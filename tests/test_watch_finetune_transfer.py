import io
import tarfile
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pc_tests.model_pipeline.scripts import watch_finetune_transfer as watcher


class WatchFinetuneTransferTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.remote = 'user@host:/results/export'
        self.pc = {'pc-case'}
        self.hpc = {'hpc-case'}

    def exercise(self, interval):
        watcher.run(self.base / 'handoff.json', self.remote, self.base / 'imported', ['123'], interval)

    def test_waits_then_collects_once_noninteractively(self):
        state = dict(status='running', active='pc-case')
        with patch.object(watcher, 'snapshot', side_effect=[(self.pc, self.hpc, set(), state),
                                                           (self.pc, self.hpc, self.pc, state)]), \
             patch.object(watcher, 'probe', side_effect=[dict(complete=[], live={'123': 'RUNNING'}),
                                                        dict(complete=['hpc-case'], live={})]), \
             patch.object(watcher.time, 'sleep') as sleep, patch.object(watcher.runner, 'collect') as collect:
            self.exercise(10)
        sleep.assert_called_once_with(600)
        collect.assert_called_once_with(self.base / 'handoff.json', self.remote,
                                        self.base / 'imported', batch_mode=True)

    def test_incomplete_export_after_exit_is_not_transferred(self):
        state = dict(status='running', active=None)
        with patch.object(watcher, 'snapshot', return_value=(self.pc, self.hpc, self.pc, state)), \
             patch.object(watcher, 'probe', return_value=dict(complete=[], live={})), \
             patch.object(watcher.runner, 'collect') as collect:
            with self.assertRaisesRegex(RuntimeError, 'exports are incomplete'):
                self.exercise(None)
        collect.assert_not_called()

    def test_single_login_uses_one_connection_and_no_scp(self):
        def fake_ssh(args, stdout, check):
            self.assertEqual(args[0], 'ssh')
            self.assertNotIn('BatchMode=yes', args)
            with tarfile.open(fileobj=stdout, mode='w|') as tar:
                for name in ('best.pth', 'receipt.json', 'study.json'):
                    member = tarfile.TarInfo(f'hpc-case/{name}')
                    member.size = 4
                    tar.addfile(member, io.BytesIO(b'test'))

        def verify_stage(*args, local_exports):
            self.assertEqual((local_exports / 'hpc-case/best.pth').read_bytes(), b'test')

        with patch.object(watcher, 'snapshot', return_value=(self.pc, self.hpc, self.pc, dict(status='running'))), \
             patch.object(watcher.subprocess, 'run', side_effect=fake_ssh) as ssh, \
             patch.object(watcher.runner, 'collect', side_effect=verify_stage) as collect:
            watcher.single_login(self.base / 'handoff.json', self.remote, self.base / 'imported', ['123'], 10)
        ssh.assert_called_once()
        collect.assert_called_once()

    def test_archive_rejects_path_escape(self):
        archive = self.base / 'unsafe.tar'
        with tarfile.open(archive, 'w') as tar:
            tar.addfile(tarfile.TarInfo('../bad'))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            watcher.unpack(archive, self.base / 'stage', self.hpc)
