"""CPU-only structure/provenance regression checks; never start experiments."""
import ast
import importlib
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pc_tests/model_pipeline')]
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT, source_path
from pc_tests.model_pipeline.provenance import manifest, verify_source, verify_protocol


class RemoveImportScaffolding(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_If(self, node):
        if ast.unparse(node.test) == 'not __package__':
            return None
        return self.generic_visit(node)


def scientific_ast(source):
    # Normalize only the documented filesystem-anchor substitutions.
    source = source.replace("Path(__file__).resolve().parents[1] / 'router_benchmark'",
                            "PIPELINE.parent / 'router_benchmark'")
    source = source.replace('Path(__file__).resolve().parents[2]', 'REPOSITORY_ROOT')
    source = source.replace('Path(__file__).resolve().parent', 'PIPELINE')
    source = re.sub(r"Path\(__file__\)\.with_name\((['\"])([^'\"]+)\1\)",
                    lambda m: 'pipeline_source(' + repr(m[2]) + ')', source)
    return ast.dump(RemoveImportScaffolding().visit(ast.parse(source)), include_attributes=False)


class PipelineStructureTests(unittest.TestCase):
    def test_legacy_imports_alias_canonical_modules(self):
        for old, new in (
            ('batch_router_policy', 'routers.scalar.batch_router_policy'),
            ('correctness_policy', 'routers.correctness.correctness_policy'),
            ('run_random_routes', 'experiments.run_random_routes'),
            ('run_random_routes_b64', 'experiments.run_random_routes_b64'),
            ('run_most_pruned_batches', 'experiments.run_most_pruned_batches'),
            ('run_p80_batches', 'experiments.run_p80_batches'),
        ):
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old),
                              importlib.import_module('pc_tests.model_pipeline.' + new))

    def test_data_roots_do_not_follow_module_depth(self):
        self.assertEqual(REPOSITORY_ROOT, ROOT)
        for name in ('run_random_routes', 'run_random_routes_b64', 'run_p80_batches'):
            self.assertEqual(importlib.import_module(name).PIPELINE, PIPELINE)
        self.assertEqual(source_path('gpu_batch_inputs.py'), PIPELINE / 'runtime/gpu_batch_inputs.py')

    def test_source_lineage_and_scientific_bodies(self):
        record = manifest()
        for old, entry in record['files'].items():
            with self.subTest(source=old):
                verify_source(old, entry['original_sha256'])
                if old.endswith('.ps1'):
                    self.assertEqual(entry['canonical_sha256'], entry['original_sha256'])
                if not old.endswith('.py') or old in ('audit_random_b64.py', 'audit_p80_batches.py'):
                    continue  # These two audits explicitly gained historical-source verification.
                original = subprocess.run(
                    ['git', 'show', record['base_commit'] + ':pc_tests/model_pipeline/' + old],
                    cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8-sig')
                current = (PIPELINE / entry['canonical']).read_text(encoding='utf-8-sig')
                self.assertEqual(scientific_ast(original), scientific_ast(current))

    def test_source_drift_fails_closed(self):
        old, entry = next(iter(manifest()['files'].items()))
        with patch('pc_tests.model_pipeline.provenance.digest', return_value='unexpected'):
            with self.assertRaises(AssertionError):
                verify_source(old, entry['original_sha256'])

    def test_protocol_comparison_does_not_relax_data_or_settings(self):
        old = 'run_random_routes_b64.py'
        entry = manifest()['files'][old]
        saved = {'script': entry['original_sha256'], 'batch': 64, 'data': 'original-data'}
        current = dict(saved, script=entry['canonical_sha256'])
        verify_protocol(saved, current, {'script': old})
        for changed in ({'batch': 128}, {'data': 'changed-data'}):
            with self.assertRaises(AssertionError):
                verify_protocol(saved, dict(current, **changed), {'script': old})
        self.assertEqual(saved['script'], entry['original_sha256'])


if __name__ == '__main__':
    unittest.main()
