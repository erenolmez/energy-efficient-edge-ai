"""Explicit historical source verification after the structure-only refactor."""
import hashlib
import json
import subprocess
from pathlib import Path
from pc_tests.model_pipeline.paths import PIPELINE, REPOSITORY_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return json.loads((PIPELINE / 'provenance/structure_refactor.json').read_text())


def verify_source(legacy_name, historical_sha256):
    """Verify both the measured Git version and the separately hashed refactor.

    This does NOT claim that new source bytes equal the measured source bytes.
    No saved experiment protocol or historical hash is rewritten.
    A full Git checkout containing base_commit is required for this audit.
    """
    record = manifest()
    entry = record['files'][legacy_name]
    if historical_sha256 != entry['original_sha256']:
        raise AssertionError(f'Unrecognized historical source: {legacy_name}')
    historical = subprocess.run(
        ['git', 'show', record['base_commit'] + ':pc_tests/model_pipeline/' + legacy_name],
        cwd=REPOSITORY_ROOT, capture_output=True, check=True).stdout
    assert hashlib.sha256(historical).hexdigest() == historical_sha256, legacy_name
    assert digest(PIPELINE / entry['canonical']) == entry['canonical_sha256'], legacy_name
    assert digest(PIPELINE / legacy_name) == entry['wrapper_sha256'], legacy_name


def verify_protocol(saved, current, source_fields):
    """All data/settings hashes remain exact; moved source is verified explicitly."""
    assert saved.keys() == current.keys(), 'Protocol fields changed'
    for key, value in saved.items():
        if key in source_fields:
            filename = source_fields[key]
            verify_source(filename, value)
            assert current[key] == manifest()['files'][filename]['canonical_sha256'], key
        else:
            assert current[key] == value, key
