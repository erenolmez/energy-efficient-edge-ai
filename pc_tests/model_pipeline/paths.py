"""Stable data roots and canonical source paths, independent of entry-point depth."""
import json
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
REPOSITORY_ROOT = PIPELINE.parents[1]


def source_path(legacy_name):
    """Resolve a former root-level source filename; never redirect data hashes."""
    manifest = json.loads((PIPELINE / 'provenance/structure_refactor.json').read_text())
    entry = manifest['files'].get(legacy_name)
    return PIPELINE / (entry['canonical'] if entry else legacy_name)
