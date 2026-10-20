"""Compatibility entry point; tests live in tests/test_p80_batches.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pc_tests.model_pipeline._compat import redirect
redirect(__name__, 'tests.test_p80_batches')
