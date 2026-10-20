"""Compatibility entry point; implementation: runtime/engine_precision_audit.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pc_tests.model_pipeline._compat import redirect
redirect(__name__, "pc_tests.model_pipeline.runtime.engine_precision_audit")
