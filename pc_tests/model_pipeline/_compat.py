"""Old imports alias the canonical module, including its globals for mock.patch."""
import importlib
import runpy
import sys


def redirect(name, target):
    if name == '__main__':
        runpy.run_module(target, run_name='__main__')
    else:
        sys.modules[name] = importlib.import_module(target)
