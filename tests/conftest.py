"""Shared pytest setup.

The project's modules (run_trial.py, analyze_context.py, ...) are flat
top-level modules, not an installed package, so make the repo root
importable regardless of which directory pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
