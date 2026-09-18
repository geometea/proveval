"""Shared pytest setup.

The project's modules (run_trial.py, analyze_context.py, ...) are flat
top-level modules, not an installed package, so make the repo root
importable regardless of which directory pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# No test may ever reach the network: every real provider call would need a
# socket, so opening one fails loudly. Fake providers monkeypatch
# model_providers.call_model (or inject fake SDK modules) and never touch
# sockets, so they are unaffected.
# ---------------------------------------------------------------------------
import socket

import pytest


class _NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise _NetworkBlocked("network access is disabled in the test suite -- a real model API call was attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
