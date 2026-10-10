"""bookgap.jobs.stop_signals: a job's child process stops at its next boundary."""
from __future__ import annotations

import os
import signal

import pytest

from bookgap import jobs


def test_a_first_signal_requests_a_stop_and_a_second_interrupts():
    with jobs.stop_signals():
        assert not jobs.stop_requested()
        os.kill(os.getpid(), signal.SIGTERM)
        assert jobs.stop_requested()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_entering_resets_a_stale_request():
    with jobs.stop_signals():
        os.kill(os.getpid(), signal.SIGINT)
    with jobs.stop_signals():
        assert not jobs.stop_requested()
