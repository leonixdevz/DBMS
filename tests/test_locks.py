"""Lock manager: compatibility, upgrades and deadlock detection."""

from __future__ import annotations

import threading
import time

import pytest

from brickdb.errors import DeadlockError, LockConflict
from brickdb.txn.locks import LockManager, S, X


def test_shared_locks_are_compatible():
    lm = LockManager()
    lm.acquire(1, "table:a", S)
    lm.acquire(2, "table:a", S)  # readers do not block readers
    assert set(lm.held_resources(1)) == {"table:a"}


def test_exclusive_lock_conflicts_with_reader():
    lm = LockManager(deadlock_timeout=0.2)
    lm.acquire(1, "table:a", X)
    with pytest.raises(LockConflict):
        lm.acquire(2, "table:a", S)


def test_writer_waits_for_reader_then_proceeds():
    lm = LockManager(deadlock_timeout=2.0)
    lm.acquire(1, "table:a", S)
    order = []

    def writer():
        lm.acquire(2, "table:a", X)  # blocks until txn 1 releases
        order.append("writer acquired")

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.1)
    order.append("releasing reader")
    lm.release_all(1)
    thread.join(timeout=3)
    assert order == ["releasing reader", "writer acquired"]


def test_release_all_frees_every_resource():
    lm = LockManager()
    lm.acquire(1, "table:a", X)
    lm.acquire(1, "table:b", S)
    lm.release_all(1)
    assert lm.held_resources(1) == {}
    lm.acquire(2, "table:a", X)  # no conflict now


def test_deadlock_is_detected_and_one_transaction_is_aborted():
    lm = LockManager(deadlock_timeout=3.0)
    lm.acquire(1, "table:a", X)
    lm.acquire(2, "table:b", X)
    errors = []

    def t1():
        try:
            lm.acquire(1, "table:b", X)  # waits for txn 2
            errors.append(("t1", "acquired"))
        except DeadlockError as exc:
            errors.append(("t1", "deadlock"))

    thread = threading.Thread(target=t1)
    thread.start()
    time.sleep(0.15)
    try:
        lm.acquire(2, "table:a", X)  # completes the cycle -> victim here
        errors.append(("t2", "acquired"))
    except DeadlockError:
        errors.append(("t2", "deadlock"))
    assert ("t2", "deadlock") in errors

    # The victim releases its locks, unblocking the other transaction.
    lm.release_all(2)
    thread.join(timeout=3)
    assert ("t1", "acquired") in errors
