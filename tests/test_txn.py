"""Transactions and crash recovery.

The interesting cases exercise the two crash windows the WAL exists for:

1. **after COMMIT is fsynced, before the data pages were written** —
   recovery must redo the pages (test_wal_redo_restores_committed_pages).
2. **before COMMIT** — the transaction's pages were never published, so
   recovery must simply leave the database at the pre-transaction state
   (test_uncommitted_transaction_is_lost_after_crash).
"""

from __future__ import annotations

import os

import pytest

from brickdb.engine import Database
from brickdb.errors import SQLError


def test_begin_commit_persists(db):
    db.execute("CREATE TABLE t (id INT)")
    db.execute("BEGIN")
    db.execute("INSERT INTO t VALUES (1)")
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[1]]  # read your own writes
    db.execute("COMMIT")
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[1]]


def test_rollback_discards_changes(db):
    db.execute("CREATE TABLE t (id INT, v TEXT)")
    db.execute("INSERT INTO t VALUES (1, 'committed')")
    db.execute("BEGIN")
    db.execute("INSERT INTO t VALUES (2, 'rolled back')")
    db.execute("UPDATE t SET v = 'changed' WHERE id = 1")
    db.execute("DELETE FROM t WHERE id = 1")
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[1]]  # sees own writes...
    db.execute("ROLLBACK")
    rows = db.execute("SELECT * FROM t").rows
    assert rows == [[1, "committed"]]  # ...but they are gone afterwards


def test_nested_begin_is_rejected(db):
    db.execute("BEGIN")
    with pytest.raises(SQLError):
        db.execute("BEGIN")
    db.execute("ROLLBACK")


def test_commit_without_transaction_is_rejected(db):
    with pytest.raises(SQLError):
        db.execute("COMMIT")
    with pytest.raises(SQLError):
        db.execute("ROLLBACK")


def test_statement_failure_aborts_the_transaction(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
    db.execute("INSERT INTO t VALUES (1)")
    db.execute("BEGIN")
    db.execute("INSERT INTO t VALUES (2)")
    with pytest.raises(Exception):
        db.execute("INSERT INTO t VALUES (2)")  # duplicate key
    # The transaction is gone; the successful earlier insert was discarded.
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[1]]
    with pytest.raises(SQLError):
        db.execute("COMMIT")


def test_failed_statement_leaves_no_partial_rows(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
    with pytest.raises(Exception):
        db.execute("INSERT INTO t VALUES (1), (2), (1)")  # third row violates the key
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[0]]


def test_wal_redo_restores_committed_pages(db_path, monkeypatch):
    """A committed transaction whose pages never reached the data file must
    come back from the log after a crash."""
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    db.execute("INSERT INTO t VALUES (1, 'flushed')")
    # Simulate the crash window: commits are logged, but page writes stall.
    monkeypatch.setattr(db.pool, "flush_page", lambda page_id: None)
    db.execute("INSERT INTO t VALUES (2, 'only in the log')")
    # "Crash": abandon the connection without close()/checkpoint.
    del db

    db2 = Database(db_path)
    try:
        assert db2.recovery_stats()["pages_redone"] >= 1
        rows = db2.execute("SELECT id, v FROM t ORDER BY id").rows
        assert rows == [[1, "flushed"], [2, "only in the log"]]
    finally:
        db2.close()


def test_uncommitted_transaction_is_lost_after_crash(db_path):
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
    db.execute("INSERT INTO t VALUES (1)")
    db.execute("BEGIN")
    db.execute("INSERT INTO t VALUES (2)")
    del db  # crash before COMMIT: staged pages were never published

    db2 = Database(db_path)
    try:
        assert db2.execute("SELECT id FROM t ORDER BY id").rows == [[1]]
    finally:
        db2.close()


def test_checkpoint_truncates_the_log(db_path):
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT)")
    db.execute("INSERT INTO t VALUES (1)")
    size_before = os.path.getsize(db.wal_path)
    assert size_before > 0
    db.checkpoint()
    assert os.path.getsize(db.wal_path) == 0
    assert db.recovery_stats()["pages_redone"] == 0
    db.close()


def test_index_roots_survive_a_crash(db_path, monkeypatch):
    """Root splits move the B+ tree entry point; the new root must be part of
    the same durable commit as the data."""
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    # Enough rows to force several splits (ORDER = 8 in this implementation).
    for i in range(200):
        db.execute(f"INSERT INTO t VALUES ({i}, 'v{i}')")
    monkeypatch.setattr(db.pool, "flush_page", lambda page_id: None)
    db.execute("INSERT INTO t VALUES (999, 'after split')")  # forces more splits
    del db

    db2 = Database(db_path)
    try:
        assert db2.execute("SELECT v FROM t WHERE id = 999").rows == [["after split"]]
        assert db2.execute("SELECT COUNT(*) FROM t").rows == [[201]]
        check = dict(db2.execute("EXPLAIN INDEX t_pkey").rows)
        assert check["check"] == "ok"
        assert check["entries"] == "201"
    finally:
        db2.close()


def test_many_commits_then_reopen(db_path):
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    for i in range(25):
        db.execute(f"INSERT INTO t VALUES ({i}, 'v{i}')")
    db.execute("BEGIN")
    for i in range(25, 30):
        db.execute(f"INSERT INTO t VALUES ({i}, 'v{i}')")
    db.execute("COMMIT")
    db.close()

    db2 = Database(db_path)
    try:
        assert db2.execute("SELECT COUNT(*) FROM t").rows == [[30]]
        assert db2.execute("SELECT v FROM t WHERE id = 29").rows == [["v29"]]
    finally:
        db2.close()


def test_locks_are_released_after_each_statement(seeded):
    # 20 sequential statements would deadlock if locks leaked.
    for i in range(20):
        seeded.execute(f"SELECT COUNT(*) FROM users WHERE id > {i}")
    assert seeded.locks.stats()["locked_resources"] == 0
