"""Shared fixtures: each test gets its own temporary database."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brickdb.engine import Database  # noqa: E402


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture()
def db(db_path):
    database = Database(db_path)
    yield database
    database.close()


@pytest.fixture()
def seeded(db):
    """A small users table plus indexes, ready for query tests."""
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL, age INT)")
    db.execute("INSERT INTO users VALUES (1, 'alice', 30), (2, 'bob', 25), (3, 'carol', 41), (4, 'dan', NULL)")
    db.execute("CREATE INDEX ix_name ON users (name)")
    return db
