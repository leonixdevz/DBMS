"""End-to-end engine tests: DDL, DML, queries, planning and constraints."""

from __future__ import annotations

import pytest

from brickdb.errors import BrickDBError, CatalogError, NotNullViolation, SQLError, UniqueViolation
from brickdb.engine import Database


# ------------------------------------------------------------------- basics

def test_create_and_list_tables(db):
    db.execute("CREATE TABLE a (x INT)")
    db.execute("CREATE TABLE b (y TEXT)")
    assert sorted(db.tables()) == ["a", "b"]
    assert db.execute("SELECT * FROM a").rows == []


def test_insert_and_select_roundtrip(db):
    db.execute("CREATE TABLE t (id INT, v TEXT)")
    assert db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')").message == "2 rows inserted"
    result = db.execute("SELECT * FROM t ORDER BY id")
    assert result.columns == ["id", "v"]
    assert result.rows == [[1, "a"], [2, "b"]]


def test_unknown_table_and_column_errors(db):
    with pytest.raises(CatalogError):
        db.execute("SELECT * FROM nope")
    db.execute("CREATE TABLE t (a INT)")
    with pytest.raises(SQLError):
        db.execute("SELECT missing FROM t")


def test_unknown_columns_are_rejected_while_planning(db):
    """Bad names must fail even when no row is ever evaluated.

    ``t`` is empty, so a lazily-resolved column would never be looked up.
    """
    db.execute("CREATE TABLE t (a INT, b INT)")
    with pytest.raises(SQLError):
        db.execute("SELECT missing FROM t")
    with pytest.raises(SQLError):
        db.execute("SELECT t.missing FROM t")
    with pytest.raises(SQLError):
        db.execute("SELECT a FROM t WHERE bogus = 1")
    with pytest.raises(SQLError):
        db.execute("SELECT a FROM t ORDER BY bogus")
    with pytest.raises(SQLError):
        db.execute("SELECT COUNT(bogus) FROM t")
    with pytest.raises(SQLError):
        db.execute("SELECT x.* FROM t")
    with pytest.raises(SQLError):
        db.execute("EXPLAIN SELECT missing FROM t")
    # Aliases and select-list names still resolve.
    assert db.execute("SELECT a AS n FROM t ORDER BY n").rows == []


def test_ambiguous_column_in_a_join_is_rejected(db):
    db.execute("CREATE TABLE a (id INT, v INT)")
    db.execute("CREATE TABLE b (id INT, w INT)")
    with pytest.raises(SQLError):
        db.execute("SELECT id FROM a JOIN b ON a.id = b.id")
    with pytest.raises(SQLError):
        db.execute("SELECT v FROM a JOIN b ON id = b.id")
    assert db.execute("SELECT a.id FROM a JOIN b ON a.id = b.id").rows == []


def test_type_coercion_and_errors(db):
    db.execute("CREATE TABLE t (i INT, f FLOAT, s TEXT)")
    db.execute("INSERT INTO t VALUES (1, 2, 'three')")
    assert db.execute("SELECT * FROM t").rows == [[1, 2.0, "three"]]
    with pytest.raises(BrickDBError):
        db.execute("INSERT INTO t VALUES ('not an int', 1, 'x')")


def test_not_null_violation(db):
    db.execute("CREATE TABLE t (a INT NOT NULL)")
    with pytest.raises(NotNullViolation):
        db.execute("INSERT INTO t VALUES (NULL)")


def test_default_values(db):
    db.execute("CREATE TABLE t (a INT, b TEXT DEFAULT 'fallback')")
    db.execute("INSERT INTO t (a) VALUES (5)")
    assert db.execute("SELECT * FROM t").rows == [[5, "fallback"]]


def test_insert_with_column_list(db):
    db.execute("CREATE TABLE t (a INT, b INT, c TEXT)")
    db.execute("INSERT INTO t (c, a) VALUES ('hi', 1)")
    assert db.execute("SELECT * FROM t").rows == [[1, None, "hi"]]
    with pytest.raises(SQLError):
        db.execute("INSERT INTO t (a, b) VALUES (1)")


# --------------------------------------------------------------- predicates

def test_where_predicates(seeded):
    assert seeded.execute("SELECT name FROM users WHERE age IS NULL").rows == [["dan"]]
    assert seeded.execute("SELECT name FROM users WHERE age IS NOT NULL AND age > 30").rows == [["carol"]]
    assert seeded.execute("SELECT name FROM users WHERE name IN ('bob', 'dan') ORDER BY name").rows == [["bob"], ["dan"]]
    assert seeded.execute("SELECT name FROM users WHERE age BETWEEN 25 AND 30 ORDER BY name").rows == [["alice"], ["bob"]]
    assert seeded.execute("SELECT name FROM users WHERE name LIKE 'a%'").rows == [["alice"]]
    assert seeded.execute("SELECT name FROM users WHERE name LIKE '_ob'").rows == [["bob"]]
    assert seeded.execute("SELECT name FROM users WHERE NOT (age > 30) ORDER BY name").rows == [["alice"], ["bob"]]


def test_null_comparison_is_unknown_not_true(seeded):
    # NULL = NULL is unknown, so nothing matches; IS NULL is the way to ask.
    assert seeded.execute("SELECT * FROM users WHERE age = NULL").rows == []
    assert seeded.execute("SELECT COUNT(*) FROM users").rows == [[4]]


def test_order_by_mixed_directions(seeded):
    rows = seeded.execute("SELECT name, age FROM users ORDER BY age DESC, name ASC").rows
    assert rows[0] == ["carol", 41]
    assert rows[-1] == ["dan", None]


def test_order_by_alias_and_limit_offset(seeded):
    rows = seeded.execute("SELECT name AS n FROM users ORDER BY n LIMIT 2 OFFSET 1").rows
    assert rows == [["bob"], ["carol"]]


def test_distinct(seeded):
    seeded.execute("INSERT INTO users VALUES (5, 'alice', 30)")
    rows = seeded.execute("SELECT DISTINCT name FROM users ORDER BY name").rows
    assert rows == [["alice"], ["bob"], ["carol"], ["dan"]]


def test_expressions_in_select(seeded):
    rows = seeded.execute("SELECT id * 10 + 1, name || '!' FROM users WHERE id = 1").rows
    assert rows == [[11, "alice!"]]


# --------------------------------------------------------------- aggregates

def test_bare_aggregates(seeded):
    assert seeded.execute("SELECT COUNT(*), COUNT(age), MIN(age), MAX(age) FROM users").rows == [[4, 3, 25, 41]]
    assert seeded.execute("SELECT AVG(age), SUM(age) FROM users").rows == [[32.0, 96]]


def test_aggregates_over_empty_table(db):
    db.execute("CREATE TABLE t (a INT)")
    assert db.execute("SELECT COUNT(*), SUM(a), AVG(a), MIN(a) FROM t").rows == [[0, None, None, None]]


def test_group_by(seeded):
    seeded.execute("CREATE TABLE orders (oid INT PRIMARY KEY, uid INT, total FLOAT)")
    seeded.execute("INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 5.5), (3, 2, 7.25)")
    rows = seeded.execute("SELECT uid, COUNT(*), SUM(total) FROM orders GROUP BY uid ORDER BY uid").rows
    assert rows == [[1, 2, 15.5], [2, 1, 7.25]]


def test_group_by_multi_column(seeded):
    rows = seeded.execute(
        "SELECT name, age, COUNT(*) FROM users GROUP BY name, age ORDER BY name"
    ).rows
    assert rows == [["alice", 30, 1], ["bob", 25, 1], ["carol", 41, 1], ["dan", None, 1]]


# -------------------------------------------------------------------- joins

def test_inner_join(seeded):
    seeded.execute("CREATE TABLE orders (oid INT PRIMARY KEY, uid INT, total FLOAT)")
    seeded.execute("INSERT INTO orders VALUES (1, 1, 10.0), (2, 2, 20.0)")
    rows = seeded.execute(
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.uid ORDER BY orders.total"
    ).rows
    assert rows == [["alice", 10.0], ["bob", 20.0]]


def test_left_join_emits_nulls_for_missing_matches(seeded):
    seeded.execute("CREATE TABLE orders (oid INT PRIMARY KEY, uid INT)")
    seeded.execute("INSERT INTO orders VALUES (1, 1)")
    rows = seeded.execute(
        "SELECT users.name, orders.oid FROM users LEFT JOIN orders ON users.id = orders.uid ORDER BY users.id"
    ).rows
    assert rows[0] == ["alice", 1]
    assert rows[1] == ["bob", None]


def test_cross_join(seeded):
    seeded.execute("CREATE TABLE t2 (k INT)")
    seeded.execute("INSERT INTO t2 VALUES (1), (2)")
    rows = seeded.execute("SELECT COUNT(*) FROM users CROSS JOIN t2").rows
    assert rows == [[8]]


# ------------------------------------------------------------------ updates

def test_update_rows(seeded):
    assert seeded.execute("UPDATE users SET age = age + 1 WHERE name = 'alice'").message == "1 row updated"
    assert seeded.execute("SELECT age FROM users WHERE name = 'alice'").rows == [[31]]


def test_update_all_rows_without_where(seeded):
    assert seeded.execute("UPDATE users SET age = 0").message == "4 rows updated"
    assert seeded.execute("SELECT DISTINCT age FROM users").rows == [[0]]


def test_update_respects_not_null(seeded):
    with pytest.raises(BrickDBError):
        seeded.execute("UPDATE users SET name = NULL WHERE id = 1")


def test_delete_rows(seeded):
    assert seeded.execute("DELETE FROM users WHERE age IS NULL").message == "1 row deleted"
    assert seeded.execute("SELECT COUNT(*) FROM users").rows == [[3]]
    assert seeded.execute("DELETE FROM users").message == "3 rows deleted"
    assert seeded.execute("SELECT COUNT(*) FROM users").rows == [[0]]


# -------------------------------------------------------------- constraints

def test_primary_key_uniqueness(seeded):
    with pytest.raises(UniqueViolation):
        seeded.execute("INSERT INTO users VALUES (1, 'clone', 1)")


def test_unique_index_enforcement(seeded):
    seeded.execute("CREATE UNIQUE INDEX ux_users_name ON users (name)")
    with pytest.raises(UniqueViolation):
        seeded.execute("INSERT INTO users VALUES (9, 'alice', 1)")


def test_primary_key_blocks_nulls(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
    with pytest.raises(NotNullViolation):
        db.execute("INSERT INTO t VALUES (NULL)")


def test_drop_table_removes_indexes(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    db.execute("CREATE INDEX ix ON t (v)")
    db.execute("DROP TABLE t")
    assert db.tables() == []
    assert db.indexes() == []
    with pytest.raises(CatalogError):
        db.execute("SELECT * FROM t")


# ------------------------------------------------------------------- indexes

def test_index_is_used_for_equality_lookup(seeded):
    plan = seeded.execute("EXPLAIN SELECT * FROM users WHERE id = 2").plan
    assert any("IndexScan" in line for line in plan)
    assert seeded.execute("SELECT name FROM users WHERE id = 2").rows == [["bob"]]


def test_index_is_used_for_text_equality(seeded):
    plan = seeded.execute("EXPLAIN SELECT * FROM users WHERE name = 'carol'").plan
    assert any("IndexScan ix_name" in line for line in plan)


def test_full_scan_when_no_index_applies(seeded):
    plan = seeded.execute("EXPLAIN SELECT * FROM users WHERE age = 30").plan
    assert any("SeqScan" in line for line in plan)


def test_indexed_range_scan(seeded):
    rows = seeded.execute("SELECT id FROM users WHERE id > 1 AND id < 4 ORDER BY id").rows
    assert rows == [[2], [3]]


def test_reindex_rebuilds_index(seeded):
    assert "Rebuilt" in seeded.execute("REINDEX users").message
    assert seeded.execute("SELECT name FROM users WHERE name = 'bob'").rows == [["bob"]]
    assert seeded.execute("EXPLAIN INDEX ix_name").rows[-1][1] == "ok"


def test_unique_index_rejects_duplicate_values_on_create(seeded):
    # A unique index over data that already contains duplicates must fail,
    # and the failed statement must leave no phantom index behind.
    seeded.execute("INSERT INTO users VALUES (7, 'alice', 99)")
    with pytest.raises(BrickDBError):
        seeded.execute("CREATE UNIQUE INDEX bad ON users (name)")
    assert "bad" not in seeded.indexes()
    # NULLs are not indexed, so several NULL ages are acceptable.
    seeded.execute("CREATE UNIQUE INDEX ok ON users (age)")


# ------------------------------------------------------------------- reports

def test_explain_table_report(seeded):
    rows = dict(seeded.execute("EXPLAIN TABLE users").rows)
    assert rows["table"] == "users"
    assert rows["rows"] == "4"
    assert "ix_name" in rows["indexes"]


def test_explain_stats_report(seeded):
    comps = [r[0] for r in seeded.execute("EXPLAIN STATS").rows]
    assert "buffer_pool" in comps and "wal" in comps and "disk" in comps


def test_explain_select_plan_shape(seeded):
    plan = seeded.execute("EXPLAIN SELECT name FROM users WHERE id = 1 ORDER BY name LIMIT 3").plan
    joined = "\n".join(plan)
    assert "IndexScan" in joined
    assert "Project" in joined
    assert "Sort" in joined
    assert "Limit" in joined


# --------------------------------------------------------------- durability

def test_data_survives_reopen(db_path):
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    db.execute("INSERT INTO t VALUES (1, 'kept')")
    db.close()

    db2 = Database(db_path)
    try:
        assert db2.execute("SELECT * FROM t").rows == [[1, "kept"]]
        assert db2.execute("SELECT v FROM t WHERE id = 1").rows == [["kept"]]
    finally:
        db2.close()


def test_many_rows_span_multiple_pages(db_path):
    db = Database(db_path)
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, payload TEXT)")
    for i in range(300):
        db.execute(f"INSERT INTO t VALUES ({i}, 'row-{i:04d}-{'x' * 60}')")
    assert db.execute("SELECT COUNT(*) FROM t").rows == [[300]]
    assert db.execute("SELECT COUNT(*) FROM t WHERE id >= 250").rows == [[50]]
    rows = dict(db.execute("EXPLAIN TABLE t").rows)
    assert int(rows["heap_pages"]) > 1
    db.close()

    db2 = Database(db_path)
    try:
        assert db2.execute("SELECT COUNT(*) FROM t").rows == [[300]]
        assert db2.execute("SELECT payload FROM t WHERE id = 42").rows == [[f"row-0042-{'x' * 60}"]]
    finally:
        db2.close()
