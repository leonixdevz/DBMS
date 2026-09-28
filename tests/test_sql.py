"""SQL front-end tests: tokenizer, AST shapes, and error reporting."""

from __future__ import annotations

import pytest

from brickdb.errors import SQLError
from brickdb.sql import ast as A
from brickdb.sql.lexer import tokenize
from brickdb.sql.parser import parse


def test_tokenizer_basics():
    kinds = [t.kind for t in tokenize("SELECT a, 12, 3.5, 'x' FROM t;")]
    assert kinds[:4] == ["kw", "ident", "op", "int"]
    assert any(t.kind == "string" and t.value == "x" for t in tokenize("SELECT 'x'"))
    assert tokenize("select * from t")[0].value == "SELECT"  # keywords are canonical


def test_tokenizer_handles_comments_and_escaped_quotes():
    tokens = tokenize("SELECT 'it''s' -- trailing comment\nFROM t")
    assert any(t.kind == "string" and t.value == "it's" for t in tokens)
    assert [t.value for t in tokens if t.kind == "kw"] == ["SELECT", "FROM"]


def test_tokenizer_rejects_unterminated_string():
    with pytest.raises(SQLError):
        tokenize("SELECT 'oops")


def test_parse_create_table():
    stmt = parse("CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL, score FLOAT UNIQUE, when DATE)")
    assert isinstance(stmt, A.CreateTableStmt)
    assert stmt.table == "t"
    assert [c.name for c in stmt.columns] == ["id", "name", "score", "when"]
    assert stmt.columns[0].primary_key is True
    assert stmt.columns[0].notnull is True
    assert stmt.columns[1].notnull is True
    assert stmt.columns[2].unique is True
    assert stmt.columns[3].kind == "DATE"


def test_parse_insert_with_columns():
    stmt = parse("INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y')")
    assert stmt.columns == ["a", "b"]
    assert len(stmt.rows) == 2
    assert isinstance(stmt.rows[0][0], A.Literal)


def test_parse_select_with_everything():
    stmt = parse(
        "SELECT DISTINCT u.name AS n, COUNT(*) FROM users u "
        "LEFT JOIN orders o ON u.id = o.uid "
        "WHERE u.age BETWEEN 20 AND 40 AND n <> 'x' "
        "GROUP BY u.name ORDER BY n DESC LIMIT 5 OFFSET 2"
    )
    assert stmt.distinct is True
    assert stmt.from_table.name == "users" and stmt.from_table.alias == "u"
    assert len(stmt.joins) == 1 and stmt.joins[0].kind == "LEFT"
    assert isinstance(stmt.items[1].expr, A.FuncCall) and stmt.items[1].expr.star
    assert isinstance(stmt.where, A.BinaryOp) and stmt.where.op == "AND"
    assert len(stmt.group_by) == 1 and len(stmt.order_by) == 1
    assert stmt.order_by[0].desc is True
    assert stmt.limit == 5 and stmt.offset == 2


def test_parse_expression_precedence():
    stmt = parse("SELECT 1 + 2 * 3 FROM t")
    expr = stmt.items[0].expr
    assert isinstance(expr, A.BinaryOp) and expr.op == "+"
    assert isinstance(expr.right, A.BinaryOp) and expr.right.op == "*"


def test_parse_predicates():
    assert isinstance(parse("SELECT * FROM t WHERE a IS NOT NULL").where, A.IsNull)
    assert isinstance(parse("SELECT * FROM t WHERE a IN (1, 2, 3)").where, A.InList)
    assert isinstance(parse("SELECT * FROM t WHERE NOT a").where, A.UnaryOp)
    like = parse("SELECT * FROM t WHERE a LIKE 'x%'").where
    assert isinstance(like, A.BinaryOp) and like.op == "LIKE"


def test_parse_ddl_statements():
    assert isinstance(parse("DROP TABLE IF EXISTS t"), A.DropTableStmt)
    assert parse("DROP TABLE IF EXISTS t").if_exists is True
    ci = parse("CREATE UNIQUE INDEX ix ON t (a, b)")
    assert ci.unique is True and ci.columns == ["a", "b"]
    assert isinstance(parse("REINDEX t"), A.ReindexStmt)
    assert isinstance(parse("EXPLAIN TABLE t"), A.ExplainStmt)
    assert isinstance(parse("EXPLAIN STATS"), A.ExplainStmt)


def test_parse_errors_are_clear():
    with pytest.raises(SQLError):
        parse("SELECT * FROM")
    with pytest.raises(SQLError):
        parse("CREATE TABLE t (id NOPE)")
    with pytest.raises(SQLError):
        parse("SELECT * FROM t WHERE a = ")
    with pytest.raises(SQLError):
        parse("INSERT INTO t VALUES (1,,2)")
    with pytest.raises(SQLError):
        parse("SELECT 1 FROM t trailing garbage")


def test_parse_accepts_optional_semicolon_and_newlines():
    stmt = parse("SELECT\n  1,\n  2\nFROM t;\n")
    assert len(stmt.items) == 2
