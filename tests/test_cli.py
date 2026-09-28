"""CLI helpers: statement splitting, table rendering, one-shot execution."""

from __future__ import annotations

from datetime import date

from brickdb.cli import _split_statements, format_table, format_value, main


def test_split_statements_ignores_semicolons_in_strings():
    parts = _split_statements("INSERT INTO t VALUES ('a;b'); SELECT 1;\n")
    assert len(parts) == 2
    assert "'a;b'" in parts[0]


def test_format_value_renders_sql_types():
    assert format_value(None) == "NULL"
    assert format_value(True) == "true"
    assert format_value(2.0) == "2.0"
    assert format_value(date(2024, 1, 5)) == "2024-01-05"
    assert format_value("x") == "x"


def test_format_table_box_drawing():
    out = format_table(["id", "name"], [[1, "alice"], [2, None]])
    lines = out.splitlines()
    assert lines[0].startswith("┌") and lines[-1].startswith("└")
    assert "alice" in out and "NULL" in out
    assert len(lines) == 6  # top border, header, separator, 2 data rows, bottom


def test_format_table_empty_result():
    out = format_table(["a", "b"], [])
    assert "a" in out and "b" in out


def test_cli_execute_one_shot(tmp_path, capsys):
    path = str(tmp_path / "cli.db")
    assert main([path, "-e", "CREATE TABLE t (id INT)"]) == 0
    assert main([path, "-e", "INSERT INTO t VALUES (1), (2)"]) == 0
    capsys.readouterr()
    assert main([path, "-e", "SELECT * FROM t ORDER BY id"]) == 0
    captured = capsys.readouterr().out
    # A query renders its rows, not its plan.
    assert "(2 rows)" in captured
    assert "SeqScan" not in captured


def test_cli_explain_prints_the_plan(tmp_path, capsys):
    path = str(tmp_path / "cli_explain.db")
    assert main([path, "-e", "CREATE TABLE t (id INT)"]) == 0
    capsys.readouterr()
    assert main([path, "-e", "EXPLAIN SELECT * FROM t WHERE id = 1"]) == 0
    captured = capsys.readouterr().out
    assert "SeqScan t" in captured
    assert "(0 rows)" not in captured


def test_cli_reports_errors_without_crashing(tmp_path, capsys):
    path = str(tmp_path / "cli2.db")
    assert main([path, "-e", "SELECT * FROM missing_table"]) == 0
    assert "Error" in capsys.readouterr().out
