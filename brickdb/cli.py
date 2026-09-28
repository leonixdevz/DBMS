"""Interactive SQL shell (REPL) for brickdb.

Usage::

    python -m brickdb [path/to/database.db]

Features: multi-line statements terminated by ``;``, meta commands starting
with a dot, boxed result tables, per-statement timing, and friendly error
reporting that keeps the session alive after a failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from typing import List, Optional

from .engine import Database, QueryResult
from .errors import BrickDBError

BANNER = """\
brickdb 0.1.0 — a tiny relational DBMS built from scratch
Type ".help" for help, ".quit" to exit. Statements end with ';'."""

HELP = """\
Meta commands:
  .help                show this help
  .tables              list tables
  .schema [table]      show the schema of one or all tables
  .indexes             list indexes
  .stats               disk / buffer pool / WAL / lock statistics
  .checkpoint          flush dirty pages and truncate the write-ahead log
  .timing [on|off]     toggle per-statement timing
  .open <file>         ... not supported: restart with a different path
  .quit / .exit        leave the shell

SQL:
  CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL);
  CREATE [UNIQUE] INDEX ix ON t (col);
  INSERT INTO t VALUES (1, 'a'), (2, 'b');
  SELECT * FROM t WHERE id > 1 ORDER BY name DESC LIMIT 5;
  UPDATE t SET name = 'x' WHERE id = 1;
  DELETE FROM t WHERE id = 1;
  BEGIN; ... COMMIT; / ROLLBACK;
  EXPLAIN SELECT ...;  EXPLAIN TABLE t;  EXPLAIN INDEX ix;  EXPLAIN STATS;
  REINDEX t;
Types: INT, FLOAT, TEXT, DATE."""


def format_value(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(v)
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def format_table(columns: List[str], rows: List[List]) -> str:
    """Render an ASCII box table, right-aligning numeric columns."""
    if not columns:
        return ""
    cells = [[format_value(c) for c in row] for row in rows]
    widths = [len(str(c)) for c in columns]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    numeric = [
        all(
            _is_number(row[i])
            for row in rows
        )
        if rows
        else False
        for i in range(len(columns))
    ]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt_row(items: List[str], header: bool = False) -> str:
        parts = []
        for i, item in enumerate(items):
            if header or not numeric[i]:
                parts.append(" " + item.ljust(widths[i]) + " ")
            else:
                parts.append(" " + item.rjust(widths[i]) + " ")
        return "│" + "│".join(parts) + "│"

    out = [line("┌", "┬", "┐"), fmt_row([str(c) for c in columns], header=True), line("├", "┼", "┤")]
    for row in cells:
        out.append(fmt_row(row))
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Shell:
    def __init__(self, db: Database):
        self.db = db
        self.timing = True

    # ------------------------------------------------------------ output

    def render(self, result: QueryResult, elapsed: float) -> None:
        if result.message is not None:
            print(result.message)
        elif result.explain:
            # EXPLAIN's payload is the plan text itself.
            print("\n".join(result.plan))
        else:
            if result.rows:
                print(format_table(result.columns, result.rows))
            n = len(result.rows)
            print(f"({n} row{'s' if n != 1 else ''})")
        if self.timing:
            print(f"Time: {elapsed * 1000:.2f} ms")

    # ---------------------------------------------------------- commands

    def run_statement(self, sql: str) -> None:
        start = time.perf_counter()
        try:
            result = self.db.execute(sql)
        except BrickDBError as exc:
            print(f"Error: {exc}")
            return
        except Exception as exc:  # defensive: never kill the REPL
            print(f"Internal error: {type(exc).__name__}: {exc}")
            return
        elapsed = time.perf_counter() - start
        self.render(result, elapsed)

    def run_meta(self, line: str, rest: str) -> bool:
        """Returns False to signal exit."""
        cmd = line.lower()
        if cmd in (".quit", ".exit", "exit", "quit"):
            return False
        if cmd == ".help":
            print(HELP)
        elif cmd == ".tables":
            tables = self.db.tables()
            print("\n".join(tables) if tables else "(no tables)")
        elif cmd == ".indexes":
            idxs = self.db.indexes()
            print("\n".join(idxs) if idxs else "(no indexes)")
        elif cmd == ".schema":
            names = [rest.strip()] if rest.strip() else self.db.tables()
            if not names:
                print("(no tables)")
            for name in names:
                try:
                    print("\n".join(self.db.schema(name)))
                except BrickDBError as exc:
                    print(f"Error: {exc}")
        elif cmd == ".stats":
            for row in self.db._explain_stats().rows:
                print(f"{row[0]:>12}: {row[1]}")
        elif cmd == ".checkpoint":
            self.db.checkpoint()
            print("checkpoint complete (pages flushed, WAL truncated)")
        elif cmd == ".timing":
            arg = rest.strip().lower()
            if arg in ("on", "off"):
                self.timing = arg == "on"
            elif arg:
                print("usage: .timing [on|off]")
            else:
                self.timing = not self.timing
            print(f"timing {'on' if self.timing else 'off'}")
        elif cmd == ".open":
            print("Not supported: exit and restart with a different database path.")
        else:
            print(f"Unknown command {line!r}; try .help")
        return True

    # ---------------------------------------------------------------- loop

    def loop(self) -> None:
        buffer = ""
        while True:
            prompt = "brickdb> " if not buffer else "      -> "
            try:
                line = input(prompt)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("^C")
                buffer = ""
                continue
            stripped = line.strip()
            if not buffer and not stripped:
                continue
            if not buffer and stripped.startswith("."):
                if not self.run_meta(stripped.split()[0], stripped[len(stripped.split()[0]):]):
                    break
                continue
            if not buffer and stripped.lower() in ("exit", "quit"):
                break
            buffer += line + "\n"
            if ";" not in buffer:
                continue
            statements = _split_statements(buffer)
            buffer = ""
            for stmt in statements:
                if stmt.strip():
                    self.run_statement(stmt)


def _split_statements(text: str) -> List[str]:
    """Split on semicolons that are not inside string literals."""
    out: List[str] = []
    current: List[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            if in_string and i + 1 < len(text) and text[i + 1] == "'":
                current.append("''")
                i += 2
                continue
            in_string = not in_string
            current.append(ch)
        elif ch == ";" and not in_string:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current)
    if tail.strip():
        out.append(tail)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brickdb", description="A tiny relational DBMS built from scratch.")
    p.add_argument("database", nargs="?", default="brickdb.db",
                   help="path to the database file (created if missing; default: brickdb.db)")
    p.add_argument("--buffer-pages", type=int, default=64, metavar="N",
                   help="buffer pool capacity in pages (default: 64)")
    p.add_argument("--execute", "-e", metavar="SQL",
                   help="run one statement and exit (useful for demos and tests)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        db = Database(args.database, buffer_pool_size=args.buffer_pages)
    except BrickDBError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    shell = Shell(db)
    try:
        if args.execute:
            shell.timing = False
            shell.run_statement(args.execute)
            return 0
        print(BANNER)
        rec = db.recovery_stats()
        if rec.get("pages_redone"):
            print(f"[recovery] redid {rec['pages_redone']} page(s) from {rec['records']} WAL record(s)")
        shell.loop()
    finally:
        db.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
