# brickdb

A small relational DBMS written from scratch in Python — no SQLite, no SQL
parsing library, no third-party runtime dependencies. Slotted pages, a buffer
pool, B+ tree indexes, a write-ahead log with crash recovery, table-level
locking with deadlock detection, a SQL parser, and a Volcano-style executor,
in about 5,100 lines of library code.

It exists to show the machinery, not to replace PostgreSQL: every layer is
small enough to read in one sitting, and each design decision is written down
next to the code that implements it.

```
$ python -m brickdb mydata.db

brickdb> CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL, age INT);
Table 'users' created

brickdb> INSERT INTO users VALUES (1, 'alice', 30), (2, 'bob', 25), (3, 'carol', 41);
3 rows inserted

brickdb> SELECT name, age FROM users WHERE id = 2;
┌──────┬─────┐
│ name │ age │
├──────┼─────┤
│ bob  │  25 │
└──────┴─────┘
(1 row)

brickdb> EXPLAIN SELECT name FROM users WHERE age > 28 ORDER BY name;
Project [name]
  Sort name
    Filter age > 28
      SeqScan users (1 page(s))

brickdb> EXPLAIN SELECT * FROM users WHERE id = 2;
Project [id, name, age]
  Filter id = 2
    IndexScan users_pkey on users
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install pytest          # the only test dependency
.venv/bin/python -m pytest -q         # 109 tests
.venv/bin/python -m brickdb data.db   # interactive shell
```

Everything runs from the repository root; `pip install -e .` is only needed if
you want the `brickdb` console script. Requires Python 3.9+, and the standard
library is the only runtime dependency.

## What's implemented

| Area | Details |
| --- | --- |
| Types | `INT`, `FLOAT`, `TEXT`, `DATE`, `NULL` |
| DDL | `CREATE TABLE` (`NOT NULL`, `PRIMARY KEY`, `UNIQUE`, `DEFAULT`), `DROP TABLE [IF EXISTS]`, `CREATE [UNIQUE] INDEX`, `DROP INDEX [IF EXISTS]`, `REINDEX` |
| DML | `INSERT` (multi-row, column lists), `UPDATE`, `DELETE` |
| Queries | `WHERE` with `AND`/`OR`/`NOT`, `= <> < <= > >=`, `IN`, `BETWEEN`, `LIKE`, `IS [NOT] NULL`, arithmetic, `||`; `GROUP BY` + `COUNT`/`SUM`/`MIN`/`MAX`/`AVG`; `DISTINCT`; multi-key `ORDER BY`; `LIMIT`/`OFFSET`; `INNER`/`LEFT`/`CROSS JOIN`; constant `SELECT` without `FROM` |
| Transactions | `BEGIN` / `COMMIT` / `ROLLBACK`, statement-level atomicity for every statement including DDL |
| Introspection | `EXPLAIN SELECT`, `EXPLAIN TABLE`, `EXPLAIN INDEX`, `EXPLAIN STATS` |

The CLI adds `.help`, `.tables`, `.schema`, `.indexes`, `.stats`,
`.checkpoint` and `.timing`. Multi-line statements end with `;`.

## Architecture

Data flows down; control flows up. Each layer may only call the layer below it.

```
                     ┌──────────────────────────────────────────┐
   cli.py            │ interactive shell: parse, execute, print  │
                     ├──────────────────────────────────────────┤
   engine.py         │ statement dispatch · planner · index      │
                     │ maintenance · commit protocol (WAL first) │
                     ├────────────┬─────────────┬───────────────┤
   sql/              │ lexer      │ AST         │ parser        │
   executors/        │ operators  │ (Volcano)   │ evaluator     │
   txn/              │ WAL        │ recovery    │ lock manager  │
   catalog.py        │ tables/index metadata (dual-slot, CRC'd)  │
   index/            │ B+ tree   │ key encoding                  │
   buffer.py         │ buffer pool: LRU, pin/unpin, dirty bits    │
   storage/          │ slotted pages · heap files · pager         │
                     │ disk manager: page 0 header + data pages   │
                     └──────────────────────────────────────────┘
```

A statement travels:

```
parse ─► plan ─► execute (pages staged in the pager)
                   │
                   ├─ commit: catalog snapshot → WAL DATA + COMMIT (fsync)
                   │          → publish pages to the pool (lazily written)
                   └─ abort:  discard staged pages, restore the catalog
```

### Storage

* **Disk manager** — one `*.db` file. Page 0 is a header (magic `BRICKDB1`,
  page size, CRC32). Pages are appended and never move, so *global page id =
  byte offset / PAGE_SIZE* is a permanent address.
* **Slotted page** (4 KiB) — header `(num_slots, free_offset)`, a slot
  directory of `(offset, length)` growing forward, row payloads growing
  backward. Deleting a row tombstones its slot; `compact()` reclaims space.
  Locators `(page_id, slot)` are stable: compaction preserves slot indexes, so
  an index entry never has to move.
* **Buffer pool** — LRU with pin counts, dirty bits, and a write-before-evict
  contract. It knows nothing about the WAL; instead it calls an `on_flush`
  hook before every physical write, which is where the write-ahead rule is
  enforced for flushes that race ahead of the pager.
* **Pager** — the transactional view of pages: `alloc`, `read`, and `write`
  (which *stages* an uncommitted image), plus `stage`/`publish`/`abort`.

### B+ tree indexes

Order 8 (`MIN_KEYS = 4`), keys capped at 480 bytes so a worst-case entry
always fits a page. Leaves are chained by right-sibling pointers for range
scans. Uniqueness is enforced by the index, not the tree.

The interesting part is what an entry *is*:

> Every entry is a `(key, locator)` pair, and so is every separator in an
> internal node. Ordering by the pair — rather than by the bare key — is what
> lets a non-unique index hold thousands of copies of one value. A separator
> must sort *strictly* between the two children it divides; no bare key can do
> that when that key exists on both sides, and a leaf full of one repeated key
> could not be split at all. Because a locator identifies exactly one row
> version, `(key, locator)` is a strict total order with no ties, so a split
> point always exists. This is the same trick as PostgreSQL ordering a
> secondary index by `(key, ctid)` and InnoDB appending the primary key to
> secondary index entries.

So the tree tolerates duplicates everywhere — `verify()` checks that entries
are strictly ascending while allowing repeated keys — and the engine relies on
it: a non-unique index keeps an entry per row, so indexing a low-cardinality
column fills leaves with the same key, and `REINDEX` rebuilds a tree from a
heap scan that can hand it thousands of entries sharing one key. The one
precondition is that a `(key, locator)` pair is never inserted twice, which
the engine guarantees because a locator is allocated once per row version and
freed with it.

Keys are encoded so that **lexicographic byte order equals logical value
order**: `INT` big-endian, `FLOAT` IEEE-754 bits transformed with sign flips,
`DATE` ordinal, `TEXT` UTF-8 shifted up by one with a `0x00` terminator (so a
prefix sorts before its extensions and composite keys split on the
terminator). Prefix comparison is what makes a composite index usable for a
predicate on its leading column.

### Catalog

The database's own metadata, stored in two reserved pages (1 and 2) as JSON
with a SHA-256 checksum and a monotonically increasing epoch. Writes alternate
slots; the highest *valid* epoch wins at load. A crash mid-write always leaves
one intact slot, which is the standard way to get atomic metadata updates when
you can't rely on an atomic rename.

### Transactions, locking and recovery

* **Concurrency:** S/X locks at **table granularity**, strict two-phase
  locking (all locks released together at commit/abort). A blocked acquirer
  records a waits-for edge; a cycle means deadlock and the youngest
  transaction is aborted, like Postgres.
* **Buffer policy:** *no-steal* / *no-force*. Uncommitted pages never reach
  the data file, and committed pages are written lazily.
* **WAL:** `len(4) | lsn(4) | txn(8) | type(1) | payload | crc32(4)`.
  `REC_BEGIN`, `REC_DATA` (a batch of full page images), `REC_COMMIT`,
  `REC_ABORT`, `REC_CKPT`. The CRC covers the record body, so replay stops at
  the last intact record — a torn tail is detected rather than trusted.
* **Recovery is redo-only.** Because of no-steal there is nothing to undo:
  recovery scans the log, takes the *newest* image of each page from every
  committed transaction, and writes it back. `EXPLAIN STATS` reports the last
  recovery pass (`pages_redone`). `.checkpoint` flushes and truncates the log
  once everything is durable.

## Testing

109 tests, standard library plus pytest. Nothing in the storage, index or SQL
layers is faked: tests write real pages through a real buffer pool into a real
file. The only interception is in the two crash tests, which lose page flushes
at the worst possible moment and then reopen the database to see what
recovery does.

| File | Tests | Covers |
| --- | --: | --- |
| `test_storage.py` | 15 | slot stable across compaction, page-full handling, row codec round-trips, disk header integrity, buffer pool LRU/pinning/flush hook |
| `test_btree.py` | 18 | key encoding order-preservation, splits, borrow/merge deletes, range scans, duplicate-heavy random insert/delete against a reference model, durability across reopen |
| `test_sql.py` | 11 | lexing, parsing, AST shape, error messages |
| `test_engine.py` | 41 | DDL, DML, predicates, aggregates, joins, constraints, plans, reports, durability |
| `test_txn.py` | 12 | WAL framing, commit durability, redo recovery, checkpointing |
| `test_locks.py` | 5 | S/X compatibility, 2PL, deadlock detection |
| `test_cli.py` | 7 | REPL formatting, row vs. plan rendering, meta commands, error recovery |

```bash
.venv/bin/python -m pytest -q
```

The B+ tree tests are the ones that earn their keep: they run hundreds of
randomized inserts and deletes against a reference multiset and check the
full structural invariant (`verify()`) every 100 steps — node ordering,
separator bounds, sibling chain, reachability.

## Design decisions and trade-offs

| Decision | Why | Cost |
| --- | --- | --- |
| Redo-only WAL (no-steal/no-force) | Recovery needs no undo pass, and a crash never exposes uncommitted data | Long transactions hold more staged pages in memory |
| Table-level locks | Simple to reason about, no lock escalation, no row-lock table | One writer per table; coarse for OLTP |
| Full-page images in the WAL | Redo is idempotent and needs no page-format knowledge | Fat log records; ARIES-style physiological logging is the real answer |
| `(key, locator)` index order | Duplicate keys work with no special cases anywhere else | 6 extra bytes per entry, separators are slightly bigger |
| Fanout 8, keys capped at 480 B | A node image always fits a page by construction | Deeper trees than a real engine (fanout is sized from key widths there) |
| Pages never move, no vacuum | Stable addresses; no free-space map to get wrong | Deleted pages are never reclaimed |
| Dual-slot catalog with epochs | Atomic-ish metadata writes on plain files | Two pages of overhead; a torn *both*-slot write is still theoretically possible |
| Statement-level atomicity only | Enough for DDL/DML to be all-or-nothing without a general undo log | A failed statement inside a transaction rolls back the statement, not earlier ones |

## Deliberate omissions

Row-level locking and MVCC; a real cost-based optimizer (the planner is
rule-based: equality on a leading index column → `IndexScan`, else `SeqScan`);
subqueries, `ALTER TABLE`, and foreign keys; `NULL`-aware `NOT IN` semantics
(`NULLs` are never indexed, so a unique index simply skips them); free-space
reclamation for pages; and log-based point-in-time recovery (checkpoint
truncates rather than archiving).

## Layout

```
brickdb/
  __main__.py        python -m brickdb
  cli.py             REPL: formatting, meta commands
  engine.py          statement dispatch, planner, index maintenance, commit
  catalog.py         table/index metadata, dual-slot persistence
  buffer.py          LRU buffer pool
  heap.py            heap file over slotted pages
  errors.py          error hierarchy
  storage/           disk.py, page.py, codec.py, pager.py
  index/             btree.py, keyenc.py
  sql/               lexer.py, ast.py, parser.py
  executors/         operators.py (Volcano), eval.py (expressions, 3-valued logic)
  txn/               wal.py, locks.py
tests/               109 tests
```

## License

MIT.
