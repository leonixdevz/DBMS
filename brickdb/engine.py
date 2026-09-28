"""Database engine: the transactional statement dispatcher.

Owns the disk manager, buffer pool, WAL, lock manager, catalog, and the
pagers that stage per-transaction page writes.

Statement flow::

    parse -> plan -> execute (stage pages in the pager)
          -> commit:  WAL DATA+COMMIT (fsync) -> publish pages to the pool
             abort:   discard staged pages, rewind the in-memory catalog

Isolation: strict two-phase locking at table granularity (S for reads, X for
writes), released only at commit/rollback. Combined with the no-steal buffer
policy, a transaction never reads another transaction's uncommitted pages.

Statement atomicity: every statement snapshots the catalog and stages its
pages privately; any error discards both, so a failed multi-row INSERT or
UPDATE leaves no partial state.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .buffer import BufferPool
from .catalog import Catalog, ColumnDef, IndexDef, TableDef
from .errors import (
    BrickDBError,
    CatalogError,
    SQLError,
    TypeViolation,
)
from .executors import eval as E
from .executors.operators import (
    Distinct,
    Filter,
    GroupAggregate,
    IndexScan,
    Limit,
    NestedLoopJoin,
    Operator,
    Project,
    SeqScan,
    Sort,
    item_name,
)
from .heap import HeapFile
from .index.btree import BTree
from .index.keyenc import encode_key, encode_key_part, pack_payload
from .sql import ast as A
from .sql.parser import parse
from .storage.codec import coerce
from .storage.disk import DiskManager
from .storage.pager import Pager
from .txn.locks import LockManager, S, X
from .txn.wal import WriteAheadLog

_FLIP_OP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


class QueryResult:
    """Result of one statement: rows + columns, or a status message.

    ``plan`` carries the rendered plan of a query for introspection. For
    ``EXPLAIN`` the plan *is* the result, which ``explain`` marks so the shell
    prints it rather than an empty row set.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        rows: Optional[List[List[Any]]] = None,
        message: Optional[str] = None,
        plan: Optional[List[str]] = None,
        explain: bool = False,
    ):
        self.columns = columns or []
        self.rows = rows or []
        self.message = message
        self.plan = plan or []
        self.explain = explain

    @property
    def is_query(self) -> bool:
        return self.message is None

    def __repr__(self) -> str:
        if self.is_query:
            return f"QueryResult(cols={self.columns}, rows={len(self.rows)})"
        return f"QueryResult({self.message!r})"


class PlanNode:
    """A node of the executable plan tree, used for EXPLAIN output."""

    def __init__(self, label: str, *children: "PlanNode"):
        self.label = label
        self.children = children

    def render(self, depth: int = 0) -> List[str]:
        lines = ["  " * depth + self.label]
        for child in self.children:
            lines.extend(child.render(depth + 1))
        return lines


class Transaction:
    """State for an explicit BEGIN..COMMIT transaction."""

    def __init__(self, txn_id: int, pager: Pager):
        self.txn_id = txn_id
        self.pager = pager
        self.begun_logged = False


class Database:
    """Embedded database engine over a single ``*.db`` file."""

    def __init__(self, path: str, buffer_pool_size: int = 64):
        self.disk = DiskManager(path)
        self.wal_path = path[:-3] + ".wal" if path.endswith(".db") else path + ".wal"
        self.wal = WriteAheadLog(self.wal_path)
        self.pool = BufferPool(self.disk, size=buffer_pool_size, on_flush=self._before_page_flush)
        self.locks = LockManager()
        self.catalog = Catalog(Pager(self.disk, self.pool))
        self._txn: Optional[Transaction] = None
        self._txn_counter = 0
        self._mtx = threading.RLock()
        self._open = True
        # Set when an in-memory catalog change (e.g. a B+ tree root moving)
        # must be persisted in the committing transaction's page batch.
        self._catalog_dirty = False
        # --- crash recovery: redo committed transactions before serving reads
        self._recovery_stats = self.wal.recover(self.disk, force=True)
        # --- catalog pages 1 and 2 exist in every database file
        if self.disk.num_pages < 3:
            self.disk.allocate_page()
            self.disk.allocate_page()
        self.catalog.load()

    # --------------------------------------------------------- lifetime

    def close(self) -> None:
        """Flush everything, checkpoint the WAL, and close the files."""
        if not self._open:
            return
        if self._txn is not None:
            self.rollback()
        self.checkpoint()
        self.pool.close()
        self.wal.close()
        self.disk.close()
        self._open = False

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def checkpoint(self) -> None:
        """Flush dirty pages, sync, and truncate the log."""
        self.pool.flush_all()
        self.disk.sync()
        self.wal.checkpoint()

    def _before_page_flush(self, page_id: int, image: bytes) -> None:
        """Write-ahead rule: the log must be durable before any page write."""
        self.wal.sync()

    # ------------------------------------------------------- transactions

    def begin(self) -> None:
        if self._txn is not None:
            raise SQLError("a transaction is already in progress")
        self._txn = Transaction(self._next_txn_id(), Pager(self.disk, self.pool))

    def commit(self) -> None:
        if self._txn is None:
            raise SQLError("no transaction in progress")
        txn = self._txn
        self._txn = None
        try:
            self._commit_pager(txn.pager, txn)
        except BrickDBError:
            self._release(txn)
            raise
        self._release(txn)

    def rollback(self) -> None:
        if self._txn is None:
            raise SQLError("no transaction in progress")
        txn = self._txn
        self._txn = None
        txn.pager.abort()
        self._release(txn)
        self.catalog.load()

    def _release(self, txn: Transaction) -> None:
        self.locks.release_all(txn.txn_id)

    def _next_txn_id(self) -> int:
        with self._mtx:
            self._txn_counter += 1
            return self._txn_counter

    def _commit_pager(self, pager: Pager, txn: Optional[Transaction] = None) -> None:
        # Persist metadata changes (index roots from splits/merges, heap page
        # growth) atomically with the data pages they describe.
        if self._catalog_dirty:
            self.catalog.save(pager)
            self._catalog_dirty = False
        pages = pager.stage()
        if pages:
            txn_id = txn.txn_id if txn else getattr(pager, "txn_id", 0)
            if txn is not None and not txn.begun_logged:
                self.wal.log_begin(txn_id)
                txn.begun_logged = True
            # 1. Durably record the batch, 2. only then publish it.
            self.wal.log_commit_pages(txn_id, "data", pages)
            published = pager.publish(pages)
            # 3. Push it toward the data file (WAL sync happens in the hook).
            for pid in published:
                self.pool.flush_page(pid)
        else:
            pager.abort()
        if txn is not None:
            self._release(txn)
        else:
            self.locks.release_all(getattr(pager, "txn_id", 0))

    # ------------------------------------------------------------ execute

    def execute(self, sql: str) -> QueryResult:
        """Parse and execute one SQL statement."""
        if not self._open:
            raise BrickDBError("database is closed")
        stmt = parse(sql)
        with self._mtx:
            return self._execute_stmt(stmt)

    def _execute_stmt(self, stmt: A.Stmt) -> QueryResult:
        # Transaction control statements.
        if isinstance(stmt, A.BeginStmt):
            self.begin()
            return QueryResult(message="BEGIN")
        if isinstance(stmt, A.CommitStmt):
            self.commit()
            return QueryResult(message="COMMIT")
        if isinstance(stmt, A.RollbackStmt):
            self.rollback()
            return QueryResult(message="ROLLBACK")

        explicit = self._txn is not None
        txn = self._txn
        if txn is None:
            pager = Pager(self.disk, self.pool)
            pager.txn_id = self._next_txn_id()
        else:
            pager = txn.pager
        snapshot = self.catalog.snapshot()
        try:
            result = self._dispatch(stmt, pager)
        except BrickDBError:
            self.catalog.restore(snapshot)
            pager.abort()
            if explicit:
                # A failed statement aborts the whole explicit transaction.
                self._txn = None
                self.locks.release_all(txn.txn_id)  # type: ignore[union-attr]
            else:
                self.locks.release_all(pager.txn_id)
            raise
        if explicit:
            return result
        # Autocommit: publish this statement's pages.
        try:
            self._commit_pager(pager)
        except BrickDBError:
            self.catalog.restore(snapshot)
            self.locks.release_all(pager.txn_id)
            raise
        return result

    # ----------------------------------------------------------- dispatch

    def _dispatch(self, stmt: A.Stmt, pager: Pager) -> QueryResult:
        if isinstance(stmt, A.ExplainStmt):
            return self._explain(stmt, pager)
        if isinstance(stmt, A.CreateTableStmt):
            return self._create_table(stmt, pager)
        if isinstance(stmt, A.DropTableStmt):
            return self._drop_table(stmt, pager)
        if isinstance(stmt, A.CreateIndexStmt):
            return self._create_index(stmt, pager)
        if isinstance(stmt, A.DropIndexStmt):
            return self._drop_index(stmt, pager)
        if isinstance(stmt, A.ReindexStmt):
            return self._reindex(stmt, pager)
        if isinstance(stmt, A.SelectStmt):
            return self._select(stmt, pager)
        if isinstance(stmt, A.InsertStmt):
            return self._insert(stmt, pager)
        if isinstance(stmt, A.UpdateStmt):
            return self._update(stmt, pager)
        if isinstance(stmt, A.DeleteStmt):
            return self._delete(stmt, pager)
        raise SQLError(f"unsupported statement: {type(stmt).__name__}")

    # ------------------------------------------------------------- schema

    def _create_table(self, stmt: A.CreateTableStmt, pager: Pager) -> QueryResult:
        if not stmt.columns:
            raise SQLError("a table needs at least one column")
        seen = set()
        for spec in stmt.columns:
            if spec.name.lower() in seen:
                raise SQLError(f"duplicate column name {spec.name!r}")
            seen.add(spec.name.lower())
        tdef = TableDef(
            stmt.table,
            [
                ColumnDef(spec.name, spec.kind, spec.notnull or spec.primary_key, self._literal(spec.default))
                for spec in stmt.columns
            ],
        )
        self.catalog.add_table(tdef)
        # PRIMARY KEY / UNIQUE constraints become unique indexes.
        pk_cols = [s.name for s in stmt.columns if s.primary_key]
        if pk_cols:
            self.catalog.add_index(IndexDef(f"{tdef.name}_pkey", tdef.name, pk_cols, True, None))
        for spec in stmt.columns:
            if spec.unique and not spec.primary_key:
                self.catalog.add_index(IndexDef(f"{tdef.name}_{spec.name}_key", tdef.name, [spec.name], True, None))
        self.catalog.save(pager)
        return QueryResult(message=f"Table '{tdef.name}' created")

    def _literal(self, expr: Optional[A.Expr]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, A.Literal):
            return expr.value
        raise SQLError("DEFAULT must be a literal value")

    def _drop_table(self, stmt: A.DropTableStmt, pager: Pager) -> QueryResult:
        try:
            tdef = self.catalog.get_table(stmt.table)
        except CatalogError:
            if stmt.if_exists:
                return QueryResult(message=f"Table '{stmt.table}' does not exist")
            raise
        for idef in self.catalog.table_indexes(tdef.name):
            self.catalog.drop_index(idef.name)
        self.catalog.drop_table(tdef.name)
        self.catalog.save(pager)
        return QueryResult(message=f"Table '{tdef.name}' dropped")

    def _create_index(self, stmt: A.CreateIndexStmt, pager: Pager) -> QueryResult:
        if not stmt.columns:
            raise SQLError("an index needs at least one column")
        tdef = self.catalog.get_table(stmt.table)
        for col in stmt.columns:
            tdef.column_index(col)  # raises if unknown
        self.catalog.add_index(IndexDef(stmt.index, tdef.name, list(stmt.columns), stmt.unique, None))
        idef = self.catalog.get_index(stmt.index)
        self._lock(pager, tdef.name, X)
        heap = self._heap(pager, tdef)
        tree = BTree(pager, None, idef.unique)
        for locator, row in heap.scan():
            key_values = [row[tdef.column_index(c)] for c in idef.columns]
            if any(v is None for v in key_values):
                continue  # NULLs are not indexed
            tree.insert(encode_key(self._kinds(tdef, idef), key_values), locator, unique=idef.unique)
        idef.root = tree.root
        self.catalog.save(pager)
        n = tree.count_keys()
        return QueryResult(message=f"Index '{idef.name}' created on {tdef.name}({', '.join(idef.columns)}) [{n} entries]")

    def _drop_index(self, stmt: A.DropIndexStmt, pager: Pager) -> QueryResult:
        try:
            idef = self.catalog.get_index(stmt.index)
        except CatalogError:
            if stmt.if_exists:
                return QueryResult(message=f"Index '{stmt.index}' does not exist")
            raise
        self.catalog.drop_index(idef.name)
        self.catalog.save(pager)
        return QueryResult(message=f"Index '{idef.name}' dropped")

    def _reindex(self, stmt: A.ReindexStmt, pager: Pager) -> QueryResult:
        tdef = self.catalog.get_table(stmt.table)
        self._lock(pager, tdef.name, X)
        heap = self._heap(pager, tdef)
        rows = [(loc, row) for loc, row in heap.scan()]
        for idef in self.catalog.table_indexes(tdef.name):
            tree = BTree(pager, None, idef.unique)
            for loc, row in rows:
                key_values = [row[tdef.column_index(c)] for c in idef.columns]
                if any(v is None for v in key_values):
                    continue
                tree.insert(encode_key(self._kinds(tdef, idef), key_values), loc, unique=idef.unique)
            idef.root = tree.root
        self.catalog.save(pager)
        return QueryResult(message=f"Rebuilt {len(self.catalog.table_indexes(tdef.name))} index(es) on '{tdef.name}'")

    def _heap(self, pager: Pager, tdef: TableDef) -> HeapFile:
        """Heap access that keeps the catalog in sync with page allocation."""
        return HeapFile(pager, tdef, on_grow=self._mark_catalog_dirty)

    def _mark_catalog_dirty(self) -> None:
        self._catalog_dirty = True

    def _kinds(self, tdef: TableDef, idef: IndexDef) -> List[str]:
        return [tdef.columns[tdef.column_index(c)].kind for c in idef.columns]

    # -------------------------------------------------------------- DML

    def _insert(self, stmt: A.InsertStmt, pager: Pager) -> QueryResult:
        tdef = self.catalog.get_table(stmt.table)
        self._lock(pager, tdef.name, X)
        heap = self._heap(pager, tdef)
        count = 0
        for row_exprs in stmt.rows:
            values = self._prepare_values(tdef, stmt.columns, row_exprs)
            locator = heap.insert(values)
            self._index_insert(pager, tdef, locator, values)
            count += 1
        return QueryResult(message=f"{count} row{'s' if count != 1 else ''} inserted")

    def _prepare_values(self, tdef: TableDef, columns: List[str], exprs: List[A.Expr]) -> List[Any]:
        values: List[Any] = [None] * tdef.ncols
        if columns:
            if len(columns) != len(exprs):
                raise SQLError("INSERT column list and VALUES list have different lengths")
            for cname, vexpr in zip(columns, exprs):
                idx = tdef.column_index(cname)
                values[idx] = self._eval_constant(vexpr)
        else:
            if len(exprs) != tdef.ncols:
                raise SQLError(f"table '{tdef.name}' has {tdef.ncols} columns but {len(exprs)} values were given")
            values = [self._eval_constant(v) for v in exprs]
        for i, col in enumerate(tdef.columns):
            if values[i] is None:
                values[i] = col.default
            values[i] = coerce(values[i], col.kind, col.name)
        return values

    def _eval_constant(self, expr: A.Expr) -> Any:
        empty = E.Row({}, {})
        return E.evaluate(expr, empty)

    def _update(self, stmt: A.UpdateStmt, pager: Pager) -> QueryResult:
        tdef = self.catalog.get_table(stmt.table)
        self._lock(pager, tdef.name, X)
        heap = self._heap(pager, tdef)
        alias = tdef.name.lower()
        colnames = [c.name.lower() for c in tdef.columns]
        assignments = [(tdef.column_index(c), v) for c, v in stmt.assignments]
        count = 0
        for locator, row in list(self._matching_rows(tdef, stmt.where, pager)):
            bound = E.Row({alias: row}, {alias: colnames})
            new_values = list(row)
            for idx, vexpr in assignments:
                new_values[idx] = E.evaluate(vexpr, bound)
            for i, col in enumerate(tdef.columns):
                new_values[i] = coerce(new_values[i], col.kind, col.name)
                if col.notnull and new_values[i] is None:
                    raise SQLError(f"column {col.name!r} cannot be NULL")
            self._index_delete(pager, tdef, locator, row)
            heap.delete(locator)
            new_loc = heap.insert(new_values)
            self._index_insert(pager, tdef, new_loc, new_values)
            count += 1
        return QueryResult(message=f"{count} row{'s' if count != 1 else ''} updated")

    def _delete(self, stmt: A.DeleteStmt, pager: Pager) -> QueryResult:
        tdef = self.catalog.get_table(stmt.table)
        self._lock(pager, tdef.name, X)
        heap = self._heap(pager, tdef)
        count = 0
        for locator, row in list(self._matching_rows(tdef, stmt.where, pager)):
            self._index_delete(pager, tdef, locator, row)
            heap.delete(locator)
            count += 1
        return QueryResult(message=f"{count} row{'s' if count != 1 else ''} deleted")

    def _matching_rows(
        self, tdef: TableDef, where: Optional[A.Expr], pager: Pager
    ) -> Iterator[Tuple[Tuple[int, int], List]]:
        """Yield (locator, row) for rows satisfying WHERE, using an index
        when the predicate starts with an equality/range on an indexed column."""
        heap = self._heap(pager, tdef)
        alias = tdef.name.lower()
        colnames = [c.name.lower() for c in tdef.columns]
        plan = self._pick_index(tdef, where)
        if plan is not None:
            idef, lo, hi, lo_inc, hi_inc = plan
            tree = BTree(pager, idef.root, idef.unique)
            for locator in tree.range_scan_trunc(lo, lo_inc, hi, hi_inc):
                row = heap.get(locator)
                if row is None:
                    continue
                if where is None or E.evaluate(where, E.Row({alias: row}, {alias: colnames})):
                    yield locator, row
            return
        for locator, row in heap.scan():
            if where is None or E.evaluate(where, E.Row({alias: row}, {alias: colnames})):
                yield locator, row

    # -------------------------------------------------------- index upkeep

    def _index_insert(self, pager: Pager, tdef: TableDef, locator, values: List[Any]) -> None:
        for idef in self.catalog.table_indexes(tdef.name):
            key_values = [values[tdef.column_index(c)] for c in idef.columns]
            if any(v is None for v in key_values):
                continue
            tree = BTree(pager, idef.root, idef.unique)
            tree.insert(encode_key(self._kinds(tdef, idef), key_values), locator, unique=idef.unique)
            if tree.root != idef.root:
                idef.root = tree.root  # a root split moves the entry point
                self._catalog_dirty = True

    def _index_delete(self, pager: Pager, tdef: TableDef, locator, row: List[Any]) -> None:
        for idef in self.catalog.table_indexes(tdef.name):
            key_values = [row[tdef.column_index(c)] for c in idef.columns]
            if any(v is None for v in key_values):
                continue
            tree = BTree(pager, idef.root, idef.unique)
            payload = None if idef.unique else pack_payload(*locator)
            tree.delete(encode_key(self._kinds(tdef, idef), key_values), payload)
            if tree.root != idef.root:
                idef.root = tree.root  # shrinking can promote a new root
                self._catalog_dirty = True

    # ------------------------------------------------------------ locking

    def _lock(self, pager: Pager, table: str, mode: str) -> None:
        txn_id = self._txn.txn_id if self._txn is not None else pager.txn_id
        self.locks.acquire(txn_id, f"table:{table.lower()}", mode)

    # ------------------------------------------------------------- select

    def _select(self, stmt: A.SelectStmt, pager: Pager) -> QueryResult:
        op, plan, out_names = self._build_select(stmt, pager)
        rows = op.all_rows()
        out = [[v for values in row.tables.values() for v in values] for row in rows]
        return QueryResult(columns=out_names, rows=out, plan=plan.render())

    def _build_select(self, stmt: A.SelectStmt, pager: Pager):
        """Return (operator, PlanNode, output column names)."""
        if stmt.from_table is None:
            values = []
            for item in stmt.items:
                if isinstance(item.expr, A.Star):
                    raise SQLError("SELECT * requires a FROM clause")
                values.append(E.evaluate(item.expr, E.Row({}, {})))
            node = _LiteralRow(values)
            names = [item_name(i) for i in stmt.items]
            return node, PlanNode(f"Constant [{', '.join(names)}]"), names

        _bind_columns(stmt, self.catalog)
        tdef = self.catalog.get_table(stmt.from_table.name)
        self._lock(pager, tdef.name, S)
        heap = self._heap(pager, tdef)
        colnames = [c.name for c in tdef.columns]
        out_names = _out_names(self.catalog, stmt)

        # --- choose the access path
        plan = self._pick_index(tdef, stmt.where) if not stmt.joins else None
        if plan is not None:
            idef, lo, hi, lo_inc, hi_inc = plan
            op: Operator = IndexScan(
                BTree(pager, idef.root, idef.unique), heap, stmt.from_table, colnames, lo, hi, lo_inc, hi_inc
            )
            node = PlanNode(f"IndexScan {idef.name} on {tdef.name}")
        else:
            op = SeqScan(heap, stmt.from_table, colnames)
            node = PlanNode(f"SeqScan {tdef.name} ({len(tdef.heap_pages)} page(s))")

        if stmt.where is not None:
            op = Filter(op, stmt.where)
            node = PlanNode(f"Filter {_expr_str(stmt.where)}", node)

        for j in stmt.joins:
            jt = self.catalog.get_table(j.table.name)
            self._lock(pager, jt.name, S)
            jheap = self._heap(pager, jt)
            jscan = SeqScan(jheap, j.table, [c.name for c in jt.columns])
            op = NestedLoopJoin(op, jscan, j.kind, j.on)
            node = PlanNode(f"NestedLoopJoin {j.kind}", node, PlanNode(f"SeqScan {j.table.name}"))

        has_agg = bool(stmt.group_by) or any(isinstance(i.expr, A.FuncCall) for i in stmt.items)

        if has_agg:
            op = GroupAggregate(op, stmt.items, stmt.group_by)
            label = "GroupAggregate" + (
                " group by " + ", ".join(_expr_str(g) for g in stmt.group_by) if stmt.group_by else ""
            )
            node = PlanNode(label, node)
            if stmt.order_by:
                items = _rewrite_order(stmt.order_by, stmt.items, mode="output")
                op = Sort(op, items)
                node = PlanNode("Sort " + _order_str(stmt.order_by), node)
        else:
            if stmt.order_by and not stmt.distinct:
                items = _rewrite_order(stmt.order_by, stmt.items, mode="source")
                op = Sort(op, items)
                node = PlanNode("Sort " + _order_str(stmt.order_by), node)
            op = Project(op, _Binder(self.catalog, stmt), stmt.items, out_names)
            node = PlanNode(f"Project [{', '.join(out_names)}]", node)
            if stmt.distinct:
                op = Distinct(op)
                node = PlanNode("Distinct", node)
                if stmt.order_by:
                    items = _rewrite_order(stmt.order_by, stmt.items, mode="output")
                    op = Sort(op, items)
                    node = PlanNode("Sort " + _order_str(stmt.order_by), node)

        if stmt.limit is not None or stmt.offset:
            op = Limit(op, stmt.limit, stmt.offset)
            node = PlanNode(f"Limit {stmt.limit if stmt.limit is not None else 'all'} offset {stmt.offset}", node)

        return op, node, out_names

    # -------------------------------------------------------------- index

    def _pick_index(self, tdef: TableDef, where: Optional[A.Expr]):
        """Pick an index for WHERE if the first column has an equality or
        range predicate. Returns (idef, lo, hi, lo_inclusive, hi_inclusive)."""
        if where is None:
            return None
        eqs: Dict[str, Any] = {}
        ranges: Dict[str, list] = {}
        if not _collect_predicates(where, eqs, ranges):
            return None
        for idef in self.catalog.table_indexes(tdef.name):
            first = idef.columns[0]
            kind = self._kinds(tdef, idef)[0]
            fl = first.lower()
            if fl in eqs:
                bound = self._encode_bound(kind, eqs[fl], first)
                if bound is None:
                    continue
                return idef, bound, bound, True, True
            if fl in ranges:
                lo, lo_inc, hi, hi_inc = ranges[fl]
                lo_b = self._encode_bound(kind, lo, first) if lo is not None else None
                hi_b = self._encode_bound(kind, hi, first) if hi is not None else None
                if lo is not None and lo_b is None:
                    continue
                if hi is not None and hi_b is None:
                    continue
                if lo_b is None and hi_b is None:
                    continue
                return idef, lo_b, hi_b, lo_inc, hi_inc
        return None

    def _encode_bound(self, kind: str, value: Any, colname: str) -> Optional[bytes]:
        if value is None:
            return None
        try:
            return encode_key_part(kind, coerce(value, kind, colname))
        except (TypeViolation, BrickDBError):
            return None

    # ------------------------------------------------------------ reports

    def _explain(self, stmt: A.ExplainStmt, pager: Pager) -> QueryResult:
        if stmt.report == "table":
            return self._explain_table(stmt.arg, pager)
        if stmt.report == "index":
            return self._explain_index(stmt.arg, pager)
        if stmt.report == "stats":
            return self._explain_stats()
        target = stmt.target
        if isinstance(target, A.SelectStmt):
            _op, plan, _names = self._build_select(target, pager)
            lines = plan.render()
            return QueryResult(columns=["plan"], rows=[], plan=lines, explain=True)
        raise SQLError("EXPLAIN supports SELECT and TABLE/INDEX/STATS reports")

    def _explain_table(self, name: str, pager: Pager) -> QueryResult:
        tdef = self.catalog.get_table(name)
        stats = self._heap(pager, tdef).stats()
        idxs = self.catalog.table_indexes(tdef.name)
        desc = ", ".join(
            f"{c.name} {c.kind}" + (" NOT NULL" if c.notnull else "") for c in tdef.columns
        )
        return QueryResult(
            columns=["property", "value"],
            rows=[
                ["table", tdef.name],
                ["columns", desc],
                ["heap_pages", str(stats["pages"])],
                ["rows", str(stats["rows"])],
                ["dead_slots", str(stats["dead_slots"])],
                ["indexes", ", ".join(i.name for i in idxs) or "(none)"],
            ],
        )

    def _explain_index(self, name: str, pager: Pager) -> QueryResult:
        idef = self.catalog.get_index(name)
        tree = BTree(pager, idef.root, idef.unique)
        problems = tree.verify()
        return QueryResult(
            columns=["property", "value"],
            rows=[
                ["index", idef.name],
                ["table", idef.table],
                ["columns", ", ".join(idef.columns)],
                ["unique", str(idef.unique)],
                ["root_page", str(idef.root)],
                ["height", str(tree.height())],
                ["entries", str(tree.count_keys())],
                ["check", "ok" if not problems else "; ".join(problems)],
            ],
        )

    def _explain_stats(self) -> QueryResult:
        b = self.pool.stats()
        w = self.wal.stats()
        d = self.disk.stats()
        lk = self.locks.stats()
        return QueryResult(
            columns=["component", "stats"],
            rows=[
                ["buffer_pool", f"resident={b['resident']}/{b['capacity']} hit_rate={b['hit_rate']:.2f} flushes={b['flushes']}"],
                ["wal", f"lsn={w['lsn']} bytes={w['file_bytes']}"],
                ["disk", f"pages={d['pages']} file_bytes={d['file_bytes']} page_size={d['page_size']}"],
                ["locks", f"locked={lk['locked_resources']} waiters={lk['waiting_edges']}"],
                ["recovery", str(self._recovery_stats)],
            ],
        )

    # ------------------------------------------------------------ helpers

    def recovery_stats(self) -> Dict[str, Any]:
        return self._recovery_stats

    def tables(self) -> List[str]:
        return [t.name for t in self.catalog.tables.values()]

    def indexes(self) -> List[str]:
        return [i.name for i in self.catalog.indexes.values()]

    def schema(self, table: str) -> List[str]:
        tdef = self.catalog.get_table(table)
        lines = [f"CREATE TABLE {tdef.name} ("]
        for i, col in enumerate(tdef.columns):
            comma = "," if i < len(tdef.columns) - 1 else ""
            flags = " NOT NULL" if col.notnull else ""
            lines.append(f"  {col.name} {col.kind}{flags}{comma}")
        lines.append(");")
        for idef in self.catalog.table_indexes(tdef.name):
            uniq = "UNIQUE " if idef.unique else ""
            lines.append(f"CREATE {uniq}INDEX {idef.name} ON {tdef.name}({', '.join(idef.columns)});")
        return lines

    def stats(self) -> Dict[str, Any]:
        return {
            "tables": self.tables(),
            "indexes": self.indexes(),
            "buffer": self.pool.stats(),
            "wal": self.wal.stats(),
            "disk": self.disk.stats(),
            "locks": self.locks.stats(),
            "recovery": self._recovery_stats,
        }


class _LiteralRow(Operator):
    """One row of constant expressions (``SELECT 1 + 1``)."""

    def __init__(self, values: List[Any]):
        self.values = values
        self._done = False

    def open(self) -> None:
        self._done = False

    def next(self) -> Optional[E.Row]:
        if self._done:
            return None
        self._done = True
        return E.Row({"": list(self.values)}, {"": [""] * len(self.values)})

    def close(self) -> None:
        pass


class _Binder:
    """Expands ``*`` select items against the row currently flowing."""

    def __init__(self, catalog: Catalog, stmt: A.SelectStmt):
        self.catalog = catalog
        self.stmt = stmt

    def expand_star(self, star: A.Star, row: E.Row) -> List[Any]:
        if star.table is None:
            return [v for values in row.tables.values() for v in values]
        alias = star.table.lower()
        if alias not in row.tables:
            raise SQLError(f"unknown table or alias {star.table!r}")
        return list(row.tables[alias])


def _iter_column_refs(expr: Optional[A.Expr]) -> Iterator[A.ColumnRef]:
    """Yield every :class:`A.ColumnRef` inside ``expr``, at any depth."""
    if expr is None:
        return
    if isinstance(expr, A.ColumnRef):
        yield expr
        return
    if isinstance(expr, (A.Star, A.Literal, A.Param)):
        return
    for f in dataclasses.fields(expr):
        value = getattr(expr, f.name)
        if isinstance(value, A.Expr):
            yield from _iter_column_refs(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, A.Expr):
                    yield from _iter_column_refs(item)


def _bind_columns(stmt: A.SelectStmt, catalog: Catalog) -> None:
    """Resolve every column reference in ``stmt`` before any row is read.

    Resolution used to happen while evaluating a row, so a misspelled or
    ambiguous name only failed if the scan produced a row — ``SELECT nope
    FROM empty_table`` returned an empty result instead of an error. Binding
    against the catalog first makes the mistake fail the same way every time;
    the probe row is all NULLs and is only ever used to resolve names.
    """
    tables: Dict[str, List] = {}
    colnames: Dict[str, List[str]] = {}
    for tref in [stmt.from_table] + [j.table for j in stmt.joins]:
        tdef = catalog.get_table(tref.name)
        alias = (tref.alias or tref.name).lower()
        tables[alias] = [None] * tdef.ncols
        colnames[alias] = [c.name.lower() for c in tdef.columns]
    probe = E.Row(tables, colnames)

    for item in stmt.items:
        if isinstance(item.expr, A.Star):
            if item.expr.table is not None and item.expr.table.lower() not in tables:
                raise SQLError(f"unknown table or alias {item.expr.table!r}")
        else:
            for ref in _iter_column_refs(item.expr):
                E.lookup(probe, ref)
    for expr in [stmt.where] + list(stmt.group_by) + [j.on for j in stmt.joins]:
        for ref in _iter_column_refs(expr):
            E.lookup(probe, ref)
    for oi in stmt.order_by:
        # An ORDER BY that names a select item (or its alias) is already
        # checked above; anything else must be a source column.
        if _match_select_item(oi.expr, stmt.items) is not None:
            continue
        for ref in _iter_column_refs(oi.expr):
            E.lookup(probe, ref)


def _out_names(catalog: Catalog, stmt: A.SelectStmt) -> List[str]:
    if stmt.from_table is None:
        return [item_name(i) for i in stmt.items]
    refs = [stmt.from_table] + [j.table for j in stmt.joins]
    names: List[str] = []
    for item in stmt.items:
        if isinstance(item.expr, A.Star):
            for tref in refs:
                if item.expr.table is None or (tref.alias or tref.name).lower() == item.expr.table.lower():
                    tdef = catalog.get_table(tref.name)
                    names.extend(c.name for c in tdef.columns)
        else:
            names.append(item_name(item))
    return names


def _collect_predicates(where: A.Expr, eqs: Dict[str, Any], ranges: Dict[str, list]) -> bool:
    """Collect equality/range predicates on columns from an AND tree.

    Returns True if at least one usable predicate was found. Predicates that
    cannot drive an index (OR, LIKE, IS NULL, …) are simply skipped — the
    Filter operator still enforces the full WHERE clause.
    """
    if isinstance(where, A.BinaryOp) and where.op == "AND":
        left = _collect_predicates(where.left, eqs, ranges)
        right = _collect_predicates(where.right, eqs, ranges)
        return left or right
    if isinstance(where, A.BinaryOp) and where.op in ("=", "<", "<=", ">", ">="):
        if isinstance(where.left, A.ColumnRef) and isinstance(where.right, A.Literal):
            col, value, op = where.left.name.lower(), where.right.value, where.op
        elif isinstance(where.right, A.ColumnRef) and isinstance(where.left, A.Literal):
            col, value, op = where.right.name.lower(), where.left.value, _FLIP_OP[where.op]
        else:
            return False
        if value is None:
            return False
        if op == "=":
            eqs[col] = value
        else:
            r = ranges.setdefault(col, [None, True, None, True])
            if op in ("<", "<="):
                r[2], r[3] = value, op == "<="
            else:
                r[0], r[1] = value, op == ">="
        return True
    if isinstance(where, A.Between):
        if (
            isinstance(where.operand, A.ColumnRef)
            and isinstance(where.low, A.Literal)
            and isinstance(where.high, A.Literal)
            and where.low.value is not None
            and where.high.value is not None
        ):
            r = ranges.setdefault(where.operand.name.lower(), [None, True, None, True])
            r[0], r[1] = where.low.value, True
            r[2], r[3] = where.high.value, True
            return True
    return False


def _sig(expr: A.Expr) -> str:
    """Canonical string for an expression, used to match ORDER BY to items."""
    if isinstance(expr, A.ColumnRef):
        return f"{(expr.table or '').lower()}.{expr.name.lower()}"
    if isinstance(expr, A.Literal):
        return repr(expr.value)
    if isinstance(expr, A.FuncCall):
        if expr.star:
            return f"{expr.name.lower()}(*)"
        return f"{expr.name.lower()}({','.join(_sig(a) for a in expr.args)})"
    if isinstance(expr, A.BinaryOp):
        return f"({_sig(expr.left)}{expr.op}{_sig(expr.right)})"
    if isinstance(expr, A.UnaryOp):
        return f"({expr.op}{_sig(expr.operand)})"
    if isinstance(expr, A.Star):
        return "*"
    return repr(expr)


def _match_select_item(expr: A.Expr, items: List[A.SelectItem]) -> Optional[A.SelectItem]:
    if isinstance(expr, A.ColumnRef) and expr.table is None:
        for it in items:
            if it.alias and it.alias.lower() == expr.name.lower():
                return it
    key = _sig(expr)
    for it in items:
        if _sig(it.expr) == key:
            return it
    return None


def _rewrite_order(order_by: List[A.OrderItem], items: List[A.SelectItem], mode: str) -> List[A.OrderItem]:
    """Rewrite ORDER BY expressions so they can be evaluated at their stage.

    ``mode='source'``: sort happens below the projection, so an ORDER BY on
    an output alias is replaced by the aliased expression, and any plain
    column reference (including unselected ones) is left alone.

    ``mode='output'``: sort happens above the projection (aggregates or
    DISTINCT), so every ORDER BY item must correspond to a select item; it is
    replaced by a reference to that output column.
    """
    out: List[A.OrderItem] = []
    for oi in order_by:
        item = _match_select_item(oi.expr, items)
        if item is None:
            if mode == "output":
                raise SQLError(
                    f"ORDER BY {_sig(oi.expr)!r} must reference a select-list item "
                    "when using DISTINCT or aggregates"
                )
            out.append(oi)
            continue
        if mode == "output":
            out.append(A.OrderItem(A.ColumnRef(None, item_name(item)), oi.desc))
        else:
            out.append(A.OrderItem(item.expr, oi.desc))
    return out


def _order_str(order_by: List[A.OrderItem]) -> str:
    return ", ".join(_expr_str(o.expr) + (" DESC" if o.desc else "") for o in order_by)


def _expr_str(expr: A.Expr) -> str:
    if isinstance(expr, A.ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if isinstance(expr, A.Literal):
        return "NULL" if expr.value is None else repr(expr.value)
    if isinstance(expr, A.BinaryOp):
        return f"{_expr_str(expr.left)} {expr.op} {_expr_str(expr.right)}"
    if isinstance(expr, A.UnaryOp):
        return f"{expr.op} {_expr_str(expr.operand)}"
    if isinstance(expr, A.FuncCall):
        if expr.star:
            return f"{expr.name}(*)"
        return f"{expr.name}({', '.join(_expr_str(a) for a in expr.args)})"
    if isinstance(expr, A.IsNull):
        return f"{_expr_str(expr.operand)} IS {'NOT ' if expr.negated else ''}NULL"
    if isinstance(expr, A.Between):
        return f"{_expr_str(expr.operand)} BETWEEN {_expr_str(expr.low)} AND {_expr_str(expr.high)}"
    if isinstance(expr, A.InList):
        return f"{_expr_str(expr.operand)} IN (...)"
    if isinstance(expr, A.Star):
        return "*"
    return type(expr).__name__
