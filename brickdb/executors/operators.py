"""Volcano-style iterator operators.

Every operator implements ``open()`` / ``next()`` / ``close()`` and pulls
tuples from its children one at a time — the classic "volcano" model. This
gives us pipelining (a LIMIT stops the scan early instead of reading the
whole table) and makes adding operators (hash join, top-N sort) additive.

Rows flowing between operators are :class:`~brickdb.executors.eval.Row`
objects: ``tables: alias -> values`` plus ``colnames: alias -> [names]`` so
joined tuples carry per-table columns and expressions can be qualified.

Sort and GroupAggregate are blocking; everything else streams.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from ..errors import SQLError
from ..sql import ast as A
from .eval import Aggregator, Row, evaluate, sort_key


class Operator:
    """Base class for all operators."""

    def open(self) -> None:
        pass

    def next(self) -> Optional[Row]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def all_rows(self) -> List[Row]:
        self.open()
        try:
            out = []
            while True:
                row = self.next()
                if row is None:
                    return out
                out.append(row)
        finally:
            self.close()

    def null_row(self) -> Dict[str, List]:
        """A row-shaped all-NULL value set (used by LEFT JOIN)."""
        return {}

    def null_colnames(self) -> Dict[str, List[str]]:
        return {}


class SeqScan(Operator):
    """Full heap scan over one table."""

    def __init__(self, heap, tref: A.TableRef, colnames: List[str]):
        self.heap = heap
        self.tref = tref
        self.colnames = [c.lower() for c in colnames]
        self._it: Optional[Iterator[Tuple[Tuple[int, int], List]]] = None
        self.alias = (tref.alias or tref.name).lower()

    def open(self) -> None:
        self._it = iter(self.heap.scan())

    def next(self) -> Optional[Row]:
        try:
            _loc, values = next(self._it)  # type: ignore[union-attr]
        except StopIteration:
            return None
        return Row({self.alias: values}, {self.alias: list(self.colnames)})

    def close(self) -> None:
        self._it = None

    def null_row(self) -> Dict[str, List]:
        return {self.alias: [None] * self.heap.tdef.ncols}

    def null_colnames(self) -> Dict[str, List[str]]:
        return {self.alias: list(self.colnames)}


class IndexScan(Operator):
    """Scan rows via a B+ tree range lookup, fetching each row by locator.

    ``lo``/``hi`` are encoded key bounds; ``lo_inclusive``/``hi_inclusive``
    control endpoint semantics. Both None means a full index scan.
    """

    def __init__(self, btree, heap, tref: A.TableRef, colnames: List[str],
                 lo: Optional[bytes] = None, hi: Optional[bytes] = None,
                 lo_inclusive: bool = True, hi_inclusive: bool = True):
        self.btree = btree
        self.heap = heap
        self.tref = tref
        self.colnames = [c.lower() for c in colnames]
        self.alias = (tref.alias or tref.name).lower()
        self.lo = lo
        self.hi = hi
        self.lo_inclusive = lo_inclusive
        self.hi_inclusive = hi_inclusive
        self._it: Optional[Iterator[Tuple[int, int]]] = None

    def open(self) -> None:
        self._it = self.btree.range_scan_trunc(
            self.lo, self.lo_inclusive, self.hi, self.hi_inclusive
        )

    def next(self) -> Optional[Row]:
        try:
            page_id, slot = next(self._it)  # type: ignore[union-attr]
        except StopIteration:
            return None
        values = self.heap.get((page_id, slot))
        if values is None:
            return self.next()  # stale index entry; skip
        return Row({self.alias: values}, {self.alias: list(self.colnames)})

    def close(self) -> None:
        self._it = None

    def null_row(self) -> Dict[str, List]:
        return {self.alias: [None] * self.heap.tdef.ncols}

    def null_colnames(self) -> Dict[str, List[str]]:
        return {self.alias: list(self.colnames)}


class Filter(Operator):
    def __init__(self, child: Operator, where: A.Expr):
        self.child = child
        self.where = where

    def open(self) -> None:
        self.child.open()

    def next(self) -> Optional[Row]:
        while True:
            row = self.child.next()
            if row is None:
                return None
            if evaluate(self.where, row):
                return row

    def close(self) -> None:
        self.child.close()

    def null_row(self) -> Dict[str, List]:
        return self.child.null_row()

    def null_colnames(self) -> Dict[str, List[str]]:
        return self.child.null_colnames()


class Project(Operator):
    """Computes the select list. Handles ``*`` expansion via the binder."""

    def __init__(self, child: Operator, binder, items: List[A.SelectItem], out_names: List[str]):
        self.child = child
        self.binder = binder
        self.items = items
        self.out_names = out_names

    def open(self) -> None:
        self.child.open()

    def next(self) -> Optional[Row]:
        row = self.child.next()
        if row is None:
            return None
        values: List[object] = []
        for item in self.items:
            expr = item.expr
            if isinstance(expr, A.Star):
                values.extend(self.binder.expand_star(expr, row))
            else:
                values.append(evaluate(expr, row))
        return Row({"": values}, {"": [n.lower() for n in self.out_names]})

    def close(self) -> None:
        self.child.close()


class NestedLoopJoin(Operator):
    """Tuple-at-a-time nested loop join supporting INNER and LEFT OUTER."""

    def __init__(self, left: Operator, right: Operator, kind: str, on: Optional[A.Expr]):
        self.left = left
        self.right = right
        self.kind = kind
        self.on = on
        self._left_row: Optional[Row] = None
        self._left_done = False
        self._matched = False

    def open(self) -> None:
        self.left.open()
        self._left_row = None
        self._left_done = False

    def next(self) -> Optional[Row]:
        while True:
            if self._left_done:
                return None
            if self._left_row is None:
                self._left_row = self.left.next()
                if self._left_row is None:
                    self._left_done = True
                    return None
                self._matched = False
                self.right.open()
            right_row = self.right.next()
            if right_row is None:
                self.right.close()
                if self.kind == "LEFT" and not self._matched:
                    out_tables = dict(self._left_row.tables)
                    out_names = dict(self._left_row.colnames)
                    out_tables.update(self.right.null_row())
                    out_names.update(self.right.null_colnames())
                    self._left_row = None
                    return Row(out_tables, out_names)
                self._left_row = None
                continue
            merged_tables = dict(self._left_row.tables)
            merged_names = dict(self._left_row.colnames)
            merged_tables.update(right_row.tables)
            merged_names.update(right_row.colnames)
            out = Row(merged_tables, merged_names)
            if self.on is None or evaluate(self.on, out):
                self._matched = True
                return out

    def close(self) -> None:
        self.left.close()
        self.right.close()

    def null_row(self) -> Dict[str, List]:
        out = dict(self.left.null_row())
        out.update(self.right.null_row())
        return out

    def null_colnames(self) -> Dict[str, List[str]]:
        out = dict(self.left.null_colnames())
        out.update(self.right.null_colnames())
        return out


class Sort(Operator):
    """Blocking sort. Multi-key ordering via stable last-to-first passes."""

    def __init__(self, child: Operator, order_by: List[A.OrderItem]):
        self.child = child
        self.order_by = order_by
        self._rows: Optional[List[Row]] = None
        self._i = 0

    def open(self) -> None:
        if self._rows is not None:
            return
        rows = self.child.all_rows()
        for item in reversed(self.order_by):
            rows.sort(key=lambda r: sort_key(evaluate(item.expr, r)), reverse=item.desc)
        self._rows = rows
        self._i = 0

    def next(self) -> Optional[Row]:
        if self._rows is None:
            self.open()
        if self._i >= len(self._rows):  # type: ignore[operator]
            return None
        row = self._rows[self._i]  # type: ignore[index]
        self._i += 1
        return row

    def close(self) -> None:
        pass


class Distinct(Operator):
    """Blocking deduplication of the child's output."""

    def __init__(self, child: Operator):
        self.child = child
        self._rows: Optional[List[Row]] = None
        self._i = 0

    @staticmethod
    def _key(row: Row):
        return tuple(tuple(v) for v in row.tables.values())

    def open(self) -> None:
        if self._rows is not None:
            return
        seen = set()
        out = []
        self.child.open()
        while True:
            row = self.child.next()
            if row is None:
                break
            k = self._key(row)
            if k not in seen:
                seen.add(k)
                out.append(row)
        self.child.close()
        self._rows = out
        self._i = 0

    def next(self) -> Optional[Row]:
        if self._rows is None:
            self.open()
        if self._i >= len(self._rows):  # type: ignore[operator]
            return None
        row = self._rows[self._i]  # type: ignore[index]
        self._i += 1
        return row

    def close(self) -> None:
        pass


class Limit(Operator):
    def __init__(self, child: Operator, limit: Optional[int], offset: int):
        self.child = child
        self.limit = limit
        self.offset = offset
        self._produced = 0
        self._skipped = 0

    def open(self) -> None:
        self.child.open()
        self._produced = 0
        self._skipped = 0

    def next(self) -> Optional[Row]:
        while self._skipped < self.offset:
            if self.child.next() is None:
                return None
            self._skipped += 1
        if self.limit is not None and self._produced >= self.limit:
            return None
        row = self.child.next()
        if row is not None:
            self._produced += 1
        return row

    def close(self) -> None:
        self.child.close()


class GroupAggregate(Operator):
    """GROUP BY + aggregates, or bare aggregates over the whole input.

    Emits one row per group in first-seen order (ORDER BY on top re-sorts).
    """

    def __init__(self, child: Operator, items: List[A.SelectItem], group_by: List[A.Expr]):
        self.child = child
        self.items = items
        self.group_by = group_by
        self._groups: Optional[List[Tuple[List[object], List[Aggregator]]]] = None
        self._i = 0

    def open(self) -> None:
        if self._groups is not None:
            return
        buckets: Dict[tuple, List[Aggregator]] = {}
        order: List[tuple] = []
        self.child.open()
        while True:
            row = self.child.next()
            if row is None:
                break
            gkey = tuple(sort_key(evaluate(e, row)) for e in self.group_by)
            if gkey not in buckets:
                buckets[gkey] = _new_aggs(self.items)
                order.append(gkey)
            _accumulate(self.items, buckets[gkey], row, bool(self.group_by))
        self.child.close()
        if not order and not self.group_by:
            # Bare aggregates over an empty input still emit one row.
            buckets[()] = _new_aggs(self.items)
            order.append(())
        self._groups = [(gkey_values(k), buckets[k]) for k in order]
        self._i = 0

    def next(self) -> Optional[Row]:
        if self._groups is None:
            self.open()
        if self._i >= len(self._groups):  # type: ignore[operator]
            return None
        gvals, aggs = self._groups[self._i]  # type: ignore[index]
        self._i += 1
        values: List[object] = []
        for item, agg in zip(self.items, aggs):
            if isinstance(item.expr, A.FuncCall):
                values.append(agg.result())
            elif self.group_by:
                values.append(self._group_value(item, gvals))
            else:
                raise SQLError("non-aggregate select item without GROUP BY")
        return Row({"": values}, {"": [item_name(item).lower() for item in self.items]})

    def _group_value(self, item: A.SelectItem, gvals: List[object]) -> object:
        for pos, gexpr in enumerate(self.group_by):
            if _same_expr(item.expr, gexpr):
                return gvals[pos]
        raise SQLError(f"select item {item_name(item)!r} must appear in GROUP BY or be an aggregate")

    def close(self) -> None:
        self.child.close()


def _new_aggs(items: List[A.SelectItem]) -> List[Aggregator]:
    return [Aggregator(it.expr.name, it.expr.star) if isinstance(it.expr, A.FuncCall) else Aggregator("COUNT", True)
            for it in items]


def _accumulate(items: List[A.SelectItem], aggs: List[Aggregator], row: Row, grouped: bool) -> None:
    for item, agg in zip(items, aggs):
        if isinstance(item.expr, A.FuncCall):
            if item.expr.star:
                agg.add(1)  # COUNT(*): only presence matters
            else:
                if len(item.expr.args) != 1:
                    raise SQLError(f"{item.expr.name}() takes exactly one argument")
                agg.add(evaluate(item.expr.args[0], row))
        elif grouped:
            pass  # group-key columns are filled from the key at emit time


def _same_expr(a: A.Expr, b: A.Expr) -> bool:
    """Structural equality for the select items that group keys satisfy."""
    if isinstance(a, A.ColumnRef) and isinstance(b, A.ColumnRef):
        return a.name.lower() == b.name.lower() and (a.table or "").lower() == (b.table or "").lower()
    if isinstance(a, A.Literal) and isinstance(b, A.Literal):
        return a.value == b.value
    return a is b


def gkey_values(k: tuple) -> List[object]:
    """Recover display values from a group key built from sort_key tuples.

    sort_key(v) returns (class, payload); only the payload survives as a
    displayable value (NULLs came from None so their payload is a
    placeholder; the value itself was not needed for grouping correctness).
    """
    return [part[1] for part in k]


def item_name(item: A.SelectItem) -> str:
    if item.alias:
        return item.alias
    expr = item.expr
    if isinstance(expr, A.ColumnRef):
        return expr.name
    if isinstance(expr, A.FuncCall):
        if expr.star:
            return f"{expr.name}(*)"
        args = ", ".join(_expr_str(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, A.Star):
        return "*"
    return _expr_str(expr)


def _expr_str(e: A.Expr) -> str:
    if isinstance(e, A.ColumnRef):
        return f"{e.table}.{e.name}" if e.table else e.name
    if isinstance(e, A.Literal):
        return "NULL" if e.value is None else str(e.value)
    if isinstance(e, A.BinaryOp):
        return f"{_expr_str(e.left)} {e.op} {_expr_str(e.right)}"
    return "expr"
