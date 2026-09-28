"""Expression evaluation over rows with SQL NULL semantics.

Rows flowing between operators are :class:`Row` objects carrying
``tables: alias -> values`` plus ``colnames: alias -> [lowercased names]``
so expressions can resolve unqualified and table-qualified column names.

SQL's three-valued logic is honored: ``NULL = NULL`` is NULL (unknown), not
true; ``NOT (NULL)`` is still NULL; comparisons with NULL are NULL and WHERE
treats NULL as false. LIKE supports ``%`` (any run) and ``_`` (one char).
``||`` is string concatenation. Division by zero raises SQLError.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..errors import SQLError
from ..sql import ast as A


class Row:
    """A row visible to the evaluator, optionally table-qualified (joins)."""

    __slots__ = ("tables", "colnames")

    def __init__(self, tables: Dict[str, List], colnames: Dict[str, List[str]]):
        self.tables = tables
        self.colnames = colnames

    @property
    def single(self) -> List:
        (values,) = self.tables.values()
        return values


def lookup(row: Row, ref: A.ColumnRef) -> object:
    lname = ref.name.lower()
    if ref.table is None:
        if len(row.tables) == 1:
            alias = next(iter(row.tables))
        else:
            matches = [a for a, names in row.colnames.items() if lname in names]
            if len(matches) == 1:
                alias = matches[0]
            elif not matches:
                raise SQLError(f"no such column: {ref.name!r}")
            else:
                raise SQLError(f"ambiguous column {ref.name!r}: qualify it with a table name")
    else:
        alias = ref.table.lower()
        if alias not in row.tables:
            raise SQLError(f"unknown table or alias {ref.table!r}")
    names = row.colnames[alias]
    try:
        idx = names.index(lname)
    except ValueError:
        raise SQLError(f"no such column: {ref.name!r}")
    return row.tables[alias][idx]


def evaluate(expr: A.Expr, row: Row) -> object:
    """Evaluate an expression against a row (Python None == SQL NULL)."""
    if isinstance(expr, A.Literal):
        return expr.value
    if isinstance(expr, A.ColumnRef):
        return lookup(row, expr)
    if isinstance(expr, A.Star):
        raise SQLError("'*' is not valid in this context")
    if isinstance(expr, A.Param):
        raise SQLError("parameter placeholders ('?') are not supported; inline literals instead")
    if isinstance(expr, A.UnaryOp):
        if expr.op == "NOT":
            v = evaluate(expr.operand, row)
            return None if v is None else (not truthy(v))
        if expr.op == "-":
            v = evaluate(expr.operand, row)
            if v is None:
                return None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SQLError("unary '-' expects a numeric operand")
            return -v
        raise SQLError(f"unknown unary operator {expr.op!r}")
    if isinstance(expr, A.IsNull):
        v = evaluate(expr.operand, row)
        # IS NULL / IS NOT NULL never evaluate to NULL themselves.
        return (v is None) != expr.negated
    if isinstance(expr, A.Between):
        v = evaluate(expr.operand, row)
        result = _and3(_cmp3(">=", v, evaluate(expr.low, row)), _cmp3("<=", v, evaluate(expr.high, row)))
        return _not3(result) if expr.negated else result
    if isinstance(expr, A.InList):
        v = evaluate(expr.operand, row)
        result = None
        for item in expr.items:
            result = _or3(result, _cmp3("=", v, evaluate(item, row)))
        return _not3(result) if expr.negated else result
    if isinstance(expr, A.FuncCall):
        raise SQLError(f"aggregate {expr.name}() is only valid as a top-level select item")
    if isinstance(expr, A.BinaryOp):
        return _binary(expr, row)
    raise SQLError(f"cannot evaluate expression node {type(expr).__name__}")


def _binary(expr: A.BinaryOp, row: Row) -> object:
    op = expr.op
    if op == "AND":
        return _and3(_to_bool3(evaluate(expr.left, row)), _to_bool3(evaluate(expr.right, row)))
    if op == "OR":
        return _or3(_to_bool3(evaluate(expr.left, row)), _to_bool3(evaluate(expr.right, row)))
    left = evaluate(expr.left, row)
    right = evaluate(expr.right, row)
    if op in ("=", "<>", "<", "<=", ">", ">="):
        return _cmp3(op, left, right)
    if op == "||":
        if left is None or right is None:
            return None
        return _to_str(left) + _to_str(right)
    if op == "LIKE":
        return _like3(left, right)
    if left is None or right is None:
        return None
    try:
        if op == "+":
            if isinstance(left, str) or isinstance(right, str):
                raise SQLError("'+' on strings is not supported; use '||' to concatenate")
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0:
                raise SQLError("division by zero")
            return left / right
        if op == "%":
            if right == 0:
                raise SQLError("division by zero")
            return left % right
    except TypeError:
        raise SQLError(f"type error: {type(left).__name__} {op} {type(right).__name__}")
    raise SQLError(f"unknown operator {op!r}")


def _to_bool3(v: object) -> Optional[bool]:
    return None if v is None else truthy(v)


def truthy(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return bool(v)
    raise SQLError(f"cannot interpret {type(v).__name__} as boolean")


def _not3(v: Optional[bool]) -> Optional[bool]:
    return None if v is None else (not v)


def _and3(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _or3(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _to_str(v: object) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    raise SQLError(f"cannot convert {type(v).__name__} to string")


def _cmp3(op: str, left: object, right: object) -> Optional[bool]:
    """Three-valued comparison; NULL never satisfies any comparison."""
    if left is None or right is None:
        return None
    if isinstance(left, str) != isinstance(right, str):
        raise SQLError(f"cannot compare {type(left).__name__} with {type(right).__name__}")
    from datetime import date as _date

    if isinstance(left, _date) != isinstance(right, _date):
        raise SQLError(f"cannot compare {type(left).__name__} with {type(right).__name__}")
    if op == "=":
        return left == right
    if op == "<>":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise SQLError(f"unknown comparison {op!r}")


def _like3(value: object, pattern: object) -> Optional[bool]:
    if value is None or pattern is None:
        return None
    if not isinstance(value, str) or not isinstance(pattern, str):
        raise SQLError("LIKE expects text operands")
    return _like_match(value, pattern)


def _like_match(s: str, pattern: str) -> bool:
    import re

    out = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.fullmatch("".join(out), s, flags=re.DOTALL) is not None


# ------------------------------------------------------------- aggregates

class Aggregator:
    """Accumulates one aggregate function's state over rows."""

    def __init__(self, func: str, star: bool):
        self.func = func
        self.star = star
        self.count = 0
        self.total = 0.0
        self.seen_min = None
        self.seen_max = None
        self.saw_any = False

    def add(self, value: object) -> None:
        if self.star or value is not None:
            self.count += 1
            self.saw_any = True
        if value is None:
            return
        if self.func in ("SUM", "AVG"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SQLError(f"{self.func}() expects numeric input")
            self.total += value
        elif self.func in ("MIN", "MAX"):
            if self.seen_min is None or value < self.seen_min:
                self.seen_min = value
            if self.seen_max is None or value > self.seen_max:
                self.seen_max = value

    def result(self) -> object:
        if self.func == "COUNT":
            return self.count
        if self.func == "SUM":
            return self.total if self.count else None
        if self.func == "AVG":
            return (self.total / self.count) if self.count else None
        if self.func == "MIN":
            return self.seen_min if self.saw_any else None
        if self.func == "MAX":
            return self.seen_max if self.saw_any else None
        raise SQLError(f"unknown aggregate {self.func!r}")


def sort_key(v: object):
    """Total-order key for ORDER BY / grouping: NULLs first, then by value.

    The second tuple slot keeps the value itself (or a stand-in for NULL) so
    group keys can recover the original display value.
    """
    if v is None:
        return (0, None)
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    from datetime import date as _date

    if isinstance(v, _date):
        return (3, v)
    return (4, str(v))
