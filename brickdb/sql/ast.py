"""AST node definitions for SQL statements and expressions.

Keeping the AST as plain dataclasses makes the parser trivial to test (parse
a string, compare to an expected tree) and makes the executor's dispatch a
simple ``isinstance`` chain or dict lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# ----------------------------------------------------------- expressions

@dataclass
class Expr:
    """Base class for all expression nodes."""


@dataclass
class Literal(Expr):
    value: Any  # int | float | str | bool | None


@dataclass
class ColumnRef(Expr):
    table: Optional[str]
    name: str


@dataclass
class Star(Expr):
    """``*`` or ``alias.*`` in a select list / COUNT(*)."""

    table: Optional[str] = None


@dataclass
class BinaryOp(Expr):
    op: str  # = <> < <= > >= + - * / % AND OR || LIKE IN
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: str  # NOT -
    operand: Expr


@dataclass
class FuncCall(Expr):
    name: str  # COUNT SUM MIN MAX AVG (upper-cased)
    args: List[Expr]
    star: bool = False  # COUNT(*)


@dataclass
class IsNull(Expr):
    operand: Expr
    negated: bool


@dataclass
class Between(Expr):
    operand: Expr
    low: Expr
    high: Expr
    negated: bool


@dataclass
class InList(Expr):
    operand: Expr
    items: List[Expr]
    negated: bool


@dataclass
class Param(Expr):
    """``?`` placeholder — not supported, parsed for good error messages."""

    index: int = 0


# ------------------------------------------------------------ statements

@dataclass
class Stmt:
    """Base class for statements."""


@dataclass
class SelectItem:
    expr: Expr
    alias: Optional[str] = None


@dataclass
class OrderItem:
    expr: Expr
    desc: bool = False


@dataclass
class TableRef:
    name: str
    alias: Optional[str] = None


@dataclass
class JoinClause:
    kind: str  # INNER | LEFT | CROSS
    table: TableRef
    on: Optional[Expr] = None


@dataclass
class SelectStmt(Stmt):
    items: List[SelectItem] = field(default_factory=list)
    distinct: bool = False
    from_table: Optional[TableRef] = None
    joins: List[JoinClause] = field(default_factory=list)
    where: Optional[Expr] = None
    group_by: List[Expr] = field(default_factory=list)
    order_by: List[OrderItem] = field(default_factory=list)
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class InsertStmt(Stmt):
    table: str = ""
    columns: List[str] = field(default_factory=list)
    rows: List[List[Expr]] = field(default_factory=list)


@dataclass
class UpdateStmt(Stmt):
    table: str = ""
    assignments: List[Tuple[str, Expr]] = field(default_factory=list)
    where: Optional[Expr] = None


@dataclass
class DeleteStmt(Stmt):
    table: str = ""
    where: Optional[Expr] = None


@dataclass
class ColumnSpec:
    name: str
    kind: str
    notnull: bool = False
    primary_key: bool = False
    unique: bool = False
    default: Optional[Expr] = None


@dataclass
class CreateTableStmt(Stmt):
    table: str = ""
    columns: List[ColumnSpec] = field(default_factory=list)


@dataclass
class DropTableStmt(Stmt):
    table: str = ""
    if_exists: bool = False


@dataclass
class CreateIndexStmt(Stmt):
    index: str = ""
    table: str = ""
    columns: List[str] = field(default_factory=list)
    unique: bool = False


@dataclass
class DropIndexStmt(Stmt):
    index: str = ""
    if_exists: bool = False


@dataclass
class BeginStmt(Stmt):
    pass


@dataclass
class CommitStmt(Stmt):
    pass


@dataclass
class RollbackStmt(Stmt):
    pass


@dataclass
class ExplainStmt(Stmt):
    # Either a query plan or a storage report
    target: Optional[Stmt] = None
    report: Optional[str] = None  # 'table' | 'index' | 'stats'
    arg: Optional[str] = None


@dataclass
class ReindexStmt(Stmt):
    table: str = ""
