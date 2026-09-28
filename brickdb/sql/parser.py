"""Recursive-descent parser: token stream -> AST.

Grammar (subset, LL(1) with a couple of backtrack points)::

    statement    := select | insert | update | delete | create | drop
                  | begin | commit | rollback | explain | reindex
    select       := SELECT [DISTINCT] items [FROM tref [joins]]
                    [WHERE e] [GROUP BY elist] [ORDER BY olist]
                    [LIMIT n [OFFSET m]]
    items        := item (',' item)*
    item         := '*' | ident '.' '*' | expr [AS ident | ident]
    expr         := or_expr
    or_expr      := and_expr (OR and_expr)*
    and_expr     := not_expr (AND not_expr)*
    not_expr     := NOT not_expr | predicate
    predicate    := additive ( (=|<>|<|<=|>|>=) additive
                              | [NOT] BETWEEN additive AND additive
                              | [NOT] IN '(' expr,... ')'
                              | [NOT] LIKE additive
                              | IS [NOT] NULL )?
    additive     := multiplicative (('+'|'-'|'||') multiplicative)*
    multiplicative := unary (('*'|'/'|'%') unary)*
    unary        := '-' unary | primary
    primary      := NUMBER | STRING | TRUE | FALSE | NULL
                  | func '(' args ')' | ident ['.' ident] | '(' expr ')'
    func         := COUNT | SUM | MIN | MAX | AVG

Precedence, lowest to highest: OR, AND, NOT, comparison / BETWEEN / IN /
LIKE / IS, additive, multiplicative, unary minus, primary.
"""

from __future__ import annotations

from typing import List, Optional

from ..errors import SQLError
from . import ast as A
from .lexer import Token, tokenize

AGGREGATES = {"COUNT", "SUM", "MIN", "MAX", "AVG"}
TYPE_NAMES = {"INT": "INT", "INTEGER": "INT", "FLOAT": "FLOAT", "DOUBLE": "FLOAT",
              "REAL": "FLOAT", "TEXT": "TEXT", "DATE": "DATE"}


class Parser:
    def __init__(self, sql: str):
        self.tokens = tokenize(sql)
        self.pos = 0

    # ------------------------------------------------------------ helpers

    def peek(self, ahead: int = 0) -> Token:
        i = min(self.pos + ahead, len(self.tokens) - 1)
        return self.tokens[i]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "eof":
            self.pos += 1
        return tok

    def expect_op(self, op: str) -> Token:
        tok = self.peek()
        if not tok.is_op(op):
            raise SQLError(f"expected {op!r} near {tok.value!r} (offset {tok.pos})")
        return self.advance()

    def expect_kw(self, kw: str) -> Token:
        tok = self.peek()
        if not tok.is_kw(kw):
            raise SQLError(f"expected {kw} near {tok.value!r} (offset {tok.pos})")
        return self.advance()

    def accept_op(self, op: str) -> Optional[Token]:
        if self.peek().is_op(op):
            return self.advance()
        return None

    def accept_kw(self, *kws: str) -> Optional[Token]:
        if self.peek().is_kw(*kws):
            return self.advance()
        return None

    def expect_ident(self) -> str:
        tok = self.peek()
        if tok.kind != "ident":
            raise SQLError(f"expected identifier near {tok.value!r} (offset {tok.pos})")
        return self.advance().value

    def at_end(self) -> bool:
        return self.peek().kind == "eof"

    # ---------------------------------------------------------- statements

    def parse(self) -> A.Stmt:
        stmt = self.parse_statement()
        self.accept_op(";")
        if not self.at_end():
            tok = self.peek()
            raise SQLError(f"unexpected trailing input near {tok.value!r} (offset {tok.pos})")
        return stmt

    def parse_statement(self) -> A.Stmt:
        tok = self.peek()
        if tok.is_kw("SELECT"):
            return self.parse_select()
        if tok.is_kw("INSERT"):
            return self.parse_insert()
        if tok.is_kw("UPDATE"):
            return self.parse_update()
        if tok.is_kw("DELETE"):
            return self.parse_delete()
        if tok.is_kw("CREATE"):
            return self.parse_create()
        if tok.is_kw("DROP"):
            return self.parse_drop()
        if tok.is_kw("BEGIN"):
            self.advance()
            return A.BeginStmt()
        if tok.is_kw("COMMIT"):
            self.advance()
            return A.CommitStmt()
        if tok.is_kw("ROLLBACK"):
            self.advance()
            return A.RollbackStmt()
        if tok.is_kw("REINDEX"):
            return self.parse_reindex()
        if tok.is_kw("EXPLAIN"):
            return self.parse_explain()
        raise SQLError(f"unexpected token {tok.value!r} at statement start (offset {tok.pos})")

    # ------------------------------------------------------------- select

    def parse_select(self) -> A.SelectStmt:
        self.expect_kw("SELECT")
        stmt = A.SelectStmt()
        if self.accept_kw("DISTINCT"):
            stmt.distinct = True
        stmt.items = [self.parse_select_item()]
        while self.accept_op(","):
            stmt.items.append(self.parse_select_item())
        if self.accept_kw("FROM"):
            stmt.from_table = self.parse_table_ref()
            while True:
                if self.accept_kw("CROSS"):
                    self.expect_kw("JOIN")
                    stmt.joins.append(A.JoinClause("CROSS", self.parse_table_ref()))
                elif self.accept_kw("INNER"):
                    self.expect_kw("JOIN")
                    tref = self.parse_table_ref()
                    self.expect_kw("ON")
                    stmt.joins.append(A.JoinClause("INNER", tref, self.parse_expr()))
                elif self.accept_kw("LEFT"):
                    self.accept_kw("OUTER")
                    self.expect_kw("JOIN")
                    tref = self.parse_table_ref()
                    self.expect_kw("ON")
                    stmt.joins.append(A.JoinClause("LEFT", tref, self.parse_expr()))
                elif self.accept_kw("JOIN"):
                    tref = self.parse_table_ref()
                    self.expect_kw("ON")
                    stmt.joins.append(A.JoinClause("INNER", tref, self.parse_expr()))
                else:
                    break
        if self.accept_kw("WHERE"):
            stmt.where = self.parse_expr()
        if self.accept_kw("GROUP"):
            self.expect_kw("BY")
            stmt.group_by.append(self.parse_expr())
            while self.accept_op(","):
                stmt.group_by.append(self.parse_expr())
        if self.accept_kw("ORDER"):
            self.expect_kw("BY")
            stmt.order_by.append(self.parse_order_item())
            while self.accept_op(","):
                stmt.order_by.append(self.parse_order_item())
        if self.accept_kw("LIMIT"):
            stmt.limit = self.parse_int("LIMIT")
            if self.accept_kw("OFFSET"):
                stmt.offset = self.parse_int("OFFSET")
        return stmt

    def parse_int(self, ctx: str) -> int:
        tok = self.peek()
        if tok.kind != "int":
            raise SQLError(f"expected integer after {ctx}, got {tok.value!r}")
        return int(self.advance().value)

    def parse_select_item(self) -> A.SelectItem:
        if self.peek().is_op("*"):
            self.advance()
            return A.SelectItem(A.Star())
        if self.peek().kind == "ident" and self.peek(1).is_op(".") and self.peek(2).is_op("*"):
            tbl = self.advance().value
            self.advance()
            self.advance()
            return A.SelectItem(A.Star(tbl))
        expr = self.parse_expr()
        alias = None
        if self.accept_kw("AS"):
            alias = self.expect_ident()
        elif self.peek().kind == "ident":
            alias = self.advance().value
        return A.SelectItem(expr, alias)

    def parse_table_ref(self) -> A.TableRef:
        name = self.expect_ident()
        alias = None
        if self.accept_kw("AS"):
            alias = self.expect_ident()
        elif self.peek().kind == "ident":
            alias = self.advance().value
        return A.TableRef(name, alias)

    def parse_order_item(self) -> A.OrderItem:
        expr = self.parse_expr()
        desc = False
        if self.accept_kw("DESC"):
            desc = True
        elif self.accept_kw("ASC"):
            pass
        return A.OrderItem(expr, desc)

    # ------------------------------------------------------------- insert

    def parse_insert(self) -> A.InsertStmt:
        self.expect_kw("INSERT")
        self.expect_kw("INTO")
        table = self.expect_ident()
        stmt = A.InsertStmt(table=table)
        if self.peek().is_op("("):
            self.advance()
            stmt.columns.append(self.expect_ident())
            while self.accept_op(","):
                stmt.columns.append(self.expect_ident())
            self.expect_op(")")
        self.expect_kw("VALUES")
        while True:
            self.expect_op("(")
            row = [self.parse_expr()]
            while self.accept_op(","):
                row.append(self.parse_expr())
            self.expect_op(")")
            stmt.rows.append(row)
            if not self.accept_op(","):
                break
        return stmt

    # ------------------------------------------------------------- update

    def parse_update(self) -> A.UpdateStmt:
        self.expect_kw("UPDATE")
        table = self.expect_ident()
        self.expect_kw("SET")
        assignments = []
        while True:
            col = self.expect_ident()
            self.expect_op("=")
            assignments.append((col, self.parse_expr()))
            if not self.accept_op(","):
                break
        where = self.parse_expr() if self.accept_kw("WHERE") else None
        return A.UpdateStmt(table, assignments, where)

    # ------------------------------------------------------------- delete

    def parse_delete(self) -> A.DeleteStmt:
        self.expect_kw("DELETE")
        self.expect_kw("FROM")
        table = self.expect_ident()
        where = self.parse_expr() if self.accept_kw("WHERE") else None
        return A.DeleteStmt(table, where)

    # ------------------------------------------------------------- create

    def parse_create(self) -> A.Stmt:
        self.expect_kw("CREATE")
        unique = bool(self.accept_kw("UNIQUE"))
        if unique or self.peek().is_kw("INDEX"):
            return self.parse_create_index(unique)
        self.expect_kw("TABLE")
        table = self.expect_ident()
        self.expect_op("(")
        columns: List[A.ColumnSpec] = []
        while True:
            columns.append(self.parse_column_spec())
            if not self.accept_op(","):
                break
        self.expect_op(")")
        return A.CreateTableStmt(table, columns)

    def parse_column_spec(self) -> A.ColumnSpec:
        name = self.expect_ident()
        tok = self.peek()
        if not (tok.is_kw(*TYPE_NAMES.keys()) or (tok.kind == "kw" and tok.value in TYPE_NAMES)):
            raise SQLError(f"expected type for column {name!r}, got {tok.value!r}")
        kind = TYPE_NAMES[self.advance().value]
        spec = A.ColumnSpec(name, kind)
        while True:
            if self.accept_kw("NOT"):
                self.expect_kw("NULL")
                spec.notnull = True
            elif self.accept_kw("PRIMARY"):
                self.expect_kw("KEY")
                spec.primary_key = True
                spec.notnull = True
            elif self.accept_kw("UNIQUE"):
                spec.unique = True
            elif self.accept_kw("DEFAULT"):
                spec.default = self.parse_literal_expr()
            else:
                break
        return spec

    def parse_literal_expr(self) -> A.Expr:
        tok = self.peek()
        if tok.kind in ("int", "float"):
            return A.Literal(self._num(self.advance()))
        if tok.kind == "string":
            return A.Literal(self.advance().value)
        if tok.is_kw("TRUE"):
            self.advance()
            return A.Literal(True)
        if tok.is_kw("FALSE"):
            self.advance()
            return A.Literal(False)
        if tok.is_kw("NULL"):
            self.advance()
            return A.Literal(None)
        if tok.is_op("-"):
            self.advance()
            inner = self.parse_literal_expr()
            if isinstance(inner, A.Literal) and isinstance(inner.value, (int, float)):
                return A.Literal(-inner.value)
            raise SQLError("DEFAULT must be a literal")
        raise SQLError(f"expected literal, got {tok.value!r}")

    @staticmethod
    def _num(tok: Token):
        return float(tok.value) if tok.kind == "float" else int(tok.value)

    def parse_create_index(self, unique: bool) -> A.CreateIndexStmt:
        self.expect_kw("INDEX")
        index = self.expect_ident()
        self.expect_kw("ON")
        table = self.expect_ident()
        self.expect_op("(")
        columns = [self.expect_ident()]
        while self.accept_op(","):
            columns.append(self.expect_ident())
        self.expect_op(")")
        return A.CreateIndexStmt(index, table, columns, unique)

    # --------------------------------------------------------------- drop

    def parse_drop(self) -> A.Stmt:
        self.expect_kw("DROP")
        if self.accept_kw("TABLE"):
            if_exists = bool(self.accept_kw("IF") and self.expect_kw("EXISTS"))
            return A.DropTableStmt(self.expect_ident(), if_exists)
        self.expect_kw("INDEX")
        if_exists = bool(self.accept_kw("IF") and self.expect_kw("EXISTS"))
        return A.DropIndexStmt(self.expect_ident(), if_exists)

    # ------------------------------------------------------------ reindex

    def parse_reindex(self) -> A.ReindexStmt:
        self.expect_kw("REINDEX")
        return A.ReindexStmt(self.expect_ident())

    # ------------------------------------------------------------ explain

    def parse_explain(self) -> A.ExplainStmt:
        self.expect_kw("EXPLAIN")
        if self.accept_kw("TABLE"):
            return A.ExplainStmt(report="table", arg=self.expect_ident())
        if self.accept_kw("INDEX"):
            return A.ExplainStmt(report="index", arg=self.expect_ident())
        if self.accept_kw("STATS"):
            return A.ExplainStmt(report="stats")
        return A.ExplainStmt(target=self.parse_statement())

    # --------------------------------------------------------- expressions

    def parse_expr(self) -> A.Expr:
        return self.parse_or()

    def parse_or(self) -> A.Expr:
        left = self.parse_and()
        while self.accept_kw("OR"):
            left = A.BinaryOp("OR", left, self.parse_and())
        return left

    def parse_and(self) -> A.Expr:
        left = self.parse_not()
        while self.accept_kw("AND"):
            left = A.BinaryOp("AND", left, self.parse_not())
        return left

    def parse_not(self) -> A.Expr:
        if self.accept_kw("NOT"):
            return A.UnaryOp("NOT", self.parse_not())
        return self.parse_predicate()

    def parse_predicate(self) -> A.Expr:
        left = self.parse_additive()
        if self.peek().is_op("=", "<>", "<", "<=", ">", ">="):
            op = self.advance().value
            return A.BinaryOp(op, left, self.parse_additive())
        negated = False
        if self.peek().is_kw("NOT") and self.peek(1).is_kw("BETWEEN", "IN", "LIKE"):
            self.advance()
            negated = True
        if self.accept_kw("BETWEEN"):
            low = self.parse_additive()
            self.expect_kw("AND")
            high = self.parse_additive()
            return A.Between(left, low, high, negated)
        if self.accept_kw("IN"):
            self.expect_op("(")
            items = [self.parse_expr()]
            while self.accept_op(","):
                items.append(self.parse_expr())
            self.expect_op(")")
            return A.InList(left, items, negated)
        if self.accept_kw("LIKE"):
            return A.BinaryOp("LIKE", left, self.parse_additive())
        if self.accept_kw("IS"):
            neg = bool(self.accept_kw("NOT"))
            self.expect_kw("NULL")
            return A.IsNull(left, neg)
        return left

    def parse_additive(self) -> A.Expr:
        left = self.parse_multiplicative()
        while self.peek().is_op("+", "-", "||"):
            op = self.advance().value
            left = A.BinaryOp(op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self) -> A.Expr:
        left = self.parse_unary()
        while self.peek().is_op("*", "/", "%"):
            op = self.advance().value
            left = A.BinaryOp(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> A.Expr:
        if self.peek().is_op("-"):
            self.advance()
            return A.UnaryOp("-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> A.Expr:
        tok = self.peek()
        if tok.kind in ("int", "float"):
            return A.Literal(self._num(self.advance()))
        if tok.kind == "string":
            return A.Literal(self.advance().value)
        if tok.is_kw("TRUE"):
            self.advance()
            return A.Literal(True)
        if tok.is_kw("FALSE"):
            self.advance()
            return A.Literal(False)
        if tok.is_kw("NULL"):
            self.advance()
            return A.Literal(None)
        if tok.is_kw("COUNT", "SUM", "MIN", "MAX", "AVG"):
            name = self.advance().value
            self.expect_op("(")
            if self.peek().is_op("*"):
                self.advance()
                self.expect_op(")")
                return A.FuncCall(name, [], star=True)
            args = [self.parse_expr()]
            while self.accept_op(","):
                args.append(self.parse_expr())
            self.expect_op(")")
            return A.FuncCall(name, args)
        if tok.kind == "ident":
            first = self.advance().value
            if self.peek().is_op("."):
                self.advance()
                return A.ColumnRef(first, self.expect_ident())
            return A.ColumnRef(None, first)
        if tok.is_op("("):
            self.advance()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner
        raise SQLError(f"unexpected token {tok.value!r} in expression (offset {tok.pos})")


def parse(sql: str) -> A.Stmt:
    return Parser(sql).parse()
