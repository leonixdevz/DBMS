"""SQL tokenizer: converts raw SQL text into a stream of tokens.

Handles keywords, identifiers (quoted with double quotes), string literals
(single quotes with '' escape), numbers (int and float), operators, and
comments (``--`` to end of line). Keyword matching is case-insensitive; the
parser receives the canonical upper-case form for keywords and the original
text for identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..errors import SQLError

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "DROP", "TABLE", "INDEX", "ON", "UNIQUE", "NOT",
    "NULL", "PRIMARY", "KEY", "AND", "OR", "AS", "ORDER", "BY", "ASC",
    "DESC", "LIMIT", "OFFSET", "JOIN", "LEFT", "INNER", "OUTER", "CROSS",
    "GROUP", "COUNT", "SUM", "MIN", "MAX", "AVG", "DISTINCT", "EXPLAIN",
    "BEGIN", "COMMIT", "ROLLBACK", "DEFAULT", "TEXT", "INT", "INTEGER",
    "FLOAT", "DOUBLE", "REAL", "DATE", "TRUE", "FALSE", "IS", "IN",
    "BETWEEN", "LIKE", "EXISTS", "REINDEX", "IF", "ANALYZE", "STATS",
}

_MULTI_OPS = ("<=", ">=", "<>", "!=", "||")
_SINGLE_OPS = set("=<>+-*/%(),.;")


@dataclass
class Token:
    kind: str  # 'kw' | 'ident' | 'string' | 'int' | 'float' | 'op' | 'eof'
    value: str
    pos: int   # character offset in the source, for error messages

    def is_kw(self, *names: str) -> bool:
        return self.kind == "kw" and self.value in names

    def is_op(self, *ops: str) -> bool:
        return self.kind == "op" and self.value in ops

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def tokenize(sql: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        # whitespace
        if c.isspace():
            i += 1
            continue
        # comment
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        # string literal
        if c == "'":
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise SQLError(f"unterminated string literal at offset {i}")
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            tokens.append(Token("string", "".join(buf), i))
            i = j + 1
            continue
        # quoted identifier
        if c == '"':
            end = sql.find('"', i + 1)
            if end == -1:
                raise SQLError(f"unterminated quoted identifier at offset {i}")
            tokens.append(Token("ident", sql[i + 1 : end], i))
            i = end + 1
            continue
        # number
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            j = i
            is_float = False
            while j < n and (sql[j].isdigit() or sql[j] == "."):
                if sql[j] == ".":
                    if is_float:
                        break
                    is_float = True
                j += 1
            text = sql[i:j]
            tokens.append(Token("float" if is_float else "int", text, i))
            i = j
            continue
        # identifier / keyword
        if c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            word = sql[i:j]
            upper = word.upper()
            if upper in KEYWORDS:
                tokens.append(Token("kw", upper, i))
            else:
                tokens.append(Token("ident", word, i))
            i = j
            continue
        # operators
        two = sql[i : i + 2]
        if two in _MULTI_OPS:
            tokens.append(Token("op", "<>" if two == "!=" else two, i))
            i += 2
            continue
        if c in _SINGLE_OPS:
            tokens.append(Token("op", c, i))
            i += 1
            continue
        raise SQLError(f"unexpected character {c!r} at offset {i}")
    tokens.append(Token("eof", "", n))
    return tokens
