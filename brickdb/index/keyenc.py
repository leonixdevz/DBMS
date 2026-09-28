"""Order-preserving key encoding for B+ tree indexes.

Index keys are compared as raw bytes, so the encoding must map values to
bytes such that *lexicographic byte order equals logical value order*, and
composite keys must be unambiguously splittable. Schemes used:

* ``INT``  — 8 bytes big-endian unsigned (values are stored unsigned).
* ``FLOAT``— IEEE-754 bits transformed so byte order matches numeric order
  (flip all bits if negative, else flip the sign bit), big-endian.
* ``DATE`` — 4 bytes big-endian ordinal.
* ``TEXT`` — UTF-8 bytes with every byte shifted up by one (``0xFE``/``0xFF``
  get two-byte escapes), terminated by ``0x00``. The terminator sorts before
  any content byte, so shorter prefixes order before their extensions and
  composite keys split cleanly on the terminator.

NULLs never appear in indexes: PRIMARY KEY implies NOT NULL, and UNIQUE
indexes simply skip rows whose key contains NULL (the same "NULLs are not
equal to each other" semantics PostgreSQL uses).
"""

from __future__ import annotations

import struct
from datetime import date
from typing import List, Tuple

from ..errors import CorruptError, TypeViolation

_U64 = 0xFFFFFFFFFFFFFFFF


def _text_escape(raw: bytes) -> bytes:
    out = bytearray()
    for b in raw:
        if b <= 0xFD:
            out.append(b + 1)
        elif b == 0xFE:
            out += b"\xff\x01"
        else:  # 0xFF
            out += b"\xff\x02"
    out.append(0x00)  # terminator
    return bytes(out)


def _text_unescape(enc: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(enc):
        b = enc[i]
        if b == 0x00:
            return bytes(out)
        if b == 0xFF:
            if i + 1 >= len(enc):
                raise CorruptError("truncated text key escape")
            out.append(0xFE if enc[i + 1] == 0x01 else 0xFF)
            i += 2
        else:
            out.append(b - 1)
            i += 1
    raise CorruptError("unterminated text key")


def _float_to_ordered(f: float) -> bytes:
    (bits,) = struct.unpack(">Q", struct.pack(">d", f))
    if bits >> 63:  # negative number: flip all bits
        bits = ~bits & _U64
    else:  # non-negative: flip sign bit only
        bits ^= 1 << 63
    return struct.pack(">Q", bits)


def _ordered_to_float(b: bytes) -> float:
    (bits,) = struct.unpack(">Q", b)
    if bits >> 63:  # sign bit set => original was non-negative
        bits ^= 1 << 63
    else:  # original was negative: re-invert all bits
        bits = ~bits & _U64
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def encode_key_part(kind: str, value) -> bytes:
    """Encode one column value as an order-preserving byte string."""
    if value is None:
        raise TypeViolation("NULL keys are not indexed")
    if kind == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeViolation(f"INT key expected, got {type(value).__name__}")
        return struct.pack(">Q", value)
    if kind == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeViolation(f"FLOAT key expected, got {type(value).__name__}")
        return _float_to_ordered(float(value))
    if kind == "DATE":
        if not isinstance(value, date):
            raise TypeViolation(f"DATE key expected, got {type(value).__name__}")
        return struct.pack(">I", value.toordinal())
    if kind in ("TEXT", "VARCHAR"):
        if not isinstance(value, str):
            raise TypeViolation(f"TEXT key expected, got {type(value).__name__}")
        return _text_escape(value.encode("utf-8"))
    raise TypeViolation(f"unsupported key type {kind!r}")


def decode_key_part(kind: str, buf: bytes, pos: int) -> Tuple[object, int]:
    """Inverse of :func:`encode_key_part` (used by EXPLAIN and verification)."""
    if kind == "INT":
        return struct.unpack_from(">Q", buf, pos)[0], pos + 8
    if kind == "FLOAT":
        return _ordered_to_float(buf[pos : pos + 8]), pos + 8
    if kind == "DATE":
        return date.fromordinal(struct.unpack_from(">I", buf, pos)[0]), pos + 4
    if kind in ("TEXT", "VARCHAR"):
        end = buf.index(0x00, pos)
        # Include the terminator so the unescaper can validate it.
        return _text_unescape(buf[pos : end + 1]).decode("utf-8"), end + 1
    raise CorruptError(f"unsupported key type {kind!r}")


def encode_key(kinds: List[str], values: List) -> bytes:
    if len(kinds) != len(values):
        raise CorruptError("key arity mismatch")
    return b"".join(encode_key_part(k, v) for k, v in zip(kinds, values))


def decode_key(kinds: List[str], buf: bytes) -> List[object]:
    out: List[object] = []
    pos = 0
    for kind in kinds:
        value, pos = decode_key_part(kind, buf, pos)
        out.append(value)
    return out


def pack_payload(page_id: int, slot: int) -> bytes:
    return struct.pack("<IH", page_id, slot)


def unpack_payload(payload: bytes) -> Tuple[int, int]:
    return struct.unpack("<IH", payload)
