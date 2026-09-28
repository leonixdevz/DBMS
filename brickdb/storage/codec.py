"""Typed value encoding for on-page payloads.

Every column value is stored as ``1 type-tag byte + fixed-size payload``.
Fixed sizes keep offsets trivial to compute and make page compaction a pure
``memmove`` problem. Layout:

    N   -> 1 byte: 0x00
    I   -> 8 bytes little-endian unsigned
    F   -> 8 bytes little-endian IEEE-754 double
    S   -> varint length prefix + UTF-8 bytes
    B   -> varint length prefix + raw bytes
    T   -> 4 bytes little-endian date ordinal

Rows are encoded back-to-back with no separators; decoding consumes the
buffer until it is exhausted (payloads are self-delimiting). SQL NULL is
represented only by the Python ``None`` object; an empty string is a valid
zero-length payload, which is why the tag byte must always be present.
"""

from __future__ import annotations

import struct
from datetime import date
from typing import Any, List, Tuple

from ..errors import CorruptError, TypeViolation

SIZEOF_N = 1 + 1
SIZEOF_I = 1 + 8
SIZEOF_F = 1 + 8


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result, shift = 0, 0
    while True:
        if pos >= len(buf):
            raise CorruptError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 35:
            raise CorruptError("varint too long")


def _encode(value: Any) -> bytes:
    """Encode a single Python value to its tagged on-page bytes."""
    if value is None:
        return b"\x00"
    if isinstance(value, bool):
        return b"\x01" + struct.pack("<Q", int(value))
    if _is_int(value):
        if value < 0 or value > 2**64 - 1:
            raise TypeViolation(f"integer {value} out of unsigned 64-bit range")
        return b"\x01" + struct.pack("<Q", value)
    if isinstance(value, float):
        return b"\x02" + struct.pack("<d", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"\x03" + _varint(len(raw)) + raw
    if isinstance(value, bytes):
        return b"\x04" + _varint(len(value)) + value
    if isinstance(value, date):
        ordinal = value.toordinal()
        if not 1 <= ordinal <= 0xFFFFFFFF:
            raise TypeViolation("date out of representable range")
        return b"\x05" + struct.pack("<I", ordinal)
    raise TypeViolation(f"unsupported value type: {type(value).__name__}")


def _decode(buf: bytes, pos: int) -> Tuple[Any, int]:
    if pos >= len(buf):
        raise CorruptError("truncated value")
    tag = buf[pos]
    pos += 1
    if tag == 0x00:
        return None, pos
    if tag == 0x01:
        if pos + 8 > len(buf):
            raise CorruptError("truncated integer")
        return struct.unpack_from("<Q", buf, pos)[0], pos + 8
    if tag == 0x02:
        if pos + 8 > len(buf):
            raise CorruptError("truncated float")
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if tag == 0x03:
        n, pos = _read_varint(buf, pos)
        if pos + n > len(buf):
            raise CorruptError("truncated string")
        return buf[pos : pos + n].decode("utf-8"), pos + n
    if tag == 0x04:
        n, pos = _read_varint(buf, pos)
        if pos + n > len(buf):
            raise CorruptError("truncated bytes")
        return bytes(buf[pos : pos + n]), pos + n
    if tag == 0x05:
        if pos + 4 > len(buf):
            raise CorruptError("truncated date")
        return date.fromordinal(struct.unpack_from("<I", buf, pos)[0]), pos + 4
    raise CorruptError(f"unknown value tag 0x{tag:02x}")


def encode_row(values: List[Any]) -> bytes:
    """Encode a whole row (list of Python values) into page payload bytes."""
    return b"".join(_encode(v) for v in values)


def decode_one(buf: bytes, pos: int = 0) -> Tuple[Any, int]:
    """Decode a single tagged value at ``pos``; returns (value, new_pos)."""
    return _decode(buf, pos)


def decode_row(buf: bytes) -> List[Any]:
    """Decode a self-delimiting row payload into a list of Python values."""
    out: List[Any] = []
    pos = 0
    while pos < len(buf):
        value, pos = _decode(buf, pos)
        out.append(value)
    if pos != len(buf):
        raise CorruptError("row payload misaligned")
    return out


def encoded_size(value: Any) -> int:
    """Number of bytes this value will occupy on a page."""
    if value is None:
        return SIZEOF_N
    if isinstance(value, bool) or _is_int(value):
        return SIZEOF_I
    if isinstance(value, float):
        return SIZEOF_F
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return 1 + len(_varint(len(raw))) + len(raw)
    if isinstance(value, bytes):
        return 1 + len(_varint(len(value))) + len(value)
    if isinstance(value, date):
        return 1 + 4
    raise TypeViolation(f"unsupported value type: {type(value).__name__}")


def row_size(values: List[Any]) -> int:
    return sum(encoded_size(v) for v in values)


def coerce(value: Any, kind: str, colname: str = "?") -> Any:
    """Coerce a Python value to the storage representation for ``kind``."""
    if value is None:
        return None
    if kind == "INT":
        if _is_int(value):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeViolation(f"column {colname!r} expects INT, got {type(value).__name__}")
    if kind == "FLOAT":
        if _is_int(value):
            return float(value)
        if isinstance(value, float):
            return value
        raise TypeViolation(f"column {colname!r} expects FLOAT, got {type(value).__name__}")
    if kind in ("TEXT", "VARCHAR"):
        if isinstance(value, str):
            return value
        raise TypeViolation(f"column {colname!r} expects TEXT, got {type(value).__name__}")
    if kind == "DATE":
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                raise TypeViolation(f"column {colname!r} expects DATE as YYYY-MM-DD, got {value!r}")
        raise TypeViolation(f"column {colname!r} expects DATE, got {type(value).__name__}")
    raise TypeViolation(f"unknown column type {kind!r}")
