"""Disk manager: the only layer allowed to touch the database files.

File layout (single ``*.db`` file):

    page 0        file header: magic, page size, CRC32 of the header
    page 1..N     data pages, addressed by their *global page id*
                  (page id == byte offset / PAGE_SIZE)

Pages are appended at the end of the file and never move, so a page id is a
stable permanent address. Reclamation of empty pages is left as future work
(no vacuum) — real systems solve this with free lists or segment compaction.

Neither the disk manager nor the buffer pool above it ever issues a write
without the WAL being informed first (see ``txn/wal.py``).
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Dict

from ..errors import CorruptError

MAGIC = b"BRICKDB1"  # exactly 8 bytes: it is unpacked as one "<8s" field
PAGE_SIZE = 4096

# Header page: magic(8) page_size(4) reserved(4080) crc32(4)
_HEADER_FMT = "<8sI4080xI"
assert struct.calcsize(_HEADER_FMT) == PAGE_SIZE


class DiskManager:
    """Append-only file of fixed-size pages, numbered by global page id."""

    def __init__(self, path: str):
        self.path = path
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT)
        if exists:
            self._load_header()
        else:
            self._init_header()

    # ------------------------------------------------------------- header

    def _init_header(self) -> None:
        header = struct.pack(_HEADER_FMT, MAGIC, PAGE_SIZE, 0)
        header = header[:-4] + struct.pack("<I", zlib.crc32(header[:-4]) & 0xFFFFFFFF)
        os.write(self._fd, header)
        os.fsync(self._fd)

    def _load_header(self) -> None:
        raw = os.pread(self._fd, PAGE_SIZE, 0)
        if len(raw) != PAGE_SIZE:
            raise CorruptError(f"database file too small for header: {self.path}")
        body, crc = raw[:-4], struct.unpack_from("<I", raw, PAGE_SIZE - 4)[0]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            raise CorruptError("database header checksum mismatch (file corrupt?)")
        magic, page_size, _crc = struct.unpack(_HEADER_FMT, raw)
        if magic != MAGIC:
            raise CorruptError(f"not a brickdb database file: {self.path}")
        if page_size != PAGE_SIZE:
            raise CorruptError(f"unexpected page size {page_size}")

    # ---------------------------------------------------------- page math

    @property
    def num_pages(self) -> int:
        """Total pages in the file, including the header page (page 0)."""
        return os.fstat(self._fd).st_size // PAGE_SIZE

    def allocate_page(self) -> int:
        """Append one zeroed page to the file and return its page id."""
        end = os.fstat(self._fd).st_size
        page_id = end // PAGE_SIZE
        os.pwrite(self._fd, b"\x00" * PAGE_SIZE, end)
        return page_id

    def read_page(self, page_id: int, buf: bytearray) -> None:
        """Read page ``page_id`` into ``buf`` (a PAGE_SIZE bytearray)."""
        off = page_id * PAGE_SIZE
        if off + PAGE_SIZE > os.fstat(self._fd).st_size:
            raise CorruptError(f"read past end of file: page {page_id}")
        data = os.pread(self._fd, PAGE_SIZE, off)
        if len(data) != PAGE_SIZE:
            raise CorruptError(f"short read on page {page_id}")
        buf[:] = data

    def write_page(self, page_id: int, buf: bytes) -> None:
        if len(buf) != PAGE_SIZE:
            raise CorruptError("write_page requires exactly PAGE_SIZE bytes")
        os.pwrite(self._fd, buf, page_id * PAGE_SIZE)

    def sync(self) -> None:
        os.fsync(self._fd)

    def close(self) -> None:
        try:
            self.sync()
        finally:
            os.close(self._fd)

    def file_size_bytes(self) -> int:
        return os.fstat(self._fd).st_size

    def stats(self) -> Dict[str, int]:
        return {
            "pages": self.num_pages,
            "file_bytes": self.file_size_bytes(),
            "page_size": PAGE_SIZE,
        }
