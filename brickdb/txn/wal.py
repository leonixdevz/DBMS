"""Write-ahead log (WAL) — the heart of crash safety.

**The write-ahead rule:** before any data page is written to the database
file, the log record describing that change must already be on stable
storage. brickdb enforces this in two places:

1. The pager publishes pages to the buffer pool only *after*
   :meth:`WriteAheadLog.log_commit_pages` has appended and fsynced the
   transaction's commit record.
2. The buffer pool's ``on_flush`` hook forces a log sync before every
   physical page write, covering flushes that race ahead of the pager.

Record format (little-endian)::

    +------+-----+-----+---------+---------+-----+
    | len4 | lsn | txn | type(1) | payload | crc |
    +------+-----+-----+---------+---------+-----+

``len`` covers everything after it; ``crc32`` covers the record body, so a
torn or corrupted tail is detected and the log is replayed only up to the
last intact record.

Recovery is **redo-only** (the no-steal buffer policy means dirty uncommitted
pages never reach the data file, so there is nothing to undo): scan the log,
collect the newest page images from every *committed* transaction's DATA
records, and write them into the data file. A checkpoint record lets the
engine truncate the log after a clean flush.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Dict, List, Optional, Tuple

from ..storage.disk import DiskManager, PAGE_SIZE

REC_BEGIN = 1
REC_DATA = 2
REC_COMMIT = 3
REC_ABORT = 4
REC_CKPT = 5

_HEADER = struct.Struct("<IIQ")  # len, lsn, txn


class WriteAheadLog:
    """Append-only CRC-protected log over a dedicated ``*.wal`` file."""

    def __init__(self, path: str):
        self.path = path
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
        self.lsn = self._scan_last_lsn()
        self._next_lsn = self.lsn + 1

    # ------------------------------------------------------------- append

    def _append(self, txn: int, rtype: int, payload: bytes) -> int:
        lsn = self._next_lsn
        tail = bytes([rtype]) + payload
        # Body layout: len(4) lsn(4) txn(8) type(1) payload(n).
        size = _HEADER.size + len(tail)
        body = _HEADER.pack(size, lsn, txn) + tail
        rec = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        os.write(self._fd, rec)
        self._next_lsn += 1
        self.lsn = lsn
        return lsn

    def sync(self) -> None:
        os.fsync(self._fd)

    # ---------------------------------------------------------- protocol

    def log_begin(self, txn: int) -> None:
        self._append(txn, REC_BEGIN, b"")

    def log_commit_pages(self, txn: int, name: str, pages: List[Tuple[int, bytes]]) -> None:
        """Append one DATA record (the page batch) + COMMIT, then fsync.

        This is the commit point: when this returns, the transaction is
        durable regardless of what happens to the data file afterwards.
        """
        if pages:
            payload = name.encode("utf-8") + b"\x00" + struct.pack("<I", len(pages))
            chunks = [payload]
            for page_id, image in pages:
                chunks.append(struct.pack("<I", page_id) + image)
            self._append(txn, REC_DATA, b"".join(chunks))
        self._append(txn, REC_COMMIT, b"")
        self.sync()

    def log_abort(self, txn: int) -> None:
        self._append(txn, REC_ABORT, b"")
        self.sync()

    def checkpoint(self) -> None:
        """Write a checkpoint marker, then truncate the log to empty.

        The engine calls this only after all committed pages have been
        flushed to the data file, so everything before the checkpoint is
        safely on disk and the log has nothing left to say.
        """
        self._append(0, REC_CKPT, b"")
        self.sync()
        os.ftruncate(self._fd, 0)
        self._next_lsn = 1
        self.lsn = 0

    # ---------------------------------------------------------- recovery

    def _scan_last_lsn(self) -> int:
        records = self._read_records()
        return records[-1][1] if records else 0

    def _read_records(self) -> List[Tuple[int, int, int, int, bytes]]:
        """Return (file_offset, lsn, txn, type, payload) for intact records."""
        size = os.fstat(self._fd).st_size
        records: List[Tuple[int, int, int, int, bytes]] = []
        pos = 0
        while pos + 4 <= size:
            (declared_len,) = struct.unpack("<I", os.pread(self._fd, 4, pos))
            if declared_len < 17 or pos + declared_len + 4 > size:
                break  # torn tail
            raw = os.pread(self._fd, declared_len + 4, pos)
            body, crc = raw[:-4], struct.unpack_from("<I", raw, declared_len)[0]
            if zlib.crc32(body) & 0xFFFFFFFF != crc:
                break  # corrupt tail: stop replay here
            _l, lsn, txn = _HEADER.unpack_from(body, 0)
            rtype = body[16]
            records.append((pos, lsn, txn, rtype, body[17:]))
            pos += declared_len + 4
        return records

    def recover(self, disk: DiskManager, force: bool = False) -> Dict[str, int]:
        """Redo committed transactions. Returns stats about what was done."""
        records = self._read_records()
        committed: Dict[int, None] = {t: None for _off, _l, t, rt, _p in records if rt == REC_COMMIT}
        # A DATA record belongs to the transaction that later committed.
        pending: Dict[int, List[Tuple[int, bytes]]] = {}
        for _off, _lsn, txn, rtype, payload in records:
            if rtype == REC_DATA and txn in committed:
                pages = self._parse_data(payload)
                pending.setdefault(txn, []).extend(pages)
        # Newest image per page wins across all committed transactions.
        images: Dict[int, bytes] = {}
        for txn in committed:
            for page_id, image in pending.get(txn, []):
                images[page_id] = image
        applied = 0
        if images:
            for page_id, image in sorted(images.items()):
                disk.write_page(page_id, image)
                applied += 1
            disk.sync()
        stats = {"records": len(records), "committed_txns": len(committed), "pages_redone": applied}
        if force and records:
            os.ftruncate(self._fd, 0)
            self._next_lsn = 1
            self.lsn = 0
        return stats

    @staticmethod
    def _parse_data(payload: bytes) -> List[Tuple[int, bytes]]:
        nul = payload.index(b"\x00")
        (count,) = struct.unpack_from("<I", payload, nul + 1)
        pos = nul + 5
        pages: List[Tuple[int, bytes]] = []
        for _ in range(count):
            (page_id,) = struct.unpack_from("<I", payload, pos)
            pos += 4
            pages.append((page_id, payload[pos : pos + PAGE_SIZE]))
            pos += PAGE_SIZE
        return pages

    # ------------------------------------------------------------ misc

    def stats(self) -> dict:
        return {"lsn": self.lsn, "file_bytes": os.fstat(self._fd).st_size, "path": self.path}

    def close(self) -> None:
        self.sync()
        os.close(self._fd)
