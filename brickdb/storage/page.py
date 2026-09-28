"""Slotted pages and heap files: the storage format for table data.

Slotted page layout (the classic design used by PostgreSQL, SQLite and most
textbooks):

    low addresses                                  high addresses
    +------------+----------------+-------------------+------------+
    | page header| slot directory |    free space     |  row data  |
    |  4 bytes   | 4 bytes/slot   |      ...          |   <----    |
    +------------+----------------+-------------------+------------+

The slot directory grows *forward* from the end of the header; row payloads
grow *backward* from the end of the page. The page is full when the two
regions meet. Each slot is an ``(offset, length)`` pair. Deleting a row only
tombstones its slot (length = 0); :meth:`SlottedPage.compact` slides live
rows down to reclaim the space.

A *heap file* is an unordered collection of pages tracked by the catalog.
Full scans visit every page in page-id order; the B+ tree index layer exists
precisely to avoid them.
"""

from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Optional, Tuple

from ..errors import CorruptError, PageFull
from .codec import decode_row, encode_row
from .disk import PAGE_SIZE

PAGE_HEADER_FMT = "<HH"  # num_slots, free_offset (start of free space)
PAGE_HEADER_SIZE = 4
SLOT_SIZE = 4  # (u16 offset, u16 length)


class SlottedPage:
    """A single slotted page, backed by a PAGE_SIZE bytearray."""

    __slots__ = ("page_id", "buf")

    def __init__(self, page_id: int, buf: Optional[bytearray] = None):
        self.page_id = page_id
        self.buf = bytearray(PAGE_SIZE) if buf is None else buf
        if len(self.buf) != PAGE_SIZE:
            raise CorruptError("page buffer must be exactly PAGE_SIZE")

    # ------------------------------------------------------------- header

    @property
    def num_slots(self) -> int:
        return struct.unpack_from(PAGE_HEADER_FMT, self.buf, 0)[0]

    @property
    def free_offset(self) -> int:
        """Offset where the free space begins (end of the slot directory)."""
        return struct.unpack_from(PAGE_HEADER_FMT, self.buf, 0)[1]

    def _set_header(self, num_slots: int, free_offset: int) -> None:
        struct.pack_into(PAGE_HEADER_FMT, self.buf, 0, num_slots, free_offset)

    # -------------------------------------------------------------- slots

    def _slot(self, i: int) -> Tuple[int, int]:
        base = PAGE_HEADER_SIZE + i * SLOT_SIZE
        return struct.unpack_from("<HH", self.buf, base)

    def _set_slot(self, i: int, off: int, length: int) -> None:
        base = PAGE_HEADER_SIZE + i * SLOT_SIZE
        struct.pack_into("<HH", self.buf, base, off, length)

    # --------------------------------------------------------------- rows

    def is_empty(self) -> bool:
        return self.num_slots == 0

    def row(self, slot: int) -> Optional[List]:
        """Decode the row in ``slot``, or None if the slot is dead/absent."""
        if slot >= self.num_slots:
            return None
        off, length = self._slot(slot)
        if length == 0:
            return None  # tombstone from DELETE
        return decode_row(bytes(self.buf[off : off + length]))

    def _free_bytes(self) -> int:
        if self.num_slots == 0:
            # A freshly allocated (all-zero) page has free_offset == 0, which
            # is not a valid offset; treat it as "everything after the
            # header is free". Slot count is the disambiguator: a page that
            # ever held rows keeps num_slots > 0 even after deletions.
            return PAGE_SIZE - PAGE_HEADER_SIZE
        return self.free_offset - (PAGE_HEADER_SIZE + SLOT_SIZE * self.num_slots)

    def has_room(self, payload_len: int) -> bool:
        return self._free_bytes() >= payload_len + SLOT_SIZE

    def insert(self, values: List) -> int:
        """Append a row and return its slot index. Raises PageFull if no room."""
        payload = encode_row(values)
        if not 0 < len(payload) <= 0xFFFF:
            raise CorruptError(f"row payload size {len(payload)} out of range")
        if not self.has_room(len(payload)):
            raise PageFull(f"page {self.page_id} has no room for {len(payload)}B payload")
        # Place the payload directly below the existing rows: on a fresh
        # page that is the page end, otherwise it is the current data start.
        limit = self.free_offset if self.num_slots else PAGE_SIZE
        start = limit - len(payload)
        self.buf[start:limit] = payload
        slot = self.num_slots
        self._set_slot(slot, start, len(payload))
        self._set_header(slot + 1, start)
        return slot

    def mark_deleted(self, slot: int) -> bool:
        """Tombstone a slot (set length to 0). True if the slot was live."""
        if slot >= self.num_slots:
            return False
        off, length = self._slot(slot)
        if length == 0:
            return False
        self._set_slot(slot, off, 0)
        return True

    def live_slots(self) -> Iterator[Tuple[int, List]]:
        """Yield ``(slot, row)`` for every live row on the page."""
        for i in range(self.num_slots):
            row = self.row(i)
            if row is not None:
                yield i, row

    def count_live(self) -> int:
        return sum(1 for _ in self.live_slots())

    def compact(self) -> None:
        """Defragment: slide all live rows to the bottom of the page.

        Slot *indexes* are preserved exactly — tombstoned slots stay as
        zero-length entries so that ``(page_id, slot)`` row locators stored
        in indexes remain valid after compaction. Only the data area moves.
        """
        entries: List[Tuple[int, int, Optional[List]]] = []
        for i in range(self.num_slots):
            off, length = self._slot(i)
            if length == 0:
                entries.append((i, 0, None))
            else:
                entries.append((i, length, self.row(i)))
        # Write live rows from the bottom up, remembering new offsets.
        new_offs: List[Tuple[int, int]] = []
        end = PAGE_SIZE
        for _, length, row in entries:
            if row is None:
                continue
            payload = encode_row(row)
            start = end - len(payload)
            self.buf[start:end] = payload
            new_offs.append((start, len(payload)))
            end = start
        # Rewrite the directory: same slot count/order, new offsets.
        k = 0
        for i, _, row in entries:
            if row is None:
                self._set_slot(i, 0, 0)
            else:
                self._set_slot(i, *new_offs[k])
                k += 1
        self._set_header(self.num_slots, end)

    def to_bytes(self) -> bytes:
        return bytes(self.buf)

    def stats(self) -> Dict[str, int]:
        return {
            "page_id": self.page_id,
            "slots": self.num_slots,
            "live_rows": self.count_live(),
            "free_bytes": self._free_bytes(),
        }
