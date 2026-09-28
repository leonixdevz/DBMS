"""Heap file: append-mostly row storage over slotted pages.

A table's rows live in the pages listed in its :class:`TableDef`. Inserts go
to the last page with room (compacting tombstones first); a new page is
allocated only when nothing fits. Rows are identified by a *locator*
``(page_id, slot)`` — the same handle the B+ tree payloads store.

Locators are permanent: compaction never moves a row between pages and never
renumbers slots, so an index entry stays valid for the row's whole life.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from .errors import CatalogError, NotNullViolation
from .storage.codec import row_size
from .storage.page import SlottedPage
from .storage.pager import Pager

Locator = Tuple[int, int]


class HeapFile:
    """Row storage for one table.

    ``on_grow`` is called whenever a new heap page is allocated; the engine
    uses it to mark the catalog dirty so the page list becomes durable
    together with the rows that landed on the new page.
    """

    def __init__(self, pager: Pager, tdef, on_grow=None):
        self.pager = pager
        self.tdef = tdef
        self.on_grow = on_grow

    # ------------------------------------------------------------ helpers

    def _page(self, page_id: int) -> SlottedPage:
        return SlottedPage(page_id, bytearray(self.pager.read(page_id)))

    def _write_page(self, page: SlottedPage) -> None:
        self.pager.write(page.page_id, page.to_bytes())

    def _new_page(self) -> SlottedPage:
        page_id = self.pager.alloc()
        page = SlottedPage(page_id)
        self.tdef.heap_pages.append(page_id)
        if self.on_grow is not None:
            self.on_grow()
        return page

    # ------------------------------------------------------------- reads

    def scan(self) -> Iterator[Tuple[Locator, List]]:
        """Yield ``(locator, row)`` for every live row in page order."""
        for page_id in list(self.tdef.heap_pages):
            page = self._page(page_id)
            for slot, row in page.live_slots():
                yield (page_id, slot), row

    def get(self, locator: Locator) -> Optional[List]:
        page_id, slot = locator
        if page_id not in self.tdef.heap_pages:
            raise CatalogError(f"page {page_id} does not belong to table {self.tdef.name!r}")
        return self._page(page_id).row(slot)

    def count_rows(self) -> int:
        return sum(1 for _ in self.scan())

    # ------------------------------------------------------------ writes

    def insert(self, values: List) -> Locator:
        """Insert one row, validating NOT NULL. Returns its locator."""
        if len(values) != self.tdef.ncols:
            raise CatalogError(
                f"table {self.tdef.name!r} expects {self.tdef.ncols} values, got {len(values)}"
            )
        for col, value in zip(self.tdef.columns, values):
            if col.notnull and value is None:
                raise NotNullViolation(f"column {col.name!r} cannot be NULL")
        payload_hint = row_size(values)
        for page_id in reversed(list(self.tdef.heap_pages)):
            page = self._page(page_id)
            if page.has_room(payload_hint):
                slot = page.insert(values)
                self._write_page(page)
                return (page_id, slot)
            # Last resort on the oldest-ish pages: compact and retry once.
            if page.count_live() < page.num_slots:
                page.compact()
                if page.has_room(payload_hint):
                    slot = page.insert(values)
                    self._write_page(page)
                    return (page_id, slot)
        page = self._new_page()
        slot = page.insert(values)
        self._write_page(page)
        return (page.page_id, slot)

    def delete(self, locator: Locator) -> bool:
        page_id, slot = locator
        page = self._page(page_id)
        if page.mark_deleted(slot):
            self._write_page(page)
            return True
        return False

    def stats(self) -> dict:
        pages = len(self.tdef.heap_pages)
        rows = 0
        dead = 0
        for page_id in self.tdef.heap_pages:
            page = self._page(page_id)
            rows += page.count_live()
            dead += page.num_slots - page.count_live()
        return {"pages": pages, "rows": rows, "dead_slots": dead}
