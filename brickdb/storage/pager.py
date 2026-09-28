"""Pager: page allocation and transactional page writes over disk + pool.

The pager is the meeting point of the storage layers and the transaction
layer. It exposes three primitives to index/catalog/heap code:

* ``alloc()``       — append a zeroed page, return its id
* ``read(pid)``     — read a page image (via the buffer pool)
* ``write(pid, b)`` — stage a new image for ``pid`` as an *uncommitted* change

Writes are **no-steal**: a staged page is held only in the pager's private
stash and is never flushed before the transaction commits, so an abort needs
no undo work — the pre-image is simply discarded.

Commit is orchestrated by the engine in the write-ahead order:

1. ``stage()`` hands the batch of (page_id, image) pairs to the engine.
2. The engine appends the DATA + COMMIT records to the WAL and fsyncs it.
3. Only then does the engine call ``publish()`` to make the pages visible in
   the buffer pool as dirty; they reach the data file lazily on flush/evict.

Crash between 2 and 3 is safe: recovery redoes the pages from the log.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .disk import DiskManager, PAGE_SIZE


class Pager:
    """Per-transaction view of pages with commit/abort semantics."""

    def __init__(self, disk: DiskManager, pool):
        self.disk = disk
        self.pool = pool
        self._staged: Dict[int, bytes] = {}

    # ----------------------------------------------------------- reading

    def read(self, page_id: int) -> bytes:
        """Return the current image of a page, including staged changes."""
        if page_id in self._staged:
            return self._staged[page_id]
        buf = self.pool.fetch(page_id, pin=False)
        return bytes(buf)

    # ----------------------------------------------------------- writing

    def alloc(self) -> int:
        """Append a zeroed page and stage it as an all-zero uncommitted page."""
        page_id = self.disk.allocate_page()
        self._staged[page_id] = b"\x00" * PAGE_SIZE
        return page_id

    def write(self, page_id: int, image: bytes) -> None:
        """Stage a full-page image as an uncommitted change."""
        if len(image) != PAGE_SIZE:
            raise ValueError("page image must be exactly PAGE_SIZE bytes")
        self._staged[page_id] = bytes(image)

    # -------------------------------------------------------- commit/abort

    def stage(self) -> List[Tuple[int, bytes]]:
        """Return the sorted batch of staged pages (for the WAL record)."""
        return sorted(self._staged.items())

    def publish(self, pages: List[Tuple[int, bytes]]) -> List[int]:
        """Make committed pages visible in the pool as dirty. Returns ids."""
        for page_id, image in pages:
            buf = self.pool.fetch(page_id, pin=False)
            buf[:] = image
            self.pool.unpin(page_id, dirty=True)
        ids = [pid for pid, _ in pages]
        self._staged.clear()
        return ids

    def abort(self) -> None:
        """Discard all staged changes. No undo work needed (no-steal)."""
        self._staged.clear()

    # -------------------------------------------------------------- stats

    def stats(self) -> dict:
        return {"staged_pages": len(self._staged)}
