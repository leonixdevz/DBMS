"""Buffer pool: a bounded in-memory cache of pages over the disk manager.

Semantics:

* Pages are *pinned* while an operator is touching them; the pool never
  evicts a pinned page. Pinning is cooperative: callers must ``unpin`` when
  done, marking the page dirty if they modified it.
* Unpinned pages are eviction candidates under LRU (a Python dict is
  insertion-ordered, so moving an entry to the back on access gives us LRU
  for free).
* Dirty pages are written back to disk *before* eviction; clean pages are
  simply dropped.
* The pool does not know about the WAL. It calls ``on_flush(page_id, bytes)``
  just before every physical write; the engine hooks this so WAL flushes are
  ordered before data-page writes (the write-ahead rule).
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

from .storage.disk import DiskManager, PAGE_SIZE


class BufferPool:
    """LRU buffer pool with pin counting and a write-before-evict contract."""

    def __init__(
        self,
        disk: DiskManager,
        size: int = 64,
        on_flush: Optional[Callable[[int, bytes], None]] = None,
    ):
        if size < 1:
            raise ValueError("buffer pool size must be >= 1")
        self.disk = disk
        self.size = size
        self.on_flush = on_flush
        self._lock = threading.RLock()
        # page_id -> [bytearray, pin_count, dirty]
        self._pages: Dict[int, list] = {}
        self._hits = 0
        self._misses = 0
        self._flushes = 0

    # ------------------------------------------------------------ fetch

    def fetch(self, page_id: int, pin: bool = True) -> bytearray:
        """Return the live bytearray for ``page_id``, pinning it by default."""
        with self._lock:
            entry = self._pages.get(page_id)
            if entry is not None:
                self._hits += 1
                if pin:
                    entry[1] += 1
                # Re-insert to move to the back of the LRU order.
                self._pages.pop(page_id)
                self._pages[page_id] = entry
                return entry[0]
            if len(self._pages) >= self.size:
                self._evict_one()
            buf = bytearray(PAGE_SIZE)
            self.disk.read_page(page_id, buf)
            self._misses += 1
            self._pages[page_id] = [buf, 1 if pin else 0, False]
            return buf

    # -------------------------------------------------------------- pins

    def unpin(self, page_id: int, dirty: bool = False) -> None:
        with self._lock:
            entry = self._pages.get(page_id)
            if entry is None:
                return
            if entry[1] > 0:
                entry[1] -= 1
            if dirty:
                entry[2] = True

    # ---------------------------------------------------------- eviction

    def _evict_one(self) -> None:
        """Evict the LRU unpinned page. Caller must hold the lock."""
        for pid in list(self._pages):
            entry = self._pages[pid]
            if entry[1] == 0:
                self._flush_page_now(pid, entry)
                del self._pages[pid]
                return
        raise RuntimeError(
            f"buffer pool exhausted: {self.size} pages, all pinned "
            "(increase pool size or fix a missing unpin)"
        )

    # ------------------------------------------------------------ writes

    def _flush_page_now(self, page_id: int, entry: list) -> None:
        """Write the page if dirty. Caller must hold the lock."""
        buf, _pin, dirty = entry
        if dirty:
            if self.on_flush is not None:
                self.on_flush(page_id, bytes(buf))
            self.disk.write_page(page_id, buf)
            self._flushes += 1
            entry[2] = False

    def flush_page(self, page_id: int) -> None:
        """Force a dirty page to disk; the page stays resident, now clean."""
        with self._lock:
            entry = self._pages.get(page_id)
            if entry is not None:
                self._flush_page_now(page_id, entry)

    def flush_all(self) -> None:
        with self._lock:
            for pid in list(self._pages):
                self._flush_page_now(pid, self._pages[pid])

    def is_dirty(self, page_id: int) -> bool:
        with self._lock:
            entry = self._pages.get(page_id)
            return bool(entry and entry[2])

    # ---------------------------------------------------------- shutdown

    def close(self) -> None:
        self.flush_all()
        self.disk.sync()

    # -------------------------------------------------------------- stats

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "capacity": self.size,
                "resident": len(self._pages),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "flushes": self._flushes,
            }
