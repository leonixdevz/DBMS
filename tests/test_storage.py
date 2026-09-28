"""Storage layer tests: codec, slotted pages, disk manager, buffer pool."""

from __future__ import annotations

from datetime import date

import pytest

from brickdb.buffer import BufferPool
from brickdb.errors import CorruptError, PageFull
from brickdb.storage.codec import coerce, decode_row, encode_row, encoded_size
from brickdb.storage.disk import PAGE_SIZE, DiskManager
from brickdb.storage.page import SlottedPage


# ------------------------------------------------------------------ codec

def test_codec_roundtrip_all_types():
    values = [None, 42, -0, 3.5, "hello", "", "unicode: héllo ✓", True, False, date(2024, 5, 17)]
    row = encode_row(values)
    assert decode_row(row) == values


def test_codec_is_self_delimiting():
    row = encode_row([1, "two", None, 3.0])
    assert decode_row(row) == [1, "two", None, 3.0]
    assert encoded_size("abc") == 5  # tag + length byte + 3 payload bytes


def test_codec_rejects_bad_types():
    with pytest.raises(CorruptError):
        decode_row(b"\xff\xff")


def test_coerce_rules():
    assert coerce(3, "FLOAT") == 3.0
    assert coerce(3.0, "INT") == 3
    assert coerce("x", "TEXT") == "x"
    assert coerce("2024-01-05", "DATE") == date(2024, 1, 5)
    assert coerce(None, "INT") is None
    with pytest.raises(Exception):
        coerce("abc", "INT")


# -------------------------------------------------------------- slotted page

def test_slotted_page_insert_read_delete():
    page = SlottedPage(7)
    a = page.insert([1, "alpha"])
    b = page.insert([2, "beta"])
    assert page.row(a) == [1, "alpha"]
    assert page.row(b) == [2, "beta"]
    assert page.count_live() == 2
    assert page.mark_deleted(a) is True
    assert page.row(a) is None
    assert page.count_live() == 1
    assert page.mark_deleted(a) is False  # already dead


def test_slotted_page_slot_indexes_are_stable_across_compaction():
    page = SlottedPage(1)
    slots = [page.insert([i, "x" * (i + 1)]) for i in range(5)]
    page.mark_deleted(slots[1])
    page.compact()
    # Slot numbers never shift: indexes rely on (page_id, slot) locators.
    assert page.row(slots[0]) == [0, "x"]
    assert page.row(slots[1]) is None
    assert page.row(slots[2]) == [2, "xxx"]
    assert page.row(slots[4]) == [4, "xxxxx"]
    assert page.num_slots == 5


def test_slotted_page_fills_and_reports_full():
    page = SlottedPage(2)
    biggest = "y" * 200
    inserted = 0
    with pytest.raises(PageFull):
        while True:
            page.insert([inserted, biggest])
            inserted += 1
    assert inserted > 10  # a 4 KiB page holds plenty of small rows
    assert page.count_live() == inserted


def test_slotted_page_rejects_oversized_row():
    page = SlottedPage(3)
    with pytest.raises(PageFull):
        page.insert(["z" * (PAGE_SIZE + 10)])


# ---------------------------------------------------------------- disk manager

def test_disk_manager_roundtrip(tmp_path):
    path = str(tmp_path / "d.db")
    disk = DiskManager(path)
    assert disk.num_pages == 1  # header page only
    pid = disk.allocate_page()
    assert pid == 1
    buf = bytearray(PAGE_SIZE)
    buf[:4] = b"data"
    disk.write_page(pid, bytes(buf))
    out = bytearray(PAGE_SIZE)
    disk.read_page(pid, out)
    assert bytes(out[:4]) == b"data"
    disk.close()

    # Reopening validates the header checksum and magic.
    disk2 = DiskManager(path)
    assert disk2.num_pages == 2
    disk2.close()


def test_disk_manager_detects_corrupt_header(tmp_path):
    path = str(tmp_path / "bad.db")
    DiskManager(path).close()
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"NOTADB\x00\x00")
    with pytest.raises(CorruptError):
        DiskManager(path)


def test_disk_manager_rejects_read_past_eof(tmp_path):
    disk = DiskManager(str(tmp_path / "e.db"))
    with pytest.raises(CorruptError):
        disk.read_page(99, bytearray(PAGE_SIZE))
    disk.close()


# ---------------------------------------------------------------- buffer pool

def test_buffer_pool_caches_and_counts_hits(tmp_path):
    disk = DiskManager(str(tmp_path / "b.db"))
    pid = disk.allocate_page()
    pool = BufferPool(disk, size=4)
    first = pool.fetch(pid, pin=False)
    first[:4] = b"abcd"
    pool.unpin(pid, dirty=True)
    again = pool.fetch(pid, pin=False)
    assert bytes(again[:4]) == b"abcd"  # served from memory
    stats = pool.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    disk.close()


def test_buffer_pool_evicts_lru_and_flushes_dirty(tmp_path):
    disk = DiskManager(str(tmp_path / "b.db"))
    for _ in range(3):
        disk.allocate_page()
    pool = BufferPool(disk, size=2)
    for pid in (1, 2):
        buf = pool.fetch(pid, pin=False)
        buf[:3] = f"p{pid} ".encode()  # exactly 3 bytes, never resizes the page
        pool.unpin(pid, dirty=True)
    # Touching page 1 makes page 2 the least recently used.
    pool.fetch(1, pin=False)
    pool.fetch(3, pin=False)  # forces an eviction
    assert pool.stats()["resident"] == 2
    pool.flush_all()
    disk.close()

    disk2 = DiskManager(str(tmp_path / "b.db"))
    out = bytearray(PAGE_SIZE)
    disk2.read_page(2, out)  # page 2 was evicted and written back
    assert bytes(out[:3]) == b"p2 "
    disk2.close()


def test_buffer_pool_refuses_to_evict_pinned_pages(tmp_path):
    disk = DiskManager(str(tmp_path / "p.db"))
    for _ in range(3):
        disk.allocate_page()
    pool = BufferPool(disk, size=2)
    pool.fetch(1)  # pinned
    pool.fetch(2)  # pinned
    with pytest.raises(RuntimeError):
        pool.fetch(3)
    pool.unpin(1)
    pool.fetch(3)  # now evictable
    disk.close()


def test_buffer_pool_calls_on_flush_hook_before_writes(tmp_path):
    disk = DiskManager(str(tmp_path / "h.db"))
    pid = disk.allocate_page()
    seen = []
    pool = BufferPool(disk, size=2, on_flush=lambda page_id, image: seen.append(page_id))
    buf = pool.fetch(pid, pin=False)
    buf[:1] = b"x"
    pool.unpin(pid, dirty=True)
    pool.flush_all()
    assert seen == [pid]  # the write-ahead hook fired before the page write
    disk.close()
