"""B+ tree tests: splits, searches, range scans, deletes and durability."""

from __future__ import annotations

import random

import pytest

from brickdb.buffer import BufferPool
from brickdb.errors import UniqueViolation
from brickdb.index.btree import BTree
from brickdb.index.keyenc import (
    decode_key,
    encode_key,
    encode_key_part,
    pack_payload,
    unpack_payload,
)
from brickdb.storage.disk import DiskManager
from brickdb.storage.pager import Pager


@pytest.fixture()
def pager(tmp_path):
    disk = DiskManager(str(tmp_path / "bt.db"))
    for _ in range(2):
        disk.allocate_page()  # keep page ids away from 0/1 like the real engine
    pool = BufferPool(disk, size=64)
    yield Pager(disk, pool)
    disk.close()


def k(n: int) -> bytes:
    return encode_key_part("INT", n)


# ---------------------------------------------------------------- key encoding

def test_integer_key_encoding_is_order_preserving():
    keys = [encode_key_part("INT", n) for n in (0, 1, 2, 9, 10, 255, 1000)]
    assert keys == sorted(keys)


def test_text_key_encoding_is_order_preserving():
    # Note the logical order: 'z' (U+007A) sorts before '\u00ff' (U+00FF).
    words = ["", "a", "aa", "ab", "b", "z" * 50, "\u00ff\u00ff"]
    keys = [encode_key_part("TEXT", w) for w in words]
    assert keys == sorted(keys)


def test_float_key_encoding_is_order_preserving():
    values = [-1e9, -1.5, -0.0, 0.0, 0.5, 1.0, 42.25, 1e9]
    keys = [encode_key_part("FLOAT", v) for v in values]
    assert keys == sorted(keys)


def test_composite_keys_roundtrip():
    kinds = ["TEXT", "INT"]
    key = encode_key(kinds, ["alice", 30])
    assert decode_key(kinds, key) == ["alice", 30]


# -------------------------------------------------------------------- search

def test_insert_and_search(pager):
    tree = BTree(pager, None)
    tree.insert(k(5), (10, 1))
    tree.insert(k(3), (10, 2))
    tree.insert(k(9), (10, 3))
    assert tree.search(k(3)) == (10, 2)
    assert tree.search(k(5)) == (10, 1)
    assert tree.search(k(9)) == (10, 3)
    assert tree.search(k(4)) is None
    assert tree.verify() == []


def test_unique_index_rejects_duplicates(pager):
    tree = BTree(pager, None)
    tree.insert(k(1), (1, 0), unique=True)
    with pytest.raises(UniqueViolation):
        tree.insert(k(1), (1, 1), unique=True)


def test_non_unique_index_keeps_duplicate_keys(pager):
    tree = BTree(pager, None)
    tree.insert(k(7), (1, 0))
    tree.insert(k(7), (1, 1))
    assert tree.search(k(7)) in ((1, 0), (1, 1))
    assert tree.count_keys() == 2


def test_many_inserts_split_and_stay_correct(pager):
    tree = BTree(pager, None)
    n = 400
    for i in range(n):
        tree.insert(k(i), (i, i % 16))
    assert tree.height() >= 2  # splits definitely happened
    assert tree.verify() == []
    for i in range(n):
        assert tree.search(k(i)) == (i, i % 16)
    assert tree.count_keys() == n


def test_random_insert_delete_matches_reference_model(pager):
    """Random insert/delete mix checked against a reference multiset.

    A non-unique index keeps every entry, so the model tracks the live
    locators per key rather than one row per key. Locators are unique here
    exactly as row handles are in the engine: two insertions of the same key
    are distinct entries, and a separator may land between them.
    """
    rng = random.Random(1234)
    tree = BTree(pager, None)
    reference = {}  # key -> live locators, in insertion order
    for step in range(600):
        key = rng.randrange(200)
        locator = (key, step)  # unique per operation
        if rng.random() < 0.6:
            tree.insert(k(key), locator)
            reference.setdefault(key, []).append(locator)
        elif reference.get(key):
            victim = reference[key][0]
            assert tree.delete(k(key), pack_payload(*victim)) is True
            reference[key].remove(victim)
            if not reference[key]:
                del reference[key]
        else:
            assert tree.delete(k(key)) is False
        if step % 100 == 0:
            assert tree.verify() == [], f"tree broke at step {step}"
    assert tree.count_keys() == sum(len(v) for v in reference.values())
    for key, locators in sorted(reference.items()):
        assert tree.search(k(key)) in locators
        assert {loc for _, loc in tree.prefix_scan(k(key))} == set(locators)
        for locator in locators:
            assert tree.delete(k(key), pack_payload(*locator)) is True
    assert tree.count_keys() == 0
    assert tree.verify() == []


# --------------------------------------------------- duplicate (non-unique) keys

def test_many_duplicate_keys_split_and_stay_reachable(pager):
    """A run of equal keys must never straddle a separator.

    Splitting a page full of one repeated key used to put that key on both
    sides of the separator, which made the entries on the wrong side
    unreachable. Entries are ordered by ``(key, locator)`` so the split point
    is always an exact boundary.
    """
    tree = BTree(pager, None)
    locators = [(1, i) for i in range(60)]
    for locator in locators:
        tree.insert(k(42), locator)
    assert tree.height() >= 2  # splits definitely happened
    assert tree.verify() == []
    assert tree.count_keys() == 60
    # Scanning the one key returns every duplicate, in locator order.
    assert list(tree.range_scan(k(42), True, k(42), True)) == locators
    assert tree.search(k(42)) in locators
    assert tree.search(k(41)) is None
    # Deleting half of them rebalances without losing the rest.
    for locator in locators[::2]:
        assert tree.delete(k(42), pack_payload(*locator)) is True
    assert tree.verify() == []
    assert tree.count_keys() == 30
    assert list(tree.range_scan(k(42), True, k(42), True)) == locators[1::2]


def test_duplicate_keys_interleave_with_neighbours(pager):
    """Duplicates mixed with other keys keep the tree ordered and complete."""
    tree = BTree(pager, None)
    expected = []
    for i in range(40):
        for slot in range(3):  # three copies of every key
            locator = (i, slot)
            tree.insert(k(i), locator)
            expected.append((i, locator))
    assert tree.verify() == []
    assert tree.count_keys() == len(expected)
    scanned = [(i, loc) for i in range(40) for loc in tree.range_scan(k(i), True, k(i), True)]
    assert scanned == expected
    for i in range(0, 40, 2):
        for locator in [(i, 0), (i, 1), (i, 2)]:
            assert tree.delete(k(i), pack_payload(*locator)) is True
    assert tree.verify() == []
    assert tree.count_keys() == 60
    assert tree.search(k(1)) in [(1, 0), (1, 1), (1, 2)]
    assert tree.search(k(2)) is None


# -------------------------------------------------------------------- scans

def test_range_scan_bounds(pager):
    tree = BTree(pager, None)
    for i in range(20):
        tree.insert(k(i), (i, 0))
    got = [unpack_payload(pack_payload(*loc)) for loc in tree.range_scan(k(5), True, k(9), True)]
    assert got == [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
    got = [loc for loc in tree.range_scan(k(5), False, k(9), True)]
    assert got == [(6, 0), (7, 0), (8, 0), (9, 0)]
    got = [loc for loc in tree.range_scan(k(5), True, k(9), False)]
    assert got == [(5, 0), (6, 0), (7, 0), (8, 0)]


def test_range_scan_on_composite_prefix(pager):
    """Bounds built from the first component of a composite key must match
    every key that starts with that component (truncated comparison)."""
    kinds = ["TEXT", "INT"]
    tree = BTree(pager, None)
    for name, n in [("alice", 1), ("alice", 2), ("bob", 1), ("carol", 9)]:
        tree.insert(encode_key(kinds, [name, n]), (n, n))
    bound = encode_key_part("TEXT", "alice")
    got = list(tree.range_scan_trunc(bound, True, bound, True))
    assert got == [(1, 1), (2, 2)]


def test_full_scan_is_sorted(pager):
    tree = BTree(pager, None)
    for i in [5, 1, 9, 3, 7]:
        tree.insert(k(i), (i, 0))
    keys = []
    for loc in tree.full_scan():
        keys.append(loc[0])
    assert keys == [1, 3, 5, 7, 9]


# ------------------------------------------------------------------ deletes

def test_delete_with_borrow_and_merge_keeps_invariants(pager):
    tree = BTree(pager, None)
    for i in range(120):
        tree.insert(k(i), (i, 0))
    for i in range(0, 120, 2):  # delete every other key, forcing rebalancing
        assert tree.delete(k(i)) is True
    assert tree.verify() == []
    assert tree.count_keys() == 60
    for i in range(1, 120, 2):
        assert tree.search(k(i)) == (i, 0)
    for i in range(0, 120, 2):
        assert tree.search(k(i)) is None


def test_delete_all_collapses_the_tree(pager):
    tree = BTree(pager, None)
    for i in range(60):
        tree.insert(k(i), (i, 0))
    for i in range(60):
        assert tree.delete(k(i)) is True
    assert tree.delete(k(0)) is False
    assert tree.count_keys() == 0
    assert tree.height() <= 1


def test_non_unique_delete_removes_exact_entry(pager):
    tree = BTree(pager, None)
    tree.insert(k(4), (1, 0))
    tree.insert(k(4), (2, 0))
    assert tree.delete(k(4), pack_payload(1, 0)) is True
    remaining = [loc for loc in tree.full_scan()]
    assert remaining == [(2, 0)]


# --------------------------------------------------------------- durability

def test_tree_survives_reopen(tmp_path):
    path = str(tmp_path / "persist.db")
    disk = DiskManager(path)
    for _ in range(2):
        disk.allocate_page()
    pool = BufferPool(disk, size=32)
    pager = Pager(disk, pool)
    tree = BTree(pager, None)
    for i in range(150):
        tree.insert(k(i), (i, i))
    root = tree.root
    pager.publish(pager.stage())
    pool.flush_all()
    disk.close()

    disk2 = DiskManager(path)
    pool2 = BufferPool(disk2, size=32)
    tree2 = BTree(Pager(disk2, pool2), root)
    assert tree2.count_keys() == 150
    assert tree2.search(k(149)) == (149, 149)
    assert tree2.verify() == []
    disk2.close()
