"""On-disk B+ tree index.

All pages go through the pager, so structural changes (splits, merges) are
staged as uncommitted full-page images and become visible atomically at
commit — readers never observe a half-updated tree.

Page image layout (little-endian)::

    leaf                                  internal
    +--------------------------+          +--------------------------+
    | kind=1 | n | rsib |  --  |          | kind=2 | n | rsib | left |
    +--------------------------+          +--------------------------+
    | klen | key | payload |...|          | child | klen | key |... |
    +--------------------------+          +--------------------------+

* ``rsib`` — right-sibling page id, chaining leaves for range scans.
* ``left`` — first (leftmost) child of an internal node.
* every entry is ``(key bytes, 6-byte locator)``, where the locator is the
  row handle ``(page_id u32, slot u16)``. Separators in internal nodes are
  whole entries too, not bare keys.
* ordering is by entry: ``child[i] < sep[i] <= child[i+1]``.

Pairing the locator with the key is what makes duplicate index keys work.
A non-unique index is a multiset, so a separator has to sit *strictly*
between the two children it divides — and no bare key can do that when that
key is present on both sides. Because locators are unique, entries are
unique, and ``(key, locator)`` is a strict total order that the split
point can always respect. This is the same trick as PostgreSQL ordering a
secondary index by ``(key, ctid)`` and InnoDB appending the primary key to
secondary index entries.

Design constants: ``ORDER = 8`` with keys capped at 480 bytes guarantees an
entry always fits a 4 KiB page in the worst case. Small fanout means deeper
trees — a deliberate trade to keep the code simple and correct; production
engines size fanout dynamically from real key widths.
"""

from __future__ import annotations

import bisect
import struct
from typing import Iterator, List, Optional, Tuple

from ..errors import CorruptError
from ..storage.disk import PAGE_SIZE
from ..storage.pager import Pager
from .keyenc import pack_payload, unpack_payload

KIND_LEAF = 1
KIND_INTERNAL = 2

ORDER = 8                 # max keys per node (split when exceeding)
MIN_KEYS = ORDER // 2     # min keys per non-root node
MAX_KEY_LEN = 480
MAX_LOCATOR = b"\xff" * 6  # sorts after every real locator; used as a probe

HEADER_FMT = "<BHII"      # kind, n, right_sibling, first_child
HEADER_SIZE = 16
assert struct.calcsize(HEADER_FMT) <= HEADER_SIZE


def _read_entry(image: bytes, pos: int, page_id: int, where: str) -> Tuple[bytes, int]:
    """Read a length-prefixed key; returns ``(key, offset just past it)``."""
    (klen,) = struct.unpack_from("<H", image, pos)
    pos += 2
    key = image[pos : pos + klen]
    if len(key) != klen:
        raise CorruptError(f"btree {where} {page_id}: truncated key")
    return key, pos + klen


class _Node:
    """Parsed view of one B+ tree page image.

    ``keys`` holds ``(key, locator)`` pairs for leaves *and* internal nodes,
    so a single comparison orders every entry in the tree.
    """

    __slots__ = ("page_id", "kind", "keys", "children", "right_sibling")

    @classmethod
    def blank(cls, page_id: int, kind: int) -> "_Node":
        """An empty node on a freshly allocated page."""
        node = cls.__new__(cls)
        node.page_id = page_id
        node.kind = kind
        node.keys = []
        node.children = []
        node.right_sibling = 0
        return node

    def __init__(self, page_id: int, image: bytes):
        if len(image) != PAGE_SIZE:
            raise CorruptError(f"btree page {page_id}: bad image size")
        kind, n, rsib, first_child = struct.unpack_from(HEADER_FMT, image, 0)
        if kind not in (KIND_LEAF, KIND_INTERNAL):
            raise CorruptError(f"btree page {page_id}: unknown kind {kind}")
        self.page_id = page_id
        self.kind = kind
        self.right_sibling = rsib
        self.keys: List[Tuple[bytes, bytes]] = []
        self.children: List[int] = []
        pos = HEADER_SIZE
        if kind == KIND_LEAF:
            for _ in range(n):
                key, pos = _read_entry(image, pos, page_id, "leaf")
                locator = image[pos : pos + 6]
                pos += 6
                self.keys.append((key, locator))
        else:
            self.children.append(first_child)
            for _ in range(n):
                (child,) = struct.unpack_from("<I", image, pos)
                pos += 4
                key, pos = _read_entry(image, pos, page_id, "internal")
                locator = image[pos : pos + 6]
                pos += 6
                self.keys.append((key, locator))
                self.children.append(child)

    # ---------------------------------------------------------- serialize

    def is_leaf(self) -> bool:
        return self.kind == KIND_LEAF

    def image(self) -> bytes:
        body = bytearray()
        if self.is_leaf():
            for key, locator in self.keys:
                body += struct.pack("<H", len(key)) + key + locator
        else:
            for child, (key, locator) in zip(self.children[1:], self.keys):
                body += (
                    struct.pack("<I", child)
                    + struct.pack("<H", len(key))
                    + key
                    + locator
                )
        first_child = self.children[0] if not self.is_leaf() else 0
        header = struct.pack(HEADER_FMT, self.kind, len(self.keys), self.right_sibling, first_child)
        assert len(header) <= HEADER_SIZE
        out = bytes(header).ljust(HEADER_SIZE, b"\x00") + bytes(body)
        if len(out) > PAGE_SIZE:
            raise CorruptError(f"btree page {self.page_id} overflow ({len(out)}B)")
        return out.ljust(PAGE_SIZE, b"\x00")


def _trunc_cmp(key: bytes, bound: bytes) -> int:  # noqa: D401
    """Compare ``key[:len(bound)]`` against ``bound`` lexicographically."""
    head = key[: len(bound)]
    if head < bound:
        return -1
    if head > bound:
        return 1
    return 0


class BTree:
    """A B+ tree over the pager. Keys are encoded byte strings.

    ``unique`` mirrors the index constraint. Entries are always ordered by
    ``(key, locator)``, so the tree stores duplicates of a key happily; a
    unique index is enforced at insert time instead (``insert(...,
    unique=True)``), while non-unique indexes behave as a multiset.

    The locator identifies one row version, so callers must not insert the
    same ``(key, locator)`` pair twice: entries would then tie, and no
    separator could order them. The engine maintains this — a locator is
    allocated once per row version and freed when that version is deleted.
    """

    def __init__(self, pager: Pager, root_page_id: Optional[int], unique: bool = False):
        self.pager = pager
        self.root: Optional[int] = root_page_id
        self.unique = unique

    # ------------------------------------------------------------ helpers

    def _load(self, page_id: int) -> _Node:
        return _Node(page_id, self.pager.read(page_id))

    def _save(self, node: _Node) -> None:
        self.pager.write(node.page_id, node.image())

    def _new_node(self, kind: int) -> _Node:
        pid = self.pager.alloc()
        return _Node.blank(pid, kind)

    def is_empty(self) -> bool:
        if self.root is None:
            return True
        node = self._load(self.root)
        return node.is_leaf() and not node.keys

    # ------------------------------------------------------------- search

    def _find_leaf(self, key: bytes, locator: bytes = b"") -> _Node:
        """Descend to the leaf where entry ``(key, locator)`` belongs.

        Probing with the smallest locator finds the first entry ``>= key``,
        which is where range scans start. Probing with ``MAX_LOCATOR`` finds
        a leaf that actually holds ``key`` when any leaf does — needed by
        :meth:`search`, because a duplicated key can span leaves.
        """
        node = self._load(self.root)  # type: ignore[index]
        probe = (key, locator)
        while not node.is_leaf():
            idx = bisect.bisect_right(node.keys, probe)
            node = self._load(node.children[idx])
        return node

    def search(self, key: bytes) -> Optional[Tuple[int, int]]:
        """Exact lookup; returns the row locator (page_id, slot) or None."""
        if self.root is None:
            return None
        leaf = self._find_leaf(key, MAX_LOCATOR)
        idx = bisect.bisect_left(leaf.keys, (key, b""))
        if idx < len(leaf.keys) and leaf.keys[idx][0] == key:
            return unpack_payload(leaf.keys[idx][1])
        return None

    # -------------------------------------------------------------- scans

    def range_scan(
        self,
        lo: Optional[bytes] = None,
        lo_inclusive: bool = True,
        hi: Optional[bytes] = None,
        hi_inclusive: bool = True,
    ) -> Iterator[Tuple[int, int]]:
        """Yield row locators with ``lo <= key <= hi`` in ascending key order."""
        yield from self.range_scan_trunc(lo, lo_inclusive, hi, hi_inclusive)

    def range_scan_trunc(
        self,
        lo: Optional[bytes] = None,
        lo_inclusive: bool = True,
        hi: Optional[bytes] = None,
        hi_inclusive: bool = True,
    ) -> Iterator[Tuple[int, int]]:
        """Range scan where bounds compare only their leading bytes.

        A bound produced by ``encode_key`` for the first k columns of a
        composite key is exactly ``len(bound)`` bytes wide, so comparing
        ``key[:len(bound)]`` against the bound implements "first k columns
        satisfy the predicate" — including equality on a prefix (which
        plain byte comparison would miss, since composite keys are strictly
        longer than their prefixes). Works for fixed-width columns (INT,
        FLOAT, DATE) and for TEXT because text parts are 0x00-terminated.
        """
        if self.root is None:
            return
        if lo is None:
            node = self._load(self.root)
            while not node.is_leaf():
                node = self._load(node.children[0])
        else:
            node = self._find_leaf(lo)
        while node is not None:
            for key, locator in node.keys:
                if lo is not None:
                    t = _trunc_cmp(key, lo)
                    if t < 0 or (t == 0 and not lo_inclusive):
                        continue
                if hi is not None:
                    t = _trunc_cmp(key, hi)
                    if t > 0 or (t == 0 and not hi_inclusive):
                        return
                yield unpack_payload(locator)
            node = self._load(node.right_sibling) if node.right_sibling else None

    def full_scan(self) -> Iterator[Tuple[int, int]]:
        yield from self.range_scan()

    # ------------------------------------------------------------- insert

    def insert(self, key: bytes, locator: Tuple[int, int], unique: bool = False) -> bool:
        """Insert ``key -> locator``.

        With ``unique=True`` a duplicate key raises UniqueViolation. Without
        it the tree behaves as a multiset (non-unique secondary index), and
        each entry is identified by ``(key, locator)`` for deletion.
        """
        if len(key) > MAX_KEY_LEN:
            raise CorruptError(f"index key too long ({len(key)}B > {MAX_KEY_LEN}B)")
        entry = (key, pack_payload(*locator))
        if self.root is None:
            root = self._new_node(KIND_LEAF)
            self._save(root)  # make the page parseable before anyone loads it
            self.root = root.page_id
        split = self._insert(self.root, entry, unique)
        if split is not None:
            sep, right_pid = split
            new_root = self._new_node(KIND_INTERNAL)
            new_root.children = [self.root, right_pid]  # type: ignore[list-item]
            new_root.keys = [sep]
            self._save(new_root)
            self.root = new_root.page_id
        return True

    def _insert(
        self, pid: int, entry: Tuple[bytes, bytes], unique: bool
    ) -> Optional[Tuple[Tuple[bytes, bytes], int]]:
        """Recursive insert. Returns (separator entry, new_right_pid) on split."""
        node = self._load(pid)
        key = entry[0]
        if node.is_leaf():
            idx = bisect.bisect_left(node.keys, entry)
            if unique and (
                (idx < len(node.keys) and node.keys[idx][0] == key)
                or (idx > 0 and node.keys[idx - 1][0] == key)
            ):
                from ..errors import UniqueViolation

                raise UniqueViolation("duplicate key in unique index")
            node.keys.insert(idx, entry)
            if len(node.keys) <= ORDER:
                self._save(node)
                return None
            return self._split_leaf(node)
        # internal node
        cidx = bisect.bisect_right(node.keys, entry)
        split = self._insert(node.children[cidx], entry, unique)
        if split is None:
            return None
        sep, right_pid = split
        node.keys.insert(cidx, sep)
        node.children.insert(cidx + 1, right_pid)
        if len(node.keys) <= ORDER:
            self._save(node)
            return None
        return self._split_internal(node)

    def _split_leaf(self, node: _Node) -> Tuple[Tuple[bytes, bytes], int]:
        """Split a leaf in half; the separator is the right half's first entry.

        Locators make entries unique, so ``left < separator <= right`` holds
        strictly even when the key repeats across the split point — the case
        that a bare-key separator cannot express.
        """
        mid = len(node.keys) // 2
        right = self._new_node(KIND_LEAF)
        right.keys = node.keys[mid:]
        node.keys = node.keys[:mid]
        right.right_sibling = node.right_sibling
        node.right_sibling = right.page_id
        self._save(node)
        self._save(right)
        return right.keys[0], right.page_id

    def _split_internal(self, node: _Node) -> Tuple[Tuple[bytes, bytes], int]:
        mid = len(node.keys) // 2
        promote = node.keys[mid]
        right = self._new_node(KIND_INTERNAL)
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        self._save(node)
        self._save(right)
        return promote, right.page_id

    # ------------------------------------------------------------- delete

    def prefix_scan(self, prefix: bytes) -> Iterator[Tuple[int, int]]:
        """Yield locators for every key starting with ``prefix``.

        Used for non-unique composite lookups; robust regardless of what the
        key suffix contains (we stop as soon as the prefix stops matching
        rather than guessing an exclusive upper bound).
        """
        for key_loc in self._prefix_items(prefix):
            yield key_loc

    def _prefix_items(self, prefix: bytes) -> Iterator[Tuple[bytes, Tuple[int, int]]]:
        if self.root is None:
            return
        node = self._find_leaf(prefix)
        while node is not None:
            for key, locator in node.keys:
                if key >= prefix and key.startswith(prefix):
                    yield key, unpack_payload(locator)
                elif key > prefix and not key.startswith(prefix):
                    return
            node = self._load(node.right_sibling) if node.right_sibling else None

    def delete(self, key: bytes, payload: Optional[bytes] = None) -> bool:
        """Remove one entry.

        ``payload=None`` removes the first entry with ``key`` (unique
        indexes). With a payload, removes exactly that ``(key, payload)``
        entry — required for non-unique indexes with duplicate keys.
        """
        if self.root is None:
            return False
        found, _ = self._delete(self.root, key, payload)
        if not found:
            return False
        root = self._load(self.root)
        # Shrinking: an internal root with no separators has one child; promote it.
        while not root.is_leaf() and not root.keys:
            self.root = root.children[0]
            root = self._load(self.root)
        return True

    def _delete(self, pid: int, key: bytes, payload: Optional[bytes]) -> Tuple[bool, bool]:
        """Returns (found, child_underflowed)."""
        node = self._load(pid)
        # With an exact locator the probe is the entry itself, landing on the
        # one child whose range contains it. Without one, probe with the
        # largest locator: that lands on a child that really does hold
        # ``key``, whereas the smallest locator can stop one child short when
        # a separator is itself a copy of ``key``.
        probe = (key, payload if payload is not None else MAX_LOCATOR)
        if node.is_leaf():
            idx = bisect.bisect_left(node.keys, (key, b""))
            target = -1
            while idx < len(node.keys) and node.keys[idx][0] == key:
                if payload is None or node.keys[idx][1] == payload:
                    target = idx
                    break
                idx += 1
            if target < 0:
                return False, False
            del node.keys[target]
            self._save(node)
            return True, len(node.keys) < MIN_KEYS
        cidx = bisect.bisect_right(node.keys, probe)
        found, underflow = self._delete(node.children[cidx], key, payload)
        if not found:
            return False, False
        if underflow:
            self._fix_child(node, cidx)
        self._save(node)
        return True, len(node.keys) < MIN_KEYS

    def _fix_child(self, parent: _Node, cidx: int) -> None:
        """Rebalance ``parent.children[cidx]`` after it underflowed.

        Separator indexes: the separator between children[i] and children[i+1]
        is parent.keys[i], so borrowing from the left sibling uses
        ``cidx - 1`` and from the right sibling uses ``cidx``.
        """
        child = self._load(parent.children[cidx])
        # Try to borrow from the left sibling.
        if cidx > 0:
            left = self._load(parent.children[cidx - 1])
            if len(left.keys) > MIN_KEYS:
                self._borrow(left, child, parent, cidx - 1, from_left=True)
                self._save(left)
                self._save(child)
                return
        # Try to borrow from the right sibling.
        if cidx + 1 < len(parent.children):
            right = self._load(parent.children[cidx + 1])
            if len(right.keys) > MIN_KEYS:
                # donor = right sibling, receiver = the underflowed child
                self._borrow(right, child, parent, cidx, from_left=False)
                self._save(right)
                self._save(child)
                return
        # Merge with a sibling (prefer left).
        if cidx > 0:
            left = self._load(parent.children[cidx - 1])
            self._merge(left, child, parent, cidx - 1)
            self._save(left)
        elif cidx + 1 < len(parent.children):
            right = self._load(parent.children[cidx + 1])
            self._merge(child, right, parent, cidx)
            self._save(child)

    def _borrow(self, donor: _Node, receiver: _Node, parent: _Node, sep_idx: int, from_left: bool) -> None:
        """Move one entry from ``donor`` to ``receiver`` and fix the separator."""
        if donor.is_leaf():
            if from_left:
                receiver.keys.insert(0, donor.keys.pop())
                # New separator = smallest entry in the receiver.
                parent.keys[sep_idx] = receiver.keys[0]
            else:
                receiver.keys.append(donor.keys.pop(0))
                # New separator = smallest entry remaining in the donor.
                parent.keys[sep_idx] = donor.keys[0]
        elif from_left:
            # Rotate through the parent separator.
            receiver.keys.insert(0, parent.keys[sep_idx])
            parent.keys[sep_idx] = donor.keys.pop()
            receiver.children.insert(0, donor.children.pop())
        else:
            receiver.keys.append(parent.keys[sep_idx])
            parent.keys[sep_idx] = donor.keys.pop(0)
            receiver.children.append(donor.children.pop(0))

    def _merge(self, left: _Node, right: _Node, parent: _Node, sep_idx: int) -> None:
        """Merge ``right`` into ``left`` and remove the separator."""
        if left.is_leaf():
            left.keys += right.keys
            left.right_sibling = right.right_sibling
        else:
            left.keys += [parent.keys[sep_idx]] + right.keys
            left.children += right.children
        del parent.keys[sep_idx]
        del parent.children[sep_idx + 1]

    # ---------------------------------------------------------- debugging

    def height(self) -> int:
        if self.root is None:
            return 0
        node = self._load(self.root)
        h = 1
        while not node.is_leaf():
            node = self._load(node.children[0])
            h += 1
        return h

    def count_keys(self) -> int:
        total = 0
        for _ in self.full_scan():
            total += 1
        return total

    def verify(self) -> List[str]:
        """Structural sanity checks; returns a list of problems (empty = ok).

        Verifies key ordering within nodes, separator invariants between
        children, sibling chains, and leaf reachability.
        """
        problems: List[str] = []
        if self.root is None:
            return problems

        leaves: List[int] = []

        def walk(
            pid: Optional[int],
            lo: Optional[Tuple[bytes, bytes]],
            hi: Optional[Tuple[bytes, bytes]],
            depth: int,
        ) -> int:
            if pid is None:
                problems.append("None child pointer")
                return 0
            if depth > 64:
                problems.append("tree deeper than 64 levels (cycle?)")
                return 0
            node = self._load(pid)
            if not node.is_leaf() and len(node.children) != len(node.keys) + 1:
                problems.append(
                    f"page {pid}: {len(node.keys)} keys but {len(node.children)} children"
                )
                return 0
            # Entries are (key, locator) pairs, so repeats of a key are legal
            # but a repeat of a whole entry is not.
            for a, b in zip(node.keys, node.keys[1:]):
                if not a < b:
                    problems.append(f"page {pid}: entries not strictly ascending")
                    break
            for key, locator in node.keys:
                if len(key) > MAX_KEY_LEN:
                    problems.append(f"page {pid}: key too long")
                if lo is not None and (key, locator) < lo:
                    problems.append(f"page {pid}: entry below lower bound")
                if hi is not None and not (key, locator) < hi:
                    problems.append(f"page {pid}: entry at/above upper bound")
            if node.is_leaf():
                leaves.append(pid)
                return len(node.keys)
            # child[0] has no separator on its left; child[i+1] is bounded
            # below by sep[i] and above by sep[i+1].
            count = walk(node.children[0], lo, node.keys[0] if node.keys else hi, depth + 1)
            for i, sep in enumerate(node.keys):
                count += walk(
                    node.children[i + 1],
                    sep,
                    node.keys[i + 1] if i + 1 < len(node.keys) else hi,
                    depth + 1,
                )
            return count

        walk(self.root, None, None, 0)
        # The leaves visited left-to-right must be exactly the right-sibling
        # chain range scans follow.
        for pid, nxt in zip(leaves, leaves[1:]):
            actual = self._load(pid).right_sibling
            if actual != nxt:
                problems.append(
                    f"page {pid}: leaf chain points to {actual}, expected {nxt}"
                )
        if leaves and self._load(leaves[-1]).right_sibling != 0:
            problems.append(f"page {leaves[-1]}: last leaf has a right sibling")
        return problems
