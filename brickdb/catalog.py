"""Catalog: metadata for tables and indexes, with crash-safe persistence.

The catalog is the database's own database. It tracks, per table, the column
schema and the list of heap page ids; per index, the key columns, uniqueness
flag and the B+ tree root page id.

Persistence uses **shadowed dual slots**: two reserved pages (ids 1 and 2)
each hold a JSON document with a SHA-256 checksum and a monotonically
increasing epoch. Writes alternate between the slots; the copy with the
highest *valid* epoch wins at load time. A crash mid-write therefore always
leaves at least one intact slot — the classic technique to get atomic-ish
metadata updates on hardware without atomic rename.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .errors import CatalogError, CorruptError
from .storage.disk import PAGE_SIZE
from .storage.pager import Pager

CATALOG_PAGE_A = 1
CATALOG_PAGE_B = 2

VALID_KINDS = {"INT", "FLOAT", "TEXT", "DATE"}


class ColumnDef:
    __slots__ = ("name", "kind", "notnull", "default")

    def __init__(self, name: str, kind: str, notnull: bool = False, default: Any = None):
        if kind not in VALID_KINDS:
            raise CatalogError(f"unsupported column type {kind!r}")
        self.name = name
        self.kind = kind
        self.notnull = notnull
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "kind": self.kind, "notnull": self.notnull}
        if self.default is not None:
            d["default"] = _dump_default(self.default)
            d["default_kind"] = type(self.default).__name__
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnDef":
        default = _load_default(d.get("default"), d.get("default_kind"))
        return cls(d["name"], d["kind"], bool(d.get("notnull", False)), default)

    def __repr__(self) -> str:
        return f"ColumnDef({self.name!r}, {self.kind}, notnull={self.notnull})"


def _dump_default(v: Any) -> Any:
    from datetime import date

    if isinstance(v, date):
        return v.isoformat()
    return v


def _load_default(v: Any, kind_name: Optional[str]) -> Any:
    if v is None:
        return None
    if kind_name == "date":
        from datetime import date

        return date.fromisoformat(v)
    return v


class TableDef:
    __slots__ = ("name", "columns", "heap_pages")

    def __init__(self, name: str, columns: List[ColumnDef], heap_pages: Optional[List[int]] = None):
        self.name = name
        self.columns = columns
        self.heap_pages: List[int] = heap_pages if heap_pages is not None else []

    @property
    def ncols(self) -> int:
        return len(self.columns)

    def column_index(self, name: str) -> int:
        lname = name.lower()
        for i, col in enumerate(self.columns):
            if col.name.lower() == lname:
                return i
        raise CatalogError(f"column {name!r} not found in table {self.name!r}")

    def has_column(self, name: str) -> bool:
        try:
            self.column_index(name)
            return True
        except CatalogError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "heap_pages": list(self.heap_pages),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableDef":
        return cls(
            d["name"],
            [ColumnDef.from_dict(c) for c in d["columns"]],
            list(d.get("heap_pages", [])),
        )


class IndexDef:
    __slots__ = ("name", "table", "columns", "unique", "root")

    def __init__(self, name: str, table: str, columns: List[str], unique: bool, root: Optional[int]):
        self.name = name
        self.table = table
        self.columns = columns  # column names, in key order
        self.unique = unique
        self.root = root  # B+ tree root page id (None = empty tree)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "columns": list(self.columns),
            "unique": self.unique,
            "root": self.root,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndexDef":
        return cls(d["name"], d["table"], list(d["columns"]), bool(d["unique"]), d["root"])


class Catalog:
    """In-memory metadata cache with dual-slot checksummed persistence."""

    def __init__(self, pager: Pager):
        self.pager = pager
        self.tables: Dict[str, TableDef] = {}
        self.indexes: Dict[str, IndexDef] = {}
        self._epoch = 0

    # ------------------------------------------------------------ lookup

    def _key(self, name: str) -> str:
        return name.lower()

    def get_table(self, name: str) -> TableDef:
        t = self.tables.get(self._key(name))
        if t is None:
            raise CatalogError(f"no such table: {name!r}")
        return t

    def get_index(self, name: str) -> IndexDef:
        ix = self.indexes.get(self._key(name))
        if ix is None:
            raise CatalogError(f"no such index: {name!r}")
        return ix

    def table_indexes(self, table_name: str) -> List[IndexDef]:
        key = self._key(table_name)
        return [ix for ix in self.indexes.values() if ix.table == key or ix.table.lower() == key]

    def unique_indexes(self, table_name: str) -> List[IndexDef]:
        return [ix for ix in self.table_indexes(table_name) if ix.unique]

    def all_indexes(self, table_name: str) -> List[IndexDef]:
        return self.table_indexes(table_name)

    # ---------------------------------------------------------- mutation

    def add_table(self, tdef: TableDef) -> None:
        key = self._key(tdef.name)
        if key in self.tables:
            raise CatalogError(f"table {tdef.name!r} already exists")
        self.tables[key] = tdef

    def drop_table(self, name: str) -> TableDef:
        key = self._key(name)
        if key not in self.tables:
            raise CatalogError(f"no such table: {name!r}")
        return self.tables.pop(key)

    def add_index(self, idef: IndexDef) -> None:
        key = self._key(idef.name)
        if key in self.indexes:
            raise CatalogError(f"index {idef.name!r} already exists")
        self.indexes[key] = idef

    def drop_index(self, name: str) -> IndexDef:
        key = self._key(name)
        if key not in self.indexes:
            raise CatalogError(f"no such index: {name!r}")
        return self.indexes.pop(key)

    # ---------------------------------------------------- statement undo

    def snapshot(self):
        """Deep copy of the metadata, for statement-level rollback.

        Staged page images are discarded by the pager on abort, so the
        in-memory catalog must be rewound too — otherwise a failed CREATE
        INDEX would leave a phantom index (or worse, a stale B+ tree root
        pointing at pages that were never published).
        """
        return (
            {k: TableDef.from_dict(v.to_dict()) for k, v in self.tables.items()},
            {k: IndexDef.from_dict(v.to_dict()) for k, v in self.indexes.items()},
            self._epoch,
        )

    def restore(self, snap) -> None:
        tables, indexes, epoch = snap
        self.tables = tables
        self.indexes = indexes
        self._epoch = epoch

    # ------------------------------------------------------- persistence

    def _serialize(self) -> bytes:
        doc = {
            "epoch": self._epoch,
            "tables": [t.to_dict() for t in self.tables.values()],
            "indexes": [i.to_dict() for i in self.indexes.values()],
        }
        raw = json.dumps(doc, sort_keys=True).encode("utf-8")
        if len(raw) > PAGE_SIZE - 40:
            raise CorruptError("catalog document too large (tables/indexes limit)")
        return raw

    def _slot_image(self, raw: bytes, epoch: int) -> bytes:
        payload = {"epoch": epoch, "sha256": hashlib.sha256(raw).hexdigest(), "len": len(raw)}
        head = json.dumps(payload, sort_keys=True).encode("utf-8")
        image = head + b"\x00" + raw
        return image.ljust(PAGE_SIZE, b"\x00")

    def save(self, pager: Pager) -> None:
        """Stage the catalog document into the next shadow slot.

        The image goes through the caller's transaction pager so catalog
        updates commit atomically together with the statement's other pages
        (WAL record first, then publish). ``self.pager`` is used read-only
        by :meth:`load`.
        """
        raw = self._serialize()
        self._epoch += 1
        slot = CATALOG_PAGE_A if self._epoch % 2 else CATALOG_PAGE_B
        pager.write(slot, self._slot_image(raw, self._epoch))

    def load(self) -> None:
        """Load from whichever slot has the highest valid epoch."""
        best: Optional[Tuple[int, Dict[str, Any]]] = None
        for pid in (CATALOG_PAGE_A, CATALOG_PAGE_B):
            try:
                image = self.pager.read(pid)
            except Exception:
                continue
            doc = self._parse_slot(image)
            if doc is None:
                continue
            if best is None or doc["epoch"] > best[0]:
                best = (doc["epoch"], doc)
        if best is None:
            return  # fresh database
        doc = best[1]
        self._epoch = doc["epoch"]
        self.tables = {}
        self.indexes = {}
        for t in doc["tables"]:
            tdef = TableDef.from_dict(t)
            self.tables[self._key(tdef.name)] = tdef
        for i in doc["indexes"]:
            idef = IndexDef.from_dict(i)
            self.indexes[self._key(idef.name)] = idef

    def _parse_slot(self, image: bytes) -> Optional[Dict[str, Any]]:
        try:
            sep = image.index(b"\x00")
            head = json.loads(image[:sep].decode("utf-8"))
            raw = image[sep + 1 : sep + 1 + head["len"]]
            if hashlib.sha256(raw).hexdigest() != head["sha256"]:
                return None
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    # -------------------------------------------------------------- info

    def summary(self) -> List[Tuple[str, str, int]]:
        return [
            (t.name, ", ".join(f"{c.name}:{c.kind}" for c in t.columns), len(t.heap_pages))
            for t in self.tables.values()
        ]
