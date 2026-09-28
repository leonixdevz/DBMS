"""brickdb — a tiny relational DBMS built from scratch.

Layers (bottom-up):

    storage/   disk manager, value codec, slotted pages, heap files
    buffer.py  buffer pool with pin/unpin, LRU eviction, dirty tracking
    index/     on-disk B+ tree indexes
    catalog.py table/index metadata with checksummed dual-slot persistence
    sql/       lexer, AST, recursive-descent parser
    executors/ volcano-style iterator operators + expression evaluator
    txn/       write-ahead log, lock manager, recovery
    engine.py  statement dispatch, planner, index maintenance
    cli.py     interactive REPL
"""

from .engine import Database
from .errors import BrickDBError

__version__ = "0.1.0"

__all__ = ["Database", "BrickDBError", "__version__"]
