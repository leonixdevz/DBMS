"""Exception hierarchy for brickdb.

Keeping every error under one base class lets the CLI catch *any* engine
failure and report it without crashing the REPL loop.
"""


class BrickDBError(Exception):
    """Base class for all brickdb errors."""


class SQLError(BrickDBError):
    """Raised for syntax errors and semantic errors in SQL statements."""


class TypeViolation(SQLError):
    """A value does not match its column's declared type."""


class NotNullViolation(SQLError):
    """A NOT NULL (or PRIMARY KEY) column received NULL."""


class UniqueViolation(SQLError):
    """A unique/primary-key index received a duplicate key."""


class CatalogError(BrickDBError):
    """Raised for missing tables/columns and catalog corruption."""


class PageFull(BrickDBError):
    """A slotted page has no room for the requested payload."""


class CorruptError(BrickDBError):
    """On-disk structures failed validation."""


class LockError(BrickDBError):
    """Base class for lock manager failures."""


class LockConflict(LockError):
    """A lock could not be granted without waiting."""


class DeadlockError(LockError):
    """The waits-for graph contains a cycle involving this transaction."""
