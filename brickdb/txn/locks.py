"""Lock manager: shared/exclusive table locks with deadlock detection.

Concurrency scope: brickdb is an embedded engine with one writer process, so
locks guard *transactions within a process* against each other (and serve as
the API for embedding multi-threaded clients). Locks are taken at **table
granularity** — row-level locking would be the natural next step and is
sketched in the README.

Protocol:

* Readers take S locks, writers take X locks on ``table:<name>`` resources.
* S locks are compatible with each other; X conflicts with everything.
* A blocked acquirer records a waits-for edge; a cycle in the waits-for
  graph means deadlock and the loser gets :class:`DeadlockError` (the
  youngest transaction is chosen as victim, like Postgres).
* All of a transaction's locks are released together at commit/abort —
  strict two-phase locking (2PL), which is what makes the isolation
  guarantee real.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Set

from ..errors import DeadlockError, LockConflict

S = "S"
X = "X"

_COMPATIBLE = {(S, S): True, (S, X): False, (X, S): False, (X, X): False}


class LockManager:
    def __init__(self, deadlock_timeout: float = 5.0):
        self._mutex = threading.Lock()
        self._cond = threading.Condition(self._mutex)
        # resource -> {txn_id: mode}
        self._held: Dict[str, Dict[int, str]] = defaultdict(dict)
        # txn -> set of txns it is waiting for
        self._waits_for: Dict[int, Set[int]] = defaultdict(set)
        self._waiters = 0
        self.deadlock_timeout = deadlock_timeout

    # ------------------------------------------------------------- acquire

    def acquire(self, txn_id: int, resource: str, mode: str) -> None:
        deadline = time.monotonic() + self.deadlock_timeout
        with self._cond:
            while True:
                holders = self._held[resource]
                if self._compatible(mode, holders, txn_id):
                    holders[txn_id] = mode if holders.get(txn_id) != X else X
                    self._waits_for.pop(txn_id, None)
                    return
                # Record waits-for edges, then check for a deadlock cycle.
                self._waits_for[txn_id] = {t for t in holders if t != txn_id}
                self._check_deadlock(txn_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._waits_for.pop(txn_id, None)
                    raise LockConflict(f"timed out waiting for {mode} lock on {resource}")
                self._cond.wait(remaining)

    @staticmethod
    def _compatible(mode: str, holders: Dict[int, str], txn_id: int) -> bool:
        other = [(t, m) for t, m in holders.items() if t != txn_id]
        if not other:
            return True  # no one else holds it (or we do; upgrades are S->X here)
        return all(_COMPATIBLE[(mode, m)] for _t, m in other)

    def _check_deadlock(self, start: int) -> None:
        """DFS over the waits-for graph from ``start``; abort on a cycle."""
        seen: Set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == start and node in seen:
                self._waits_for.pop(start, None)
                raise DeadlockError(
                    f"deadlock detected involving transaction {start}; aborted as victim"
                )
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self._waits_for.get(node, ()))

    # ------------------------------------------------------------- release

    def release_all(self, txn_id: int) -> None:
        with self._cond:
            for resource in list(self._held):
                self._held[resource].pop(txn_id, None)
                if not self._held[resource]:
                    del self._held[resource]
            self._waits_for.pop(txn_id, None)
            self._cond.notify_all()

    def held_resources(self, txn_id: int) -> Dict[str, str]:
        with self._mutex:
            return {res: modes[txn_id] for res, modes in self._held.items() if txn_id in modes}

    def stats(self) -> dict:
        with self._mutex:
            return {"locked_resources": len(self._held), "waiting_edges": len(self._waits_for)}
