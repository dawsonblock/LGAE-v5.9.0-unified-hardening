"""Crash-safe transactions via write-ahead log (Phase 30).

A WAL ensures that committed transactions survive crashes. The protocol:

  1. BEGIN: write a BEGIN record with transaction ID and state hash
  2. WRITE: write each mutation as a WAL record
  3. COMMIT: write a COMMIT record and fsync
  4. APPLY: apply the mutations to the authoritative state
  5. CHECKPOINT: truncate the WAL after a successful checkpoint

On recovery, the WAL is replayed:
  - If a transaction has COMMIT: re-apply it
  - If a transaction has no COMMIT: discard it (rollback)

This is the standard ARIES-style WAL protocol, simplified for the runtime's
single-writer model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class WALRecordType(str, Enum):
    BEGIN = "begin"
    WRITE = "write"
    COMMIT = "commit"
    ABORT = "abort"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class WALRecord:
    """One record in the write-ahead log."""
    txn_id: int
    record_type: WALRecordType
    lsn: int  # log sequence number
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def serialize(self) -> str:
        return json.dumps({
            "txn_id": int(self.txn_id),
            "record_type": self.record_type.value,
            "lsn": int(self.lsn),
            "payload": self.payload,
            "timestamp": float(self.timestamp),
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def deserialize(cls, line: str) -> "WALRecord":
        data = json.loads(line)
        return cls(
            txn_id=int(data["txn_id"]),
            record_type=WALRecordType(data["record_type"]),
            lsn=int(data["lsn"]),
            payload=data["payload"],
            timestamp=float(data["timestamp"]),
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "txn_id": int(self.txn_id),
            "record_type": self.record_type.value,
            "lsn": int(self.lsn),
            "payload": self.payload,
            "timestamp": float(self.timestamp),
        }


@dataclass(slots=True)
class WALTransaction:
    """An in-progress transaction."""
    txn_id: int
    records: list[WALRecord] = field(default_factory=list)
    committed: bool = False
    aborted: bool = False


class WriteAheadLog:
    """A crash-safe write-ahead log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lsn = 0
        self._next_txn_id = 0
        self._active_txns: dict[int, WALTransaction] = {}

    def _append(self, record: WALRecord) -> WALRecord:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.serialize() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def begin(self, metadata: dict[str, Any] | None = None) -> int:
        """Begin a new transaction. Returns the transaction ID."""
        txn_id = self._next_txn_id
        self._next_txn_id += 1
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.BEGIN, lsn=self._lsn,
            payload=dict(metadata or {}), timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id] = WALTransaction(txn_id=txn_id)
        return txn_id

    def write(self, txn_id: int, mutation: dict[str, Any]) -> WALRecord:
        """Write a mutation within a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.WRITE, lsn=self._lsn,
            payload=mutation, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        return record

    def commit(self, txn_id: int) -> WALRecord:
        """Commit a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.COMMIT, lsn=self._lsn,
            payload={}, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].committed = True
        del self._active_txns[txn_id]
        return record

    def abort(self, txn_id: int) -> WALRecord:
        """Abort a transaction (rollback)."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.ABORT, lsn=self._lsn,
            payload={}, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].aborted = True
        del self._active_txns[txn_id]
        return record

    def checkpoint(self) -> WALRecord:
        """Write a checkpoint record and truncate the log."""
        self._lsn += 1
        record = WALRecord(
            txn_id=-1, record_type=WALRecordType.CHECKPOINT, lsn=self._lsn,
            payload={"active_txns": list(self._active_txns.keys())},
            timestamp=time.time(),
        )
        self._append(record)
        return record

    def truncate(self) -> None:
        """Truncate the WAL (after a successful checkpoint)."""
        self.path.write_text("")

    def iter_records(self) -> Iterator[WALRecord]:
        """Iterate over all records in the log."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield WALRecord.deserialize(line)


def recover_transactions(records: list[WALRecord]) -> dict[int, list[dict[str, Any]]]:
    """Recover committed transactions from WAL records.

    Returns a dict of {txn_id: [mutations]} for committed transactions only.
    Transactions without a COMMIT record are discarded (rollback).
    """
    txns: dict[int, list[dict[str, Any]]] = {}
    committed: set[int] = set()
    aborted: set[int] = set()
    for record in records:
        if record.record_type == WALRecordType.BEGIN:
            txns[record.txn_id] = []
        elif record.record_type == WALRecordType.WRITE:
            if record.txn_id in txns:
                txns[record.txn_id].append(record.payload)
        elif record.record_type == WALRecordType.COMMIT:
            committed.add(record.txn_id)
        elif record.record_type == WALRecordType.ABORT:
            aborted.add(record.txn_id)
    # Return only committed, non-aborted transactions.
    return {
        txn_id: mutations
        for txn_id, mutations in txns.items()
        if txn_id in committed and txn_id not in aborted
    }
