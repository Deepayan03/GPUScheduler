"""
sqliteStore.py

SQLite-backed persistence for scheduler state and recovery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

from gpuscheduler.daemon.job import Job


class SQLiteJobStore:
    def __init__(self, dbPath: Path | str = "state/jobs.db"):
        self.dbPath = Path(dbPath)
        self.dbPath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.dbPath),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._initSchema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _initSchema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status_updated
                ON jobs(status, updated_at DESC)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daemon_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def upsertJobs(self, jobs: Iterable[Job]) -> None:
        now = time.time()
        rows = []
        for job in jobs:
            rows.append(
                (
                    job.id,
                    job.status.value,
                    json.dumps(job.toDict(), sort_keys=True),
                    now,
                )
            )

        if not rows:
            return

        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO jobs(id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
            self._conn.commit()

    def listJobs(
        self,
        statuses: Optional[List[str]] = None,
    ) -> List[Job]:
        with self._lock:
            cur = self._conn.cursor()
            if statuses:
                placeholders = ",".join(["?"] * len(statuses))
                query = (
                    "SELECT payload_json FROM jobs "
                    f"WHERE status IN ({placeholders}) "
                    "ORDER BY updated_at ASC"
                )
                cur.execute(query, statuses)
            else:
                cur.execute(
                    "SELECT payload_json FROM jobs ORDER BY updated_at ASC"
                )
            rows = cur.fetchall()

        result: List[Job] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
                result.append(Job.fromDict(payload))
            except Exception:
                continue
        return result

    def setDaemonState(self, key: str, value) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO daemon_state(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), now),
            )
            self._conn.commit()

    def getDaemonState(self, key: str, default=None):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT value_json FROM daemon_state WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

