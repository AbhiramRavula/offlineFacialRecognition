"""
NHAI FaceGuard SDK — Local Face Database
SQLite-backed store for face embeddings. Zero network, zero cloud.
Thread-safe for concurrent API access.
"""

import sqlite3
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.config import DB_PATH, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# SQL Schema (Updated with Attendance Logging table for Sync & Purge)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT    UNIQUE NOT NULL,
    person_name   TEXT    NOT NULL,
    embedding     BLOB    NOT NULL,
    registered_at TEXT    DEFAULT (datetime('now', 'localtime')),
    metadata      TEXT    DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_person_id ON faces(person_id);

CREATE TABLE IF NOT EXISTS attendance_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT    NOT NULL,
    timestamp     TEXT    DEFAULT (datetime('now', 'localtime')),
    liveness_score REAL    NOT NULL,
    verification_mode TEXT NOT NULL,
    is_synced     INTEGER DEFAULT 0
);
"""


def _emb_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class FaceDatabase:
    """
    Thread-safe SQLite database for storing and searching face embeddings.
    Contains support for local attendance transaction logging to facilitate
    offline-to-online Sync & Purge protocols.

    Usage:
        db = FaceDatabase()
        db.register("emp001", "Arjun Sharma", embedding)
        name, dist = db.identify(query_embedding)
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._cache: Dict[str, Tuple[str, np.ndarray]] = {}  # person_id → (name, emb)
        self._load_cache()
        logger.info(
            "FaceDatabase ready at '%s' (%d enrolled faces).",
            db_path, len(self._cache)
        )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_cache(self) -> None:
        """Load all embeddings into memory for fast cosine search."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_id, person_name, embedding FROM faces"
            ).fetchall()
        self._cache = {
            row["person_id"]: (row["person_name"], _blob_to_emb(row["embedding"]))
            for row in rows
        }

    # ── CRUD Operations ────────────────────────────────────────────────────

    def register(
        self,
        person_id: str,
        person_name: str,
        embedding: np.ndarray,
        metadata: str = "{}",
        overwrite: bool = False,
    ) -> bool:
        """
        Register a new face in the database.

        Args:
            person_id: Unique identifier (e.g., employee ID).
            person_name: Display name.
            embedding: 512-dim L2-normalized embedding.
            metadata: Optional JSON string for extra fields.
            overwrite: If True, update existing record.

        Returns:
            True on success, False on failure.
        """
        blob = _emb_to_blob(embedding)
        with self._lock:
            try:
                with self._connect() as conn:
                    if overwrite:
                        conn.execute(
                            """INSERT INTO faces (person_id, person_name, embedding, metadata)
                               VALUES (?, ?, ?, ?)
                               ON CONFLICT(person_id) DO UPDATE SET
                                 person_name=excluded.person_name,
                                 embedding=excluded.embedding,
                                 metadata=excluded.metadata,
                                 registered_at=datetime('now','localtime')""",
                            (person_id, person_name, blob, metadata),
                        )
                    else:
                        conn.execute(
                            """INSERT INTO faces (person_id, person_name, embedding, metadata)
                               VALUES (?, ?, ?, ?)""",
                            (person_id, person_name, blob, metadata),
                        )
                # Update in-memory cache
                self._cache[person_id] = (person_name, embedding)
                logger.info("Registered face: %s (%s)", person_id, person_name)
                return True
            except sqlite3.IntegrityError:
                logger.warning(
                    "Person ID '%s' already exists. Use overwrite=True to update.",
                    person_id,
                )
                return False
            except Exception as e:
                logger.error("Registration failed for %s: %s", person_id, e)
                return False

    def delete(self, person_id: str) -> bool:
        """Remove a face from the database."""
        with self._lock:
            with self._connect() as conn:
                result = conn.execute(
                    "DELETE FROM faces WHERE person_id = ?", (person_id,)
                )
            if result.rowcount > 0:
                self._cache.pop(person_id, None)
                logger.info("Deleted face: %s", person_id)
                return True
            logger.warning("Person ID '%s' not found for deletion.", person_id)
            return False

    def get(self, person_id: str) -> Optional[Dict]:
        """Fetch a single person's record."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT person_id, person_name, registered_at, metadata FROM faces WHERE person_id=?",
                (person_id,),
            ).fetchone()
        if row:
            return dict(row)
        return None

    def list_all(self) -> List[Dict]:
        """List all enrolled persons (without embeddings)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_id, person_name, registered_at, metadata FROM faces ORDER BY person_name"
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Return the number of enrolled faces."""
        return len(self._cache)

    # ── Search ──────────────────────────────────────────────────────────────

    def identify(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.35,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the closest matching face(s) in the database.

        Uses cosine distance (lower = more similar) against the in-memory cache.

        Args:
            query_embedding: 512-dim L2-normalized query embedding.
            top_k: Return up to this many matches.
            threshold: Maximum cosine distance to accept (0.35 recommended).

        Returns:
            List of (person_id, person_name, distance) tuples, sorted by distance asc.
            Empty list if no match within threshold.
        """
        if not self._cache:
            return []

        scores = []
        for person_id, (name, stored_emb) in self._cache.items():
            dist = float(1.0 - np.dot(query_embedding, stored_emb))
            scores.append((person_id, name, dist))

        scores.sort(key=lambda x: x[2])
        filtered = [(pid, name, d) for pid, name, d in scores if d <= threshold]
        return filtered[:top_k]

    def verify(
        self,
        person_id: str,
        query_embedding: np.ndarray,
        threshold: float = 0.35,
    ) -> Tuple[bool, float]:
        """
        1:1 Verification — check if query matches a specific person.

        Args:
            person_id: The person to verify against.
            query_embedding: Query face embedding.
            threshold: Maximum cosine distance to accept.

        Returns:
            (is_match: bool, distance: float)
        """
        record = self._cache.get(person_id)
        if record is None:
            logger.warning("Person ID '%s' not found in database.", person_id)
            return False, float("inf")

        _, stored_emb = record
        dist = float(1.0 - np.dot(query_embedding, stored_emb))
        return dist <= threshold, dist

    # ── Sync & Purge Local Logging Operations ────────────────────────────────

    def log_attendance(
        self, person_id: str, liveness_score: float, verification_mode: str
    ) -> int:
        """Log a local verification session event (for offline queuing)."""
        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute(
                        """INSERT INTO attendance_logs (person_id, liveness_score, verification_mode)
                           VALUES (?, ?, ?)""",
                        (person_id, liveness_score, verification_mode),
                    )
                    log_id = cursor.lastrowid
                logger.info("Logged attendance locally for '%s' (Row ID: %d)", person_id, log_id)
                return log_id
            except Exception as e:
                logger.error("Failed to log local attendance: %s", e)
                return -1

    def get_unsynced_logs(self) -> List[Dict]:
        """Fetch all offline verification logs waiting for AWS synchronization."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, person_id, timestamp, liveness_score, verification_mode FROM attendance_logs WHERE is_synced = 0"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_logs_as_synced(self, log_ids: List[int]) -> bool:
        """Mark local logs as successfully uploaded to the cloud."""
        if not log_ids:
            return True
        with self._lock:
            try:
                with self._connect() as conn:
                    placeholders = ",".join("?" for _ in log_ids)
                    conn.execute(
                        f"UPDATE attendance_logs SET is_synced = 1 WHERE id IN ({placeholders})",
                        tuple(log_ids)
                    )
                logger.info("Marked %d logs as successfully synchronized.", len(log_ids))
                return True
            except Exception as e:
                logger.error("Failed to mark sync state: %s", e)
                return False

    def purge_synced_logs(self) -> int:
        """Purge synced records from local SQLite to release device memory."""
        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute("DELETE FROM attendance_logs WHERE is_synced = 1")
                    deleted_count = cursor.rowcount
                logger.info("Purged %d transaction logs from local database.", deleted_count)
                return deleted_count
            except Exception as e:
                logger.error("Failed to purge synced logs: %s", e)
                return 0
