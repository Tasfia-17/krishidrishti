"""
KrishiDrishti SQLite Database
Stores diagnosis history with full JSON serialization of DiagnosisRecord.
Uses aiosqlite for async access — keeps the FastAPI server non-blocking.

Schema: one table (diagnoses) with JSON-serialized diagnosis data.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from config import settings
from storage.models import DiagnosisRecord, DiagnosisHistoryEntry

logger = logging.getLogger("krishidrishti.storage.database")

# ── Schema ────────────────────────────────────────────────────────────────────
CREATE_DIAGNOSES_TABLE = """
CREATE TABLE IF NOT EXISTS diagnoses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    image_path      TEXT,
    farmer_note     TEXT,
    analysis_mode   TEXT    NOT NULL DEFAULT 'balanced',
    crop_name       TEXT    NOT NULL,
    disease_detected TEXT   NOT NULL,
    severity        TEXT    NOT NULL,
    urgency         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    is_healthy      INTEGER NOT NULL DEFAULT 0,
    diagnosis_json  TEXT    NOT NULL
)
"""

CREATE_INDEX_TIMESTAMP = """
CREATE INDEX IF NOT EXISTS idx_diagnoses_timestamp ON diagnoses(timestamp DESC)
"""

CREATE_INDEX_CROP = """
CREATE INDEX IF NOT EXISTS idx_diagnoses_crop ON diagnoses(crop_name)
"""

CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS diagnoses_fts USING fts5(
    crop_name,
    disease_detected,
    symptoms,
    farmer_note,
    content='diagnoses',
    content_rowid='id'
)
"""

CREATE_FTS_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS diagnoses_ai AFTER INSERT ON diagnoses BEGIN
        INSERT INTO diagnoses_fts(rowid, crop_name, disease_detected, symptoms, farmer_note)
        VALUES (new.id, new.crop_name, new.disease_detected, '', new.farmer_note);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS diagnoses_ad AFTER DELETE ON diagnoses BEGIN
        INSERT INTO diagnoses_fts(diagnoses_fts, rowid, crop_name, disease_detected, symptoms, farmer_note)
        VALUES ('delete', old.id, old.crop_name, old.disease_detected, '', old.farmer_note);
    END
    """,
]


class DiagnosisDatabase:
    """
    Async SQLite database for KrishiDrishti diagnosis history.
    Usage:
        db = DiagnosisDatabase()
        await db.initialize()
        entry_id = await db.save_diagnosis(record, image_path, farmer_note, mode)
        history = await db.get_history(limit=20)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self._initialized = False

    async def initialize(self):
        """Create database file and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_DIAGNOSES_TABLE)
            await db.execute(CREATE_INDEX_TIMESTAMP)
            await db.execute(CREATE_INDEX_CROP)
            try:
                await db.execute(CREATE_FTS_TABLE)
                for trigger in CREATE_FTS_TRIGGERS:
                    await db.execute(trigger)
            except Exception as e:
                # FTS5 may not be available in all SQLite builds — non-critical
                logger.warning(f"FTS5 setup failed (search will use LIKE): {e}")
            await db.commit()
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    async def save_diagnosis(
        self,
        record: DiagnosisRecord,
        image_path: Optional[str] = None,
        farmer_note: Optional[str] = None,
        mode: str = "balanced",
    ) -> int:
        """
        Persist a diagnosis to the database.

        Args:
            record: DiagnosisRecord from the crop analyzer
            image_path: Optional path to the uploaded image
            farmer_note: Farmer's description (if any)
            mode: Analysis mode used (fast/balanced/accurate)

        Returns:
            Auto-incremented row ID.
        """
        timestamp = datetime.utcnow().isoformat()
        diagnosis_json = record.model_dump_json()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO diagnoses
                    (timestamp, image_path, farmer_note, analysis_mode,
                     crop_name, disease_detected, severity, urgency,
                     confidence, is_healthy, diagnosis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    image_path,
                    farmer_note,
                    mode,
                    record.crop_name,
                    record.disease_detected,
                    record.severity,
                    record.urgency,
                    record.confidence,
                    int(record.is_healthy),
                    diagnosis_json,
                ),
            )
            await db.commit()
            entry_id = cursor.lastrowid
            logger.info(
                f"Saved diagnosis #{entry_id}: {record.crop_name} — {record.disease_detected} "
                f"(confidence={record.confidence:.2f})"
            )
            return entry_id

    async def get_history(
        self,
        limit: int = 20,
        offset: int = 0,
        crop_filter: Optional[str] = None,
    ) -> List[DiagnosisHistoryEntry]:
        """
        Retrieve diagnosis history, newest first.

        Args:
            limit: Max entries to return
            offset: Pagination offset
            crop_filter: Optional crop name to filter by

        Returns:
            List of DiagnosisHistoryEntry objects.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if crop_filter:
                rows = await db.execute_fetchall(
                    """
                    SELECT * FROM diagnoses
                    WHERE crop_name LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (f"%{crop_filter}%", limit, offset),
                )
            else:
                rows = await db.execute_fetchall(
                    """
                    SELECT * FROM diagnoses
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )

        return [self._row_to_entry(row) for row in rows]

    async def get_by_id(self, entry_id: int) -> Optional[DiagnosisHistoryEntry]:
        """Retrieve a single diagnosis by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM diagnoses WHERE id = ?", (entry_id,)
            )
        if not rows:
            return None
        return self._row_to_entry(rows[0])

    async def search(self, query: str, limit: int = 10) -> List[DiagnosisHistoryEntry]:
        """
        Full-text search on crop name, disease, and farmer notes.
        Falls back to LIKE search if FTS5 is not available.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                # Try FTS5 first
                rows = await db.execute_fetchall(
                    """
                    SELECT d.* FROM diagnoses d
                    JOIN diagnoses_fts fts ON d.id = fts.rowid
                    WHERE diagnoses_fts MATCH ?
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                    """,
                    (query, limit),
                )
            except Exception:
                # Fallback to LIKE search
                like = f"%{query}%"
                rows = await db.execute_fetchall(
                    """
                    SELECT * FROM diagnoses
                    WHERE crop_name LIKE ? OR disease_detected LIKE ? OR farmer_note LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (like, like, like, limit),
                )

        return [self._row_to_entry(row) for row in rows]

    async def get_stats(self) -> dict:
        """Return aggregate statistics for the dashboard."""
        async with aiosqlite.connect(self.db_path) as db:
            total = await db.execute_fetchall("SELECT COUNT(*) FROM diagnoses")
            total_count = total[0][0] if total else 0

            healthy = await db.execute_fetchall(
                "SELECT COUNT(*) FROM diagnoses WHERE is_healthy = 1"
            )
            healthy_count = healthy[0][0] if healthy else 0

            top_crops = await db.execute_fetchall(
                """
                SELECT crop_name, COUNT(*) as cnt
                FROM diagnoses
                GROUP BY crop_name
                ORDER BY cnt DESC
                LIMIT 5
                """
            )

            top_diseases = await db.execute_fetchall(
                """
                SELECT disease_detected, COUNT(*) as cnt
                FROM diagnoses
                WHERE is_healthy = 0
                GROUP BY disease_detected
                ORDER BY cnt DESC
                LIMIT 5
                """
            )

        return {
            "total_diagnoses": total_count,
            "healthy_count": healthy_count,
            "diseased_count": total_count - healthy_count,
            "top_crops": [{"name": r[0], "count": r[1]} for r in top_crops],
            "top_diseases": [{"name": r[0], "count": r[1]} for r in top_diseases],
        }

    async def delete_diagnosis(self, entry_id: int) -> bool:
        """Delete a diagnosis record by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM diagnoses WHERE id = ?", (entry_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    def _row_to_entry(self, row) -> DiagnosisHistoryEntry:
        """Convert a database row to a DiagnosisHistoryEntry."""
        diagnosis_data = json.loads(row["diagnosis_json"])
        return DiagnosisHistoryEntry(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            image_path=row["image_path"],
            farmer_note=row["farmer_note"],
            analysis_mode=row["analysis_mode"],
            diagnosis=DiagnosisRecord(**diagnosis_data),
        )


# ── Module-level singleton (shared across the app) ────────────────────────────
_db: Optional[DiagnosisDatabase] = None


def get_database() -> DiagnosisDatabase:
    """Get the module-level database singleton."""
    global _db
    if _db is None:
        _db = DiagnosisDatabase()
    return _db
