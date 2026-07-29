"""SQLite-backed HistoryRepositoryPort. Uses the stdlib sqlite3 module only --
no extra ORM dependency needed for what's essentially one small table."""
import json
import sqlite3
from pathlib import Path

from app.domain.entities import HistoryEntry
from app.domain.ports import HistoryRepositoryPort

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    thumbnail_data_url TEXT NOT NULL
)
"""


class SqliteHistoryRepository(HistoryRepositoryPort):
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def add(self, entry: HistoryEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history "
                "(id, created_at, model_name, label, confidence, probabilities, thumbnail_data_url) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.created_at,
                    entry.model_name,
                    entry.label,
                    entry.confidence,
                    json.dumps(entry.probabilities),
                    entry.thumbnail_data_url,
                ),
            )

    def list_recent(self, limit: int) -> list[HistoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, model_name, label, confidence, probabilities, thumbnail_data_url "
                "FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            HistoryEntry(
                id=row[0],
                created_at=row[1],
                model_name=row[2],
                label=row[3],
                confidence=row[4],
                probabilities=json.loads(row[5]),
                thumbnail_data_url=row[6],
            )
            for row in rows
        ]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
