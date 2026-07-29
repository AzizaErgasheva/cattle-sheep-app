from app.domain.entities import HistoryEntry
from app.infrastructure.history_repository import SqliteHistoryRepository


def _entry(id_: str, created_at: str) -> HistoryEntry:
    return HistoryEntry(
        id=id_,
        created_at=created_at,
        model_name="resnet50",
        label="cow",
        confidence=0.9,
        probabilities={"cow": 0.9, "sheep": 0.1},
        thumbnail_data_url="data:image/jpeg;base64,ZmFrZQ==",
    )


def test_add_and_list_recent_orders_newest_first(tmp_path):
    repo = SqliteHistoryRepository(db_path=tmp_path / "history.db")

    repo.add(_entry("1", "2026-01-01T00:00:00+00:00"))
    repo.add(_entry("2", "2026-01-02T00:00:00+00:00"))

    result = repo.list_recent(limit=10)

    assert [e.id for e in result] == ["2", "1"]
    assert result[0].probabilities == {"cow": 0.9, "sheep": 0.1}


def test_list_recent_respects_limit(tmp_path):
    repo = SqliteHistoryRepository(db_path=tmp_path / "history.db")
    for i in range(5):
        repo.add(_entry(str(i), f"2026-01-0{i+1}T00:00:00+00:00"))

    result = repo.list_recent(limit=2)

    assert len(result) == 2


def test_clear_removes_all_entries(tmp_path):
    repo = SqliteHistoryRepository(db_path=tmp_path / "history.db")
    repo.add(_entry("1", "2026-01-01T00:00:00+00:00"))

    repo.clear()

    assert repo.list_recent(limit=10) == []
