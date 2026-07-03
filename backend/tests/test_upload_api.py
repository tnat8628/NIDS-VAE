"""Focused contract tests for POST /upload."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.api import upload as upload_module
from backend.app.db.database import get_db
from backend.app.services.prediction_storage import UploadNotFoundError
from backend.app.services.upload_storage import UploadStorageError


def _test_db() -> Generator[MagicMock, None, None]:
    yield MagicMock(spec=Session)


app = FastAPI()
app.include_router(upload_module.router)
app.dependency_overrides[get_db] = _test_db
client = TestClient(app)


def test_upload_returns_upload_id_and_keeps_existing_fields(monkeypatch) -> None:
    upload_id = uuid.uuid4()
    storage = MagicMock(return_value=upload_id)
    monkeypatch.setattr(upload_module, "save_csv_upload", storage)

    response = client.post(
        "/upload",
        files={"file": ("one-row.csv", b"name,value\nflow-1,10\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "upload_id": str(upload_id),
        "filename": "one-row.csv",
        "row_count": 1,
        "col_count": 2,
        "message": "File và toàn bộ các dòng dữ liệu đã được lưu vào database.",
    }
    storage.assert_called_once()
    assert storage.call_args.kwargs["contents"] == b"name,value\nflow-1,10\n"


def test_upload_rejects_empty_csv_without_calling_storage(monkeypatch) -> None:
    storage = MagicMock()
    monkeypatch.setattr(upload_module, "save_csv_upload", storage)

    response = client.post(
        "/upload",
        files={"file": ("empty.csv", b"name,value\n", "text/csv")},
    )

    assert response.status_code == 400
    storage.assert_not_called()


def test_upload_rejects_non_csv_without_calling_storage(monkeypatch) -> None:
    storage = MagicMock()
    monkeypatch.setattr(upload_module, "save_csv_upload", storage)

    response = client.post(
        "/upload",
        files={"file": ("flows.txt", b"name,value\nflow-1,10\n", "text/plain")},
    )

    assert response.status_code == 400
    storage.assert_not_called()


def test_upload_returns_503_when_atomic_storage_fails(monkeypatch) -> None:
    storage = MagicMock(side_effect=UploadStorageError("database failed"))
    monkeypatch.setattr(upload_module, "save_csv_upload", storage)

    response = client.post(
        "/upload",
        files={"file": ("flows.csv", b"name,value\nflow-1,10\n", "text/csv")},
    )

    assert response.status_code == 503


def test_list_uploads_contract_supports_filter_and_pagination(monkeypatch) -> None:
    upload_id = uuid.uuid4()
    run_id = uuid.uuid4()
    created_at = datetime(2026, 7, 2, 10, 0, tzinfo=UTC)
    storage = MagicMock(
        return_value={
            "status": "ok",
            "items": [
                {
                    "upload_id": upload_id,
                    "filename": "sample.csv",
                    "row_count": 500,
                    "col_count": 66,
                    "created_at": created_at,
                    "analysis_status": "completed",
                    "latest_run_id": run_id,
                    "latest_predicted_at": created_at,
                    "anomaly_count": 31,
                    "normal_count": 469,
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_items": 1,
                "total_pages": 1,
                "has_previous": False,
                "has_next": False,
            },
        }
    )
    monkeypatch.setattr(upload_module, "list_uploads", storage)

    response = client.get("/uploads?filter=analyzed&page=1&page_size=20")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["upload_id"] == str(upload_id)
    assert payload["items"][0]["latest_run_id"] == str(run_id)
    assert payload["items"][0]["analysis_status"] == "completed"
    storage.assert_called_once()
    assert storage.call_args.kwargs["upload_filter"] == "analyzed"


def test_delete_upload_returns_json_message(monkeypatch) -> None:
    upload_id = uuid.uuid4()
    storage = MagicMock()
    monkeypatch.setattr(upload_module, "delete_upload", storage)

    response = client.delete(f"/uploads/{upload_id}")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "upload_id": str(upload_id),
        "message": "ÄÃ£ xÃ³a file vÃ  toÃ n bá»™ dá»¯ liá»‡u liÃªn quan.",
    }
    storage.assert_called_once()


def test_delete_upload_returns_404_for_missing_upload(monkeypatch) -> None:
    upload_id = uuid.uuid4()
    storage = MagicMock(side_effect=UploadNotFoundError("missing"))
    monkeypatch.setattr(upload_module, "delete_upload", storage)

    response = client.delete(f"/uploads/{upload_id}")

    assert response.status_code == 404