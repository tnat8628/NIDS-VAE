"""Contract tests for the global database-backed flow explorer endpoint."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.api import dashboard as dashboard_module
from backend.app.db.database import get_db


def _test_db() -> Generator[MagicMock, None, None]:
    yield MagicMock(spec=Session)


app = FastAPI()
app.include_router(dashboard_module.router)
app.dependency_overrides[get_db] = _test_db
client = TestClient(app)


def test_global_flows_contract_supports_prediction_filter(monkeypatch) -> None:
    upload_id = uuid.uuid4()
    run_id = uuid.uuid4()
    created_at = datetime(2026, 7, 2, 11, 0, tzinfo=UTC)
    storage = MagicMock(
        return_value={
            "status": "ok",
            "items": [
                {
                    "upload_id": upload_id,
                    "filename": "file-b.csv",
                    "run_id": run_id,
                    "row_index": 25,
                    "reconstruction_error": 7.12,
                    "prediction": 1,
                    "prediction_label": "anomaly",
                    "created_at": created_at,
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 25,
                "total_items": 1,
                "total_pages": 1,
                "has_previous": False,
                "has_next": False,
            },
        }
    )
    monkeypatch.setattr(dashboard_module, "list_global_flows", storage)

    response = client.get("/dashboard/flows?prediction=anomaly&page=1&page_size=25")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["upload_id"] == str(upload_id)
    assert payload["items"][0]["run_id"] == str(run_id)
    assert payload["items"][0]["prediction_label"] == "anomaly"
    storage.assert_called_once()
    assert storage.call_args.kwargs["prediction_filter"] == "anomaly"