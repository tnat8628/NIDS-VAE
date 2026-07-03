"""Transactional persistence for validated CSV uploads."""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import insert
from sqlalchemy.orm import Session

from backend.app.db.models import CsvRow, CsvUpload

ROW_INSERT_BATCH_SIZE = 1_000


class UploadStorageError(RuntimeError):
    """Raised when a complete CSV upload cannot be persisted."""


def _to_json_value(value: Any) -> Any:
    """Convert pandas/numpy values to values accepted by PostgreSQL JSONB."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return None if not value.is_finite() else str(value)

    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        try:
            return _to_json_value(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        if value.strip().lower() in {
            "nan",
            "nat",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            return None
        return value

    if isinstance(value, (int, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def dataframe_rows(dataframe: pd.DataFrame):
    """Yield zero-based row indexes and JSON-safe payloads without dropping duplicates."""
    columns = [str(column) for column in dataframe.columns]
    for row_index, values in enumerate(dataframe.itertuples(index=False, name=None)):
        yield row_index, {
            column: _to_json_value(value)
            for column, value in zip(columns, values, strict=True)
        }


def save_csv_upload(
    session: Session,
    *,
    original_filename: str,
    contents: bytes,
    dataframe: pd.DataFrame,
) -> uuid.UUID:
    """Persist upload metadata and every CSV row atomically."""
    upload = CsvUpload(
        original_filename=original_filename,
        row_count=len(dataframe),
        col_count=len(dataframe.columns),
        columns=[str(column) for column in dataframe.columns],
        file_sha256=hashlib.sha256(contents).hexdigest(),
    )

    try:
        with session.begin():
            session.add(upload)
            session.flush()

            batch: list[dict[str, object]] = []
            for row_index, payload in dataframe_rows(dataframe):
                batch.append(
                    {
                        "id": uuid.uuid4(),
                        "upload_id": upload.id,
                        "row_index": row_index,
                        "payload": payload,
                    }
                )
                if len(batch) >= ROW_INSERT_BATCH_SIZE:
                    session.execute(insert(CsvRow), batch)
                    batch.clear()

            if batch:
                session.execute(insert(CsvRow), batch)
    except Exception as exc:
        raise UploadStorageError("Không thể lưu trọn vẹn file CSV vào database.") from exc

    return upload.id