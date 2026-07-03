"""Upload management and global flow explorer queries."""

from __future__ import annotations

import math
import uuid
from typing import Literal

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import CsvUpload, FlowPrediction
from backend.app.services.dashboard_storage import _latest_runs_per_upload
from backend.app.services.prediction_storage import UploadNotFoundError

UploadFilter = Literal["all", "analyzed", "pending"]
PredictionFilter = Literal["all", "anomaly", "normal"]


def _pagination(page: int, page_size: int, total_items: int) -> dict[str, int | bool]:
    total_pages = math.ceil(total_items / page_size) if total_items else 0
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
    }


def _apply_upload_filter(query: Select, latest_runs, upload_filter: UploadFilter) -> Select:
    if upload_filter == "analyzed":
        return query.where(latest_runs.c.run_id.is_not(None))
    if upload_filter == "pending":
        return query.where(latest_runs.c.run_id.is_(None))
    return query


def list_uploads(
    session: Session,
    *,
    page: int,
    page_size: int,
    upload_filter: UploadFilter = "all",
) -> dict[str, object]:
    """Return a database-paginated upload list with latest-run aggregates."""
    latest_runs = _latest_runs_per_upload()

    count_query = (
        select(func.count())
        .select_from(CsvUpload)
        .outerjoin(latest_runs, latest_runs.c.upload_id == CsvUpload.id)
    )
    count_query = _apply_upload_filter(count_query, latest_runs, upload_filter)
    total_items = int(session.scalar(count_query) or 0)

    rows_query = (
        select(
            CsvUpload.id.label("upload_id"),
            CsvUpload.original_filename.label("filename"),
            CsvUpload.row_count,
            CsvUpload.col_count,
            CsvUpload.created_at,
            latest_runs.c.run_id.label("latest_run_id"),
            latest_runs.c.predicted_at.label("latest_predicted_at"),
            latest_runs.c.anomaly_count,
            latest_runs.c.normal_count,
        )
        .outerjoin(latest_runs, latest_runs.c.upload_id == CsvUpload.id)
        .order_by(
            CsvUpload.row_count.desc(),
            CsvUpload.created_at.desc(),
            CsvUpload.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows_query = _apply_upload_filter(rows_query, latest_runs, upload_filter)

    items = []
    for row in session.execute(rows_query).mappings().all():
        latest_run_id = row["latest_run_id"]
        items.append(
            {
                "upload_id": row["upload_id"],
                "filename": row["filename"],
                "row_count": int(row["row_count"]),
                "col_count": int(row["col_count"]),
                "created_at": row["created_at"],
                "analysis_status": "completed" if latest_run_id else "pending",
                "latest_run_id": latest_run_id,
                "latest_predicted_at": row["latest_predicted_at"],
                "anomaly_count": int(row["anomaly_count"] or 0),
                "normal_count": int(row["normal_count"] or 0),
            }
        )

    return {
        "status": "ok",
        "items": items,
        "pagination": _pagination(page, page_size, total_items),
    }


def delete_upload(session: Session, upload_id: uuid.UUID) -> None:
    """Delete one upload and rely on PostgreSQL ON DELETE CASCADE for children."""
    with session.begin():
        upload = session.get(CsvUpload, upload_id)
        if upload is None:
            raise UploadNotFoundError(f"Upload '{upload_id}' khÃ´ng tá»“n táº¡i.")
        session.delete(upload)


def _apply_prediction_filter(
    query: Select,
    prediction_filter: PredictionFilter,
) -> Select:
    if prediction_filter == "anomaly":
        return query.where(FlowPrediction.prediction == 1)
    if prediction_filter == "normal":
        return query.where(FlowPrediction.prediction == 0)
    return query


def list_global_flows(
    session: Session,
    *,
    page: int,
    page_size: int,
    prediction_filter: PredictionFilter = "all",
) -> dict[str, object]:
    """Return globally paginated predictions from only each upload's latest run."""
    latest_runs = _latest_runs_per_upload()

    count_query = (
        select(func.count())
        .select_from(FlowPrediction)
        .join(latest_runs, latest_runs.c.run_id == FlowPrediction.inference_run_id)
    )
    count_query = _apply_prediction_filter(count_query, prediction_filter)
    total_items = int(session.scalar(count_query) or 0)

    rows_query = (
        select(
            latest_runs.c.upload_id,
            CsvUpload.original_filename.label("filename"),
            latest_runs.c.run_id,
            FlowPrediction.row_index,
            FlowPrediction.reconstruction_error,
            FlowPrediction.prediction,
            FlowPrediction.prediction_label,
            FlowPrediction.created_at,
        )
        .select_from(FlowPrediction)
        .join(latest_runs, latest_runs.c.run_id == FlowPrediction.inference_run_id)
        .join(CsvUpload, CsvUpload.id == latest_runs.c.upload_id)
        .order_by(
            FlowPrediction.created_at.desc(),
            latest_runs.c.run_id.desc(),
            FlowPrediction.row_index,
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows_query = _apply_prediction_filter(rows_query, prediction_filter)

    items = [
        {
            "upload_id": row["upload_id"],
            "filename": row["filename"],
            "run_id": row["run_id"],
            "row_index": int(row["row_index"]),
            "reconstruction_error": float(row["reconstruction_error"]),
            "prediction": int(row["prediction"]),
            "prediction_label": row["prediction_label"],
            "created_at": row["created_at"],
        }
        for row in session.execute(rows_query).mappings().all()
    ]

    return {
        "status": "ok",
        "items": items,
        "pagination": _pagination(page, page_size, total_items),
    }