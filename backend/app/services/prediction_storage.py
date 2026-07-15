"""Persistence and paginated reads for database-backed VAE results."""

from __future__ import annotations

import csv
import io
import math
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import Integer, func, insert, select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    CsvRow,
    CsvUpload,
    FlowPrediction,
    InferenceRun,
)

PREDICTION_INSERT_BATCH_SIZE = 1_000
HISTOGRAM_BINS = 30
TOP_ANOMALIES_LIMIT = 100
GROUND_TRUTH_LABEL_KEYS = {"label", "actual_label", "true_label", "binary_label"}
ExportPredictionFilter = Literal["all", "anomaly", "normal"]
ExportSort = Literal["idx", "err_desc", "err_asc"]


class UploadNotFoundError(LookupError):
    """Raised when an upload ID does not exist."""


class PredictionRunNotFoundError(LookupError):
    """Raised when no completed prediction run exists for an upload."""


class PredictionStorageError(RuntimeError):
    """Raised when a prediction run cannot be persisted atomically."""


@dataclass(frozen=True)
class PersistedUploadData:
    """DataFrame and stable csv_rows IDs loaded in row order."""

    dataframe: pd.DataFrame
    row_ids: dict[int, uuid.UUID]


def load_upload_dataframe(
    session: Session,
    upload_id: uuid.UUID,
) -> PersistedUploadData:
    """Load all persisted CSV payloads for one inference execution."""
    upload = session.get(CsvUpload, upload_id)
    if upload is None:
        raise UploadNotFoundError(f"Upload '{upload_id}' không tồn tại.")

    rows = session.execute(
        select(CsvRow.id, CsvRow.row_index, CsvRow.payload)
        .where(CsvRow.upload_id == upload_id)
        .order_by(CsvRow.row_index)
    ).all()

    if len(rows) != upload.row_count:
        raise PredictionStorageError(
            f"Upload '{upload_id}' thiếu dữ liệu: metadata={upload.row_count}, rows={len(rows)}."
        )

    dataframe = pd.DataFrame(
        [dict(row.payload) for row in rows],
        columns=upload.columns,
    )
    dataframe.index = [row.row_index for row in rows]
    row_ids = {row.row_index: row.id for row in rows}
    return PersistedUploadData(dataframe=dataframe, row_ids=row_ids)


def save_prediction_run(
    session: Session,
    *,
    upload_id: uuid.UUID,
    row_ids: dict[int, uuid.UUID],
    raw_result: dict[str, object],
) -> uuid.UUID:
    """Persist one inference run and every per-row result in one transaction."""
    result_items = raw_result["results"]
    if not isinstance(result_items, list):
        raise PredictionStorageError("Inference results không phải là một danh sách.")

    total_flows = int(raw_result["total_flows"])
    if total_flows != len(result_items) or total_flows != len(row_ids):
        raise PredictionStorageError(
            "Số prediction không khớp với số csv_rows của upload."
        )

    run = InferenceRun(
        upload_id=upload_id,
        total_flows=total_flows,
        anomaly_count=int(raw_result["anomaly_count"]),
        normal_count=int(raw_result["normal_count"]),
        anomaly_rate=float(raw_result["anomaly_rate"]),
        threshold=float(raw_result["threshold"]),
    )

    try:
        with session.begin():
            session.add(run)
            session.flush()

            batch: list[dict[str, object]] = []
            for item in result_items:
                row_index = int(item["row_index"])
                csv_row_id = row_ids.get(row_index)
                if csv_row_id is None:
                    raise PredictionStorageError(
                        f"Không tìm thấy csv_row cho row_index={row_index}."
                    )

                batch.append(
                    {
                        "id": uuid.uuid4(),
                        "inference_run_id": run.id,
                        "csv_row_id": csv_row_id,
                        "row_index": row_index,
                        "reconstruction_error": float(item["reconstruction_error"]),
                        "prediction": int(item["prediction"]),
                        "prediction_label": str(item["prediction_label"]),
                    }
                )

                if len(batch) >= PREDICTION_INSERT_BATCH_SIZE:
                    session.execute(insert(FlowPrediction), batch)
                    batch.clear()

            if batch:
                session.execute(insert(FlowPrediction), batch)
    except PredictionStorageError:
        raise
    except Exception as exc:
        raise PredictionStorageError(
            "Không thể lưu trọn vẹn kết quả inference vào database."
        ) from exc

    return run.id


def _severity(error: float, threshold: float) -> str:
    if error > threshold * 2:
        return "critical"
    if error > threshold * 1.5:
        return "high"
    if error > threshold:
        return "medium"
    return "low"


def _fold_text(value: object) -> str:
    text = str(value).strip().casefold()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_ground_truth_label(value: object | None) -> int | None:
    """Map a source CSV ground-truth label to 0=normal or 1=attack."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 0.0:
            return 0
        if float(value) == 1.0:
            return 1

    folded = _fold_text(value)
    if not folded:
        return None
    if folded in {"0", "benign", "normal", "binh thuong"}:
        return 0
    if folded == "1":
        return 1

    return 1


def extract_ground_truth_label(payload: dict[str, object]) -> tuple[object | None, int | None]:
    """Return the first ground-truth label found in a stored CSV row payload."""
    for key, value in payload.items():
        if key.strip().casefold() in GROUND_TRUTH_LABEL_KEYS:
            return value, normalize_ground_truth_label(value)
    return None, None


def compare_prediction(prediction: int, actual_binary: int | None) -> str:
    """Return TP/FP/TN/FN when ground truth exists, otherwise N/A."""
    if actual_binary is None:
        return "N/A"
    if prediction == 1 and actual_binary == 1:
        return "TP"
    if prediction == 1 and actual_binary == 0:
        return "FP"
    if prediction == 0 and actual_binary == 0:
        return "TN"
    return "FN"


def _safe_ratio(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.6f}"


def _find_run(
    session: Session,
    *,
    upload_id: uuid.UUID,
    inference_run_id: uuid.UUID | None = None,
) -> InferenceRun:
    run_query = select(InferenceRun).where(InferenceRun.upload_id == upload_id)
    if inference_run_id is not None:
        run_query = run_query.where(InferenceRun.id == inference_run_id)
    else:
        run_query = run_query.order_by(
            InferenceRun.created_at.desc(),
            InferenceRun.id.desc(),
        )

    run = session.scalars(run_query.limit(1)).first()
    if run is None:
        upload_exists = session.scalar(
            select(func.count()).select_from(CsvUpload).where(CsvUpload.id == upload_id)
        )
        if not upload_exists:
            raise UploadNotFoundError(f"Upload '{upload_id}' không tồn tại.")
        raise PredictionRunNotFoundError(
            f"Upload '{upload_id}' chưa có kết quả prediction."
        )

    return run


def _prediction_item(
    row: FlowPrediction,
    threshold: float,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    actual_label, actual_binary = extract_ground_truth_label(payload or {})
    return {
        "row_index": row.row_index,
        "reconstruction_error": row.reconstruction_error,
        "prediction": row.prediction,
        "prediction_label": row.prediction_label,
        "severity": _severity(row.reconstruction_error, threshold),
        "actual_label": actual_label,
        "actual_binary": actual_binary,
    }


def _histogram(session: Session, run: InferenceRun) -> list[dict[str, object]]:
    min_error, max_error = session.execute(
        select(
            func.min(FlowPrediction.reconstruction_error),
            func.max(FlowPrediction.reconstruction_error),
        ).where(FlowPrediction.inference_run_id == run.id)
    ).one()

    if min_error is None or max_error is None:
        return []

    min_error = float(min_error)
    max_error = float(max_error)
    step = (max_error - min_error) / HISTOGRAM_BINS if max_error > min_error else 1.0
    bins = [
        {
            "bin": f"{min_error + index * step:.4f}",
            "normal": 0,
            "anomaly": 0,
        }
        for index in range(HISTOGRAM_BINS)
    ]

    if max_error == min_error:
        bins[0]["normal"] = run.normal_count
        bins[0]["anomaly"] = run.anomaly_count
        return bins

    bucket = func.least(
        HISTOGRAM_BINS,
        func.width_bucket(
            FlowPrediction.reconstruction_error,
            min_error,
            max_error,
            HISTOGRAM_BINS,
        ),
    ).cast(Integer)

    counts = session.execute(
        select(
            bucket.label("bucket"),
            FlowPrediction.prediction,
            func.count().label("count"),
        )
        .where(FlowPrediction.inference_run_id == run.id)
        .group_by(bucket, FlowPrediction.prediction)
    ).all()

    for bucket_number, prediction, count in counts:
        target = bins[max(0, int(bucket_number) - 1)]
        target["anomaly" if prediction == 1 else "normal"] = int(count)

    return bins


def get_paginated_results(
    session: Session,
    *,
    upload_id: uuid.UUID,
    page: int,
    page_size: int,
    inference_run_id: uuid.UUID | None = None,
) -> dict[str, object]:
    """Read one bounded page plus aggregates computed from the complete DB run."""
    run = _find_run(
        session,
        upload_id=upload_id,
        inference_run_id=inference_run_id,
    )

    items = session.execute(
        select(FlowPrediction, CsvRow.payload)
        .join(CsvRow, FlowPrediction.csv_row_id == CsvRow.id)
        .where(FlowPrediction.inference_run_id == run.id)
        .order_by(FlowPrediction.row_index)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    top_anomalies = session.execute(
        select(FlowPrediction, CsvRow.payload)
        .join(CsvRow, FlowPrediction.csv_row_id == CsvRow.id)
        .where(
            FlowPrediction.inference_run_id == run.id,
            FlowPrediction.prediction == 1,
        )
        .order_by(
            FlowPrediction.reconstruction_error.desc(),
            FlowPrediction.row_index,
        )
        .limit(TOP_ANOMALIES_LIMIT)
    ).all()

    total_pages = math.ceil(run.total_flows / page_size) if run.total_flows else 0
    return {
        "status": "ok",
        "upload_id": upload_id,
        "inference_run_id": run.id,
        "summary": {
            "total_flows": run.total_flows,
            "anomaly_count": run.anomaly_count,
            "normal_count": run.normal_count,
            "anomaly_rate": run.anomaly_rate,
            "threshold": run.threshold,
        },
        "items": [
            _prediction_item(item, run.threshold, dict(payload))
            for item, payload in items
        ],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": run.total_flows,
            "total_pages": total_pages,
            "has_previous": page > 1,
            "has_next": page < total_pages,
        },
        "aggregates": {
            "histogram": _histogram(session, run),
            "top_anomalies": [
                _prediction_item(item, run.threshold, dict(payload))
                for item, payload in top_anomalies
            ],
        },
    }


def build_results_export_csv(
    session: Session,
    *,
    upload_id: uuid.UUID,
    inference_run_id: uuid.UUID | None = None,
    prediction: ExportPredictionFilter = "all",
    sort: ExportSort = "idx",
) -> bytes:
    """Build a full CSV export for one inference run, without pagination."""
    run = _find_run(
        session,
        upload_id=upload_id,
        inference_run_id=inference_run_id,
    )

    stmt = (
        select(FlowPrediction, CsvRow.payload)
        .join(CsvRow, FlowPrediction.csv_row_id == CsvRow.id)
        .where(FlowPrediction.inference_run_id == run.id)
    )
    if prediction == "anomaly":
        stmt = stmt.where(FlowPrediction.prediction == 1)
    elif prediction == "normal":
        stmt = stmt.where(FlowPrediction.prediction == 0)

    if sort == "err_desc":
        stmt = stmt.order_by(
            FlowPrediction.reconstruction_error.desc(),
            FlowPrediction.row_index,
        )
    elif sort == "err_asc":
        stmt = stmt.order_by(
            FlowPrediction.reconstruction_error.asc(),
            FlowPrediction.row_index,
        )
    else:
        stmt = stmt.order_by(FlowPrediction.row_index)

    rows = session.execute(stmt).all()

    has_source_file = any("_source_file" in payload for _, payload in rows)
    base_columns = [
        "flow_no",
        "row_index",
        "reconstruction_error",
        "threshold",
        "prediction",
        "predicted_label_en",
        "predicted_label_vi",
        "severity",
        "actual_label",
        "actual_binary",
        "compare_result",
    ]
    if has_source_file:
        base_columns.append("original_source_file")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=base_columns)
    writer.writeheader()

    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    has_ground_truth = False

    for flow, payload in rows:
        payload = dict(payload)
        actual_label, actual_binary = extract_ground_truth_label(payload)
        result = compare_prediction(flow.prediction, actual_binary)
        if result != "N/A":
            has_ground_truth = True
            confusion[result] += 1

        record = {
            "flow_no": flow.row_index + 1,
            "row_index": flow.row_index,
            "reconstruction_error": flow.reconstruction_error,
            "threshold": run.threshold,
            "prediction": flow.prediction,
            "predicted_label_en": "anomaly" if flow.prediction == 1 else "normal",
            "predicted_label_vi": "Bất thường" if flow.prediction == 1 else "Bình thường",
            "severity": _severity(flow.reconstruction_error, run.threshold),
            "actual_label": actual_label if actual_label is not None else "N/A",
            "actual_binary": actual_binary if actual_binary is not None else "N/A",
            "compare_result": result,
        }
        if has_source_file:
            record["original_source_file"] = payload.get("_source_file", "")

        writer.writerow(record)

    tp = confusion["TP"]
    fp = confusion["FP"]
    tn = confusion["TN"]
    fn = confusion["FN"]
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)

    output.write("\n\n---- SUMMARY ----\n")
    summary_writer = csv.writer(output)
    summary_writer.writerow(["metric", "value"])
    summary_writer.writerow(["exported_filter", prediction])
    summary_writer.writerow(["total_exported", len(rows)])
    summary_writer.writerow(["has_ground_truth", str(has_ground_truth).lower()])
    for key in ("TP", "FP", "TN", "FN"):
        summary_writer.writerow([key, confusion[key] if has_ground_truth else "N/A"])
    summary_writer.writerow(["accuracy", _safe_ratio(tp + tn, tp + tn + fp + fn)])
    summary_writer.writerow(["precision", _safe_ratio(tp, tp + fp)])
    summary_writer.writerow(["recall", _safe_ratio(tp, tp + fn)])
    if precision is None or recall is None or precision + recall == 0:
        f1_score = "N/A"
    else:
        f1_score = f"{(2 * precision * recall) / (precision + recall):.6f}"
    summary_writer.writerow(["f1_score", f1_score])

    return output.getvalue().encode("utf-8-sig")