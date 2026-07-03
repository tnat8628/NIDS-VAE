"""System-wide PostgreSQL aggregates for the Overview dashboard."""

from __future__ import annotations

from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import CsvUpload, FlowPrediction, InferenceRun
from backend.app.services.prediction_storage import HISTOGRAM_BINS


def _latest_runs_per_upload():
    """Return a subquery containing exactly the latest persisted run per upload."""
    ranked = select(
        InferenceRun.id.label("run_id"),
        InferenceRun.upload_id.label("upload_id"),
        InferenceRun.total_flows.label("total_flows"),
        InferenceRun.anomaly_count.label("anomaly_count"),
        InferenceRun.normal_count.label("normal_count"),
        InferenceRun.created_at.label("predicted_at"),
        func.row_number()
        .over(
            partition_by=InferenceRun.upload_id,
            order_by=(InferenceRun.created_at.desc(), InferenceRun.id.desc()),
        )
        .label("run_rank"),
    ).subquery()

    return (
        select(
            ranked.c.run_id,
            ranked.c.upload_id,
            ranked.c.total_flows,
            ranked.c.anomaly_count,
            ranked.c.normal_count,
            ranked.c.predicted_at,
        )
        .where(ranked.c.run_rank == 1)
        .subquery()
    )


def _overview_histogram(
    session: Session,
    latest_runs,
    *,
    normal_count: int,
    anomaly_count: int,
) -> list[dict[str, object]]:
    selected_run_ids = select(latest_runs.c.run_id)
    min_error, max_error = session.execute(
        select(
            func.min(FlowPrediction.reconstruction_error),
            func.max(FlowPrediction.reconstruction_error),
        ).where(FlowPrediction.inference_run_id.in_(selected_run_ids))
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
        bins[0]["normal"] = normal_count
        bins[0]["anomaly"] = anomaly_count
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
        .where(FlowPrediction.inference_run_id.in_(selected_run_ids))
        .group_by(bucket, FlowPrediction.prediction)
    ).all()

    for bucket_number, prediction, count in counts:
        target = bins[max(0, int(bucket_number) - 1)]
        target["anomaly" if prediction == 1 else "normal"] = int(count)

    return bins


def get_dashboard_overview(session: Session) -> dict[str, object]:
    """Aggregate uploads and only the latest successful run for each upload."""
    total_uploads, total_uploaded_flows = session.execute(
        select(
            func.count(CsvUpload.id),
            func.coalesce(func.sum(CsvUpload.row_count), 0),
        )
    ).one()

    latest_runs = _latest_runs_per_upload()
    (
        analyzed_uploads,
        total_analyzed_flows,
        anomaly_count,
        normal_count,
    ) = session.execute(
        select(
            func.count(latest_runs.c.run_id),
            func.coalesce(func.sum(latest_runs.c.total_flows), 0),
            func.coalesce(func.sum(latest_runs.c.anomaly_count), 0),
            func.coalesce(func.sum(latest_runs.c.normal_count), 0),
        ).select_from(latest_runs)
    ).one()

    total_uploads = int(total_uploads)
    total_uploaded_flows = int(total_uploaded_flows)
    analyzed_uploads = int(analyzed_uploads)
    total_analyzed_flows = int(total_analyzed_flows)
    anomaly_count = int(anomaly_count)
    normal_count = int(normal_count)
    anomaly_rate = (
        anomaly_count / total_analyzed_flows if total_analyzed_flows else 0.0
    )

    latest_activity = session.execute(
        select(
            latest_runs.c.upload_id,
            latest_runs.c.run_id,
            CsvUpload.original_filename,
            CsvUpload.created_at,
            latest_runs.c.predicted_at,
        )
        .join(CsvUpload, CsvUpload.id == latest_runs.c.upload_id)
        .order_by(
            latest_runs.c.predicted_at.desc(),
            latest_runs.c.run_id.desc(),
        )
        .limit(1)
    ).one_or_none()

    if latest_activity is None:
        latest_upload = session.scalars(
            select(CsvUpload)
            .order_by(CsvUpload.created_at.desc(), CsvUpload.id.desc())
            .limit(1)
        ).first()
        activity_payload = {
            "latest_upload_id": latest_upload.id if latest_upload else None,
            "latest_run_id": None,
            "latest_filename": (
                latest_upload.original_filename if latest_upload else None
            ),
            "latest_uploaded_at": latest_upload.created_at if latest_upload else None,
            "latest_predicted_at": None,
        }
    else:
        activity_payload = {
            "latest_upload_id": latest_activity.upload_id,
            "latest_run_id": latest_activity.run_id,
            "latest_filename": latest_activity.original_filename,
            "latest_uploaded_at": latest_activity.created_at,
            "latest_predicted_at": latest_activity.predicted_at,
        }

    return {
        "status": "ok",
        "uploads": {
            "total_uploads": total_uploads,
            "total_uploaded_flows": total_uploaded_flows,
        },
        "analysis": {
            "analyzed_uploads": analyzed_uploads,
            "total_analyzed_flows": total_analyzed_flows,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_rate": anomaly_rate,
        },
        "histogram": _overview_histogram(
            session,
            latest_runs,
            normal_count=normal_count,
            anomaly_count=anomaly_count,
        ),
        "classification": {
            "normal": normal_count,
            "anomaly": anomaly_count,
        },
        "latest_activity": activity_payload,
    }