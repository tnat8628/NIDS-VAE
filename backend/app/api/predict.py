"""
backend/app/api/predict.py
----------------------------
Router FastAPI cho endpoint POST /predict.

Nhận file CSV từ multipart/form-data, chạy toàn bộ pipeline suy diễn VAE,
và trả về kết quả phân loại anomaly cho từng network flow.

Quy trình:
  1. Validate file (tồn tại, đuôi .csv, không rỗng)
  2. Đọc CSV thành pandas DataFrame
  3. Gọi VAEInferenceService.predict_dataframe(df)
  4. Chuyển kết quả sang Pydantic schemas
  5. Trả về PredictionResponse
"""

import io
import logging
import os
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.core.inference import (
    ArtifactLoadError,
    InferenceError,
    ModelDimensionError,
    VAEInferenceService,
)
from backend.app.core.preprocessing import PreprocessingError
from backend.app.core.thresholding import ThresholdingError
from backend.app.config import MAX_UPLOAD_BYTES
from backend.app.schemas.response_schema import (
    ErrorResponse,
    FlowPrediction,
    PredictionResponse,
    PredictionRunResponse,
    PredictionSummary,
)
from backend.app.core.debug_inference import debug_single_sample_flow
from backend.app.db.database import get_db
from backend.app.services.prediction_storage import (
    PredictionStorageError,
    UploadNotFoundError,
    load_upload_dataframe,
    save_prediction_run,
)
# Import hàm lưu kết quả vào cache để GET /results có thể fetch lại
# Import muộn để tránh circular imports (results.py không import predict.py)
from backend.app.api import results as results_module

logger = logging.getLogger(__name__)

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter()

# ── Singleton inference service ───────────────────────────────────────────────
# Được tạo một lần khi module được import — không tái tạo cho mỗi request
# Nếu load artifact thất bại khi khởi động, lỗi sẽ được ghi log rõ ràng
_inference_service: VAEInferenceService | None = None
_service_error: str | None = None

try:
    _inference_service = VAEInferenceService()
    logger.info("VAEInferenceService singleton đã khởi tạo thành công.")
except (ArtifactLoadError, ModelDimensionError, Exception) as _exc:
    _service_error = str(_exc)
    logger.error("Không thể khởi tạo VAEInferenceService: %s", _service_error)


def get_inference_service() -> VAEInferenceService:
    """
    Trả về singleton VAEInferenceService.

    Raises:
        HTTPException 503: Nếu dịch vụ chưa được khởi tạo thành công.
    """
    if _inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service chưa sẵn sàng: {_service_error}",
        )
    return _inference_service


# ── POST /predict ─────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Dự đoán anomaly từ CSV network flows",
    description=(
        "Nhận file CSV chứa network flow features, chạy qua pipeline VAE, "
        "và trả về phân loại normal/anomaly cho từng flow cùng reconstruction error."
    ),
    responses={
        200: {"description": "Dự đoán thành công"},
        400: {"model": ErrorResponse, "description": "File không hợp lệ hoặc lỗi preprocessing"},
        422: {"description": "Thiếu file hoặc sai định dạng"},
        503: {"model": ErrorResponse, "description": "Inference service chưa sẵn sàng"},
    },
)
async def predict(
    file: UploadFile,
) -> PredictionResponse:
    """
    Chạy pipeline dự đoán VAE anomaly detection trên CSV đầu vào.

    Args:
        file: File CSV multipart/form-data với network flow features.

    Returns:
        PredictionResponse với summary và per-flow results.

    Raises:
        HTTPException 400: File rỗng, không phải CSV, hoặc lỗi preprocessing.
        HTTPException 503: Inference service chưa sẵn sàng.
    """
    svc = get_inference_service()

    # ── Bước 1: Validate file ─────────────────────────────────────────────────
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận file .csv. "
                   f"File nhận được: '{file.filename or 'unknown'}'",
        )

    # ── Bước 2: Đọc nội dung file ─────────────────────────────────────────────
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Không thể đọc file tải lên: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc file: {exc}",
        ) from exc

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                "Hãy chia nhỏ CSV trước khi upload."
            ),
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV rỗng — không có dữ liệu để xử lý.",
        )

    # ── Bước 3: Parse CSV thành DataFrame ─────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        logger.warning("Không thể parse CSV '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc file CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV không có dữ liệu (0 hàng sau khi parse).",
        )

    logger.info("Nhận file '%s': %d hàng × %d cột", file.filename, len(df), len(df.columns))

    # ── Bước 4: Chạy inference pipeline ──────────────────────────────────────
    if os.getenv("DEBUG_INFERENCE", "false").lower() == "true":
        debug_single_sample_flow(svc, df)
    try:
        raw_result = svc.predict_dataframe(df)
    except PreprocessingError as exc:
        logger.warning("Lỗi preprocessing '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi tiền xử lý dữ liệu: {exc}",
        ) from exc
    except (InferenceError, ModelDimensionError, ThresholdingError) as exc:
        logger.error("Lỗi inference '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi suy diễn: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Lỗi không xác định khi predict '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi nội bộ: {exc}",
        ) from exc

    # ── Bước 5: Chuyển kết quả sang Pydantic schemas ─────────────────────────
    summary = PredictionSummary(
        total_flows=raw_result["total_flows"],
        anomaly_count=raw_result["anomaly_count"],
        normal_count=raw_result["normal_count"],
        anomaly_rate=raw_result["anomaly_rate"],
        threshold=raw_result["threshold"],
    )

    results = [
        FlowPrediction(
            row_index=item["row_index"],
            reconstruction_error=item["reconstruction_error"],
            prediction=item["prediction"],
            prediction_label=item["prediction_label"],
        )
        for item in raw_result["results"]
    ]

    response = PredictionResponse(status="ok", summary=summary, results=results)

    # ── Lưu kết quả vào cache để GET /results có thể truy xuất lại ───────────
    results_module.store_latest_result(response)

    logger.info(
        "Predict '%s': %d flows, %d anomaly (%.1f%%)",
        file.filename,
        summary.total_flows,
        summary.anomaly_count,
        summary.anomaly_rate * 100,
    )

    return response


@router.post(
    "/uploads/{upload_id}/predict",
    response_model=PredictionRunResponse,
    summary="Chạy VAE cho một CSV upload đã lưu",
    description=(
        "Đọc csv_rows từ PostgreSQL, chạy pipeline VAE hiện có, rồi lưu "
        "atomically inference_run và toàn bộ flow_predictions."
    ),
    responses={
        200: {"description": "Prediction đã được lưu vào PostgreSQL"},
        404: {"model": ErrorResponse, "description": "Upload không tồn tại"},
        503: {"model": ErrorResponse, "description": "Không thể lưu prediction"},
    },
)
def predict_persisted_upload(
    upload_id: UUID,
    db: Session = Depends(get_db),
) -> PredictionRunResponse:
    """Run and persist VAE inference for an existing upload ID."""
    svc = get_inference_service()

    try:
        persisted = load_upload_dataframe(db, upload_id)
    except UploadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    # Kết thúc read transaction trước khi chạy model. Transaction ghi phía
    # dưới chỉ bao quanh inference_run và toàn bộ flow_predictions.
    db.rollback()

    if os.getenv("DEBUG_INFERENCE", "false").lower() == "true":
        debug_single_sample_flow(svc, persisted.dataframe)

    try:
        raw_result = svc.predict_dataframe(persisted.dataframe)
    except PreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi tiền xử lý dữ liệu: {exc}",
        ) from exc
    except (InferenceError, ModelDimensionError, ThresholdingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi suy diễn: {exc}",
        ) from exc

    try:
        inference_run_id = save_prediction_run(
            db,
            upload_id=upload_id,
            row_ids=persisted.row_ids,
            raw_result=raw_result,
        )
    except PredictionStorageError as exc:
        logger.exception("Không thể lưu prediction cho upload '%s'", upload_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Không thể lưu kết quả prediction vào database. "
                "Không có inference run dở dang được tạo."
            ),
        ) from exc

    summary = PredictionSummary(
        total_flows=raw_result["total_flows"],
        anomaly_count=raw_result["anomaly_count"],
        normal_count=raw_result["normal_count"],
        anomaly_rate=raw_result["anomaly_rate"],
        threshold=raw_result["threshold"],
    )
    results_url = f"/uploads/{upload_id}/results?inference_run_id={inference_run_id}"
    logger.info(
        "Đã lưu inference run '%s' cho upload '%s': %d flows",
        inference_run_id,
        upload_id,
        summary.total_flows,
    )
    return PredictionRunResponse(
        status="ok",
        upload_id=upload_id,
        inference_run_id=inference_run_id,
        summary=summary,
        results_url=results_url,
    )