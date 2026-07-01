"""
backend/app/api/upload.py
---------------------------
Router FastAPI cho endpoint POST /upload.

Nhận file CSV, validate cơ bản, và trả về thông tin xác nhận.
Upload endpoint này là bước đầu tiên — dành cho luồng 2-bước
(upload trước, predict sau). Hiện tại chỉ validate và trả metadata.

LƯU Ý: Trong MVP này, /upload không lưu file lên disk.
Để phân tích trực tiếp, dùng POST /predict.
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, status

from backend.app.config import MAX_UPLOAD_BYTES
from backend.app.schemas.response_schema import UploadResponse

logger = logging.getLogger(__name__)

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter()


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Validate và xác nhận file CSV đầu vào",
    description=(
        "Nhận file CSV network flows, kiểm tra định dạng cơ bản, "
        "và trả về metadata. Dùng POST /predict để chạy phân tích ngay lập tức."
    ),
    responses={
        200: {"description": "File hợp lệ"},
        400: {"description": "File không hợp lệ hoặc không phải CSV"},
    },
)
async def upload(file: UploadFile) -> UploadResponse:
    """
    Validate file CSV đầu vào và trả về metadata.

    Args:
        file: File CSV multipart/form-data.

    Returns:
        UploadResponse với thông tin về file đã upload.

    Raises:
        HTTPException 400: File thiếu, không phải CSV, rỗng, hoặc không parse được.
    """
    # ── Validate tên file và đuôi mở rộng ────────────────────────────────────
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chỉ chấp nhận file .csv. "
                   f"File nhận được: '{file.filename or 'unknown'}'",
        )

    # ── Đọc và validate nội dung ──────────────────────────────────────────────
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc file: {exc}",
        ) from exc

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV rỗng.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                "Hãy chia nhỏ CSV trước khi upload."
            ),
        )

    # ── Parse CSV để đếm hàng/cột ─────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể parse file CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV không có dữ liệu (0 hàng).",
        )

    logger.info(
        "Upload '%s' hợp lệ: %d hàng × %d cột",
        file.filename,
        len(df),
        len(df.columns),
    )

    return UploadResponse(
        status="ok",
        filename=file.filename,
        row_count=len(df),
        col_count=len(df.columns),
        message="File hợp lệ. Gửi POST /predict với cùng file để nhận kết quả dự đoán.",
    )
