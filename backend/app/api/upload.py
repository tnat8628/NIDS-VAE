"""
backend/app/api/upload.py
---------------------------
Router FastAPI cho endpoint POST /upload.

Nhận file CSV, validate cơ bản, lưu metadata cùng toàn bộ data rows vào
PostgreSQL, và trả về thông tin xác nhận.

Raw file không được lưu lên disk. Để phân tích trực tiếp, dùng POST /predict.
"""

import io
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.config import MAX_UPLOAD_BYTES
from backend.app.db.database import get_db
from backend.app.schemas.response_schema import (
    DeleteUploadResponse,
    UploadListResponse,
    UploadResponse,
)
from backend.app.services.prediction_storage import UploadNotFoundError
from backend.app.services.upload_management import (
    UploadFilter,
    delete_upload,
    list_uploads,
)
from backend.app.services.upload_storage import UploadStorageError, save_csv_upload

logger = logging.getLogger(__name__)

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter()


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.get(
    "/uploads",
    response_model=UploadListResponse,
    summary="Danh sÃ¡ch file CSV Ä‘Ã£ upload",
    description=(
        "PhÃ¢n trang danh sÃ¡ch upload tá»« PostgreSQL, kÃ¨m latest successful "
        "inference run cá»§a má»—i file náº¿u Ä‘Ã£ phÃ¢n tÃ­ch."
    ),
)
def uploads(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    upload_filter: UploadFilter = Query("all", alias="filter"),
    db: Session = Depends(get_db),
) -> UploadListResponse:
    return UploadListResponse.model_validate(
        list_uploads(
            db,
            page=page,
            page_size=page_size,
            upload_filter=upload_filter,
        )
    )


@router.delete(
    "/uploads/{upload_id}",
    response_model=DeleteUploadResponse,
    summary="XÃ³a file CSV vÃ  toÃ n bá»™ dá»¯ liá»‡u liÃªn quan",
    responses={404: {"description": "Upload khÃ´ng tá»“n táº¡i"}},
)
def remove_upload(
    upload_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DeleteUploadResponse:
    try:
        delete_upload(db, upload_id)
    except UploadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload '{upload_id}' khÃ´ng tá»“n táº¡i.",
        ) from exc

    return DeleteUploadResponse(
        upload_id=upload_id,
        message="ÄÃ£ xÃ³a file vÃ  toÃ n bá»™ dá»¯ liá»‡u liÃªn quan.",
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Validate và lưu file CSV đầu vào",
    description=(
        "Nhận file CSV network flows, kiểm tra định dạng cơ bản, "
        "lưu metadata và từng dòng dữ liệu vào PostgreSQL. "
        "Dùng POST /predict để chạy phân tích ngay lập tức."
    ),
    responses={
        200: {"description": "File hợp lệ"},
        400: {"description": "File không hợp lệ hoặc không phải CSV"},
    },
)
async def upload(
    file: UploadFile,
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Validate file CSV đầu vào, lưu atomically, và trả về metadata.

    Args:
        file: File CSV multipart/form-data.

    Returns:
        UploadResponse với ID và thông tin về file đã lưu.

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

    try:
        upload_id = save_csv_upload(
            db,
            original_filename=file.filename,
            contents=contents,
            dataframe=df,
        )
    except UploadStorageError as exc:
        logger.exception("Không thể lưu upload '%s' vào database", file.filename)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Không thể lưu file CSV vào database. Không có dữ liệu dở dang được tạo.",
        ) from exc

    logger.info(
        "Đã lưu upload '%s' (%s): %d hàng × %d cột",
        file.filename,
        upload_id,
        len(df),
        len(df.columns),
    )

    return UploadResponse(
        status="ok",
        upload_id=upload_id,
        filename=file.filename,
        row_count=len(df),
        col_count=len(df.columns),
        message="File và toàn bộ các dòng dữ liệu đã được lưu vào database.",
    )
