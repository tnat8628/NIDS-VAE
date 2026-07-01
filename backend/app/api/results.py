"""
backend/app/api/results.py
----------------------------
Router FastAPI cho endpoint GET /results.

Trả về kết quả dự đoán gần nhất được cache trong bộ nhớ.
Endpoint này cho phép dashboard fetch lại kết quả mà không cần
upload lại file — hữu ích khi người dùng reload trang.

LƯU Ý: Trong MVP này, results được lưu trong bộ nhớ (in-memory).
Không có persistence qua khởi động lại server.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.response_schema import ErrorResponse, PredictionResponse

logger = logging.getLogger(__name__)

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter()

# ── Cache kết quả gần nhất trong bộ nhớ ──────────────────────────────────────
# Được cập nhật bởi POST /predict sau mỗi lần dự đoán thành công
_latest_result: PredictionResponse | None = None


def store_latest_result(result: PredictionResponse) -> None:
    """
    Lưu kết quả dự đoán gần nhất vào cache bộ nhớ.

    Được gọi từ predict router sau khi dự đoán thành công.

    Args:
        result: PredictionResponse cần cache.
    """
    global _latest_result
    _latest_result = result
    logger.debug("Đã cache kết quả: %d flows", result.summary.total_flows)


# ── GET /results ──────────────────────────────────────────────────────────────

@router.get(
    "/results",
    response_model=PredictionResponse,
    summary="Lấy kết quả dự đoán gần nhất",
    description=(
        "Trả về kết quả dự đoán gần nhất từ POST /predict. "
        "Hữu ích cho dashboard reload mà không cần upload lại file."
    ),
    responses={
        200: {"description": "Kết quả dự đoán gần nhất"},
        404: {"model": ErrorResponse, "description": "Chưa có kết quả nào"},
    },
)
def get_results() -> PredictionResponse:
    """
    Trả về kết quả dự đoán gần nhất đã được cache.

    Returns:
        PredictionResponse của lần dự đoán gần nhất.

    Raises:
        HTTPException 404: Nếu chưa có lần dự đoán nào được thực hiện.
    """
    if _latest_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chưa có kết quả dự đoán nào. Hãy gửi POST /predict trước.",
        )

    logger.debug("Trả về kết quả cache: %d flows", _latest_result.summary.total_flows)
    return _latest_result
