"""
backend/app/main.py
---------------------
Điểm khởi động chính của FastAPI NIDS VAE backend.

Chịu trách nhiệm:
  - Tạo ứng dụng FastAPI với metadata đầy đủ
  - Cấu hình CORS cho frontend React (localhost:3000)
  - Đăng ký tất cả API routers
  - Cung cấp GET /health endpoint để kiểm tra trạng thái
  - Thiết lập logging cơ bản

Khởi động:
  uvicorn backend.app.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import predict as predict_module
from backend.app.api import results as results_module
from backend.app.api import upload as upload_module
from backend.app.schemas.response_schema import ErrorResponse, HealthResponse

# ── Cấu hình logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Tạo FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS VAE Anomaly Detection API",
    description=(
        "API phát hiện anomaly trên network traffic sử dụng Variational Autoencoder "
        "huấn luyện trên bộ dữ liệu CICIDS2017. "
        "Upload CSV chứa network flow features để nhận phân loại normal/anomaly."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Cho phép React frontend (localhost:3000) gọi API trong môi trường development
# Trong production, thay thế "*" bằng domain cụ thể
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server mặc định
        "http://127.0.0.1:3000",
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Đăng ký routers ───────────────────────────────────────────────────────────
# Tất cả routes đều có prefix "/" — không dùng versioning trong MVP
app.include_router(predict_module.router, tags=["Prediction"])
app.include_router(upload_module.router, tags=["Upload"])
app.include_router(results_module.router, tags=["Results"])


# ── Custom exception handler: chuẩn hóa lỗi theo ErrorResponse schema ────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Trả về JSON theo ErrorResponse schema thay vì format mặc định {"detail": "..."}.
    Đảm bảo frontend luôn nhận được {"status": "error", "message": "..."}.
    """
    body = ErrorResponse(
        message=str(exc.detail),
        detail=f"HTTP {exc.status_code}",
    ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái dịch vụ",
    description="Xác nhận dịch vụ đang chạy và tất cả artifacts đã tải thành công.",
    tags=["Health"],
)
def health_check() -> HealthResponse:
    """
    Kiểm tra trạng thái sẵn sàng của dịch vụ và các artifacts.

    Kiểm tra xem VAEInferenceService singleton đã được khởi tạo thành công
    bằng cách truy cập trực tiếp vào module predict.

    Returns:
        HealthResponse với trạng thái dịch vụ và từng artifact.
    """
    svc = predict_module._inference_service

    if svc is not None:
        # Dịch vụ đã tải — kiểm tra từng component
        model_loaded = svc.model is not None
        scaler_loaded = svc.scaler is not None
        threshold_loaded = svc.threshold is not None
        overall_status = "ok" if (model_loaded and scaler_loaded and threshold_loaded) else "degraded"
    else:
        # Dịch vụ chưa tải — trả về trạng thái lỗi
        model_loaded = False
        scaler_loaded = False
        threshold_loaded = False
        overall_status = "error"

    logger.debug(
        "Health check: status=%s, model=%s, scaler=%s, threshold=%s",
        overall_status,
        model_loaded,
        scaler_loaded,
        threshold_loaded,
    )

    return HealthResponse(
        status=overall_status,
        model_loaded=model_loaded,
        scaler_loaded=scaler_loaded,
        threshold_loaded=threshold_loaded,
        service_name="NIDS VAE Anomaly Detection",
    )


# ── GET / (root) ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect hint cho root URL."""
    return {
        "message": "NIDS VAE API. Xem tài liệu tại /docs hoặc /redoc.",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "upload": "POST /upload",
            "results": "GET /results",
        },
    }
