"""
backend/app/schemas/response_schema.py
----------------------------------------
Pydantic schemas cho tất cả các response của FastAPI NIDS VAE.

Schemas:
  - FlowPrediction     : Kết quả dự đoán cho một network flow
  - PredictionSummary  : Tóm tắt tổng hợp toàn bộ batch
  - PredictionResponse : Response đầy đủ cho POST /predict
  - HealthResponse     : Response cho GET /health
  - ErrorResponse      : Response lỗi chuẩn hóa

Schemas này là nguồn duy nhất định nghĩa cấu trúc JSON trả về — không
được thay đổi field name mà không cập nhật frontend và docs/api-spec.md.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── Per-flow prediction ───────────────────────────────────────────────────────

class FlowPrediction(BaseModel):
    """
    Kết quả dự đoán cho một network flow đơn lẻ.

    Attributes:
        row_index            : Chỉ số hàng gốc trong DataFrame đầu vào.
        reconstruction_error : Giá trị MSE reconstruction error của VAE.
        prediction           : Nhãn số (0 = normal, 1 = anomaly).
        prediction_label     : Nhãn chuỗi ("normal" hoặc "anomaly").
    """

    row_index: int = Field(..., description="Chỉ số hàng gốc trong CSV đầu vào")
    reconstruction_error: float = Field(
        ..., description="MSE reconstruction error của VAE cho flow này"
    )
    prediction: int = Field(..., description="0 = normal, 1 = anomaly")
    prediction_label: str = Field(..., description="'normal' hoặc 'anomaly'")


# ── Summary ───────────────────────────────────────────────────────────────────

class PredictionSummary(BaseModel):
    """
    Tóm tắt thống kê cho toàn bộ batch dự đoán.

    Attributes:
        total_flows   : Tổng số flow đã xử lý.
        anomaly_count : Số flow bị phân loại là anomaly.
        normal_count  : Số flow bình thường.
        anomaly_rate  : Tỷ lệ anomaly (0.0 đến 1.0).
        threshold     : Ngưỡng phân loại đã được sử dụng.
    """

    total_flows: int = Field(..., description="Tổng số network flows đã xử lý")
    anomaly_count: int = Field(..., description="Số flows bị phân loại là anomaly")
    normal_count: int = Field(..., description="Số flows bình thường")
    anomaly_rate: float = Field(..., description="Tỷ lệ anomaly (0.0 đến 1.0)")
    threshold: float = Field(..., description="Ngưỡng reconstruction error đã dùng")


# ── Full prediction response ──────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Response đầy đủ cho POST /predict.

    Bao gồm thông tin trạng thái, tóm tắt và danh sách chi tiết từng flow.

    Attributes:
        status  : Trạng thái xử lý ("ok" khi thành công).
        summary : Tóm tắt thống kê batch.
        results : Danh sách kết quả chi tiết từng flow.
    """

    status: str = Field(default="ok", description="Trạng thái xử lý")
    summary: PredictionSummary = Field(..., description="Thống kê tóm tắt toàn batch")
    results: list[FlowPrediction] = Field(
        ..., description="Kết quả chi tiết từng network flow"
    )


class PredictionRunResponse(BaseModel):
    """Compact response after predicting a persisted upload."""

    status: str = Field(default="ok")
    upload_id: UUID
    inference_run_id: UUID
    summary: PredictionSummary
    results_url: str


class StoredFlowPrediction(FlowPrediction):
    """One paginated database-backed prediction row."""

    severity: Literal["low", "medium", "high", "critical"]
    actual_label: str | int | float | bool | None = None
    actual_binary: Literal[0, 1] | None = None


class PaginationResponse(BaseModel):
    """Server-side pagination metadata."""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_previous: bool
    has_next: bool


class HistogramBinResponse(BaseModel):
    """One aggregate reconstruction-error histogram bin."""

    bin: str
    normal: int
    anomaly: int


class PredictionAggregatesResponse(BaseModel):
    """Bounded aggregates computed from the complete inference run."""

    histogram: list[HistogramBinResponse]
    top_anomalies: list[StoredFlowPrediction]


class PaginatedPredictionResponse(BaseModel):
    """Database-backed prediction page for one upload."""

    status: str = Field(default="ok")
    upload_id: UUID
    inference_run_id: UUID
    summary: PredictionSummary
    items: list[StoredFlowPrediction]
    pagination: PaginationResponse
    aggregates: PredictionAggregatesResponse


class DashboardUploadAggregate(BaseModel):
    total_uploads: int
    total_uploaded_flows: int


class DashboardAnalysisAggregate(BaseModel):
    analyzed_uploads: int
    total_analyzed_flows: int
    anomaly_count: int
    normal_count: int
    anomaly_rate: float


class DashboardClassification(BaseModel):
    normal: int
    anomaly: int


class DashboardLatestActivity(BaseModel):
    latest_upload_id: UUID | None
    latest_run_id: UUID | None
    latest_filename: str | None
    latest_uploaded_at: datetime | None
    latest_predicted_at: datetime | None


class DashboardOverviewResponse(BaseModel):
    """System-wide aggregates, independent from the Results page context."""

    status: str = Field(default="ok")
    uploads: DashboardUploadAggregate
    analysis: DashboardAnalysisAggregate
    histogram: list[HistogramBinResponse]
    classification: DashboardClassification
    latest_activity: DashboardLatestActivity


# ── Health response ───────────────────────────────────────────────────────────

class UploadListItemResponse(BaseModel):
    """One CSV upload row for the upload management page."""

    upload_id: UUID
    filename: str
    row_count: int
    col_count: int
    created_at: datetime
    analysis_status: Literal["pending", "completed"]
    latest_run_id: UUID | None
    latest_predicted_at: datetime | None
    anomaly_count: int
    normal_count: int


class UploadListResponse(BaseModel):
    """Paginated CSV upload management response."""

    status: str = Field(default="ok")
    items: list[UploadListItemResponse]
    pagination: PaginationResponse


class DeleteUploadResponse(BaseModel):
    """Response returned after deleting one upload and all dependent data."""

    status: str = Field(default="ok")
    upload_id: UUID
    message: str


class GlobalFlowItemResponse(BaseModel):
    """One globally browsable prediction row from the latest run of an upload."""

    upload_id: UUID
    filename: str
    run_id: UUID
    row_index: int
    reconstruction_error: float
    prediction: int
    prediction_label: Literal["normal", "anomaly"]
    created_at: datetime


class GlobalFlowListResponse(BaseModel):
    """Database-backed global flow explorer response."""

    status: str = Field(default="ok")
    items: list[GlobalFlowItemResponse]
    pagination: PaginationResponse


class HealthResponse(BaseModel):
    """
    Response cho GET /health.

    Kiểm tra xem dịch vụ và tất cả artifacts đã được tải thành công chưa.

    Attributes:
        status           : "ok" nếu dịch vụ sẵn sàng, "error" nếu có vấn đề.
        model_loaded     : True nếu VAE checkpoint đã tải thành công.
        scaler_loaded    : True nếu StandardScaler đã tải thành công.
        threshold_loaded : True nếu ngưỡng phân loại đã tải thành công.
        service_name     : Tên dịch vụ.
    """

    status: str = Field(..., description="'ok' hoặc 'error'")
    model_loaded: bool = Field(..., description="VAE model đã tải thành công")
    scaler_loaded: bool = Field(..., description="StandardScaler đã tải thành công")
    threshold_loaded: bool = Field(..., description="Threshold đã tải thành công")
    service_name: str = Field(
        default="NIDS VAE Anomaly Detection", description="Tên dịch vụ"
    )


# ── Error response ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Response lỗi chuẩn hóa cho tất cả HTTP error cases.

    Attributes:
        status  : Luôn là "error".
        message : Mô tả lỗi ngắn gọn.
        detail  : Chi tiết bổ sung nếu có (tùy chọn).
    """

    status: str = Field(default="error", description="Luôn là 'error'")
    message: str = Field(..., description="Mô tả lỗi ngắn gọn")
    detail: str | None = Field(default=None, description="Chi tiết bổ sung về lỗi")


# ── Upload response ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """
    Response cho POST /upload.

    Attributes:
        status     : "ok" nếu upload hợp lệ.
        upload_id  : UUID của upload đã lưu trong database.
        filename   : Tên file đã tải lên.
        row_count  : Số hàng đọc được từ CSV.
        col_count  : Số cột đọc được từ CSV.
        message    : Hướng dẫn tiếp theo.
    """

    status: str = Field(default="ok")
    upload_id: UUID = Field(..., description="UUID của upload đã lưu")
    filename: str = Field(..., description="Tên file đã upload")
    row_count: int = Field(..., description="Số hàng trong CSV")
    col_count: int = Field(..., description="Số cột trong CSV")
    message: str = Field(..., description="Hướng dẫn bước tiếp theo")