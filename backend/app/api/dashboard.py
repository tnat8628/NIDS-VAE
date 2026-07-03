"""Database-backed aggregate endpoint for the Overview dashboard."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.response_schema import (
    DashboardOverviewResponse,
    GlobalFlowListResponse,
)
from backend.app.services.dashboard_storage import get_dashboard_overview
from backend.app.services.upload_management import PredictionFilter, list_global_flows

router = APIRouter()


@router.get(
    "/dashboard/overview",
    response_model=DashboardOverviewResponse,
    summary="Thống kê tổng hợp toàn hệ thống",
    description=(
        "Aggregate trực tiếp từ PostgreSQL. Mỗi upload chỉ đóng góp latest "
        "persisted inference run để không double count khi prediction được chạy lại."
    ),
)
def dashboard_overview(
    db: Session = Depends(get_db),
) -> DashboardOverviewResponse:
    return DashboardOverviewResponse.model_validate(get_dashboard_overview(db))


@router.get(
    "/dashboard/flows",
    response_model=GlobalFlowListResponse,
    summary="Danh sÃ¡ch flow toÃ n há»‡ thá»‘ng",
    description=(
        "PhÃ¢n trang tá»« PostgreSQL vÃ  chá»‰ láº¥y flow_predictions thuá»™c latest "
        "persisted inference run cá»§a má»—i upload Ä‘á»ƒ trÃ¡nh double count."
    ),
)
def dashboard_flows(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    prediction: PredictionFilter = Query("all"),
    db: Session = Depends(get_db),
) -> GlobalFlowListResponse:
    return GlobalFlowListResponse.model_validate(
        list_global_flows(
            db,
            page=page,
            page_size=page_size,
            prediction_filter=prediction,
        )
    )