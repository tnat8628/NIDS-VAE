"""
backend/app/core/preprocessing.py
-----------------------------------
Module tiền xử lý dữ liệu đầu vào cho backend NIDS VAE.

Quy trình tiền xử lý phải khớp chính xác với quy trình huấn luyện:
  1. Xóa khoảng trắng tên cột
  2. Thay ±Inf bằng NaN
  3. Đảm bảo đủ 66 cột theo feature schema
  4. Thêm cột thiếu dưới dạng NaN, xóa cột thừa
  5. Sắp xếp lại cột theo đúng thứ tự schema
  6. Chuyển về numeric, impute NaN bằng median từ training
  7. Kiểm tra không còn NaN hoặc Inf
  8. Trả về DataFrame đã làm sạch (chưa scale)

LƯU Ý QUAN TRỌNG:
  - Không fit scaler mới trong backend.
  - Không tính median mới từ input của user.
  - Thứ tự cột đầu vào phải khớp với artifacts/feature_schema/feature_columns.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import FEATURE_SCHEMA_PATH, IMPUTATION_MEDIANS_PATH

logger = logging.getLogger(__name__)


# ── Ngoại lệ tùy chỉnh ───────────────────────────────────────────────────────

class PreprocessingError(Exception):
    """Lỗi xảy ra trong quá trình tiền xử lý đầu vào."""


class MissingArtifactError(PreprocessingError):
    """File artifact cần thiết không tồn tại."""


# ── Hàm tải artifact ─────────────────────────────────────────────────────────

def load_feature_schema(schema_path: Path = FEATURE_SCHEMA_PATH) -> list[str]:
    """
    Tải danh sách tên cột đặc trưng từ file JSON.

    Args:
        schema_path: Đường dẫn đến file feature_columns.json.

    Returns:
        Danh sách tên cột theo đúng thứ tự huấn luyện.

    Raises:
        MissingArtifactError: Nếu file không tồn tại.
        PreprocessingError: Nếu định dạng JSON không hợp lệ.
    """
    if not schema_path.exists():
        raise MissingArtifactError(
            f"Feature schema không tồn tại: {schema_path}. "
            "Hãy chạy scripts/export_artifacts.py để tạo artifact."
        )

    with schema_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ cả hai định dạng: {"feature_columns": [...]} hoặc [...]
    if isinstance(data, list):
        columns = data
    elif isinstance(data, dict) and "feature_columns" in data:
        columns = data["feature_columns"]
    else:
        raise PreprocessingError(
            f"Định dạng feature schema không hợp lệ trong {schema_path}. "
            "Cần có key 'feature_columns' hoặc là một danh sách."
        )

    if not columns:
        raise PreprocessingError("Feature schema trống — không có cột nào được định nghĩa.")

    logger.info("Đã tải feature schema: %d cột", len(columns))
    return columns


def load_imputation_medians(
    medians_path: Path = IMPUTATION_MEDIANS_PATH,
) -> dict[str, float]:
    """
    Tải giá trị median imputation từ tập train để xử lý NaN trong inference.

    Args:
        medians_path: Đường dẫn đến file imputation_medians.json.

    Returns:
        Dict ánh xạ tên cột → giá trị median.

    Raises:
        MissingArtifactError: Nếu file không tồn tại.
    """
    if not medians_path.exists():
        raise MissingArtifactError(
            f"Imputation medians không tồn tại: {medians_path}. "
            "Hãy chạy scripts/export_artifacts.py để tạo artifact."
        )

    with medians_path.open("r", encoding="utf-8") as f:
        medians = json.load(f)

    logger.info("Đã tải imputation medians: %d cột", len(medians))
    return medians


# ── Hàm tiền xử lý chính ─────────────────────────────────────────────────────

def preprocess_input_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    imputation_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Tiền xử lý DataFrame đầu vào để chuẩn bị cho inference VAE.

    Thực hiện đúng quy trình đã dùng trong training:
      1. Xóa khoảng trắng từ tên cột đầu vào
      2. Thay ±Inf bằng NaN
      3. Đảm bảo tất cả 66 cột cần thiết tồn tại
      4. Thêm cột thiếu với NaN, xóa cột không cần thiết
      5. Sắp xếp cột theo đúng thứ tự feature schema
      6. Chuyển tất cả cột về kiểu số
      7. Impute NaN bằng median từ tập train (không tính lại)
      8. Kiểm tra không còn NaN hoặc Inf

    Args:
        df: DataFrame thô từ CSV đầu vào của user.
        feature_columns: Danh sách cột theo thứ tự schema. Nếu None, tự tải từ artifact.
        imputation_medians: Dict median để impute. Nếu None, tự tải từ artifact.

    Returns:
        DataFrame đã làm sạch với đúng 66 cột, chưa scale, không NaN/Inf.

    Raises:
        PreprocessingError: Nếu đầu vào không hợp lệ hoặc quá trình xử lý thất bại.
        MissingArtifactError: Nếu artifact cần thiết không tồn tại.
    """
    # ── Bước 0: Kiểm tra đầu vào cơ bản ─────────────────────────────────────
    if df is None or not isinstance(df, pd.DataFrame):
        raise PreprocessingError("Đầu vào phải là pandas DataFrame.")

    if df.empty:
        raise PreprocessingError("DataFrame đầu vào rỗng — không có dữ liệu để xử lý.")

    # ── Bước 1: Tải artifact nếu chưa được cung cấp ──────────────────────────
    if feature_columns is None:
        feature_columns = load_feature_schema()

    if imputation_medians is None:
        imputation_medians = load_imputation_medians()

    # ── Bước 2: Làm sạch tên cột ─────────────────────────────────────────────
    # Xóa khoảng trắng đầu/cuối khỏi tên cột (lỗi phổ biến khi đọc CSV)
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    logger.debug("Đầu vào: %d hàng, %d cột", len(df), len(df.columns))

    # ── Bước 3: Thay ±Inf bằng NaN ───────────────────────────────────────────
    # Cần chuyển về float trước để replace hoạt động chính xác
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # ── Bước 4: Đảm bảo đủ cột theo feature schema ───────────────────────────
    existing_cols = set(df.columns)
    required_cols = set(feature_columns)

    # Tìm cột thiếu — sẽ được thêm vào với NaN và sau đó impute
    missing_cols = required_cols - existing_cols
    if missing_cols:
        logger.warning(
            "Thiếu %d cột trong đầu vào, sẽ impute bằng training medians: %s",
            len(missing_cols),
            sorted(missing_cols),
        )
        for col in missing_cols:
            df[col] = np.nan

    # Tìm cột thừa — sẽ bị loại bỏ
    extra_cols = existing_cols - required_cols
    if extra_cols:
        logger.debug("Xóa %d cột thừa không có trong feature schema.", len(extra_cols))
        df = df.drop(columns=list(extra_cols))

    # ── Bước 5: Sắp xếp lại cột theo đúng thứ tự schema ─────────────────────
    # Thứ tự cột PHẢI khớp chính xác với thứ tự khi fit scaler trong training
    df = df[feature_columns]

    # ── Bước 6: Chuyển tất cả cột về kiểu số ─────────────────────────────────
    # Giá trị không thể chuyển (chuỗi không phải số) sẽ trở thành NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Thay ±Inf có thể xuất hiện sau khi chuyển kiểu
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── Bước 7: Impute NaN bằng training medians ─────────────────────────────
    # Chỉ dùng median đã tính từ tập train — KHÔNG tính lại từ dữ liệu đầu vào
    for col in feature_columns:
        if df[col].isna().any():
            if col in imputation_medians:
                fill_value = imputation_medians[col]
                df[col] = df[col].fillna(fill_value)
                logger.debug("Impute cột '%s' bằng median = %s", col, fill_value)
            else:
                # Fallback: dùng 0 nếu không có median cho cột này
                logger.warning(
                    "Không tìm thấy median cho cột '%s', dùng 0 làm giá trị impute.", col
                )
                df[col] = df[col].fillna(0.0)

    # ── Bước 8: Kiểm tra an toàn cuối cùng ───────────────────────────────────
    # Đảm bảo không còn NaN hoặc Inf nào — đây là yêu cầu bắt buộc trước khi scale
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        raise PreprocessingError(
            f"Vẫn còn NaN sau khi impute trong các cột: {nan_cols.to_dict()}. "
            "Kiểm tra lại imputation_medians.json."
        )

    inf_mask = np.isinf(df.values)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        raise PreprocessingError(
            f"Vẫn còn {n_inf} giá trị Inf sau khi xử lý. "
            "Kiểm tra lại dữ liệu đầu vào."
        )

    logger.info(
        "Tiền xử lý hoàn tất: %d hàng × %d cột, không có NaN/Inf.",
        len(df),
        len(df.columns),
    )

    return df
