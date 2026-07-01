"""
backend/app/core/thresholding.py
----------------------------------
Module phân loại anomaly dựa trên ngưỡng reconstruction error.

Quy tắc phân loại:
  reconstruction_error > threshold  =>  anomaly (nhãn 1)
  reconstruction_error <= threshold =>  normal  (nhãn 0)

Ngưỡng được tải từ artifacts/threshold/threshold.json — KHÔNG hard-code
giá trị ngưỡng ở nhiều nơi, chỉ dùng file artifact này làm nguồn duy nhất.
"""

import json
import logging
from pathlib import Path

import numpy as np

from backend.app.config import THRESHOLD_PATH

logger = logging.getLogger(__name__)


# ── Ngoại lệ tùy chỉnh ───────────────────────────────────────────────────────

class ThresholdingError(Exception):
    """Lỗi liên quan đến tải hoặc áp dụng ngưỡng phân loại."""


class MissingThresholdError(ThresholdingError):
    """File threshold artifact không tồn tại."""


# ── Tải ngưỡng từ artifact ───────────────────────────────────────────────────

def load_threshold(threshold_path: Path = THRESHOLD_PATH) -> float:
    """
    Tải giá trị ngưỡng phân loại từ file JSON artifact.

    Ngưỡng này được đọc trực tiếp từ key "threshold" trong artifact.
    Percentile cụ thể (ví dụ P99) đến từ file JSON, không hard-code trong backend.

    Args:
        threshold_path: Đường dẫn đến file threshold.json.

    Returns:
        Giá trị ngưỡng (float) để so sánh với reconstruction error.

    Raises:
        MissingThresholdError: Nếu file không tồn tại.
        ThresholdingError: Nếu định dạng JSON không hợp lệ hoặc thiếu key 'threshold'.
    """
    if not threshold_path.exists():
        raise MissingThresholdError(
            f"Threshold artifact không tồn tại: {threshold_path}. "
            "Hãy chạy scripts/evaluate.py để tạo artifact."
        )

    raw_text = threshold_path.read_text(encoding="utf-8")
    if any(marker in raw_text for marker in ("<<<<<<<", "=======", ">>>>>>>")):
        raise ThresholdingError(
            f"File {threshold_path} chứa Git conflict markers. "
            "Hãy resolve artifact trước khi khởi động backend."
        )

    threshold_data = json.loads(raw_text)

    if "threshold" not in threshold_data:
        raise ThresholdingError(
            f"Thiếu key 'threshold' trong file {threshold_path}. "
            f"Các key hiện có: {list(threshold_data.keys())}"
        )

    threshold_value = float(threshold_data["threshold"])

    if not np.isfinite(threshold_value) or threshold_value <= 0:
        raise ThresholdingError(
            f"Giá trị threshold không hợp lệ: {threshold_value}. "
            "Threshold phải là số dương hữu hạn."
        )

    logger.info(
        "Đã tải threshold: %.6f (phương pháp: %s)",
        threshold_value,
        threshold_data.get("selection_method", "không xác định"),
    )

    return threshold_value


# ── Phân loại dựa trên ngưỡng ────────────────────────────────────────────────

def classify_errors(
    errors: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Phân loại từng mẫu là anomaly hoặc normal dựa trên reconstruction error.

    Quy tắc:
      error > threshold  =>  1 (anomaly)
      error <= threshold =>  0 (normal)

    Args:
        errors: Mảng reconstruction error cho từng mẫu, shape (n_samples,).
        threshold: Ngưỡng phân loại (từ load_threshold()).

    Returns:
        Mảng nhãn nhị phân int, shape (n_samples,):
          1 = anomaly, 0 = normal.

    Raises:
        ThresholdingError: Nếu errors chứa NaN hoặc Inf.
    """
    errors = np.asarray(errors, dtype=np.float64)

    # ── Kiểm tra đầu vào ─────────────────────────────────────────────────────
    if not np.isfinite(errors).all():
        n_bad = (~np.isfinite(errors)).sum()
        raise ThresholdingError(
            f"Mảng errors chứa {n_bad} giá trị NaN/Inf. "
            "Kiểm tra lại quá trình tính reconstruction error."
        )

    if not np.isfinite(threshold) or threshold <= 0:
        raise ThresholdingError(
            f"Threshold không hợp lệ: {threshold}. Phải là số dương hữu hạn."
        )

    # ── Phân loại: error > threshold => 1 (anomaly), ngược lại => 0 (normal) ─
    predictions = (errors > threshold).astype(np.int32)

    n_anomaly = int(predictions.sum())
    n_total = len(predictions)
    logger.debug(
        "Phân loại: %d anomaly / %d normal trong %d mẫu (ngưỡng=%.6f)",
        n_anomaly,
        n_total - n_anomaly,
        n_total,
        threshold,
    )

    return predictions
