"""
Kiểm tra nhanh các artifact runtime mà backend sử dụng khi suy luận.

Script này không load model vào GPU/CPU để predict; nó chỉ xác nhận đường dẫn,
JSON artifact và các giá trị cấu hình quan trọng để phát hiện lệch artifact sớm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (  # noqa: E402
    FEATURE_SCHEMA_PATH,
    IMPUTATION_MEDIANS_PATH,
    MODEL_CHECKPOINT_PATH,
    MODEL_CONFIG_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
)


ARTIFACT_PATHS = {
    "model_checkpoint": MODEL_CHECKPOINT_PATH,
    "model_config": MODEL_CONFIG_PATH,
    "scaler": SCALER_PATH,
    "imputation_medians": IMPUTATION_MEDIANS_PATH,
    "feature_schema": FEATURE_SCHEMA_PATH,
    "threshold": THRESHOLD_PATH,
}


def read_json_artifact(path: Path) -> dict[str, Any]:
    """Đọc JSON và chặn artifact còn Git conflict markers."""
    raw_text = path.read_text(encoding="utf-8")
    if any(marker in raw_text for marker in ("<<<<<<<", "=======", ">>>>>>>")):
        raise ValueError(f"{path} con Git conflict markers")
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} phai la JSON object")
    return data


def path_status(path: Path) -> dict[str, Any]:
    """Trả về trạng thái tồn tại của từng artifact."""
    return {
        "path": str(path),
        "exists": path.exists(),
    }


def build_report() -> dict[str, Any]:
    """Tổng hợp những giá trị backend sẽ dùng lúc khởi động inference."""
    artifact_status = {
        name: path_status(path)
        for name, path in ARTIFACT_PATHS.items()
    }

    missing = [
        name
        for name, status in artifact_status.items()
        if not status["exists"]
    ]

    if missing:
        return {
            "status": "error",
            "reason": "missing_artifacts",
            "missing": missing,
            "artifacts": artifact_status,
        }

    model_config = read_json_artifact(MODEL_CONFIG_PATH)
    threshold_data = read_json_artifact(THRESHOLD_PATH)
    feature_schema = read_json_artifact(FEATURE_SCHEMA_PATH)

    return {
        "status": "ok",
        "runtime_artifacts": artifact_status,
        "model": {
            "checkpoint_path": str(MODEL_CHECKPOINT_PATH),
            "config_path": str(MODEL_CONFIG_PATH),
            "input_dim": model_config.get("input_dim"),
            "latent_dim": model_config.get("latent_dim"),
            "hidden_dims": model_config.get("hidden_dims"),
        },
        "features": {
            "schema_path": str(FEATURE_SCHEMA_PATH),
            "n_features": feature_schema.get("n_features"),
        },
        "preprocessing": {
            "scaler_path": str(SCALER_PATH),
            "imputation_medians_path": str(IMPUTATION_MEDIANS_PATH),
        },
        "threshold": {
            "path": str(THRESHOLD_PATH),
            "threshold": threshold_data.get("threshold"),
            "percentile": threshold_data.get("percentile"),
            "selection_method": threshold_data.get("selection_method"),
        },
    }


def main() -> int:
    try:
        report = build_report()
    except Exception as exc:
        report = {
            "status": "error",
            "reason": type(exc).__name__,
            "message": str(exc),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
