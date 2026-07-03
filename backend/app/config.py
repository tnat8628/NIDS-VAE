"""
backend/app/config.py
----------------------
Cấu hình tập trung cho backend FastAPI NIDS VAE.

Định nghĩa tất cả đường dẫn artifact và các hằng số suy diễn.
Tất cả path đều được xây dựng tương đối từ project root để dễ di chuyển.
"""

import os
from pathlib import Path

# ── Gốc project ──────────────────────────────────────────────────────────────
# PROJECT_ROOT là thư mục chứa toàn bộ project (nids-vae-project/)
# File này nằm tại: backend/app/config.py
#   parents[0] = backend/app
#   parents[1] = backend
#   parents[2] = nids-vae-project  <-- project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Thư mục artifact chính ───────────────────────────────────────────────────
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
DATA_DIR: Path = PROJECT_ROOT / "data"

# PostgreSQL connection used by SQLAlchemy and Alembic.
# Docker Compose overrides this default with hostname "db".
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://nids_vae:nids_vae_dev@localhost:5432/nids_vae",
)

# ── Đường dẫn Model ──────────────────────────────────────────────────────────
# File checkpoint PyTorch tốt nhất từ quá trình huấn luyện
MODEL_CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "models" / "vae_best.pth"

# Cấu hình kiến trúc mô hình (input_dim, latent_dim, hidden_dims, ...)
MODEL_CONFIG_PATH: Path = ARTIFACTS_DIR / "models" / "model_config.json"

# ── Đường dẫn Scaler ─────────────────────────────────────────────────────────
# StandardScaler đã được fit trên tập train — không được fit lại trong backend
SCALER_PATH: Path = ARTIFACTS_DIR / "scaler" / "scaler.joblib"

# Giá trị median đã tính từ tập train để impute NaN — không tính lại từ input
IMPUTATION_MEDIANS_PATH: Path = ARTIFACTS_DIR / "scaler" / "imputation_medians.json"

# ── Đường dẫn Threshold ──────────────────────────────────────────────────────
# Ngưỡng phân loại anomaly: reconstruction_error > threshold => anomaly
THRESHOLD_PATH: Path = ARTIFACTS_DIR / "threshold" / "threshold.json"

# ── Đường dẫn Feature Schema ─────────────────────────────────────────────────
# Thứ tự cột đặc trưng chuẩn (66 features) — phải khớp với quá trình huấn luyện
FEATURE_SCHEMA_PATH: Path = ARTIFACTS_DIR / "feature_schema" / "feature_columns.json"

# Đường dẫn thay thế trong data/processed/ (dùng khi artifact chưa được export)
# LƯU Ý: Fallback này chưa được implement trong preprocessing.py — chỉ để tham khảo
# FEATURE_SCHEMA_FALLBACK_PATH: Path = DATA_DIR / "processed" / "feature_names.json"

# ── Tham số suy diễn mặc định ────────────────────────────────────────────────
# Kích thước batch mặc định cho xử lý CSV lớn — cân bằng bộ nhớ và hiệu suất
DEFAULT_BATCH_SIZE: int = 4096

# Giới hạn kích thước file upload — ngăn chặn DoS qua CSV cực lớn
# 100 MB là đủ cho ~1 triệu flow CICIDS2017 (mỗi flow ~100 bytes CSV)
MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024  # 100 MB

# ── Sample batch để kiểm tra ─────────────────────────────────────────────────
SAMPLE_BATCH_PATH: Path = ARTIFACTS_DIR / "sample_batch" / "fixed_batch.csv"

# ── Chế độ Debug Inference ────────────────────────────────────────────────────
# Khi DEBUG_INFERENCE=True, hệ thống sẽ log chi tiết từng bước trong pipeline
# inference cho mẫu đầu tiên của mỗi batch: preprocessing → scaler → encoder
# → latent z → decoder → reconstruction error → phân loại threshold.
#
# Cách bật:  đặt biến môi trường DEBUG_INFERENCE=true  (trong .env hoặc shell)
# Cách tắt:  DEBUG_INFERENCE=false  (mặc định)
#
# QUAN TRỌNG: Chỉ bật khi phát triển/debug — tắt trước khi deploy production
# vì mỗi request sẽ ghi thêm ~30 dòng log.
DEBUG_INFERENCE: bool = os.getenv("DEBUG_INFERENCE", "false").lower() in ("1", "true", "yes")