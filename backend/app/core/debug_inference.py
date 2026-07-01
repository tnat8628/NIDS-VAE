"""
backend/app/core/debug_inference.py
-------------------------------------
Module debug để quan sát chi tiết từng bước trong pipeline inference VAE.

Chỉ được kích hoạt khi DEBUG_INFERENCE=true trong biến môi trường.
Không ảnh hưởng đến logic predict và response API khi debug tắt.

Các bước được log:
  [STEP 1] Read CSV        — shape và danh sách cột đầu vào
  [STEP 2] Preprocessing   — shape sau khi chuẩn hóa và impute
  [STEP 3] Scaling         — shape và mẫu 5 giá trị đầu sau StandardScaler
  [STEP 4] VAE Encoder     — shape mu, logvar và giá trị mẫu
  [STEP 5] Latent Vector   — shape z và giá trị mẫu
  [STEP 6] Decoder Reconstruction — shape x_hat và giá trị mẫu
  [STEP 7] Reconstruction Error   — giá trị error của dòng đầu tiên
  [STEP 8] Threshold Classification — so sánh error với threshold

Cách sử dụng:
  Gọi debug_single_sample_flow(service, df_raw) sau khi đọc CSV nhưng trước
  khi gọi predict_dataframe() hàng loạt — hàm này chạy độc lập, không can
  thiệp vào batch inference chính.
  $env:DEBUG_INFERENCE="true"
uvicorn backend.app.main:app --reload
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    # Import kiểu dữ liệu chỉ khi type-checking để tránh circular import
    from backend.app.core.inference import VAEInferenceService

logger = logging.getLogger(__name__)

# Separator dùng cho tiêu đề mỗi bước — dễ đọc trong log output
_SEP = "=" * 60


def _log_step(step: int, title: str, lines: list[str]) -> None:
    """
    In log theo định dạng chuẩn với tiêu đề và danh sách nội dung.

    Args:
        step  : Số thứ tự bước (1–8).
        title : Tiêu đề mô tả bước.
        lines : Danh sách dòng nội dung cần log.
    """
    logger.debug(_SEP)
    logger.debug("[STEP %d] %s", step, title)
    logger.debug(_SEP)
    for line in lines:
        logger.debug("  %s", line)


def debug_single_sample_flow(
    service: "VAEInferenceService",
    df_raw: pd.DataFrame,
) -> None:
    """
    Chạy toàn bộ pipeline inference cho **dòng đầu tiên** của DataFrame và
    log chi tiết từng bước. Không thay đổi kết quả batch inference chính.

    Hàm này được thiết kế để gọi song song với predict_dataframe() khi
    DEBUG_INFERENCE=true — nó chạy độc lập trên 1 mẫu duy nhất.

    Args:
        service : Instance của VAEInferenceService đã khởi tạo xong.
        df_raw  : DataFrame thô nguyên bản từ CSV (chưa qua bất kỳ xử lý nào).
    """
    logger.debug("\n")
    logger.debug("%" * 60)
    logger.debug("   DEBUG INFERENCE — SINGLE SAMPLE WALKTHROUGH")
    logger.debug("%" * 60)

    # ── STEP 1: Đọc CSV — quan sát shape và cột đầu vào ──────────────────────
    # Mục đích: kiểm tra xem CSV có đúng số cột và đúng tên cột không
    _log_step(1, "Read CSV", [
        f"Shape DataFrame gốc          : {df_raw.shape}  "
        f"({df_raw.shape[0]:,} dòng x {df_raw.shape[1]} cột)",
        f"Số cột kỳ vọng (feature schema): {len(service.feature_columns)}",
        f"Các cột đầu vào (10 cột đầu) : {list(df_raw.columns[:10])}",
        f"Các cột đầu vào (10 cột cuối): {list(df_raw.columns[-10:])}",
    ])

    # ── STEP 2: Preprocessing — chuẩn hóa cột, impute NaN ───────────────────
    # Mục đích: đảm bảo 66 cột đúng schema, không còn NaN/Inf
    from backend.app.core.preprocessing import preprocess_input_dataframe

    df_clean = preprocess_input_dataframe(
        df=df_raw,
        feature_columns=service.feature_columns,
        imputation_medians=service.imputation_medians,
    )

    # Lấy dòng đầu tiên để theo dõi xuyên suốt
    first_row_clean = df_clean.iloc[0]

    _log_step(2, "Preprocessing", [
        f"Shape sau preprocessing       : {df_clean.shape}",
        f"NaN còn lại                   : {df_clean.isnull().sum().sum()}",
        f"5 giá trị đầu tiên của dòng 0 : {first_row_clean.values[:5].tolist()}",
        f"  (feature names)             : {list(df_clean.columns[:5])}",
    ])

    # ── STEP 3: Scaling — StandardScaler transform ───────────────────────────
    # Mục đích: đưa dữ liệu về cùng phân phối với tập train (mean=0, std=1)
    X_scaled = service.scaler.transform(df_clean)
    first_row_scaled = X_scaled[0]

    _log_step(3, "Scaling (StandardScaler)", [
        f"Shape sau scaler              : {X_scaled.shape}",
        f"5 giá trị đầu tiên dòng 0     : {np.round(first_row_scaled[:5], 6).tolist()}",
        f"  (mean≈0, std≈1 sau scaling)",
        f"Min toàn batch                : {X_scaled.min():.4f}",
        f"Max toàn batch                : {X_scaled.max():.4f}",
    ])

    # ── STEP 4: VAE Encoder — encode dòng đầu tiên qua encoder ──────────────
    # Mục đích: quan sát không gian tiềm ẩn — mu (trung bình) và logvar (phương sai)
    sample_tensor = torch.tensor(
        first_row_scaled.reshape(1, -1),
        dtype=torch.float32,
        device=service.device,
    )

    with torch.no_grad():
        # Chạy qua encoder để lấy mu và logvar
        # h: (1, hidden_dim_cuoi) → mu, logvar: (1, latent_dim)
        h = service.model.encoder(sample_tensor)
        mu = service.model.fc_mu(h)
        logvar = service.model.fc_logvar(h)

    mu_vals = mu.cpu().numpy()[0]
    logvar_vals = logvar.cpu().numpy()[0]

    _log_step(4, "VAE Encoder → (mu, logvar)", [
        f"Shape tensor đầu vào          : {tuple(sample_tensor.shape)}  "
        f"= (1 mẫu x {sample_tensor.shape[1]} features)",
        f"Shape mu                      : {tuple(mu.shape)}  "
        f"= (1 mẫu x {mu.shape[1]} chiều latent)",
        f"Shape logvar                  : {tuple(logvar.shape)}",
        f"mu[:5]     (trung bình phân phối tiềm ẩn): "
        f"{np.round(mu_vals[:5], 6).tolist()}",
        f"logvar[:5] (log phương sai — đo độ không chắc): "
        f"{np.round(logvar_vals[:5], 6).tolist()}",
    ])

    # ── STEP 5: Latent Vector z — reparameterize ─────────────────────────────
    # Mục đích: quan sát vector z đại diện cho flow trong không gian tiềm ẩn
    # Trong inference (eval mode): z = mu (deterministic, không lấy mẫu ngẫu nhiên)
    with torch.no_grad():
        z = service.model.reparameterize(mu, logvar)

    z_vals = z.cpu().numpy()[0]

    _log_step(5, "Latent Vector z (reparameterize)", [
        f"Shape z                       : {tuple(z.shape)}  "
        f"= (1 mẫu x {z.shape[1]} chiều latent)",
        f"z[:5] (= mu[:5] vì đang ở eval mode): "
        f"{np.round(z_vals[:5], 6).tolist()}",
        f"  → z dùng để đại diện ngắn gọn cho 66 features thành {z.shape[1]} số",
    ])

    # ── STEP 6: Decoder Reconstruction — giải nén z về x_hat ─────────────────
    # Mục đích: xem VAE tái tạo lại dữ liệu như thế nào từ z
    with torch.no_grad():
        x_hat = service.model.decode(z)

    x_hat_vals = x_hat.cpu().numpy()[0]

    _log_step(6, "Decoder Reconstruction → x_hat", [
        f"Shape x_hat                   : {tuple(x_hat.shape)}  "
        f"= (1 mẫu x {x_hat.shape[1]} features — khớp với input gốc)",
        f"x_hat[:5]  (giá trị tái tạo)  : {np.round(x_hat_vals[:5], 6).tolist()}",
        f"x_orig[:5] (giá trị gốc)      : {np.round(first_row_scaled[:5], 6).tolist()}",
        f"  → So sánh x_orig và x_hat để thấy mức độ tái tạo",
    ])

    # ── STEP 7: Reconstruction Error — tính MSE per-sample ───────────────────
    # Mục đích: đây là "anomaly score" — càng lớn càng bất thường
    # Công thức: error = mean((x - x_hat)^2) theo chiều features
    x_orig_tensor = sample_tensor
    with torch.no_grad():
        error_tensor = torch.nn.functional.mse_loss(
            x_hat, x_orig_tensor, reduction="none"
        ).mean(dim=1)

    error_val = float(error_tensor.cpu().numpy()[0])
    diff_vals = np.abs(x_hat_vals[:5] - first_row_scaled[:5])

    _log_step(7, "Reconstruction Error (MSE per-sample)", [
        f"Công thức: error = mean((x - x_hat)²) qua {x_hat.shape[1]} features",
        f"Reconstruction error dòng 0   : {error_val:.6f}",
        f"|x - x_hat|[:5] (sai lệch tuyệt đối 5 feat đầu): "
        f"{np.round(diff_vals, 6).tolist()}",
    ])

    # ── STEP 8: Threshold Classification — so sánh error với ngưỡng ──────────
    # Mục đích: quyết định cuối cùng là normal hay anomaly
    # Quy tắc: error > threshold → anomaly (1), ngược lại → normal (0)
    is_anomaly = error_val > service.threshold
    label = "ANOMALY ⚠" if is_anomaly else "NORMAL ✓"
    margin = error_val - service.threshold

    _log_step(8, "Threshold Classification", [
        f"Threshold (ngưỡng từ artifact): {service.threshold:.6f}",
        f"Reconstruction error dòng 0   : {error_val:.6f}",
        f"error > threshold?            : {error_val:.6f} > {service.threshold:.6f} "
        f"→ {is_anomaly}",
        f"Kết quả phân loại             : {label}",
        f"Khoảng cách đến ngưỡng        : {margin:+.6f}  "
        f"({'vượt ngưỡng' if is_anomaly else 'dưới ngưỡng'})",
    ])

    logger.debug(_SEP)
    logger.debug("   KẾT THÚC DEBUG — toàn bộ batch vẫn xử lý bình thường")
    logger.debug(_SEP)
    logger.debug("\n")
