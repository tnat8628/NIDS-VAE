"""
scripts/verify_vae_collapse.py
-------------------------------
Script kiểm tra nhanh xem model VAE có bị Posterior Collapse không.

Chạy sau mỗi lần training để xác nhận encoder và decoder hoạt động đúng.

Cách dùng:
    python scripts/verify_vae_collapse.py
    python scripts/verify_vae_collapse.py --checkpoint artifacts/models/vae_best.pth
    python scripts/verify_vae_collapse.py --threshold-mu-std 0.01

Tiêu chí đánh giá (mặc định):
    OK       : mu.std > 0.01 AND decoder_diff > 0.01
    WARNING  : mu.std > 0.001 OR decoder_diff > 0.001
    COLLAPSED: mu.std < 0.001 AND decoder_diff < 0.001

Exit code:
    0 = OK
    1 = WARNING
    2 = COLLAPSED
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Thiết lập đường dẫn gốc của project ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.vae import VAE  # noqa: E402

# ── Đường dẫn mặc định ───────────────────────────────────────────────────────
DEFAULT_CHECKPOINT   = PROJECT_ROOT / "artifacts" / "models" / "vae_best.pth"
DEFAULT_CONFIG       = PROJECT_ROOT / "artifacts" / "models" / "model_config.json"
DEFAULT_VAL_DATA     = PROJECT_ROOT / "data" / "validation" / "X_val.npy"
DEFAULT_FEATURE_SCHEMA = PROJECT_ROOT / "artifacts" / "feature_schema" / "feature_columns.json"

# ── Ngưỡng đánh giá ──────────────────────────────────────────────────────────
THRESHOLD_MU_STD_OK       = 0.01    # mu.std > này => encoder có thông tin
THRESHOLD_DECODER_DIFF_OK = 0.01    # decoder diff > này => decoder dùng z
THRESHOLD_MU_STD_WARN     = 0.001
THRESHOLD_DECODER_DIFF_WARN = 0.001


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Kiểm tra Posterior Collapse của model VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Đường dẫn đến file checkpoint .pth",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Đường dẫn đến model_config.json",
    )
    parser.add_argument(
        "--val-data", type=Path, default=DEFAULT_VAL_DATA,
        help="Đường dẫn đến X_val.npy",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000,
        help="Số mẫu val dùng để kiểm tra (dùng ít hơn để chạy nhanh hơn)",
    )
    return parser.parse_args()


def load_model(config_path: Path, checkpoint_path: Path) -> VAE:
    """Tải model VAE từ config và checkpoint."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    model = VAE(
        input_dim   = cfg["input_dim"],
        latent_dim  = cfg["latent_dim"],
        hidden_dims = cfg["hidden_dims"],
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def run_verification(args: argparse.Namespace) -> int:
    """
    Chạy toàn bộ kiểm tra collapse.
    Trả về exit code: 0=OK, 1=WARNING, 2=COLLAPSED.
    """
    print("=" * 70)
    print("VAE POSTERIOR COLLAPSE VERIFICATION")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Val data   : {args.val_data}")
    print()

    # ── Tải model ────────────────────────────────────────────────────────────
    model = load_model(args.config, args.checkpoint)
    print(f"Model loaded: input_dim={model.input_dim}, "
          f"latent_dim={model.latent_dim}, hidden_dims={model.hidden_dims}")

    # ── Tải dữ liệu val ──────────────────────────────────────────────────────
    X_val_np = np.load(args.val_data).astype(np.float32)
    # Giới hạn số mẫu nếu cần
    n = min(args.n_samples, len(X_val_np))
    X_val = torch.from_numpy(X_val_np[:n])
    print(f"Val samples used: {n:,} / {len(X_val_np):,}")
    print()

    # ── Chạy inference ───────────────────────────────────────────────────────
    with torch.no_grad():
        x_hat_all, mu_all, logvar_all = model(X_val)
        errors_all = ((x_hat_all - X_val) ** 2).mean(dim=1)

    # ── Test decoder sensitivity: decode(z1) vs decode(z2) ───────────────────
    torch.manual_seed(0)
    z1 = torch.randn(1000, model.latent_dim)
    z2 = torch.randn(1000, model.latent_dim)
    with torch.no_grad():
        out1 = model.decode(z1)
        out2 = model.decode(z2)
    decoder_diff = (out1 - out2).abs().mean().item()

    # ── Baseline MSE ─────────────────────────────────────────────────────────
    mse_baseline = (X_val ** 2).mean().item()
    mse_model    = ((X_val - x_hat_all) ** 2).mean().item()
    improvement  = (1.0 - mse_model / mse_baseline) * 100.0

    # ── In kết quả ───────────────────────────────────────────────────────────
    mu_std    = mu_all.std().item()
    mu_max    = mu_all.abs().max().item()
    lv_mean   = logvar_all.mean().item()
    lv_std    = logvar_all.std().item()
    xhat_mean = x_hat_all.abs().mean().item()
    xhat_std  = x_hat_all.std().item()

    print("─" * 70)
    print("LATENT SPACE (ENCODER)")
    print("─" * 70)
    print(f"  mu.mean()      = {mu_all.mean().item():.8f}")
    print(f"  mu.std()       = {mu_std:.8f}   {'✓' if mu_std > THRESHOLD_MU_STD_OK else ('!' if mu_std > THRESHOLD_MU_STD_WARN else '✗')}")
    print(f"  mu.abs().max() = {mu_max:.8f}")
    std_from_lv = torch.exp(0.5 * logvar_all)
    print(f"  logvar.mean()  = {lv_mean:.8f}")
    print(f"  logvar.std()   = {lv_std:.8f}")
    print(f"  std=exp(0.5*logvar): mean={std_from_lv.mean().item():.6f}  "
          f"min={std_from_lv.min().item():.6f}  max={std_from_lv.max().item():.6f}")
    print()
    print("  Per-dim mu.std() [should be > 0.01 for non-collapsed dims]:")
    mu_per_dim_std = mu_all.std(dim=0)
    for i, v in enumerate(mu_per_dim_std.tolist()):
        flag = "✓" if v > THRESHOLD_MU_STD_OK else ("!" if v > THRESHOLD_MU_STD_WARN else "✗")
        print(f"    dim {i:2d}: {v:.4e} {flag}")

    print()
    print("─" * 70)
    print("DECODER SENSITIVITY TO z")
    print("─" * 70)
    print(f"  mean|decode(z1) - decode(z2)| = {decoder_diff:.12f}   "
          f"{'✓' if decoder_diff > THRESHOLD_DECODER_DIFF_OK else ('!' if decoder_diff > THRESHOLD_DECODER_DIFF_WARN else '✗')}")

    print()
    print("─" * 70)
    print("RECONSTRUCTION OUTPUT (x_hat)")
    print("─" * 70)
    print(f"  mean(abs(x_hat)) = {xhat_mean:.8f}   {'✓' if xhat_std > 0.01 else '✗'}")
    print(f"  std(x_hat)       = {xhat_std:.8f}")
    per_sample_std = x_hat_all.std(dim=1)
    print(f"  per-sample std (mean) = {per_sample_std.mean().item():.8f}")
    print(f"  per-sample std (std)  = {per_sample_std.std().item():.8f}")

    print()
    print("─" * 70)
    print("RECONSTRUCTION ERROR DISTRIBUTION (val set)")
    print("─" * 70)
    pcts   = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    labels = ["min", "p10", "p25", "med", "p75", "p90", "p95", "p99", "max"]
    for pct, lab in zip(pcts, labels):
        v = torch.quantile(errors_all, pct).item()
        print(f"  {lab:4s} = {v:.6f}")
    print(f"  mean = {errors_all.mean().item():.6f}")
    print(f"  std  = {errors_all.std().item():.6f}")

    print()
    print("─" * 70)
    print("BASELINE COMPARISON")
    print("─" * 70)
    print(f"  MSE(x_val, 0)     = {mse_baseline:.8f}  [zero predictor]")
    print(f"  MSE(x_val, x_hat) = {mse_model:.8f}  [model]")
    print(f"  Improvement       = {improvement:.5f}%   {'✓' if improvement > 5.0 else ('!' if improvement > 0.1 else '✗')}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)

    is_collapsed = (
        mu_std < THRESHOLD_MU_STD_WARN
        and decoder_diff < THRESHOLD_DECODER_DIFF_WARN
    )
    is_ok = (
        mu_std > THRESHOLD_MU_STD_OK
        and decoder_diff > THRESHOLD_DECODER_DIFF_OK
    )

    if is_collapsed:
        verdict = "COLLAPSED"
        exit_code = 2
        detail = (
            "Encoder không mã hóa thông tin input. "
            "Decoder bỏ qua z. "
            "Model là constant predictor. "
            "=> Cần train lại với KL Annealing (beta_start=0, warmup=30 epochs)."
        )
    elif is_ok:
        verdict = "OK"
        exit_code = 0
        detail = (
            "Encoder mã hóa thông tin vào latent space. "
            "Decoder phụ thuộc z. "
            "Model đang học reconstruction tốt."
        )
    else:
        verdict = "WARNING"
        exit_code = 1
        detail = (
            "Một số dấu hiệu collapse nhưng chưa hoàn toàn. "
            "Có thể cần tăng beta_warmup_epochs hoặc giảm beta_end."
        )

    verdict_icon = {"OK": "✅", "WARNING": "⚠️ ", "COLLAPSED": "❌"}[verdict]
    print(f"VERDICT: {verdict_icon} {verdict}")
    print()
    print(f"  {detail}")
    print()
    print(f"  mu.std          = {mu_std:.4e}   (ngưỡng OK: > {THRESHOLD_MU_STD_OK})")
    print(f"  decoder diff    = {decoder_diff:.4e}   (ngưỡng OK: > {THRESHOLD_DECODER_DIFF_OK})")
    print(f"  improvement     = {improvement:.3f}%    (tốt nếu > 5%)")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_verification(args)
    sys.exit(exit_code)
