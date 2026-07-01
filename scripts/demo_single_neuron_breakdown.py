"""
scripts/demo_single_neuron_breakdown.py
----------------------------------------
Demo phân tích chi tiết cách một neuron trong lớp Encoder đầu tiên được tính toán.

Mục tiêu: hiểu từng bước của công thức
    h1[0] = ReLU( W1[0,:] · x + b1[0] )

Quy trình:
  1. Load 1 flow thực tế từ artifacts/sample_batch/fixed_batch.csv + scaler.joblib
  2. Load model đã train từ artifacts/models/vae_best.pth (nếu không có thì dùng model demo)
  3. Giải phẫu neuron #0 của lớp fc1: liệt kê W[i]*x[i] cho cả 66 feature
  4. Xếp hạng 10 feature đóng góp lớn nhất theo |W*x|
  5. Tính thủ công từng bước: Σ(W*x), bias, pre_activation
  6. Áp dụng ReLU và so sánh với output thực tế của PyTorch
  7. Bonus: in top-5 neuron có pre_activation lớn nhất trong toàn bộ lớp h1

Chạy:
    python scripts/demo_single_neuron_breakdown.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# ĐƯỜNG DẪN ARTIFACT
# ──────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
FEATURE_PATH  = ROOT / "artifacts" / "feature_schema" / "feature_columns.json"
SCALER_PATH   = ROOT / "artifacts" / "scaler" / "scaler.joblib"
MEDIANS_PATH  = ROOT / "artifacts" / "scaler" / "imputation_medians.json"
SAMPLE_PATH   = ROOT / "artifacts" / "sample_batch" / "fixed_batch.csv"
MODEL_PATH    = ROOT / "artifacts" / "models" / "vae_best.pth"
MODEL_CFG     = ROOT / "artifacts" / "models" / "model_config.json"

# ──────────────────────────────────────────────────────────────────────────────
# KIẾN TRÚC VAE (bản sao gọn, khớp với backend/app/models/vae.py)
# ──────────────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    """Variational Autoencoder dùng cho NIDS — kiến trúc khớp với vae_best.pth."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        # Encoder: input_dim → hidden_dims
        enc = []
        in_f = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        self.encoder   = nn.Sequential(*enc)
        self.fc_mu     = nn.Linear(in_f, latent_dim)
        self.fc_logvar = nn.Linear(in_f, latent_dim)
        # Decoder: latent_dim → reversed(hidden_dims) → input_dim
        dec = []
        in_f = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        dec.append(nn.Linear(in_f, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        h      = self.encoder(x)
        mu, lv = self.fc_mu(h), self.fc_logvar(h)
        z      = mu  # inference: dùng mu để deterministic
        return self.decoder(z), mu, lv


# ──────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ HIỂN THỊ
# ──────────────────────────────────────────────────────────────────────────────
SEP  = "=" * 66
SEP2 = "-" * 66


# ──────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ──────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    """Đọc cấu hình kiến trúc model từ model_config.json."""
    with open(MODEL_CFG) as f:
        return json.load(f)


def load_feature_names() -> list[str]:
    """Đọc danh sách tên 66 feature theo thứ tự chuẩn."""
    with open(FEATURE_PATH) as f:
        return json.load(f)["feature_columns"]


def load_medians() -> dict:
    """Đọc giá trị median để impute NaN."""
    with open(MEDIANS_PATH) as f:
        return json.load(f)


def load_and_preprocess_flow(feature_names: list[str], medians: dict) -> np.ndarray:
    """
    Đọc flow đầu tiên từ fixed_batch.csv, impute NaN bằng median,
    rồi scale bằng scaler.joblib đã fit lúc training.
    Trả về mảng numpy shape (66,) — đã scale, sẵn sàng đưa vào model.
    """
    df = pd.read_csv(SAMPLE_PATH, nrows=1)

    # Đảm bảo đúng 66 cột theo thứ tự chuẩn
    df = df.reindex(columns=feature_names)

    # Thay Inf bằng NaN rồi impute bằng median của training set
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_names:
        if df[col].isna().any():
            df[col].fillna(medians.get(col, 0.0), inplace=True)

    # Load scaler đã fit, transform flow
    scaler = joblib.load(SCALER_PATH)
    x_scaled = scaler.transform(df[feature_names].values)  # shape (1, 66)
    return x_scaled[0]                                      # shape (66,)


def load_model(cfg: dict) -> tuple[VAE, bool]:
    """
    Cố gắng load vae_best.pth.
    Nếu không tìm thấy, trả về model random với flag is_real=False.
    """
    model = VAE(
        input_dim   = cfg["input_dim"],
        hidden_dims = cfg["hidden_dims"],
        latent_dim  = cfg["latent_dim"],
    )
    if MODEL_PATH.exists():
        # Load weights đã train — map_location để tương thích CPU
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, True
    else:
        # Fallback: model random, chỉ dùng để minh họa cơ chế
        model.eval()
        return model, False


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – HIỂN THỊ FLOW THỰC TẾ (GIÁ TRỊ ĐÃ SCALE)
# ──────────────────────────────────────────────────────────────────────────────
def print_flow(x_scaled: np.ndarray, feature_names: list[str]):
    print(SEP)
    print("PHẦN 1 – FLOW THỰC TẾ (đã preprocess + scale)")
    print(SEP)
    print()
    print(f"  Nguồn: {SAMPLE_PATH.relative_to(ROOT)}")
    print(f"  Flow index: 0  (dòng đầu tiên)")
    print()
    print(f"  {'Index':>5}  {'Feature Name':<32}  {'Scaled Value':>14}")
    print("  " + "-" * 56)
    for i, (name, val) in enumerate(zip(feature_names, x_scaled)):
        print(f"  {i:>5}  {name:<32}  {val:>14.6f}")
    print()
    print(f"  Tổng số feature: {len(x_scaled)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – LOAD MODEL VÀ IN THÔNG TIN FC1
# ──────────────────────────────────────────────────────────────────────────────
def print_model_info(model: VAE, is_real: bool):
    print(SEP)
    print("PHẦN 2 – MODEL ĐÃ LOAD")
    print(SEP)
    print()
    # encoder.0 là Linear đầu tiên (fc1)
    fc1 = model.encoder[0]
    W   = fc1.weight.detach().numpy()   # shape (128, 66)
    b   = fc1.bias.detach().numpy()     # shape (128,)

    src = str(MODEL_PATH.relative_to(ROOT)) if is_real else "MODEL DEMO (random weights)"
    print(f"  Nguồn: {src}")
    print(f"  Trạng thái: {'✓ Model đã train (vae_best.pth)' if is_real else '⚠ Model demo — chưa train'}")
    print()
    print(f"  fc1.weight  shape = {list(W.shape)}   → [n_neurons, n_features] = [128, 66]")
    print(f"  fc1.bias    shape = {list(b.shape)}   → [n_neurons] = [128]")
    print()
    print("  Ý nghĩa:")
    print("    W[i, j] = trọng số kết nối feature j đến neuron i")
    print("    b[i]    = bias của neuron i")
    print("    h1[i]   = ReLU( Σ_j W[i,j]·x[j] + b[i] )")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3 – GIẢI PHẪU NEURON #0
# ──────────────────────────────────────────────────────────────────────────────
def print_neuron_anatomy(model: VAE, x_scaled: np.ndarray, feature_names: list[str]):
    print(SEP)
    print("PHẦN 3 – GIẢI PHẪU NEURON #0 của LỚP h1 (fc1)")
    print(SEP)
    print()
    print("  Neuron số 0 tính:")
    print("    pre_activation = W1[0,:] · x + b1[0]")
    print("    h1[0]          = ReLU(pre_activation)")
    print()

    fc1 = model.encoder[0]
    W0  = fc1.weight[0].detach().numpy()   # shape (66,)  — hàng 0 của W
    b0  = fc1.bias[0].detach().item()      # scalar — bias của neuron 0

    contributions = W0 * x_scaled          # shape (66,) — W[0,j] * x[j] từng feature

    # In bảng đầy đủ 66 feature
    col_w = 32   # chiều rộng cột tên feature
    print(f"  {'Feature':<{col_w}}  {'x (scaled)':>13}  {'W[0,j]':>12}  {'W[0,j]·x[j]':>14}")
    print("  " + "-" * (col_w + 46))
    for j, (name, xj, wj, cj) in enumerate(
        zip(feature_names, x_scaled, W0, contributions)
    ):
        # Đánh dấu đóng góp lớn (|W*x| > 0.5)
        marker = " ◀" if abs(cj) > 0.5 else ""
        print(
            f"  {name:<{col_w}}  {xj:>13.6f}  {wj:>12.8f}  {cj:>14.8f}{marker}"
        )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 4 – TOP 10 FEATURE ĐÓNG GÓP LỚN NHẤT
# ──────────────────────────────────────────────────────────────────────────────
def print_top_features(model: VAE, x_scaled: np.ndarray, feature_names: list[str]):
    print(SEP)
    print("PHẦN 4 – TOP 10 FEATURE ĐÓNG GÓP LỚN NHẤT VÀO NEURON #0")
    print(SEP)
    print()
    print("  Xếp hạng theo |W[0,j] · x[j]|  (đóng góp tuyệt đối)")
    print()

    fc1  = model.encoder[0]
    W0   = fc1.weight[0].detach().numpy()
    b0   = fc1.bias[0].detach().item()
    contributions = W0 * x_scaled
    total_contrib = contributions.sum()     # Σ(W*x) — chưa cộng bias

    # Sắp xếp theo |đóng góp| giảm dần
    sorted_idx = np.argsort(np.abs(contributions))[::-1]
    top10      = sorted_idx[:10]

    print(
        f"  {'Rank':>4}  {'Feature':<32}  {'Weight W[0,j]':>14}  "
        f"{'Value x[j]':>12}  {'Contribution':>14}  {'Contrib%':>9}"
    )
    print("  " + "-" * 94)
    for rank, j in enumerate(top10, start=1):
        name    = feature_names[j]
        w       = W0[j]
        xv      = x_scaled[j]
        contrib = contributions[j]
        # Phần trăm đóng góp tuyệt đối so với tổng Σ|W*x|
        pct     = 100.0 * abs(contrib) / (np.abs(contributions).sum() + 1e-12)
        bar     = "█" * max(1, int(pct / 2))    # thanh biểu đồ đơn giản
        print(
            f"  {rank:>4}  {name:<32}  {w:>14.8f}  "
            f"{xv:>12.6f}  {contrib:>+14.8f}  {pct:>8.2f}%  {bar}"
        )

    print()
    print(f"  Σ|W[0,j]·x[j]| = {np.abs(contributions).sum():.6f}")
    print(f"  Σ W[0,j]·x[j]  = {total_contrib:+.6f}  (tổng có dấu)")
    print(f"  bias b1[0]      = {b0:+.6f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 5 – TÍNH TOÁN THỦ CÔNG
# ──────────────────────────────────────────────────────────────────────────────
def print_manual_calculation(model: VAE, x_scaled: np.ndarray, feature_names: list[str]):
    print(SEP)
    print("PHẦN 5 – TÍNH TOÁN THỦ CÔNG")
    print(SEP)
    print()

    fc1  = model.encoder[0]
    W0   = fc1.weight[0].detach().numpy()
    b0   = fc1.bias[0].detach().item()
    contributions = W0 * x_scaled

    # Lấy tối đa 10 dòng đầu để in cho gọn (tránh tràn màn hình)
    N_SHOW = 10
    print(f"  Bước 1: Tính Σ(W[0,j] · x[j])  — chỉ in {N_SHOW}/{len(feature_names)} dòng đầu")
    print()
    print("  Σ(W·x) =")
    running = 0.0
    for j in range(N_SHOW):
        w  = W0[j]
        xv = x_scaled[j]
        c  = contributions[j]
        running += c
        sign = "+" if j > 0 else " "
        print(f"    {sign} ({w:+.8f}) × ({xv:+.8f})  =  {c:+.10f}   ← {feature_names[j]}")

    remaining_sum = contributions[N_SHOW:].sum()
    print(f"    + ... ({len(feature_names) - N_SHOW} feature còn lại)  =  {remaining_sum:+.10f}")
    total_wx = contributions.sum()
    print()
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Σ(W[0,j] · x[j])  = {total_wx:+.10f}")
    print()

    # Bước 2: cộng bias
    print("  Bước 2: Cộng bias")
    print()
    pre = total_wx + b0
    print(f"  pre_activation = Σ(W·x) + b1[0]")
    print(f"                 = ({total_wx:+.10f})")
    print(f"                 + ({b0:+.10f})")
    print(f"                 = {pre:+.10f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 6 – RELU
# ──────────────────────────────────────────────────────────────────────────────
def print_relu(model: VAE, x_scaled: np.ndarray):
    print(SEP)
    print("PHẦN 6 – ÁP DỤNG ReLU  →  h1[0]")
    print(SEP)
    print()

    fc1  = model.encoder[0]
    W0   = fc1.weight[0].detach().numpy()
    b0   = fc1.bias[0].detach().item()
    pre  = float(W0 @ x_scaled + b0)           # dot product + bias = pre_activation

    print(f"  pre_activation = {pre:+.10f}")
    print()
    print("  Quy tắc ReLU:")
    print("    Nếu pre_activation > 0  →  h1[0] = pre_activation  (neuron kích hoạt)")
    print("    Nếu pre_activation ≤ 0  →  h1[0] = 0               (neuron bị triệt tiêu)")
    print()

    h1_0 = max(0.0, pre)

    if pre > 0:
        print(f"  pre_activation = {pre:+.10f}  >  0")
        print(f"  ⟹  h1[0] = {h1_0:+.10f}   ✓ Neuron KÍCH HOẠT")
    else:
        print(f"  pre_activation = {pre:+.10f}  ≤  0")
        print(f"  ⟹  h1[0] = 0.0000000000    ✗ Neuron BỊ TRIỆT TIÊU")

    print()

    # Kiểm chứng bằng PyTorch — đưa x qua encoder thực tế
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 66)
    with torch.no_grad():
        # encoder.0 = Linear, encoder.1 = ReLU → chạy cả 2
        h_linear = model.encoder[0](x_tensor)         # sau Linear, trước ReLU
        h_relu   = model.encoder[:2](x_tensor)        # sau ReLU
        pytorch_pre  = h_linear[0, 0].item()
        pytorch_h1_0 = h_relu[0, 0].item()

    print("  ─── Kiểm chứng với PyTorch ───────────────────────")
    print(f"  pre_activation (thủ công)  = {pre:+.10f}")
    print(f"  pre_activation (PyTorch)   = {pytorch_pre:+.10f}")
    print(f"  h1[0]          (thủ công)  = {h1_0:+.10f}")
    print(f"  h1[0]          (PyTorch)   = {pytorch_h1_0:+.10f}")
    diff_pre = abs(pre - pytorch_pre)
    diff_h1  = abs(h1_0 - pytorch_h1_0)
    print()
    print(f"  |Δ pre_activation| = {diff_pre:.2e}  {'✓ khớp' if diff_pre < 1e-4 else '⚠ sai lệch!'}")
    print(f"  |Δ h1[0]|          = {diff_h1:.2e}  {'✓ khớp' if diff_h1 < 1e-4 else '⚠ sai lệch!'}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# BONUS – TOP 5 NEURON KÍCH HOẠT MẠNH NHẤT TRONG h1
# ──────────────────────────────────────────────────────────────────────────────
def print_top_neurons(model: VAE, x_scaled: np.ndarray):
    print(SEP)
    print("BONUS – TOP 5 NEURON KÍCH HOẠT MẠNH NHẤT TRONG LỚP h1 (128 neuron)")
    print(SEP)
    print()

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        h_linear = model.encoder[0](x_tensor).squeeze(0).numpy()  # shape (128,) trước ReLU
        h_relu   = np.maximum(0, h_linear)                         # sau ReLU

    n_active    = int((h_relu > 0).sum())
    n_suppressed = len(h_relu) - n_active

    print(f"  Tổng số neuron trong h1:        128")
    print(f"  Số neuron KÍCH HOẠT (h1[i]>0): {n_active}")
    print(f"  Số neuron BỊ TRIỆT TIÊU (=0):  {n_suppressed}")
    print()
    print("  Top 5 neuron có h1[i] lớn nhất (sau ReLU):")
    print()
    print(f"  {'Rank':>4}  {'Neuron #':>8}  {'pre_activation':>16}  {'h1[i]':>14}  {'Trạng thái'}")
    print("  " + "-" * 62)
    top5_idx = np.argsort(h_relu)[::-1][:5]
    for rank, i in enumerate(top5_idx, start=1):
        pre_i = h_linear[i]
        h_i   = h_relu[i]
        status = "KÍCH HOẠT" if h_i > 0 else "triệt tiêu"
        print(f"  {rank:>4}  {i:>8}  {pre_i:>+16.8f}  {h_i:>14.8f}  {status}")

    print()
    print("  Ý nghĩa: Mỗi neuron học cách phản ứng với một tổ hợp tuyến tính")
    print("  khác nhau của 66 feature. ReLU chỉ giữ lại những neuron dương,")
    print("  tạo ra biểu diễn thưa (sparse) — đặc điểm quan trọng cho")
    print("  việc phân biệt BENIGN vs ATTACK sau này.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print(SEP)
    print("  DEMO: GIẢI PHẪU NEURON — NIDS VAE PROJECT")
    print("  Mục tiêu: Hiểu chính xác h1[0] = ReLU(W1[0,:]·x + b1[0])")
    print(SEP)
    print()

    # ── Đọc cấu hình và artifacts ────────────────────────────────────────────
    cfg          = load_config()
    feature_names = load_feature_names()
    medians      = load_medians()

    print(f"  Kiến trúc: input_dim={cfg['input_dim']}, "
          f"hidden_dims={cfg['hidden_dims']}, latent_dim={cfg['latent_dim']}")
    print()

    # ── Kiểm tra sự tồn tại của artifacts ────────────────────────────────────
    for p in [FEATURE_PATH, SCALER_PATH, MEDIANS_PATH, SAMPLE_PATH]:
        if not p.exists():
            print(f"  [LỖI] Không tìm thấy: {p}")
            sys.exit(1)

    # ── Load và preprocess flow thực tế ──────────────────────────────────────
    x_scaled = load_and_preprocess_flow(feature_names, medians)

    # ── Load model ────────────────────────────────────────────────────────────
    model, is_real = load_model(cfg)
    if not is_real:
        print("  [CẢNH BÁO] Không tìm thấy vae_best.pth — dùng model demo với random weights.")
        print("             Kết quả minh họa cơ chế tính toán, không phản ánh model thực.")
        print()

    # ── Chạy từng phần ────────────────────────────────────────────────────────
    print_flow(x_scaled, feature_names)
    print_model_info(model, is_real)
    print_neuron_anatomy(model, x_scaled, feature_names)
    print_top_features(model, x_scaled, feature_names)
    print_manual_calculation(model, x_scaled, feature_names)
    print_relu(model, x_scaled)
    print_top_neurons(model, x_scaled)

    print(SEP)
    print("  DEMO HOÀN TẤT")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
