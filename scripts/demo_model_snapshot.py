"""
scripts/demo_model_snapshot.py
-------------------------------
Demo trực quan cách toàn bộ VAE thay đổi qua từng epoch và
giải thích chính xác "Best Epoch" thực chất lưu những gì.

Chạy:
    python scripts/demo_model_snapshot.py

KHÔNG yêu cầu dữ liệu CICIDS2017 thực. Dataset giả lập (BENIGN synthetic).
KHÔNG thay đổi bất kỳ artifact nào đang có trong thư mục artifacts/.
"""

import copy
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Không mở cửa sổ GUI — tương thích mọi môi trường
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH DEMO
# ──────────────────────────────────────────────────────────────────────────────
INPUT_DIM   = 66      # Số đặc trưng, khớp với CICIDS2017 thực
HIDDEN_DIMS = [128, 64]
LATENT_DIM  = 16
N_EPOCHS    = 20      # Số epoch demo nhỏ
N_SAMPLES   = 500     # Số mẫu dữ liệu giả lập
BATCH_SIZE  = 64
LR          = 1e-3
SEED        = 42

# Đường dẫn lưu biểu đồ — tạo nếu chưa có
OUTPUT_DIR  = Path("artifacts") / "demo_snapshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_PATH  = OUTPUT_DIR / "epoch_vs_weight_changes.png"

# Các tham số được chọn để theo dõi suốt quá trình huấn luyện
WATCHED_PARAMS = {
    "fc1.weight[0,0]":     ("encoder.0.weight",  (0, 0)),   # Lớp encoder đầu tiên
    "fc1.weight[0,1]":     ("encoder.0.weight",  (0, 1)),
    "fc2.weight[0,0]":     ("encoder.2.weight",  (0, 0)),   # Lớp encoder thứ hai
    "fc_mu.weight[0,0]":   ("fc_mu.weight",      (0, 0)),   # Lớp mu (mean)
    "fc_logvar.weight[0,0]": ("fc_logvar.weight",(0, 0)),   # Lớp logvar
}

# Phân cách trực quan
SEP  = "=" * 60
SEP2 = "-" * 60


# ──────────────────────────────────────────────────────────────────────────────
# KIẾN TRÚC VAE (bản sao nhỏ gọn, khớp với backend/app/models/vae.py)
# ──────────────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    """
    Variational Autoencoder đơn giản cho demo.
    Kiến trúc: 66 → 128 → 64 → (mu16, logvar16) → z16 → 64 → 128 → 66
    """

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()

        # Encoder: input_dim → hidden_dims
        encoder_layers = []
        in_f = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Hai nhánh song song mu và logvar
        self.fc_mu     = nn.Linear(in_f, latent_dim)
        self.fc_logvar = nn.Linear(in_f, latent_dim)

        # Decoder: latent_dim → reversed(hidden_dims) → input_dim
        decoder_layers = []
        in_f = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        decoder_layers.append(nn.Linear(in_f, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + eps * std
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu  # Inference: dùng mu để deterministic

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0):
    """
    Hàm mất mát VAE = Reconstruction Loss (MSE) + beta * KL Divergence.
    MSE: trung bình trên tất cả feature và sample.
    KL : -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / n_samples
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()


# ──────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH: đọc giá trị tham số từ tên và index
# ──────────────────────────────────────────────────────────────────────────────
def get_param_val(model: nn.Module, param_name: str, idx: tuple) -> float:
    """Lấy giá trị scalar của tham số tại vị trí idx (r, c)."""
    # Duyệt qua state_dict để truy cập tham số theo tên
    tensor = dict(model.named_parameters())[param_name]
    if len(idx) == 2:
        return tensor[idx[0], idx[1]].item()
    return tensor[idx[0]].item()


def snapshot_watched(model: nn.Module) -> dict:
    """Chụp toàn bộ giá trị tham số được theo dõi tại thời điểm gọi."""
    return {
        label: get_param_val(model, pname, pidx)
        for label, (pname, pidx) in WATCHED_PARAMS.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – HIỂN THỊ KIẾN TRÚC MODEL
# ──────────────────────────────────────────────────────────────────────────────
def print_architecture():
    print(SEP)
    print("PHẦN 1 – KIẾN TRÚC VAE")
    print(SEP)
    print()
    print("Luồng dữ liệu qua model:")
    print()
    print("       66          ← input_dim (số feature CICIDS2017 sau cleaning)")
    print("        ↓")
    print("      128          ← fc1  (encoder hidden layer 1)")
    print("        ↓")
    print("       64          ← fc2  (encoder hidden layer 2)")
    print("        ↙   ↘")
    print("     μ16  logvar16  ← fc_mu, fc_logvar (hai nhánh song song)")
    print("        ↘   ↙")
    print("        z16         ← z = μ + ε·σ  (reparameterization trick)")
    print("         ↓")
    print("        64          ← decoder_fc1")
    print("         ↓")
    print("       128          ← decoder_fc2")
    print("         ↓")
    print("        66          ← decoder_fc3  (output, linear activation)")
    print()
    print("Danh sách layer và tham số:")
    print()
    # Encoder layers
    print("  encoder.0.weight   (fc1.weight)   — shape: [128, 66]")
    print("  encoder.0.bias     (fc1.bias)     — shape: [128]")
    print()
    print("  encoder.2.weight   (fc2.weight)   — shape: [64, 128]")
    print("  encoder.2.bias     (fc2.bias)     — shape: [64]")
    print()
    print("  fc_mu.weight                      — shape: [16, 64]")
    print("  fc_mu.bias                        — shape: [16]")
    print()
    print("  fc_logvar.weight                  — shape: [16, 64]")
    print("  fc_logvar.bias                    — shape: [16]")
    print()
    print("  decoder.0.weight   (decoder_fc1)  — shape: [64, 16]")
    print("  decoder.0.bias                    — shape: [64]")
    print()
    print("  decoder.2.weight   (decoder_fc2)  — shape: [128, 64]")
    print("  decoder.2.bias                    — shape: [128]")
    print()
    print("  decoder.4.weight   (decoder_fc3)  — shape: [66, 128]")
    print("  decoder.4.bias                    — shape: [66]")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – THEO DÕI THAM SỐ (giá trị khởi tạo ngẫu nhiên)
# ──────────────────────────────────────────────────────────────────────────────
def print_initial_params(model: nn.Module):
    print(SEP)
    print("PHẦN 2 – GIÁ TRỊ THAM SỐ KHỞI TẠO NGẪU NHIÊN")
    print(SEP)
    print()
    print("Epoch 0 (Random Init — trước khi train bất kỳ batch nào)")
    print()
    vals = snapshot_watched(model)
    for label, val in vals.items():
        print(f"  {label:<28} = {val:+.8f}")
    print()
    print("Ghi chú: PyTorch khởi tạo Linear layers bằng Kaiming Uniform")
    print("         nên các giá trị ngẫu nhiên nhỏ quanh 0.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3+4+5 – TRAIN NHỎ, SNAPSHOT, VÀ BEST EPOCH
# ──────────────────────────────────────────────────────────────────────────────
def run_training(model: nn.Module, X_train: torch.Tensor, X_val: torch.Tensor):
    """
    Chạy vòng lặp huấn luyện nhỏ, in chi tiết mỗi epoch,
    thu thập snapshot và xác định best epoch dựa trên val_loss.

    Trả về:
        history       : list[dict] — snapshot mỗi epoch
        best_epoch    : int        — epoch có val_loss thấp nhất
        best_state    : dict       — state_dict tại best epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    history   = []
    best_val  = float("inf")
    best_epoch = -1
    best_state  = None

    print(SEP)
    print("PHẦN 3 – TRAIN NHỎ (20 EPOCH)")
    print(SEP)
    print()

    dataset = torch.utils.data.TensorDataset(X_train)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()

        # ── Lưu giá trị TRƯỚC khi update ──────────────────────────────────
        pre_vals = snapshot_watched(model)

        # ── Training step ──────────────────────────────────────────────────
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch)
            loss, _, _        = vae_loss(batch, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_loss = epoch_loss / len(X_train)

        # ── Validation loss ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            x_hat_v, mu_v, logvar_v = model(X_val)
            val_loss, _, _          = vae_loss(X_val, x_hat_v, mu_v, logvar_v)
            val_loss = val_loss.item()

        # ── Lưu giá trị SAU khi update ────────────────────────────────────
        post_vals = snapshot_watched(model)

        # ── In chi tiết epoch ──────────────────────────────────────────────
        print(f"Epoch {epoch:02d}")
        print(SEP2)
        print(f"  Train Loss = {train_loss:.6f}  |  Val Loss = {val_loss:.6f}")
        print()

        # In trước/sau/delta cho 4 tham số chính
        for label in [
            "fc1.weight[0,0]",
            "fc2.weight[0,0]",
            "fc_mu.weight[0,0]",
            "fc_logvar.weight[0,0]",
        ]:
            pre  = pre_vals[label]
            post = post_vals[label]
            delta = post - pre
            print(f"  {label:<28}")
            print(f"    trước update  = {pre:+.8f}")
            print(f"    sau update    = {post:+.8f}")
            print(f"    Δ             = {delta:+.8f}")

        # ── Kiểm tra best epoch ────────────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())  # Chụp toàn bộ trạng thái model
            print()
            print(f"  → ★ New Best Epoch  (val_loss = {val_loss:.6f})")
            print(f"  → Saving Model Snapshot for Epoch {epoch}")

        print()

        # ── Thu thập snapshot cho biểu đồ ─────────────────────────────────
        history.append({
            "epoch":             epoch,
            "train_loss":        train_loss,
            "val_loss":          val_loss,
            "fc1_weight_00":     post_vals["fc1.weight[0,0]"],
            "fc2_weight_00":     post_vals["fc2.weight[0,0]"],
            "fc_mu_weight_00":   post_vals["fc_mu.weight[0,0]"],
            "fc_logvar_weight_00": post_vals["fc_logvar.weight[0,0]"],
        })

    return history, best_epoch, best_state


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 4 – BẢNG SNAPSHOT
# ──────────────────────────────────────────────────────────────────────────────
def print_snapshot_table(history: list):
    print(SEP)
    print("PHẦN 4 – BẢNG SNAPSHOT TOÀN BỘ EPOCH")
    print(SEP)
    print()
    # Header
    print(f"  {'Epoch':>5} | {'Loss':>10} | {'fc1[0,0]':>12} | {'fc2[0,0]':>12} | {'mu[0,0]':>12} | {'logvar[0,0]':>12}")
    print("  " + "-" * 72)
    for snap in history:
        print(
            f"  {snap['epoch']:>5} | "
            f"{snap['train_loss']:>10.6f} | "
            f"{snap['fc1_weight_00']:>+12.8f} | "
            f"{snap['fc2_weight_00']:>+12.8f} | "
            f"{snap['fc_mu_weight_00']:>+12.8f} | "
            f"{snap['fc_logvar_weight_00']:>+12.8f}"
        )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 6 – SO SÁNH BEST EPOCH vs FINAL EPOCH
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison(model: nn.Module, best_state: dict, best_epoch: int, history: list):
    """
    Nạp lại best_state vào model để đọc tham số tại best epoch,
    rồi so sánh với tham số tại epoch cuối cùng.
    """
    final_epoch = history[-1]["epoch"]

    # Đọc giá trị từ best_state mà KHÔNG thay đổi model gốc
    tmp_model = VAE(INPUT_DIM, HIDDEN_DIMS, LATENT_DIM)
    tmp_model.load_state_dict(best_state)

    best_vals  = snapshot_watched(tmp_model)
    final_vals = snapshot_watched(model)   # Model hiện tại là trạng thái sau epoch cuối

    print(SEP)
    print("PHẦN 6 – SO SÁNH: BEST EPOCH vs FINAL EPOCH")
    print(SEP)
    print()

    print(f"Best Epoch = {best_epoch}  (val_loss thấp nhất)")
    print()
    for label, val in best_vals.items():
        print(f"  {label:<28} = {val:+.8f}")
    print()

    print(f"Final Epoch = {final_epoch}")
    print()
    for label, val in final_vals.items():
        print(f"  {label:<28} = {val:+.8f}")
    print()

    # Tính độ lệch giữa best và final
    print("  Δ (Final − Best):")
    for label in best_vals:
        diff = final_vals[label] - best_vals[label]
        marker = " ← KHÁC NHAU" if abs(diff) > 1e-9 else ""
        print(f"  {label:<28}  Δ = {diff:+.8f}{marker}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 7 – GIẢI THÍCH (tiếng Việt)
# ──────────────────────────────────────────────────────────────────────────────
def print_explanation():
    print(SEP)
    print("PHẦN 7 – GIẢI THÍCH: BEST EPOCH LƯU GÌ?")
    print(SEP)
    print()
    print("  Best Epoch KHÔNG phải chỉ là W1 (fc1.weight).")
    print()
    print("  Best Epoch là SNAPSHOT TOÀN BỘ MODEL tại thời điểm")
    print("  validation loss thấp nhất trong quá trình huấn luyện.")
    print()
    print("  Checkpoint (vae_best.pth) lưu đồng thời:")
    print()
    print("    Encoder:")
    print("      • encoder.0.weight  (fc1.weight)")
    print("      • encoder.0.bias    (fc1.bias)")
    print("      • encoder.2.weight  (fc2.weight)")
    print("      • encoder.2.bias    (fc2.bias)")
    print()
    print("    Bottleneck:")
    print("      • fc_mu.weight      (W_mu)")
    print("      • fc_mu.bias")
    print("      • fc_logvar.weight  (W_logvar)")
    print("      • fc_logvar.bias")
    print()
    print("    Decoder:")
    print("      • decoder.0.weight  (decoder_fc1.weight)")
    print("      • decoder.0.bias")
    print("      • decoder.2.weight  (decoder_fc2.weight)")
    print("      • decoder.2.bias")
    print("      • decoder.4.weight  (decoder_fc3.weight)")
    print("      • decoder.4.bias")
    print()
    print("  ──────────────────────────────────────────────────────")
    print("  Tại sao KHÔNG thể lấy W1 từ Epoch A và W2 từ Epoch B?")
    print("  ──────────────────────────────────────────────────────")
    print()
    print("  Tất cả layer được học ĐỒNG THỜI qua cùng một")
    print("  vòng lặp backward. Gradient của fc2 phụ thuộc vào")
    print("  giá trị đầu ra của fc1. Gradient của fc_mu phụ thuộc")
    print("  vào trạng thái của fc2. Chúng không thể hoán đổi cho")
    print("  nhau vì không có sự nhất quán nội bộ nếu trộn lẫn")
    print("  các epoch khác nhau.")
    print()
    print("  Ví dụ: Nếu fc1 tại Epoch 10 học được cách nén feature A")
    print("  vào chiều 0, nhưng fc2 tại Epoch 20 đã học rằng chiều 0")
    print("  của fc1 mang thông tin feature B, thì kết hợp hai trạng")
    print("  thái đó sẽ tạo ra đầu ra vô nghĩa và reconstruction")
    print("  error sẽ không còn ý nghĩa phân loại anomaly.")
    print()
    print("  ──────────────────────────────────────────────────────")
    print("  vae_best.pth là gì?")
    print("  ──────────────────────────────────────────────────────")
    print()
    print("  vae_best.pth là ẢNH CHỤP TOÀN BỘ TRẠNG THÁI MODEL")
    print("  (state_dict) tại Best Epoch — tức là thời điểm model")
    print("  có khả năng tái tạo dữ liệu BENIGN tốt nhất (val_loss")
    print("  thấp nhất) trước khi bắt đầu overfit.")
    print()
    print("  Khi load lại để inference, model sẽ tái tạo BENIGN")
    print("  với lỗi thấp và tái tạo ATTACK với lỗi cao — đó chính")
    print("  là cơ chế phát hiện anomaly của hệ thống NIDS.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 8 – BIỂU ĐỒ epoch_vs_weight_changes.png
# ──────────────────────────────────────────────────────────────────────────────
def plot_weight_changes(history: list, best_epoch: int):
    """
    Vẽ biểu đồ đường thể hiện giá trị 3 trọng số theo epoch.
    Đánh dấu best epoch bằng đường dọc.
    """
    epochs       = [s["epoch"] for s in history]
    fc1_vals     = [s["fc1_weight_00"] for s in history]
    fc2_vals     = [s["fc2_weight_00"] for s in history]
    fc_mu_vals   = [s["fc_mu_weight_00"] for s in history]
    val_losses   = [s["val_loss"] for s in history]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        "Trọng số VAE thay đổi qua từng Epoch\n"
        "(Demo nhỏ — dữ liệu BENIGN giả lập)",
        fontsize=13, fontweight="bold"
    )

    # ── Subplot 1: giá trị trọng số ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, fc1_vals,   marker="o", linewidth=1.8, label="fc1.weight[0,0]   (encoder lớp 1)")
    ax1.plot(epochs, fc2_vals,   marker="s", linewidth=1.8, label="fc2.weight[0,0]   (encoder lớp 2)")
    ax1.plot(epochs, fc_mu_vals, marker="^", linewidth=1.8, label="fc_mu.weight[0,0]  (bottleneck μ)")
    ax1.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.5,
                label=f"Best Epoch = {best_epoch}")
    ax1.set_ylabel("Giá trị trọng số")
    ax1.set_title("Giá trị tham số được theo dõi")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Subplot 2: validation loss ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, val_losses, color="purple", marker="D", linewidth=1.8,
             label="Validation Loss")
    ax2.axvline(x=best_epoch, color="red", linestyle="--", linewidth=1.5,
                label=f"Best Epoch = {best_epoch}")
    # Đánh dấu điểm val_loss thấp nhất
    best_idx = best_epoch - 1
    ax2.scatter([best_epoch], [val_losses[best_idx]], color="red", zorder=5,
                s=80, label=f"Min val_loss = {val_losses[best_idx]:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss theo Epoch")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(SEP)
    print("PHẦN 8 – BIỂU ĐỒ epoch_vs_weight_changes.png")
    print(SEP)
    print()
    print(f"  Biểu đồ đã được lưu tại: {CHART_PATH.resolve()}")
    print()
    print("  Biểu đồ gồm 2 panel:")
    print("    Panel 1 – Giá trị 3 trọng số theo epoch:")
    print("              fc1.weight[0,0]   (encoder lớp 1)")
    print("              fc2.weight[0,0]   (encoder lớp 2)")
    print("              fc_mu.weight[0,0] (bottleneck μ)")
    print()
    print("    Panel 2 – Validation Loss theo epoch.")
    print()
    print("    Đường đỏ đứt: Best Epoch (val_loss thấp nhất).")
    print("    Quan sát: các trọng số tiếp tục thay đổi sau Best Epoch,")
    print("    nhưng checkpoint chỉ giữ lại trạng thái tốt nhất.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 5 – IN LỊCH SỬ VAL LOSS (tóm tắt)
# ──────────────────────────────────────────────────────────────────────────────
def print_val_loss_summary(history: list, best_epoch: int):
    print(SEP)
    print("PHẦN 5 – LỊCH SỬ VALIDATION LOSS")
    print(SEP)
    print()
    for snap in history:
        marker = "  ← ★ Best Epoch" if snap["epoch"] == best_epoch else ""
        print(f"  Epoch {snap['epoch']:02d}  val_loss = {snap['val_loss']:.6f}{marker}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Khởi tạo seed để kết quả tái tạo được ────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print()
    print(SEP)
    print("  DEMO: VAE MODEL SNAPSHOT — NIDS VAE PROJECT")
    print("  Mục tiêu: Hiểu cách trọng số thay đổi qua từng epoch")
    print("            và Best Epoch thực chất lưu những gì.")
    print(SEP)
    print()

    # ── Tạo dữ liệu BENIGN giả lập ───────────────────────────────────────────
    # Mô phỏng traffic bình thường: phân phối chuẩn quanh 0, đã scale
    print("Đang tạo dữ liệu BENIGN giả lập ...")
    X = torch.randn(N_SAMPLES, INPUT_DIM) * 0.5

    # Chia train/val theo tỷ lệ 80/20
    split   = int(0.8 * N_SAMPLES)
    X_train = X[:split]
    X_val   = X[split:]
    print(f"  Train: {len(X_train)} mẫu  |  Val: {len(X_val)} mẫu")
    print(f"  Mỗi mẫu: {INPUT_DIM} feature")
    print()

    # ── Khởi tạo model ────────────────────────────────────────────────────────
    model = VAE(INPUT_DIM, HIDDEN_DIMS, LATENT_DIM)

    # ── In thông tin tổng số tham số ─────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {n_params:,}")
    print()

    # ── PHẦN 1: Kiến trúc ─────────────────────────────────────────────────────
    print_architecture()

    # ── PHẦN 2: Giá trị khởi tạo ─────────────────────────────────────────────
    print_initial_params(model)

    # ── PHẦN 3+4+5: Training, Snapshot, Best Epoch ───────────────────────────
    history, best_epoch, best_state = run_training(model, X_train, X_val)

    # ── PHẦN 4: Bảng snapshot ────────────────────────────────────────────────
    print_snapshot_table(history)

    # ── PHẦN 5: Tóm tắt val loss ─────────────────────────────────────────────
    print_val_loss_summary(history, best_epoch)

    # ── PHẦN 6: So sánh Best vs Final ────────────────────────────────────────
    print_comparison(model, best_state, best_epoch, history)

    # ── PHẦN 7: Giải thích ───────────────────────────────────────────────────
    print_explanation()

    # ── PHẦN 8: Biểu đồ ──────────────────────────────────────────────────────
    plot_weight_changes(history, best_epoch)

    print(SEP)
    print("  DEMO HOÀN TẤT")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
