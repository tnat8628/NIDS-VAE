"""
scripts/train.py
-----------------
Pipeline huấn luyện Variational Autoencoder (VAE) cho bài toán phát hiện
xâm nhập mạng (NIDS) trên dataset CICIDS2017.

Quy trình:
  1. Tải dữ liệu đã xử lý (X_train.npy, X_val.npy) từ data/train và data/validation
  2. Khởi tạo VAE với kiến trúc chuẩn (input=66, hidden=[128,64], latent=16)
  3. Huấn luyện với Adam optimizer, early stopping, gradient clipping
  4. KL Annealing: beta tăng dần từ beta_start → beta_end trong warmup_epochs đầu
     để tránh posterior collapse (VAE loss đặc thù).
  5. Lưu artifacts: vae_best.pth, model_config.json, training_history.json,
     training_summary.json

Sử dụng:
    python scripts/train.py [--epochs 100] [--batch-size 1024] [--lr 1e-3]
                            [--patience 10] [--beta-end 1.0] [--latent-dim 16]
                            [--beta-start 0.0] [--beta-warmup-epochs 30]
                            [--kl-annealing-type linear]
                            [--use-free-bits] [--free-bits-lambda 0.05]
                            [--no-cuda]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Thiết lập đường dẫn gốc của project ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.vae import VAE, vae_loss, vae_loss_free_bits  # noqa: E402

# ── Hằng số mặc định ─────────────────────────────────────────────────────────
INPUT_DIM   = 66            # Số feature sau khi loại constant/near-constant cols
LATENT_DIM  = 16            # Số chiều không gian tiềm ẩn
HIDDEN_DIMS = [128, 64]     # Kích thước các hidden layer

LEARNING_RATE  = 1e-3       # Learning rate cho Adam optimizer
WEIGHT_DECAY   = 1e-5       # L2 regularization
BATCH_SIZE     = 1024       # Số mẫu mỗi batch
MAX_EPOCHS     = 100        # Số epoch tối đa
PATIENCE       = 10         # Số epoch chịu đựng không cải thiện trước khi dừng
GRAD_CLIP_NORM = 5.0        # Giới hạn max norm của gradient
BETA           = 1.0        # Hệ số cân bằng KL divergence (beta_end cuối cùng)
RANDOM_SEED    = 42         # Seed cho tái tạo kết quả

# ── Hằng số KL Annealing — tránh posterior collapse ──────────────────────────
# Posterior collapse xảy ra khi beta=1.0 ngay từ đầu, encoder học mu→0 để giảm KL.
# Giải pháp: beta tăng dần từ 0 → beta_end trong warmup_epochs đầu.
BETA_START          = 0.0   # Beta ban đầu (0 = chỉ optimize reconstruction)
BETA_END            = 1.0   # Beta cuối sau warm-up (= BETA)
BETA_WARMUP_EPOCHS  = 30    # Số epoch để beta tăng từ beta_start → beta_end
KL_ANNEALING_TYPE   = "linear"  # linear | quadratic | constant

# ── Free Bits — ràng buộc tối thiểu KL mỗi latent dim (tắt mặc định) ────────
# Khi bật: KL của mỗi dim không được thấp hơn free_bits_lambda (nats).
# Tránh latent dimensions bị collapse về prior hoàn toàn.
USE_FREE_BITS      = False  # Không bật mặc định — dùng KL annealing trước
FREE_BITS_LAMBDA   = 0.05   # Ngưỡng KL tối thiểu mỗi dim (nếu bật)

# ── Đường dẫn artifacts ──────────────────────────────────────────────────────
DATA_TRAIN_DIR   = PROJECT_ROOT / "data" / "train"
DATA_VAL_DIR     = PROJECT_ROOT / "data" / "validation"
ARTIFACTS_DIR    = PROJECT_ROOT / "artifacts" / "models"

MODEL_CHECKPOINT  = ARTIFACTS_DIR / "vae_best.pth"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
HISTORY_PATH      = ARTIFACTS_DIR / "training_history.json"
SUMMARY_PATH      = ARTIFACTS_DIR / "training_summary.json"


# ────────────────────────────────────────────────────────────────────────────
# 1b. KL Annealing — tính beta động theo epoch
# ────────────────────────────────────────────────────────────────────────────

def get_beta_for_epoch(
    epoch: int,
    beta_start: float,
    beta_end: float,
    warmup_epochs: int,
    annealing_type: str,
) -> float:
    """
    Tính giá trị beta cho epoch hiện tại theo lịch trình KL annealing.

    Mục đích: Tránh posterior collapse bằng cách bắt đầu với beta thấp
    (chỉ học reconstruction), sau đó tăng dần KL penalty.

    Args:
        epoch          : Epoch hiện tại (bắt đầu từ 1)
        beta_start     : Beta ban đầu (thường = 0.0)
        beta_end       : Beta mục tiêu sau warm-up (thường = 1.0)
        warmup_epochs  : Số epoch để tăng từ beta_start → beta_end
        annealing_type : Kiểu tăng: 'linear' | 'quadratic' | 'constant'

    Returns:
        beta: Giá trị beta cho epoch này, trong [beta_start, beta_end]
    """
    # Nếu không warm-up hoặc constant: trả beta_end ngay
    if warmup_epochs <= 0 or annealing_type == "constant":
        return beta_end

    # Tỉ lệ tiến độ trong [0, 1]
    progress = min(float(epoch) / float(warmup_epochs), 1.0)

    if annealing_type == "linear":
        # Tăng đều: beta = beta_start + (beta_end - beta_start) * t
        beta = beta_start + (beta_end - beta_start) * progress
    elif annealing_type == "quadratic":
        # Tăng chậm lúc đầu, nhanh dần: beta = beta_start + (beta_end - beta_start) * t²
        beta = beta_start + (beta_end - beta_start) * (progress ** 2)
    else:
        raise ValueError(
            f"annealing_type không hợp lệ: '{annealing_type}'. "
            "Chọn: linear | quadratic | constant"
        )

    return float(beta)


# ────────────────────────────────────────────────────────────────────────────
# 1. Thiết lập seed cho tái tạo kết quả
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Đặt random seed cho Python, NumPy và PyTorch để kết quả reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Đảm bảo deterministic mode cho CUDA (có thể chậm hơn một chút)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ────────────────────────────────────────────────────────────────────────────
# 2. Tải dữ liệu đã tiền xử lý
# ────────────────────────────────────────────────────────────────────────────

def load_data(
    train_dir: Path,
    val_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tải X_train và X_val từ thư mục đã xử lý.
    Trả về: (X_train, X_val) dạng numpy array float32.
    """
    x_train_path = train_dir / "X_train.npy"
    x_val_path   = val_dir   / "X_val.npy"

    # Kiểm tra file tồn tại
    for p in [x_train_path, x_val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu: {p}\n"
                "Hãy chạy scripts/clean_data.py trước."
            )

    print("Đang tải dữ liệu huấn luyện...")
    X_train = np.load(x_train_path).astype(np.float32)
    X_val   = np.load(x_val_path).astype(np.float32)

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    return X_train, X_val


# ────────────────────────────────────────────────────────────────────────────
# 3. Kiểm tra dữ liệu đầu vào
# ────────────────────────────────────────────────────────────────────────────

def validate_data(X_train: np.ndarray, X_val: np.ndarray) -> None:
    """
    Xác nhận dữ liệu đầu vào không chứa NaN, Inf và có đúng số feature.
    Dừng sớm nếu bất kỳ điều kiện nào bị vi phạm.
    """
    assert X_train.shape[1] == INPUT_DIM, (
        f"X_train có {X_train.shape[1]} features, kỳ vọng {INPUT_DIM}"
    )
    assert X_val.shape[1] == INPUT_DIM, (
        f"X_val có {X_val.shape[1]} features, kỳ vọng {INPUT_DIM}"
    )
    assert not np.isnan(X_train).any(), "X_train chứa giá trị NaN"
    assert not np.isnan(X_val).any(),   "X_val chứa giá trị NaN"
    assert not np.isinf(X_train).any(), "X_train chứa giá trị Inf"
    assert not np.isinf(X_val).any(),   "X_val chứa giá trị Inf"

    print("Kiểm tra dữ liệu: OK (không có NaN/Inf, đúng số feature)")


# ────────────────────────────────────────────────────────────────────────────
# 4. Tạo DataLoader
# ────────────────────────────────────────────────────────────────────────────

def make_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Chuyển numpy arrays thành TensorDataset và tạo DataLoader.
    Train loader: shuffle=True để tránh overfitting.
    Val loader  : shuffle=False để kết quả ổn định.
    """
    train_tensor = torch.from_numpy(X_train)
    val_tensor   = torch.from_numpy(X_val)

    train_ds = TensorDataset(train_tensor)
    val_ds   = TensorDataset(val_tensor)

    # Số worker phụ thuộc vào hệ điều hành (Windows có giới hạn với fork)
    num_workers = 0  # Dùng 0 để tương thích Windows

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ────────────────────────────────────────────────────────────────────────────
# 5. Một epoch huấn luyện
# ────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    grad_clip: float,
    use_free_bits: bool = False,
    free_bits_lam: float = 0.05,
) -> dict[str, float]:
    """
    Huấn luyện model qua một epoch.
    Trả về dict chứa các giá trị loss trung bình (total, recon, kl).

    Args:
        beta          : Hệ số KL hiện tại (dynamic — từ KL annealing schedule)
        use_free_bits : Nếu True, dùng free bits để tránh KL collapse từng dim
        free_bits_lam : Ngưỡng KL tối thiểu mỗi latent dim (nats)
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    n_batches      = 0

    # Chọn hàm loss phù hợp: chuẩn hoặc free bits
    loss_fn = vae_loss_free_bits if use_free_bits else vae_loss

    for (batch_x,) in loader:
        batch_x = batch_x.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        x_hat, mu, logvar = model(batch_x)

        # Tính loss với beta động
        if use_free_bits:
            loss, recon, kl = loss_fn(batch_x, x_hat, mu, logvar,
                                      beta=beta, lambda_free_bits=free_bits_lam)
        else:
            loss, recon, kl = loss_fn(batch_x, x_hat, mu, logvar, beta=beta)

        # Backward pass
        loss.backward()

        # Gradient clipping: ngăn gradient exploding
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum    += kl.item()
        n_batches      += 1

    return {
        "loss"  : total_loss_sum / n_batches,
        "recon" : recon_loss_sum / n_batches,
        "kl"    : kl_loss_sum    / n_batches,
    }


# ────────────────────────────────────────────────────────────────────────────
# 6. Đánh giá trên validation set
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    use_free_bits: bool = False,
    free_bits_lam: float = 0.05,
) -> dict[str, float]:
    """
    Đánh giá model trên validation set mà không cập nhật gradient.
    Trả về dict chứa các giá trị loss trung bình.
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0
    n_batches      = 0

    loss_fn = vae_loss_free_bits if use_free_bits else vae_loss

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)

            x_hat, mu, logvar = model(batch_x)
            if use_free_bits:
                loss, recon, kl = loss_fn(batch_x, x_hat, mu, logvar,
                                          beta=beta, lambda_free_bits=free_bits_lam)
            else:
                loss, recon, kl = loss_fn(batch_x, x_hat, mu, logvar, beta=beta)

            total_loss_sum += loss.item()
            recon_loss_sum += recon.item()
            kl_loss_sum    += kl.item()
            n_batches      += 1

    return {
        "loss"  : total_loss_sum / n_batches,
        "recon" : recon_loss_sum / n_batches,
        "kl"    : kl_loss_sum    / n_batches,
    }


# ────────────────────────────────────────────────────────────────────────────
# 7. Lưu artifacts
# ────────────────────────────────────────────────────────────────────────────

def save_model_config(
    output_path: Path,
    input_dim: int,
    latent_dim: int,
    hidden_dims: list[int],
    beta: float,
) -> None:
    """Lưu cấu hình kiến trúc model để có thể tái tạo lại sau này."""
    config = {
        "architecture": "VAE",
        "input_dim"   : input_dim,
        "latent_dim"  : latent_dim,
        "hidden_dims" : hidden_dims,
        "beta"        : beta,
        "activation"  : "ReLU",
        "output_activation": "Linear",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu model config: {output_path}")


def save_training_history(
    output_path: Path,
    history: dict,
) -> None:
    """Lưu lịch sử loss theo từng epoch để phân tích sau."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu training history: {output_path}")


def save_training_summary(
    output_path: Path,
    summary: dict,
) -> None:
    """Lưu tóm tắt quá trình huấn luyện: hyperparameters, kết quả, thời gian."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu training summary: {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# 8. Early stopping helper
# ────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Dừng huấn luyện sớm nếu validation loss không giảm sau `patience` epoch.
    Lưu checkpoint của model tốt nhất.
    """

    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience          = patience
        self.checkpoint_path   = checkpoint_path
        self.best_val_loss     = float("inf")
        self.epochs_no_improve = 0
        self.best_epoch        = 0

    def step(self, val_loss: float, model: VAE, epoch: int) -> bool:
        """
        Cập nhật trạng thái early stopping.
        Trả về True nếu nên dừng huấn luyện.
        """
        if val_loss < self.best_val_loss:
            # Cải thiện: lưu checkpoint và reset counter
            self.best_val_loss     = val_loss
            self.epochs_no_improve = 0
            self.best_epoch        = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  → Model tốt nhất được lưu (val_loss={val_loss:.6f})")
        else:
            self.epochs_no_improve += 1
            print(
                f"  → Không cải thiện {self.epochs_no_improve}/{self.patience}"
            )
            if self.epochs_no_improve >= self.patience:
                return True  # Dừng lại
        return False


# ────────────────────────────────────────────────────────────────────────────
# 9. Pipeline huấn luyện chính
# ────────────────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    """
    Hàm chính điều phối toàn bộ quá trình huấn luyện VAE.
    """
    start_time = time.time()

    # Đặt seed trước tiên để tái tạo kết quả
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Chọn device: ưu tiên CUDA, fallback sang CPU
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Sử dụng CPU")

    # Tạo thư mục output nếu chưa tồn tại
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Bước 1: Tải và kiểm tra dữ liệu ─────────────────────────────────────
    print("\n=== Bước 1: Tải dữ liệu ===")
    X_train, X_val = load_data(DATA_TRAIN_DIR, DATA_VAL_DIR)
    validate_data(X_train, X_val)

    # ── Bước 2: Tạo DataLoader ───────────────────────────────────────────────
    print("\n=== Bước 2: Tạo DataLoader ===")
    train_loader, val_loader = make_loaders(X_train, X_val, args.batch_size)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Bước 3: Khởi tạo model ───────────────────────────────────────────────
    print("\n=== Bước 3: Khởi tạo VAE ===")
    model = VAE(
        input_dim   = INPUT_DIM,
        latent_dim  = args.latent_dim,
        hidden_dims = HIDDEN_DIMS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Kiến trúc   : {INPUT_DIM} → {HIDDEN_DIMS} → (mu,logvar)[{args.latent_dim}]")
    print(f"  Số tham số  : {n_params:,}")

    # ── Bước 4: Thiết lập optimizer ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience        = args.patience,
        checkpoint_path = MODEL_CHECKPOINT,
    )

    # ── Bước 5: Vòng lặp huấn luyện ─────────────────────────────────────────
    print("\n=== Bước 5: Huấn luyện ===")
    history: dict[str, list] = {
        "train_loss"  : [],
        "train_recon" : [],
        "train_kl"    : [],
        "val_loss"    : [],
        "val_recon"   : [],
        "val_kl"      : [],
        "beta"        : [],   # Beta thực tế dùng trong mỗi epoch (KL annealing)
    }

    stopped_early = False
    final_epoch   = args.epochs

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Huấn luyện
        # Tính beta động cho epoch này theo KL annealing schedule
        current_beta = get_beta_for_epoch(
            epoch            = epoch,
            beta_start       = args.beta_start,
            beta_end         = args.beta_end,
            warmup_epochs    = args.beta_warmup_epochs,
            annealing_type   = args.kl_annealing_type,
        )

        # Huấn luyện một epoch với beta hiện tại
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            beta          = current_beta,
            grad_clip     = args.grad_clip,
            use_free_bits = args.use_free_bits,
            free_bits_lam = args.free_bits_lambda,
        )
        # Đánh giá trên val set với cùng beta
        val_metrics = evaluate(
            model, val_loader, device,
            beta          = current_beta,
            use_free_bits = args.use_free_bits,
            free_bits_lam = args.free_bits_lambda,
        )

        # Ghi lịch sử loss và beta theo epoch
        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])
        history["beta"].append(current_beta)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Beta={current_beta:.4f} | "
            f"Train Loss={train_metrics['loss']:.6f} "
            f"(recon={train_metrics['recon']:.6f}, kl={train_metrics['kl']:.6f}) | "
            f"Val Loss={val_metrics['loss']:.6f} "
            f"(recon={val_metrics['recon']:.6f}, kl={val_metrics['kl']:.6f}) | "
            f"Time={elapsed:.1f}s"
        )

        # Kiểm tra early stopping
        # - Trong warmup (epoch <= beta_warmup_epochs): beta đang tăng nên total
        #   val_loss tự nhiên tăng. Không fire early stopping, nhưng vẫn lưu
        #   checkpoint nếu val_recon cải thiện.
        # - Bước vào post-warmup: reset early stopping để so sánh total val_loss mới.
        # - Sau warmup: monitor total val_loss bình thường.
        in_warmup = epoch <= args.beta_warmup_epochs
        just_entered_post_warmup = (epoch == args.beta_warmup_epochs + 1)

        if just_entered_post_warmup:
            # Reset để post-warmup so sánh total val_loss từ đầu (không bị ảnh hưởng
            # bởi val_recon nhỏ trong warmup)
            early_stopping.best_val_loss     = float("inf")
            early_stopping.epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_CHECKPOINT)
            early_stopping.best_val_loss = val_metrics["loss"]
            early_stopping.best_epoch    = epoch
            print(f"  → [Post-Warmup] Bắt đầu monitor total val_loss={val_metrics['loss']:.6f}")
            should_stop = False
        elif in_warmup:
            # Chỉ lưu checkpoint nếu val_recon cải thiện, không đếm patience
            if val_metrics["recon"] < early_stopping.best_val_loss:
                early_stopping.best_val_loss     = val_metrics["recon"]
                early_stopping.best_epoch        = epoch
                early_stopping.epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_CHECKPOINT)
                print(f"  → [Warmup] Checkpoint lưu (val_recon={val_metrics['recon']:.6f})")
            else:
                print(f"  → [Warmup] beta={current_beta:.4f}, val_recon={val_metrics['recon']:.6f}")
            should_stop = False
        else:
            # Sau warmup: dùng total val_loss, kích hoạt early stopping bình thường
            should_stop = early_stopping.step(val_metrics["loss"], model, epoch)
        if should_stop:
            print(f"\nEarly stopping kích hoạt sau {epoch} epoch.")
            stopped_early = True
            final_epoch   = epoch
            break

    # ── Bước 6: Tải lại model tốt nhất để xác nhận ──────────────────────────
    print(f"\nTải model tốt nhất từ epoch {early_stopping.best_epoch}...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    # Đánh giá lại với model tốt nhất — dùng beta_end (full KL)
    final_val = evaluate(
        model, val_loader, device,
        beta          = args.beta_end,
        use_free_bits = args.use_free_bits,
        free_bits_lam = args.free_bits_lambda,
    )
    print(
        f"Val Loss cuối cùng: {final_val['loss']:.6f} "
        f"(recon={final_val['recon']:.6f}, kl={final_val['kl']:.6f})"
    )

    # ── Bước 7: Lưu tất cả artifacts ─────────────────────────────────────────
    print("\n=== Bước 7: Lưu artifacts ===")

    total_time = time.time() - start_time

    # Lưu cấu hình kiến trúc (ghi cả beta_end để backward compatible)
    save_model_config(
        MODEL_CONFIG_PATH,
        input_dim   = INPUT_DIM,
        latent_dim  = args.latent_dim,
        hidden_dims = HIDDEN_DIMS,
        beta        = args.beta_end,
    )

    # Lưu lịch sử training
    save_training_history(HISTORY_PATH, history)

    # Lưu tóm tắt training
    summary = {
        "hyperparameters": {
            "input_dim"    : INPUT_DIM,
            "latent_dim"   : args.latent_dim,
            "hidden_dims"  : HIDDEN_DIMS,
            "beta"         : args.beta_end,
            "learning_rate": args.lr,
            "weight_decay" : args.weight_decay,
            "batch_size"   : args.batch_size,
            "max_epochs"   : args.epochs,
            "patience"     : args.patience,
            "grad_clip"    : args.grad_clip,
            "seed"         : args.seed,
        },
        "kl_annealing": {
            "beta_start"         : args.beta_start,
            "beta_end"           : args.beta_end,
            "beta_warmup_epochs" : args.beta_warmup_epochs,
            "kl_annealing_type" : args.kl_annealing_type,
            "use_free_bits"      : args.use_free_bits,
            "free_bits_lambda"   : args.free_bits_lambda,
        },
        "results": {
            "best_epoch"       : early_stopping.best_epoch,
            "final_epoch"      : final_epoch,
            "stopped_early"    : stopped_early,
            "best_val_loss"    : early_stopping.best_val_loss,
            "best_val_recon"   : history["val_recon"][early_stopping.best_epoch - 1],
            "best_val_kl"      : history["val_kl"][early_stopping.best_epoch - 1],
            "n_params"         : n_params,
            "n_train_samples"  : int(X_train.shape[0]),
            "n_val_samples"    : int(X_val.shape[0]),
        },
        "device"    : str(device),
        "total_time_seconds": round(total_time, 2),
        "artifacts": {
            "checkpoint"   : str(MODEL_CHECKPOINT),
            "model_config" : str(MODEL_CONFIG_PATH),
            "history"      : str(HISTORY_PATH),
            "summary"      : str(SUMMARY_PATH),
        },
    }
    save_training_summary(SUMMARY_PATH, summary)

    print(f"\n=== Huấn luyện hoàn tất ===")
    print(f"Thời gian tổng: {total_time:.1f}s ({total_time/60:.1f} phút)")
    print(f"Best epoch    : {early_stopping.best_epoch}")
    print(f"Best val loss : {early_stopping.best_val_loss:.6f}")
    print(f"Checkpoint    : {MODEL_CHECKPOINT}")


# ────────────────────────────────────────────────────────────────────────────
# 10. Phân tích tham số dòng lệnh
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Phân tích các tham số từ command line với giá trị mặc định từ spec."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện VAE cho NIDS CICIDS2017",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help="Số epoch tối đa",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Kích thước batch",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate cho Adam optimizer",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY,
        help="L2 weight decay cho Adam optimizer",
    )
    parser.add_argument(
        "--patience", type=int, default=PATIENCE,
        help="Số epoch chịu đựng không cải thiện (early stopping)",
    )
    parser.add_argument(
        "--beta", type=float, default=BETA,
        help="[Deprecated — dùng --beta-end] Hệ số KL cuối (backward compat)",
    )
    parser.add_argument(
        "--beta-end", type=float, default=None,
        help="Beta mục tiêu sau warm-up (mặc định = --beta hoặc 1.0)",
    )
    parser.add_argument(
        "--beta-start", type=float, default=BETA_START,
        help="Beta ban đầu (0.0 = chỉ học reconstruction đầu tiên)",
    )
    parser.add_argument(
        "--beta-warmup-epochs", type=int, default=BETA_WARMUP_EPOCHS,
        help="Số epoch warm-up để beta tăng từ beta_start → beta_end",
    )
    parser.add_argument(
        "--kl-annealing-type", type=str, default=KL_ANNEALING_TYPE,
        choices=["linear", "quadratic", "constant"],
        help="Kiểu KL annealing schedule",
    )
    parser.add_argument(
        "--use-free-bits", action="store_true", default=USE_FREE_BITS,
        help="Bật Free Bits để tránh latent dim bị collapse về prior",
    )
    parser.add_argument(
        "--free-bits-lambda", type=float, default=FREE_BITS_LAMBDA,
        help="Ngưỡng KL tối thiểu mỗi latent dim khi dùng free bits (nats)",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=LATENT_DIM,
        help="Số chiều không gian tiềm ẩn",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=GRAD_CLIP_NORM,
        help="Max norm cho gradient clipping",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed cho tái tạo kết quả",
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Bắt buộc dùng CPU ngay cả khi CUDA khả dụng",
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    # Giải quyết --beta-end: nếu không truyền, dùng --beta (backward compat)
    if args.beta_end is None:
        args.beta_end = args.beta
    run_training(args)
