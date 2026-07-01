"""
demo_vae_encoder.py
-------------------
Mô phỏng toàn bộ quá trình Encoder của VAE theo kiến trúc 66 → 128 → 64
của dự án NIDS-VAE, sử dụng thuần numpy (không PyTorch / TensorFlow).

Các phần chính:
  1. Dữ liệu giả lập: flow DoS với giá trị bất thường
  2. Tính toán từng bước qua lớp encoder (W1, h1, W2, h2)
  3. Minh họa trọng số học được sau 10 epoch huấn luyện đơn giản
  4. So sánh biểu diễn latent của flow BENIGN vs DoS
"""

import numpy as np

# ---------------------------------------------------------------------------
# Tên 66 cột đặc trưng theo chuẩn CICIDS2017 (khớp với feature_columns.json)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

assert len(FEATURE_NAMES) == 66, "Phải có đúng 66 tên cột"

# ===========================================================================
# PHẦN 1 – Dữ liệu giả lập: flow DoS
# ===========================================================================

def separator(title: str) -> None:
    """In tiêu đề phân cách giữa các phần cho dễ đọc."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def create_dos_flow() -> np.ndarray:
    """
    Tạo vector 66 chiều mô phỏng một flow DoS điển hình:
    - Flow Duration rất ngắn (tấn công dồn dập)
    - Flow Bytes/s và Flow Packets/s rất cao
    - Phần lớn các cờ (SYN, ACK) bật liên tục
    - Kích thước gói tin nhỏ, đều nhau
    """
    x = np.zeros(66)

    # Destination Port – cổng HTTP thông thường (bình thường)
    x[0]  = 80.0

    # Flow Duration (µs) – rất ngắn → dấu hiệu flood
    x[1]  = 1500.0

    # Total Fwd / Bwd Packets – số lượng gói lớn bất thường
    x[2]  = 950.0
    x[3]  = 5.0

    # Total Length of Fwd / Bwd Packets (bytes)
    x[4]  = 57000.0
    x[5]  = 300.0

    # Fwd Packet Length Max/Min/Mean/Std – gói nhỏ đồng đều
    x[6]  = 60.0
    x[7]  = 60.0
    x[8]  = 60.0
    x[9]  = 0.0

    # Bwd Packet Length Max/Min/Mean/Std
    x[10] = 60.0
    x[11] = 60.0
    x[12] = 60.0
    x[13] = 0.0

    # Flow Bytes/s và Flow Packets/s – RẤT CAO (đặc trưng DoS)
    x[14] = 38000000.0
    x[15] = 633333.0

    # Flow IAT Mean/Std/Max/Min (µs) – inter-arrival time cực nhỏ
    x[16] = 2.4
    x[17] = 0.5
    x[18] = 5.0
    x[19] = 1.0

    # Fwd IAT Total/Mean/Std/Max/Min
    x[20] = 1500.0
    x[21] = 1.6
    x[22] = 0.4
    x[23] = 3.0
    x[24] = 1.0

    # Bwd IAT Total/Mean/Std/Max/Min – gần như không có phản hồi
    x[25] = 100.0
    x[26] = 100.0
    x[27] = 0.0
    x[28] = 100.0
    x[29] = 100.0

    # Fwd PSH Flags
    x[30] = 0.0

    # Fwd Header Length / Bwd Header Length
    x[31] = 19000.0
    x[32] = 100.0

    # Fwd Packets/s / Bwd Packets/s
    x[33] = 633333.0
    x[34] = 3333.0

    # Min / Max Packet Length
    x[35] = 60.0
    x[36] = 60.0

    # Packet Length Mean / Std / Variance
    x[37] = 60.0
    x[38] = 0.0
    x[39] = 0.0

    # Flag Counts – SYN flood: nhiều SYN, nhiều ACK
    x[40] = 0.0   # FIN
    x[41] = 950.0 # SYN – đặc trưng SYN flood
    x[42] = 0.0   # PSH
    x[43] = 950.0 # ACK
    x[44] = 0.0   # URG

    # Down/Up Ratio – tỉ lệ thấp (upload nhiều, download ít)
    x[45] = 0.005

    # Average Packet Size / Avg Fwd Segment Size / Avg Bwd Segment Size
    x[46] = 60.0
    x[47] = 60.0
    x[48] = 60.0

    # Fwd Header Length.1
    x[49] = 19000.0

    # Subflow Fwd Packets / Bytes / Bwd Packets / Bytes
    x[50] = 950.0
    x[51] = 57000.0
    x[52] = 5.0
    x[53] = 300.0

    # Init_Win_bytes_forward / backward
    x[54] = 65535.0
    x[55] = 229.0

    # act_data_pkt_fwd / min_seg_size_forward
    x[56] = 0.0
    x[57] = 20.0

    # Active Mean / Std / Max / Min (µs)
    x[58] = 0.0
    x[59] = 0.0
    x[60] = 0.0
    x[61] = 0.0

    # Idle Mean / Std / Max / Min (µs)
    x[62] = 0.0
    x[63] = 0.0
    x[64] = 0.0
    x[65] = 0.0

    return x


def create_benign_flow() -> np.ndarray:
    """
    Tạo vector 66 chiều mô phỏng một flow BENIGN (traffic bình thường):
    - Flow Duration vừa phải
    - Flow Bytes/s và Packets/s ở mức bình thường
    - Tỉ lệ Fwd/Bwd cân đối
    """
    x = np.zeros(66)

    x[0]  = 443.0        # Destination Port – HTTPS
    x[1]  = 2500000.0    # Flow Duration (µs) – ~2.5 giây
    x[2]  = 12.0         # Total Fwd Packets
    x[3]  = 10.0         # Total Backward Packets
    x[4]  = 8400.0       # Total Length of Fwd Packets
    x[5]  = 12000.0      # Total Length of Bwd Packets
    x[6]  = 1200.0       # Fwd Packet Length Max
    x[7]  = 40.0         # Fwd Packet Length Min
    x[8]  = 700.0        # Fwd Packet Length Mean
    x[9]  = 380.0        # Fwd Packet Length Std
    x[10] = 1400.0       # Bwd Packet Length Max
    x[11] = 52.0         # Bwd Packet Length Min
    x[12] = 1200.0       # Bwd Packet Length Mean
    x[13] = 290.0        # Bwd Packet Length Std
    x[14] = 8160.0       # Flow Bytes/s – bình thường
    x[15] = 8.8          # Flow Packets/s – bình thường
    x[16] = 113636.0     # Flow IAT Mean (µs)
    x[17] = 200000.0     # Flow IAT Std
    x[18] = 800000.0     # Flow IAT Max
    x[19] = 500.0        # Flow IAT Min
    x[20] = 2000000.0    # Fwd IAT Total
    x[21] = 200000.0     # Fwd IAT Mean
    x[22] = 280000.0     # Fwd IAT Std
    x[23] = 900000.0     # Fwd IAT Max
    x[24] = 1000.0       # Fwd IAT Min
    x[25] = 1800000.0    # Bwd IAT Total
    x[26] = 200000.0     # Bwd IAT Mean
    x[27] = 250000.0     # Bwd IAT Std
    x[28] = 750000.0     # Bwd IAT Max
    x[29] = 2000.0       # Bwd IAT Min
    x[30] = 1.0          # Fwd PSH Flags
    x[31] = 480.0        # Fwd Header Length
    x[32] = 400.0        # Bwd Header Length
    x[33] = 4.8          # Fwd Packets/s
    x[34] = 4.0          # Bwd Packets/s
    x[35] = 40.0         # Min Packet Length
    x[36] = 1400.0       # Max Packet Length
    x[37] = 930.0        # Packet Length Mean
    x[38] = 490.0        # Packet Length Std
    x[39] = 240000.0     # Packet Length Variance
    x[40] = 1.0          # FIN Flag Count
    x[41] = 1.0          # SYN Flag Count
    x[42] = 2.0          # PSH Flag Count
    x[43] = 18.0         # ACK Flag Count
    x[44] = 0.0          # URG Flag Count
    x[45] = 0.83         # Down/Up Ratio
    x[46] = 930.0        # Average Packet Size
    x[47] = 700.0        # Avg Fwd Segment Size
    x[48] = 1200.0       # Avg Bwd Segment Size
    x[49] = 480.0        # Fwd Header Length.1
    x[50] = 12.0         # Subflow Fwd Packets
    x[51] = 8400.0       # Subflow Fwd Bytes
    x[52] = 10.0         # Subflow Bwd Packets
    x[53] = 12000.0      # Subflow Bwd Bytes
    x[54] = 65535.0      # Init_Win_bytes_forward
    x[55] = 65535.0      # Init_Win_bytes_backward
    x[56] = 8.0          # act_data_pkt_fwd
    x[57] = 20.0         # min_seg_size_forward
    x[58] = 350000.0     # Active Mean
    x[59] = 120000.0     # Active Std
    x[60] = 500000.0     # Active Max
    x[61] = 200000.0     # Active Min
    x[62] = 900000.0     # Idle Mean
    x[63] = 300000.0     # Idle Std
    x[64] = 1500000.0    # Idle Max
    x[65] = 400000.0     # Idle Min

    return x


# ===========================================================================
# PHẦN 2 – Khởi tạo trọng số và tính forward pass từng bước
# ===========================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """Hàm kích hoạt ReLU: trả về max(0, z) từng phần tử."""
    return np.maximum(0.0, z)


def init_weights(seed: int = 42) -> tuple:
    """
    Khởi tạo trọng số cho 2 lớp encoder theo kiến trúc 66→128→64.
    Dùng He initialization (phù hợp cho ReLU) để tránh gradient vanishing.

    Returns:
        W1 (128×66), b1 (128,), W2 (64×128), b2 (64,)
    """
    rng = np.random.default_rng(seed)

    # He initialization: std = sqrt(2 / n_in)
    W1 = rng.normal(0, np.sqrt(2.0 / 66),  size=(128, 66))
    b1 = np.zeros(128)

    W2 = rng.normal(0, np.sqrt(2.0 / 128), size=(64, 128))
    b2 = np.zeros(64)

    return W1, b1, W2, b2


def encoder_forward(
    x: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
) -> tuple:
    """
    Tính forward pass qua 2 lớp encoder.

    x (66,) → z1 (128,) → h1 (128,) → z2 (64,) → h2 (64,)

    z = W·x + b  (pre-activation)
    h = ReLU(z)  (post-activation)

    Returns:
        z1, h1, z2, h2
    """
    z1 = W1 @ x + b1   # pre-activation lớp 1: shape (128,)
    h1 = relu(z1)       # post-activation lớp 1

    z2 = W2 @ h1 + b2  # pre-activation lớp 2: shape (64,)
    h2 = relu(z2)       # post-activation lớp 2

    return z1, h1, z2, h2


def print_flow_input(x: np.ndarray) -> None:
    """In vector đầu vào x với tên cột tương ứng."""
    separator("PHẦN 1 – VECTOR ĐẦU VÀO (Flow DoS giả lập, 66 chiều)")
    print(f"\n{'Chỉ số':>6}  {'Tên cột':<40}  {'Giá trị':>15}")
    print("-" * 68)
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, x)):
        print(f"{i:>6}  {name:<40}  {val:>15.4f}")


def print_layer1_detail(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                        z1: np.ndarray, h1: np.ndarray) -> None:
    """
    In chi tiết 5 neuron đầu tiên của lớp 1 (lớp ẩn 128 neuron):
    - Tổng tích vô hướng W1[j]·x (trước bias)
    - Sau khi cộng bias b1[j]
    - Sau ReLU
    """
    separator("PHẦN 2a – LỚP 1: W1 (128×66) · x + b1 → ReLU → h1 (128,)")

    print("\nKiến trúc lớp 1:")
    print(f"  Đầu vào  : {x.shape[0]} chiều")
    print(f"  Ma trận  : W1 {W1.shape}  →  {W1.shape[0]} neuron đầu ra")
    print(f"  Bias     : b1 {b1.shape}")

    print("\nChi tiết 5 neuron đầu tiên:")
    print(f"\n{'Neuron':>8}  {'W1[j]·x (pre-bias)':>22}  "
          f"{'+ b1 (pre-ReLU)':>18}  {'ReLU (h1)':>12}")
    print("-" * 70)
    for j in range(5):
        dot_val  = W1[j] @ x          # tích vô hướng hàng j với x
        pre_relu = dot_val + b1[j]    # cộng bias → z1[j]
        post_act = h1[j]              # sau ReLU
        print(f"{j:>8}  {dot_val:>22.6f}  {pre_relu:>18.6f}  {post_act:>12.6f}")

    # Thống kê toàn lớp
    n_dead = int(np.sum(h1 == 0))
    print(f"\nToàn lớp h1 (128 neuron): {n_dead} neuron bị tắt (= 0 sau ReLU)"
          f"  [{n_dead/128*100:.1f}%]")
    print(f"  min h1 = {h1.min():.4f}  |  max h1 = {h1.max():.4f}"
          f"  |  mean h1 = {h1.mean():.4f}")


def print_layer2_detail(W2: np.ndarray, b2: np.ndarray,
                        z2: np.ndarray, h2: np.ndarray) -> None:
    """
    In chi tiết lớp 2 (64 neuron) tương tự lớp 1.
    Đầu vào của lớp này là h1 (128,).
    """
    separator("PHẦN 2b – LỚP 2: W2 (64×128) · h1 + b2 → ReLU → h2 (64,)")

    print("\nKiến trúc lớp 2:")
    print(f"  Đầu vào  : {W2.shape[1]} chiều (h1)")
    print(f"  Ma trận  : W2 {W2.shape}  →  {W2.shape[0]} neuron đầu ra")
    print(f"  Bias     : b2 {b2.shape}")

    print("\nChi tiết 5 neuron đầu tiên của h2:")
    print(f"\n{'Neuron':>8}  {'W2[j]·h1 (pre-bias)':>22}  "
          f"{'+ b2 (pre-ReLU)':>18}  {'ReLU (h2)':>12}")
    print("-" * 70)
    for j in range(5):
        pre_bias = z2[j] - b2[j]   # trích lại tích vô hướng từ z2
        pre_relu = z2[j]
        post_act = h2[j]
        print(f"{j:>8}  {pre_bias:>22.6f}  {pre_relu:>18.6f}  {post_act:>12.6f}")

    n_dead = int(np.sum(h2 == 0))
    print(f"\nToàn lớp h2 (64 neuron): {n_dead} neuron bị tắt (= 0 sau ReLU)"
          f"  [{n_dead/64*100:.1f}%]")
    print(f"  min h2 = {h2.min():.4f}  |  max h2 = {h2.max():.4f}"
          f"  |  mean h2 = {h2.mean():.4f}")


def print_compression_summary(x: np.ndarray,
                               h1: np.ndarray, h2: np.ndarray) -> None:
    """In bảng tóm tắt quá trình nén: 66 → 128 → 64."""
    separator("PHẦN 2c – SO SÁNH: x (66) → h1 (128) → h2 (64)")

    n_dead_h1 = int(np.sum(h1 == 0))
    n_dead_h2 = int(np.sum(h2 == 0))

    rows = [
        ("x   (đầu vào)", 66,  0,        0.0,    f"{x.min():.2f}", f"{x.max():.2f}"),
        ("h1  (lớp 1)",  128, n_dead_h1, n_dead_h1/128*100, f"{h1.min():.4f}", f"{h1.max():.4f}"),
        ("h2  (lớp 2)",   64, n_dead_h2, n_dead_h2/64*100,  f"{h2.min():.4f}", f"{h2.max():.4f}"),
    ]
    print(f"\n{'Tensor':<18}  {'Kích thước':>10}  {'Dead':>6}  "
          f"{'Dead%':>7}  {'Min':>12}  {'Max':>12}")
    print("-" * 74)
    for name, size, dead, pct, mn, mx in rows:
        print(f"{name:<18}  {size:>10}  {dead:>6}  {pct:>6.1f}%  "
              f"{mn:>12}  {mx:>12}")


# ===========================================================================
# PHẦN 3 – Autoencoder đơn giản 66→128→64→66 huấn luyện 10 epoch
# ===========================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid ổn định số (tránh overflow với giá trị âm lớn)."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def generate_benign_dataset(n: int = 100, seed: int = 7) -> np.ndarray:
    """
    Tạo 100 flow BENIGN giả lập: lấy flow BENIGN chuẩn rồi thêm nhiễu
    Gaussian nhỏ ±10% để tạo sự đa dạng trong tập huấn luyện.
    """
    rng   = np.random.default_rng(seed)
    base  = create_benign_flow()
    noise = rng.normal(0, 0.1, size=(n, 66))

    # Nhân nhiễu theo tỉ lệ giá trị gốc (relative noise)
    X = base + noise * np.abs(base + 1e-8)

    # Chuẩn hóa min-max đơn giản để ổn định việc huấn luyện
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom   = np.where(col_max - col_min > 1e-8, col_max - col_min, 1.0)
    X_norm  = (X - col_min) / denom
    return X_norm


def autoencoder_forward(x: np.ndarray,
                        W1: np.ndarray, b1: np.ndarray,
                        W2: np.ndarray, b2: np.ndarray,
                        W3: np.ndarray, b3: np.ndarray,
                        W4: np.ndarray, b4: np.ndarray) -> tuple:
    """
    Forward pass của autoencoder 4 lớp: 66 → 128 → 64 → 128 → 66.

    Encoder: x → h1 → h2
    Decoder: h2 → h3 → x_hat (không dùng activation ở lớp cuối)

    Returns:
        h1, h2, h3, x_hat
    """
    # --- Encoder ---
    h1    = relu(W1 @ x + b1)     # lớp ẩn 1: 128 neuron
    h2    = relu(W2 @ h1 + b2)    # lớp ẩn 2: 64 neuron (bottleneck)

    # --- Decoder (đối xứng với encoder) ---
    h3    = relu(W3 @ h2 + b3)    # lớp ẩn 3: 128 neuron
    x_hat = W4 @ h3 + b4          # đầu ra tái tạo: 66 chiều (linear)

    return h1, h2, h3, x_hat


def train_autoencoder(X_train: np.ndarray,
                      lr: float = 0.01,
                      n_epochs: int = 10,
                      seed: int = 42) -> tuple:
    """
    Huấn luyện autoencoder đơn giản bằng Gradient Descent (SGD batch).
    Hàm mất mát: Mean Squared Error (MSE) giữa x và x_hat.

    Backpropagation:
        - dL/dx_hat = 2*(x_hat - x)/n
        - Lan truyền ngược qua từng lớp theo chain rule

    Returns:
        W1, b1, W2, b2, W3, b3, W4, b4
        epoch_weights: dict chứa W1 snapshot tại epoch 1, 5, 10
    """
    rng = np.random.default_rng(seed)

    # Khởi tạo trọng số 4 lớp với He initialization
    W1 = rng.normal(0, np.sqrt(2.0 / 66),  (128, 66))
    b1 = np.zeros(128)
    W2 = rng.normal(0, np.sqrt(2.0 / 128), (64, 128))
    b2 = np.zeros(64)
    W3 = rng.normal(0, np.sqrt(2.0 / 64),  (128, 64))
    b3 = np.zeros(128)
    W4 = rng.normal(0, np.sqrt(2.0 / 128), (66, 128))
    b4 = np.zeros(66)

    n_samples = X_train.shape[0]
    epoch_weights: dict = {}  # lưu snapshot W1 theo epoch

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0

        # Xáo trộn dữ liệu mỗi epoch để tránh overfitting theo thứ tự
        idx = rng.permutation(n_samples)

        for i in idx:
            x = X_train[i]

            # --- Forward pass ---
            h1, h2, h3, x_hat = autoencoder_forward(
                x, W1, b1, W2, b2, W3, b3, W4, b4
            )

            # --- Tính loss MSE ---
            diff = x_hat - x
            loss = np.mean(diff ** 2)
            total_loss += loss

            # ----------------------------------------------------------------
            # Backpropagation (chain rule từ đầu ra về đầu vào)
            # ----------------------------------------------------------------

            # Gradient tại lớp đầu ra (linear activation, không cần mask)
            d_xhat = 2.0 * diff / 66          # dL/dx_hat: shape (66,)

            # Lớp 4: x_hat = W4·h3 + b4
            dW4 = np.outer(d_xhat, h3)
            db4 = d_xhat
            d_h3 = W4.T @ d_xhat              # gradient lan về h3

            # Lớp 3: h3 = ReLU(W3·h2 + b3)  — nhân mask ReLU
            d_h3 *= (h3 > 0)                  # đạo hàm ReLU: 1 nếu z>0
            dW3 = np.outer(d_h3, h2)
            db3 = d_h3
            d_h2 = W3.T @ d_h3               # gradient lan về h2

            # Lớp 2: h2 = ReLU(W2·h1 + b2)
            d_h2 *= (h2 > 0)
            dW2 = np.outer(d_h2, h1)
            db2 = d_h2
            d_h1 = W2.T @ d_h2               # gradient lan về h1

            # Lớp 1: h1 = ReLU(W1·x + b1)
            d_h1 *= (h1 > 0)
            dW1 = np.outer(d_h1, x)
            db1 = d_h1

            # Cập nhật trọng số (SGD, learning rate cố định)
            W1 -= lr * dW1;  b1 -= lr * db1
            W2 -= lr * dW2;  b2 -= lr * db2
            W3 -= lr * dW3;  b3 -= lr * db3
            W4 -= lr * dW4;  b4 -= lr * db4

        avg_loss = total_loss / n_samples

        # Lưu snapshot W1 tại epoch 1, 5, 10
        if epoch in (1, 5, 10):
            epoch_weights[epoch] = W1.copy()

        print(f"  Epoch {epoch:>2} / {n_epochs}  |  Loss trung bình: {avg_loss:.6f}")

    return W1, b1, W2, b2, W3, b3, W4, b4, epoch_weights


def print_top_weights(epoch_weights: dict, neuron_idx: int = 0) -> None:
    """
    In top 5 cột có trọng số tuyệt đối lớn nhất cho neuron `neuron_idx`
    của W1 tại các epoch 1, 5, 10 để thấy sự thay đổi.
    """
    separator(f"PHẦN 3b – TOP 5 CỘT ĐƯỢC W1 QUAN TÂM NHẤT (Neuron #{neuron_idx})")

    print(f"\nNeuron #{neuron_idx} của lớp ẩn thứ nhất theo dõi đầu vào nào nhất?\n")

    header = f"{'Epoch':<8}  {'Hạng':>5}  {'Cột':>4}  {'Tên cột':<40}  {'|w|':>10}"
    for epoch, W1_snap in sorted(epoch_weights.items()):
        row = W1_snap[neuron_idx]                       # trọng số 66 chiều
        top5_idx = np.argsort(np.abs(row))[::-1][:5]   # 5 chỉ số lớn nhất

        print(f"--- Epoch {epoch} ---")
        print(header)
        print("-" * 76)
        for rank, col_i in enumerate(top5_idx, 1):
            print(f"{'':8}  {rank:>5}  {col_i:>4}  "
                  f"{FEATURE_NAMES[col_i]:<40}  {abs(row[col_i]):>10.6f}")
        print()

    # So sánh sự thay đổi |w| từ epoch 1 → epoch 10 cho neuron này
    if 1 in epoch_weights and 10 in epoch_weights:
        w1_ep1  = epoch_weights[1][neuron_idx]
        w1_ep10 = epoch_weights[10][neuron_idx]
        change  = np.abs(w1_ep10) - np.abs(w1_ep1)

        top5_change = np.argsort(np.abs(change))[::-1][:5]
        print(f"Thay đổi |w| lớn nhất từ Epoch 1 → Epoch 10 (Neuron #{neuron_idx}):")
        print(f"{'Hạng':>5}  {'Cột':>4}  {'Tên cột':<40}  {'Δ|w|':>12}")
        print("-" * 64)
        for rank, col_i in enumerate(top5_change, 1):
            print(f"{rank:>5}  {col_i:>4}  "
                  f"{FEATURE_NAMES[col_i]:<40}  {change[col_i]:>+12.6f}")


# ===========================================================================
# PHẦN 4 – So sánh biểu diễn h2: BENIGN vs DoS
# ===========================================================================

def normalize_flow(x: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa min-max một flow đơn lẻ theo thống kê của tập tham chiếu X_ref.
    Đảm bảo flow DoS và BENIGN dùng cùng scale khi đẩy qua encoder.
    """
    col_min = X_ref.min(axis=0)
    col_max = X_ref.max(axis=0)
    denom   = np.where(col_max - col_min > 1e-8, col_max - col_min, 1.0)
    return (x - col_min) / denom


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Khoảng cách Euclidean giữa hai vector biểu diễn."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Khoảng cách cosine: 1 - cosine_similarity.
    = 0 nếu hai vector hoàn toàn giống hướng.
    = 1 nếu vuông góc (không liên quan).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def print_benign_vs_dos(
    W1_tr: np.ndarray, b1_tr: np.ndarray,
    W2_tr: np.ndarray, b2_tr: np.ndarray,
    X_train: np.ndarray,
) -> None:
    """
    Đẩy flow BENIGN và DoS qua encoder đã huấn luyện.
    In h2 của cả hai và tính các khoảng cách giữa chúng.
    """
    separator("PHẦN 4 – SO SÁNH BENIGN vs DoS QUA ENCODER ĐÃ TRAIN")

    # Tạo 2 flow và chuẩn hóa theo cùng thống kê tập train
    x_benign_raw = create_benign_flow()
    x_dos_raw    = create_dos_flow()

    # Mở rộng X_train bằng 2 flow để tính min/max nhất quán
    X_ref    = np.vstack([X_train,
                          x_benign_raw.reshape(1, -1),
                          x_dos_raw.reshape(1, -1)])
    x_benign = normalize_flow(x_benign_raw, X_ref)
    x_dos    = normalize_flow(x_dos_raw,    X_ref)

    # Forward pass chỉ qua encoder (2 lớp đầu)
    _, h2_benign, _, _ = autoencoder_forward(
        x_benign, W1_tr, b1_tr, W2_tr, b2_tr,
        np.zeros((128, 64)), np.zeros(128),  # W3/b3 không dùng ở encoder
        np.zeros((66, 128)),  np.zeros(66),  # W4/b4 không dùng ở encoder
    )
    _, h2_dos, _, _ = autoencoder_forward(
        x_dos, W1_tr, b1_tr, W2_tr, b2_tr,
        np.zeros((128, 64)), np.zeros(128),
        np.zeros((66, 128)),  np.zeros(66),
    )

    # --- In h2 của cả hai flow ---
    print("\nh2 BENIGN (64 chiều) – 10 giá trị đầu tiên:")
    print("  " + "  ".join(f"{v:8.4f}" for v in h2_benign[:10]))

    print("\nh2 DoS (64 chiều) – 10 giá trị đầu tiên:")
    print("  " + "  ".join(f"{v:8.4f}" for v in h2_dos[:10]))

    # --- Tính các khoảng cách ---
    euc  = euclidean_distance(h2_benign, h2_dos)
    cos  = cosine_distance(h2_benign, h2_dos)
    l1   = float(np.sum(np.abs(h2_benign - h2_dos)))

    print("\n--- Khoảng cách giữa biểu diễn BENIGN và DoS ---")
    print(f"  Euclidean distance  : {euc:.6f}")
    print(f"  Cosine distance     : {cos:.6f}  (0 = giống hướng, 1 = vuông góc)")
    print(f"  L1 (Manhattan)      : {l1:.6f}")

    # --- Thống kê nhanh ---
    n_diff  = int(np.sum(np.abs(h2_benign - h2_dos) > 0.01))
    n_total = 64
    print(f"\n  Số neuron khác nhau >0.01 : {n_diff}/{n_total}"
          f"  [{n_diff/n_total*100:.1f}%]")

    # Kết luận
    print("\n  Kết luận:")
    if euc > 1.0:
        print("  → Hai luồng cho ra biểu diễn KHÁC NHAU rõ ràng trong không gian h2.")
        print("    Encoder đã học phân tách các pattern BENIGN và DoS.")
    else:
        print("  → Hai luồng cho ra biểu diễn gần nhau; cần huấn luyện thêm.")

    # --- Bảng so sánh neuron theo neuron (10 neuron đầu) ---
    print("\n--- Chi tiết 10 neuron đầu của h2 ---")
    print(f"{'Neuron':>8}  {'h2_BENIGN':>12}  {'h2_DoS':>12}  {'|Δ|':>12}")
    print("-" * 52)
    for j in range(10):
        print(f"{j:>8}  {h2_benign[j]:>12.4f}  {h2_dos[j]:>12.4f}  "
              f"{abs(h2_benign[j] - h2_dos[j]):>12.4f}")


# ===========================================================================
# MAIN – Chạy tuần tự 4 phần
# ===========================================================================

def main() -> None:
    """Hàm chính: chạy toàn bộ demo từ Phần 1 đến Phần 4."""

    print("\n" + "#" * 72)
    print("#  DEMO: QUÁ TRÌNH ENCODER CỦA VAE (66 → 128 → 64)")
    print("#  Kiến trúc: NIDS-VAE / CICIDS2017  |  Numpy thuần")
    print("#" * 72)

    # ------------------------------------------------------------------ P1
    x_dos = create_dos_flow()
    print_flow_input(x_dos)

    # ------------------------------------------------------------------ P2
    W1, b1, W2, b2 = init_weights(seed=42)

    separator("PHẦN 2 – KHỞI TẠO TRỌNG SỐ (He initialization, seed=42)")
    print(f"\n  W1: shape={W1.shape},  std={W1.std():.6f}")
    print(f"  b1: shape={b1.shape},  (khởi tạo = 0)")
    print(f"  W2: shape={W2.shape},  std={W2.std():.6f}")
    print(f"  b2: shape={b2.shape},  (khởi tạo = 0)")

    z1, h1, z2, h2 = encoder_forward(x_dos, W1, b1, W2, b2)

    print_layer1_detail(x_dos, W1, b1, z1, h1)
    print_layer2_detail(W2, b2, z2, h2)
    print_compression_summary(x_dos, h1, h2)

    # ------------------------------------------------------------------ P3
    separator("PHẦN 3 – HUẤN LUYỆN AUTOENCODER 66→128→64→128→66 (10 epoch)")
    print("\nTạo 100 flow BENIGN giả lập và chuẩn hóa...")
    X_train = generate_benign_dataset(n=100, seed=7)
    print(f"  X_train shape: {X_train.shape}  |  min={X_train.min():.4f}"
          f"  max={X_train.max():.4f}")
    print("\nBắt đầu huấn luyện:")

    W1_tr, b1_tr, W2_tr, b2_tr, W3_tr, b3_tr, W4_tr, b4_tr, epoch_weights = \
        train_autoencoder(X_train, lr=0.005, n_epochs=10, seed=42)

    print_top_weights(epoch_weights, neuron_idx=0)

    # ------------------------------------------------------------------ P4
    print_benign_vs_dos(W1_tr, b1_tr, W2_tr, b2_tr, X_train)

    separator("HOÀN THÀNH")
    print("\nDemo kết thúc. Toàn bộ kết quả đã được in phía trên.\n")


if __name__ == "__main__":
    main()
