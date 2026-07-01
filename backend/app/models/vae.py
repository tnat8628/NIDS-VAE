"""
backend/app/models/vae.py
--------------------------
Định nghĩa kiến trúc Variational Autoencoder (VAE) cho bài toán phát hiện
xâm nhập mạng (NIDS).

Kiến trúc:
  Encoder: input_dim → hidden_dims → (mu, logvar) [latent_dim]
  Decoder: latent_dim → reversed(hidden_dims) → input_dim (linear output)

File này là nguồn duy nhất cho kiến trúc VAE — được dùng chung bởi
scripts/train.py (huấn luyện) và backend inference.

Công thức reconstruction error dùng cho inference:
  error_i = mean((x_i - x_hat_i)^2)   (MSE theo chiều feature, mỗi mẫu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder cho bài toán phát hiện anomaly trên network flow.

    Tham số:
        input_dim   : số lượng đặc trưng đầu vào (66 với CICIDS2017)
        latent_dim  : số chiều không gian tiềm ẩn
        hidden_dims : danh sách số units của các hidden layer encoder
                      (decoder dùng thứ tự ngược lại)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # ── Encoder ──────────────────────────────────────────────────────────
        # Xây dựng encoder: input_dim → hidden_dims → (mu, logvar)
        encoder_layers: list[nn.Module] = []
        in_features = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_features, h_dim))
            encoder_layers.append(nn.ReLU())
            in_features = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Hai nhánh song song để xuất mu và log-variance
        self.fc_mu     = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Xây dựng decoder: latent_dim → reversed(hidden_dims) → input_dim
        # Output activation: Linear (không dùng sigmoid/tanh vì data đã được scale)
        decoder_layers: list[nn.Module] = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_features, h_dim))
            decoder_layers.append(nn.ReLU())
            in_features = h_dim

        # Lớp output cuối — activation tuyến tính để tái tạo giá trị đã scale
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ── Encoder helpers ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chạy qua encoder, trả về (mu, logvar) của phân phối tiềm ẩn.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std, eps ~ N(0,1)
        Trong quá trình training, lấy mẫu ngẫu nhiên để gradient có thể lan truyền.
        Trong inference (model.eval()), trả về mu để cho kết quả deterministic.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # Inference: dùng mean để reconstruction error ổn định và tái tạo được
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Chạy qua decoder để tái tạo input từ vector tiềm ẩn z."""
        return self.decoder(z)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass hoàn chỉnh.
        Trả về: (x_hat, mu, logvar)
          x_hat  : reconstruction của x
          mu     : mean của phân phối tiềm ẩn
          logvar : log-variance của phân phối tiềm ẩn
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    # ── Reconstruction error (dùng cho inference / thresholding) ─────────────

    @staticmethod
    def reconstruction_error(
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính MSE reconstruction error cho từng mẫu.
        Công thức: error_i = mean((x_i - x_hat_i)^2) theo chiều feature.

        Args:
            x     : input gốc, shape (batch, input_dim)
            x_hat : reconstruction, shape (batch, input_dim)
        Returns:
            errors: shape (batch,) — một giá trị lỗi cho mỗi mẫu
        """
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tính VAE loss = Reconstruction Loss + beta * KL Divergence.

    Reconstruction Loss: MSE trung bình trên toàn batch và features.
    KL Divergence: -0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar)))

    Args:
        x      : input gốc
        x_hat  : reconstruction
        mu     : mean của phân phối tiềm ẩn
        logvar : log-variance của phân phối tiềm ẩn
        beta   : hệ số cân bằng giữa reconstruction và KL (mặc định 1.0)

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss: MSE trung bình trên batch và features
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")

    # KL divergence: đo khoảng cách giữa q(z|x) và prior N(0,I)
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) cho mỗi mẫu, rồi trung bình
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    kl_loss = kl_per_sample.mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def vae_loss_free_bits(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    lambda_free_bits: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss với Free Bits — ràng buộc KL tối thiểu mỗi latent dim.

    Mục đích: Tránh posterior collapse bằng cách đảm bảo mỗi chiều tiềm ẩn
    phải encode ít nhất `lambda_free_bits` nats thông tin từ input.

    Công thức KL mỗi dim: max(lambda_free_bits, KL_dim_j)
    Tổng KL = sum_j max(λ, kl_j), average over batch.

    Args:
        x                : Input gốc
        x_hat            : Reconstruction
        mu               : Mean của phân phối tiềm ẩn
        logvar           : Log-variance của phân phối tiềm ẩn
        beta             : Hệ số cân bằng KL (thường từ annealing schedule)
        lambda_free_bits : Ngưỡng KL tối thiểu mỗi dim (nats), thường 0.05–0.25

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss: MSE trung bình trên batch và features
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")

    # KL divergence per dimension (không sum theo dim, để clamp từng dim)
    # kl_per_dim shape: (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Free bits: đảm bảo mỗi dim có KL >= lambda_free_bits
    kl_clamped = torch.clamp(kl_per_dim, min=lambda_free_bits)

    # Tổng theo latent dim, mean theo batch
    kl_loss = kl_clamped.sum(dim=1).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
