"""
backend/app/core/inference.py
-------------------------------
Dịch vụ suy diễn (inference) trung tâm cho hệ thống NIDS VAE.

Lớp VAEInferenceService chịu trách nhiệm:
  1. Tải và giữ các artifact (model, scaler, threshold, feature schema)
  2. Nhận DataFrame đầu vào, tiền xử lý, scale, chạy VAE, tính reconstruction error
  3. Phân loại từng flow là normal/anomaly so với ngưỡng
  4. Trả về dict tóm tắt và chi tiết từng flow

Quy trình suy diễn:
  DataFrame -> preprocess -> scale (StandardScaler) -> VAE (batch) -> 
  reconstruction_error -> classify -> kết quả

LƯU Ý:
  - Scaler đã được fit trên tập train, không fit lại.
  - Model chạy ở chế độ eval() với torch.no_grad().
  - Xử lý theo batch để tránh tràn bộ nhớ với CSV lớn.
  - Dùng CPU mặc định, tự động dùng CUDA nếu có.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from backend.app.config import (
    DEBUG_INFERENCE,
    DEFAULT_BATCH_SIZE,
    FEATURE_SCHEMA_PATH,
    IMPUTATION_MEDIANS_PATH,
    MODEL_CHECKPOINT_PATH,
    MODEL_CONFIG_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
)
from backend.app.core.preprocessing import (
    MissingArtifactError,
    PreprocessingError,
    load_feature_schema,
    load_imputation_medians,
    preprocess_input_dataframe,
)
from backend.app.core.thresholding import (
    MissingThresholdError,
    ThresholdingError,
    classify_errors,
    load_threshold,
)
from backend.app.models.vae import VAE

logger = logging.getLogger(__name__)


# ── Ngoại lệ tùy chỉnh ───────────────────────────────────────────────────────

class InferenceError(Exception):
    """Lỗi xảy ra trong quá trình suy diễn."""


class ArtifactLoadError(InferenceError):
    """Không thể tải một hoặc nhiều artifact cần thiết."""


class ModelDimensionError(InferenceError):
    """Input dimension không khớp với mô hình đã lưu."""


# ── Dịch vụ suy diễn chính ───────────────────────────────────────────────────

class VAEInferenceService:
    """
    Dịch vụ suy diễn VAE cho phát hiện anomaly trên network flow.

    Tải tất cả artifact một lần khi khởi tạo và tái sử dụng cho nhiều
    lần gọi predict_dataframe(). Thread-safe ở chế độ read-only.

    Attributes:
        model: Mô hình VAE đã tải và ở chế độ eval.
        scaler: StandardScaler đã fit trên tập train.
        threshold: Ngưỡng phân loại anomaly (float).
        feature_columns: Danh sách tên cột theo thứ tự schema.
        imputation_medians: Dict median để impute NaN.
        device: PyTorch device (cpu hoặc cuda).
        input_dim: Chiều đầu vào của mô hình (phải là 66).
    """

    def __init__(
        self,
        model_checkpoint_path: Path = MODEL_CHECKPOINT_PATH,
        model_config_path: Path = MODEL_CONFIG_PATH,
        scaler_path: Path = SCALER_PATH,
        threshold_path: Path = THRESHOLD_PATH,
        feature_schema_path: Path = FEATURE_SCHEMA_PATH,
        imputation_medians_path: Path = IMPUTATION_MEDIANS_PATH,
    ) -> None:
        """
        Khởi tạo dịch vụ và tải tất cả artifact.

        Args:
            model_checkpoint_path: Đường dẫn file checkpoint .pth.
            model_config_path: Đường dẫn file cấu hình model JSON.
            scaler_path: Đường dẫn file scaler.joblib.
            threshold_path: Đường dẫn file threshold.json.
            feature_schema_path: Đường dẫn file feature_columns.json.
            imputation_medians_path: Đường dẫn file imputation_medians.json.

        Raises:
            ArtifactLoadError: Nếu bất kỳ artifact nào không thể tải.
            ModelDimensionError: Nếu input_dim trong config không khớp với schema.
        """
        # ── Xác định device ──────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Sử dụng device: %s", self.device)

        # ── Tải feature schema và medians ────────────────────────────────────
        try:
            self.feature_columns = load_feature_schema(feature_schema_path)
            self.imputation_medians = load_imputation_medians(imputation_medians_path)
        except MissingArtifactError as exc:
            raise ArtifactLoadError(f"Không thể tải feature artifacts: {exc}") from exc

        # ── Tải ngưỡng phân loại ─────────────────────────────────────────────
        try:
            self.threshold = load_threshold(threshold_path)
        except (MissingThresholdError, ThresholdingError) as exc:
            raise ArtifactLoadError(f"Không thể tải threshold: {exc}") from exc

        # ── Tải scaler ───────────────────────────────────────────────────────
        self.scaler = self._load_scaler(scaler_path)

        # ── Tải cấu hình và model VAE ─────────────────────────────────────────
        model_config = self._load_model_config(model_config_path)
        self.input_dim: int = model_config["input_dim"]

        # Kiểm tra chiều đầu vào khớp với feature schema
        n_features = len(self.feature_columns)
        if self.input_dim != n_features:
            raise ModelDimensionError(
                f"input_dim trong model config ({self.input_dim}) không khớp "
                f"với số cột trong feature schema ({n_features}). "
                "Đảm bảo dùng cùng một phiên bản artifact."
            )

        self.model = self._load_vae_model(
            model_checkpoint_path=model_checkpoint_path,
            input_dim=model_config["input_dim"],
            latent_dim=model_config.get("latent_dim", 16),
            hidden_dims=model_config.get("hidden_dims", [128, 64]),
        )

        logger.info(
            "VAEInferenceService sẵn sàng. "
            "input_dim=%d, threshold=%.6f, device=%s",
            self.input_dim,
            self.threshold,
            self.device,
        )

    # ── Các hàm tải artifact nội bộ ──────────────────────────────────────────

    def _load_scaler(self, scaler_path: Path):
        """
        Tải StandardScaler đã fit từ file joblib.

        Raises:
            ArtifactLoadError: Nếu file không tồn tại hoặc tải thất bại.
        """
        if not scaler_path.exists():
            raise ArtifactLoadError(
                f"Scaler artifact không tồn tại: {scaler_path}. "
                "Hãy chạy scripts/train.py để tạo artifact."
            )
        try:
            scaler = joblib.load(scaler_path)
            logger.info("Đã tải scaler từ: %s", scaler_path)
            return scaler
        except Exception as exc:
            raise ArtifactLoadError(
                f"Không thể tải scaler từ {scaler_path}: {exc}"
            ) from exc

    def _load_model_config(self, config_path: Path) -> dict[str, Any]:
        """
        Tải cấu hình kiến trúc mô hình từ file JSON.

        Raises:
            ArtifactLoadError: Nếu file không tồn tại hoặc thiếu key bắt buộc.
        """
        if not config_path.exists():
            raise ArtifactLoadError(
                f"Model config không tồn tại: {config_path}. "
                "Hãy chạy scripts/export_artifacts.py để tạo artifact."
            )

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        required_keys = ["input_dim", "latent_dim", "hidden_dims"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ArtifactLoadError(
                f"Thiếu các key bắt buộc trong model config: {missing}"
            )

        logger.info(
            "Đã tải model config: input_dim=%d, latent_dim=%d, hidden_dims=%s",
            config["input_dim"],
            config["latent_dim"],
            config["hidden_dims"],
        )
        return config

    def _load_vae_model(
        self,
        model_checkpoint_path: Path,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
    ) -> VAE:
        """
        Khởi tạo kiến trúc VAE và tải trọng số từ checkpoint.

        Model được đặt ở chế độ eval() ngay sau khi tải.

        Raises:
            ArtifactLoadError: Nếu file checkpoint không tồn tại hoặc tải thất bại.
        """
        if not model_checkpoint_path.exists():
            raise ArtifactLoadError(
                f"Model checkpoint không tồn tại: {model_checkpoint_path}. "
                "Hãy chạy scripts/train.py để tạo artifact."
            )

        # Khởi tạo kiến trúc VAE với cùng tham số đã dùng khi train
        model = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )

        try:
            # Tải state dict, map về đúng device
            state_dict = torch.load(
                model_checkpoint_path,
                map_location=self.device,
                weights_only=True,  # Bảo mật: chỉ tải tensor, không chạy code tùy ý
            )
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise ArtifactLoadError(
                f"Không thể tải model weights từ {model_checkpoint_path}: {exc}"
            ) from exc

        # Đặt model ở chế độ eval — tắt dropout, BatchNorm dùng running stats,
        # và reparameterize() sẽ trả về mu thay vì lấy mẫu ngẫu nhiên
        model.to(self.device)
        model.eval()

        logger.info("Đã tải VAE checkpoint: %s", model_checkpoint_path)
        return model

    # ── Hàm suy diễn chính ───────────────────────────────────────────────────

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, Any]:
        """
        Chạy pipeline suy diễn đầy đủ trên DataFrame đầu vào.

        Quy trình:
          1. Tiền xử lý DataFrame (xử lý NaN, chuẩn hóa cột, imputation)
          2. Scale dữ liệu bằng scaler đã lưu
          3. Chạy VAE theo từng batch với torch.no_grad()
          4. Tính reconstruction error (MSE per-sample)
          5. Phân loại từng flow là normal/anomaly
          6. Tổng hợp và trả về kết quả

        Args:
            df: DataFrame thô từ CSV đầu vào (chưa qua tiền xử lý).
            batch_size: Số hàng xử lý mỗi lần — điều chỉnh theo RAM/VRAM.

        Returns:
            Dict chứa:
              - total_flows (int): Tổng số flow đã xử lý.
              - anomaly_count (int): Số flow bị phân loại là anomaly.
              - normal_count (int): Số flow bình thường.
              - anomaly_rate (float): Tỷ lệ anomaly (0.0 đến 1.0).
              - threshold (float): Ngưỡng đã dùng để phân loại.
              - results (list[dict]): Chi tiết từng flow.

            Mỗi item trong results:
              - row_index (int): Chỉ số hàng gốc trong DataFrame.
              - reconstruction_error (float): Giá trị MSE reconstruction error.
              - prediction (int): Nhãn số (0=normal, 1=anomaly).
              - prediction_label (str): Nhãn chuỗi ("normal" hoặc "anomaly").

        Raises:
            PreprocessingError: Nếu tiền xử lý thất bại.
            InferenceError: Nếu scale hoặc chạy model thất bại.
        """
        # ── [DEBUG] Chạy walkthrough chi tiết 1 mẫu đầu tiên nếu debug bật ────
        # Hàm debug_single_sample_flow() chạy độc lập, không ảnh hưởng đến
        # kết quả batch inference bên dưới. Chỉ active khi DEBUG_INFERENCE=true.
        if DEBUG_INFERENCE:
            from backend.app.core.debug_inference import debug_single_sample_flow

            # Thiết lập logging level DEBUG để các dòng logger.debug() hiện ra
            logging.getLogger("backend.app.core.debug_inference").setLevel(
                logging.DEBUG
            )
            debug_single_sample_flow(service=self, df_raw=df)

        # ── Bước 1: Tiền xử lý ───────────────────────────────────────────────
        try:
            df_clean = preprocess_input_dataframe(
                df=df,
                feature_columns=self.feature_columns,
                imputation_medians=self.imputation_medians,
            )
        except (PreprocessingError, MissingArtifactError) as exc:
            raise PreprocessingError(
                f"Tiền xử lý đầu vào thất bại: {exc}"
            ) from exc

        n_samples = len(df_clean)
        original_indices = df_clean.index.tolist()

        # ── Bước 2: Scale dữ liệu ────────────────────────────────────────────
        # Dùng scaler đã fit từ training — KHÔNG fit lại
        # Truyền DataFrame (không phải numpy array) để sklearn khớp feature names
        try:
            X_scaled = self.scaler.transform(df_clean)
        except Exception as exc:
            raise InferenceError(
                f"Scaler transform thất bại: {exc}. "
                "Kiểm tra xem scaler có khớp với feature schema không."
            ) from exc

        # Kiểm tra chiều sau khi scale
        if X_scaled.shape[1] != self.input_dim:
            raise ModelDimensionError(
                f"Sau khi scale: shape[1]={X_scaled.shape[1]} "
                f"nhưng model input_dim={self.input_dim}."
            )

        # ── Bước 3 & 4: Chạy VAE theo batch, tính reconstruction error ───────
        all_errors = self._run_vae_batched(X_scaled, batch_size=batch_size)

        # ── Bước 5: Phân loại theo ngưỡng ────────────────────────────────────
        predictions = classify_errors(all_errors, self.threshold)

        # ── Bước 6: Tổng hợp kết quả ─────────────────────────────────────────
        return self._build_result_dict(
            original_indices=original_indices,
            errors=all_errors,
            predictions=predictions,
        )

    def _run_vae_batched(
        self,
        X_scaled: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """
        Chạy forward pass VAE theo từng batch để tránh tràn bộ nhớ.

        Dùng torch.no_grad() để tắt tính toán gradient (inference only).
        Ở chế độ eval(), reparameterize() trả về mu (deterministic).

        Args:
            X_scaled: Mảng đã scale, shape (n_samples, input_dim).
            batch_size: Số hàng mỗi batch.

        Returns:
            Mảng reconstruction error, shape (n_samples,).

        Raises:
            InferenceError: Nếu quá trình chạy model thất bại.
        """
        n_samples = len(X_scaled)
        all_errors: list[np.ndarray] = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_np = X_scaled[start_idx:end_idx]

                # Chuyển batch numpy → PyTorch tensor, đúng device
                batch_tensor = torch.tensor(
                    batch_np,
                    dtype=torch.float32,
                    device=self.device,
                )

                try:
                    # Forward pass: trả về (x_hat, mu, logvar)
                    x_hat, _mu, _logvar = self.model(batch_tensor)

                    # Tính MSE per-sample: error_i = mean((x_i - x_hat_i)^2)
                    batch_errors = VAE.reconstruction_error(batch_tensor, x_hat)

                    # Chuyển về numpy CPU để xử lý tiếp
                    all_errors.append(batch_errors.cpu().numpy())

                except Exception as exc:
                    raise InferenceError(
                        f"Model forward pass thất bại tại batch [{start_idx}:{end_idx}]: {exc}"
                    ) from exc

        # Ghép tất cả batch errors thành một mảng duy nhất
        return np.concatenate(all_errors, axis=0)

    def _build_result_dict(
        self,
        original_indices: list[int],
        errors: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, Any]:
        """
        Xây dựng dict kết quả trả về từ arrays errors và predictions.

        Args:
            original_indices: Chỉ số hàng gốc từ DataFrame đầu vào.
            errors: Mảng reconstruction error, shape (n_samples,).
            predictions: Mảng nhãn nhị phân, shape (n_samples,).

        Returns:
            Dict kết quả chuẩn theo API contract.
        """
        n_total = len(predictions)
        n_anomaly = int(predictions.sum())
        n_normal = n_total - n_anomaly
        anomaly_rate = n_anomaly / n_total if n_total > 0 else 0.0

        # Xây dựng danh sách chi tiết từng flow
        results = [
            {
                "row_index": int(original_indices[i]),
                "reconstruction_error": float(errors[i]),
                "prediction": int(predictions[i]),
                "prediction_label": "anomaly" if predictions[i] == 1 else "normal",
            }
            for i in range(n_total)
        ]

        logger.info(
            "Kết quả: %d flows, %d anomaly (%.1f%%), threshold=%.6f",
            n_total,
            n_anomaly,
            anomaly_rate * 100,
            self.threshold,
        )

        return {
            "total_flows": n_total,
            "anomaly_count": n_anomaly,
            "normal_count": n_normal,
            "anomaly_rate": round(anomaly_rate, 6),
            "threshold": self.threshold,
            "results": results,
        }
