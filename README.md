# NIDS VAE – Hệ Thống Phát Hiện Xâm Nhập Mạng

Hệ thống phát hiện xâm nhập mạng (Network Intrusion Detection System) sử dụng Variational Autoencoder (VAE) huấn luyện trên bộ dữ liệu CICIDS2017. Mỗi flow mạng được phân loại là **bình thường** hoặc **bất thường** dựa trên reconstruction error so với một ngưỡng đã học.

```
CSV/PCAP → Trích xuất đặc trưng → Chuẩn hóa → VAE → Reconstruction Error → Threshold → Normal / Anomaly → Dashboard
```

---

## Bắt Đầu Nhanh (Quick Start)

> **Yêu cầu:** Python 3.10+, Git, Windows PowerShell

```powershell
# 1. Clone dự án
git clone <repo-url>
cd nids-vae-project

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Cài dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Chạy backend (artifacts đã có sẵn trong repo)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# 5. Kiểm tra
# Mở trình duyệt: http://127.0.0.1:8000/health
# Swagger UI:     http://127.0.0.1:8000/docs
```

**Kiểm tra predict với file mẫu:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Form @{file=Get-Item "artifacts\sample_batch\fixed_batch.csv"}
```

Kết quả mong đợi: `total_flows: 128`, `anomaly_count: 1`, `normal_count: 127`.

---

## Hướng Dẫn Chi Tiết

Xem [docs/setup-guide.md](docs/setup-guide.md) để có hướng dẫn đầy đủ bao gồm:

- Cài đặt phần mềm cần thiết
- Hai chế độ khởi chạy (dùng artifacts có sẵn / xây dựng lại từ dữ liệu thô)
- Chuẩn bị dataset CICIDS2017
- Hướng dẫn chạy frontend
- Checklist xác nhận
- Bảng lệnh tóm tắt

Xem [docs/troubleshooting.md](docs/troubleshooting.md) để biết cách xử lý các lỗi phổ biến.

---

## Cấu Trúc Dự Án

```
nids-vae-project/
├── artifacts/          # Model, scaler, threshold artifacts
├── backend/            # FastAPI service (Python)
├── data/               # Dữ liệu thô và đã xử lý
├── docs/               # Tài liệu dự án
├── frontend/           # React dashboard (đang phát triển)
├── notebooks/          # Jupyter notebooks phân tích
├── scripts/            # Scripts xử lý dữ liệu, huấn luyện, đánh giá
└── requirements.txt    # Dependencies Python
```

---

## Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Mô hình ML | PyTorch (VAE) |
| Xử lý dữ liệu | pandas, NumPy, scikit-learn |
| Frontend | React + Vite |
| Tests | pytest |

---

## Kết Quả Mô Hình

Huấn luyện trên CICIDS2017 (train: 402,229 flows BENIGN):

| Chỉ số | Giá trị |
|---|---|
| Best epoch | 58 |
| Final epoch | 68 |
| Best validation loss | 0.740856 |
| Threshold (P99 val) | 6.031263 |
| F1 Score | 0.693 |
| ROC AUC | 0.784 |
| Accuracy | 0.892 |
| Precision | 0.865 |
| Recall | 0.578 |

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Kiểm tra trạng thái service |
| POST | `/upload` | Upload CSV, validate format |
| POST | `/predict` | Chạy phát hiện anomaly, trả kết quả |
| GET | `/results` | Lấy kết quả predict gần nhất |

Tài liệu API đầy đủ: [docs/api-spec.md](docs/api-spec.md)

---

## Tài Liệu

- [Hướng dẫn cài đặt](docs/setup-guide.md)
- [Xử lý lỗi](docs/troubleshooting.md)
- [API Spec](docs/api-spec.md)
- [Xử lý dữ liệu](docs/data-processing.md)
- [Thiết kế mô hình](docs/model-design.md)
- [Tổng quan dự án](docs/project-overview.md)
- [Notebook walkthrough NIDS-VAE tiếng Việt](notebooks/nids_vae_full_walkthrough_vi.ipynb)

---

## Chạy Bằng Docker

Chế độ Docker dùng cho production/local demo:

- Backend FastAPI chạy bằng Uvicorn trên port `8000`.
- Frontend React/Vite được build static và serve bằng Nginx trên port `5173`.
- Docker image không copy `data/`, `notebooks/`, `reports/`, `venv/`, `frontend/node_modules/`, `frontend/dist/` hoặc `__pycache__/`.

### Yêu Cầu Artifact Runtime

Backend container cần các file sau tồn tại trong repo trước khi build:

```text
artifacts/models/vae_best.pth
artifacts/models/model_config.json
artifacts/scaler/scaler.joblib
artifacts/scaler/imputation_medians.json
artifacts/threshold/threshold.json
artifacts/feature_schema/feature_columns.json
artifacts/sample_batch/fixed_batch.csv
```

Kiểm tra nhanh các artifact runtime:

```powershell
python scripts/check_runtime_artifacts.py
```

`vae_best.pth` và `scaler.joblib` là artifact runtime bắt buộc. Hai file này hiện có kích thước nhỏ, nên `.gitignore` đã được cấu hình để cho phép track đúng hai artifact này thay vì bỏ qua toàn bộ binary artifact. Nếu sau này model lớn hơn, nên chuyển `*.pth`/`*.joblib` sang Git LFS hoặc phát hành qua GitHub Release.

### Khởi Động

```bash
docker compose up --build
```

Frontend sẽ gọi backend qua `VITE_API_BASE_URL`. Mặc định trong Docker Compose là:

```text
http://localhost:8000
```

Muốn đổi URL backend khi build frontend:

```bash
VITE_API_BASE_URL=http://localhost:8000 docker compose up --build
```

### Kiểm Tra

```bash
docker compose config
docker compose up --build
```

Sau khi container khởi động:

- Backend health: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- Frontend: http://localhost:5173

Kiểm tra nhanh bằng sample CSV:

```bash
curl -F "file=@artifacts/sample_batch/fixed_batch.csv" http://localhost:8000/predict
```
