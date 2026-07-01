"""
scripts/clean_data.py
---------------------
Tiền xử lý dữ liệu CICIDS2017 thô và xuất các tập train/val/test
đã được làm sạch, impute, scale — sẵn sàng cho việc huấn luyện VAE.

Chiến lược phân chia dữ liệu:
  - Train  : Monday BENIGN flows (80%)
  - Val    : Monday BENIGN flows (20%)
  - Test   : Tất cả các file còn lại (BENIGN + Attack)

Các bước xử lý (theo thứ tự):
  1. Đọc tất cả CSV thô, chuẩn hóa tên cột.
  2. Chuyển đổi giá trị sang kiểu số, thay thế ±Inf bằng NaN.
  3. Xóa các cột hằng số (zero-variance).
  4. Xóa các hàng trùng lặp.
  5. Tạo nhãn nhị phân (BinaryLabel: BENIGN=0, Attack=1).
  6. Phân chia: Monday → train/val, non-Monday → test.
  7. Tính median trên X_train, impute tất cả các tập.
  8. Fit StandardScaler trên X_train, transform tất cả.
  9. Kiểm tra assertion: không còn NaN hay giá trị vô cực.
  10. Lưu tất cả các artifact.

Cách chạy:
  python scripts/clean_data.py
  python scripts/clean_data.py --raw-dir data/raw --output-dir data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Hằng số cấu hình ─────────────────────────────────────────────────────────
# Tên file CSV chứa toàn bộ BENIGN — dùng làm nguồn dữ liệu train/val
MONDAY_FILE = "Monday-WorkingHours.pcap_ISCX.csv"

# Cột nhãn trong dataset CICIDS2017 (sau khi strip khoảng trắng)
LABEL_COL = "Label"

# Giá trị nhãn bình thường
BENIGN_LABEL = "BENIGN"

# Cột hằng số (zero-variance) đã xác định trong EDA — loại bỏ trước khi huấn luyện
CONSTANT_COLS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Cột gần hằng số (≥99% giá trị là 0) — loại bỏ do variance cực thấp
NEAR_CONSTANT_COLS = [
    "Fwd URG Flags",
    "RST Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
]

# Tỷ lệ phân chia train/val từ Monday data
# VAL_RATIO được suy ra từ TRAIN_RATIO để đảm bảo một nguồn sự thật duy nhất
TRAIN_RATIO = 0.80
VAL_RATIO   = 1.0 - TRAIN_RATIO

# Số lượng mẫu trong fixed batch (dùng để kiểm tra tính nhất quán)
FIXED_BATCH_SIZE = 128

# Random seed để tái tạo kết quả
RANDOM_STATE = 42


# ── Hàm đọc và nối dữ liệu ────────────────────────────────────────────────────

def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    """
    Đọc tất cả CSV trong raw_dir, chuẩn hóa tên cột, thêm cột nguồn.
    Trả về DataFrame đã nối.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong: {raw_dir}")

    log.info(f"Tìm thấy {len(csv_files)} file CSV trong {raw_dir}")

    dfs = []
    for f in csv_files:
        log.info(f"  Đọc: {f.name}")
        df_tmp = pd.read_csv(f, low_memory=False)

        # Strip khoảng trắng trong tên cột — quirk của CICIDS2017
        df_tmp.columns = [c.strip() for c in df_tmp.columns]
        df_tmp["_source_file"] = f.name
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Dữ liệu sau khi nối: {df.shape[0]:,} hàng x {df.shape[1]} cột")
    return df


# ── Hàm làm sạch dữ liệu ──────────────────────────────────────────────────────

def standardize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện các bước làm sạch dữ liệu:
      - Chuyển đổi cột đặc trưng sang kiểu số
      - Thay thế ±Inf bằng NaN
      - Xóa cột hằng số và gần hằng số
      - Xóa hàng trùng lặp
    """
    assert LABEL_COL in df.columns, f"Không tìm thấy cột nhãn '{LABEL_COL}'"

    # Xác định các cột đặc trưng (loại trừ nhãn và metadata)
    meta_cols    = [LABEL_COL, "_source_file"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Chuyển đổi sang kiểu số — các giá trị không hợp lệ thành NaN
    log.info("Chuyển đổi các cột đặc trưng sang kiểu số...")
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Thay thế ±Inf bằng NaN (Flow Bytes/s và Flow Packets/s bị ảnh hưởng)
    inf_count = np.isinf(df[feature_cols].values).sum()
    log.info(f"Thay thế {inf_count:,} giá trị vô cực bằng NaN...")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Xóa các cột hằng số đã xác định trong EDA
    cols_to_drop = [c for c in CONSTANT_COLS + NEAR_CONSTANT_COLS if c in df.columns]
    log.info(f"Xóa {len(cols_to_drop)} cột hằng số / gần hằng số: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # Xóa hàng trùng lặp (giữ lần xuất hiện đầu tiên)
    n_before = len(df)
    df = df.drop_duplicates(
        subset=[c for c in df.columns if c != "_source_file"],
        keep="first",
    )
    n_dropped = n_before - len(df)
    log.info(f"Xóa {n_dropped:,} hàng trùng lặp ({n_dropped/n_before*100:.2f}%)")

    return df


# ── Hàm tạo nhãn nhị phân ─────────────────────────────────────────────────────

def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm cột BinaryLabel: BENIGN=0, Attack=1.
    """
    df = df.copy()
    df["BinaryLabel"] = (df[LABEL_COL] != BENIGN_LABEL).astype(int)
    benign_count = (df["BinaryLabel"] == 0).sum()
    attack_count = (df["BinaryLabel"] == 1).sum()
    log.info(
        f"Phân phối nhãn nhị phân — BENIGN: {benign_count:,}  Attack: {attack_count:,}"
    )
    return df


# ── Hàm phân chia dữ liệu ─────────────────────────────────────────────────────

def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phân chia dữ liệu theo chiến lược:
      - Monday (BENIGN) → train + val
      - Non-Monday      → test

    Trả về: (df_train, df_val, df_test)
    """
    # Phân tách Monday và non-Monday
    mask_monday    = df["_source_file"] == MONDAY_FILE
    df_monday      = df[mask_monday].copy()
    df_non_monday  = df[~mask_monday].copy()

    log.info(f"Monday (BENIGN training source): {len(df_monday):,} hàng")
    log.info(f"Non-Monday (test set)          : {len(df_non_monday):,} hàng")

    # Kiểm tra: Monday phải toàn bộ là BENIGN
    assert (df_monday[LABEL_COL] == BENIGN_LABEL).all(), \
        "Monday file chứa các nhãn không phải BENIGN — kiểm tra lại dữ liệu thô!"

    # Phân chia Monday thành train / val (theo tỷ lệ định sẵn)
    # Dùng 1 - TRAIN_RATIO trực tiếp để đảm bảo khớp với hằng số
    df_train, df_val = train_test_split(
        df_monday,
        test_size=1.0 - TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    log.info(
        f"Phân chia: train={len(df_train):,}  val={len(df_val):,}  "
        f"test={len(df_non_monday):,}"
    )
    return df_train, df_val, df_non_monday


# ── Hàm impute và scale ───────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Trả về danh sách các cột đặc trưng số (loại trừ cột metadata và nhãn).
    """
    exclude = {LABEL_COL, "_source_file", "BinaryLabel"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def impute_with_train_medians(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Tính median từ X_train và áp dụng cho tất cả các tập.
    Trả về: (X_train_imp, X_val_imp, X_test_imp, median_dict)

    Lưu ý: chỉ fit median trên X_train để tránh data leakage.
    """
    train_medians = X_train.median()
    median_dict   = train_medians.to_dict()

    X_train_imp = X_train.fillna(train_medians)
    X_val_imp   = X_val.fillna(train_medians)
    X_test_imp  = X_test.fillna(train_medians)

    log.info(
        f"Impute bằng median từ train — NaN còn lại: "
        f"train={X_train_imp.isnull().sum().sum()}  "
        f"val={X_val_imp.isnull().sum().sum()}  "
        f"test={X_test_imp.isnull().sum().sum()}"
    )
    return X_train_imp, X_val_imp, X_test_imp, median_dict


def fit_and_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler trên X_train rồi transform tất cả ba tập.
    Trả về: (X_train_sc, X_val_sc, X_test_sc, scaler)
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    log.info(
        f"StandardScaler fit trên {len(X_train):,} mẫu train. "
        f"mean[0]={scaler.mean_[0]:.4f}  std[0]={scaler.scale_[0]:.4f}"
    )
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ── Kiểm tra tính toàn vẹn sau xử lý ─────────────────────────────────────────

def assert_no_nonfinite(name: str, arr: np.ndarray) -> None:
    """Xác nhận mảng không chứa NaN hay giá trị vô cực."""
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    assert n_nan == 0, f"{name}: còn {n_nan:,} giá trị NaN!"
    assert n_inf == 0, f"{name}: còn {n_inf:,} giá trị vô cực!"
    log.info(f"✓ {name}: không có NaN hay vô cực ({arr.shape})")


# ── Hàm lưu artifact ──────────────────────────────────────────────────────────

def save_outputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_unscaled: pd.DataFrame,
    test_labels: pd.Series,
    test_sources: pd.Series,
    feature_cols: list[str],
    scaler: StandardScaler,
    median_dict: dict,
    output_dir: Path,
    artifacts_dir: Path,
) -> None:
    """
    Lưu tất cả các artifact đầu ra:
      - .npy arrays cho train/val/test
      - StandardScaler (.joblib)
      - Median imputation values (.json)
      - Danh sách đặc trưng (.json)
      - Metadata tiền xử lý (.json)
      - Fixed batch CSV (128 mẫu unscaled từ X_test)
    """
    # Tạo thư mục nếu chưa tồn tại
    for d in [
        output_dir / "train",
        output_dir / "validation",
        output_dir / "test",
        output_dir / "processed",
        artifacts_dir / "scaler",
        artifacts_dir / "sample_batch",
        artifacts_dir / "feature_schema",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Lưu numpy arrays ──────────────────────────────────────────────────────
    log.info("Lưu numpy arrays...")
    np.save(output_dir / "train"      / "X_train.npy", X_train)
    np.save(output_dir / "train"      / "y_train.npy", y_train)
    np.save(output_dir / "validation" / "X_val.npy",   X_val)
    np.save(output_dir / "validation" / "y_val.npy",   y_val)
    np.save(output_dir / "test"       / "X_test.npy",  X_test)
    np.save(output_dir / "test"       / "y_test.npy",  y_test)

    test_metadata = pd.DataFrame(
        {
            "label": test_labels.reset_index(drop=True),
            "source_file": test_sources.reset_index(drop=True),
            "binary_label": y_test,
        }
    )
    test_metadata_path = output_dir / "test" / "test_metadata.csv"
    test_metadata.to_csv(test_metadata_path, index=False)
    log.info(f"Test metadata da luu: {test_metadata_path}")

    # ── Lưu scaler ────────────────────────────────────────────────────────────
    scaler_path = artifacts_dir / "scaler" / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler đã lưu: {scaler_path}")

    # ── Lưu median imputation values ──────────────────────────────────────────
    medians_path = artifacts_dir / "scaler" / "imputation_medians.json"
    with open(medians_path, "w", encoding="utf-8") as f:
        json.dump(median_dict, f, indent=2)
    log.info(f"Imputation medians đã lưu: {medians_path}")

    # ── Lưu danh sách đặc trưng (feature schema) ──────────────────────────────
    schema_path_1 = output_dir / "processed" / "feature_names.json"
    schema_path_2 = artifacts_dir / "feature_schema" / "feature_columns.json"
    feature_schema = {"feature_columns": feature_cols, "n_features": len(feature_cols)}
    for path in [schema_path_1, schema_path_2]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=2)
    log.info(f"Feature schema đã lưu: {schema_path_1}  &  {schema_path_2}")

    # ── Lưu metadata tiền xử lý ───────────────────────────────────────────────
    # Lưu đường dẫn dưới dạng tương đối (posix) so với project root
    # để metadata có thể di chuyển được giữa các môi trường
    project_root = output_dir.parent

    def _rel(p: Path) -> str:
        """Chuyển đổi absolute path thành relative posix path từ project root."""
        try:
            return p.relative_to(project_root).as_posix()
        except ValueError:
            return p.as_posix()

    metadata = {
        "created_at"            : pd.Timestamp.now().isoformat(),
        "random_state"          : RANDOM_STATE,
        "monday_file"           : MONDAY_FILE,
        "train_ratio"           : TRAIN_RATIO,
        "val_ratio"             : round(VAL_RATIO, 10),
        "n_features"            : len(feature_cols),
        "n_train"               : int(X_train.shape[0]),
        "n_val"                 : int(X_val.shape[0]),
        "n_test"                : int(X_test.shape[0]),
        "constant_cols_dropped" : CONSTANT_COLS,
        "near_constant_cols_dropped": NEAR_CONSTANT_COLS,
        "inf_handling"          : "replace ±Inf with NaN",
        "nan_imputation"        : "median (computed on X_train only)",
        "scaling"               : "StandardScaler (fit on X_train only)",
        "label_encoding"        : "BENIGN=0, Attack=1",
        "scaler_path"           : _rel(scaler_path),
        "medians_path"          : _rel(medians_path),
        "feature_schema_path"   : _rel(schema_path_2),
    }
    meta_path = output_dir / "processed" / "preprocessing_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Preprocessing metadata đã lưu: {meta_path}")

    # ── Lưu fixed batch CSV (128 mẫu unscaled từ X_test) ─────────────────────
    # Dùng để kiểm tra tính nhất quán giữa training và inference
    batch_path = artifacts_dir / "sample_batch" / "fixed_batch.csv"
    fixed_batch = X_test_unscaled.iloc[:FIXED_BATCH_SIZE].copy()
    fixed_batch.to_csv(batch_path, index=False)
    log.info(f"Fixed batch ({len(fixed_batch)} mẫu unscaled) đã lưu: {batch_path}")


# ── Hàm tổng hợp chính ────────────────────────────────────────────────────────

def run_pipeline(raw_dir: Path, output_dir: Path, artifacts_dir: Path) -> None:
    """
    Chạy toàn bộ pipeline tiền xử lý từ đầu đến cuối.
    """
    log.info("=" * 60)
    log.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ CICIDS2017")
    log.info("=" * 60)

    # ── Bước 1: Đọc dữ liệu thô ───────────────────────────────────────────────
    df = load_raw_csvs(raw_dir)

    # ── Bước 2: Làm sạch dữ liệu ──────────────────────────────────────────────
    df = standardize_and_clean(df)

    # ── Bước 3: Tạo nhãn nhị phân ─────────────────────────────────────────────
    df = create_binary_label(df)

    # ── Bước 4: Phân chia dữ liệu ─────────────────────────────────────────────
    df_train, df_val, df_test = split_dataset(df)

    # ── Bước 5: Xác định cột đặc trưng ────────────────────────────────────────
    feature_cols = get_feature_columns(df_train)
    log.info(f"Số cột đặc trưng cuối cùng: {len(feature_cols)}")

    X_train_raw = df_train[feature_cols].reset_index(drop=True)
    X_val_raw   = df_val[feature_cols].reset_index(drop=True)
    X_test_raw  = df_test[feature_cols].reset_index(drop=True)

    y_train = df_train["BinaryLabel"].to_numpy()
    y_val   = df_val["BinaryLabel"].to_numpy()
    y_test  = df_test["BinaryLabel"].to_numpy()

    # ── Bước 6: Impute NaN bằng median của X_train ────────────────────────────
    X_train_imp, X_val_imp, X_test_imp, median_dict = impute_with_train_medians(
        X_train_raw, X_val_raw, X_test_raw
    )

    # ── Bước 7: Lưu X_test unscaled trước khi scale (cho fixed batch) ─────────
    X_test_unscaled = X_test_imp.copy()

    # ── Bước 8: Scale dữ liệu ─────────────────────────────────────────────────
    X_train_sc, X_val_sc, X_test_sc, scaler = fit_and_scale(
        X_train_imp, X_val_imp, X_test_imp
    )

    # ── Bước 9: Kiểm tra assertion ────────────────────────────────────────────
    log.info("Chạy kiểm tra tính toàn vẹn dữ liệu...")
    assert_no_nonfinite("X_train", X_train_sc)
    assert_no_nonfinite("X_val",   X_val_sc)
    assert_no_nonfinite("X_test",  X_test_sc)

    # Kiểm tra shape nhất quán
    assert X_train_sc.shape[1] == len(feature_cols), "Shape X_train không khớp feature_cols!"
    assert X_val_sc.shape[1]   == len(feature_cols), "Shape X_val không khớp feature_cols!"
    assert X_test_sc.shape[1]  == len(feature_cols), "Shape X_test không khớp feature_cols!"

    # Kiểm tra nhãn train phải toàn bộ là BENIGN (0)
    assert y_train.sum() == 0, "Tập train chứa nhãn Attack — chỉ được dùng BENIGN để train VAE!"
    assert y_val.sum()   == 0, "Tập val chứa nhãn Attack — chỉ được dùng BENIGN để validate VAE!"

    log.info("✓ Tất cả kiểm tra tính toàn vẹn ĐÃ PASSED")

    # ── Bước 10: Lưu tất cả artifact ──────────────────────────────────────────
    save_outputs(
        X_train=X_train_sc,
        y_train=y_train,
        X_val=X_val_sc,
        y_val=y_val,
        X_test=X_test_sc,
        y_test=y_test,
        X_test_unscaled=X_test_unscaled,
        test_labels=df_test[LABEL_COL],
        test_sources=df_test["_source_file"],
        feature_cols=feature_cols,
        scaler=scaler,
        median_dict=median_dict,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
    )

    # ── Tóm tắt kết quả ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("HOÀN THÀNH PIPELINE TIỀN XỬ LÝ")
    log.info(f"  Train  : {X_train_sc.shape}  (tất cả BENIGN, y=0)")
    log.info(f"  Val    : {X_val_sc.shape}    (tất cả BENIGN, y=0)")
    log.info(f"  Test   : {X_test_sc.shape}   (BENIGN + Attack)")
    log.info(f"  Attack trong test: {y_test.sum():,} / {len(y_test):,} ({y_test.mean()*100:.1f}%)")
    log.info(f"  Số đặc trưng     : {len(feature_cols)}")
    log.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu CICIDS2017 cho VAE NIDS"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Thư mục chứa CSV thô (mặc định: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Thư mục gốc để lưu train/val/test (mặc định: data)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Thư mục lưu scaler, schema, sample batch (mặc định: artifacts)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve về absolute paths để tránh nhầm lẫn working directory
    raw_dir       = args.raw_dir.resolve()
    output_dir    = args.output_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()

    run_pipeline(raw_dir, output_dir, artifacts_dir)
