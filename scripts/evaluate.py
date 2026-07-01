"""
scripts/evaluate.py
-------------------
Evaluate the VAE NIDS model on CICIDS2017 processed validation/test arrays.

Backward-compatible default:
    python scripts/evaluate.py --percentile 99

New capabilities:
  - score modes: recon, kl, recon_kl
  - threshold sweep at fixed percentiles
  - final threshold selection by fixed percentile, max F2 under FPR, or max Recall under FPR
  - F2, FNR, PR-AUC metrics
  - optional false-negative and per-attack reports when data/test/test_metadata.csv exists
  - evaluation manifest with run id and hashes
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.vae import VAE  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_VAL_DIR = PROJECT_ROOT / "data" / "validation"
DATA_TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
THRESHOLD_DIR = PROJECT_ROOT / "artifacts" / "threshold"

MODEL_CHECKPOINT = MODEL_DIR / "vae_best.pth"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
TEST_METADATA_PATH = DATA_TEST_DIR / "test_metadata.csv"

THRESHOLD_JSON = THRESHOLD_DIR / "threshold.json"
VAL_ERRORS_PATH = THRESHOLD_DIR / "validation_errors.npy"
TEST_ERRORS_PATH = THRESHOLD_DIR / "test_errors.npy"
EVAL_METRICS_PATH = THRESHOLD_DIR / "evaluation_metrics.json"
CONFUSION_MAT_PATH = THRESHOLD_DIR / "confusion_matrix.json"
THRESHOLD_SWEEP_PATH = THRESHOLD_DIR / "threshold_sweep.csv"
FALSE_NEGATIVE_PATH = THRESHOLD_DIR / "false_negative_analysis.csv"
PER_ATTACK_METRICS_PATH = THRESHOLD_DIR / "per_attack_metrics.csv"
MANIFEST_PATH = THRESHOLD_DIR / "artifact_manifest.json"

SWEEP_PERCENTILES = [90, 92, 93, 94, 95, 96, 97, 97.5, 98, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE NIDS model and select anomaly threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--score-mode",
        choices=["recon", "kl", "recon_kl"],
        default="recon",
        help="Anomaly score source. recon keeps the historical behavior.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Weight for normalized KL when --score-mode recon_kl.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["fixed_percentile", "max_f2_under_fpr", "max_recall_under_fpr"],
        default="fixed_percentile",
        help="How to choose the final threshold after computing the sweep.",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.07,
        help="FPR ceiling for max_f2_under_fpr and max_recall_under_fpr.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def load_model(device: torch.device) -> tuple[VAE, int]:
    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_CHECKPOINT}")
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing model config: {MODEL_CONFIG_PATH}")

    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    input_dim = int(config["input_dim"])
    model = VAE(
        input_dim=input_dim,
        latent_dim=int(config["latent_dim"]),
        hidden_dims=config["hidden_dims"],
    )
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model: input_dim=%d, latent_dim=%s, hidden_dims=%s",
        input_dim,
        config["latent_dim"],
        config["hidden_dims"],
    )
    return model, input_dim


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_val = np.load(DATA_VAL_DIR / "X_val.npy")
    y_val = np.load(DATA_VAL_DIR / "y_val.npy")
    X_test = np.load(DATA_TEST_DIR / "X_test.npy")
    y_test = np.load(DATA_TEST_DIR / "y_test.npy")
    logger.info(
        "Loaded data: X_val=%s, y_val=%s, X_test=%s, y_test=%s",
        X_val.shape,
        y_val.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_val, y_val, X_test, y_test


def validate_data(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> None:
    assert X_val.shape[1] == input_dim
    assert X_test.shape[1] == input_dim
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()
    assert not np.isinf(X_val).any()
    assert not np.isinf(X_test).any()
    assert y_val.sum() == 0, "Validation set must contain only BENIGN labels."
    labels = np.unique(y_test)
    assert 0 in labels and 1 in labels, f"y_test must contain both classes, got {labels}"
    logger.info(
        "Test labels: %d normal, %d anomaly",
        int((y_test == 0).sum()),
        int((y_test == 1).sum()),
    )


@torch.no_grad()
def compute_score_components(
    model: VAE,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor_X = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_X), batch_size=batch_size, shuffle=False, num_workers=0)

    recon_values: list[np.ndarray] = []
    kl_values: list[np.ndarray] = []

    for (batch,) in loader:
        batch = batch.to(device)
        x_hat, mu, logvar = model(batch)
        recon = VAE.reconstruction_error(batch, x_hat)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        recon_values.append(recon.cpu().numpy())
        kl_values.append(kl.cpu().numpy())

    return np.concatenate(recon_values), np.concatenate(kl_values)


def _normalization_stats(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"mean": mean, "std": std if std > 0 else 1.0}


def _normalize(values: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return (values - stats["mean"]) / stats["std"]


def build_scores(
    val_recon: np.ndarray,
    val_kl: np.ndarray,
    test_recon: np.ndarray,
    test_kl: np.ndarray,
    score_mode: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    recon_stats = _normalization_stats(val_recon)
    kl_stats = _normalization_stats(val_kl)

    if score_mode == "recon":
        val_scores = val_recon
        test_scores = test_recon
    elif score_mode == "kl":
        val_scores = val_kl
        test_scores = test_kl
    elif score_mode == "recon_kl":
        val_scores = _normalize(val_recon, recon_stats) + alpha * _normalize(val_kl, kl_stats)
        test_scores = _normalize(test_recon, recon_stats) + alpha * _normalize(test_kl, kl_stats)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    score_stats = {
        "score_mode": score_mode,
        "alpha": alpha,
        "recon_normalization": recon_stats,
        "kl_normalization": kl_stats,
        "validation_score_stats": score_distribution_stats(val_scores),
    }
    return val_scores.astype(np.float32), test_scores.astype(np.float32), score_stats


def score_distribution_stats(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p97_5": float(np.percentile(scores, 97.5)),
        "p99": float(np.percentile(scores, 99)),
    }


def predict_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(np.int32)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int]]:
    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    anomaly_rate = float(y_pred.mean())

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_positive_rate": tpr,
        "anomaly_rate": anomaly_rate,
        "n_samples": int(len(y_true)),
        "n_predicted_anomaly": int(y_pred.sum()),
        "n_predicted_normal": int((y_pred == 0).sum()),
        "n_true_anomaly": int(y_true.sum()),
        "n_true_normal": int((y_true == 0).sum()),
    }
    return metrics, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_threshold_sweep(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for percentile in SWEEP_PERCENTILES:
        threshold = float(np.percentile(val_scores, percentile))
        y_pred = predict_labels(test_scores, threshold)
        metrics, cm = compute_metrics(y_test, y_pred, test_scores)
        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "f2_score": metrics["f2_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "fpr": metrics["false_positive_rate"],
                "fnr": metrics["false_negative_rate"],
                "accuracy": metrics["accuracy"],
                "tp": cm["tp"],
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    val_scores: np.ndarray,
    sweep_df: pd.DataFrame,
    percentile: float,
    strategy: str,
    max_fpr: float,
) -> tuple[float, float, str]:
    if strategy == "fixed_percentile":
        return float(np.percentile(val_scores, percentile)), float(percentile), strategy

    eligible = sweep_df[sweep_df["fpr"] <= max_fpr].copy()
    if eligible.empty:
        logger.warning(
            "No sweep percentile satisfies max_fpr=%.4f; choosing the lowest-FPR row.",
            max_fpr,
        )
        eligible = sweep_df.sort_values(["fpr", "percentile"], ascending=[True, False]).head(1)

    if strategy == "max_f2_under_fpr":
        selected = eligible.sort_values(
            ["f2_score", "recall", "fpr"],
            ascending=[False, False, True],
        ).iloc[0]
    elif strategy == "max_recall_under_fpr":
        selected = eligible.sort_values(
            ["recall", "f2_score", "fpr"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    return float(selected["threshold"]), float(selected["percentile"]), strategy


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_threshold_artifact(
    threshold: float,
    percentile: float,
    val_stats: dict[str, float],
    run_id: str,
    score_mode: str,
    alpha: float,
    strategy: str,
    max_fpr: float,
    created_at: str,
) -> None:
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "threshold": threshold,
        "percentile": percentile,
        "selection_method": strategy,
        "max_fpr": max_fpr,
        "score_mode": score_mode,
        "alpha": alpha,
        "validation_error_stats": val_stats,
        "model_checkpoint": str(MODEL_CHECKPOINT),
        "run_id": run_id,
        "created_at": created_at,
    }
    with THRESHOLD_JSON.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    logger.info("Saved threshold artifact: %s", THRESHOLD_JSON)


def save_evaluation_artifacts(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    metrics: dict[str, Any],
    cm_dict: dict[str, int],
    threshold: float,
    percentile: float,
    score_mode: str,
    alpha: float,
    strategy: str,
    max_fpr: float,
    score_stats: dict[str, Any],
    created_at: str,
) -> None:
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    np.save(VAL_ERRORS_PATH, val_scores)
    np.save(TEST_ERRORS_PATH, test_scores)
    logger.info("Saved validation_errors.npy and test_errors.npy")

    metrics_artifact: dict[str, Any] = {
        "metrics": metrics,
        "threshold": threshold,
        "percentile": percentile,
        "score_mode": score_mode,
        "alpha": alpha,
        "selection_method": strategy,
        "max_fpr": max_fpr,
        "score_stats": score_stats,
        "created_at": created_at,
    }
    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_artifact, f, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation_metrics.json")

    cm_artifact: dict[str, Any] = {
        "confusion_matrix": cm_dict,
        "labels": {"0": "normal", "1": "anomaly"},
        "threshold": threshold,
        "score_mode": score_mode,
        "created_at": created_at,
    }
    with CONFUSION_MAT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cm_artifact, f, indent=2, ensure_ascii=False)
    logger.info("Saved confusion_matrix.json")


def save_manifest(
    run_id: str,
    threshold: float,
    percentile: float,
    score_mode: str,
    alpha: float,
    val_scores: np.ndarray,
    created_at: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "model_checkpoint_path": str(MODEL_CHECKPOINT),
        "model_config_hash": _sha256_file(MODEL_CONFIG_PATH),
        "validation_errors_hash": _sha256_array(val_scores),
        "threshold": threshold,
        "percentile": percentile,
        "score_mode": score_mode,
        "alpha": alpha,
        "created_at": created_at,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved artifact manifest: %s", MANIFEST_PATH)


def save_label_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    recon: np.ndarray,
    kl: np.ndarray,
) -> None:
    if not TEST_METADATA_PATH.exists():
        logger.info("No test_metadata.csv found; skipping per-attack reports.")
        return

    metadata = pd.read_csv(TEST_METADATA_PATH)
    if len(metadata) != len(y_true):
        logger.warning(
            "test_metadata.csv length (%d) does not match y_test (%d); skipping label reports.",
            len(metadata),
            len(y_true),
        )
        return

    df = metadata.copy()
    df["binary_label"] = y_true.astype(int)
    df["prediction"] = y_pred.astype(int)
    df["score"] = scores
    df["reconstruction_score"] = recon
    df["kl_score"] = kl
    df["is_false_negative"] = (df["binary_label"] == 1) & (df["prediction"] == 0)

    false_negatives = df[df["is_false_negative"]].copy()
    false_negatives.insert(0, "test_index", false_negatives.index)
    false_negatives.to_csv(FALSE_NEGATIVE_PATH, index=False)
    logger.info("Saved false-negative analysis: %s", FALSE_NEGATIVE_PATH)

    rows = []
    for label, group in df.groupby("label", dropna=False):
        y_group = group["binary_label"].to_numpy()
        pred_group = group["prediction"].to_numpy()
        positives = int((y_group == 1).sum())
        negatives = int((y_group == 0).sum())
        tp = int(((y_group == 1) & (pred_group == 1)).sum())
        fn = int(((y_group == 1) & (pred_group == 0)).sum())
        fp = int(((y_group == 0) & (pred_group == 1)).sum())
        tn = int(((y_group == 0) & (pred_group == 0)).sum())
        rows.append(
            {
                "label": label,
                "n_samples": int(len(group)),
                "n_attack": positives,
                "n_benign": negatives,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": float(tp / positives) if positives else np.nan,
                "fpr": float(fp / negatives) if negatives else np.nan,
                "mean_score": float(group["score"].mean()),
                "median_score": float(group["score"].median()),
            }
        )

    pd.DataFrame(rows).sort_values(["n_attack", "n_samples"], ascending=False).to_csv(
        PER_ATTACK_METRICS_PATH,
        index=False,
    )
    logger.info("Saved per-attack metrics: %s", PER_ATTACK_METRICS_PATH)


def print_results(
    metrics: dict[str, Any],
    cm_dict: dict[str, int],
    threshold: float,
    percentile: float,
    score_mode: str,
    strategy: str,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  VAE NIDS EVALUATION")
    print(sep)
    print(f"  Score mode                 : {score_mode}")
    print(f"  Threshold strategy         : {strategy}")
    print(f"  Selected percentile        : {percentile:.1f}")
    print(f"  Threshold                  : {threshold:.6f}")
    print(sep)
    print(f"  {'Accuracy':<30}: {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<30}: {metrics['precision']:.4f}")
    print(f"  {'Recall':<30}: {metrics['recall']:.4f}")
    print(f"  {'F1 Score':<30}: {metrics['f1_score']:.4f}")
    print(f"  {'F2 Score':<30}: {metrics['f2_score']:.4f}")
    print(f"  {'ROC AUC':<30}: {metrics['roc_auc']:.4f}")
    print(f"  {'PR AUC':<30}: {metrics['pr_auc']:.4f}")
    print(f"  {'False Positive Rate':<30}: {metrics['false_positive_rate']:.4f}")
    print(f"  {'False Negative Rate':<30}: {metrics['false_negative_rate']:.4f}")
    print(sep)
    print(f"  {'True Positive  (TP)':<30}: {cm_dict['tp']:,}")
    print(f"  {'True Negative  (TN)':<30}: {cm_dict['tn']:,}")
    print(f"  {'False Positive (FP)':<30}: {cm_dict['fp']:,}")
    print(f"  {'False Negative (FN)':<30}: {cm_dict['fn']:,}")
    print(sep)
    print(f"  Total test samples         : {metrics['n_samples']:,}")
    print(f"  True anomalies             : {metrics['n_true_anomaly']:,}")
    print(f"  Predicted anomalies        : {metrics['n_predicted_anomaly']:,}")
    print(f"{sep}\n")


def main() -> None:
    args = parse_args()
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(
        "Device=%s, score_mode=%s, alpha=%.4f, strategy=%s, percentile=%.1f, max_fpr=%.4f",
        device,
        args.score_mode,
        args.alpha,
        args.threshold_strategy,
        args.percentile,
        args.max_fpr,
    )

    start_time = time.time()
    model, input_dim = load_model(device)
    X_val, y_val, X_test, y_test = load_data()
    validate_data(X_val, y_val, X_test, y_test, input_dim)

    logger.info("Computing validation score components...")
    val_recon, val_kl = compute_score_components(model, X_val, args.batch_size, device)
    logger.info("Computing test score components...")
    test_recon, test_kl = compute_score_components(model, X_test, args.batch_size, device)

    val_scores, test_scores, score_stats = build_scores(
        val_recon=val_recon,
        val_kl=val_kl,
        test_recon=test_recon,
        test_kl=test_kl,
        score_mode=args.score_mode,
        alpha=args.alpha,
    )

    sweep_df = build_threshold_sweep(val_scores, test_scores, y_test)
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    logger.info("Saved threshold sweep: %s", THRESHOLD_SWEEP_PATH)

    threshold, selected_percentile, selection_method = select_threshold(
        val_scores=val_scores,
        sweep_df=sweep_df,
        percentile=args.percentile,
        strategy=args.threshold_strategy,
        max_fpr=args.max_fpr,
    )

    y_pred = predict_labels(test_scores, threshold)
    metrics, cm_dict = compute_metrics(y_test, y_pred, test_scores)
    val_stats = score_distribution_stats(val_scores)

    logger.info("Saving artifacts...")
    save_threshold_artifact(
        threshold=threshold,
        percentile=selected_percentile,
        val_stats=val_stats,
        run_id=run_id,
        score_mode=args.score_mode,
        alpha=args.alpha,
        strategy=selection_method,
        max_fpr=args.max_fpr,
        created_at=created_at,
    )
    save_evaluation_artifacts(
        val_scores=val_scores,
        test_scores=test_scores,
        metrics=metrics,
        cm_dict=cm_dict,
        threshold=threshold,
        percentile=selected_percentile,
        score_mode=args.score_mode,
        alpha=args.alpha,
        strategy=selection_method,
        max_fpr=args.max_fpr,
        score_stats=score_stats,
        created_at=created_at,
    )
    save_manifest(
        run_id=run_id,
        threshold=threshold,
        percentile=selected_percentile,
        score_mode=args.score_mode,
        alpha=args.alpha,
        val_scores=val_scores,
        created_at=created_at,
    )
    save_label_reports(y_test, y_pred, test_scores, test_recon, test_kl)

    print_results(
        metrics=metrics,
        cm_dict=cm_dict,
        threshold=threshold,
        percentile=selected_percentile,
        score_mode=args.score_mode,
        strategy=selection_method,
    )

    logger.info("Evaluation completed in %.1f seconds", time.time() - start_time)
    logger.info("Artifacts saved under: %s", THRESHOLD_DIR)


if __name__ == "__main__":
    main()
