from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _load_mask(mask_path: str | float | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or (isinstance(mask_path, float) and np.isnan(mask_path)):
        return None

    mask_path = str(mask_path)
    if not mask_path:
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def pixel_auroc_from_scores(df: pd.DataFrame) -> tuple[float, int]:
    y_true_parts: list[np.ndarray] = []
    y_score_parts: list[np.ndarray] = []
    anomaly_masks_used = 0

    for _, row in df.iterrows():
        heatmap_path = Path(row["heatmap_npy"])
        if not heatmap_path.exists():
            continue
        heatmap = np.load(heatmap_path).astype(np.float32)

        label = int(row["label"])
        if label == 0:
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
        else:
            mask = _load_mask(row.get("mask_path"), heatmap.shape)
            if mask is None:
                continue
            anomaly_masks_used += 1

        y_true_parts.append(mask.reshape(-1))
        y_score_parts.append(heatmap.reshape(-1))

    if anomaly_masks_used == 0 or not y_true_parts:
        return float("nan"), anomaly_masks_used

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    if np.unique(y_true).size < 2:
        return float("nan"), anomaly_masks_used
    return float(roc_auc_score(y_true, y_score)), anomaly_masks_used


def evaluate_scores_csv(scores_csv: str | Path) -> dict[str, float | int]:
    df = pd.read_csv(scores_csv)
    img_auc = image_auroc(df["label"].to_numpy(), df["image_score"].to_numpy())
    pix_auc, mask_count = pixel_auroc_from_scores(df)

    avg_time = float(df["inference_time_sec"].mean()) if "inference_time_sec" in df else float("nan")
    memory_size = int(df["memory_bank_size"].iloc[0]) if "memory_bank_size" in df and len(df) else 0

    return {
        "image_auroc": img_auc,
        "pixel_auroc": pix_auc,
        "pixel_anomaly_masks_used": int(mask_count),
        "average_inference_time_sec": avg_time,
        "memory_bank_size": memory_size,
        "num_test_images": int(len(df)),
    }

