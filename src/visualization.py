from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def normalize_heatmap(heatmap: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value - min_value < eps:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_value) / (max_value - min_value)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    normalized = normalize_heatmap(heatmap)
    cmap = plt.get_cmap("jet")
    rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_visualizations(
    image_path: str | Path,
    heatmap: np.ndarray,
    sample_id: str,
    output_dir: str | Path,
    alpha: float = 0.45,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    heatmap_dir = ensure_dir(output_dir / "heatmaps")
    overlay_dir = ensure_dir(output_dir / "overlays")
    side_by_side_dir = ensure_dir(output_dir / "side_by_side")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    heatmap_npy_path = heatmap_dir / f"{sample_id}.npy"
    heatmap_png_path = heatmap_dir / f"{sample_id}.png"
    overlay_path = overlay_dir / f"{sample_id}.png"
    side_by_side_path = side_by_side_dir / f"{sample_id}.png"

    np.save(heatmap_npy_path, heatmap.astype(np.float32))
    heatmap_bgr = colorize_heatmap(heatmap)
    overlay = cv2.addWeighted(image, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
    side_by_side = np.concatenate([image, heatmap_bgr, overlay], axis=1)

    cv2.imwrite(str(heatmap_png_path), heatmap_bgr)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(side_by_side_path), side_by_side)

    return {
        "heatmap_npy": str(heatmap_npy_path),
        "heatmap_png": str(heatmap_png_path),
        "overlay_path": str(overlay_path),
        "side_by_side_path": str(side_by_side_path),
    }

