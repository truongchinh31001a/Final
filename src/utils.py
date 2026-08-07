from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def l2_normalize_np(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def l2_normalize_torch(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(features, p=2, dim=dim)


def iter_image_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_sample_id(image_path: str | Path, relative_to: str | Path | None = None) -> str:
    image_path = Path(image_path)
    try:
        rel = image_path.relative_to(Path(relative_to)) if relative_to else image_path.name
    except ValueError:
        rel = image_path.name
    text = str(rel.with_suffix("")) if isinstance(rel, Path) else str(rel)
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", text)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "skipped"
    return f"{value:.4f}"

