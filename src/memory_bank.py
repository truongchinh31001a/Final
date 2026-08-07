from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .utils import ensure_dir, l2_normalize_np, torch_load


@dataclass
class FeatureProcessor:
    """Small robust feature representation module for patch features."""

    pca_dim: int | None = None
    standardize: bool = False
    whiten: bool = False
    eps: float = 1e-12

    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    pca_explained_variance: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureProcessor":
        features = np.asarray(features, dtype=np.float32)
        x = l2_normalize_np(features, eps=self.eps)

        if self.standardize:
            self.feature_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
            self.feature_std = x.std(axis=0, keepdims=True).astype(np.float32)
            x = (x - self.feature_mean) / np.maximum(self.feature_std, self.eps)

        max_components = min(x.shape[0], x.shape[1])
        if self.pca_dim and 0 < self.pca_dim < x.shape[1] and self.pca_dim <= max_components:
            pca = PCA(n_components=self.pca_dim, svd_solver="randomized", random_state=0)
            pca.fit(x)
            self.pca_mean = pca.mean_.astype(np.float32)
            self.pca_components = pca.components_.astype(np.float32)
            self.pca_explained_variance = pca.explained_variance_.astype(np.float32)
        elif self.pca_dim and (self.pca_dim >= x.shape[1] or self.pca_dim > max_components):
            warnings.warn(
                "Skipping PCA because pca_dim must be smaller than feature_dim and no larger than "
                f"min(num_patches, feature_dim). pca_dim={self.pca_dim}, "
                f"num_patches={x.shape[0]}, feature_dim={x.shape[1]}",
                RuntimeWarning,
            )

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = l2_normalize_np(features, eps=self.eps)

        if self.standardize:
            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("FeatureProcessor was not fitted before transform")
            x = (x - self.feature_mean) / np.maximum(self.feature_std, self.eps)

        if self.pca_components is not None:
            if self.pca_mean is None:
                raise RuntimeError("FeatureProcessor has PCA components but no PCA mean")
            x = (x - self.pca_mean) @ self.pca_components.T
            if self.whiten:
                if self.pca_explained_variance is None:
                    raise RuntimeError("FeatureProcessor has whitening enabled but no PCA variance")
                x = x / np.sqrt(self.pca_explained_variance + self.eps)

        return l2_normalize_np(x, eps=self.eps).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "pca_dim": self.pca_dim,
            "standardize": self.standardize,
            "whiten": self.whiten,
            "eps": self.eps,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "pca_mean": self.pca_mean,
            "pca_components": self.pca_components,
            "pca_explained_variance": self.pca_explained_variance,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureProcessor":
        processor = cls(
            pca_dim=state.get("pca_dim"),
            standardize=bool(state.get("standardize", False)),
            whiten=bool(state.get("whiten", False)),
            eps=float(state.get("eps", 1e-12)),
        )
        processor.feature_mean = state.get("feature_mean")
        processor.feature_std = state.get("feature_std")
        processor.pca_mean = state.get("pca_mean")
        processor.pca_components = state.get("pca_components")
        processor.pca_explained_variance = state.get("pca_explained_variance")
        return processor


def subsample_coreset(
    features: torch.Tensor,
    ratio: float,
    method: str = "random",
    seed: int = 0,
    greedy_limit: int = 5000,
) -> torch.Tensor:
    if not 0 < ratio <= 1:
        raise ValueError("coreset_ratio must be in (0, 1]")

    num_features = features.shape[0]
    keep = max(1, int(round(num_features * ratio)))
    if keep >= num_features:
        return features

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if method == "random":
        indices = torch.randperm(num_features, generator=generator)[:keep]
        return features[indices]

    if method != "greedy":
        raise ValueError("coreset method must be 'random' or 'greedy'")

    if num_features > greedy_limit or keep > greedy_limit:
        warnings.warn(
            "Greedy coreset is expensive for this memory size; falling back to random sampling. "
            f"num_features={num_features}, keep={keep}, greedy_limit={greedy_limit}",
            RuntimeWarning,
        )
        indices = torch.randperm(num_features, generator=generator)[:keep]
        return features[indices]

    x = F.normalize(features.float().cpu(), p=2, dim=1)
    selected = torch.empty(keep, dtype=torch.long)
    current = torch.randint(num_features, size=(1,), generator=generator).item()
    min_dist = torch.full((num_features,), float("inf"))

    for i in range(keep):
        selected[i] = current
        similarity = torch.mv(x, x[current])
        distance = torch.clamp(2.0 - 2.0 * similarity, min=0.0)
        min_dist = torch.minimum(min_dist, distance)
        current = int(torch.argmax(min_dist).item())

    return features[selected]


def build_memory_bank(
    feature_batches: list[torch.Tensor],
    processor: FeatureProcessor,
    coreset_ratio: float = 1.0,
    coreset_method: str = "random",
    seed: int = 0,
) -> tuple[torch.Tensor, FeatureProcessor, int]:
    if not feature_batches:
        raise ValueError("No feature batches were provided")

    raw_features = torch.cat(feature_batches, dim=0).float().cpu()
    raw_count = raw_features.shape[0]

    processor.fit(raw_features.numpy())
    processed = torch.from_numpy(processor.transform(raw_features.numpy())).float()
    processed = subsample_coreset(processed, ratio=coreset_ratio, method=coreset_method, seed=seed)
    processed = F.normalize(processed, p=2, dim=1).cpu()
    return processed, processor, raw_count


def save_memory_bank(
    path: str | Path,
    memory_features: torch.Tensor,
    processor: FeatureProcessor,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "memory_features": memory_features.cpu(),
            "processor": processor.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_memory_bank(path: str | Path) -> tuple[torch.Tensor, FeatureProcessor, dict[str, Any]]:
    artifact = torch_load(path, map_location="cpu")
    memory_features = artifact["memory_features"].float()
    processor = FeatureProcessor.from_state_dict(artifact["processor"])
    metadata = artifact.get("metadata", {})
    return memory_features, processor, metadata
