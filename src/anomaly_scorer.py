from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class NearestNeighborAnomalyScorer:
    """Patch-level nearest-neighbor scoring against a normal memory bank."""

    def __init__(
        self,
        memory_features: torch.Tensor,
        device: torch.device,
        memory_chunk_size: int = 16384,
        patch_chunk_size: int = 2048,
    ) -> None:
        self.device = device
        self.memory_features = F.normalize(memory_features.float().to(device), p=2, dim=1)
        self.memory_chunk_size = memory_chunk_size
        self.patch_chunk_size = patch_chunk_size

    @torch.no_grad()
    def patch_scores(self, patch_features: torch.Tensor) -> torch.Tensor:
        patches = F.normalize(patch_features.float().to(self.device), p=2, dim=1)
        all_scores: list[torch.Tensor] = []

        for patch_start in range(0, patches.shape[0], self.patch_chunk_size):
            patch_chunk = patches[patch_start : patch_start + self.patch_chunk_size]
            best_similarity = torch.full(
                (patch_chunk.shape[0],),
                -float("inf"),
                device=self.device,
                dtype=patch_chunk.dtype,
            )

            for memory_start in range(0, self.memory_features.shape[0], self.memory_chunk_size):
                memory_chunk = self.memory_features[memory_start : memory_start + self.memory_chunk_size]
                similarity = patch_chunk @ memory_chunk.T
                best_similarity = torch.maximum(best_similarity, similarity.max(dim=1).values)

            distance = torch.sqrt(torch.clamp(2.0 - 2.0 * best_similarity, min=0.0))
            all_scores.append(distance.cpu())

        return torch.cat(all_scores, dim=0)

    def score(
        self,
        patch_features: torch.Tensor,
        grid_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        scores = self.patch_scores(patch_features)
        grid_h, grid_w = grid_size
        if grid_h * grid_w != scores.numel():
            raise ValueError(
                f"Patch grid {grid_size} does not match number of scores {scores.numel()}"
            )

        patch_map = scores.reshape(grid_h, grid_w).numpy()
        orig_h, orig_w = original_size
        heatmap = cv2.resize(patch_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap = np.maximum(heatmap, 0.0).astype(np.float32)
        image_score = float(scores.max().item())
        return patch_map.astype(np.float32), heatmap, image_score

