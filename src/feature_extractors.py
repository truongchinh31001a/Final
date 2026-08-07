from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


BackboneName = Literal["resnet50", "dinov2"]


class PatchFeatureExtractor(nn.Module):
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        raise NotImplementedError


class ResNet50PatchExtractor(PatchFeatureExtractor):
    """Frozen ResNet50 feature map extractor using the layer3 activation map."""

    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.feature_dim = 1024
        self._freeze()

    def _freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature_map = self.encoder(images)
        batch, channels, height, width = feature_map.shape
        features = feature_map.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        features = F.normalize(features, p=2, dim=-1)
        return features, (height, width)


class DINOv2PatchExtractor(PatchFeatureExtractor):
    """Frozen DINOv2 patch-token extractor loaded from torch.hub."""

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self._freeze()

    def _freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _infer_grid(num_patches: int) -> tuple[int, int]:
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise ValueError(f"Cannot infer a square patch grid from {num_patches} patch tokens")
        return side, side

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if hasattr(self.model, "forward_features"):
            outputs = self.model.forward_features(images)
            if isinstance(outputs, dict) and "x_norm_patchtokens" in outputs:
                patch_tokens = outputs["x_norm_patchtokens"]
            elif isinstance(outputs, dict) and "x_prenorm" in outputs:
                patch_tokens = outputs["x_prenorm"][:, 1:]
            else:
                raise RuntimeError("DINOv2 forward_features returned an unsupported output format")
        elif hasattr(self.model, "get_intermediate_layers"):
            patch_tokens = self.model.get_intermediate_layers(
                images,
                n=1,
                reshape=False,
                return_class_token=False,
            )[0]
        else:
            raise RuntimeError("The loaded DINOv2 model does not expose patch-token features")

        if patch_tokens.dim() == 4:
            batch, channels, height, width = patch_tokens.shape
            features = patch_tokens.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            grid = (height, width)
        elif patch_tokens.dim() == 3:
            grid = self._infer_grid(patch_tokens.shape[1])
            features = patch_tokens
        else:
            raise RuntimeError(f"Unsupported DINOv2 patch feature shape: {tuple(patch_tokens.shape)}")

        features = F.normalize(features, p=2, dim=-1)
        return features, grid


def build_feature_extractor(
    backbone: BackboneName,
    device: torch.device,
    dinov2_model: str = "dinov2_vits14",
) -> PatchFeatureExtractor:
    if backbone == "resnet50":
        extractor: PatchFeatureExtractor = ResNet50PatchExtractor()
    elif backbone == "dinov2":
        extractor = DINOv2PatchExtractor(model_name=dinov2_model)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    extractor.to(device)
    extractor.eval()
    return extractor

