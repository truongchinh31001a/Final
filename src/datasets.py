from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import IMAGE_EXTENSIONS, iter_image_files


@dataclass(frozen=True)
class MVTecSample:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


def _find_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None

    mask_dir = category_root / "ground_truth" / defect_type
    if not mask_dir.exists():
        return None

    stem = image_path.stem
    candidate_names: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        candidate_names.append(f"{stem}_mask{ext}")
        candidate_names.append(f"{stem}{ext}")

    for name in candidate_names:
        candidate = mask_dir / name
        if candidate.exists():
            return candidate

    matches = sorted(
        path
        for path in mask_dir.glob(f"{stem}*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0] if matches else None


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class MVTecDataset(Dataset):
    """Dataset reader for MVTec AD-style category folders."""

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        split: str,
        image_size: int = 224,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.dataset_root = Path(dataset_root)
        self.category = category
        self.category_root = self.dataset_root / category
        self.split = split
        self.image_size = image_size
        self.transform = build_transform(image_size)
        self.train_good_dir = self.category_root / "train" / "good"
        self.test_dir = self.category_root / "test"
        self.samples = self._collect_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No images found for {self.category_root} split={split}. "
                "Check that the dataset uses an MVTec AD-style structure."
            )

    def _collect_samples(self) -> list[MVTecSample]:
        if self.split == "train":
            return [
                MVTecSample(path=path, label=0, defect_type="good")
                for path in iter_image_files(self.train_good_dir)
            ]

        samples: list[MVTecSample] = []
        if not self.test_dir.exists():
            return samples

        for defect_dir in sorted(path for path in self.test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in iter_image_files(defect_dir):
                mask_path = _find_mask_path(self.category_root, defect_type, image_path)
                samples.append(
                    MVTecSample(
                        path=image_path,
                        label=label,
                        defect_type=defect_type,
                        mask_path=mask_path,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            img = img.convert("RGB")
            width, height = img.size
            tensor = self.transform(img)

        return {
            "image": tensor,
            "path": str(sample.path),
            "label": int(sample.label),
            "defect_type": sample.defect_type,
            "mask_path": str(sample.mask_path) if sample.mask_path else "",
            "orig_size": torch.tensor([height, width], dtype=torch.int32),
        }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

