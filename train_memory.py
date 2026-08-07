from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a normal patch-feature memory bank.")
    parser.add_argument("--dataset_root", required=True, help="Root folder containing MVTec AD-style categories.")
    parser.add_argument("--category", required=True, help="Category name, for example bottle.")
    parser.add_argument("--backbone", choices=["resnet50", "dinov2"], default="resnet50")
    parser.add_argument("--dinov2_model", default="dinov2_vits14", help="torch.hub DINOv2 model name.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--coreset_ratio", type=float, default=1.0)
    parser.add_argument("--coreset_method", choices=["random", "greedy"], default="random")
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--whiten", action="store_true", help="Apply PCA whitening when PCA is enabled.")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--memory_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    import torch

    from src.datasets import MVTecDataset, make_dataloader
    from src.feature_extractors import build_feature_extractor
    from src.memory_bank import FeatureProcessor, build_memory_bank, save_memory_bank
    from src.utils import ensure_dir, get_device, set_seed

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    memory_path = Path(args.memory_path) if args.memory_path else output_dir / "memory_bank.pt"
    device = get_device(args.device)

    dataset = MVTecDataset(
        dataset_root=args.dataset_root,
        category=args.category,
        split="train",
        image_size=args.image_size,
    )
    loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    extractor = build_feature_extractor(args.backbone, device=device, dinov2_model=args.dinov2_model)

    feature_batches: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        features, grid_size = extractor.extract(images)
        feature_batches.append(features.reshape(-1, features.shape[-1]).cpu())
        if batch_index == 1 or batch_index % 20 == 0:
            print(
                f"[memory] batch={batch_index}/{len(loader)} "
                f"grid={grid_size} patches_collected={sum(x.shape[0] for x in feature_batches)}"
            )

    processor = FeatureProcessor(
        pca_dim=args.pca_dim,
        standardize=args.standardize,
        whiten=args.whiten,
    )
    memory_features, processor, raw_count = build_memory_bank(
        feature_batches=feature_batches,
        processor=processor,
        coreset_ratio=args.coreset_ratio,
        coreset_method=args.coreset_method,
        seed=args.seed,
    )

    metadata = {
        "dataset_root": str(Path(args.dataset_root)),
        "category": args.category,
        "backbone": args.backbone,
        "dinov2_model": args.dinov2_model,
        "image_size": args.image_size,
        "train_images": len(dataset),
        "raw_patch_features": int(raw_count),
        "memory_bank_size": int(memory_features.shape[0]),
        "feature_dim": int(memory_features.shape[1]),
        "coreset_ratio": args.coreset_ratio,
        "coreset_method": args.coreset_method,
        "pca_dim": args.pca_dim,
        "standardize": args.standardize,
        "whiten": args.whiten,
    }
    save_memory_bank(memory_path, memory_features, processor, metadata)
    print(
        f"[memory] saved={memory_path} size={memory_features.shape[0]} "
        f"dim={memory_features.shape[1]} raw_patches={raw_count}"
    )
    return memory_path


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
