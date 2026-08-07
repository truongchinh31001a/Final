from __future__ import annotations

import argparse
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patch nearest-neighbor anomaly inference.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backbone", choices=["resnet50", "dinov2"], default=None)
    parser.add_argument("--dinov2_model", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--memory_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory_chunk_size", type=int, default=16384)
    parser.add_argument("--patch_chunk_size", type=int, default=2048)
    return parser


def run(args: argparse.Namespace) -> Path:
    import pandas as pd
    import torch

    from src.anomaly_scorer import NearestNeighborAnomalyScorer
    from src.datasets import MVTecDataset
    from src.feature_extractors import build_feature_extractor
    from src.memory_bank import load_memory_bank
    from src.utils import ensure_dir, get_device, make_sample_id, set_seed
    from src.visualization import save_visualizations

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    memory_path = Path(args.memory_path) if args.memory_path else output_dir / "memory_bank.pt"
    memory_features, processor, metadata = load_memory_bank(memory_path)

    backbone = args.backbone or metadata.get("backbone", "resnet50")
    dinov2_model = args.dinov2_model or metadata.get("dinov2_model", "dinov2_vits14")
    image_size = args.image_size or int(metadata.get("image_size", 224))
    device = get_device(args.device)

    if metadata.get("backbone") and metadata["backbone"] != backbone:
        print(
            f"[test] warning: memory bank backbone={metadata['backbone']} "
            f"but inference backbone={backbone}"
        )

    dataset = MVTecDataset(
        dataset_root=args.dataset_root,
        category=args.category,
        split="test",
        image_size=image_size,
    )
    extractor = build_feature_extractor(backbone, device=device, dinov2_model=dinov2_model)
    scorer = NearestNeighborAnomalyScorer(
        memory_features=memory_features,
        device=device,
        memory_chunk_size=args.memory_chunk_size,
        patch_chunk_size=args.patch_chunk_size,
    )

    rows: list[dict[str, object]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        original_size = tuple(int(x) for x in sample["orig_size"].tolist())

        start = time.perf_counter()
        features, grid_size = extractor.extract(image)
        processed = processor.transform(features.reshape(-1, features.shape[-1]).cpu().numpy())
        patch_features = torch.from_numpy(processed).float()
        patch_map, heatmap, image_score = scorer.score(
            patch_features=patch_features,
            grid_size=grid_size,
            original_size=original_size,
        )
        elapsed = time.perf_counter() - start

        sample_id = make_sample_id(sample["path"], relative_to=dataset.test_dir)
        paths = save_visualizations(
            image_path=sample["path"],
            heatmap=heatmap,
            sample_id=sample_id,
            output_dir=output_dir,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": sample["path"],
                "label": int(sample["label"]),
                "defect_type": sample["defect_type"],
                "mask_path": sample["mask_path"],
                "image_score": image_score,
                "patch_grid_h": int(patch_map.shape[0]),
                "patch_grid_w": int(patch_map.shape[1]),
                "inference_time_sec": elapsed,
                "memory_bank_size": int(memory_features.shape[0]),
                **paths,
            }
        )

        print(
            f"[test] {index + 1}/{len(dataset)} id={sample_id} "
            f"score={image_score:.4f} time={elapsed:.3f}s"
        )

    scores_csv = output_dir / "scores.csv"
    pd.DataFrame(rows).to_csv(scores_csv, index=False)
    avg_time = sum(float(row["inference_time_sec"]) for row in rows) / max(1, len(rows))
    print(f"[test] saved={scores_csv} average_inference_time={avg_time:.4f}s/image")
    return scores_csv


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
