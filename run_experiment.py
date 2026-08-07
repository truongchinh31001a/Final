from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import test_anomaly
import train_memory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end industrial visual anomaly detection baseline."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backbone", choices=["resnet50", "dinov2"], default="resnet50")
    parser.add_argument("--dinov2_model", default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--coreset_ratio", type=float, default=1.0)
    parser.add_argument("--coreset_method", choices=["random", "greedy"], default="random")
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--memory_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory_chunk_size", type=int, default=16384)
    parser.add_argument("--patch_chunk_size", type=int, default=2048)
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int]:
    from src.utils import ensure_dir

    output_dir = ensure_dir(args.output_dir)
    memory_path = Path(args.memory_path) if args.memory_path else output_dir / "memory_bank.pt"

    train_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        category=args.category,
        backbone=args.backbone,
        dinov2_model=args.dinov2_model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        coreset_ratio=args.coreset_ratio,
        coreset_method=args.coreset_method,
        pca_dim=args.pca_dim,
        standardize=args.standardize,
        whiten=args.whiten,
        output_dir=str(output_dir),
        memory_path=str(memory_path),
        device=args.device,
        seed=args.seed,
    )
    train_memory.run(train_args)

    test_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        category=args.category,
        backbone=args.backbone,
        dinov2_model=args.dinov2_model,
        image_size=args.image_size,
        output_dir=str(output_dir),
        memory_path=str(memory_path),
        device=args.device,
        seed=args.seed,
        memory_chunk_size=args.memory_chunk_size,
        patch_chunk_size=args.patch_chunk_size,
    )
    scores_csv = test_anomaly.run(test_args)

    eval_args = argparse.Namespace(scores_csv=str(scores_csv), output_dir=str(output_dir))
    return evaluate.run(eval_args)


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
