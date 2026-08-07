from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate image-level and pixel-level AUROC.")
    parser.add_argument("--scores_csv", default=None)
    parser.add_argument("--output_dir", default="./outputs")
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int]:
    from src.metrics import evaluate_scores_csv
    from src.utils import ensure_dir, format_metric, write_json

    output_dir = ensure_dir(args.output_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else output_dir / "scores.csv"
    metrics = evaluate_scores_csv(scores_csv)
    write_json(output_dir / "metrics.json", metrics)

    print("[eval] Image AUROC:", format_metric(float(metrics["image_auroc"])))
    print("[eval] Pixel AUROC:", format_metric(float(metrics["pixel_auroc"])))
    print("[eval] Average inference time per image:", f"{metrics['average_inference_time_sec']:.4f}s")
    print("[eval] Memory bank size:", metrics["memory_bank_size"])
    print("[eval] Metrics saved:", output_dir / "metrics.json")
    return metrics


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
