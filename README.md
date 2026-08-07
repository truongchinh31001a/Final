# Industrial Visual Anomaly Detection Prototype

This repository is a clean Python prototype for testing whether frozen foundation model features improve industrial visual anomaly detection over a conventional CNN baseline.

The experiment follows a PatchCore-style baseline:

1. Load only normal `train/good` images.
2. Extract frozen patch-level features from ResNet50 or DINOv2.
3. Normalize features and optionally apply standardization, PCA, and PCA whitening.
4. Store normal patch features in a memory bank, optionally with coreset sampling.
5. Score each test patch by nearest-neighbor distance to the memory bank.
6. Produce an image-level anomaly score and a pixel-level anomaly heatmap.

This is not the final thesis method. It is a baseline prototype for comparing feature representations.

## Dataset Structure

The loader expects an MVTec AD-style category layout:

```text
dataset/
  bottle/
    train/
      good/
    test/
      good/
      broken_large/
      broken_small/
      contamination/
    ground_truth/
      broken_large/
      broken_small/
      contamination/
```

Ground-truth masks are optional. If anomaly masks are missing, pixel-level AUROC is skipped gracefully.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

DINOv2 is loaded from `torch.hub`:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
```

The first DINOv2 run may download model code and weights.

## Run A Full Experiment

```bash
python run_experiment.py \
  --dataset_root ./data/mvtec \
  --category bottle \
  --backbone dinov2 \
  --image_size 224 \
  --coreset_ratio 0.1 \
  --output_dir ./outputs/bottle_dinov2
```

On Windows PowerShell:

```powershell
python run_experiment.py `
  --dataset_root ./data/mvtec `
  --category bottle `
  --backbone dinov2 `
  --image_size 224 `
  --coreset_ratio 0.1 `
  --output_dir ./outputs/bottle_dinov2
```

## Individual Scripts

Build the memory bank:

```bash
python train_memory.py \
  --dataset_root ./data/mvtec \
  --category bottle \
  --backbone resnet50 \
  --image_size 224 \
  --coreset_ratio 1.0 \
  --output_dir ./outputs/bottle_resnet50
```

Run inference:

```bash
python test_anomaly.py \
  --dataset_root ./data/mvtec \
  --category bottle \
  --output_dir ./outputs/bottle_resnet50
```

Evaluate saved scores:

```bash
python evaluate.py --output_dir ./outputs/bottle_resnet50
```

## Robust Feature Options

The feature processor is intentionally simple and modular:

```bash
python run_experiment.py \
  --dataset_root ./data/mvtec \
  --category bottle \
  --backbone dinov2 \
  --image_size 224 \
  --standardize \
  --pca_dim 128 \
  --whiten \
  --coreset_ratio 0.1 \
  --output_dir ./outputs/bottle_dinov2_pca128
```

Notes:

- L2 normalization is always applied.
- `--standardize` applies per-feature mean/std normalization fitted on normal train patches.
- `--pca_dim` reduces feature dimensionality.
- `--whiten` applies PCA whitening when PCA is enabled.
- `--coreset_ratio` reduces the memory bank after feature processing.

## Expected Outputs

Each experiment writes:

```text
outputs/
  memory_bank.pt
  scores.csv
  metrics.json
  heatmaps/
    defect__000.npy
    defect__000.png
  overlays/
    defect__000.png
  side_by_side/
    defect__000.png
```

The side-by-side visualization is:

```text
original image | heatmap | overlay
```

Printed metrics:

- Image AUROC
- Pixel AUROC, if masks are available
- Average inference time per image
- Memory bank size

## Compare ResNet50 vs DINOv2

Run four baseline variants:

```bash
python run_experiment.py --dataset_root ./data/mvtec --category bottle --backbone resnet50 --coreset_ratio 1.0 --output_dir ./outputs/bottle_resnet50_full
python run_experiment.py --dataset_root ./data/mvtec --category bottle --backbone dinov2   --coreset_ratio 1.0 --output_dir ./outputs/bottle_dinov2_full
python run_experiment.py --dataset_root ./data/mvtec --category bottle --backbone resnet50 --coreset_ratio 0.1 --output_dir ./outputs/bottle_resnet50_core01
python run_experiment.py --dataset_root ./data/mvtec --category bottle --backbone dinov2   --coreset_ratio 0.1 --output_dir ./outputs/bottle_dinov2_core01
```

Compare `metrics.json` and `scores.csv` across output folders. The main research question is whether DINOv2 patch tokens form a stronger frozen representation than ResNet50 layer3 features for nearest-neighbor anomaly scoring.

