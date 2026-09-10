# DMSA-KANet

**Improving Breast Cancer Segmentation in Mammography through Deep Supervision Multi-Scale Attention and Kolmogorov-Arnold Networks**

This repository contains the training code for DMSA-KANet, a hybrid deep learning architecture for breast cancer lesion segmentation in mammography. DMSA-KANet combines a hierarchical CNN encoder (EfficientNetB4) with a dedicated high-resolution path, a multi-scale bottleneck (dilated convolutions + Kolmogorov-Arnold Network layers + Vision Transformer), a KAN-based CBAM attention module, and a decoder with KAN-gated dense skip connections and deep supervision.

Manuscript submitted to the *International Journal of Intelligent Engineering and Systems* (IJIES).

## Architecture overview

```
Input (256×256×3)
  └─ HR-Encoder (256×256, no downsampling)
  └─ Encoder ×2 → ×4 → ×8 → ×16 → ×32 (EfficientNetB4 backbone, feature projection to 64/128/256/512/512 ch)
       └─ MSFE Bottleneck (dilated convs, rates 1/3/6/12) → 1024ch
       └─ ViT Bottleneck (4 blocks, 8 heads, CLS token, learnable positional embeddings) → 1024ch
       └─ KAN-CBAM (KAN-based channel attention + spatial attention)
  └─ Decoder ×16 → ×8 → ×4 → ×2 → HR-Decoder
       (KAN-gated dense skip connections at each level; 4-scale deep supervision output)
```

- **Loss:** Focal Tversky (0.4) + Lovász hinge (0.3) + Focal (0.2) + Dice (0.1)
- **Training:** two-phase (Phase 1: frozen backbone, LR = 1e-3; Phase 2: partial backbone unfreeze, LR = 5e-5)
- **Evaluation:** 5-fold cross-validation

## Datasets

The model was evaluated on two public mammography datasets:

- **CDD-CESM** — Categorized Digital Database for Low Energy and Subtracted Contrast Enhanced Spectral Mammography Images, available via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).
- **CBIS-DDSM** — Curated Breast Imaging Subset of the Digital Database for Screening Mammography (mass subset used here).

Dataset files are not included in this repository. Update `image_directory` and `mask_directory` at the top of `train_dmsa_kanet.py` to point to your local copies before running.

## Requirements

See `requirements.txt`. Tested with Python 3.10 and TensorFlow 2.x on an NVIDIA GPU (RTX A6000 / RTX 2080).

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_dmsa_kanet.py
```

The script performs 5-fold cross-validation with per-fold checkpointing (safe to interrupt and resume — completed folds are automatically skipped on the next run). Outputs:

- `checkpoints_v11/fold_N_result.json` — per-fold metrics
- `checkpoints_v11/fold_N_history.json` — per-fold training history
- `amedde_v9_fold_N_best_dice.keras` — best model weights per fold
- `amedde_v11_kfold_results.csv`, `amedde_v11_kfold_summary.csv` — aggregated results
- `amedde_v11_complexity.csv` — parameter count (FLOPs estimate from this script covers convolutional layers only; see note below)

## Notes on this release

- This is the corrected version of the training script. An earlier internal version contained an algebra error in the F1-score formula (it computed `2·Precision·Recall` instead of `2·Precision·Recall / (Precision + Recall)`); for binary segmentation these two quantities are mathematically required to be identical to Dice, so the corrected formula is used throughout. See the `calculate_metrics()` function docstring for detail.
- The built-in `estimate_flops()` utility only walks the model's top-level Keras layers and does not recurse into the custom `ViTBottleneck` / `KANLayer` layers, so its FLOPs estimate undercounts the true computational cost. The manuscript reports a corrected estimate (~53.1 GFLOPs) that includes the Vision Transformer and KAN spline computations.

## Citation

If you use this code, please cite:

> [Author list], "DMSA-KANet: Improving Breast Cancer Segmentation in Mammography through Deep Supervision Multi-Scale Attention and Kolmogorov-Arnold Networks," *International Journal of Intelligent Engineering and Systems*, in press.

## License

See `LICENSE`.
