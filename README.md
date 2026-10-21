# DMSA-KANet

Code, data-preparation scripts, patient-level fold files, and per-fold results for

> D. Titisari et al., "DMSA-KANet: Improving Breast Cancer Segmentation in Mammography through Deep
> Supervision Multi-Scale Attention and Kolmogorov-Arnold Networks", *International Journal of
> Intelligent Engineering and Systems* (under review, paper ID 20265008).

DMSA-KANet combines an EfficientNetB4 encoder, a high-resolution path, a multi-scale bottleneck
(dilated convolutions, Kolmogorov-Arnold Network layers, and a Vision Transformer), a KAN-based CBAM
module, and a decoder with KAN-gated dense skip connections and deep supervision.

**Release used for the manuscript:** tag `v2.0-ijies` (see *Code version* below).

## Repository layout

| Folder | Contents |
|---|---|
| `preprocessing/` | Mask generation, resizing, inclusion rules, and patient-level split scripts for CDD-CESM and CBIS-DDSM |
| `splits/` | Fold files listing the role (train / val / test) of every image in every outer fold, for both datasets |
| `training/` | Training and evaluation scripts for DMSA-KANet and the six baselines, the complexity script, the nested learning-rate selection, and all per-fold results (`training/results/`) |
| `legacy/` | The earlier image-level training script of the first submission. **Not used for any result in the current manuscript** |

## Settings used in the manuscript

| Setting | Value |
|---|---|
| Evaluation | Patient-level 5-fold cross-validation (`StratifiedGroupKFold`, seed 42); ~15% of the training patients of each outer fold form the validation set (`GroupShuffleSplit`, seed 42 + fold) |
| Model selection | Checkpoint with the highest validation Dice; LR halved after 5 epochs without improvement; threshold chosen on the validation set from 0.01–0.90 by pooled IoU; test fold used once |
| DMSA-KANet schedule | Phase 1: frozen encoder, **LR1 = 1e-3**, 20 epochs. Phase 2: blocks 4–7 unfrozen, **LR2 = 1e-5** (Trial 8 of the search in Table 5) |
| Batch size / epochs | 8 / 50 (CDD-CESM, 6× augmentation), 30 (CBIS-DDSM, 3× augmentation) |
| Loss | Focal Tversky 0.4 + Lovász 0.3 + Focal 0.2 + Dice 0.1; deep-supervision weights 0.5 / 0.2 / 0.2 / 0.1 |
| Metrics | TP, FP, FN, TN pooled over all pixels of a test fold; Dice, IoU, precision, recall, FPR, FNR, ROC-AUC from the pooled counts (F1 equals Dice); per-image Dice in addition; mean ± SD over five folds |
| Complexity | 120.50 M parameters (103.56 M trainable in Phase 1); **56.70 GFLOPs** (see below) |

The full protocol, the baseline configurations, and the commands are in `training/README.md`.

## Which file produced which table

| Manuscript | Produced by | Stored output |
|---|---|---|
| Table 2 (dataset composition) | `preprocessing/*` | `splits/*.csv` |
| Tables 6–7 (patient-level results) | `training/train_dmsa_kanet_patientlevel.py`, `training/train_baseline_patientlevel.py` (default settings), collected by `training/collect_results.py` | `training/results/all_fold_results.csv`, `all_summary.csv` |
| Nested learning-rate selection (Section 5.1) | `training/run_nested_cbis.sh`, `training/select_nested.py` | `training/results/nested_selection_cbisddsm.csv` |
| Table 9 (complexity) | `training/compute_complexity.py` | printed output, reproduced in `training/README.md` |
| Tables 5 and 8 (search and ablation) | earlier image-level experiments (single 80/20 split of CBIS-DDSM, seed 789); not protocol-matched and not reproduced by these scripts | values in the manuscript |

## FLOPs convention

56.70 GFLOPs is the number of floating-point operations of one forward pass (batch size 1,
256 × 256 × 3 input, all four outputs) counted by the TensorFlow 2.15 profiler
(`tf.compat.v1.profiler`, `float_operation()` option) on the traced inference graph, in which one
multiply–accumulate counts as two operations. The profiler counts only operations that have a
registered FLOP rule. Run `CUDA_VISIBLE_DEVICES="" python training/compute_complexity.py` to
reproduce it. An earlier version of this README quoted "~53.1 GFLOPs"; that was a hand estimate and
is superseded by the profiler value. The `estimate_flops()` helper inside the training scripts walks
only the top-level Keras layers and undercounts; it is not used for Table 9.

## Code version

The tag `v2.0-ijies` marks the code of the revised manuscript. Relative to the scripts that produced
Tables 6 and 7, `train_dmsa_kanet_patientlevel.py` differs only in (i) an optional learning-rate
override (`LR1`, `LR2`, `RUN_TAG` environment variables) and two extra fields in the result files
(configuration and best validation Dice), and (ii) identifier names: the model builder, the trainer
class and the Keras package of the custom layers, which still carried the project's earlier working
title (AMEDDE-Net), were renamed to DMSA-KANet. The old layer names are registered as aliases, so the
saved checkpoints still load. With the variables unset, the script is identical in behaviour
(same architecture, 120,497,866 parameters, 56.70 GFLOPs). Earlier versions of this README described
the first-submission settings (LR2 = 5e-5, image-level folds); those settings are not used in the
current manuscript.

## Environment

Python 3.11.7, TensorFlow 2.15.0, CUDA 12.2, cuDNN 8, scikit-learn 1.3.0, keras-unet-collection 0.1.13,
one NVIDIA RTX A6000 (48 GB). Exact versions of the packages used by the scripts: `requirements.txt`;
full list of the installed packages (`pip freeze`): `requirements_full_paperspace.txt`.

## Data

CDD-CESM and CBIS-DDSM are available from The Cancer Imaging Archive and are not redistributed here.
Patient identifiers in the fold files are the anonymized indices of each dataset. Trained checkpoints
are available from the corresponding author on request.

## License

MIT (see `LICENSE`).
