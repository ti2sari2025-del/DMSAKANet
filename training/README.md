# Training and evaluation (patient-level protocol)

Scripts that train and evaluate DMSA-KANet and the six baselines with the patient-level
five-fold protocol reported in the paper, and the per-fold results behind Tables 6 and 7.
The data preparation steps (masks, resizing, fold files) are in `../preprocessing/`, and the
fold files are in `../splits/`.

## Files

| File | Purpose |
|---|---|
| `train_dmsa_kanet_patientlevel.py` | Trains and evaluates DMSA-KANet on all five folds of one dataset |
| `train_baseline_patientlevel.py` | Trains and evaluates one baseline (U-Net, Attention U-Net, R2 U-Net, U-Net++, TransUNet, ViT) |
| `run_queue_priority.sh` | Runs every remaining model/dataset combination in sequence; safe to restart |
| `reevaluate_thresholds.py` | Re-selects the threshold on the validation set with the 0.01–0.90 grid from a saved checkpoint (used for the CDD-CESM DMSA-KANet run) |
| `collect_results.py` | Collects all finished folds and writes `all_fold_results.csv` and `all_summary.csv` |
| `run_nested_cbis.sh` | Nested learning-rate selection of DMSA-KANet on CBIS-DDSM (three extra configurations, see below) |
| `select_nested.py` | Chooses the configuration per outer fold on the validation patients and writes `nested_selection_cbisddsm.csv` |
| `compute_complexity.py` | Parameter count and FLOPs of DMSA-KANet (Table 9) |
| `collect_per_image.py` | Merges the per-image test metrics of all models into `results/per_image_<dataset>.csv` |
| `results/per_image_cddcesm.csv`, `results/per_image_cbisddsm.csv` | Dice and IoU of every test image, per model and fold |
| `results/nested_selection_cbisddsm.csv` | One row per fold × configuration, with the configuration chosen on validation marked |
| `results/all_fold_results.csv` | One row per dataset × model × fold (all metrics, selected threshold) |
| `results/all_summary.csv` | Mean ± SD over the five folds (Tables 6 and 7) |

## Environment

Python 3.11.7, TensorFlow 2.15.0, CUDA 12.2, cuDNN 8, scikit-learn 1.3.0,
keras-unet-collection 0.1.13. All runs used one NVIDIA RTX A6000 (48 GB).

```
pip install tensorflow==2.15.0 scikit-learn==1.3.0 keras-unet-collection==0.1.13 opencv-python pandas matplotlib
```

## Expected folder layout

```
CDD_DM_650_256/   images/  masks/  fold_assignments_cddcesm_650.csv
CBIS_DDSM_256/    images/  masks/  fold_assignments_cbisddsm.csv
train_dmsa_kanet_patientlevel.py
train_baseline_patientlevel.py
...
```

The datasets are not redistributed. Build both folders with the scripts in `../preprocessing/`.

## Running

```
# DMSA-KANet
DATASET=cddcesm  python train_dmsa_kanet_patientlevel.py
DATASET=cbisddsm python train_dmsa_kanet_patientlevel.py

# one baseline: unet | attunet | r2unet | unetpp | transunet | vit
DATASET=cddcesm  python train_baseline_patientlevel.py unet common
DATASET=cbisddsm python train_baseline_patientlevel.py unet common

# everything, in sequence, in the background
nohup bash run_queue_priority.sh > queue_priority_log.txt 2>&1 &

# summary of all finished folds
python collect_results.py
```

Results are written to `results_patientlevel_<dataset>/` (DMSA-KANet) and
`results_patientlevel_<dataset>/baseline_<model>_common/` (baselines): one
`fold_<k>_result.json` per fold, per-image metrics, and the test-set probability maps.

## Protocol (identical for all seven models)

- **Split:** patient-level five-fold cross-validation (`StratifiedGroupKFold`, seed 42). In each
  outer fold about 15% of the training patients form an inner validation set
  (`GroupShuffleSplit`, seed 42 + fold index). The role of every image is read from the fold files.
- **Model selection:** checkpoint with the highest validation Dice; learning rate halved when the
  validation Dice does not improve for 5 epochs (minimum 1e-7). The test fold is used once.
- **Threshold:** chosen on the validation set from 0.01–0.90 (step 0.01) by pooled IoU, then
  applied unchanged to the test fold.
- **Loss / optimizer / batch:** composite loss (Focal Tversky 0.4 + Lovász 0.3 + Focal 0.2 +
  Dice 0.1), Adam, batch size 8. Deep supervision and the two-phase schedule apply to DMSA-KANet only.
- **Epochs and augmentation:** CDD-CESM 50 epochs, 6× augmentation (original, horizontal flip,
  vertical flip, rotation ±15°, gamma 0.7–1.3, cutout). CBIS-DDSM 30 epochs, 3× augmentation
  (original, horizontal flip, one of the other four per image). No early stopping.
- **Test-time augmentation:** mean of the predictions on the original, horizontally, vertically,
  and doubly flipped image.
- **Metrics:** TP, FP, FN, and TN pooled over all pixels of a test fold; Dice, IoU, precision,
  recall, FPR, FNR, and ROC-AUC from the pooled counts (F1 equals Dice and is not reported
  separately); mean per-image Dice in addition. Reported as mean ± SD over the five folds.
- **Seed:** 42 (Python, NumPy, TensorFlow).

### Model-specific settings

| Model | Implementation | Initialization | Learning rate | Parameters |
|---|---|---|---|---|
| U-Net | keras-unet-collection `unet_2d`, filters 64–1024 | random | 1e-3 | 34.53 M |
| Attention U-Net | keras-unet-collection `att_unet_2d` | random | 1e-3 | 31.92 M |
| R2 U-Net | keras-unet-collection `r2_unet_2d`, 2 recurrent steps | random | 1e-3 | 101.98 M |
| U-Net++ | Keras re-implementation, filters 64–1024 | random | 1e-3 | 36.62 M |
| TransUNet | Keras re-implementation, 8 blocks, 768-d, 12 heads | random | 1e-4 | 61.64 M |
| ViT | 16×16 patches, 8 blocks, 768-d, 12 heads, conv decoder | random | 1e-4 | 46.87 M |
| DMSA-KANet | this work | EfficientNetB4 encoder (ImageNet), other layers random | 1e-3 (phase 1, frozen encoder, 20 epochs) → 1e-5 (phase 2) | 120.50 M |

## Results (pooled Dice, mean ± SD over five patient-level folds, %)

| Model | CDD-CESM | CBIS-DDSM |
|---|---|---|
| TransUNet | **51.50 ± 3.37** | **51.06 ± 4.26** |
| U-Net | 51.02 ± 3.94 | 46.80 ± 5.86 |
| Attention U-Net | 49.42 ± 3.82 | 46.32 ± 5.11 |
| R2 U-Net | 49.30 ± 4.12 | 47.65 ± 3.97 |
| ViT | 47.83 ± 2.76 | 16.36 ± 9.25 |
| U-Net++ | 46.60 ± 5.69 | 2.66 ± 0.25 |
| DMSA-KANet | 43.97 ± 9.11 | 39.98 ± 8.40 |

Full metrics are in `results/all_summary.csv` and per fold in `results/all_fold_results.csv`.
Under this protocol DMSA-KANet does not outperform the baselines. On CBIS-DDSM, U-Net++ collapsed
to all-lesion predictions in every fold and ViT trained unreliably; both are reported as observed,
without re-tuning, because every model received the same recipe.

## Nested learning-rate selection (CBIS-DDSM)

The learning rates of DMSA-KANet were originally chosen on an image-level split of CBIS-DDSM whose
validation images overlap the patient-level test folds. To keep that choice inside the training
patients, `run_nested_cbis.sh` trains three further configurations on the same folds, inner
validation patients, seeds, and epochs: LR1 ∈ {1e-3, 1e-4} × LR2 ∈ {5e-6, 1e-5} (batch size 8;
LR1 = 1e-3 and LR2 = 1e-5 is the run of Table 7). The grid is the 2 × 2 grid spanned by the three
best trials of the earlier search (Trials 8, 6, and 5, all with batch size 8) and was fixed before
any of these runs.
For each outer fold, `select_nested.py` takes the configuration with the highest best-validation
Dice and reports its test result; the test fold plays no part in the choice.

```
nohup bash run_nested_cbis.sh > nested_log.txt 2>&1 &   # resumable
python select_nested.py
```

## Complexity (Table 9)

```
CUDA_VISIBLE_DEVICES="" python compute_complexity.py
```

Output (TensorFlow 2.15.0):

```
Total parameters      : 120,497,866 (120.50 M)
Trainable (phase 1)   : 103,555,163 (103.56 M)
Non-trainable         : 16,942,703 (16.94 M)
Trainable (phase 2)   : 120,028,017 (120.03 M)
FLOPs, all four outputs (deep-supervision heads included): 56,700,734,031 (56.70 G)
FLOPs, main output only                                   : 56,697,042,505 (56.70 G)   # OUTPUTS=main
```

FLOPs are counted by the TensorFlow profiler (`float_operation()`) on the traced inference graph of
one forward pass at batch size 1 and 256 × 256 × 3 input; one multiply–accumulate counts as two
operations, and only operations with a registered FLOP rule are counted. The encoder is built
without ImageNet weights for this count, which does not change it. The `estimate_flops()` helper in
the training script covers only top-level Conv2D/Dense layers and is not used for Table 9.

## Implementation notes

- **Resuming:** `train_dmsa_kanet_patientlevel.py` saves the model weights and the training state
  (phase, epoch, learning rate, plateau counter, best validation values) after every epoch and
  resumes from the last finished epoch after an interruption. The Adam moment estimates are
  re-initialised on resume. The CBIS-DDSM DMSA-KANet run was resumed in this way after machine
  shutdowns. The baseline script skips finished folds; an interrupted fold restarts from epoch 1.
- **Threshold grid:** the CDD-CESM DMSA-KANet run was first evaluated with a narrower grid
  (0.05–0.575). Its saved best-validation checkpoints were re-evaluated with the 0.01–0.90 grid by
  `reevaluate_thresholds.py`, so every model uses the same threshold rule.
- **Not protocol-matched:** the hyperparameter search (Table 5) and the ablation (Table 8) were run
  earlier on an image-level 80/20 split of CBIS-DDSM and are not reproduced by these scripts.

## Checkpoints

Trained checkpoints are too large for this repository and are available from the corresponding
author on request.
