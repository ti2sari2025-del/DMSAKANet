# Data preparation (CDD-CESM)

Scripts that rebuild the CDD-CESM segmentation set and the patient-level folds used in the paper.
Run them in order.

| Step | File | Where it was run | Output |
|---|---|---|---|
| 1 | `01_generate_masks_cddcesm.ipynb` | Google Colab | Binary masks at native resolution from the radiologists' hand-drawn ROIs (`Radiology_hand_drawn_segmentations_v2.csv`), plus `master_table.csv` with labels and quality flags |
| 2 | `02_resize_cddcesm_256.py` | Google Colab | 650 images and masks resized to 256 × 256, plus `manifest_cddcesm_650.csv` |
| 3 | `03_patient_level_split_cddcesm.py` | Paperspace | `fold_assignments_cddcesm_650.csv` (patient-level 5-fold split with train/val/test roles) |

## Source data

CDD-CESM (Khaled et al., *Scientific Data* 9:122, 2022, https://doi.org/10.1038/s41597-022-01238-0), distributed by
The Cancer Imaging Archive under CC BY 4.0: https://www.cancerimagingarchive.net/collection/cdd-cesm/

Download the low-energy images, `Radiology-manual-annotations.xlsx` and `Radiology_hand_drawn_segmentations_v2.csv`.
Images and annotation files are not redistributed in this repository.

## Inclusion criteria

```
1003 low-energy (DM) images, 326 patients
 -> excluded 341 Normal images
 -> excluded 12 benign/malignant images without an ROI annotation
 =  650 images (324 benign, 326 malignant) from 282 patients
```

No mask became empty after resizing to 256 × 256.

## Mask generation

Polygon ROIs are filled; ellipse and circle ROIs are drawn as filled ellipses using the annotated rotation angle;
point annotations are drawn as discs of 25-pixel radius; open polylines are closed and filled.
All ROIs of an image are merged into one binary mask (0/255).

## Resizing

Images are read as 3-channel and resized directly to 256 × 256 (no aspect-ratio preservation) with `cv2.INTER_AREA`.
Masks are resized with `cv2.INTER_NEAREST` and binarized to {0, 255}. Files are saved as PNG.

## Patient-level split

- Outer 5-fold cross-validation grouped by patient (`StratifiedGroupKFold`, seed 42), so every patient appears in exactly one test fold.
- Stratification label per patient: malignant if any of the patient's images is malignant, otherwise benign
  (57 patients have both benign and malignant images).
- For each outer fold, about 15% of the training patients form a validation set, drawn from the training folds only.
  It is used for checkpoint selection, learning-rate scheduling and threshold selection. The test fold is used once, for the final evaluation.
- The role of every image in every fold is stored explicitly in `splits/fold_assignments_cddcesm_650.csv`
  (columns `role_fold0` … `role_fold4`). Patient IDs are the anonymized dataset indices (`P1` … `P326`).
