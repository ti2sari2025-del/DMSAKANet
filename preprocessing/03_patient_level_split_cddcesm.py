"""
Step 3 — Patient-level, label-stratified 5-fold split for CDD-CESM (650 images)
==============================================================================

Input : manifest_cddcesm_650.csv   (written by 02_resize_cddcesm_256.py)
Output: fold_assignments_cddcesm_650.csv

Protocol
--------
* Outer 5-fold cross-validation, grouped by Patient_ID (StratifiedGroupKFold),
  so that all images of a patient fall in exactly one test fold.
* Stratification label per patient: 'Malignant' if ANY of the patient's images
  is malignant, otherwise 'Benign'.
* For every outer fold k, a patient-level validation subset (~VAL_FRACTION of the
  training patients) is drawn from the remaining four folds only. It is used for
  early stopping / checkpoint selection. The test fold is never used for any
  model selection.
* The resulting role of every image in every outer fold is written explicitly
  (columns role_fold0 ... role_fold4 with values train / val / test), so the
  exact split can be reproduced and audited.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

# ============================== CONFIG ==============================
MANIFEST     = "./CDD_DM_650_256/manifest_cddcesm_650.csv"
OUT_CSV      = "./CDD_DM_650_256/fold_assignments_cddcesm_650.csv"
N_SPLITS     = 5
VAL_FRACTION = 0.15
SEED         = 42
# ======================================================================


def main():
    df = pd.read_csv(MANIFEST)
    df = df[["Image_name", "Patient_ID", "Label"] +
            [c for c in df.columns if c.startswith("flag_")]].copy()

    pat_label = df.groupby("Patient_ID")["Label"].agg(
        lambda s: "Malignant" if (s == "Malignant").any() else "Benign")
    df["patient_label"] = df["Patient_ID"].map(pat_label)
    mixed = df.groupby("Patient_ID")["Label"].nunique()
    print(f"Images: {len(df)} | patients: {df.Patient_ID.nunique()} "
          f"| patients with both benign and malignant images: {(mixed > 1).sum()}")

    # ---- outer folds ----
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["patient_label"], groups=df["Patient_ID"])):
        df.iloc[te, df.columns.get_loc("fold")] = k

    # ---- inner validation (patient-level, inside the training folds only) ----
    for k in range(N_SPLITS):
        role = np.where(df["fold"] == k, "test", "train").astype(object)
        tr_idx = np.where(df["fold"] != k)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED + k)
        _, va_rel = next(gss.split(tr_idx, groups=df["Patient_ID"].values[tr_idx]))
        role[tr_idx[va_rel]] = "val"
        df[f"role_fold{k}"] = role

    # ---- leakage checks ----
    assert (df.groupby("Patient_ID")["fold"].nunique() == 1).all(), "Patient in >1 test fold"
    for k in range(N_SPLITS):
        roles_per_patient = df.groupby("Patient_ID")[f"role_fold{k}"].nunique()
        assert (roles_per_patient == 1).all(), f"Patient split across roles in fold {k}"
    print("Leakage checks passed: every patient has exactly one role in every fold.\n")

    # ---- summary ----
    print("Outer folds (test sets):")
    summ = df.groupby("fold").agg(patients=("Patient_ID", "nunique"), images=("Image_name", "size"))
    summ = summ.join(pd.crosstab(df["fold"], df["Label"]))
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        summ = summ.join(df.groupby("fold")[flag_cols].sum())
    print(summ.to_string(), "\n")

    print("Train / val / test sizes per outer fold (images | patients):")
    for k in range(N_SPLITS):
        parts = []
        for r in ("train", "val", "test"):
            sub = df[df[f"role_fold{k}"] == r]
            parts.append(f"{r} {len(sub):3d} | {sub.Patient_ID.nunique():3d}")
        print(f"  fold {k}:  " + "   ".join(parts))

    out_cols = ["Image_name", "Patient_ID", "Label", "patient_label", "fold"] + \
               [f"role_fold{k}" for k in range(N_SPLITS)]
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
