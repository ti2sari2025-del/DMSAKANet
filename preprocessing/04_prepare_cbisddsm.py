"""
Step 4 — Build the CBIS-DDSM segmentation set (256 x 256) and its patient-level 5-fold split
============================================================================================
Input : Roboflow export of CBIS-DDSM full mammograms (already resized to 512 x 512)
        images: <prefix>.rf.<hash>.jpg
        masks : <prefix>.rf.<hash>_mask.png
        where <prefix> looks like  Mass-Training_P_00428_LEFT_CC_FULL_PRE_png
Output: CBIS_DDSM_256/images/<prefix>.png, CBIS_DDSM_256/masks/<prefix>.png
        CBIS_DDSM_256/manifest_cbisddsm.csv
        CBIS_DDSM_256/fold_assignments_cbisddsm.csv   (same format as the CDD-CESM fold file)

Preprocessing is identical to CDD-CESM: images read as 3-channel and resized directly to
256 x 256 with INTER_AREA; masks resized with INTER_NEAREST and binarized to {0, 255}.

Split protocol is identical to CDD-CESM: outer 5-fold StratifiedGroupKFold grouped by patient
(seed 42); for every outer fold ~15% of the training patients form a validation set.
CBIS-DDSM file names carry no pathology label, so folds are stratified by the patient's
lesion-size tercile (median mask area), which keeps small and large lesions balanced across folds.
The official CBIS-DDSM Training/Test assignment is recorded but not used.

Run from DATASET_LAMA:  python 04_prepare_cbisddsm.py
"""

import os
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

# ============================== CONFIG ==============================
SRC_IMG_DIR  = '../DATASET_BARU/Coding_Part_2_resized/train/images'
SRC_MASK_DIR = '../DATASET_BARU/Coding_Part_2_resized/train/masks'
OUT_DIR      = 'CBIS_DDSM_256'
SIZE         = 256
N_SPLITS     = 5
VAL_FRACTION = 0.15
SEED         = 42
# ======================================================================

PAT = re.compile(r'^(Mass|Calc)-(Training|Test)_(P_\d+)_(LEFT|RIGHT)_(CC|MLO)')


def prefix_of(fn):
    stem = os.path.splitext(fn)[0]
    stem = re.sub(r'_mask$', '', stem)
    return stem.split('.rf.')[0]


def main():
    imgs = sorted(f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    msks = sorted(f for f in os.listdir(SRC_MASK_DIR) if f.lower().endswith('.png'))
    print(f"Source: {len(imgs)} images, {len(msks)} masks")

    mask_by_stem = {os.path.splitext(m)[0].replace('_mask', ''): m for m in msks}
    rows, unmatched, unparsed, dup = [], [], [], []
    seen = set()
    for im in imgs:
        stem = os.path.splitext(im)[0]
        m = mask_by_stem.get(stem)
        if m is None:
            unmatched.append(im); continue
        pre = prefix_of(im)
        g = PAT.match(pre)
        if not g:
            unparsed.append(im); continue
        if pre in seen:
            dup.append(im); continue
        seen.add(pre)
        rows.append({'Image_name': pre, 'src_image': im, 'src_mask': m,
                     'Abnormality': g.group(1), 'Official_split': g.group(2),
                     'Patient_ID': g.group(3), 'Side': g.group(4), 'View': g.group(5)})
    print(f"Matched pairs: {len(rows)} | images without mask: {len(unmatched)} "
          f"| unparsed names: {len(unparsed)} | duplicate prefixes skipped: {len(dup)}")
    for lbl, lst in [('without mask', unmatched), ('unparsed', unparsed), ('duplicates', dup)]:
        if lst:
            print(f"  e.g. {lbl}: {lst[:3]}")

    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'masks'), exist_ok=True)
    keep = []
    for r in rows:
        img = cv2.imread(os.path.join(SRC_IMG_DIR, r['src_image']), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(SRC_MASK_DIR, r['src_mask']), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        msk = ((cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127) * 255).astype(np.uint8)
        r['area_pct'] = float((msk > 0).mean() * 100)
        r['empty_mask'] = bool(msk.max() == 0)
        cv2.imwrite(os.path.join(OUT_DIR, 'images', f"{r['Image_name']}.png"), img)
        cv2.imwrite(os.path.join(OUT_DIR, 'masks', f"{r['Image_name']}.png"), msk)
        keep.append(r)

    df = pd.DataFrame(keep)
    n_empty = int(df['empty_mask'].sum())
    if n_empty:
        print(f"Excluded {n_empty} images whose mask is empty at {SIZE}x{SIZE}")
        df = df[~df['empty_mask']].reset_index(drop=True)
    print(f"\nINCLUDED: {len(df)} images from {df.Patient_ID.nunique()} patients")
    print(df['Abnormality'].value_counts().to_string())
    print(df['Official_split'].value_counts().to_string())
    df.drop(columns=['src_image', 'src_mask']).to_csv(os.path.join(OUT_DIR, 'manifest_cbisddsm.csv'), index=False)

    # ---------------- patient-level split ----------------
    pat_area = df.groupby('Patient_ID')['area_pct'].median()
    terc = pd.qcut(pat_area, 3, labels=['small', 'medium', 'large'])
    df['strat'] = df['Patient_ID'].map(terc).astype(str)
    if df['Abnormality'].nunique() > 1:
        df['strat'] = df['Abnormality'] + '_' + df['strat']

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df['strat'], groups=df['Patient_ID'])):
        df.iloc[te, df.columns.get_loc('fold')] = k
    for k in range(N_SPLITS):
        role = np.where(df['fold'] == k, 'test', 'train').astype(object)
        tr = np.where(df['fold'] != k)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED + k)
        _, va = next(gss.split(tr, groups=df['Patient_ID'].values[tr]))
        role[tr[va]] = 'val'
        df[f'role_fold{k}'] = role

    assert (df.groupby('Patient_ID')['fold'].nunique() == 1).all()
    for k in range(N_SPLITS):
        assert (df.groupby('Patient_ID')[f'role_fold{k}'].nunique() == 1).all()
    print("\nLeakage checks passed.")

    summ = df.groupby('fold').agg(patients=('Patient_ID', 'nunique'), images=('Image_name', 'size'),
                                  median_area_pct=('area_pct', 'median'))
    print(summ.to_string())
    for k in range(N_SPLITS):
        parts = [f"{r} {int((df[f'role_fold{k}'] == r).sum()):4d}" for r in ('train', 'val', 'test')]
        print(f"  fold {k}: " + "  ".join(parts))

    df['Label'] = df['Abnormality']          # kept for compatibility with the training scripts
    df['patient_label'] = df['strat']
    cols = ['Image_name', 'Patient_ID', 'Label', 'patient_label', 'fold'] + [f'role_fold{k}' for k in range(N_SPLITS)]
    df[cols].to_csv(os.path.join(OUT_DIR, 'fold_assignments_cbisddsm.csv'), index=False)
    print(f"\nSaved: {OUT_DIR}/fold_assignments_cbisddsm.csv and manifest_cbisddsm.csv")


if __name__ == '__main__':
    main()
