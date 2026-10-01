"""
Step 2 — Build the CDD-CESM segmentation set (650 images) and resize to 256x256
===============================================================================

Input : outputs of step 1 (01_generate_masks_cddcesm.ipynb)
        - masks_generated/master_table.csv
        - original low-energy (DM) images + native-resolution masks

Inclusion rule (reproduces the flowchart in the manuscript):
    Type == 'DM'                       (exclude subtracted CESM images)
    Label in {'Benign', 'Malignant'}   (exclude Normal)
    has_roi == True                    (exclude lesions without an ROI annotation)

Preprocessing (same recipe as used for CBIS-DDSM):
    images : read as 3-channel (cv2.IMREAD_COLOR), resized directly to SIZE x SIZE
             (no aspect-ratio preservation), cv2.INTER_AREA, saved as PNG
    masks  : read as grayscale, resized with cv2.INTER_NEAREST, binarized to {0, 255}, saved as PNG
    Image and mask share the same file name: {Image_name}.png

Outputs (in OUT_DIR):
    images/{Image_name}.png
    masks/{Image_name}.png
    manifest_cddcesm_650.csv   (image list + label + flags; no absolute paths)
    resize_log.txt             (counts, and any mask that became empty after resizing)

Run in Google Colab (where the step-1 outputs live), then zip OUT_DIR and upload to Paperspace.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
BASE       = "/content/drive/MyDrive/CDD-CESM"
MASTER_CSV = os.path.join(BASE, "masks_generated", "master_table.csv")
OUT_DIR    = os.path.join(BASE, "CDD_DM_650_256")
SIZE       = 256
# ======================================================================


def main():
    master = pd.read_csv(MASTER_CSV)

    dm = master[master["Type"].astype(str).str.strip() == "DM"]
    abn = dm[dm["Label"].isin(["Benign", "Malignant"])]
    sel = abn[abn["has_roi"] == True].copy()
    sel["Image_name"] = (sel["Image_name"].astype(str)
                         .str.replace(r"\.\w+$", "", regex=True).str.strip())

    log = []
    def say(msg):
        print(msg)
        log.append(msg)

    say("Inclusion flowchart")
    say(f"  All rows in master table          : {len(master)}")
    say(f"  Low-energy (DM) images            : {len(dm)}")
    say(f"  Excluded Normal                   : {(dm['Label'] == 'Normal').sum()}")
    say(f"  Benign + Malignant                : {len(abn)}")
    say(f"  Excluded (no ROI annotation)      : {len(abn) - len(sel)}")
    say(f"  INCLUDED images                   : {len(sel)}")
    say(f"  INCLUDED patients                 : {sel['Patient_ID'].nunique()}")
    for lab, n in sel["Label"].value_counts().items():
        say(f"    {lab:10s}: {n} images, {sel.loc[sel.Label == lab, 'Patient_ID'].nunique()} patients")

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

    empty_after_resize, size_mismatch, failed = [], [], []
    for r in sel.itertuples():
        img = cv2.imread(r.image_path, cv2.IMREAD_COLOR)
        msk = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            failed.append(r.Image_name)
            continue
        if img.shape[:2] != msk.shape[:2]:
            size_mismatch.append(r.Image_name)
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        img_r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        msk_r = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        msk_r = ((msk_r > 127) * 255).astype(np.uint8)
        if msk_r.max() == 0:
            empty_after_resize.append(r.Image_name)

        cv2.imwrite(os.path.join(OUT_DIR, "images", f"{r.Image_name}.png"), img_r)
        cv2.imwrite(os.path.join(OUT_DIR, "masks", f"{r.Image_name}.png"), msk_r)

    say("\nQuality checks")
    say(f"  Failed to read                    : {len(failed)} {failed[:10]}")
    say(f"  Image/mask size mismatch          : {len(size_mismatch)} {size_mismatch[:10]}")
    say(f"  Mask EMPTY after resize to {SIZE}   : {len(empty_after_resize)} {empty_after_resize[:10]}")

    sel["empty_after_resize"] = sel["Image_name"].isin(empty_after_resize)
    keep_cols = [c for c in sel.columns if c not in ("image_path", "mask_path")]
    sel[keep_cols].to_csv(os.path.join(OUT_DIR, "manifest_cddcesm_650.csv"), index=False)

    with open(os.path.join(OUT_DIR, "resize_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")

    n_img = len(os.listdir(os.path.join(OUT_DIR, "images")))
    n_msk = len(os.listdir(os.path.join(OUT_DIR, "masks")))
    say(f"\nWritten: {n_img} images, {n_msk} masks -> {OUT_DIR}")


if __name__ == "__main__":
    main()
