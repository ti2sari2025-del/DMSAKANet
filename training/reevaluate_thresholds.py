"""
Re-evaluate saved best-validation checkpoints with the wide threshold grid (0.01 ... 0.90)
=========================================================================================
No retraining. For every completed fold it:
  1. loads the best-validation checkpoint that training saved,
  2. selects the threshold on the VALIDATION set (flip TTA, pooled IoU) using the wide grid,
  3. evaluates the TEST fold once with that threshold,
and writes the results to <OUT_DIR>/reeval_widegrid/.

Use it for folds that were evaluated with the older, narrower grid (0.05 ... 0.575), so that
every fold of every model uses exactly the same threshold rule.

Usage (from DATASET_LAMA, after the training run of that model has finished):
    python reevaluate_thresholds.py dmsa
    python reevaluate_thresholds.py unet          # or attunet, unetpp, r2unet, transunet, vit
"""

import ast
import json
import os
import sys

import numpy as np
import pandas as pd

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'dmsa'
TRAIN_SCRIPT = 'train_dmsa_kanet_patientlevel.py' if MODEL == 'dmsa' else 'train_baseline_patientlevel.py'

# ------------------------------------------------------------------ load definitions only
src = open(TRAIN_SCRIPT).read()
if MODEL != 'dmsa':
    src = src.replace("MODEL_NAME = 'unet'", f"MODEL_NAME = '{MODEL}'", 1)

tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        keep.append(node)
    elif isinstance(node, ast.Assign):
        names = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, ast.Tuple):
                names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        if names and all(n == n.upper() for n in names):   # configuration constants only
            keep.append(node)
ns = {'__name__': 'reeval'}
sys_argv_backup = sys.argv
sys.argv = [TRAIN_SCRIPT]
exec(compile(ast.Module(body=keep, type_ignores=[]), TRAIN_SCRIPT, 'exec'), ns)
sys.argv = sys_argv_backup

tf = ns['tf']
cv2 = ns['cv2']
SIZE = ns['SIZE']
DATA_DIR, FOLD_CSV, OUT_DIR = ns['DATA_DIR'], ns['FOLD_CSV'], ns['OUT_DIR']
tta_predict = ns['tta_predict']
calculate_metrics = ns['calculate_metrics']
per_image_metrics = ns['per_image_metrics']
N_SPLITS = ns.get('N_SPLITS', 5)

GRID = np.round(np.arange(0.01, 0.91, 0.01), 2)


def select_threshold(y_true, y_prob):
    t = (y_true > 0.5).ravel()
    p = y_prob.ravel()
    best_t, best_iou = float(GRID[0]), -1.0
    for thr in GRID:
        b = p > thr
        tp = np.logical_and(t, b).sum(); fp = np.logical_and(~t, b).sum(); fn = np.logical_and(t, ~b).sum()
        iou = tp / (tp + fp + fn + 1e-7)
        if iou > best_iou:
            best_iou, best_t = iou, float(thr)
    return best_t, float(best_iou)


def load_model(fold_num):
    if MODEL == 'dmsa':
        path = f"{ns['CKPT_PREFIX']}_fold_{fold_num}_best_dice.keras"
        return tf.keras.models.load_model(path, compile=False), path
    path = os.path.join(OUT_DIR, f'fold_{fold_num}_best.weights.h5')
    m = ns['BUILDER']()
    m.load_weights(path)
    return m, path


# ------------------------------------------------------------------ data
folds_df = pd.read_csv(FOLD_CSV)
names = folds_df['Image_name'].astype(str).tolist()
imgs, msks = [], []
for n in names:
    imgs.append(cv2.imread(os.path.join(DATA_DIR, 'images', f'{n}.png'), cv2.IMREAD_COLOR))
    msks.append(cv2.imread(os.path.join(DATA_DIR, 'masks', f'{n}.png'), cv2.IMREAD_GRAYSCALE))
X = np.array(imgs, dtype=np.float32) / 255.0
Y = (np.expand_dims(np.array(msks, dtype=np.float32), 3) / 255.0 > 0.5).astype(np.float32)

out = os.path.join(OUT_DIR, 'reeval_widegrid')
os.makedirs(out, exist_ok=True)
rows = []
for fold_num in range(1, N_SPLITS + 1):
    k = fold_num - 1
    role = folds_df[f'role_fold{k}'].values
    va = np.where(role == 'val')[0]; te = np.where(role == 'test')[0]
    try:
        model, path = load_model(fold_num)
    except Exception as e:
        print(f"fold {fold_num}: no checkpoint ({e.__class__.__name__}) — skipped")
        continue
    thr, val_iou = select_threshold(Y[va], tta_predict(model, X[va]))
    y_pred = tta_predict(model, X[te])
    m = {k_: float(v) for k_, v in calculate_metrics(Y[te], y_pred, threshold=thr).items()}
    pim = per_image_metrics(Y[te], y_pred, thr, [names[i] for i in te])
    pim.insert(0, 'fold', fold_num)
    pim = pim.merge(folds_df[['Image_name', 'Patient_ID', 'Label']], on='Image_name', how='left')
    pim.to_csv(os.path.join(out, f'per_image_metrics_fold_{fold_num}.csv'), index=False)
    m.update({'fold': fold_num, 'val_iou_at_threshold': val_iou, 'checkpoint': path,
              'dice_per_image_mean': float(pim['dice'].mean()),
              'iou_per_image_mean': float(pim['iou'].mean())})
    json.dump(m, open(os.path.join(out, f'fold_{fold_num}_result.json'), 'w'), indent=2)
    rows.append(m)
    print(f"fold {fold_num}: thr={thr:.2f} (val IoU {val_iou:.4f}) | test Dice {m['dice']:.4f} "
          f"IoU {m['iou']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} | per-image Dice {m['dice_per_image_mean']:.4f}")
    tf.keras.backend.clear_session()

if rows:
    df = pd.DataFrame(rows).sort_values('fold')
    df.to_csv(os.path.join(out, 'fold_results.csv'), index=False)
    cols = ['dice', 'iou', 'precision', 'recall', 'f1_score', 'accuracy', 'roc_auc', 'fpr', 'fnr',
            'dice_per_image_mean', 'iou_per_image_mean', 'threshold']
    summ = pd.DataFrame({'mean': df[cols].mean(), 'std': df[cols].std()})
    summ.to_csv(os.path.join(out, 'summary.csv'))
    print(f"\n{MODEL}: mean ± SD over {len(df)} folds (wide threshold grid)")
    print(summ.round(4).to_string())
