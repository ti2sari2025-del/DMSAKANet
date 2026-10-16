"""
Nested selection for DMSA-KANet on CBIS-DDSM (revision 4).
For every outer fold, the learning-rate configuration with the highest best-validation Dice
(the same criterion used for checkpoint selection; validation patients come from the training
folds only) is selected, and the TEST result of that configuration is reported. The test fold
plays no part in the choice.
Writes nested_selection_cbisddsm.csv (one row per fold x configuration, with the choice marked)
and prints the nested mean +- SD.
Run from DATASET_LAMA:  python select_nested.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = 'results_patientlevel_cbisddsm'
RUNS = {('1e-3', '1e-5'): os.path.join(ROOT, 'checkpoints')}           # run of Table 7
for d in sorted(glob.glob(os.path.join(ROOT, 'nested', 'lr1_*_lr2_*'))):
    _, lr1, _, lr2 = os.path.basename(d).split('_')
    RUNS[(lr1, lr2)] = os.path.join(d, 'checkpoints')

KEYS = ['dice', 'iou', 'precision', 'recall', 'roc_auc', 'fpr', 'fnr', 'dice_per_image_mean', 'threshold']
rows = []
for (lr1, lr2), d in RUNS.items():
    for f in sorted(glob.glob(os.path.join(d, 'fold_*_result.json'))):
        k = int(os.path.basename(f).split('_')[1])
        r = json.load(open(f))
        h = json.load(open(os.path.join(d, f'fold_{k}_history.json')))
        v = h['val_dice_coefficient']
        rows.append({'fold': k, 'lr1': lr1, 'lr2': lr2, 'best_val_dice': max(v),
                     'best_epoch': int(np.argmax(v)) + 1, **{x: r[x] for x in KEYS}})
df = pd.DataFrame(rows).sort_values(['fold', 'lr1', 'lr2'])
n_cfg = df.groupby('fold').size()
complete = n_cfg[n_cfg == len(RUNS)].index
df['selected'] = False
for k in complete:
    g = df[df.fold == k]
    df.loc[g.best_val_dice.idxmax(), 'selected'] = True
df.to_csv('nested_selection_cbisddsm.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f"\nConfigurations: {len(RUNS)}; folds with all configurations finished: {list(complete)}")
sel = df[df.selected]
if len(sel):
    print("\nNested result (configuration chosen per fold on validation patients only), % mean +- SD:")
    for x in KEYS[:-1]:
        print(f"  {x:<22} {100*sel[x].mean():6.2f} +- {100*sel[x].std(ddof=1) if len(sel) > 1 else 0:5.2f}")
    print("  chosen per fold:", ', '.join(f"F{r.fold}: LR1={r.lr1}, LR2={r.lr2}" for r in sel.itertuples()))
