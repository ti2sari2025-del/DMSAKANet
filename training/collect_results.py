"""
Collect every finished fold of every model and print mean ± SD (in %).
Writes all_fold_results.csv (one row per model × fold) and all_summary.csv.
Run from DATASET_LAMA:  python collect_results.py
"""
import glob
import json
import os

import pandas as pd

SOURCES = {
    ('CDD-CESM', 'DMSA-KANet'): 'results_patientlevel_cddcesm/reeval_widegrid/fold_*_result.json',
    ('CBIS-DDSM', 'DMSA-KANet'): 'results_patientlevel_cbisddsm/checkpoints/fold_*_result.json',
}
for ds_dir, ds in [('cddcesm', 'CDD-CESM'), ('cbisddsm', 'CBIS-DDSM')]:
    for d in sorted(glob.glob(f'results_patientlevel_{ds_dir}/baseline_*_common')):
        name = os.path.basename(d).replace('baseline_', '').replace('_common', '')
        SOURCES[(ds, name)] = os.path.join(d, 'fold_*_result.json')

KEYS = ['dice', 'iou', 'precision', 'recall', 'f1_score', 'roc_auc', 'fpr', 'fnr',
        'dice_per_image_mean', 'iou_per_image_mean', 'threshold']
rows = []
for (ds, model), pattern in SOURCES.items():
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        row = {'dataset': ds, 'model': model, 'fold': int(r.get('fold', os.path.basename(f).split('_')[1]))}
        row.update({k: float(r[k]) for k in KEYS if k in r})
        rows.append(row)

df = pd.DataFrame(rows).sort_values(['dataset', 'model', 'fold'])
df.to_csv('all_fold_results.csv', index=False)

pct = [k for k in KEYS if k != 'threshold' and k in df.columns]
out = []
for (ds, model), g in df.groupby(['dataset', 'model']):
    rec = {'dataset': ds, 'model': model, 'folds': len(g)}
    for k in pct:
        rec[k] = f"{100 * g[k].mean():.2f} ± {100 * g[k].std(ddof=1):.2f}" if len(g) > 1 else f"{100 * g[k].mean():.2f}"
    out.append(rec)
summ = pd.DataFrame(out)
summ.to_csv('all_summary.csv', index=False)
pd.set_option('display.width', 250)
print(summ.to_string(index=False))
print("\nPer-fold Dice (%):")
print((df.pivot_table(index=['dataset', 'fold'], columns='model', values='dice') * 100).round(2).to_string())
