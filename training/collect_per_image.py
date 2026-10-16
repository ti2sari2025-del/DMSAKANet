"""
Merge the per-image test metrics of every model, dataset and fold into one CSV per dataset
(training/results/per_image_<dataset>.csv) for the repository.
Run from DATASET_LAMA:  python collect_per_image.py
"""
import glob
import os

import pandas as pd

SRC = {
    ('cddcesm', 'DMSA-KANet'): 'results_patientlevel_cddcesm/reeval_widegrid',
    ('cbisddsm', 'DMSA-KANet'): 'results_patientlevel_cbisddsm',
}
for ds in ('cddcesm', 'cbisddsm'):
    for d in sorted(glob.glob(f'results_patientlevel_{ds}/baseline_*_common')):
        SRC[(ds, os.path.basename(d).replace('baseline_', '').replace('_common', ''))] = d

for ds in ('cddcesm', 'cbisddsm'):
    parts = []
    for (dd, model), d in SRC.items():
        if dd != ds:
            continue
        files = sorted(glob.glob(os.path.join(d, 'per_image_metrics_fold_*.csv')))
        for f in files:
            t = pd.read_csv(f)
            t.insert(0, 'model', model)
            parts.append(t)
        print(f"{ds:9s} {model:12s} {len(files)} fold files")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(f'per_image_{ds}.csv', index=False)
    print(f"-> per_image_{ds}.csv: {len(out)} rows")
