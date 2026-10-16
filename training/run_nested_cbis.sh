#!/bin/bash
# Nested learning-rate selection for DMSA-KANet on CBIS-DDSM (revision 4).
# Grid (fixed before any run): LR1 in {1e-3, 1e-4} x LR2 in {5e-6, 1e-5}; batch size 8.
# It is the 2x2 grid spanned by the three best trials of the earlier search (Trials 8, 6, 5; all batch size 8).
# The configuration LR1=1e-3, LR2=1e-5 is the finished run of Table 7 and is not repeated.
# Every run uses the same patient-level folds, inner validation patients, seeds and epochs.
# Fully resumable: finished folds are skipped and an interrupted fold resumes from its last epoch.
# After every machine (re)start, run from DATASET_LAMA:
#   nohup bash run_nested_cbis.sh > nested_log.txt 2>&1 &
cd "$(dirname "$0")"
while pgrep -f "train_baseline_patientlevel.py|train_dmsa_kanet_patientlevel.py" > /dev/null; do
    echo "$(date) : another training run is active, waiting ..."
    sleep 300
done
for CFG in "1e-4 1e-5" "1e-3 5e-6" "1e-4 5e-6"; do
    set -- $CFG
    TAG="nested/lr1_$1_lr2_$2"
    echo "===== $(date) : CBIS-DDSM DMSA-KANet LR1=$1 LR2=$2 ====="
    DATASET=cbisddsm LR1=$1 LR2=$2 RUN_TAG=$TAG python train_dmsa_kanet_patientlevel.py \
        >> "log_nested_lr1_$1_lr2_$2.txt" 2>&1
done
python select_nested.py
echo "===== $(date) : NESTED FINISHED ====="
