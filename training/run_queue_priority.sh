#!/bin/bash
# Priority order for the remaining runs (fully resumable: finished folds are skipped).
#   1) DMSA-KANet on CBIS-DDSM            -> decides how the paper is framed
#   2) U-Net and Attention U-Net on CBIS-DDSM (strongest baselines on CDD-CESM)
#   3) remaining baselines on CDD-CESM
#   4) remaining baselines on CBIS-DDSM
# After every machine (re)start, run from DATASET_LAMA:
#   nohup bash run_queue_priority.sh > queue_priority_log.txt 2>&1 &
# If another training run is still using the GPU, this script waits for it to finish first.

cd "$(dirname "$0")"
while pgrep -f "train_baseline_patientlevel.py|train_dmsa_kanet_patientlevel.py" > /dev/null; do
    echo "$(date) : another training run is active, waiting ..."
    sleep 300
done
pip install -q keras-unet-collection==0.1.13

echo "===== $(date) : CBIS-DDSM DMSA-KANet ====="
DATASET=cbisddsm python train_dmsa_kanet_patientlevel.py >> log_dmsa_kanet_cbisddsm.txt 2>&1

for MODEL in unet attunet; do
    echo "===== $(date) : CBIS-DDSM $MODEL ====="
    DATASET=cbisddsm python train_baseline_patientlevel.py "$MODEL" common >> "log_cbis_baseline_${MODEL}_common.txt" 2>&1
done

for MODEL in unetpp r2unet transunet vit; do
    echo "===== $(date) : CDD-CESM $MODEL ====="
    DATASET=cddcesm python train_baseline_patientlevel.py "$MODEL" common >> "log_baseline_${MODEL}_common.txt" 2>&1
done

for MODEL in unetpp r2unet transunet vit; do
    echo "===== $(date) : CBIS-DDSM $MODEL ====="
    DATASET=cbisddsm python train_baseline_patientlevel.py "$MODEL" common >> "log_cbis_baseline_${MODEL}_common.txt" 2>&1
done
echo "===== $(date) : ALL FINISHED ====="
