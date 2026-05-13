#!/bin/bash
# Запуск CoProU на DynaKITTI + Shibuya + Sintel
# Внутри контейнера: bash /workspace/CoProU/run_eval_all.sh

set -e
cd /workspace/CoProU

CKPT=/workspace/CoProU/checkpoints/exp_pose_checkpoint_kitti.pth.tar
H=256
W=832
RESULTS=/workspace/CoProU/results

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Чекпоинт не найден: $CKPT"
    exit 1
fi

mkdir -p $RESULTS

echo "============================================"
echo "  CoProU Evaluation"
echo "  Checkpoint: $(basename $CKPT)"
echo "  Results:    $RESULTS"
echo "============================================"

# ── DynaKITTI ──────────────────────────────────
echo ""
echo "=== DynaKITTI ==="
python test_vo_dynakitti.py \
    --pretrained-posenet $CKPT \
    --img-height $H --img-width $W \
    --dataset-dir /workspace/data/DynaKITTI \
    --output-dir $RESULTS/dynakitti \
    2>&1 | tee $RESULTS/dynakitti_run.log
echo "DynaKITTI done ✓"

# ── Shibuya ────────────────────────────────────
echo ""
echo "=== Shibuya ==="
python test_vo_shibuya.py \
    --pretrained-posenet $CKPT \
    --img-height $H --img-width $W \
    --dataset-dir /workspace/data/shibuya \
    --output-dir $RESULTS/shibuya \
    2>&1 | tee $RESULTS/shibuya_run.log
echo "Shibuya done ✓"

# ── Sintel final ───────────────────────────────
echo ""
echo "=== Sintel (final) ==="
python test_vo_sintel.py \
    --pretrained-posenet $CKPT \
    --img-height $H --img-width $W \
    --sintel-dir /workspace/data/MPI-Sintel-complete \
    --output-dir $RESULTS/sintel_final \
    --sintel-pass final \
    2>&1 | tee $RESULTS/sintel_final_run.log
echo "Sintel done ✓"

echo ""
echo "============================================"
echo "  Всё готово. Результаты: $RESULTS"
echo "============================================"