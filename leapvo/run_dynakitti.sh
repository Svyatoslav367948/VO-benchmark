#!/bin/bash
# Запуск LeapVO на всех последовательностях DynaKITTI
# Внутри контейнера: bash /workspace/leapvo/run_dynakitti.sh

DATA=/workspace/data/DynaKITTI
SAVEDIR=/workspace/leapvo/results/dynakitti
CONFIG_PATH=/workspace/leapvo/configs

cd /workspace/leapvo

# Получаем все последовательности (папки вида 00_1, 00_2, 05_1 и т.д.)
for SEQ_DIR in $DATA/*/; do
    SEQ=$(basename $SEQ_DIR)
    IMG_DIR=$SEQ_DIR/image_2
    CALIB=$SEQ_DIR/calib.txt
    GT=$SEQ_DIR/pose_left.txt

    if [ ! -d "$IMG_DIR" ]; then
        echo "[skip] image_2 не найден: $IMG_DIR"
        continue
    fi
    if [ ! -f "$GT" ]; then
        echo "[skip] pose_left.txt не найден: $GT"
        continue
    fi
    if [ ! -f "$CALIB" ]; then
        echo "[skip] calib.txt не найден: $CALIB"
        continue
    fi

    echo "====== DynaKITTI seq $SEQ ======"
    python main/eval.py \
        --config-path=$CONFIG_PATH \
        --config-name=dynakitti \
        data.imagedir=$IMG_DIR \
        data.calib=$CALIB \
        data.gt_traj=$GT \
        data.name=dynakitti-$SEQ \
        data.savedir=$SAVEDIR \
        data.traj_format=kitti \
        exp_name=leapvo_dynakitti_$SEQ \
        save_video=false
done

echo ""
echo "====== Все последовательности обработаны ======"
echo "Результаты: $SAVEDIR"