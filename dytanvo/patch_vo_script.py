# """
# patch_vo_script.py
# ──────────────────
# Патч для vo_trajectory_from_folder.py — добавляет поддержку
# стандартного KITTI Odometry (--kitti-standard) и MPI-Sintel (--sintel).

# Запускать ОДИН РАЗ внутри контейнера после монтирования кода:
#   docker exec dytanvo python patch_vo_script.py

# Что делает:
#   1. Читает vo_trajectory_from_folder.py
#   2. Добавляет два новых аргумента в argparse
#   3. Добавляет две ветки загрузки датасетов
#   4. Сохраняет оригинал как vo_trajectory_from_folder.py.bak
# """

import shutil
from pathlib import Path

VO_SCRIPT = Path("vo_trajectory_from_folder.py")

if not VO_SCRIPT.exists():
    raise FileNotFoundError(f"{VO_SCRIPT} не найден. Запускайте из /workspace/DytanVO")

# ── Бэкап 
bak = Path("vo_trajectory_from_folder.py.bak")
if not bak.exists():
    shutil.copy(VO_SCRIPT, bak)
    print(f"Бэкап: {bak}")
else:
    print("Бэкап уже существует, пропускаем.")

src = VO_SCRIPT.read_text()

# ── Патч 1: аргументы 
NEW_ARGS = """
    # ── Добавлено patch_vo_script.py ──
    parser.add_argument('--kitti-standard', action='store_true',
        help='Стандартный KITTI Odometry (poses/XX.txt, 3x4 матрицы)')
    parser.add_argument('--sintel', action='store_true',
        help='MPI-Sintel dataset (camdata_left/scene/camdata.txt)')
    # ─────────────────────────────────
"""

# Вставляем после строки с --sceneflow (или --airdos)
MARKER = "parser.add_argument('--airdos'"
if MARKER not in src:
    MARKER = "parser.add_argument('--sceneflow'"

if MARKER not in src:
    print("WARN: маркер для аргументов не найден, пропускаем патч аргументов.")
else:
    # Найти конец строки после маркера
    idx = src.index(MARKER)
    end = src.index("\n", idx) + 1
    src = src[:end] + NEW_ARGS + src[end:]
    print("Патч аргументов применён.")

# ── Патч 2: загрузка датасетов
NEW_DATASET_CODE = """
    # ── Добавлено patch_vo_script.py ──
    if args.kitti_standard:
        from Datasets.kitti_standard import (
            KITTIStandardVODataset, load_kitti_standard_poses,
            load_kitti_standard_intrinsics)
        dataset = KITTIStandardVODataset(args.test_dir, transform=transform)
        pose_std = load_kitti_standard_poses(args.pose_file)
        intrinsics = load_kitti_standard_intrinsics(args.kitti_intrinsics_file)
    elif args.sintel:
        from Datasets.sintel_vo import (
            SintelVODataset, load_sintel_poses, SINTEL_K)
        dataset = SintelVODataset(args.test_dir, transform=transform)
        pose_std = load_sintel_poses(args.pose_file)
        intrinsics = SINTEL_K
    # ─────────────────────────────────
"""

# Вставляем перед elif args.airdos  (или elif args.sceneflow)
MARKER2 = "elif args.airdos:"
if MARKER2 not in src:
    MARKER2 = "elif args.sceneflow:"

if MARKER2 not in src:
    print("WARN: маркер для загрузки датасетов не найден.")
else:
    idx2 = src.index(MARKER2)
    src = src[:idx2] + NEW_DATASET_CODE + src[idx2:]
    print("Патч загрузки датасетов применён.")

VO_SCRIPT.write_text(src)
print("Готово. vo_trajectory_from_folder.py обновлён.")