"""
fix_vo_script.py — исправляет синтаксическую ошибку в vo_trajectory_from_folder.py,
которую оставил patch_vo_script.py, и восстанавливает файл из бэкапа
с правильным патчем.
"""
from pathlib import Path
import shutil, re, sys

VO   = Path("vo_trajectory_from_folder.py")
BAK  = Path("vo_trajectory_from_folder.py.bak")

if not BAK.exists():
    sys.exit("[ERROR] .bak не найден — патч не был применён ранее")

# Восстанавливаем чистый оригинал
shutil.copy(BAK, VO)
print("[1] Восстановлен оригинал из .bak")

src = VO.read_text()

# ── найдём блок parser.add_argument для --kitti / --airdos / --sceneflow ──────
# Вставим наши аргументы ПОСЛЕ последнего известного dataset-флага.
# Ищем строку вида:  parser.add_argument('--airdos'  ИЛИ  '--sceneflow'
MARKERS = ["'--airdos'", "'--sceneflow'", "'--kitti'"]
insert_after_idx = -1
for marker in MARKERS:
    idx = src.rfind(marker)      # берём последнее вхождение
    if idx != -1:
        # найти конец этого вызова add_argument (до следующей непустой строки начинающейся не с ' ')
        eol = src.index("\n", idx)
        # skip continuation lines (lines ending with ,  or ) )
        pos = eol + 1
        while pos < len(src):
            line = src[pos : src.index("\n", pos) + 1]
            stripped = line.strip()
            if stripped == "" or stripped.startswith("parser") or stripped.startswith("#"):
                break
            pos = src.index("\n", pos) + 1
        insert_after_idx = pos
        break

if insert_after_idx == -1:
    sys.exit("[ERROR] Не найден маркер для вставки аргументов")

NEW_ARGS = (
    "    parser.add_argument('--kitti-standard', action='store_true',\n"
    "                        help='Standard KITTI Odometry (poses/XX.txt 3x4)')\n"
    "    parser.add_argument('--sintel', action='store_true',\n"
    "                        help='MPI-Sintel dataset')\n"
)
src = src[:insert_after_idx] + NEW_ARGS + src[insert_after_idx:]
print("[2] Аргументы --kitti-standard и --sintel добавлены")

# ── найдём место загрузки датасета — вставим ДО elif args.airdos ──────────────
DATASET_MARKERS = ["elif args.airdos:", "elif args.sceneflow:"]
ds_idx = -1
for dm in DATASET_MARKERS:
    idx = src.find(dm)
    if idx != -1:
        ds_idx = idx
        break

if ds_idx == -1:
    sys.exit("[ERROR] Не найден маркер для вставки загрузчика датасетов")

NEW_DATASET = (
    "    if args.kitti_standard:\n"
    "        from Datasets.kitti_standard import (\n"
    "            KITTIStandardVODataset, load_kitti_poses as _lkp,\n"
    "            load_kitti_standard_intrinsics)\n"
    "        datastr = 'kitti'\n"
    "        dataset = KITTIStandardVODataset(args.test_dir, transform=transform)\n"
    "        pose_std = _lkp(args.pose_file)\n"
    "        intrinsics = load_kitti_standard_intrinsics(args.kitti_intrinsics_file)\n"
    "        focalx, focaly = float(intrinsics[0,0]), float(intrinsics[1,1])\n"
    "        centerx, centery = float(intrinsics[0,2]), float(intrinsics[1,2])\n"
    "    elif args.sintel:\n"
    "        from Datasets.sintel_vo import SintelVODataset, load_sintel_poses, SINTEL_K\n"
    "        datastr = 'sintel'\n"
    "        dataset = SintelVODataset(args.test_dir, transform=transform)\n"
    "        pose_std = load_sintel_poses(args.pose_file)\n"
    "        focalx, focaly = float(SINTEL_K[0,0]), float(SINTEL_K[1,1])\n"
    "        centerx, centery = float(SINTEL_K[0,2]), float(SINTEL_K[1,2])\n"
    "    el"   # продолжение: elif args.airdos / sceneflow
)

# Заменяем первые два символа "el" из маркера, чтобы не дублировать "elif"
src = src[:ds_idx] + NEW_DATASET + src[ds_idx + 2:]  # убираем "el" из маркера
print("[3] Загрузчики kitti_standard и sintel вставлены")

VO.write_text(src)

# Финальная проверка синтаксиса
import py_compile, tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
tmp.write(src.encode()); tmp.close()
try:
    py_compile.compile(tmp.name, doraise=True)
    print("[4] Синтаксис OK")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Синтаксическая ошибка после патча:\n{e}")
    # Восстанавливаем оригинал чтобы не сломать файл
    shutil.copy(BAK, VO)
    print("      Оригинал восстановлен из .bak")
finally:
    os.unlink(tmp.name)

print("\nГотово. vo_trajectory_from_folder.py обновлён.")