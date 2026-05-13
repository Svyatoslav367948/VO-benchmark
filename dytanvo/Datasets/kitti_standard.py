# """
# Datasets/kitti_standard.py
# ──────────────────────────
# Загрузчик стандартного KITTI Odometry для DytanVO.

# Разница с DynaKITTI:
#   - DynaKITTI:  pose_left.txt  (4×4 matrices, абсолютные)
#   - KITTI стандарт: poses/XX.txt  (3×4 matrices, абсолютные)
#   - Калибровка: calib.txt  →  P2 / K  формат KITTI Odometry

# Использование в vo_trajectory_from_folder.py:
#   python vo_trajectory_from_folder.py \
#     --kitti-standard \
#     --test-dir /workspace/data/kitti/sequences/09/image_2 \
#     --pose-file /workspace/data/kitti/poses/09.txt \
#     --kitti-intrinsics-file /workspace/data/kitti/sequences/09/calib.txt \
#     --vo-model-name models/vonet_ft.pkl \
#     --seg-model-name models/segnet-kitti.pth
# """

import numpy as np
import os
from torch.utils.data import Dataset
from os import listdir
from PIL import Image


def load_kitti_standard_poses(pose_file: str) -> np.ndarray:
    # """
    # Читает poses/XX.txt  — строки вида 12 чисел (3×4 матрица).
    # Возвращает (N, 4, 4) float32.
    # """
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
            poses.append(T)
    return np.stack(poses)


def load_kitti_standard_intrinsics(calib_file: str) -> np.ndarray:
    """
    Читает calib.txt KITTI Odometry.
    Возвращает K (3×3) из строки P2 (левая цветная камера).
    """
    with open(calib_file) as f:
        for line in f:
            if line.startswith("P2:") or line.startswith("P_rect_02:"):
                vals = list(map(float, line.split()[1:]))
                P = np.array(vals, dtype=np.float32).reshape(3, 4)
                return P[:3, :3]   # K
    raise ValueError(f"P2 не найдена в {calib_file}")


class KITTIStandardVODataset(Dataset):
    """
    Dataset для стандартного KITTI Odometry.
    Совместим с интерфейсом VoDataset из DytanVO.
    """

    def __init__(self, datadir: str, transform=None,
                 img_height: int = 448, img_width: int = 640):
        self.datadir = datadir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width

        # Сортировка по имени файла — KITTI именует кадры числами
        exts = {".png", ".jpg", ".jpeg"}
        self.imgs = sorted(
            [os.path.join(datadir, f) for f in listdir(datadir)
             if os.path.splitext(f)[1].lower() in exts]
        )

    def __len__(self):
        # N-1 пар для N кадров
        return max(0, len(self.imgs) - 1)

    def __getitem__(self, idx):
        img0 = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        img1 = np.array(Image.open(self.imgs[idx + 1]).convert("RGB"))
        sample = {"img1": img0, "img2": img1}
        if self.transform:
            sample = self.transform(sample)
        return sample