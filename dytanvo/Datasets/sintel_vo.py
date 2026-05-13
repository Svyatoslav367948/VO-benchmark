# """
# Datasets/sintel_vo.py
# 
# Загрузчик MPI-Sintel для DytanVO.

# Структура MPI-Sintel-complete/training/:
#   clean/<scene>/frame_XXXX.png   ← RGB кадры
#   final/<scene>/frame_XXXX.png
#   camdata_left/<scene>/camdata.txt  ← позы камеры (KITTI 3×4 формат)

# Использование:
#   python run_sintel.py  (скрипт ниже)
# """

import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


def load_sintel_poses(camdata_file: str) -> np.ndarray:
    # """
    # Читает camdata.txt MPI-Sintel.
    # Формат: каждые 3 строки по 4 числа = одна 3×4 матрица позы.
    # Возвращает (N, 4, 4) float32.
    # """
    with open(camdata_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    poses = []
    i = 0
    while i + 2 < len(lines):
        try:
            row0 = list(map(float, lines[i].split()))
            row1 = list(map(float, lines[i + 1].split()))
            row2 = list(map(float, lines[i + 2].split()))
            if len(row0) == 4 and len(row1) == 4 and len(row2) == 4:
                T = np.eye(4, dtype=np.float32)
                T[0] = row0
                T[1] = row1
                T[2] = row2
                poses.append(T)
                i += 3
            else:
                i += 1
        except ValueError:
            i += 1

    if not poses:
        raise ValueError(f"Не удалось разобрать позы из {camdata_file}")
    return np.stack(poses)


# Intrinsics MPI-Sintel (фиксированные для всех сцен):
#   fx = fy = 1120, cx = 511.5, cy = 217.5  (1024×436 изображения)
SINTEL_K = np.array([
    [1120.0,    0.0,  511.5],
    [   0.0, 1120.0,  217.5],
    [   0.0,    0.0,    1.0],
], dtype=np.float32)


class SintelVODataset(Dataset):
    """
    Dataset для одной сцены MPI-Sintel.
    """

    def __init__(self, scene_dir: str, transform=None):
        self.scene_dir = scene_dir
        self.transform = transform
        exts = {".png", ".jpg"}
        self.imgs = sorted(
            [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)
             if os.path.splitext(f)[1].lower() in exts]
        )

    def __len__(self):
        return max(0, len(self.imgs) - 1)

    def __getitem__(self, idx):
        img0 = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        img1 = np.array(Image.open(self.imgs[idx + 1]).convert("RGB"))
        sample = {"img1": img0, "img2": img1}
        if self.transform:
            sample = self.transform(sample)
        return sample