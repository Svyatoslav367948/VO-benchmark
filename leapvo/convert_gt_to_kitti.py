import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# Пути внутри контейнера
DATA_DIR = "/workspace/data/DynaKITTI"
RESULTS_DIR = "/workspace/leapvo/results/dynakitti/gt_kitti"

# Создаём папку для результатов
os.makedirs(RESULTS_DIR, exist_ok=True)

for traj in os.listdir(DATA_DIR):
    pose_7_file = os.path.join(DATA_DIR, traj, "pose_left.txt")
    
    if not os.path.exists(pose_7_file):
        continue
    
    # Читаем 7 колонок
    poses_7 = np.loadtxt(pose_7_file)
    poses_12 = np.zeros((len(poses_7), 12))
    
    # Конвертируем в 12 колонок
    for i, p in enumerate(poses_7):
        t = p[:3]
        q = p[3:7]
        rot = R.from_quat(q).as_matrix()
        poses_12[i] = np.hstack((rot, t.reshape(3, 1))).flatten()
    
    # Сохраняем в results (не в datasets!)
    pose_12_file = os.path.join(RESULTS_DIR, f"{traj}_pose_kitti.txt")
    np.savetxt(pose_12_file, poses_12, fmt='%.12f')
    print(f"✅ {traj}: {len(poses_7)} поз -> {pose_12_file}")
