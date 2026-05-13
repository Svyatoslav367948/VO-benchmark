#!/usr/bin/env python3
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def convert_kitti_poses_to_quat(input_file, output_file):
    """Читает KITTI poses (N×12), преобразует в (N×7): x y z qx qy qz qw"""
    poses = np.loadtxt(input_file)
    N = poses.shape[0]
    result = np.zeros((N, 7))
    
    for i in range(N):
        mat = poses[i].reshape(3, 4)
        result[i, 0:3] = mat[:, 3]
        r = R.from_matrix(mat[:, 0:3])
        quat = r.as_quat()
        result[i, 3:7] = quat
    
    np.savetxt(output_file, result, fmt='%.12f')
    print(f"✅ Конвертировано: {input_file} -> {output_file}")

if __name__ == "__main__":
    # Откуда читаем оригиналы (read-only)
    pose_dir_read = "/workspace/data/kitti/dataset/poses"
    # Куда пишем конвертированные (туда, где есть запись)
    pose_dir_write = "/workspace/DytanVO/results/gt_quat"
    os.makedirs(pose_dir_write, exist_ok=True)
    
    for seq in ["00", "05", "07", "09"]:
        input_file = f"{pose_dir_read}/{seq}.txt"
        output_file = f"{pose_dir_write}/{seq}_quat.txt"
        if os.path.exists(input_file):
            convert_kitti_poses_to_quat(input_file, output_file)
        else:
            print(f"⚠️ {input_file} не найден")
