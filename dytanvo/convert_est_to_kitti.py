#!/usr/bin/env python3
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def convert_quat_to_kitti(input_file, output_file):
    """Конвертирует est_poses.txt (7 колонок) в формат KITTI (12 колонок)"""
    poses = np.loadtxt(input_file)
    N = poses.shape[0]
    result = np.zeros((N, 12))
    
    for i in range(N):
        # Извлекаем перенос и кватернион
        t = poses[i, 0:3]
        q = poses[i, 3:7]  # [x, y, z, w]
        
        # Кватернион → матрица поворота 3×3
        r = R.from_quat(q)
        rot = r.as_matrix()
        
        # Собираем 3×4 матрицу
        mat_34 = np.hstack((rot, t.reshape(3, 1)))
        result[i, :] = mat_34.flatten()
    
    np.savetxt(output_file, result, fmt='%.12f')
    print(f"✅ Конвертировано: {input_file} -> {output_file}")
    print(f"   Формат: {result.shape[1]} колонок (KITTI style)")

if __name__ == "__main__":
    # Конвертируем все est_poses.txt для KITTI
    kitti_seqs = ["00", "05", "07", "09"]
    
    for seq in kitti_seqs:
        input_file = f"/workspace/DytanVO/results/kitti/seq_{seq}/est_poses.txt"
        output_file = f"/workspace/DytanVO/results/kitti/seq_{seq}/est_poses_kitti.txt"
        
        if os.path.exists(input_file):
            # Проверяем, сколько колонок
            sample = np.loadtxt(input_file)
            if sample.shape[1] == 7:
                convert_quat_to_kitti(input_file, output_file)
            else:
                print(f"⚠️ {input_file} уже в KITTI формате ({sample.shape[1]} колонок)")
        else:
            print(f"⚠️ {input_file} не найден")
    
    print("\n✅ Все конвертации завершены!")
