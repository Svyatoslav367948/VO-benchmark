"""
Запусти это в контейнере leapvo:
python3 /tmp/debug_rpe.py
(или скопируй содержимое в python3 <<)
"""
import numpy as np
import glob
from main.utils import load_traj, make_traj
from main.utils_rpe_fix import umeyama_full, poses_to_se3
from evo.tools import file_interface
from evo.core import sync

# GT
gt_raw, gt_ts = load_traj(
    "/workspace/data/DynaKITTI/00_1/pose_left.txt",
    traj_format="kitti"
)
traj_ref = make_traj((gt_raw, gt_ts))

# Pred — берём последний
traj_files = sorted(glob.glob(
    "/workspace/leapvo/results/dynakitti/*/leapvo_traj.txt"))
print("Pred файл:", traj_files[-1])
traj_est = file_interface.read_tum_trajectory_file(traj_files[-1])

# Синхронизация
if traj_ref.timestamps.shape[0] == traj_est.timestamps.shape[0]:
    traj_ref.timestamps = traj_est.timestamps
traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est)

gt_xyz   = traj_ref_s.positions_xyz
pred_xyz = traj_est_s.positions_xyz

print(f"\nN кадров: GT={len(gt_xyz)}, pred={len(pred_xyz)}")

# Шаги между соседними кадрами
gt_steps   = np.linalg.norm(np.diff(gt_xyz,   axis=0), axis=1)
pred_steps = np.linalg.norm(np.diff(pred_xyz, axis=0), axis=1)

print(f"\nGT   шаг/кадр: mean={gt_steps.mean():.4f}  std={gt_steps.std():.4f}  "
      f"min={gt_steps.min():.4f}  max={gt_steps.max():.4f}")
print(f"Pred шаг/кадр: mean={pred_steps.mean():.4f}  std={pred_steps.std():.4f}  "
      f"min={pred_steps.min():.4f}  max={pred_steps.max():.4f}")

ratio = gt_steps.mean() / (pred_steps.mean() + 1e-10)
print(f"\nЛокальный ratio GT/pred = {ratio:.4f}")

# Первые 5 шагов GT и pred
print(f"\nПервые 5 шагов GT   (м): {gt_steps[:5]}")
print(f"Первые 5 шагов pred    : {pred_steps[:5]}")

# После Umeyama
R, t, c = umeyama_full(pred_xyz, gt_xyz)
print(f"\nUmeyama: scale={c:.4f}")

# Применяем полное выравнивание
pred_aligned = (c * R @ pred_xyz.T).T + t
pred_steps_al = np.linalg.norm(np.diff(pred_aligned, axis=0), axis=1)
print(f"Pred aligned шаг/кадр: mean={pred_steps_al.mean():.4f}  "
      f"std={pred_steps_al.std():.4f}")
print(f"Первые 5 шагов pred aligned: {pred_steps_al[:5]}")

# RPE вручную для первых 5 пар
gt_se3   = poses_to_se3(gt_xyz,       traj_ref_s.orientations_quat_wxyz)
pred_se3 = poses_to_se3(pred_aligned,
    traj_est_s.orientations_quat_wxyz)  # используем aligned xyz

for i in range(min(5, len(gt_se3)-1)):
    gt_rel   = np.linalg.inv(gt_se3[i])   @ gt_se3[i+1]
    pred_rel = np.linalg.inv(pred_se3[i]) @ pred_se3[i+1]
    err44    = np.linalg.inv(pred_rel)    @ gt_rel
    t_err    = np.linalg.norm(err44[:3, 3])
    print(f"  кадр {i}->{i+1}: gt_step={gt_steps[i]:.4f}  "
          f"pred_step={pred_steps_al[i]:.4f}  RPE-t={t_err:.4f}")
