import os
from copy import deepcopy
from pathlib import Path

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, plot
from scipy.spatial.transform import Rotation
from main.utils_rpe_fix import compute_rpe_scaled, kitti_rpe_scaled


def sintel_cam_read(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    TAG_FLOAT = 202021.25

    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N


def load_replica_traj(gt_file):
    traj_w_c = np.loadtxt(gt_file)
    assert traj_w_c.shape[1] == 12 or traj_w_c.shape[1] == 16
    poses = [
        np.array(
            [
                [r[0], r[1], r[2], r[3]],
                [r[4], r[5], r[6], r[7]],
                [r[8], r[9], r[10], r[11]],
                [0, 0, 0, 1],
            ]
        )
        for r in traj_w_c
    ]

    pose_path = PosePath3D(poses_se3=poses)
    timestamps_mat = np.arange(traj_w_c.shape[0]).astype(float)

    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    # quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    quat = traj.orientations_quat_wxyz

    traj_tum = np.column_stack((xyz, quat))
    return (traj_tum, timestamps_mat)


def load_sintel_traj(gt_file):
    # Refer to ParticleSfM
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [os.path.join(gt_file, x) for x in gt_pose_lists]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_cam_read(f)[1] for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, wxyz], 0)
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(
        tum_gt_poses[:, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt

def load_kitti_traj(gt_file):
    """
    Читает pose_left.txt DynaKITTI.
    Формат: N строк, 7 чисел каждая: tx ty tz qx qy qz qw (scalar-last).
    Возвращает:
      traj_tum    (N, 7): tx ty tz qx qy qz qw  — как TUM но без timestamp
      timestamps  (N,)  : просто индексы 0,1,2,...
    """
    raw = np.loadtxt(gt_file)          # (N, 7)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]       # на случай одной строки

    # pose_left.txt уже в формате tx ty tz qx qy qz qw — ничего конвертировать не нужно
    # но LeapVO ожидает wxyz (scalar-first) в колонках [3:7]
    # scipy as_quat() = xyzw, TUM хранит xyzw, evo ожидает wxyz
    # Здесь данные: qx qy qz qw (cols 3,4,5,6) — нужно переставить в wxyz для evo
    xyz  = raw[:, :3]                          # tx ty tz
    xyzw = raw[:, 3:7]                         # qx qy qz qw (scalar-last)
    wxyz = np.roll(xyzw, 1, axis=1)            # -> qw qx qy qz (scalar-first для evo)

    traj_tum     = np.column_stack([xyz, wxyz])  # (N, 7): tx ty tz qw qx qy qz
    timestamps   = np.arange(len(traj_tum)).astype(float)
    return traj_tum, timestamps

def load_traj(gt_traj_file, traj_format="replica", skip=0, stride=1):
    """Read trajectory format. Return in TUM-RGBD format.
    Returns:
        traj_tum (N, 7): camera to world poses in (x,y,z,qx,qy,qz,qw)
        timestamps_mat (N, 1): timestamps
    """
    if traj_format == "replica":
        traj_tum, timestamps_mat = load_replica_traj(gt_traj_file)
    elif traj_format == "sintel":
        traj_tum, timestamps_mat = load_sintel_traj(gt_traj_file)
    elif traj_format == 'tartanair':
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        xyz = xyz[:,[1,2,0]]
        quat = traj.orientations_quat_wxyz
        quat = quat[:,[0,2,3,1]]
        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))
    elif traj_format == "tum":
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        # shift -1 column -> w in back column
        # quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
        quat = traj.orientations_quat_wxyz
        
        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))

    elif traj_format == "kitti":
        traj_tum, timestamps_mat = load_kitti_traj(gt_traj_file)

    else:
        raise NotImplementedError

    traj_tum = traj_tum[skip::stride]
    timestamps_mat = timestamps_mat[skip::stride]
    return traj_tum, timestamps_mat


def update_timestamps(gt_file, traj_format, skip=0, stride=1):
    """Update timestamps given a"""
    if traj_format == "tum":
        traj_t_map_file = gt_file.replace("groundtruth.txt", "rgb.txt")
        timestamps = load_timestamps(traj_t_map_file, traj_format)
        return timestamps[skip::stride]
    elif traj_format == "tartanair":
        traj_t_map_file = gt_file.replace("gt_pose.txt", "times.txt")
        timestamps = load_timestamps(traj_t_map_file, traj_format)
        return timestamps[skip::stride]


def load_timestamps(time_file, traj_format="replica"):
    if traj_format in ["tum", "tartanair"]:
        with open(time_file, "r+") as f:
            lines = f.readlines()
        timestamps_mat = [
            float(x.split(" ")[0]) for x in lines if not x.startswith("#")
        ]
        return timestamps_mat


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def _kitti_rpe(traj_ref, traj_est_aligned):
    """
    KITTI-style t_rel (%) и r_rel (deg/100m) на уже выровненных траекториях.
    """
    import math

    xyz_ref   = traj_ref.positions_xyz
    poses_ref = traj_ref.poses_se3
    poses_est = traj_est_aligned.poses_se3

    n = min(len(poses_ref), len(poses_est))
    if n < 2:
        return float("nan"), float("nan")

    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.linalg.norm(xyz_ref[i] - xyz_ref[i-1])

    total_dist = dist[-1]
    if total_dist < 100:
        # Последовательность короче 100м — считаем по всем доступным длинам
        lengths = [max(10, total_dist * f) for f in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]]
    else:
        lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    t_errs, r_errs = [], []

    for s in range(n):
        for L in lengths:
            e = s
            while e < n and dist[e] - dist[s] < L:
                e += 1
            if e >= n:
                continue

            T_ref_s = np.array(poses_ref[s])
            T_ref_e = np.array(poses_ref[e])
            T_est_s = np.array(poses_est[s])
            T_est_e = np.array(poses_est[e])

            T_ref_rel = np.linalg.inv(T_ref_s) @ T_ref_e
            T_est_rel = np.linalg.inv(T_est_s) @ T_est_e
            T_err     = np.linalg.inv(T_est_rel) @ T_ref_rel

            actual_len = dist[e] - dist[s]
            if actual_len < 1.0:
                continue

            t_err = np.linalg.norm(T_err[:3, 3])
            cos_a = np.clip((np.trace(T_err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            r_err = math.degrees(abs(math.acos(cos_a)))

            t_errs.append(t_err / actual_len * 100.0)
            r_errs.append(r_err / actual_len * 100.0)

    if not t_errs:
        return float("nan"), float("nan")

    return float(np.mean(t_errs)), float(np.mean(r_errs))

def eval_metrics(pred_traj, gt_traj=None, seq="", filename=""):
    import json, math
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(f"WARNING: pred={pred_traj.timestamps.shape[0]} gt={gt_traj.timestamps.shape[0]}")
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    traj_ref = gt_traj
    traj_est = pred_traj

    # ── ATE: Sim(3) Umeyama alignment ────────────────────────────────────
    ate_result = main_ape.ape(
        traj_ref, traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=True,
    )
    ate = ate_result.stats["rmse"]

    # ── RPE: правильный расчёт с scale correction ─────────────────────────
    # Идентично DyTanVO: scale применяется к translation каждой 4x4 матрицы
    rpe_t_rmse, rpe_r_mean, rpe_t_mean, scale = compute_rpe_scaled(
        traj_ref, traj_est, param_delta=1
    )

    # ── KITTI t_rel / r_rel ───────────────────────────────────────────────
    t_rel_pct, r_rel_degm = kitti_rpe_scaled(traj_ref, traj_est, scale)

    # ── Сохраняем отчёт ───────────────────────────────────────────────────
    with open(filename, "w+") as f:
        f.write(f"Seq: {seq}\n\n")
        f.write(f"=== ATE (Sim3 Umeyama, correct_scale=True) ===\n")
        f.write(f"{ate_result}\n")
        f.write(f"=== RPE translation (m, delta=1, scale-corrected) ===\n")
        f.write(f"  RMSE : {rpe_t_rmse:.6f} m\n")
        f.write(f"  mean : {rpe_t_mean:.6f} m\n")
        f.write(f"=== RPE rotation (deg, delta=1, scale-corrected) ===\n")
        f.write(f"  mean : {rpe_r_mean:.6f} deg\n")
        f.write(f"=== KITTI-style metrics ===\n")
        f.write(f"t_rel: {t_rel_pct:.4f} %\n")
        f.write(f"r_rel: {r_rel_degm:.4f} deg/100m\n")
        f.write(f"scale: {scale:.4f}\n")

    print(f"Save results to {filename}")
    print(f"  ATE RMSE  : {ate:.6f} m")
    print(f"  RPE-t RMSE: {rpe_t_rmse:.6f} m  (scale={scale:.2f}x)")
    print(f"  RPE-r mean: {rpe_r_mean:.6f} deg")
    print(f"  t_rel     : {t_rel_pct:.4f} %")
    print(f"  r_rel     : {r_rel_degm:.4f} deg/100m")

    return ate, rpe_t_rmse, rpe_r_mean


def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)


def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plot_collection.export(filename, confirm_overwrite=False)
        plt.close(fig=fig)
    print(f"Saved trajectory to {filename}")


def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: " ".join(map(str, a))
    with Path(filename).open("w") as f:
        for i in range(traj.num_poses):
            f.write(
                f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n"
            )
    print(f"Saved trajectory to {filename}")
