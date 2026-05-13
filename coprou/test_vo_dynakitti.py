"""
test_vo_dynakitti.py — инференс CoProU на DynaKITTI.

Структура DynaKITTI:
  {dataset_dir}/00_1/image_2/*.png
  {dataset_dir}/00_1/pose_left.txt   ← tx ty tz qx qy qz qw
  {dataset_dir}/00_1/calib.txt       ← P0:/P2: формат

Запуск:
  python test_vo_dynakitti.py \
    --pretrained-posenet checkpoints/exp_pose_checkpoint_kitti.pth.tar \
    --img-height 256 --img-width 832 \
    --dataset-dir /workspace/data/DynaKITTI \
    --output-dir results/dynakitti
"""

import argparse
import json
import math
import numpy as np
import torch
from PIL import Image
from path import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_warp import pose_vec2mat
import models

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-posenet", type=str, default=None)
parser.add_argument("--pretrained-model",   type=str, default=None)
parser.add_argument("--img-height", type=int, default=256)
parser.add_argument("--img-width",  type=int, default=832)
parser.add_argument("--dataset-dir", type=str, required=True)
parser.add_argument("--output-dir",  type=str, required=True)
parser.add_argument("--sequences",   type=str, nargs="*", default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════
# Утилиты — используются также из test_vo_shibuya и test_vo_sintel
# ════════════════════════════════════════════════════════

def load_tensor_image(path, h, w):
    img = Image.open(str(path)).convert("RGB")
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    t = ((torch.from_numpy(img).unsqueeze(0) / 255.0) - 0.45) / 0.225
    return t.to(device)


def load_posenet(args):
    net = models.PoseResNet().to(device)
    if args.pretrained_posenet:
        w = torch.load(args.pretrained_posenet, map_location=device)
        net.load_state_dict(w["state_dict"], strict=False)
        print(f"PoseNet loaded: {args.pretrained_posenet}")
    elif args.pretrained_model:
        ckpt = torch.load(args.pretrained_model, map_location=device)
        pw = {k.replace("pose_net.", ""): v
              for k, v in ckpt["state_dict"].items()
              if k.startswith("pose_net.")}
        net.load_state_dict(pw, strict=False)
        print(f"PoseNet loaded from combined: {args.pretrained_model}")
    else:
        raise ValueError("Нужен --pretrained-posenet или --pretrained-model")
    net.eval()
    return net


def umeyama(src_xyz, dst_xyz):
    """7DoF Umeyama: возвращает (scale, R, t) такие что scale*R@src+t ≈ dst."""
    n = src_xyz.shape[0]
    mu_s = src_xyz.mean(0)
    mu_d = dst_xyz.mean(0)
    sc = src_xyz - mu_s
    dc = dst_xyz - mu_d
    var_s = (sc ** 2).sum() / n
    if var_s < 1e-10:
        return 1.0, np.eye(3), mu_d - mu_s
    cov = dc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1, 1, float(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    c = float((S @ D.diagonal()) / var_s)
    t = mu_d - c * R @ mu_s
    return c, R, t


def kitti_trel_rrel(gt_4x4_list, pred_4x4_list, scale=1.0):
    """KITTI t_rel (%) и r_rel (deg/100m) после scale correction."""
    n = min(len(gt_4x4_list), len(pred_4x4_list))
    gt_xyz = np.array([T[:3, 3] for T in gt_4x4_list[:n]])

    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.linalg.norm(gt_xyz[i] - gt_xyz[i-1])

    total = dist[-1]
    if total < 10:
        return float("nan"), float("nan")

    if total >= 100:
        lengths = [L for L in [100, 200, 300, 400, 500, 600, 700, 800]
                   if L <= total]
    else:
        lengths = [total * f for f in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]]

    t_errs, r_errs = [], []
    for s in range(n):
        for L in lengths:
            e = s
            while e < n and dist[e] - dist[s] < L:
                e += 1
            if e >= n:
                continue
            T_gt_rel   = np.linalg.inv(gt_4x4_list[s])   @ gt_4x4_list[e]
            T_pred_rel = np.linalg.inv(pred_4x4_list[s]) @ pred_4x4_list[e]
            T_pred_sc  = T_pred_rel.copy()
            T_pred_sc[:3, 3] *= scale
            T_err = np.linalg.inv(T_pred_sc) @ T_gt_rel
            actual = dist[e] - dist[s]
            if actual < 1.0:
                continue
            t_errs.append(np.linalg.norm(T_err[:3, 3]) / actual * 100.0)
            cos_a = np.clip((np.trace(T_err[:3, :3]) - 1) / 2, -1.0, 1.0)
            r_errs.append(math.degrees(abs(math.acos(cos_a))) / actual * 100.0)

    if not t_errs:
        return float("nan"), float("nan")
    return float(np.mean(t_errs)), float(np.mean(r_errs))


def save_trajectory_plot(gt_xyz, pred_xyz, title, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xyz[:, 0],   gt_xyz[:, 2],   "b-",  lw=2,   label="GT")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], "r--", lw=1.5, label="CoProU")
    ax.scatter([gt_xyz[0, 0]], [gt_xyz[0, 2]], c="green", s=60,
               zorder=5, label="Start")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out = Path(out_dir) / "trajectory.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  График: {out}")


def compute_and_save_metrics(pred_kitti12, gt_kitti12, seq_name, out_dir):
    """
    pred_kitti12, gt_kitti12 — массивы (N, 12), KITTI 3×4 формат.
    Считает ATE (7DoF Umeyama), RPE delta=1, KITTI t_rel/r_rel.
    Сохраняет metrics.json, metrics.txt, trajectory.png.
    """
    n = min(len(pred_kitti12), len(gt_kitti12))
    pred_kitti12 = pred_kitti12[:n]
    gt_kitti12   = gt_kitti12[:n]

    def to_4x4(row12):
        T = np.eye(4)
        T[:3, :] = row12.reshape(3, 4)
        return T

    pred_4x4 = [to_4x4(r) for r in pred_kitti12]
    gt_4x4   = [to_4x4(r) for r in gt_kitti12]

    pred_xyz = np.array([T[:3, 3] for T in pred_4x4])
    gt_xyz   = np.array([T[:3, 3] for T in gt_4x4])

    # 7DoF Umeyama alignment
    scale, R_um, t_um = umeyama(pred_xyz, gt_xyz)
    pred_aligned = (scale * R_um @ pred_xyz.T).T + t_um

    # ATE
    ate_err  = np.linalg.norm(gt_xyz - pred_aligned, axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_err ** 2)))
    ate_mean = float(np.mean(ate_err))

    # Применяем alignment к 4x4 позам для RPE
    pred_4x4_aligned = []
    for T in pred_4x4:
        T_al = T.copy()
        T_al[:3, 3] = scale * R_um @ T[:3, 3] + t_um
        T_al[:3, :3] = R_um @ T[:3, :3]
        pred_4x4_aligned.append(T_al)

    # RPE delta=1
    t_errs, r_errs = [], []
    for i in range(n - 1):
        dgt   = np.linalg.inv(gt_4x4[i])           @ gt_4x4[i + 1]
        dpred = np.linalg.inv(pred_4x4_aligned[i]) @ pred_4x4_aligned[i + 1]
        derr  = np.linalg.inv(dpred) @ dgt
        t_errs.append(np.linalg.norm(derr[:3, 3]))
        cos_a = np.clip((np.trace(derr[:3, :3]) - 1) / 2, -1, 1)
        r_errs.append(math.degrees(abs(math.acos(cos_a))))

    rpe_t = float(np.sqrt(np.mean(np.array(t_errs) ** 2)))
    rpe_r = float(np.mean(r_errs))

    # KITTI t_rel / r_rel
    t_rel, r_rel = kitti_trel_rrel(gt_4x4, pred_4x4, scale)

    metrics = {
        "seq":           seq_name,
        "n_frames":      n,
        "ate_rmse":      ate_rmse,
        "ate_mean":      ate_mean,
        "rpe_t_rmse":    rpe_t,
        "rpe_r_mean_deg": rpe_r,
        "t_rel_pct":     t_rel,
        "r_rel_deg100m": r_rel,
        "scale_factor":  scale,
    }

    out_dir = Path(out_dir)
    out_dir.makedirs_p()

    with open(str(out_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(str(out_dir / "metrics.txt"), "w") as f:
        f.write(f"Sequence : {seq_name}\n")
        f.write(f"Frames   : {n}\n")
        f.write(f"Scale    : {scale:.4f}\n\n")
        f.write(f"ATE RMSE : {ate_rmse:.5f} m\n")
        f.write(f"ATE mean : {ate_mean:.5f} m\n")
        f.write(f"RPE-t    : {rpe_t:.5f} m\n")
        f.write(f"RPE-r    : {rpe_r:.5f} deg\n")
        f.write(f"t_rel    : {t_rel:.4f} %\n")
        f.write(f"r_rel    : {r_rel:.4f} deg/100m\n")

    print(f"  ATE RMSE : {ate_rmse:.5f} m")
    print(f"  RPE-t    : {rpe_t:.5f} m")
    print(f"  RPE-r    : {rpe_r:.5f} deg")
    print(f"  t_rel    : {t_rel:.4f} %")
    print(f"  r_rel    : {r_rel:.4f} deg/100m")

    save_trajectory_plot(gt_xyz, pred_aligned, seq_name, out_dir)


def print_summary(output_dir):
    out_dir = Path(output_dir)
    jsons = sorted(out_dir.walkfiles("metrics.json"))
    if not jsons:
        print("Нет метрик.")
        return

    print(f"\n{'='*72}")
    print(f"{'Seq':<22} {'ATE[m]':>8} {'RPE-t[m]':>10} "
          f"{'RPE-r[°]':>10} {'t_rel%':>8} {'r°/100m':>10}")
    print("-" * 72)

    all_m = []
    for jf in jsons:
        try:
            m = json.loads(open(str(jf)).read())
            all_m.append(m)
            print(f"{m['seq']:<22} {m['ate_rmse']:>8.4f} "
                  f"{m['rpe_t_rmse']:>10.4f} {m['rpe_r_mean_deg']:>10.4f} "
                  f"{m.get('t_rel_pct', float('nan')):>8.2f} "
                  f"{m.get('r_rel_deg100m', float('nan')):>10.4f}")
        except Exception:
            pass

    if all_m:
        summary_path = out_dir / "summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_m, f, indent=2)
        print(f"\nСводка: {summary_path}")


# ════════════════════════════════════════════════════════
# DynaKITTI специфика
# ════════════════════════════════════════════════════════

def load_gt_dynakitti(pose_file):
    """pose_left.txt: tx ty tz qx qy qz qw → список 4x4."""
    raw = np.loadtxt(str(pose_file))
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    poses = []
    for row in raw:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3]  = row[:3]
        T[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
        poses.append(T)
    return poses


@torch.no_grad()
def run_sequence(seq_dir, pose_net, args, output_dir):
    seq_dir   = Path(seq_dir)
    img_dir   = seq_dir / "image_2"
    pose_file = seq_dir / "pose_left.txt"
    seq_name  = seq_dir.name

    images = sorted(img_dir.files("*.png") + img_dir.files("*.jpg"))
    if len(images) < 2:
        print(f"[skip] {seq_name}: меньше 2 кадров")
        return

    print(f"\n{'='*50}")
    print(f"  DynaKITTI: {seq_name}  ({len(images)} кадров)")

    global_pose = np.eye(4)
    pred_poses  = [global_pose[:3, :].flatten()]

    t1 = load_tensor_image(images[0], args.img_height, args.img_width)
    for i in tqdm(range(len(images) - 1), desc=seq_name):
        t2       = load_tensor_image(images[i + 1], args.img_height, args.img_width)
        pose     = pose_net(t1, t2)
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()  # (3,4)
        pose_mat = np.vstack([pose_mat, [0, 0, 0, 1]])
        global_pose = global_pose @ np.linalg.inv(pose_mat)
        pred_poses.append(global_pose[:3, :].flatten())
        t1 = t2

    pred_kitti = np.array(pred_poses)  # (N, 12)

    out_dir = Path(output_dir) / seq_name
    out_dir.makedirs_p()
    np.savetxt(str(out_dir / "pred_kitti.txt"), pred_kitti, fmt="%1.8e")

    if pose_file.exists():
        gt_4x4   = load_gt_dynakitti(pose_file)
        gt_kitti = np.array([T[:3, :].flatten() for T in gt_4x4])
        np.savetxt(str(out_dir / "gt_kitti.txt"), gt_kitti, fmt="%1.8e")
        compute_and_save_metrics(pred_kitti, gt_kitti, seq_name, out_dir)
    else:
        print(f"  [warn] pose_left.txt не найден — метрики пропущены")


def main():
    args = parser.parse_args()
    pose_net = load_posenet(args)

    dataset_dir = Path(args.dataset_dir)
    if args.sequences:
        seq_dirs = [dataset_dir / s for s in args.sequences]
    else:
        seq_dirs = sorted([d for d in dataset_dir.dirs()
                           if (d / "image_2").exists()])

    print(f"DynaKITTI: найдено {len(seq_dirs)} последовательностей")
    for sd in seq_dirs:
        run_sequence(sd, pose_net, args, args.output_dir)

    print_summary(args.output_dir)


if __name__ == "__main__":
    main()