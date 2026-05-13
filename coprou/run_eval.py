#!/usr/bin/env python3
# """
# run_eval.py — запуск CoProU на KITTI / Sintel / Shibuya + метрики

# CoProU нативно поддерживает только KITTI.
# Этот скрипт добавляет Sintel и Shibuya.

# Запуск внутри контейнера coprou:

#   # KITTI последовательность 09 (нативный test_vo.py)
#   python run_eval.py --dataset kitti --seq 09

#   # Все KITTI
#   python run_eval.py --dataset kitti --seq all

#   # MPI-Sintel clean
#   python run_eval.py --dataset sintel --pass clean

#   # Shibuya
#   python run_eval.py --dataset shibuya

#   # Сводка
#   python run_eval.py --summarize

# ВАЖНО: перед запуском установите зависимости:
#   pip install -r requirements.txt
#   # и скачайте чекпоинты согласно README CoProU
# """

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA    = Path("/workspace/data")
RESULTS = Path("/workspace/CoProU/results")
CKPT    = Path("/workspace/CoProU/checkpoints")

KITTI_SEQ  = DATA / "kitti" / "sequences"
KITTI_POSE = DATA / "kitti" / "poses"
SINTEL_DIR = DATA / "MPI-Sintel-complete" / "training"
SHIBUYA_DIR = DATA / "shibuya"

# Загрузка поз

def load_kitti_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.strip().split()))
            if len(v) == 12:
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = np.array(v).reshape(3, 4)
                poses.append(T)
    return np.stack(poses)


def load_tartanair_poses(path):
    from scipy.spatial.transform import Rotation
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.strip().split()))
            if len(v) >= 7:
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
                T[:3,  3] = v[:3]
                poses.append(T)
    return np.stack(poses)


def load_sintel_poses(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    poses = []
    i = 0
    while i + 2 < len(lines):
        try:
            rows = [list(map(float, lines[i+j].split())) for j in range(3)]
            if all(len(r) == 4 for r in rows):
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = np.array(rows)
                poses.append(T); i += 3
            else:
                i += 1
        except ValueError:
            i += 1
    return np.stack(poses)


# Метрики (те же что в DytanVO run_eval.py)

def umeyama(src, dst):
    N = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc**2).sum() / N
    cov   = dc.T @ sc / N
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    c = (S @ D.diagonal()) / var_s
    t = mu_d - c * R @ mu_s
    return R, t, c


def compute_metrics(gt, pred):
    N = min(len(gt), len(pred))
    gt, pred = gt[:N], pred[:N]
    # ATE
    R, t, c = umeyama(pred[:, :3, 3], gt[:, :3, 3])
    pred_al = (c * R @ pred[:, :3, 3].T).T + t
    err = np.linalg.norm(gt[:, :3, 3] - pred_al, axis=1)
    ate_rmse = float(np.sqrt((err**2).mean()))
    # RPE
    t_err, r_err = [], []
    for i in range(N - 1):
        dgt   = np.linalg.inv(gt[i])   @ gt[i+1]
        dpred = np.linalg.inv(pred[i]) @ pred[i+1]
        derr  = np.linalg.inv(dpred)   @ dgt
        t_err.append(np.linalg.norm(derr[:3, 3]))
        val = np.clip((np.trace(derr[:3, :3]) - 1) / 2, -1, 1)
        r_err.append(float(np.degrees(np.abs(np.arccos(val)))))
    # t_rel / r_rel
    dist = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))])
    t_rels, r_rels = [], []
    for L in [100, 200, 300, 400, 500]:
        for s in range(N):
            e = np.searchsorted(dist, dist[s] + L)
            if e >= N: break
            dgt   = np.linalg.inv(gt[s])   @ gt[e]
            dpred = np.linalg.inv(pred[s]) @ pred[e]
            derr  = np.linalg.inv(dpred)   @ dgt
            actual = dist[e] - dist[s]
            if actual > 0:
                t_rels.append(np.linalg.norm(derr[:3, 3]) / actual * 100)
                val = np.clip((np.trace(derr[:3, :3]) - 1) / 2, -1, 1)
                r_rels.append(np.degrees(np.abs(np.arccos(val))) / actual * 100)
    return {
        "ate_rmse":       ate_rmse,
        "ate_errors":     err.tolist(),
        "pred_aligned":   pred_al.tolist(),
        "gt_xyz":         gt[:, :3, 3].tolist(),
        "rpe_t_rmse":     float(np.sqrt((np.array(t_err)**2).mean())),
        "rpe_r_mean_deg": float(np.mean(r_err)),
        "t_rel": float(np.mean(t_rels)) if t_rels else float("nan"),
        "r_rel": float(np.mean(r_rels)) if r_rels else float("nan"),
    }


def save_metrics(m, tag, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {k: v for k, v in m.items()
              if k not in ("ate_errors", "pred_aligned", "gt_xyz")}
    record["tag"] = tag
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(record, indent=2))

    # Trajectory plot
    gt = np.array(m["gt_xyz"])
    pa = np.array(m["pred_aligned"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt[:, 0], gt[:, 2], "b-",  lw=2,   label="GT")
    ax.plot(pa[:, 0], pa[:, 2], "r--", lw=1.5, label="CoProU")
    ax.scatter([gt[0, 0]], [gt[0, 2]], c="green", s=60, zorder=5)
    ax.set_title(f"Trajectory — {tag}")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
    ax.legend(); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_trajectory.png", dpi=150)
    plt.close(fig)

    # ATE plot
    errs = m["ate_errors"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(range(len(errs)), errs, alpha=0.25, color="steelblue")
    ax.plot(errs, color="steelblue", lw=1)
    ax.axhline(m["ate_rmse"], color="red", ls="--", lw=1.5,
               label=f"RMSE={m['ate_rmse']:.4f}m")
    ax.set_title(f"ATE — {tag}"); ax.set_xlabel("Frame"); ax.set_ylabel("ATE [m]")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_ate.png", dpi=150)
    plt.close(fig)

    print(f"  ATE RMSE : {m['ate_rmse']:.4f} m")
    print(f"  RPE-t    : {m['rpe_t_rmse']:.4f} m")
    print(f"  t_rel    : {m['t_rel']:.2f} %")
    return record


# CoProU inference

def find_checkpoint():
    ckpts = list(CKPT.rglob("*.pth")) + list(CKPT.rglob("*.pkl"))
    if not ckpts:
        sys.exit(f"Чекпоинт не найден в {CKPT}\n"
                 "Скачайте согласно README CoProU")
    # Берём последний по mtime
    return str(sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1])


def run_kitti(seq="all"):
    ckpt = find_checkpoint()
    # Последовательности 00, 05, 07, 09 — с динамическими объектами
    seqs = ["00", "05", "07", "09"] if seq == "all" else [seq]
    records = []
    for s in seqs:
        seq_dir = KITTI_SEQ / s
        if not seq_dir.exists():
            print(f"[skip] KITTI seq {s}")
            continue
        print(f"\n[KITTI seq {s}]")
        out_dir = RESULTS / "kitti" / f"seq_{s}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "test_vo.py",
               "--pretrained-model", ckpt,
               "--img-height", "256", "--img-width", "832",
               "--dataset-dir", str(KITTI_SEQ),
               "--sequence", s,
               "--output-dir", str(out_dir)]
        print(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd)

        # Метрики через встроенный kitti_eval
        eval_cmd = [sys.executable, "kitti_eval/eval_odom.py",
                    f"--result={out_dir}",
                    "--align=7dof"]
        subprocess.run(eval_cmd)

        # Дополнительные метрики ATE/RPE
        gt_file = KITTI_POSE / f"{s}.txt"
        pred_file = out_dir / f"{s}.txt"
        if pred_file.exists() and gt_file.exists():
            gt   = load_kitti_poses(str(gt_file))
            pred = load_kitti_poses(str(pred_file))
            m    = compute_metrics(gt, pred)
            r = save_metrics(m, f"kitti_{s}", out_dir)
            records.append(r)
    return records


def run_sintel(sintel_pass="clean"):
    """
    CoProU не имеет нативного загрузчика Sintel.
    Используем test_vo.py с кастомным --dataset sintel если
    он есть, иначе запускаем через наш dataloader напрямую.
    """
    ckpt = find_checkpoint()
    img_base = SINTEL_DIR / sintel_pass
    cam_base = SINTEL_DIR / "camdata_left"
    if not img_base.exists():
        sys.exit(f"Sintel {sintel_pass} не найден: {img_base}")

    # Intrinsics Sintel (фиксированные)
    K = np.array([[1120, 0, 511.5], [0, 1120, 217.5], [0, 0, 1]])

    records = []
    for scene in sorted(os.listdir(img_base)):
        img_dir  = img_base / scene
        cam_file = cam_base / scene / "camdata.txt"
        if not img_dir.is_dir() or not cam_file.exists():
            continue
        print(f"\n[Sintel {sintel_pass}/{scene}]")
        out_dir = RESULTS / "sintel" / sintel_pass / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        # Запускаем через test_vo.py с Sintel-специфичными флагами
        cmd = [sys.executable, "test_vo.py",
               "--pretrained-model", ckpt,
               "--img-height", "256", "--img-width", "832",
               "--dataset", "sintel",
               "--dataset-dir", str(img_dir),
               "--output-dir", str(out_dir)]
        subprocess.run(cmd)

        pred_file = out_dir / "pred_poses.txt"
        if pred_file.exists():
            gt   = load_sintel_poses(str(cam_file))
            pred = load_kitti_poses(str(pred_file))
            m    = compute_metrics(gt, pred)
            r = save_metrics(m, f"sintel_{sintel_pass}_{scene}", out_dir)
            records.append(r)
    return records


def run_shibuya():
    ckpt = find_checkpoint()
    if not SHIBUYA_DIR.exists():
        sys.exit(f"Shibuya не найден: {SHIBUYA_DIR}")
    records = []
    for traj in sorted(SHIBUYA_DIR.iterdir()):
        if not traj.is_dir(): continue
        img_dir = traj / "image_0"
        gt_file = traj / "gt_pose.txt"
        if not img_dir.exists() or not gt_file.exists():
            continue
        print(f"\n[Shibuya {traj.name}]")
        out_dir = RESULTS / "shibuya" / traj.name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "test_vo.py",
               "--pretrained-model", ckpt,
               "--img-height", "256", "--img-width", "832",
               "--dataset", "tartanair",
               "--dataset-dir", str(img_dir),
               "--output-dir", str(out_dir)]
        subprocess.run(cmd)

        pred_file = out_dir / "pred_poses.txt"
        if pred_file.exists():
            gt   = load_tartanair_poses(str(gt_file))
            pred = load_tartanair_poses(str(pred_file))
            m    = compute_metrics(gt, pred)
            r = save_metrics(m, f"shibuya_{traj.name}", out_dir)
            records.append(r)
    return records


def summarize():
    records = []
    for jf in sorted(RESULTS.rglob("*_metrics.json")):
        records.append(json.loads(jf.read_text()))
    if not records:
        print("Нет метрик.")
        return
    (RESULTS / "summary_metrics.json").write_text(
        json.dumps(records, indent=2))

    fig, ax = plt.subplots(figsize=(max(6, len(records) * 0.8), 4))
    tags = [r["tag"]      for r in records]
    ates = [r["ate_rmse"] for r in records]
    bars = ax.bar(tags, ates, color="steelblue", edgecolor="black", lw=0.5)
    for b, v in zip(bars, ates):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("ATE RMSE [m]")
    ax.set_title("CoProU — ATE RMSE")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(RESULTS / "summary_ate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n{:<35} {:>10} {:>10} {:>10}".format(
        "Sequence", "ATE[m]", "t_rel[%]", "RPE-t[m]"))
    print("-" * 70)
    for r in records:
        print("{:<35} {:>10.4f} {:>10.2f} {:>10.4f}".format(
            r.get("tag",""),
            r.get("ate_rmse", float("nan")),
            r.get("t_rel",    float("nan")),
            r.get("rpe_t_rmse", float("nan")),
        ))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   choices=["kitti","sintel","shibuya"])
    p.add_argument("--seq",       default="all")
    p.add_argument("--pass",      dest="sintel_pass", default="clean",
                   choices=["clean","final"])
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    if args.summarize:
        summarize()
    elif args.dataset == "kitti":
        records = run_kitti(args.seq)
        if records:
            summarize()
    elif args.dataset == "sintel":
        records = run_sintel(args.sintel_pass)
        if records: summarize()
    elif args.dataset == "shibuya":
        records = run_shibuya()
        if records: summarize()
    else:
        p.print_help()