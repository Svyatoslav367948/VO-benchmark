#!/usr/bin/env python3
"""
run_eval.py — запуск DytanVO на KITTI / Sintel / Shibuya + метрики

Реальная структура данных (внутри контейнера /workspace/data/):

  kitti/
    clean_calib/              <- ПАПКА с файлами 00.txt, 02.txt ...
                                 каждый файл: 4 строки x 12 чисел без заголовков (P0-P3)
    dataset/
      poses/
        00.txt  05.txt  07.txt  09.txt   <- GT позы, N строк x 12 чисел
      sequences/
        00/
          image_2/      <- цветные кадры (используем их)
          image_3/
          calib.txt     <- P0: ... P1: ... P2: ... P3: (с заголовками)
          calib_leap.txt
          times.txt
        05/  07/  09/  ...

  MPI-Sintel-complete/training/
    clean/<scene>/frame_XXXX.png
    final/<scene>/frame_XXXX.png
    camdata_left/<scene>/camdata.txt

  shibuya/
    RoadCrossing03/
      image_0/     <- кадры
      gt_pose.txt  <- tx ty tz qx qy qz qw
    RoadCrossing04/ ...

Веса моделей (/workspace/DytanVO/models/):
  vonet.pkl         <- VO модель (у тебя vonet.pkl, не vonet_ft.pkl)
  flownet.pkl
  posenet.pkl
  segnet-kitti.pth
  segnet-sf.pth

КЛЮЧЕВОЕ по калибровке KITTI:
  Используем calib.txt из sequences/NN/ — он содержит строки "P0: ...", "P2: ..."
  Это НАТИВНЫЙ формат для --kitti-intrinsics-file в DytanVO.
  clean_calib/ НЕ используется (там нет заголовков, DytanVO его не поймёт).

Использование:
  python run_eval.py --dataset kitti
  python run_eval.py --dataset kitti --seq 09
  python run_eval.py --dataset sintel --pass clean
  python run_eval.py --dataset sintel --pass final
  python run_eval.py --dataset sintel --pass clean --scene alley_1
  python run_eval.py --dataset shibuya
  python run_eval.py --summarize
"""

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

# ================================================================
# Пути внутри контейнера
# ================================================================
DATA    = Path("/workspace/data")
RESULTS = Path("/workspace/DytanVO/results")
MODELS  = Path("/workspace/DytanVO/models")

# KITTI
KITTI_POSE_DIR = DATA / "kitti" / "dataset" / "poses"
KITTI_SEQ_DIR  = DATA / "kitti" / "dataset" / "sequences"
# calib.txt лежит прямо в sequences/NN/ — используем его (P0:/P1:/P2:/P3:)

# Sintel
SINTEL_DIR = DATA / "MPI-Sintel-complete" / "training"

# Shibuya
SHIBUYA_DIR = DATA / "shibuya"


# ================================================================
# Загрузка поз
# ================================================================

def load_kitti_poses(path):
    """Читает poses/XX.txt: N строк x 12 чисел (3x4 матрица). -> (N,4,4)."""
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.strip().split()))
            if len(v) == 12:
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = np.array(v).reshape(3, 4)
                poses.append(T)
    if not poses:
        raise ValueError(f"Нет поз в {path}")
    return np.stack(poses)


def load_tartanair_poses(path):
    """Читает gt_pose.txt TartanAir: tx ty tz qx qy qz qw."""
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
    if not poses:
        raise ValueError(f"Нет поз в {path}")
    return np.stack(poses)


def load_sintel_poses(path):
    """Читает camdata.txt Sintel: блоки 3 строки x 4 числа = одна 3x4 матрица."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    poses, i = [], 0
    while i + 2 < len(lines):
        try:
            rows = [list(map(float, lines[i + j].split())) for j in range(3)]
            if all(len(r) == 4 for r in rows):
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = np.array(rows)
                poses.append(T)
                i += 3
            else:
                i += 1
        except ValueError:
            i += 1
    if not poses:
        raise ValueError(f"Нет поз в {path}")
    return np.stack(poses)


# ================================================================
# Метрики ATE / RPE / t_rel / r_rel
# ================================================================

def umeyama(src, dst):
    N = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / N
    cov   = dc.T @ sc / N
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1, 1, float(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    c = (S @ D.diagonal()) / var_s
    t = mu_d - c * R @ mu_s
    return R, t, float(c)


def compute_ate(gt, pred):
    R, t, c = umeyama(pred[:, :3, 3], gt[:, :3, 3])
    pred_al = (c * R @ pred[:, :3, 3].T).T + t
    err = np.linalg.norm(gt[:, :3, 3] - pred_al, axis=1)
    return {
        "ate_rmse": float(np.sqrt((err ** 2).mean())),
        "ate_mean": float(err.mean()),
        "_errors":  err.tolist(),
        "_gt_xyz":  gt[:, :3, 3].tolist(),
        "_pred_al": pred_al.tolist(),
    }


def compute_rpe(gt, pred):
    t_err, r_err = [], []
    for i in range(len(gt) - 1):
        dgt   = np.linalg.inv(gt[i])   @ gt[i + 1]
        dpred = np.linalg.inv(pred[i]) @ pred[i + 1]
        derr  = np.linalg.inv(dpred)   @ dgt
        t_err.append(float(np.linalg.norm(derr[:3, 3])))
        val = np.clip((np.trace(derr[:3, :3]) - 1) / 2, -1.0, 1.0)
        r_err.append(float(np.degrees(np.abs(np.arccos(val)))))
    return {
        "rpe_t_rmse":     float(np.sqrt((np.array(t_err) ** 2).mean())),
        "rpe_r_mean_deg": float(np.mean(r_err)),
    }


def compute_t_rel_r_rel(gt, pred):
    dist = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))])
    t_rels, r_rels = [], []
    for L in [100, 200, 300, 400, 500, 600, 700, 800]:
        for s in range(len(gt)):
            e = np.searchsorted(dist, dist[s] + L)
            if e >= len(gt):
                break
            dgt   = np.linalg.inv(gt[s])   @ gt[e]
            dpred = np.linalg.inv(pred[s]) @ pred[e]
            derr  = np.linalg.inv(dpred)   @ dgt
            actual = float(dist[e] - dist[s])
            if actual > 0:
                t_rels.append(np.linalg.norm(derr[:3, 3]) / actual * 100)
                val = np.clip((np.trace(derr[:3, :3]) - 1) / 2, -1.0, 1.0)
                r_rels.append(np.degrees(np.abs(np.arccos(val))) / actual * 100)
    return {
        "t_rel": float(np.mean(t_rels)) if t_rels else float("nan"),
        "r_rel": float(np.mean(r_rels)) if r_rels else float("nan"),
    }


# ================================================================
# Графики
# ================================================================

def plot_trajectory(gt_xyz, pred_al, tag, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xyz[:, 0],  gt_xyz[:, 2],  "b-",  lw=2,   label="GT")
    ax.plot(pred_al[:, 0], pred_al[:, 2], "r--", lw=1.5, label="DytanVO")
    ax.scatter([gt_xyz[0, 0]], [gt_xyz[0, 2]], c="green", s=60, zorder=5, label="Start")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(f"Trajectory — {tag}")
    ax.legend(); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_trajectory.png", dpi=150)
    plt.close(fig)


def plot_ate(errors, rmse, tag, outdir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(range(len(errors)), errors, alpha=0.25, color="steelblue")
    ax.plot(errors, color="steelblue", lw=1)
    ax.axhline(rmse, color="red", ls="--", lw=1.5, label=f"RMSE={rmse:.4f} m")
    ax.set_xlabel("Frame"); ax.set_ylabel("ATE [m]")
    ax.set_title(f"ATE — {tag}"); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_ate.png", dpi=150)
    plt.close(fig)


def plot_summary_bar(records, outdir):
    if not records:
        return
    tags = [r["tag"]      for r in records]
    ates = [r["ate_rmse"] for r in records]
    fig, ax = plt.subplots(figsize=(max(6, len(tags) * 0.9), 5))
    bars = ax.bar(tags, ates, color="steelblue", edgecolor="black", lw=0.5)
    for b, v in zip(bars, ates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("ATE RMSE [m]")
    ax.set_title("DytanVO — ATE RMSE")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "summary_ate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Запуск инференса + метрики
# ================================================================

def run_inference(extra_args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        [sys.executable, "-W", "ignore::UserWarning",
         "vo_trajectory_from_folder.py"]
        + [str(a) for a in extra_args]
        + ["--outdir", str(out_dir)]
    )
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/workspace/DytanVO")
    return result.returncode == 0


def compute_and_save(pred_file, gt_poses, tag, out_dir):
    """Загружает предсказанные позы, считает метрики, сохраняет JSON и PNG."""
    if not pred_file.exists():
        print(f"  [skip] предсказанные позы не найдены: {pred_file}")
        return None

    # Предсказанные позы DytanVO — всегда KITTI-формат (12 чисел на строку)
    # Пытаемся загрузить конвертированный KITTI файл
    pred_file_kitti = str(pred_file).replace('.txt', '_kitti.txt')
    if os.path.exists(pred_file_kitti):
        pred = load_kitti_poses(pred_file_kitti)
    else:
        pred = load_kitti_poses(str(pred_file))
    gt   = gt_poses

    N = min(len(gt), len(pred))
    if N < 2:
        print(f"  [skip] слишком мало кадров: {N}")
        return None
    gt, pred = gt[:N], pred[:N]

    m_ate  = compute_ate(gt, pred)
    m_rpe  = compute_rpe(gt, pred)
    m_trel = compute_t_rel_r_rel(gt, pred)

    record = {
        "tag":            tag,
        "ate_rmse":       m_ate["ate_rmse"],
        "ate_mean":       m_ate["ate_mean"],
        "rpe_t_rmse":     m_rpe["rpe_t_rmse"],
        "rpe_r_mean_deg": m_rpe["rpe_r_mean_deg"],
        "t_rel":          m_trel["t_rel"],
        "r_rel":          m_trel["r_rel"],
    }
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(record, indent=2))

    gt_xyz  = np.array(m_ate["_gt_xyz"])
    pred_al = np.array(m_ate["_pred_al"])
    plot_trajectory(gt_xyz, pred_al, tag, out_dir)
    plot_ate(m_ate["_errors"], m_ate["ate_rmse"], tag, out_dir)

    print(f"  ATE RMSE : {record['ate_rmse']:.4f} m")
    print(f"  RPE-t    : {record['rpe_t_rmse']:.4f} m")
    print(f"  RPE-r    : {record['rpe_r_mean_deg']:.4f} deg")
    print(f"  t_rel    : {record['t_rel']:.2f} %")
    print(f"  r_rel    : {record['r_rel']:.4f} deg/100m")
    return record


def _check_model(name):
    p = MODELS / name
    if not p.exists():
        available = sorted(MODELS.iterdir()) if MODELS.exists() else []
        msg = f"\n[ERROR] Модель не найдена: {p}\n"
        if available:
            msg += "Доступные файлы в models/:\n"
            msg += "\n".join(f"  {f.name}" for f in available)
        sys.exit(msg)


# ================================================================
# KITTI
# ================================================================

def run_kitti(seq="all"):
    _check_model("vonet.pkl")
    _check_model("segnet-kitti.pth")

    DYNAMIC_SEQS = ["00", "05", "07", "09"]
    seqs = DYNAMIC_SEQS if seq == "all" else [seq]

    records = []
    for s in seqs:
        seq_dir   = KITTI_SEQ_DIR / s
        img_dir   = seq_dir / "image_2"
        pose_file = KITTI_POSE_DIR / f"{s}.txt"
        # calib.txt лежит прямо в sequences/NN/ — формат P0:/P1:/P2:/P3:
        # это нативный формат --kitti-intrinsics-file для DytanVO
        calib_file = seq_dir / "calib.txt"

        # Проверяем наличие всего необходимого
        missing = []
        if not img_dir.exists():    missing.append(str(img_dir))
        if not pose_file.exists():  missing.append(str(pose_file))
        if not calib_file.exists(): missing.append(str(calib_file))
        if missing:
            print(f"\n[skip] KITTI seq {s} — не найдены:")
            for m in missing: print(f"  {m}")
            continue

        print(f"\n{'='*55}")
        print(f"  KITTI seq {s}")
        print(f"  images : {img_dir}")
        print(f"  poses  : {pose_file}")
        print(f"  calib  : {calib_file}")

        out_dir = RESULTS / "kitti" / f"seq_{s}"

        ok = run_inference([
            "--kitti",                          # нативный флаг DytanVO для KITTI
            "--kitti-intrinsics-file", calib_file,
            "--test-dir",  img_dir,
            "--pose-file", pose_file,
            "--vo-model-name",  "vonet.pkl",
            "--seg-model-name", "segnet-kitti.pth",
            "--batch-size", "1",
            "--worker-num", "8",
        ], out_dir)

        if ok:
            gt = load_kitti_poses(str(pose_file))
            r  = compute_and_save(out_dir / "est_poses.txt", gt, f"kitti_{s}", out_dir)
            if r:
                records.append(r)

    return records


# ================================================================
# Sintel
# ================================================================

def run_sintel(sintel_pass="clean", scene="all"):
    _check_model("flownet.pkl")
    _check_model("posenet.pkl")
    _check_model("segnet-sf.pth")

    img_base = SINTEL_DIR / sintel_pass
    cam_base = SINTEL_DIR / "camdata_left"

    if not img_base.exists():
        sys.exit(f"\n[ERROR] Sintel {sintel_pass} не найден: {img_base}")
    if not cam_base.exists():
        sys.exit(f"\n[ERROR] Sintel camdata_left не найден: {cam_base}")

    scenes = [scene] if scene != "all" else sorted(os.listdir(img_base))

    records = []
    for sc in scenes:
        img_dir = img_base / sc
        cam_dir = cam_base / sc   # ← ПАПКА, а не файл

        if not img_dir.is_dir():
            continue
        if not cam_dir.is_dir():   # ← проверяем, что ПАПКА существует
            print(f"\n[skip] Sintel: папка {cam_dir} не найдена для сцены {sc}")
            continue

        print(f"\n{'='*55}")
        print(f"  Sintel {sintel_pass} / {sc}")
        print(f"  images : {img_dir}")
        print(f"  poses  : {cam_dir}")

        out_dir = RESULTS / "sintel" / sintel_pass / sc

        ok = run_inference([
            "--sintel",
            "--test-dir",        str(img_dir),
            "--pose-file",       str(cam_dir),   # папка с frame_XXXX.cam
            "--flow-model-name", "flownet.pkl",
            "--pose-model-name", "posenet.pkl",
            "--seg-model-name",  "segnet-sf.pth",
            "--batch-size", "1",
            "--worker-num", "4",
        ], out_dir)

        if ok:
            from scipy.spatial.transform import Rotation as R_
            # Читаем GT из .cam файлов
            import glob, struct
            cam_files = sorted(glob.glob(str(cam_dir / "frame_*.cam")))
            gt_4x4 = []
            for cf in cam_files:
                with open(cf, "rb") as f:
                    floats = struct.unpack("21f", f.read(84))
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = np.array(floats[9:21]).reshape(3, 4)
                gt_4x4.append(T)
            gt_4x4 = np.stack(gt_4x4)
            # Конвертируем в 7-колоночный формат для compute_and_save
            gt_quats = []
            for T in gt_4x4:
                t = T[:3, 3]
                q = R_.from_matrix(T[:3, :3]).as_quat()
                gt_quats.append(np.concatenate([t, q]))
            gt = np.array(gt_quats, dtype=np.float32)
            r = compute_and_save(out_dir / "est_poses.txt", gt,
                                f"sintel_{sintel_pass}_{sc}", out_dir)
            if r:
                records.append(r)

    return records


# ================================================================
# Shibuya
# ================================================================

def run_shibuya():
    _check_model("flownet.pkl")
    _check_model("posenet.pkl")
    _check_model("segnet-sf.pth")

    if not SHIBUYA_DIR.exists():
        sys.exit(f"\n[ERROR] Shibuya не найден: {SHIBUYA_DIR}")

    records = []
    for traj in sorted(SHIBUYA_DIR.iterdir()):
        if not traj.is_dir():
            continue

        img_dir = traj / "image_0"
        gt_file = traj / "gt_pose.txt"

        if not img_dir.exists():
            print(f"\n[skip] image_0 не найден в {traj}")
            continue
        if not gt_file.exists():
            print(f"\n[skip] gt_pose.txt не найден в {traj}")
            continue

        print(f"\n{'='*55}")
        print(f"  Shibuya / {traj.name}")
        print(f"  images : {img_dir}")
        print(f"  poses  : {gt_file}")

        out_dir = RESULTS / "shibuya" / traj.name

        ok = run_inference([
            "--airdos",
            "--test-dir",       img_dir,
            "--pose-file",      gt_file,
            "--flow-model-name","flownet.pkl",
            "--pose-model-name","posenet.pkl",
            "--seg-model",      "segnet-sf.pth",
            "--batch-size", "1",
            "--worker-num", "8",
        ], out_dir)

        if ok:
            gt = load_tartanair_poses(str(gt_file))
            r  = compute_and_save(out_dir / "est_poses.txt", gt,
                                  f"shibuya_{traj.name}", out_dir)
            if r:
                records.append(r)

    return records


# ================================================================
# Сводка
# ================================================================

def summarize():
    records = []
    for jf in sorted(RESULTS.rglob("*_metrics.json")):
        if jf.name == "summary_metrics.json":
            continue  # пропускаем сам summary чтобы не задублировать
        try:
            data = json.loads(jf.read_text())
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except Exception:
            pass

    if not records:
        print("Нет метрик. Сначала запустите инференс.")
        return

    (RESULTS / "summary_metrics.json").write_text(json.dumps(records, indent=2))
    plot_summary_bar(records, RESULTS)

    hdr = f"{'Sequence':<38} {'ATE[m]':>9} {'t_rel%':>8} {'r_rel°/100m':>13} {'RPE-t[m]':>10} {'RPE-r[°]':>10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in records:
        print(
            f"{r.get('tag',''):<38} "
            f"{r.get('ate_rmse',float('nan')):>9.4f} "
            f"{r.get('t_rel',float('nan')):>8.2f} "
            f"{r.get('r_rel',float('nan')):>13.4f} "
            f"{r.get('rpe_t_rmse',float('nan')):>10.4f} "
            f"{r.get('rpe_r_mean_deg',float('nan')):>10.4f}"
        )
    print(f"\nГрафики -> {RESULTS}/summary_ate.png")
    print(f"JSON    -> {RESULTS}/summary_metrics.json")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--dataset",   choices=["kitti", "sintel", "shibuya"])
    p.add_argument("--seq",       default="all",
                   help="KITTI: 00/05/07/09 или 'all'")
    p.add_argument("--pass",      dest="sintel_pass", default="clean",
                   choices=["clean", "final"])
    p.add_argument("--scene",     default="all",
                   help="Sintel: имя сцены или 'all'")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    if args.summarize:
        summarize()
    elif args.dataset == "kitti":
        recs = run_kitti(args.seq)
        if recs:
            plot_summary_bar(recs, RESULTS / "kitti")
        summarize()
    elif args.dataset == "sintel":
        recs = run_sintel(args.sintel_pass, args.scene)
        if recs:
            plot_summary_bar(recs, RESULTS / "sintel" / args.sintel_pass)
        summarize()
    elif args.dataset == "shibuya":
        recs = run_shibuya()
        if recs:
            plot_summary_bar(recs, RESULTS / "shibuya")
        summarize()
    else:
        p.print_help()
