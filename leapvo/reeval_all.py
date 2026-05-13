#!/usr/bin/env python3
"""
reeval_all.py — пересчитывает метрики для всех DynaKITTI последовательностей
используя правильную формулу RPE (world-frame translation diff).

Запуск внутри контейнера leapvo:
  cd /workspace/leapvo
  python3 reeval_all.py

Читает готовые leapvo_traj.txt из results/dynakitti/
и GT позы из /workspace/data/DynaKITTI/
Пишет обновлённые metrics.json и сводную таблицу.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from evo.core import sync
from evo.tools import file_interface
from main.utils import load_traj, make_traj
from main.utils_rpe_fix import compute_rpe_scaled, kitti_rpe_scaled
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

# ── Конфигурация ─────────────────────────────────────────────
RESULTS_ROOT = Path("/workspace/leapvo/results/dynakitti")
DATA_ROOT    = Path("/workspace/data/DynaKITTI")

# Маппинг: имя папки результата → подпапка DynaKITTI с GT
# Формат: "папка_в_results" → "папка_в_DynaKITTI"
# Добавь свои если нужно
SEQ_MAP = {
    "00_1": "00_1",
    "01_0": "01_0",
    "01_1": "01_1",
    "02_0": "02_0",
    "02_1": "02_1",
    "02_2": "02_2",
    "03_0": "03_0",
    "04_0": "04_0",
    "07_1": "07_1",
    "08_0": "08_0",
    "09_0": "09_0",
    "10_0": "10_0",
    "10_1": "10_1",
}

# ─────────────────────────────────────────────────────────────

def reeval_sequence(traj_file: Path, gt_file: Path, seq_name: str):
    """Пересчитывает ATE + RPE для одной последовательности."""
    print(f"\n{'='*55}")
    print(f"  {seq_name}")
    print(f"  pred : {traj_file}")
    print(f"  GT   : {gt_file}")

    # Загрузка GT
    gt_raw, gt_ts = load_traj(str(gt_file), traj_format="kitti")
    traj_ref = make_traj((gt_raw, gt_ts))

    # Загрузка pred (TUM формат: timestamp x y z qx qy qz qw)
    traj_est = file_interface.read_tum_trajectory_file(str(traj_file))

    # Синхронизация timestamps
    if traj_ref.timestamps.shape[0] == traj_est.timestamps.shape[0]:
        traj_ref.timestamps = traj_est.timestamps
    traj_ref_s, traj_est_s = sync.associate_trajectories(traj_ref, traj_est)

    print(f"  кадров: {len(traj_ref_s.positions_xyz)}")

    # ATE (evo, Sim3 Umeyama)
    ate_result = main_ape.ape(
        traj_ref_s, traj_est_s,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=True,
    )
    ate = ate_result.stats["rmse"]

    # RPE-t правильная формула (world-frame translation diff)
    rpe_t, rpe_r, rpe_t_mean, scale = compute_rpe_scaled(traj_ref_s, traj_est_s)

    # KITTI t_rel / r_rel
    t_rel, r_rel = kitti_rpe_scaled(traj_ref_s, traj_est_s, scale)

    print(f"  ATE RMSE : {ate:.4f} m")
    print(f"  RPE-t    : {rpe_t:.4f} m  (scale={scale:.2f})")
    print(f"  RPE-r    : {rpe_r:.4f} deg")
    print(f"  t_rel    : {t_rel:.2f} %")
    print(f"  r_rel    : {r_rel:.4f} deg/100m")

    return {
        "seq":           seq_name,
        "ate_rmse":      round(ate,   6),
        "rpe_t_rmse":    round(rpe_t, 6),
        "rpe_r_mean":    round(rpe_r, 6),
        "t_rel_pct":     round(t_rel, 4) if not math.isnan(t_rel) else None,
        "r_rel_deg100m": round(r_rel, 4) if not math.isnan(r_rel) else None,
        "scale":         round(scale, 4),
    }


def main():
    all_results = []
    not_found   = []

    # Ищем все leapvo_traj.txt в results/dynakitti/
    # Структура: results/dynakitti/<seq>/<seq>/leapvo_traj.txt
    #        или results/dynakitti/<seq>/leapvo_traj.txt
    traj_files = sorted(RESULTS_ROOT.rglob("leapvo_traj.txt"))
    print(f"Найдено {len(traj_files)} файлов leapvo_traj.txt")

    for traj_file in traj_files:
        # Определяем имя последовательности из пути
        # Пробуем распознать seq из имени родительской папки
        parent = traj_file.parent.name  # напр. "00_1" или "dynakitti-00_1-final"

        # Ищем подходящий ключ в SEQ_MAP
        matched_key = None
        for key in SEQ_MAP:
            if key in parent:
                matched_key = key
                break

        if matched_key is None:
            print(f"\n[skip] Не удалось определить seq для: {traj_file}")
            not_found.append(str(traj_file))
            continue

        gt_subdir = SEQ_MAP[matched_key]
        gt_file   = DATA_ROOT / gt_subdir / "pose_left.txt"

        if not gt_file.exists():
            print(f"\n[skip] GT не найден: {gt_file}")
            not_found.append(str(traj_file))
            continue

        try:
            result = reeval_sequence(traj_file, gt_file, matched_key)
            all_results.append(result)

            # Обновляем metrics.json рядом с traj файлом
            metrics_path = traj_file.parent / "metrics_corrected.json"
            metrics_path.write_text(json.dumps(result, indent=2))
            print(f"  → {metrics_path}")

        except Exception as e:
            print(f"\n[ERROR] {traj_file}: {e}")
            not_found.append(str(traj_file))

    # Сводная таблица
    print(f"\n\n{'='*70}")
    print("ИТОГОВАЯ ТАБЛИЦА (правильный RPE-t)")
    print(f"{'='*70}")
    print(f"{'Seq':<12} {'ATE[m]':>8} {'RPE-t[m]':>10} {'RPE-r[°]':>10} "
          f"{'t_rel%':>8} {'scale':>7}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['seq']:<12} "
              f"{r['ate_rmse']:>8.4f} "
              f"{r['rpe_t_rmse']:>10.4f} "
              f"{r['rpe_r_mean']:>10.4f} "
              f"{str(r['t_rel_pct'] or 'nan'):>8} "
              f"{r['scale']:>7.2f}")

    if all_results:
        avg_ate   = np.mean([r["ate_rmse"]   for r in all_results])
        avg_rpet  = np.mean([r["rpe_t_rmse"] for r in all_results])
        avg_rper  = np.mean([r["rpe_r_mean"] for r in all_results])
        t_rels    = [r["t_rel_pct"] for r in all_results if r["t_rel_pct"] is not None]
        avg_trel  = np.mean(t_rels) if t_rels else float("nan")
        print("-" * 70)
        print(f"{'MEAN':<12} {avg_ate:>8.4f} {avg_rpet:>10.4f} "
              f"{avg_rper:>10.4f} {avg_trel:>8.2f}")

    # Сохраняем общий JSON
    summary_path = RESULTS_ROOT / "summary_corrected.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nСводный JSON → {summary_path}")

    if not_found:
        print(f"\nПропущено ({len(not_found)}):")
        for p in not_found: print(f"  {p}")


if __name__ == "__main__":
    main()
