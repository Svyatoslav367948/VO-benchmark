import math
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from main.leapvo import LEAPVO
from main.stream import dataset_stream, sintel_stream, video_stream, kitti_stream
from main.utils import (eval_metrics, load_traj, plot_trajectory,
                        save_trajectory_tum_format, update_timestamps)

import time
start = time.time()
# ... инференс на последовательности ...
elapsed = time.time() - start
# fps = n_frames / elapsed
print(f"Время: {elapsed:.1f}с")

@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig):

    slam = None
    skip = 0

    imagedir, calib, stride, skip = (
        cfg.data.imagedir,
        cfg.data.calib,
        cfg.data.stride,
        cfg.data.skip,
    )

    if os.path.isdir(imagedir):
        if cfg.data.traj_format == "sintel":
            dataloader = sintel_stream(imagedir, calib, stride, skip)
        elif cfg.data.traj_format == "kitti":
            dataloader = kitti_stream(imagedir, calib, stride, skip)
        else:
            dataloader = dataset_stream(
                imagedir, calib, stride, skip, mode=cfg.data.traj_format
            )
    else:
        dataloader = video_stream(imagedir, calib, stride, skip)

    image_list = []
    intrinsics_list = []
    for i, (t, image, intrinsics) in enumerate(tqdm(dataloader)):
        if t < 0:
            break

        image_list.append(image)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        # initialization
        if slam is None:
            slam = LEAPVO(cfg, ht=image.shape[1], wd=image.shape[2])

        slam(t, image, intrinsics)

    pred_traj = slam.terminate()

    if "gt_traj" in cfg.data and cfg.data.gt_traj != "":
        gt_traj = load_traj(
            cfg.data.gt_traj,
            cfg.data.traj_format,
            skip=cfg.data.skip,
            stride=cfg.data.stride,
        )
    else:
        gt_traj = None

    os.makedirs(f"{cfg.data.savedir}/{cfg.data.name}", exist_ok=True)

    pred_traj = list(pred_traj)
    if gt_traj is not None:
        if cfg.data.traj_format in ["tum", "tartanair"]:
            pred_traj[1] = update_timestamps(
                cfg.data.gt_traj, cfg.data.traj_format, cfg.data.skip, cfg.data.stride
            )

    if cfg.save_trajectory:
        save_trajectory_tum_format(
            pred_traj, f"{cfg.data.savedir}/{cfg.data.name}/leapvo_traj.txt"
        )

    if cfg.save_video:
        slam.visualizer.save_video(filename=cfg.slam.PATCH_GEN)

    if cfg.save_plot:
        plot_trajectory(
            pred_traj,
            gt_traj=gt_traj,
            title=f"LEAPVO Trajectory Prediction for {cfg.exp_name}",
            filename=f"{cfg.data.savedir}/{cfg.data.name}/traj_plot.pdf",
        )

    if gt_traj is not None:
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj=gt_traj,
            seq=cfg.exp_name,
            filename=os.path.join(cfg.data.savedir, cfg.data.name, "eval_metrics.txt"),
        )

        # Сохраняем JSON
        import json, re
        # Читаем t_rel и r_rel из eval_metrics.txt который только что записан
        t_rel_val = r_rel_val = float("nan")
        try:
            with open(os.path.join(cfg.data.savedir, cfg.data.name, "eval_metrics.txt")) as _f:
                for _line in _f:
                    if _line.startswith("t_rel:"):
                        t_rel_val = float(_line.split(":")[1].strip().split()[0])
                    if _line.startswith("r_rel:"):
                        r_rel_val = float(_line.split(":")[1].strip().split()[0])
        except Exception:
            pass

        metrics = {
            "seq":         cfg.data.name,
            "ate_rmse":    float(ate),
            "rpe_trans":   float(rpe_trans),
            "rpe_rot_deg": float(rpe_rot),
            "t_rel_pct":   t_rel_val,
            "r_rel_deg100m": r_rel_val,
        }
        json_path = os.path.join(cfg.data.savedir, cfg.data.name, "metrics.json")
        with open(json_path, "w") as jf:
            json.dump(metrics, jf, indent=2)
        print(f"[OK] Метрики JSON: {json_path}")

        # Сохраняем PNG траектории
        png_path = os.path.join(cfg.data.savedir, cfg.data.name, "traj_plot.png")
        plot_trajectory(
            pred_traj,
            gt_traj=gt_traj,
            title=f"LeapVO DynaKITTI {cfg.data.name} | ATE={ate:.4f}m",
            filename=png_path,
        )
        print(f"[OK] График PNG: {png_path}")

        # Общий лог
        with open(os.path.join(cfg.data.savedir, "error_sum.txt"), "a+") as f:
            f.write(
                f"{cfg.data.name:<25} | ATE: {ate:.5f} | "
                f"RPE-t: {rpe_trans:.5f} | RPE-r: {rpe_rot:.5f}\n"
            )

    # # visualization
    # if cfg.viz:
    #     vis_rerun(slam, image_list, intrinsics_list)


if __name__ == "__main__":
    main()
