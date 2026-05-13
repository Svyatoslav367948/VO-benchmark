"""
test_vo_sintel.py — CoProU на MPI-Sintel.

Структура:
  {sintel_dir}/training/final/{scene}/frame_XXXX.png
  {sintel_dir}/training/camdata_left/{scene}/frame_XXXX.cam

Запуск:
  python test_vo_sintel.py \
    --pretrained-posenet checkpoints/exp_pose_checkpoint_kitti.pth.tar \
    --img-height 256 --img-width 832 \
    --sintel-dir /workspace/data/MPI-Sintel-complete \
    --output-dir results/sintel_final \
    --sintel-pass final
"""

import argparse
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from inverse_warp import pose_vec2mat
import models
from test_vo_dynakitti import (load_posenet, load_tensor_image,
                                compute_and_save_metrics, print_summary)

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-posenet", type=str, default=None)
parser.add_argument("--pretrained-model",   type=str, default=None)
parser.add_argument("--img-height", type=int, default=256)
parser.add_argument("--img-width",  type=int, default=832)
parser.add_argument("--sintel-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--sintel-pass", choices=["clean", "final"], default="final")
parser.add_argument("--scenes",     type=str, nargs="*", default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG_FLOAT = 202021.25


def read_sintel_cam(cam_file):
    """Бинарный .cam файл → (K 3x3, N 3x4)."""
    with open(str(cam_file), "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert abs(tag - TAG_FLOAT) < 1e-2, \
            f"Неверный тег {tag} в {cam_file}"
        K = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        N = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    return K, N


def load_sintel_gt(cam_dir):
    """
    Читает все .cam файлы сцены.
    N — world2cam 3x4 → инвертируем → cam2world 4x4.
    """
    cam_dir   = Path(cam_dir)
    cam_files = sorted(cam_dir.files("frame_*.cam"))
    if not cam_files:
        raise FileNotFoundError(f"Нет .cam файлов в {cam_dir}")
    poses = []
    for cf in cam_files:
        _, N = read_sintel_cam(cf)
        T_w2c = np.vstack([N, [0, 0, 0, 1]])
        T_c2w = np.linalg.inv(T_w2c)
        poses.append(T_c2w)
    return poses


@torch.no_grad()
def run_scene(img_dir, cam_dir, scene_name, pose_net, args, output_dir):
    img_dir  = Path(img_dir)
    images   = sorted(img_dir.files("*.png"))

    if len(images) < 2:
        print(f"[skip] {scene_name}: меньше 2 кадров")
        return

    print(f"\n{'='*50}")
    print(f"  Sintel: {scene_name}  ({len(images)} кадров)")

    global_pose = np.eye(4)
    pred_poses  = [global_pose[:3, :].flatten()]

    t1 = load_tensor_image(images[0], args.img_height, args.img_width)
    for i in tqdm(range(len(images) - 1), desc=scene_name):
        t2       = load_tensor_image(images[i + 1], args.img_height, args.img_width)
        pose     = pose_net(t1, t2)
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, [0, 0, 0, 1]])
        global_pose = global_pose @ np.linalg.inv(pose_mat)
        pred_poses.append(global_pose[:3, :].flatten())
        t1 = t2

    pred_kitti = np.array(pred_poses)

    out_dir = Path(output_dir) / scene_name
    out_dir.makedirs_p()
    np.savetxt(str(out_dir / "pred_kitti.txt"), pred_kitti, fmt="%1.8e")

    cam_dir = Path(cam_dir)
    if cam_dir.exists():
        try:
            gt_4x4   = load_sintel_gt(cam_dir)
            gt_kitti = np.array([T[:3, :].flatten() for T in gt_4x4])
            np.savetxt(str(out_dir / "gt_kitti.txt"), gt_kitti, fmt="%1.8e")
            compute_and_save_metrics(pred_kitti, gt_kitti, scene_name, out_dir)
        except Exception as e:
            print(f"  [warn] Ошибка GT: {e}")
    else:
        print(f"  [warn] cam_dir не найден: {cam_dir}")


def main():
    args = parser.parse_args()
    pose_net = load_posenet(args)

    sintel_root = Path(args.sintel_dir)
    img_base    = sintel_root / "training" / args.sintel_pass
    cam_base    = sintel_root / "training" / "camdata_left"

    if not img_base.exists():
        raise FileNotFoundError(f"Не найден: {img_base}")

    scenes = args.scenes or sorted([d.name for d in img_base.dirs()])
    print(f"Sintel ({args.sintel_pass}): {len(scenes)} сцен")

    for scene in scenes:
        img_dir = img_base / scene
        cam_dir = cam_base / scene
        if img_dir.exists():
            run_scene(img_dir, cam_dir, scene, pose_net, args, args.output_dir)

    print_summary(args.output_dir)


if __name__ == "__main__":
    main()