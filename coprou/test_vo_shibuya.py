"""
test_vo_shibuya.py — CoProU на Shibuya (TartanAir формат).

Структура:
  {dataset_dir}/RoadCrossing03/image_0/*.png
  {dataset_dir}/RoadCrossing03/gt_pose.txt  ← tx ty tz qx qy qz qw

Запуск:
  python test_vo_shibuya.py \
    --pretrained-posenet checkpoints/exp_pose_checkpoint_kitti.pth.tar \
    --img-height 256 --img-width 832 \
    --dataset-dir /workspace/data/shibuya \
    --output-dir results/shibuya
"""

import argparse
import numpy as np
import torch
from PIL import Image
from path import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from inverse_warp import pose_vec2mat
import models
from test_vo_dynakitti import (load_posenet, load_tensor_image,
                                compute_and_save_metrics, print_summary)

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-posenet", type=str, default=None)
parser.add_argument("--pretrained-model",   type=str, default=None)
parser.add_argument("--img-height", type=int, default=256)
parser.add_argument("--img-width",  type=int, default=832)
parser.add_argument("--dataset-dir", type=str, required=True)
parser.add_argument("--output-dir",  type=str, required=True)
parser.add_argument("--sequences",   type=str, nargs="*", default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gt_shibuya(pose_file):
    """gt_pose.txt: tx ty tz qx qy qz qw → список 4x4."""
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
    seq_name  = seq_dir.name

    # Shibuya использует image_0
    img_dir = seq_dir / "image_0"
    if not img_dir.exists():
        img_dir = seq_dir / "image_2"
    if not img_dir.exists():
        print(f"[skip] {seq_name}: нет image_0 или image_2")
        return

    pose_file = seq_dir / "gt_pose.txt"
    images    = sorted(img_dir.files("*.png") + img_dir.files("*.jpg"))

    if len(images) < 2:
        print(f"[skip] {seq_name}: меньше 2 кадров")
        return

    print(f"\n{'='*50}")
    print(f"  Shibuya: {seq_name}  ({len(images)} кадров)")

    global_pose = np.eye(4)
    pred_poses  = [global_pose[:3, :].flatten()]

    t1 = load_tensor_image(images[0], args.img_height, args.img_width)
    for i in tqdm(range(len(images) - 1), desc=seq_name):
        t2       = load_tensor_image(images[i + 1], args.img_height, args.img_width)
        pose     = pose_net(t1, t2)
        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, [0, 0, 0, 1]])
        global_pose = global_pose @ np.linalg.inv(pose_mat)
        pred_poses.append(global_pose[:3, :].flatten())
        t1 = t2

    pred_kitti = np.array(pred_poses)

    out_dir = Path(output_dir) / seq_name
    out_dir.makedirs_p()
    np.savetxt(str(out_dir / "pred_kitti.txt"), pred_kitti, fmt="%1.8e")

    if pose_file.exists():
        gt_4x4   = load_gt_shibuya(pose_file)
        gt_kitti = np.array([T[:3, :].flatten() for T in gt_4x4])
        np.savetxt(str(out_dir / "gt_kitti.txt"), gt_kitti, fmt="%1.8e")
        compute_and_save_metrics(pred_kitti, gt_kitti, seq_name, out_dir)
    else:
        print(f"  [warn] gt_pose.txt не найден")


def main():
    args = parser.parse_args()
    pose_net = load_posenet(args)

    dataset_dir = Path(args.dataset_dir)
    if args.sequences:
        seq_dirs = [dataset_dir / s for s in args.sequences]
    else:
        img_dirs = ["image_0", "image_2"]
        seq_dirs = sorted([d for d in dataset_dir.dirs()
                           if any((d / img).exists() for img in img_dirs)])

    print(f"Shibuya: найдено {len(seq_dirs)} последовательностей")
    for sd in seq_dirs:
        run_sequence(sd, pose_net, args, args.output_dir)

    print_summary(args.output_dir)


if __name__ == "__main__":
    main()