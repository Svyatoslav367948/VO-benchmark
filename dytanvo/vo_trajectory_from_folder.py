from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, ResizeData, dataset_intrinsics, DownscaleFlow
from Datasets.utils import plot_traj, visflow, load_kiiti_intrinsics, load_sceneflow_extrinsics
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from evaluator.transformation import pose_quats2motion_ses, motion_ses2pose_quats
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.evaluator_base import per_frame_scale_alignment
from DytanVO import DytanVO

import os
import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

def load_sintel_poses_from_cam(camdata_dir):
    """
    Читает бинарные .cam файлы MPI-Sintel из папки.
    Каждый файл: 84 байта = 21 float32
      [0:9]  = intrinsics 3x3 (row-major)
      [9:21] = extrinsics 3x4 (row-major)
    Возвращает (N, 4, 4) float32 — абсолютные позы камеры.
    """
    import glob
    import struct

    cam_files = sorted(glob.glob(os.path.join(camdata_dir, "frame_*.cam")))
    if not cam_files:
        raise ValueError(f"Не найдены .cam файлы в {camdata_dir}")

    poses = []
    for cam_file in cam_files:
        with open(cam_file, "rb") as f:
            data = f.read()
        # 21 float32 = 84 bytes
        if len(data) < 84:
            continue
        floats = struct.unpack("21f", data[:84])
        # extrinsics: позиции [9:21] — матрица 3x4
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = np.array(floats[9:21], dtype=np.float32).reshape(3, 4)
        poses.append(T)

    if not poses:
        raise ValueError(f"Не удалось прочитать позы из {camdata_dir}")
    return np.stack(poses)

def get_args():
    parser = argparse.ArgumentParser(description='Inference code of DytanVO')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained VO model (default: "")')
    parser.add_argument('--flow-model-name', default='',
                        help='name of pretrained flow model (default: "")')
    parser.add_argument('--pose-model-name', default='',
                        help='name of pretrained pose model (default: "")')
    parser.add_argument('--seg-model-name', default='',
                        help='name of pretrained segmentation model (default: "")')
    parser.add_argument('--airdos', action='store_true', default=False,
                        help='airdos test (default: False)')
    parser.add_argument('--rs_d435', action='store_true', default=False,
                        help='realsense d435i test (default: False)')
    parser.add_argument('--sceneflow', action='store_true', default=False,
                        help='sceneflow test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--commaai', action='store_true', default=False,
                        help='commaai test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--seg-thresh', type=float, default=0.7,
                        help='threshold for motion segmentation')
    parser.add_argument('--iter-num', type=int, default=2,
                        help='number of iterations')
    # ── Добавлено patch_vo_script.py ──
    parser.add_argument('--kitti-standard', action='store_true',
        help='Стандартный KITTI Odometry (poses/XX.txt, 3x4 матрицы)')
    parser.add_argument('--sintel', action='store_true',
        help='MPI-Sintel dataset (camdata_left/scene/camdata.txt)')
    # ─────────────────────────────────  
    parser.add_argument('--outdir', type=str, default='./results',
                    help='Output directory for results (default: ./results)')                      

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.outdir, exist_ok=True)

    testvo = DytanVO(args.vo_model_name, args.seg_model_name, args.image_height, args.image_width, 
                    args.kitti, args.flow_model_name, args.pose_model_name)

    # load trajectory data from a folder
    if args.kitti:
        datastr = 'kitti'
    
    #     # ПОДМЕНА GT НА КОНВЕРТИРОВАННЫЙ ФАЙЛ (7 колонок)
    #     # Конвертированные GT лежат в ./dytanvo/results/gt_quat на хосте
    #     # В контейнере это /workspace/DytanVO/results/gt_quat
    #     gt_quat_dir = "/workspace/DytanVO/results/gt_quat"
    #     seq_name = args.pose_file.split('/')[-1].replace('.txt', '')
    #     gt_quat_file = f"{gt_quat_dir}/{seq_name}_quat.txt"
        
    #     if os.path.exists(gt_quat_file):
    #         print(f"✅ Используем конвертированный GT (7 колонок): {gt_quat_file}")
    #         args.pose_file = gt_quat_file
    #     else:
    #         print(f"⚠️ Конвертированный GT не найден: {gt_quat_file}")
    #         print("   Сначала запусти: python convert_kitti_gt.py")

    # # ── Добавлено patch_vo_script.py ──
    # if args.kitti_standard:
    #     from Datasets.kitti_standard import (
    #         KITTIStandardVODataset, load_kitti_standard_poses,
    #         load_kitti_standard_intrinsics)
    #     dataset = KITTIStandardVODataset(args.test_dir, transform=transform)
    #     pose_std = load_kitti_standard_poses(args.pose_file)
    #     intrinsics = load_kitti_standard_intrinsics(args.kitti_intrinsics_file)
    elif args.sintel:
        datastr = 'sceneflow' 
        from Datasets.sintel_vo import SINTEL_K
        # Загружаем позы из отдельных .cam файлов
        print(f"[Sintel] pose_file = {args.pose_file}")
    # ─────────────────────────────────
    elif args.airdos:
        datastr = 'airdos'
    elif args.rs_d435:
        datastr = 'rs_d435'
    elif args.sceneflow:
        datastr = 'sceneflow'
    elif args.commaai:
        datastr = 'commaai'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery, baseline = dataset_intrinsics(datastr, '15mm' in args.test_dir) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
        focalx, focaly, centerx, centery, baseline = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    if args.sintel:
        from Datasets.sintel_vo import SINTEL_K
        focalx = SINTEL_K[0, 0]
        focaly = SINTEL_K[1, 1]
        centerx = SINTEL_K[0, 2]
        centery = SINTEL_K[1, 2]
        baseline = 0.0
        print(f"✅ Sintel intrinsics: fx={focalx}, fy={focaly}, cx={centerx}, cy={centery}")

    if datastr == 'kitti':
        transform = Compose([ResizeData((args.image_height, 1226)), CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    else:
        transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    # if datastr == 'sintel':
    #     from Datasets.sintel_vo import SintelVODataset, load_sintel_poses, SINTEL_K
    #     dataset = SintelVODataset(args.test_dir, transform=transform)
    #     pose_std = load_sintel_poses(args.pose_file)
    #     intrinsics = SINTEL_K

    testDataset = TrajFolderDataset(args.test_dir, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    if args.sintel:
        pose_std = load_sintel_poses_from_cam(args.pose_file)
        print(f"✅ Загружено {len(pose_std)} GT поз из {args.pose_file}")
    
    motionlist = []
    testname = datastr + '_' + args.vo_model_name.split('.')[0] + '_' + args.test_dir.split('/')[-1]
    if args.save_flow:
        flowdir = os.path.join(args.outdir, testname + '_flow')
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break

        motion, flow = testvo.test_batch(sample, [focalx, centerx, centery, baseline], args.seg_thresh, args.iter_num)
        motionlist.append(motion)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1

    motions = np.array(motionlist)

    # print("🔧 Применяем коррекцию масштаба к motions...")
    # scale_correction = 0.01  # Попробуйте 0.01, 0.02, 0.005
    # motions[:, :3] = motions[:, :3] * scale_correction
    # print(f"   Коэффициент: {scale_correction}")

    # ✅ СОХРАНЯЕМ ПРЕДСКАЗАННЫЕ ПОЗЫ ПЕРВЫМ ДЕЛОМ
    est_poses = motion_ses2pose_quats(motions)  # преобразуем в 7 колонок
    np.savetxt(os.path.join(args.outdir, 'est_poses.txt'), est_poses)
    print(f"✅ Позы сохранены в {os.path.join(args.outdir, 'est_poses.txt')}")

    # calculate ATE, RPE, KITTI-RPE
    if args.sintel:
        gtposes = pose_std
        # Конвертируем 4x4 матрицы в 7-колоночный формат (tx ty tz qx qy qz qw)
        from scipy.spatial.transform import Rotation
        gt_quats = []
        for T in gtposes:
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
            gt_quats.append(np.concatenate([t, q]))
        gtposes = np.array(gt_quats, dtype=np.float32)
        
        min_len = min(len(gtposes), len(est_poses))
        gtposes = gtposes[:min_len]
        estposes = est_poses[:min_len]
        
        # Преобразуем в формат для evaluator (4x4 матрицы)
        # from evaluator.transformation import quats2SEs
        # gt_se3 = quats2SEs(gtposes)
        # est_se3 = quats2SEs(estposes)

        # gtmotions = pose_quats2motion_ses(gtposes)
        # estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
        # estposes = motion_ses2pose_quats(estmotion_scale)

        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(gtposes, estposes, scale=True, kittitype=False)
        
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # Сохраняем метрики в текстовый файл
        metrics_file = os.path.join(args.outdir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"ATE: {results['ate_score']:.6f}\n")
            f.write(f"ATE_RMSE: {results['ate_score']:.6f}\n")
            f.write(f"KITTI_R: {results['kitti_score'][0]:.6f}\n")
            f.write(f"KITTI_t: {results['kitti_score'][1]:.6f}\n")
            if 'rpe_score' in results:
                f.write(f"RPE_t_RMSE: {results['rpe_score'][0]:.6f}\n")
                f.write(f"RPE_r_mean: {results['rpe_score'][1]:.6f}\n")
                scale_to_meters = results.get('scale', 1.0)
                f.write(f"RPE_t_RMSE_meters: {results['rpe_score'][0] * scale_to_meters:.6f}\n")
                f.write(f"RPE_r_mean_rad: {np.radians(results['rpe_score'][1]):.6f}\n")

        print(f"✅ Метрики сохранены в {metrics_file}")

        # save results and visualization
        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], 
                vis=False, 
                savefigname=os.path.join(args.outdir, testname + '.png'), 
                title='ATE %.4f' %(results['ate_score']))
        np.savetxt(os.path.join(args.outdir, testname + '.txt'), results['est_aligned'])
    
    elif args.pose_file.endswith('.txt'):
        # ==================== KITTI / SHIBUYA / AIRDOS ====================
        if datastr == 'sceneflow':
            gtposes = load_sceneflow_extrinsics(args.pose_file)
        else:
            gtposes = np.loadtxt(args.pose_file)
            if datastr == 'airdos':
                gtposes = gtposes[:,1:]  # remove the first column of timestamps
        
        gtmotions = pose_quats2motion_ses(gtposes)
        estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
        estposes = motion_ses2pose_quats(estmotion_scale)
        
        evaluator = TartanAirEvaluator()
        # Для KITTI kittitype=True, для остальных False
        results = evaluator.evaluate_one_trajectory(gtposes, estposes, scale=True, kittitype=(datastr=='kitti'))
        
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
        
        metrics_file = os.path.join(args.outdir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"ATE: {results['ate_score']:.6f}\n")
            f.write(f"KITTI_R: {results['kitti_score'][0]:.6f}\n")
            f.write(f"KITTI_t: {results['kitti_score'][1]:.6f}\n")
            if 'rpe_score' in results:
                f.write(f"RPE_t_RMSE: {results['rpe_score'][0]:.6f}\n")
                f.write(f"RPE_r_mean: {results['rpe_score'][1]:.6f}\n")
                scale_to_meters = results.get('scale', 1.0)
                f.write(f"RPE_t_RMSE_meters: {results['rpe_score'][0] * scale_to_meters:.6f}\n")
                f.write(f"RPE_r_mean_rad: {np.radians(results['rpe_score'][1]):.6f}\n")

        print(f"✅ Метрики сохранены в {metrics_file}")
        plot_traj(results['gt_aligned'], results['est_aligned'], 
                vis=False, 
                savefigname=os.path.join(args.outdir, testname + '.png'), 
                title='ATE %.4f' %(results['ate_score']))
        np.savetxt(os.path.join(args.outdir, testname + '.txt'), results['est_aligned'])
    
    else:
        np.savetxt(os.path.join(args.outdir, testname + '.txt'), motion_ses2pose_quats(motions))