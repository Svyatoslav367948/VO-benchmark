import numpy as np
import math


def umeyama_full(src_xyz, dst_xyz):
    """R, t, scale: dst ≈ scale * R @ src + t"""
    n = src_xyz.shape[0]
    mu_s = src_xyz.mean(0)
    mu_d = dst_xyz.mean(0)
    sc = src_xyz - mu_s
    dc = dst_xyz - mu_d
    var_s = (sc ** 2).sum() / n
    if var_s < 1e-10:
        return np.eye(3), mu_d - mu_s, 1.0
    cov = dc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    c = float((S @ D.diagonal()) / var_s)
    t = mu_d - c * R @ mu_s
    return R, t, c


def poses_to_se3(positions_xyz, orientations_quat_wxyz):
    from scipy.spatial.transform import Rotation
    poses = []
    for xyz, wxyz in zip(positions_xyz, orientations_quat_wxyz):
        T = np.eye(4)
        T[:3, 3] = xyz.copy()
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        T[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
        poses.append(T)
    return poses


def compute_rpe_scaled(traj_ref, traj_est, param_delta=1):
    """
    Правильный RPE для up-to-scale VO (LeapVO, DyTanVO и др.)

    ПОЧЕМУ inv(pred_rel) @ gt_rel НЕПРАВИЛЬНО для monocular VO:
    После Umeyama alignment позиции pred выровнены с GT, но rotation
    матрицы SE3 остаются в разных системах координат (pred и GT могут
    иметь разную ориентацию мировых осей). inv(pred_rel) @ gt_rel
    смешивает rotation из двух систем, внося фиктивную ошибку в
    translation часть err44 (~1.9м при реальной ошибке ~0.04м).

    ПРАВИЛЬНАЯ ФОРМУЛА:
    RPE-t = norm(gt_step_world - pred_step_world)
    где оба вектора в одном world frame после Umeyama alignment.
    Это идентично тому как считает DyTanVO:
      - transform_trajs выравнивает обе траектории в один world frame
      - затем relative translation берётся напрямую из позиций

    param_delta: шаг между кадрами (1 = соседние кадры)
    """
    gt_xyz   = traj_ref.positions_xyz
    pred_xyz = traj_est.positions_xyz

    # Полное Umeyama: выравниваем pred → GT world frame
    R, t, scale = umeyama_full(pred_xyz, gt_xyz)
    pred_aligned = (scale * R @ pred_xyz.T).T + t

    print(f"  [RPE] scale = {scale:.4f}")

    # RPE-t: разница шагов в world frame
    n = min(len(gt_xyz), len(pred_aligned))
    t_errs = []
    for i in range(n - param_delta):
        j = i + param_delta
        gt_step   = gt_xyz[j]       - gt_xyz[i]
        pred_step = pred_aligned[j] - pred_aligned[i]
        t_errs.append(float(np.linalg.norm(gt_step - pred_step)))

    # RPE-r: rotation ошибка через SE3 (rotation не зависит от систем координат
    # если применить R_align к rotation pred)
    gt_se3   = poses_to_se3(gt_xyz, traj_ref.orientations_quat_wxyz)
    from scipy.spatial.transform import Rotation as Rot
    pred_se3_aligned = []
    for xyz_al, wxyz in zip(pred_aligned, traj_est.orientations_quat_wxyz):
        T = np.eye(4)
        T[:3, 3] = xyz_al
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        T[:3, :3] = R @ Rot.from_quat(xyzw).as_matrix()
        pred_se3_aligned.append(T)

    r_errs = []
    for i in range(n - param_delta):
        j = i + param_delta
        gt_rel   = np.linalg.inv(gt_se3[i])              @ gt_se3[j]
        pred_rel = np.linalg.inv(pred_se3_aligned[i])    @ pred_se3_aligned[j]
        R_err    = gt_rel[:3, :3] @ pred_rel[:3, :3].T
        cos_a    = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        r_errs.append(math.degrees(abs(math.acos(cos_a))))

    rpe_t_rmse = float(np.sqrt(np.mean(np.array(t_errs) ** 2)))
    rpe_r_mean = float(np.mean(r_errs))
    rpe_t_mean = float(np.mean(t_errs))

    print(f"  [RPE] t RMSE = {rpe_t_rmse:.6f} m")
    print(f"  [RPE] r mean = {rpe_r_mean:.6f} deg")

    return rpe_t_rmse, rpe_r_mean, rpe_t_mean, scale


def kitti_rpe_scaled(traj_ref, traj_est, scale):
    """
    KITTI t_rel (%) и r_rel (deg/100m).
    Использует ту же логику: разница шагов в world frame после alignment.
    """
    gt_xyz   = traj_ref.positions_xyz
    pred_xyz = traj_est.positions_xyz

    R, t, scale = umeyama_full(pred_xyz, gt_xyz)
    pred_aligned = (scale * R @ pred_xyz.T).T + t

    n = min(len(gt_xyz), len(pred_aligned))

    # Кумулятивные расстояния по GT
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.linalg.norm(gt_xyz[i] - gt_xyz[i-1])

    total = dist[-1]
    print(f"  [KITTI] GT path = {total:.2f} m")

    if total < 10:
        return float("nan"), float("nan")

    lengths = ([L for L in [100, 200, 300, 400, 500, 600, 700, 800]
                if L <= total]
               if total >= 100
               else [total * f for f in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]])

    # Rotation для r_rel
    gt_se3 = poses_to_se3(gt_xyz, traj_ref.orientations_quat_wxyz)
    from scipy.spatial.transform import Rotation as Rot
    pred_se3_aligned = []
    for xyz_al, wxyz in zip(pred_aligned, traj_est.orientations_quat_wxyz):
        T = np.eye(4)
        T[:3, 3] = xyz_al
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        T[:3, :3] = R @ Rot.from_quat(xyzw).as_matrix()
        pred_se3_aligned.append(T)

    t_errs, r_errs = [], []
    for s in range(n):
        for L in lengths:
            e = s
            while e < n and dist[e] - dist[s] < L:
                e += 1
            if e >= n:
                continue
            actual = dist[e] - dist[s]
            if actual < 1.0:
                continue

            # t_rel: разница шагов в world frame
            gt_step   = gt_xyz[e]       - gt_xyz[s]
            pred_step = pred_aligned[e] - pred_aligned[s]
            t_errs.append(np.linalg.norm(gt_step - pred_step) / actual * 100.0)

            # r_rel: через rotation матрицы
            gt_rel   = np.linalg.inv(gt_se3[s])           @ gt_se3[e]
            pred_rel = np.linalg.inv(pred_se3_aligned[s]) @ pred_se3_aligned[e]
            R_err    = gt_rel[:3, :3] @ pred_rel[:3, :3].T
            cos_a    = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
            r_errs.append(math.degrees(abs(math.acos(cos_a))) / actual * 100.0)

    if not t_errs:
        return float("nan"), float("nan")

    return float(np.mean(t_errs)), float(np.mean(r_errs))
