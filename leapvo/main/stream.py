import os
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

RUN_REPLICA = True
TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"


def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth(filename):
    depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    depth = depth / 1000.0  # turn depth from mm to meter
    return depth


def cam_read_sintel(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
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


def sintel_stream(imagedir, calib_root, stride, skip=0):
    """image generator"""

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[
        skip::stride
    ]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camfile = str(imfile).split("/")[-1].replace(".png", ".cam")
        K, _ = cam_read_sintel(os.path.join(calib_root, camfile))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        calib = [fx, fy, cx, cy]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]
        yield (t, image, intrinsics)

    yield (-1, image, intrinsics)


def dataset_stream(imagedir, calib, stride, skip=0, mode="replica"):
    """image generator"""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[
        skip::stride
    ]

    print("imagedir", imagedir)
    print("image_list", image_list[:5])
    # for replica
    if mode == "replica":
        image_list = sorted(
            image_list,
            key=lambda x: int(str(x).split("/")[-1].split(".")[0].split("_")[-1]),
        )

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])

        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        yield (t, image, intrinsics)

    yield (-1, image, intrinsics)


def replica_stream(scene_dir, calib, stride, skip=0):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    # load camera
    traj_file = os.path.join(scene_dir, "traj_w_c.txt")
    Ts_full_ctow = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
    Ts_full_ctow = Ts_full_ctow[skip::stride]

    imagedir = os.path.join(scene_dir, "rgb")
    depthdir = os.path.join(scene_dir, "depth")

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))
    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in img_exts))

    image_list = sorted(
        image_list,
        key=lambda x: int(str(x).split("/")[-1].split(".")[0].split("_")[-1]),
    )[skip::stride]
    depth_list = sorted(
        depth_list,
        key=lambda x: int(str(x).split("/")[-1].split(".")[0].split("_")[-1]),
    )[skip::stride]

    assert len(depth_list) == len(image_list)
    for t, imfile in enumerate(image_list):
        depthfile = depth_list[t]
        # image = cv2.imread(str(imfile))
        image = load_image(str(imfile))
        depth = load_depth(str(depthfile))

        intrinsics = np.array([fx, fy, cx, cy])

        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]
        depth = depth[: h - h % 16, : w - w % 16]
        # queue.put((t, image, intrinsics))
        yield (t, image, depth, intrinsics, Ts_full_ctow[t])

    # queue.put((-1, image, intrinsics))
    yield (-1, image, depth, intrinsics, Ts_full_ctow[0])


def video_stream(imagedir, calib, stride, skip=0):
    """video generator"""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])
        # queue.put((t, image, intrinsics))
        yield (t, image, intrinsics, "")

        t += 1

    # queue.put((-1, image, intrinsics))
    yield (-1, image, intrinsics, "")
    cap.release()

def kitti_stream(imagedir, calib_file, stride, skip=0):
    """
    Генератор кадров для DynaKITTI.
    calib_file — путь к calib.txt (строки P0:/P2: с 12 числами)
                 ИЛИ путь к calib_numeric.txt (12 чисел без заголовка)
    imagedir   — папка с PNG кадрами (image_2/)
    """
    fx = fy = cx = cy = None

    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Формат с заголовком: "P2: v1 v2 ... v12"
            if line.startswith("P2:"):
                vals = list(map(float, line.split(":")[1].split()))
                fx, cx = vals[0], vals[2]
                fy, cy = vals[5], vals[6]
                break
            # Формат без заголовка: 12 чисел (calib_numeric.txt)
            parts = line.split()
            if len(parts) == 12:
                vals = list(map(float, parts))
                fx, cx = vals[0], vals[2]
                fy, cy = vals[5], vals[6]
                break

    if fx is None:
        raise ValueError(f"Не удалось прочитать intrinsics из {calib_file}")

    img_exts = ["*.png", "*.jpg", "*.jpeg"]
    image_list = sorted(
        chain.from_iterable(Path(imagedir).glob(e) for e in img_exts)
    )[skip::stride]

    if not image_list:
        raise ValueError(f"Нет изображений в {imagedir}")

    intrinsics = np.array([fx, fy, cx, cy])
    image = None

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if image is None:
            print(f"[warn] не удалось прочитать {imfile}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]
        yield (t, image, intrinsics)

    if image is not None:
        yield (-1, image, intrinsics)
