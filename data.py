"""COLMAP sparse loader (incl. EQUIRECTANGULAR model 17) + 6-face pinhole split.

Reads a COLMAP-format sparse reconstruction produced for equirectangular
panoramas (e.g. Spirula Studio's built-in SfM output, or a COLMAP run using
the EQUIRECTANGULAR camera model) and provides:

  * binary readers for cameras.bin / images.bin / points3D.bin (no pycolmap),
  * the equirectangular projection math (matches COLMAP model 17 and
    Spirula's sfm/core/Camera.h),
  * the cube-map face split used during training: every pano camera becomes
    6 pinhole faces of 90-degree FOV (Spirula's warp_spherical_to_pinhole
    idea),
  * a torch grid_sample warp that resamples a panorama into those faces with
    exactly the same ray math, so rendered faces and GT faces line up.

Convention notes
----------------
* COLMAP EQUIRECTANGULAR (model 17) has two parameters (width, height).
  Internally fx = w/(2*pi), fy = h/pi, cx = w/2, cy = h/2 (pixels per radian).
* The pinhole face warp samples each face pixel at its CENTER (u = i + 0.5),
  matching gsplat's rasterizer (RasterizeToPixels3DGSFwd.cu: "px = j + 0.5f")
  and Spirula's WarpFace.cuh ("u = (i + 0.5 - cx) / fx").  Using the corner
  convention instead puts GT and render half a pixel apart and caps quality.
"""

from __future__ import annotations

import math
import os
import struct
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# COLMAP binary readers (only the fields this pipeline needs)
# ---------------------------------------------------------------------------

def read_cameras_bin(path: str) -> List[dict]:
    """cameras.bin -> list of {id, model, width, height, params}."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    count = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    cameras = []
    for _ in range(count):
        cam_id, model_id, width, height = struct.unpack_from("<IiQQ", data, pos)
        pos += 4 + 4 + 8 + 8
        # EQUIRECTANGULAR has 2 params; pinhole-family models carry 3-8.
        if model_id == 17:
            nparams = 2
        else:
            # For other models the parameter count varies; refuse rather than
            # mis-parse.  COLMAP ids: 0 SIMPLE_PINHOLE, 1 PINHOLE,
            # 2 SIMPLE_RADIAL, 3 RADIAL, 4 OPENCV, 5 OPENCV_FISHEYE.
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}.get(model_id)
            if nparams is None:
                raise ValueError(f"camera model {model_id} not supported")
        params = struct.unpack_from("<%dd" % nparams, data, pos)
        pos += 8 * nparams
        cameras.append(
            dict(id=cam_id, model=model_id, width=width, height=height,
                 params=list(params))
        )
    return cameras


def camera_is_equirect(cam: dict) -> bool:
    return cam["model"] == 17


def camera_K(cam: dict) -> np.ndarray:
    """3x3 intrinsics for pinhole-family cameras (None for equirectangular).

    Distortion (models 2-4) is ignored: the caller must feed distortion-free
    images (or accept the approximation).  OPENCV_FISHEYE (5) is rejected
    because rendering it correctly needs the distortion model.
    """
    m, p = cam["model"], cam["params"]
    if m == 17:
        return None
    if m in (0, 2, 3):            # f, cx, cy (+radial k*)
        f, cx, cy = p[0], p[1], p[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    if m in (1, 4):               # fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    raise ValueError(f"camera model {m} not supported for direct rendering")


def read_images_bin(path: str) -> List[dict]:
    """images.bin -> list of {image_id, qvec, tvec, camera_id, name}."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    count = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    images = []
    for _ in range(count):
        image_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        qvec = struct.unpack_from("<4d", data, pos)
        pos += 32
        tvec = struct.unpack_from("<3d", data, pos)
        pos += 24
        camera_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        end = data.index(b"\x00", pos)
        name = data[pos:end].decode("utf-8")
        pos = end + 1
        n_points2d = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        xys = np.empty((n_points2d, 2), np.float64)
        p3d_ids = np.empty(n_points2d, np.int64)
        for k in range(n_points2d):
            xys[k] = struct.unpack_from("<2d", data, pos)
            pos += 16
            p3d_ids[k] = struct.unpack_from("<q", data, pos)[0]
            pos += 8
        images.append(
            dict(image_id=image_id, qvec=qvec, tvec=tvec,
                 camera_id=camera_id, name=name,
                 xys=xys, point3D_ids=p3d_ids)
        )
    return images


def read_points3d_bin(path: str):
    """points3D.bin -> (ids [N] u64, xyz [N,3] f64, rgb [N,3] u8, errors [N] f64)."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    count = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    ids = np.empty(count, np.uint64)
    xyz = np.empty((count, 3), np.float64)
    rgb = np.empty((count, 3), np.uint8)
    err = np.empty(count, np.float64)
    for i in range(count):
        ids[i] = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        xyz[i] = struct.unpack_from("<3d", data, pos)
        pos += 24
        rgb[i] = struct.unpack_from("<3B", data, pos)
        pos += 3
        err[i] = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        track_len = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        pos += 8 * track_len  # (image_id u32, point2D_idx u32) per track
    return ids, xyz, rgb, err


# ---------------------------------------------------------------------------
# Equirectangular camera math (COLMAP model 17 / Spirula sfm convention)
# ---------------------------------------------------------------------------

class EquirectCamera:  #全景相机类
    """Spherical (equirectangular) camera: the image IS the calibration."""

    def __init__(self, width: int, height: int):
        self.width = width  #分幅的透视投影像素
        self.height = height
        self.fx = width / (2.0 * math.pi)  #根据用户定义的分幅影像素，计算内参
        self.fy = height / math.pi
        self.cx = width / 2.0
        self.cy = height / 2.0

    def project(self, xyz_cam: torch.Tensor) -> torch.Tensor:
        """Camera-frame unit direction (..., 3) -> pixel coords (..., 2)."""
        #把 3D 空间方向（射线）映射到全景图像的 2D 像素坐标
        #把相机系下的 3D 向量 (x, y, z) 通过两个角度（θ 经度、φ 纬度）映射到全景图的像素 (u, v)。
        theta = torch.atan2(xyz_cam[..., 0], xyz_cam[..., 2])
        phi = torch.atan2(-xyz_cam[..., 1],
                          torch.hypot(xyz_cam[..., 0], xyz_cam[..., 2]))
        u = self.fx * theta + self.cx
        v = self.cy - self.fy * phi
        return torch.stack([u, v], dim=-1)


# ---------------------------------------------------------------------------
# Cube-map face split (the Spirula warp_spherical_to_pinhole idea)
# ---------------------------------------------------------------------------

# Face view directions in the *pano camera* frame: front/back/left/right/up/down.
FACE_DIRS = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])   #把向量归一化，避免除0错误


def face_rotation(direction) -> np.ndarray:
    """Build R_face (3x3) mapping pano-camera coords -> face-camera coords.

    The face camera looks along `direction` (in the pano camera frame), with
    its up as close to the pano camera's up (+Y) as possible.  R_face is
    orthonormal, so d_face = R_face @ d_pano and the inverse is R_face^T.
    """
    #给定一个面相机的视向方向 direction， R_face这个变量用来把全景相机坐标系的坐标变换到面相机坐标系
    d = _normalize(np.asarray(direction, dtype=np.float64))
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(d, up)) > 0.9:
        up = np.array([0.0, 0.0, -1.0])
    x = _normalize(np.cross(up, d))
    y = np.cross(d, x)
    return np.stack([x, y, d], axis=0)  


def build_face_c2w(c2w_pano: np.ndarray, R_face: np.ndarray) -> np.ndarray:
    """Face camera-to-world from the pano's c2w and the face rotation."""
    C = c2w_pano[:3, 3]
    R = c2w_pano[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ R_face.T  # face w2c rotation = R_face @ R_w2c; invert
    out[:3, 3] = C              # same camera center
    return out  #把全幅的c2w变为面的c2w


def pinhole_intrinsics(face_size: int) -> np.ndarray:
    """90-degree FOV pinhole K.  tan(45 deg) == 1, so f == S/2."""
    f = face_size / 2.0
    return np.array([[f, 0, f], [0, f, f], [0, 0, 1]], dtype=np.float64)


def face_pixel_grid(face_size: int, device="cpu") -> torch.Tensor:
    """Continuous pixel-CENTER coordinates for every face pixel, [S, S, 2]."""
    u = torch.arange(face_size, dtype=torch.float32, device=device) + 0.5
    v = torch.arange(face_size, dtype=torch.float32, device=device) + 0.5
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([grid_u, grid_v], dim=-1)  # [S, S, 2]  #算出每个面的中心坐标


def face_ray_directions(face_size: int, device="cpu") -> torch.Tensor:
    """Unit ray directions in the face camera frame, [S, S, 3]."""
    f = face_size / 2.0
    px = face_pixel_grid(face_size, device=device)
    x = (px[..., 0] - f) / f
    y = (px[..., 1] - f) / f   #算出像素相对于面中心的偏移坐标
    d = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return F.normalize(d, dim=-1)  #把面像素转换为面系的射线方向向量


def warp_pano_to_faces(
    pano: torch.Tensor,
    equirect: EquirectCamera,
    face_size: int,
    device="cuda",
) -> torch.Tensor:
    #核心的照片分幅函数
    """用反向映射把球面全景图切成 6 个立方体面:
    对每个面像素,依次做 ①针孔逆投影(像素→面系射线)→ 
    ②旋转(面系→全景系)→ ③球面投影(全景系射线→全景像素坐标)→
    ④归一化→ ⑤grid_sample 双线性采样,得到这一面的 GT 图。
    6 个面共享第①步的射线表,只是第②步的旋转矩阵不同,所以采样到全景图的不同区域
    """
    """Resample one panorama (H,W,3 float [0,1]) into the 6 cube faces.
    Returns [6, S, S, 3] in face order (front, back, left, right, up, down).
    """
    H, W = pano.shape[:2]  #全幅的像素
    pano = pano.to(device)  
    d_face = face_ray_directions(face_size, device=device)  # [S,S,3]算出针孔相机的射线方向向量
    #d_face: [S,S,3] 透视相机中心到透视分幅的射线方向向量，6个面共享
    faces = []
    for direction in FACE_DIRS.values():
        R = torch.from_numpy(face_rotation(direction)).float().to(device)  #6个面相对于全幅面的旋转矩阵
        # face -> pano ray: d_pano = R_face^T @ d_face.  With row vectors,
        # einsum('hwi,ij->hwj', d_face, R) equals R^T @ d_face; using R.T here
        # would compute R @ d_face and swap left/right + up/down faces.
        d_pano = torch.einsum("hwi,ij->hwj", d_face, R) #将面系的射线方向向量旋转到全景系，举例如果是r基准方向其实就需用旋转
        uv = equirect.project(d_pano)  # [S,S,2] source pixel coords  根据射线获取对应的全景像素坐标
        # normalized coords for grid_sample (align_corners=True)
        gx = 2.0 * uv[..., 0] / (W - 1) - 1.0  #归一化坐标
        gy = 2.0 * uv[..., 1] / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1,S,S,2]
        img = pano.permute(2, 0, 1).unsqueeze(0)          # [1,3,H,W]
        face = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0].permute(1, 2, 0)                              # [S,S,3]
        faces.append(face)
    return torch.stack(faces, dim=0)  #返回6个个面的GT图


def load_sparse(data_dir: str) -> dict:   #获取相机内参
    """Load a COLMAP sparse model, requiring EQUIRECTANGULAR cameras."""
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        sparse_dir = os.path.join(data_dir, "sparse")
    cameras = read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))   #返回一个字典组成的列表，每一个相机内参是一个字典
    images = read_images_bin(os.path.join(sparse_dir, "images.bin"))   #同样的外参字典组成的列表
    ids, xyz, rgb, err = read_points3d_bin(os.path.join(sparse_dir, "points3D.bin"))

    return dict(cameras=cameras, images=images,
                points_ids=ids, points_xyz=xyz, points_rgb=rgb,
                points_err=err)  #字典嵌套列表嵌套字典


# ---------------------------------------------------------------------------
# 数据层辅助（从原 train_pano.py 移入）：位姿、归一化、图像/分幅加载
# ---------------------------------------------------------------------------

def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def build_camtoworlds(images):
    """COLMAP qvec/tvec (world-to-camera) -> camera-to-world 4x4 matrices."""
    c2w = []
    for im in images:
        R = qvec_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64).reshape(3, 1)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:] = t
        c2w.append(np.linalg.inv(w2c))
    return np.stack(c2w, axis=0)


def normalization_params(c2w: np.ndarray):
    """只读：返回 normalize_world 使用的 (center, scale)，不修改输入。

    p_norm = (p_raw - center) * scale，因此还原绝对坐标：
    p_raw = p_norm / scale + center。若输入 sparse 本身就是 ENU/UTM 等
    绝对坐标，还原后的模型即为地理坐标模型。
    """
    centers = np.asarray(c2w[:, :3, 3], np.float64)  #所有相机位置
    center = centers.mean(axis=0)
    scale = 1.0 / (np.linalg.norm(centers - center, axis=1).mean() + 1e-8)
    return center, float(scale)


def normalize_world(c2w: np.ndarray, points: np.ndarray):
    """Center cameras at origin and scale so mean camera radius ~ 1."""
    center, scale = normalization_params(c2w)
    c2w = c2w.copy()
    c2w[:, :3, 3] = (c2w[:, :3, 3] - center) * scale
    points = (points - center) * scale #按照相机变化等同变化点的坐标，保持相对位置
    return c2w, points


def load_pano(path: str) -> torch.Tensor:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(img)


# 面缓存版本：warp 算法或像素约定一旦改动，必须递增此版本，
# 否则训练会静默加载旧版错误缓存（曾因旋转方向修复导致缓存失效）。
CACHE_VERSION = 3


def get_faces(pano_path: str, equirect: EquirectCamera, face_size: int,
              cache_dir: str, device: str) -> torch.Tensor:
    """Warped GT faces [6, S, S, 3] float32 [0,1] for one panorama."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            f"{os.path.splitext(os.path.basename(pano_path))[0]}_{face_size}"
            f"_v{CACHE_VERSION}.pt",
        )
        if os.path.exists(cache_path):
            faces_u8 = torch.load(cache_path, map_location="cpu")
            return faces_u8.float().to(device) / 255.0
    pano = load_pano(pano_path)   #加载全景影像
    faces = warp_pano_to_faces(pano, equirect, face_size, device=device)
    if cache_dir:
        torch.save((faces.detach().cpu() * 255).to(torch.uint8), cache_path)
    return faces


def build_sparse_depths(sparse: dict, images: list, face_rots=None,
                        S: int = 0, is_pano: bool = True,
                        c2w=None, pts=None) -> list:
      #1,||Xc|| vs Xc[2]必须与gsplat 渲染端深度输出方式一致。如果渲染输出的是 Z 深度，这里要改成 Xc[2]
     #2像素中心约定：+f（不是 +f-0.5 或 +f+0.5），和 face_pixel_grid 的 arange + 0.5 配套
     #位姿来源分支：c2w is not None 时用的是归一化场景下的位姿，对应归一化后的 points_xyz
     #性能瓶颈 三重循环 np.append 的二次复制
    """Per-image sparse depth supervision from COLMAP observations.

    对每张图（或全景的每个面），取该图可见的稀疏 3D 点，把相机系距离
    （与 gsplat expected-depth 渲染一致）记录在对应像素坐标上。
    
    返回:
      is_pano=True : 每张全景一个 list[6]，元素 (px [M,2], depth [M])
      is_pano=False: 每张图一个 (px [M,2], depth [M])
    """
    if pts is None:
        pts = sparse["points_xyz"]
    id_to_idx = {int(pid): i for i, pid in enumerate(sparse["points_ids"])}
    out = []
    for i, im in enumerate(images):
        if c2w is not None:
            # 归一化后的位姿：训练场景尺度与渲染 ED 深度一致
            w2c = np.linalg.inv(c2w[i])
            R = w2c[:3, :3]
            t = w2c[:3, 3]
        else:
            R = qvec_to_rotmat(im["qvec"])
            t = np.asarray(im["tvec"], np.float64)
        if is_pano:
            per_face = []
            for _ in range(len(face_rots)):
                per_face.append((np.zeros((0, 2)), np.zeros(0)))
            for k in range(len(im["point3D_ids"])):  #把每个3D点分到6个面，这不得老慢了
                pid = im["point3D_ids"][k]
                if pid < 0:
                    continue  #point3D_id == -1 表示该 2D 观测没有对应的三角化点
                Xc = R @ pts[id_to_idx[int(pid)]] + t  #每个3D点投影到6个面的面坐标系
                depth = float(np.linalg.norm(Xc))
                for fi, R_face in enumerate(face_rots):
                    Xf = R_face @ Xc  #每个3D点投影到6个面的像素坐标系
                    if Xf[2] <= 0:
                        continue
                    f = S / 2.0
                    u = f * Xf[0] / Xf[2] + f
                    v = f * Xf[1] / Xf[2] + f
                    if 0 <= u < S and 0 <= v < S:
                        px, d = per_face[fi]
                        per_face[fi] = (
                            np.append(px, [[u, v]], axis=0),
                            np.append(d, depth),
                        )
            out.append(per_face)
        else:  #单张图像直接服用colmap的2D观测
            px, d = [], []
            for k in range(len(im["point3D_ids"])):
                pid = im["point3D_ids"][k]
                if pid < 0:
                    continue
                Xc = R @ pts[id_to_idx[int(pid)]] + t
                if Xc[2] <= 0:
                    continue
                px.append(im["xys"][k])
                d.append(float(np.linalg.norm(Xc)))
            out.append((np.array(px, dtype=np.float64).reshape(-1, 2),
                        np.array(d, dtype=np.float64)))
    return out
