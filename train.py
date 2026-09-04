"""Train 3D Gaussian Splatting on equirectangular (360-degree) panoramas.

Pipeline (mirrors Spirula Studio's 360 route, implemented on top of gsplat):

  1. Read a COLMAP sparse reconstruction with EQUIRECTANGULAR cameras
     (sparse/0) -> camera poses + sparse points.
  2. Split every panorama into 6 pinhole faces of 90-degree FOV
     (warp_spherical_to_pinhole idea) and resample GT accordingly.
  3. Seed Gaussians from the sparse points (KNN-scale initialization).
  4. Train with gsplat's differentiable pinhole rasterizer + Adam,
     DefaultStrategy (Inria clone/split) or MCMCStrategy densification.
  5. Save a checkpoint (.pt) and a viewable PLY.

Run (smoke test):
  C:\\Users\\syk\\.conda\\envs\\gsplat\\python.exe train.py ^
      --data-dir D:\\gaussian_splatting\\spirula ^
      --max-images 4 --face-size 256 --steps 300 --preview-every 100
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from data import (
    CACHE_VERSION,
    FACE_DIRS,
    EquirectCamera,
    build_face_c2w,
    build_camtoworlds,
    build_sparse_depths,
    camera_K,
    camera_is_equirect,
    face_rotation,
    get_faces,
    load_pano,
    load_sparse,
    normalization_params,
    normalize_world,
    pinhole_intrinsics,
)

from fused_ssim import fused_ssim
from gsplat.exporter import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from sklearn.neighbors import NearestNeighbors

try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAVE_TORCHMETRICS = True
except Exception:
    HAVE_TORCHMETRICS = False

# 稠密深度预处理缓存版本：缓存为“对齐后的 fp16 视差 + 每面有效像素计数”。
# 改了对齐逻辑或换了 MoGe 输出（分辨率/尺度）时把这个版本号 +1 即可强制重建。
DENSE_CACHE_VERSION = 1


def load_config_file(path: str) -> dict:
    """加载 JSON/YAML 配置文件（键 = 参数名，连字符键会被归一化为下划线）。"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a JSON/YAML object")
    return {str(k).replace("-", "_"): v for k, v in cfg.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="",
                   help="JSON/YAML config file providing defaults (CLI args override)")
    p.add_argument("--data-dir", default=r"D:\gaussian_splatting\spirula",
                   help="dataset dir containing images/ and sparse/0")
    p.add_argument("--image-dir", default="",
                   help="override image folder (default <data-dir>/images)")
    p.add_argument("--out-dir", default=r"D:\gaussian_splatting\pano_gsplat\outputs\run")
    p.add_argument("--max-images", type=int, default=0,
                   help="limit to the first N panoramas (0 = all)")
    p.add_argument("--face-size", type=int, default=512,
                   help="edge length of each of the 6 pinhole faces")
    p.add_argument("--batch-size", type=int, default=1,
                   help="panoramas per training step (6 faces each)")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    p.add_argument("--antialiased", action="store_true",
                   help="use gsplat's antialiased rasterization (Mip-Splatting "
                        "compensation factor), closer to Spirula's mip primitive")
    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--init-opacity", type=float, default=None,
                   help="initial opacity (MCMC default 0.5, else 0.1)")
    p.add_argument("--init-scale", type=float, default=None,
                   help="initial scale multiplier (MCMC default 0.1, else 1.0)")
    p.add_argument("--scale-reg", type=float, default=None,
                   help="scale regularization weight (MCMC default 0.01)")
    p.add_argument("--opacity-reg", type=float, default=None,
                   help="opacity regularization weight (MCMC default 0.01)")
    p.add_argument("--means-lr", type=float, default=1.6e-4)
    p.add_argument("--scales-lr", type=float, default=5e-3)
    p.add_argument("--opacities-lr", type=float, default=5e-2)
    p.add_argument("--quats-lr", type=float, default=1e-3)
    p.add_argument("--sh0-lr", type=float, default=2.5e-3)
    p.add_argument("--shn-lr", type=float, default=2.5e-3 / 20)
    p.add_argument("--near-plane", type=float, default=0.01)
    p.add_argument("--far-plane", type=float, default=1e10)
    # --- 增密 / 效率---
    p.add_argument("--absgrad", action="store_true",
                   help="AbsGS: absolute 2D gradients for densification (default strategy)")
    p.add_argument("--max-gaussians", type=int, default=0,
                   help="cap Gaussian count for mcmc strategy "
                        "(maps to gsplat native cap_max; 0 = off, "
                        "default 1e6). ignored by default strategy")
    p.add_argument("--grow-grad2d", type=float, default=None,
                   help="densify threshold on 2D gradient (default: "
                        "0.0008 with --absgrad, else 0.0002); "
                        "lower = densify earlier/more aggressively")
    p.add_argument("--packed", action="store_true",
                   help="packed rasterization (less memory, slightly slower)")
    p.add_argument("--sparse-grad", action="store_true",
                   help="sparse gradients (requires --packed)")
    # --- 评测---
    p.add_argument("--test-every", type=int, default=0,
                   help="hold out every Nth image for validation (0 = none)")
    p.add_argument("--eval-every", type=int, default=0,
                   help="run validation every N steps (0 = only at end)")
    p.add_argument("--eval-max-images", type=int, default=20,
                   help="cap validation images per eval pass")
    p.add_argument("--save-eval-images", action="store_true",
                   help="save validation GT|render montages")
    # --- 粗到细---
    p.add_argument("--coarse-face-size", type=int, default=0,
                   help="first-stage face size (0 = disabled)")
    p.add_argument("--coarse-steps", type=int, default=0,
                   help="switch from coarse to full face size after this many steps")
    # --- bilagrid 曝光/白平衡校正---
    p.add_argument("--bilagrid", action="store_true",
                   help="per-view bilateral grid exposure/white-balance correction")
    p.add_argument("--bilagrid-shape", default="16,16,8",
                   help="grid X,Y,W (comma separated)")
    # --- PPISP 光度校正（Spirula 语义：每个 post-split 视角一个 slot）---
    p.add_argument("--ppisp", action="store_true",
                   help="PPISP per-view photometric correction "
                        "(exposure/vignetting/color/CRF)")
    p.add_argument("--ppisp-mode", choices=["per_view", "hybrid"],
                   default="per_view",
                   help="per_view=每个 post-split 视角独立 slot（Spirula）；"
                        "hybrid=曝光/颜色按全景共享、晕影/CRF 按面方向（全景）"
                        "或按物理相机（透视）")
    p.add_argument("--ppisp-lr", type=float, default=0.002,
                   help="PPISP main-parameter learning rate")
    p.add_argument("--ppisp-reg-scale", type=float, default=1.0,
                   help="weight for PPISP regularization loss")
    # --- 深度监督---
    p.add_argument("--depth-supervision-weight", type=float, default=0.0,
                   help="weight for sparse (COLMAP) depth supervision")
    p.add_argument("--depth-dir", default="",
                   help="MoGe 稠密深度目录（gen_depth_moge.py 输出，每张全景一个 .pt）")
    # --- 位姿优化 / 随机背景---
    p.add_argument("--pose-opt", action="store_true",
                   help="optimize per-view camera poses (rig-aware: pano faces share one pose)")
    p.add_argument("--pose-opt-lr", type=float, default=1e-5)
    p.add_argument("--random-bkgd", action="store_true",
                   help="random background color per step (discourage floaters)")
    p.add_argument("--face-cache", default="",
                   help="optional dir to cache warped faces (uint8 .pt per pano)")
    p.add_argument("--face-cache-cpu", action="store_true",
                   help="preload warped faces into CPU RAM once at startup "
                        "(requires --face-cache); skips per-step disk "
                        "load+pickle, trading ~1.6GB RAM for 354 panos at 512")
    p.add_argument("--save-every", type=int, default=0,
                   help="save ckpt+ply every N steps (0 = final only)")
    p.add_argument("--preview-every", type=int, default=0,
                   help="save GT|render montage every N steps")
    p.add_argument("--loss-log-every", type=int, default=1,
                   help="record loss history every N steps "
                        "(saved to loss_history.csv/json at end)")
    p.add_argument("--export-absolute", action="store_true",
                   help="训练结束额外输出还原到输入坐标系的 splat_absolute.ply"
                        "（若 sparse 为 ENU/UTM 等绝对坐标，即地理坐标模型）"
                        " + norm.json（归一化 center/scale）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dump-config", default="",
                   help="write the effective config to this path (JSON) and exit")

    # 两遍解析：先看 --config，把文件值设为默认值，再正式解析命令行传入的参数（CLI 覆盖文件）
    pre, _ = p.parse_known_args()
    if pre.config:
        cfg = load_config_file(pre.config)
        known = {a.dest for a in p._actions if a.dest != "help"}
        unknown = set(cfg) - known
        if unknown:
            print(f"[cfg] ignoring unknown config keys: {sorted(unknown)}")
        p.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    args = p.parse_args()

    if args.dump_config:
        import json
        d = {k: v for k, v in vars(args).items()
             if k not in ("config", "dump_config")}
        with open(args.dump_config, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        print(f"[cfg] dumped effective config to {args.dump_config}")
        raise SystemExit(0)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def knn_scale(points_np: np.ndarray, k: int = 4) -> np.ndarray:
    """Mean distance to k nearest neighbours (same init as gsplat/Spirula)."""
    model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(points_np)
    dists, _ = model.kneighbors(points_np)
    return dists[:, 1:].mean(axis=1)  # [N]


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

#点云初始化为高斯
def make_splats(points_xyz, points_rgb, cfg, device):
    """Seed Gaussians from the sparse point cloud (gsplat-style init)."""
    means = torch.from_numpy(points_xyz).float().to(device)
    rgbs = torch.from_numpy(points_rgb / 255.0).float().to(device)

    dist_avg = torch.from_numpy(knn_scale(points_xyz)).float().to(device)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    quats = torch.rand((len(means), 4), device=device)
    opacities = torch.logit(
        torch.full((len(means),), cfg.init_opacity, device=device)
    )

    sh_degree = cfg.sh_degree
    colors = torch.zeros((len(means), (sh_degree + 1) ** 2, 3), device=device)
    colors[:, 0, :] = rgb_to_sh(rgbs)

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
    })
    return splats


def make_optimizers(splats, cfg, scene_scale: float):
    """One Adam per parameter group; LR scaled like gsplat's simple_trainer."""
    BS = cfg.batch_size * len(FACE_DIRS)  # 每次训练光栅化针孔相机数量
    lrs = {
        "means": cfg.means_lr * scene_scale,   #场景尺度缩放让位置学习率适应场景尺度
        #归一化消除了"原始单位差异"(比如米 vs 厘米),但没消除"场景几何分布差异"(紧凑 vs 分散)
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh0": cfg.sh0_lr,
        "shN": cfg.shn_lr,
    }
    optimizers = {}
    optimizer_cls = torch.optim.SparseAdam if cfg.sparse_grad else torch.optim.Adam
    for name, lr in lrs.items():
        opts = dict(
            params=[{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),   #调整学习率倍数，因为6张要放大
        )
        if not cfg.sparse_grad:
            opts["fused"] = True  # SparseAdam 不支持 fused
        optimizers[name] = optimizer_cls(**opts)
        #batch_size=6(或更大)这一事实反映到 Adam 的所有自适应超参上,放大学习率,使训练更稳定
        # 让训练在 6 路并行监督下仍稳定收敛。
    return optimizers


def make_strategy(cfg, splats, optimizers, scene_scale):
    if cfg.strategy == "default":
        # AbsGS：用绝对 2D 梯度做增密，按 gsplat 文档 grow_grad2d 应提到 0.0008
        strategy = DefaultStrategy(
            absgrad=cfg.absgrad,
            grow_grad2d=(cfg.grow_grad2d
                         if cfg.grow_grad2d is not None
                         else (0.0008 if cfg.absgrad else 0.0002)),
        )
        strategy.check_sanity(splats, optimizers)  #这个函数是干啥的
        state = strategy.initialize_state(scene_scale=scene_scale)
        if cfg.max_gaussians > 0:
            print("[cfg] warning: --max-gaussians applies to mcmc strategy "
                  "(native cap_max); ignored with default strategy")
    else:
        if cfg.absgrad:
            print("[cfg] warning: --absgrad only applies to default strategy; "
                  "ignored with mcmc")
        strategy = MCMCStrategy(cap_max=cfg.max_gaussians or 1_000_000)
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state()
        if cfg.max_gaussians > 0:
            print(f"[cfg] max_gaussians={cfg.max_gaussians} "
                  f"(mcmc strategy -> cap_max)")
    return strategy, state


def save_ply(splats, path):
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),
        quats=splats["quats"].detach(),
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=path,
    )


def save_preview(gt, render, path, device):
    """Save a GT|render montage of the batch (faces or images) as one PNG."""
    canvas = torch.cat([gt, render], dim=2)  # [C, S, 2S, 3]
    canvas = canvas.reshape(-1, canvas.shape[2], 3).clamp(0, 1)
    imageio.imwrite(path, (canvas.cpu().numpy() * 255).astype(np.uint8))


def evaluate(cfg, splats, device, val_idx, images, pano_paths, is_pano,
             equirect, cam_by_id, face_c2w, Ks, S, step, out_dir):
    """Render held-out views and report PSNR/SSIM (+LPIPS if available)."""
    import json
    #加入ppisp和双边网格之类的评价函数不需要改吗
    if not val_idx:
        return
    if not HAVE_TORCHMETRICS:
        print("[eval] torchmetrics unavailable, skipping validation")
        return
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = None
    try:
        lpips_m = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    except Exception:
        lpips_m = None  # lpips 包未安装时跳过

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for vi in val_idx[: cfg.eval_max_images]:
            if is_pano:
                gt = get_faces(pano_paths[vi], equirect, S, cfg.face_cache, device)
                c2w_t = torch.from_numpy(face_c2w[vi]).float().to(device)
                ks_t = Ks.to(device)
                H, W = S, S
            else:
                img = load_pano(pano_paths[vi]).to(device)  #读取单张图像的函数
                cam_cur = cam_by_id[images[vi]["camera_id"]]
                K0 = torch.from_numpy(camera_K(cam_cur)).float().to(device)
                sx = img.shape[1] / cam_cur["width"]
                sy = img.shape[0] / cam_cur["height"]
                K0 = K0.clone()
                K0[0] *= sx
                K0[1] *= sy
                gt = img.unsqueeze(0)
                c2w_t = torch.from_numpy(face_c2w[vi]).float().to(device)
                ks_t = K0.unsqueeze(0)
                H, W = img.shape[0], img.shape[1]

            colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
            renders, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=torch.linalg.inv(c2w_t),
                Ks=ks_t,
                width=W,
                height=H,
                packed=False,
                absgrad=False,
                rasterize_mode="classic",
                camera_model="pinhole",
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
            )
            r = renders.permute(0, 3, 1, 2).clamp(0, 1)
            g = gt.permute(0, 3, 1, 2).clamp(0, 1)
            psnrs.append(float(psnr_m(r, g)))
            ssims.append(float(ssim_m(r, g)))
            if lpips_m is not None:
                lpipss.append(float(lpips_m(r, g)))
            if cfg.save_eval_images:
                canvas = torch.cat([gt, renders], dim=2)
                canvas = canvas.reshape(-1, canvas.shape[2], 3).clamp(0, 1)
                imageio.imwrite(
                    os.path.join(out_dir, f"eval_{step:06d}_v{vi:03d}.png"),
                    (canvas.cpu().numpy() * 255).astype(np.uint8),
                )

    res = {"step": step,
           "psnr": float(np.mean(psnrs)),
           "ssim": float(np.mean(ssims))}
    if lpipss:
        res["lpips"] = float(np.mean(lpipss))
    mpath = os.path.join(out_dir, "metrics.json")
    data = []
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(res)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[eval] step {step}: PSNR={res['psnr']:.2f} SSIM={res['ssim']:.4f}"
          + (f" LPIPS={res['lpips']:.4f}" if "lpips" in res else ""))


def main():
    cfg = parse_args()
    # MCMC needs its companion hyperparameters; without scale/opacity
    # regularization the position noise (lr * noise_lr * scale^2) enters a
    # positive feedback loop and splats fly away.  Apply the recommended
    # values unless the user overrode them.
    if cfg.strategy == "mcmc":
        if cfg.init_scale is None:
            cfg.init_scale = 0.1
        if cfg.init_opacity is None:
            cfg.init_opacity = 0.5
        if cfg.scale_reg is None:
            cfg.scale_reg = 0.01
        if cfg.opacity_reg is None:
            cfg.opacity_reg = 0.01
    else:
        if cfg.init_scale is None:
            cfg.init_scale = 1.0
        if cfg.init_opacity is None:
            cfg.init_opacity = 0.1
        if cfg.scale_reg is None:
            cfg.scale_reg = 0.0
        if cfg.opacity_reg is None:
            cfg.opacity_reg = 0.0
    print(f"[cfg] strategy={cfg.strategy} init_scale={cfg.init_scale} "
          f"init_opacity={cfg.init_opacity} scale_reg={cfg.scale_reg} "
          f"opacity_reg={cfg.opacity_reg}")
    if cfg.sparse_grad and not cfg.packed:
        print("[cfg] --sparse-grad requires --packed; enabling packed mode")
        cfg.packed = True
    if cfg.depth_dir and cfg.depth_supervision_weight <= 0.0:
        print("[cfg] --depth-dir given without weight; using 0.1")
        cfg.depth_supervision_weight = 0.1
    set_seed(cfg.seed)
    device = cfg.device
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---- 1. sparse model + panoramas -------------------------------------
    sparse = load_sparse(cfg.data_dir)  #加载稀疏重建文件，返回一个字典
    cameras = sparse["cameras"]  #取出字典
    images = sorted(sparse["images"], key=lambda im: im["name"])
    points_xyz = sparse["points_xyz"]
    points_rgb = sparse["points_rgb"]

    cam = cameras[0] #查看相机的id
    # 数据模式：全等距柱状 = 全景（每张切成 6 面）；全针孔族 = 透视（每张直接用）
    is_pano = all(camera_is_equirect(c) for c in cameras)     #bool判断是否所有相机都是全景相机
    if not is_pano:
        for c in cameras:
            if camera_is_equirect(c) or c["model"] == 5:  #不支持混合相机类型
                raise ValueError("mixed equirect/fisheye cameras not supported")
    #读取全景的参数焦距和图像宽高参数
    equirect = EquirectCamera(cam["width"], cam["height"]) if is_pano else None  #创建全景相机对象
    #设置全景相机一张分6张面，透视就是一张
    faces_per_image = 6 if is_pano else 1
    print(f"[data] {len(images)} images, camera model {cam['model']} "
          f"{cam['width']}x{cam['height']}, {len(points_xyz)} sparse points, "
          f"mode={'pano' if is_pano else 'perspective'}")  #打印稀疏数据的读取结果
    #这个基本用不上
    if cfg.max_images > 0:
        images = images[: cfg.max_images]  #限制读取相机数量是为了测试数据时更快，测试时降低训练轮数和分幅出图像的大小可以减少内存占用
    print(f"[data] using {len(images)} panoramas")

    image_dir = cfg.image_dir if cfg.image_dir else os.path.join(cfg.data_dir, "images")
    pano_paths = []
    for im in images:
        path = os.path.join(image_dir, im["name"])
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        pano_paths.append(path) #存储每个全景照片的路径

    # 训练/验证拆分（--test-every 每 N 张留 1 张做验证）  用全幅做验证还是用分出来的单张做验证更合理？
    if cfg.test_every > 0:
        val_idx = list(range(cfg.test_every - 1, len(images), cfg.test_every))  #按照全景图像做验证
        train_idx = [i for i in range(len(images)) if i not in set(val_idx)]
    else:
        train_idx = list(range(len(images)))  #训练集的原始图像编号
        val_idx = []  #验证集的原始图像编号
    if not train_idx:
        raise ValueError("empty training set; adjust --test-every / --max-images")
    #原始编号到训练数据的映射   
    grid_pos = {orig: pos for pos, orig in enumerate(train_idx)}  #调试看一下什么数据
    images_train = [images[i] for i in train_idx]   #只包含训练集的图像字典
    print(f"[data] train {len(train_idx)} / val {len(val_idx)}")

    # ---- 2. world normalization + face camera table ----------------------
    c2w = build_camtoworlds(images)  #相机转世界
    #归一化参数（center/scale），供 --export-absolute 还原绝对/输入坐标系
    norm_center, norm_scale = normalization_params(c2w)
    c2w, points_xyz = normalize_world(c2w, points_xyz)  #归一化坐标
    scene_scale = np.linalg.norm(c2w[:, :3, 3], axis=1).max() * 1.1  #创建整个场景的尺度，尺度为归一化后的相机到原点的距离的最大值的1.1倍，确保所有相机都在场景内
    print(f"[data] normalized scene scale = {scene_scale:.4f}")

    S = cfg.face_size  #设置分幅照片的尺寸
    cam_by_id = {c["id"]: c for c in cameras}   #根据相机id查找相机参数
    if is_pano:
        K = pinhole_intrinsics(S)  #设置虚拟针孔相机的内参
        # 6 个面共享同一套内参，复制 6 份，复制6份所以命名为Ks
        Ks = torch.from_numpy(np.repeat(K[None], len(FACE_DIRS), axis=0)).float()
        # [n_pano, 6, 4, 4]：为每一张全幅的相机构建 6 个面的相机转世界矩阵
        face_c2w = np.stack([
            np.stack([build_face_c2w(c2w[i], face_rotation(d))
                      for d in FACE_DIRS.values()])
            for i in range(len(c2w))
        ]) 
        print(f"[data] 6 faces per pano, face size {S}x{S}")
    else:
        # 透视相机：K=1，直接用相机自身内参（无切面）
        Ks = None
        face_c2w = c2w[:, None, :, :]  # [n, 1, 4, 4]
        print("[data] perspective: 1 image per camera, native resolution")

    # 稀疏深度监督（COLMAP 观测点；MoGe 稠密深度后续可复用同一接口）
    depth_train = None   #稀疏深度监督，不知道shape
    #将稀疏点云投影回照片，计算图像的深度
    if cfg.depth_supervision_weight > 0.0:
        face_rots = [face_rotation(d) for d in FACE_DIRS.values()] if is_pano else None
        #取出面的朝向矩阵
        depth_train = build_sparse_depths(
            sparse, images_train, face_rots, S, is_pano,
            c2w=c2w[train_idx], pts=points_xyz,
        )  

    # MoGe 稠密深度：加载并按面与稀疏深度做中位比例对齐（归一化场景尺度）
    dense_aligned = None  #对齐后的稠密深度图
    #MoGe（单目深度估计网络）输出的稠密深度图是无尺度的相对深度——需要归一化场景尺度，
    # 用colmap的系数点当作锚点，给稠密深度乘比例系数让两者在重合像素上对齐
    if cfg.depth_dir and is_pano:
        # 稠密深度预处理缓存：首次构建时把原始 MoGe 深度与 COLMAP 稀疏深度
        # 对齐（中位比例），转成 fp16 视差（0=无效）并记录每面有效像素数，
        # 之后启动直接读缓存，免去重复加载 + CPU grid_sample 对齐 + 每步换算。
        cache_dir = os.path.join(cfg.depth_dir, f"aligned_v{DENSE_CACHE_VERSION}")
        cache_files = [
            os.path.join(cache_dir, os.path.splitext(im["name"])[0] + ".pt")
            for im in images_train
        ]
        dense_aligned = []  #每张全景 [6,S,S] fp16 视差（0=无效，已对齐场景尺度）
        dense_counts = []   #每张 [6] int32：各面有效像素数（原始 S 分辨率）
        if all(os.path.exists(cp) for cp in cache_files):
            print(f"[data] loading dense-depth cache: {cache_dir}")
            for cp in cache_files:
                obj = torch.load(cp, map_location="cpu")
                dense_aligned.append(obj["disp"])
                dense_counts.append(obj["count"])
        else:
            print(f"[data] building dense-depth cache -> {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            dense_raw = []    #原始稠密深度图
            for im in images_train:
                dp = os.path.join(cfg.depth_dir,
                                  os.path.splitext(im["name"])[0] + ".pt")  #每张全景一个pt文件
                                    #里面是wrap的6个面的稠密深度图
                if not os.path.exists(dp):
                    raise FileNotFoundError(dp)
                dense_raw.append(torch.load(dp, map_location="cpu").float())  # [6,S,S]
            dense_aligned_f32 = []  #对齐后的稠密深度图（fp32，构建缓存时临时持有）
            all_ratios = []
            for pos in range(len(images_train)):
                aligned = dense_raw[pos].clone()  #克隆一份深度图，为什么评价不加深度图?
                dense_raw[pos] = None  #逐张释放原始深度，降低构建缓存时的 CPU 峰值
                ratios = []
                for f in range(6):
                    px_f, d_f = depth_train[pos][f]  ## 该面的稀疏监督深度图 (px[M,2], d[M])
                    if len(px_f) < 5:
                        continue
                    grid = torch.from_numpy(np.stack(
                        [px_f[:, 0] / (S - 1) * 2 - 1,
                         px_f[:, 1] / (S - 1) * 2 - 1], -1)
                    ).float().unsqueeze(0).unsqueeze(0)  #稀疏点像素坐标归一化到[-1,1]
                    samp = F.grid_sample(
                        aligned[f][None, None], grid, align_corners=True
                    )[0, 0, 0, :]  # 用稀疏点像素位置去稠密深度图上采样
                    d_gt_t = torch.from_numpy(d_f).float()
                    m = (samp > 0) & (d_gt_t > 0)  #统计有效样本数量
                    if m.sum() >= 5:  #有效样本大于5才使用中位数计算
                        r = d_gt_t[m].median() / samp[m].median()
                        ratios.append(float(r))
                        aligned[f] = aligned[f] * r
                for f in range(6):
                    if len(depth_train[pos][f][0]) < 5:
                        ratios.append(float("nan"))   #小于5的先设置为nan
                all_ratios.extend(ratios)
                dense_aligned_f32.append(aligned)
            del dense_raw
            ratios = torch.tensor([r for r in all_ratios if np.isfinite(r)])
            if len(ratios) == 0:
                raise RuntimeError("no sparse depth overlap for dense alignment")
            global_r = float(ratios.median())  #都做完之后统计全局比例，为之前nan的设置全局比例
            for pos in range(len(dense_aligned_f32)):  #对于那些没有有效样本的面，用全局比例对齐
                for f in range(6):
                    if len(depth_train[pos][f][0]) < 5:
                        dense_aligned_f32[pos][f] = dense_aligned_f32[pos][f] * global_r
            print(f"[data] dense depth aligned (global scale {global_r:.4f})")
            # 对齐后的深度转 fp16 视差写缓存：深度>0 且有限 -> 1/d，否则 0
            for pos, aligned in enumerate(dense_aligned_f32):
                valid = (aligned > 0) & torch.isfinite(aligned)
                count = valid.float().sum(dim=(1, 2)).to(torch.int32)
                disp = torch.where(
                    valid, 1.0 / aligned, torch.zeros_like(aligned)
                ).half()
                torch.save({"disp": disp, "count": count}, cache_files[pos])
                dense_aligned_f32[pos] = None
                dense_aligned.append(disp)
                dense_counts.append(count)
            del dense_aligned_f32

    # 位姿优化：全景按张（6 面共享）、透视按图；默认零初始化，靠训练学小修正
    pose_adjust = None  #位姿修正，每张全景9个参数6个旋转3个平移
    pose_optimizer = None  #位姿优化器，用于训练时更新相机位姿
    if cfg.pose_opt:
        from pose import CameraOptModule
        pose_adjust = CameraOptModule(len(train_idx)).to(device)
        pose_adjust.zero_init()
        pose_optimizer = torch.optim.Adam(
            pose_adjust.parameters(),
            lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size * faces_per_image),
            #这里区分了全景和透视相机的优化率
            weight_decay=1e-6,
        )
    

    # bilagrid 曝光/白平衡校正（全景按面、透视按图，每个训练视角一个网格）  全幅用得上这个吗？
    bil_grids = None  #每个全幅影像的双边网格，用于曝光/白平衡校正
    bil_optimizer = None  #曝光/白平衡校正优化器，用于训练时更新曝光/白平衡参数
    if cfg.bilagrid:
        from lib_bilagrid import BilateralGrid
        gx, gy, gw = (int(v) for v in cfg.bilagrid_shape.split(","))
        #解析预设的网格形状 gx,gy,gw
        # gx: 网格宽度，gy: 网格高度，gw: 颜色引导分辨率 每一小格12个参数9个颜色变化系数3个偏执量
        n_grids = len(train_idx) * faces_per_image #每张分幅图像一个双边网格
        bil_grids = BilateralGrid(n_grids, grid_X=gx, grid_Y=gy, grid_W=gw).to(device)
        BS_grid = cfg.batch_size * faces_per_image
        bil_optimizer = torch.optim.Adam(
            bil_grids.parameters(), lr=2e-3 * math.sqrt(BS_grid), eps=1e-15
        )  #学习率随batch_size变化

    # PPISP 光度校正（无控制器，避免每视角一个 CNN 造成显存爆炸；
    # per_view=Spirula 语义，hybrid=物理优先绑定）
    ppisp = None   #光度校正模型，用于光度矫正正
    ppisp_optimizers = []  #多个校正优化器，因为不同的参数需要不同的优化率
    ppisp_schedulers = []
    if cfg.ppisp:
        from ppisp import PPISP, PPISPConfig
        if cfg.ppisp_mode == "hybrid":
            n_frames = len(train_idx)   #帧数
            n_cameras = faces_per_image if is_pano else 1  #相机数，全景按面、透视按图
            print(f"[ppisp] hybrid binding: num_frames={n_frames} "
                  f"num_cameras={n_cameras}")
            #让每一帧的曝光和白平衡一致，6个面的渐晕和CRF按照方向独立学习
        else:
            n_frames = n_cameras = len(train_idx) * faces_per_image  #另一种模式每个训练视角一个网格
            print(f"[ppisp] {n_frames} per-view slots "
                  f"(Spirula style, controller disabled)")
        ppisp_cfg = PPISPConfig(use_controller=False)  #不使用控制器，也就是在训练器光度矫正，并未使用新视角自动曝光能力，这样对吗？
        ppisp = PPISP(num_cameras=n_cameras, num_frames=n_frames,
                      config=ppisp_cfg).to(device)
        ppisp_optimizers = ppisp.create_optimizers()
        ppisp_schedulers = ppisp.create_schedulers(ppisp_optimizers, cfg.steps)

    # 粗到细：先用低分辨率面热身，再切到完整分辨率
    Ks_coarse = None   #计算粗阶段相机内参
    if is_pano and cfg.coarse_face_size > 0:   #只对全景生效
        Kc = pinhole_intrinsics(cfg.coarse_face_size)
        Ks_coarse = torch.from_numpy(
            np.repeat(Kc[None], len(FACE_DIRS), axis=0)
        ).float()   #粗分辨率内参
        print(f"[data] coarse-to-fine: {cfg.coarse_face_size} -> {S} "
              f"at step {cfg.coarse_steps}")

    # ---- 2b. 启动期预计算（把训练循环里每步重复的固定操作挪到循环外）----
    # GPU 常驻的位姿/内参：循环内不再 from_numpy + float + to(device)
    c2w_t = torch.from_numpy(c2w).float().to(device)          # [N,4,4]
    if is_pano:
        face_c2w_t = torch.from_numpy(face_c2w).float().to(device)  # [N,6,4,4]
    else:
        face_c2w_t = c2w_t[:, None, :, :].contiguous()        # [N,1,4,4]
    Ks = Ks.float().to(device) if is_pano else Ks             # [6,3,3] 预转 GPU
    if Ks_coarse is not None:
        Ks_coarse = Ks_coarse.to(device)

    # CPU 常驻面缓存：启动时把所有全景的 warp 面一次性读进 RAM
    # （uint8 原样保存，约 1.6GB/354 张@512），训练循环里只做内存索引，
    # 跳过每步的磁盘 torch.load + pickle 反序列化。
    faces_cpu = {}   # {face_size: list of [6,S,S,3] uint8 CPU tensors}
    if cfg.face_cache and cfg.face_cache_cpu and is_pano:
        sizes_to_preload = {S}
        if cfg.coarse_face_size > 0:
            sizes_to_preload.add(cfg.coarse_face_size)
        for sz in sorted(sizes_to_preload):
            arr = []
            for pano_path in pano_paths:
                cache_path = os.path.join(
                    cfg.face_cache,
                    f"{os.path.splitext(os.path.basename(pano_path))[0]}"
                    f"_{sz}_v{CACHE_VERSION}.pt",
                )
                if not os.path.exists(cache_path):
                    # 缺缓存：现切一张（GPU warp），顺带落盘
                    get_faces(pano_path, equirect, sz, cfg.face_cache, device)
                arr.append(torch.load(cache_path, map_location="cpu"))
            faces_cpu[sz] = arr
            print(f"[data] preloaded {len(arr)} panos @ face {sz} "
                  f"into CPU RAM "
                  f"({sum(t.numel() for t in arr) * 1 / 1024**3:.2f} GiB)")

    # ---- 3. splats + optimizer + strategy --------------------------------
    splats = make_splats(points_xyz, points_rgb, cfg, device)  # 点云注册为高斯椭球存到gpu
    optimizers = make_optimizers(splats, cfg, scene_scale)
    print(f"[model] initialized {len(splats['means'])} Gaussians")
    strategy, strategy_state = make_strategy(cfg, splats, optimizers, scene_scale)

    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps)
    )  #位置学习率衰减，训练越后期高斯移动幅度越小

    # ---- 4. training loop -------------------------------------------------
    B = cfg.batch_size  #每一步几张全景影像
    pbar_steps = cfg.steps
    # per-epoch 确定性抽样：预生成打乱后的训练索引池，顺序消费；
    # 池耗尽时重新打乱（每 epoch 每张全景恰好出现一次），
    # 替代 random.choice 的放回抽样，避免同一张被重复监督的浪费。
    epoch_pool = list(train_idx)
    random.shuffle(epoch_pool)
    pool_pos = 0
    tic = time.time()
    loss_log = []  #训练过程 loss 历史（step/loss/l1/ssim/depth_loss/gs/mem/time）
    for step in range(pbar_steps):
        depthloss = None  #预初始化：仅深度监督启用时才会被赋值，供 loss 日志读取
        n_used = 0
        idxs = []
        for _ in range(B):
            if pool_pos >= len(epoch_pool):
                random.shuffle(epoch_pool)   #一个 epoch 结束，重新打乱
                pool_pos = 0
            idxs.append(epoch_pool[pool_pos])
            pool_pos += 1
        # 粗到细：切换当前面的尺寸与内参
        if is_pano and Ks_coarse is not None and step < cfg.coarse_steps:  #粗阶段
            S_cur, Ks_cur = cfg.coarse_face_size, Ks_coarse
        else:
            S_cur, Ks_cur = S, Ks
        if step == cfg.coarse_steps and is_pano and Ks_coarse is not None:
            print(f"[train] coarse-to-fine: switched to face size {S}")
        gts = []  #[B·6, S_cur, S_cur, 3]
        c2ws = [] #[B·6, 4, 4]
        Ks_list = []   #每个面的内参列表
        for idx in idxs:  #为每一张全幅影像分幅出6个面（或透视图直接用），一般idx==1，除非电脑性能较高
            if is_pano:
                if faces_cpu:  #CPU 常驻：只做内存索引 + 转 float 上 GPU
                    faces_u8 = faces_cpu[S_cur][idx]
                    gts.append((faces_u8.float().to(device) / 255.0))
                else:  #原来的磁盘缓存路径
                    gts.append(get_faces(pano_paths[idx], equirect, S_cur,
                                         cfg.face_cache, device))   #分幅出6个面的图像并存到gpu
                if pose_adjust is not None:
                    # rig 感知位姿优化：先修正全景位姿，再派生 6 个面
                    from pose import adjust_pano_pose
                    pano_c2w_t = c2w_t[idx]  #取当前全景的位姿（已预转 GPU）
                    adj = adjust_pano_pose(pose_adjust, pano_c2w_t,
                                           grid_pos[idx])[0]  # [4,4]  #修正后的全景位姿
                    faces_t = []  #从修正后的位姿计算6个面的位姿
                    for d in FACE_DIRS.values():  
                        Rf = torch.from_numpy(face_rotation(d)).float().to(device)
                        out = torch.eye(4, device=device)
                        out[:3, :3] = adj[:3, :3] @ Rf.T
                        out[:3, 3] = adj[:3, 3]
                        faces_t.append(out)
                    c2ws.append(torch.stack(faces_t))
                else:
                    c2ws.append(face_c2w_t[idx])
                Ks_list.append(Ks_cur)  #每个面的内参（已预转 GPU）
                H, W = S_cur, S_cur #照片尺寸
            else: #针孔相机
                img = load_pano(pano_paths[idx]).to(device)     # [H, W, 3]
                cam_cur = cam_by_id[images[idx]["camera_id"]]
                K0 = torch.from_numpy(camera_K(cam_cur)).float().to(device)
                sx = img.shape[1] / cam_cur["width"]   # 图片实际尺寸与标定尺寸的缩放
                sy = img.shape[0] / cam_cur["height"]
                K0 = K0.clone()
                K0[0] *= sx
                K0[1] *= sy
                gts.append(img.unsqueeze(0))
                if pose_adjust is not None:
                    ids = torch.tensor([grid_pos[idx]], device=device)
                    c2w_adj = pose_adjust(c2w_t[idx].unsqueeze(0), ids)[0]
                    c2ws.append(c2w_adj.unsqueeze(0))
                else:
                    c2ws.append(face_c2w_t[idx])
                Ks_list.append(K0.unsqueeze(0))
                H, W = img.shape[0], img.shape[1]
        gt = torch.cat(gts, dim=0)                       # [B*faces, H, W, 3] 分幅出的真实图像（已在 GPU）
        camtoworlds = torch.cat(c2ws, dim=0)             # [B*faces, 4, 4] 分幅后的相机位姿
        Ks_b = torch.cat(Ks_list, dim=0)                 # [B*faces, 3, 3] 分幅后的内参
        # bilagrid 网格索引：全景按面、透视按图（每个训练视角独立网格）
        if bil_grids is not None:
            grid_ids = []   #双边网格的编号
            for idx in idxs:  #为相机视角加上训练的双边网格
                pos = grid_pos[idx]
                for f in range(faces_per_image):
                    grid_ids.append(pos * faces_per_image + f)  # pos*6+f f=0-5·
            grid_ids = torch.tensor(grid_ids, device=device)

        sh_degree = min(step // 1000, cfg.sh_degree)   #3000步以下用低阶分量
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        render_mode = "RGB+ED" if cfg.depth_supervision_weight > 0.0 else "RGB"
        #renders[B·6,S,S,4]颜色/深度  alphas[B·6,S,S,1]累计不透明度 info光栅化中间信息
        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks_b,
            width=W,
            height=H,
            packed=cfg.packed,
            absgrad=isinstance(strategy, DefaultStrategy) and strategy.absgrad,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
            camera_model="pinhole",
            sh_degree=sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            sparse_grad=cfg.sparse_grad,
            render_mode=render_mode,
        )
        if render_mode == "RGB+ED":
            colors_rgb = renders[..., 0:3]
            depths_ed = renders[..., 3:4]
        else:
            colors_rgb = renders
            depths_ed = None

        # PPISP 光度校正（Spirula 默认先于 bilagrid）
        if ppisp is not None:
            cam_ids, frm_ids = [], []
            for idx in idxs:
                pos = grid_pos[idx]  #网格的编号
                for f in range(faces_per_image):
                    if cfg.ppisp_mode == "hybrid":
                        cam_ids.append(f if is_pano else 0)  #每面一个相机  用来存渐晕和CRF
                        frm_ids.append(pos)  #每个网格一个帧  用来存曝光和白平衡
                    else:  #另一个per_view模式一张图为一帧一个相机
                        cam_ids.append(pos * faces_per_image + f)
                        frm_ids.append(pos * faces_per_image + f)
            rgb_corr = []
            for ci in range(colors_rgb.shape[0]):
                rgb_corr.append(ppisp(
                    colors_rgb[ci], pixel_coords=None, resolution=(W, H),
                    camera_idx=cam_ids[ci], frame_idx=frm_ids[ci]))  #校正后的rgb
            colors_rgb = torch.stack(rgb_corr)  #覆盖原值

        # bilagrid 曝光/白平衡：渲染结果按像素坐标和灰度切网格做仿射变换
        if bil_grids is not None:
            from lib_bilagrid import slice as bilagrid_slice
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(H, device=device) + 0.5) / H,
                (torch.arange(W, device=device) + 0.5) / W,
                indexing="ij",
            )  #像素归一化网格
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            grid_xy = grid_xy.expand(colors_rgb.shape[0], -1, -1, -1) #广播到 12 个渲染面
            colors_rgb = bilagrid_slice(
                bil_grids, grid_xy, colors_rgb, grid_ids.unsqueeze(-1)
            )["rgb"]  #仿射变换
        # 随机背景：每步随机颜色，抑制透明漂浮物（类似 Spirula 的背景噪声热身）
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors_rgb = colors_rgb + bkgd * (1.0 - alphas)

        strategy.step_pre_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=info,
        )  #反向前钩子，记录当前步的渲染结果，为后续是否进行增值做准备，
        #现在只有这一步训练的渲染信息，哪些高斯被看到投影半径之类的，还没有梯度

        l1 = F.l1_loss(colors_rgb, gt)
        ssim = fused_ssim(colors_rgb.permute(0, 3, 1, 2),
                          gt.permute(0, 3, 1, 2), padding="valid")
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
        # 稀疏深度监督：渲染期望深度 ED 在观测像素处采样，做视差 L1
        if depths_ed is not None:
            depthloss = torch.tensor(0.0, device=device)  #深度损失
            n_used = 0   #本次训练使用面个数
            for j, idx in enumerate(idxs):
                pos = grid_pos[idx]
                for f in range(faces_per_image):
                    ci = j * faces_per_image + f   #当前渲染面的编号
                    # 稠密深度路径（MoGe）：逐像素视差 L1，深度>0 视为有效
                    if dense_aligned is not None:  #稠密路径：缓存已是 fp16 视差（0=无效）
                        disp_map = dense_aligned[pos][f].to(device)  #第 pos 张全景第 f 个面的对齐视差（S×S，fp16）
                        if disp_map.shape[0] != S_cur:  #缓存分辨率与当前训练分辨率不一致（含粗阶段）时缩放
                            disp_map = F.interpolate(
                                disp_map[None, None], size=(S_cur, S_cur),
                                mode="bilinear", align_corners=False,
                            )[0, 0]  #双线性插值获取稠密视差
                        if int(dense_counts[pos][f]) < 100:  #有效像素不足100个，跳过（计数已在缓存预计算）
                            continue
                        valid = disp_map > 0  #有效像素（视差>0，无效像素缓存里存的是 0）
                        disp_gt = disp_map[valid].float()  #GT 视差（缓存已做过 1/深度）
                        disp_rend = 1.0 / depths_ed[ci, ..., 0][valid]  #渲染视差 = 1/深度
                        depthloss = depthloss + F.l1_loss(disp_rend, disp_gt) * scene_scale
                        #使用视差的原因是更符合感知和 SfM 误差模型，防止远处损失被稀释近处被夸大
                        n_used += 1
                        continue
                    if is_pano:  #稀疏深度监督
                        px_f, d_f = depth_train[pos][f]  #  稀疏监督数据 px_f[M, 2]该面稀疏点的像素坐标（u, v）
                        #, d_f[M]对应的相机系欧氏距离（COLMAP 真值） M该面可见的稀疏点数
                    else:
                        px_f, d_f = depth_train[pos]
                    if len(px_f) < 5:
                        continue
                    if is_pano and S_cur != S:  #粗分辨率阶段走这里
                        px_f = (S_cur / 2.0) + (px_f - S / 2.0) * (S_cur / S)
                    grid_d = torch.from_numpy(np.stack(
                        [px_f[:, 0] / (W - 1) * 2 - 1,
                         px_f[:, 1] / (H - 1) * 2 - 1], -1)  #归一化坐标到grid_sample格式
                    ).float().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,M,2]
                    d_gt = torch.from_numpy(d_f).float().to(device)
                   #用稀疏点的像素位置去渲染深度图里采样
                    d_rend = F.grid_sample(
                        depths_ed[ci:ci + 1].permute(0, 3, 1, 2), grid_d,
                        align_corners=True,
                    )[0, 0, 0, :]  # [M]
                    disp = torch.where(
                        d_rend > 0, 1.0 / d_rend, torch.zeros_like(d_rend)
                    )  #渲染视差 = 1/深度
                    depthloss = depthloss + F.l1_loss(disp, 1.0 / d_gt) * scene_scale
                    n_used += 1
            if n_used > 0:
                loss = loss + (depthloss / n_used) * cfg.depth_supervision_weight
        if cfg.scale_reg > 0.0:
            loss = loss + cfg.scale_reg * torch.exp(splats["scales"]).mean()
        if cfg.opacity_reg > 0.0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if ppisp is not None:
            loss = loss + cfg.ppisp_reg_scale * ppisp.get_regularization_loss()
        loss.backward()
        # sparse-grad：把稠密梯度转成稀疏（只保留被光栅化的高斯）
        if cfg.sparse_grad:
            gaussian_ids = info["gaussian_ids"]
            for k in splats.keys():
                grad = splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=grad[gaussian_ids],
                    size=splats[k].size(),
                    is_coalesced=len(Ks_b) == 1,
                )
        #现在出现梯度，判断是否需要增值，同时更新优化器状态
        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info, packed=cfg.packed,
            )
        else:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
                lr=means_scheduler.get_last_lr()[0],
            )  
        
        for opt in optimizers.values():
            opt.step() #更新参数 
            opt.zero_grad(set_to_none=True) #清空梯度
        if bil_optimizer is not None:
            bil_optimizer.step()
            bil_optimizer.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        if ppisp_optimizers:
            for opt in ppisp_optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in ppisp_schedulers:
                sched.step()
        means_scheduler.step() #更新学习率

        if step % 50 == 0 or step == pbar_steps - 1:  #每五十次播报一次
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[train] step {step:6d} loss {loss.item():.4f} "
                  f"l1 {l1.item():.4f} ssim {ssim.item():.4f} "
                  f"GS {len(splats['means'])} mem {mem:.2f} GiB "
                  f"time {time.time()-tic:.0f}s")

        if step % cfg.loss_log_every == 0 or step == pbar_steps - 1:
            dl = depthloss.item() if (depthloss is not None and n_used > 0) else 0.0
            loss_log.append({
                "step": step,
                "loss": float(loss.item()),
                "l1": float(l1.item()),
                "ssim": float(ssim.item()),
                "depth_loss": float(dl),
                "gs": int(len(splats["means"])),
                "mem_gib": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
                "time_s": float(time.time() - tic),
            })

        if cfg.preview_every and (step % cfg.preview_every == 0
                                  or step == pbar_steps - 1):
            save_preview(gt, colors_rgb.detach(),
                         os.path.join(cfg.out_dir, f"preview_{step:06d}.png"),
                         device)
        if cfg.save_every and (step % cfg.save_every == 0
                               or step == pbar_steps - 1):
            torch.save(splats.state_dict(),
                       os.path.join(cfg.out_dir, f"ckpt_{step:06d}.pt"))
            save_ply(splats, os.path.join(cfg.out_dir, f"splat_{step:06d}.ply"))
        # 验证集评测（功能 1）
        if val_idx and ((cfg.eval_every and step > 0 and step % cfg.eval_every == 0)
                        or step == pbar_steps - 1):
            evaluate(cfg, splats, device, val_idx, images, pano_paths, is_pano,
                     equirect, cam_by_id, face_c2w, Ks, S, step, cfg.out_dir)

    # ---- 5. final save -----------------------------------------------------
    torch.save(splats.state_dict(), os.path.join(cfg.out_dir, "ckpt_final.pt"))
    save_ply(splats, os.path.join(cfg.out_dir, "splat.ply"))
    if ppisp is not None:
        torch.save(ppisp.state_dict(),
                   os.path.join(cfg.out_dir, "ppisp_final.pt"))
    if cfg.export_absolute:
        #还原到输入坐标系：位置除以 scale 再加 center；高斯线性尺度放大 1/scale
        #（scales 是 log 空间参数，故 log 域加 -log(scale)）。无旋转，四元数不变。
        import json as _json
        norm_info = {
            "center": [float(x) for x in norm_center],
            "scale": float(norm_scale),
            "note": "p_raw = p_norm / scale + center；若输入 sparse 为 "
                    "ENU/UTM 等绝对坐标，splat_absolute.ply 即地理坐标模型",
        }
        norm_path = os.path.join(cfg.out_dir, "norm.json")
        with open(norm_path, "w", encoding="utf-8") as f:
            _json.dump(norm_info, f, indent=2)
        means_abs = splats["means"].detach() / norm_scale + torch.from_numpy(
            norm_center
        ).float().to(device)
        scales_abs = splats["scales"].detach() - math.log(norm_scale)
        abs_path = os.path.join(cfg.out_dir, "splat_absolute.ply")
        export_splats(
            means=means_abs,
            scales=scales_abs,
            quats=splats["quats"].detach(),
            opacities=splats["opacities"].detach(),
            sh0=splats["sh0"].detach(),
            shN=splats["shN"].detach(),
            format="ply",
            save_to=abs_path,
        )
        print(f"[done] absolute-coordinate model → {abs_path}")
        print(f"[done] normalization params → {norm_path}")
    if loss_log:  #训练过程 loss 历史落盘（CSV 方便 Excel/画图，JSON 方便程序读取）
        import csv
        import json
        hist_csv = os.path.join(cfg.out_dir, "loss_history.csv")
        with open(hist_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(loss_log[0].keys()))
            w.writeheader()
            w.writerows(loss_log)
        hist_json = os.path.join(cfg.out_dir, "loss_history.json")
        with open(hist_json, "w", encoding="utf-8") as f:
            json.dump(loss_log, f, indent=2)
        print(f"[done] loss history → {hist_csv} / {hist_json} "
              f"({len(loss_log)} entries)")
    print(f"[done] outputs in {cfg.out_dir}")


if __name__ == "__main__":
    main()
