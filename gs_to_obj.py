# -*- coding: utf-8 -*-
"""3DGS 模型 → OBJ 网格（路线 A：多视角深度渲染 + TSDF 融合 + Marching Cubes）。

原理（与训练器同坐标系，不改训练）：
  1. 从 splat.ply 读高斯（复用 clean_floaters 的 PLY 读取）；
  2. 从 COLMAP sparse 重建渲染相机：
       - 全景（EQUIRECTANGULAR model 17）：每张图切成 6 个 90° 针孔面，
         与 train.py 的 warp 方案一致（6 面共享同一全景位姿）；
       - 透视（PINHOLE 族）：直接用原相机位姿 + 内参；
     --normalize ours   : data.normalize_world（本项目 train.py 的归一化）；
     --normalize gsplat : 复现 gsplat simple_trainer 的归一化
                          （similarity_from_cameras + align_principal_axes，
                          源码来自 gsplat/examples/datasets/normalize.py，
                          Apache-2.0），用于 gsplat 官方训练器产出的模型；
  3. gsplat 光栅化批量渲染 RGB + 期望深度 + alpha（render_mode="RGB+ED"，
     无需自写 CUDA）；
  4. 深度图反投影为世界坐标点（alpha 掩膜滤掉不可靠像素）；
  5. TSDF 体素融合（numpy 向量化自实现，不依赖 open3d；流式逐视图更新，
     不落盘大点云）；
  6. scikit-image Marching Cubes 提零等值面 → trimesh 导出 OBJ（带顶点色）
     + 可选彩色点云 PLY / 深度图（调试）。

用法示例：
  # 本项目 train.py 训出的模型（归一化一致）
  python gs_to_obj.py --ply outputs/run/splat_clean.ply \
      --data-dir D:\gaussian_splatting\spirula --out-dir outputs/run/mesh

  # gsplat simple_trainer 训出的模型（garden 等，factor=2 匹配 images_2）
  python gs_to_obj.py --ply ...\splat.ply --data-dir ...\garden \
      --normalize gsplat --factor 2 --render-size 800

  # 合成数据自检
  python gs_to_obj.py --selftest

主要函数：
  load_splats        : PLY → 高斯参数张量 + sh_degree
  build_cameras      : sparse → 渲染相机列表（全景切面 / 透视，归一化）
  render_views       : 批量渲染 RGB+ED+alpha（流式生成器）
  unproject_depth    : 单视图深度 → 世界坐标点云
  tsdf_fusion        : 多视图深度 → TSDF 体素场
  marching_cubes_mesh: TSDF → trimesh 网格
  gs_to_obj          : 主流程（库接口）
"""

import argparse
import os

import numpy as np
import torch
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

import data as data_mod
from clean_floaters import load_ply

try:
    from gsplat.rendering import rasterization
except Exception:  # 老版本 gsplat 把接口放在顶层
    from gsplat import rasterization


FACE_DIRS = data_mod.FACE_DIRS


# ---------------------------------------------------------------------------
# 1. 加载高斯（PLY → 张量）
# ---------------------------------------------------------------------------


def load_splats(ply_path, device="cuda:0"):
    """读取 gsplat 导出的 splat.ply，返回渲染所需参数。

    PLY 里 opacity/scales 存的是原始 logit/log 参数（与训练一致，可直接喂
    光栅化器）；f_dc/f_rest 组装成 (N, (deg+1)^2, 3) 的 SH 系数。
    忽略法线等额外属性。
    """
    d, props, _, _ = load_ply(ply_path)
    names = [a for a, _ in props]
    n = len(d["x"])
    means = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    scales = np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1).astype(np.float32)
    quats = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1).astype(
        np.float32
    )
    opacities = np.asarray(d["opacity"], np.float32)
    f_dc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1).astype(np.float32)
    rest_cols = [x for x in names if x.startswith("f_rest_")]
    k_rest = len(rest_cols) // 3
    if k_rest > 0:
        f_rest = np.stack([d[x] for x in rest_cols], 1).astype(np.float32)
        f_rest = f_rest.reshape(n, k_rest, 3)
    else:
        f_rest = np.zeros((n, 0, 3), np.float32)
    colors = np.concatenate([f_dc[:, None, :], f_rest], axis=1)  # (N, K+1, 3)
    deg = int(round(np.sqrt(colors.shape[1]))) - 1
    if (deg + 1) ** 2 != colors.shape[1]:
        raise ValueError(f"SH 基数量 {colors.shape[1]} 不是平方数，PLY 可能不完整")

    def t(x):
        return torch.from_numpy(x).float().to(device)

    return {
        "means": t(means),
        # PLY 里 scales 存的是 log 空间参数、opacity 存的是 logit 参数，
        # 而 gsplat rendering.rasterization 期望线性尺度与 0-1 不透明度
        #（与 train.py 训练时 exp/sigmoid 后的输入一致）
        "scales": t(np.exp(scales)),
        "quats": t(quats),
        "opacities": t(1.0 / (1.0 + np.exp(-opacities))),
        "colors": t(colors),
        "sh_degree": deg,
        "n": n,
        "means_np": means,  # 用于网格范围估计
    }


# ---------------------------------------------------------------------------
# 2. 相机重建（支持全景 6 面 / 透视；ours / gsplat 两种归一化）
# ---------------------------------------------------------------------------


def _similarity_from_cameras(c2w):
    """gsplat normalize.py similarity_from_cameras（Apache-2.0，来源同上）。"""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = ups.mean(axis=0)
    world_up /= np.linalg.norm(world_up) + 1e-12
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0.0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(point_cloud):
    """gsplat normalize.py align_principal_axes（Apache-2.0，来源同上）。"""
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    cov = np.cov(translated, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    rot = eigvecs.T
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ centroid
    return transform


def _apply_se3(se3, xyz):
    """SE(3) 变换点（N,3）。"""
    return xyz @ se3[:3, :3].T + se3[:3, 3]


def _transform_cameras(se3, camtoworlds):
    """SE(3) 变换相机 c2w（N,4,4）。"""
    out = np.einsum("nij, ki -> nkj", camtoworlds, se3)
    scale = np.linalg.norm(out[:, 0, :3], axis=1)
    out[:, :3, :3] /= scale[:, None, None]
    return out


def build_cameras(
    data_dir,
    normalize="ours",
    factor=1,
    face_size=512,
    max_images=0,
    seed=42,
):
    """从 COLMAP sparse 构建渲染相机列表。

    返回 dict:
      views     : [ {c2w(4,4), K(3,3), W, H}, ... ]
      is_pano   : 是否全景（6 面切分）
      c2w_all   : 归一化后的原始相机（调试用）
    """
    sparse = data_mod.load_sparse(data_dir)
    cameras, images = sparse["cameras"], sparse["images"]
    c2w = data_mod.build_camtoworlds(images)
    points = sparse["points_xyz"]

    if normalize == "ours":
        c2w, points = data_mod.normalize_world(c2w, points)
    elif normalize == "gsplat":
        t1 = _similarity_from_cameras(c2w)
        c2w = _transform_cameras(t1, c2w)
        points = _apply_se3(t1, points)
        t2 = _align_principal_axes(points)
        c2w = _transform_cameras(t2, c2w)
        points = _apply_se3(t2, points)
    else:
        raise ValueError(f"unknown --normalize {normalize}")

    is_pano = any(data_mod.camera_is_equirect(c) for c in cameras)
    cam_by_id = {c["id"]: c for c in cameras}
    views = []
    n_use = len(images) if max_images <= 0 else min(max_images, len(images))
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(images))
    if max_images > 0 and max_images < len(images):
        idxs = rng.choice(idxs, max_images, replace=False)
    idxs = np.sort(idxs)[:n_use]

    if is_pano:
        K = data_mod.pinhole_intrinsics(face_size)
        for i in idxs:
            for d in FACE_DIRS.values():
                face_c2w = data_mod.build_face_c2w(c2w[i], data_mod.face_rotation(d))
                views.append(
                    dict(
                        c2w=face_c2w,
                        K=K.copy(),
                        W=face_size,
                        H=face_size,
                        name=f"{images[i]['name']}@{d}",
                    )
                )
    else:
        for i in idxs:
            cam = cam_by_id[images[i]["camera_id"]]
            K = data_mod.camera_K(cam)
            K = K.astype(np.float64)
            W = int(cam["width"] // factor)
            H = int(cam["height"] // factor)
            if factor > 1:
                K[:2] /= factor
            views.append(dict(c2w=c2w[i], K=K, W=W, H=H, name=images[i]["name"]))

    print(f"[gs2obj] {len(views)} 个渲染视图（{'全景 6 面' if is_pano else '透视'}，"
          f"归一化={normalize}，factor={factor}）")
    return dict(views=views, is_pano=is_pano, c2w_all=c2w, points_norm=points)


# ---------------------------------------------------------------------------
# 3. 渲染（RGB + 期望深度 + alpha）
# ---------------------------------------------------------------------------


def render_views(
    splats,
    views,
    device="cuda:0",
    batch_size=8,
    render_width=0,
    near_plane=0.01,
):
    """批量渲染视图，流式产出 (rgb, depth, alpha, view)。"""
    n = len(views)
    for start in range(0, n, batch_size):
        batch = views[start : start + batch_size]
        # 处理宽高不一致：garden/factor 后通常一致，全景一致；不一致则逐视图
        if len({(v["W"], v["H"]) for v in batch}) > 1:
            for v in batch:
                yield from _render_one(splats, v, device, render_width, near_plane)
            continue
        v0 = batch[0]
        W, H = v0["W"], v0["H"]
        c2w = np.stack([v["c2w"] for v in batch], 0).astype(np.float32)
        Ks = np.stack([v["K"] for v in batch], 0).astype(np.float32)
        if render_width > 0:
            s = render_width / W
            W, H = int(round(W * s)), int(round(H * s))
            Ks = Ks.copy()
            Ks[:, :2] *= s
        viewmats = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)
        Ks_t = torch.from_numpy(Ks).float().to(device)
        renders, alphas, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmats,
            Ks=Ks_t,
            width=W,
            height=H,
            sh_degree=splats["sh_degree"],
            render_mode="RGB+ED",
            near_plane=near_plane,
        )
        rgb = renders[..., :3].clamp(0, 1)
        depth = renders[..., 3:4]
        alpha = alphas
        for i, v in enumerate(batch):
            yield (
                rgb[i].detach().cpu().numpy(),
                depth[i].detach().cpu().numpy(),
                alpha[i].detach().cpu().numpy(),
                dict(c2w=v["c2w"], K=Ks[i], W=W, H=H),
            )


def _render_one(splats, v, device, render_width, near_plane):
    W, H = v["W"], v["H"]
    K = v["K"].astype(np.float32)
    if render_width > 0:
        s = render_width / W
        W, H = int(round(W * s)), int(round(H * s))
        K = K.copy()
        K[:2] *= s
    c2w = torch.from_numpy(v["c2w"].astype(np.float32))[None].to(device)
    K_t = torch.from_numpy(K)[None].to(device)
    renders, alphas, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=splats["scales"],
        opacities=splats["opacities"],
        colors=splats["colors"],
        viewmats=torch.linalg.inv(c2w),
        Ks=K_t,
        width=W,
        height=H,
        sh_degree=splats["sh_degree"],
        render_mode="RGB+ED",
        near_plane=near_plane,
    )
    yield (
        renders[0, ..., :3].clamp(0, 1).detach().cpu().numpy(),
        renders[0, ..., 3:4].detach().cpu().numpy(),
        alphas[0].detach().cpu().numpy(),
        dict(c2w=v["c2w"], K=K, W=W, H=H),
    )


# ---------------------------------------------------------------------------
# 4. 深度反投影
# ---------------------------------------------------------------------------


def unproject_depth(depth, alpha, rgb, K, c2w, alpha_thresh=0.5, near=0.01):
    """单视图深度图 → 世界坐标点云。

    gsplat 的 ED 深度是相机坐标系 z（z 向前，COLMAP 约定），
    反投影公式：X = R_c2w @ (z * K^-1 [u,v,1]) + t_c2w。
    """
    H, W = depth.shape[:2]
    mask = (alpha[..., 0] >= alpha_thresh) & (depth[..., 0] > near) & np.isfinite(
        depth[..., 0]
    )
    ys, xs = np.nonzero(mask)
    z = depth[mask, 0]
    px = np.stack(
        [xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, np.float64)], 1
    )
    cam_pts = (np.linalg.inv(K) @ px.T).T * z[:, None]  # (M,3)
    world = (c2w[:3, :3] @ cam_pts.T).T + c2w[:3, 3]
    col = rgb[mask]
    return world.astype(np.float32), col.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. TSDF 融合（自实现，numpy 向量化，逐视图流式）
# ---------------------------------------------------------------------------


def tsdf_fusion(
    splats,
    views,
    device="cuda:0",
    batch_size=8,
    render_width=0,
    voxel_size=0.0,
    trunc_mult=4.0,
    alpha_thresh=0.5,
    near_plane=0.01,
    max_voxels=30_000_000,
    margin_mult=0.2,
    save_tsdf="",
    grid_max_dist_mult=8.0,
):
    """渲染所有视图并融合进 TSDF 体素场。

    返回 (tsdf, weights, origin, voxel_size, dims)。
    网格范围 = 高斯中心 bbox + margin；voxel_size=0 时按 bbox/200 自动。
    未观测体素 tsdf 置 1（远离表面），避免 Marching Cubes 数值问题。
    """
    means = splats["means_np"]
    # 网格范围只覆盖“场景主体”：3DGS 常有飞点/真实远景（窗外、天空），
    # 全量 min/max 会把体素范围拉爆；按“距中位数的距离”截断后取分位数。
    med = np.median(means, axis=0)
    d = np.linalg.norm(means - med, axis=1)
    sel = d <= np.median(d) * grid_max_dist_mult
    if sel.sum() < 100:
        raise ValueError("grid_max_dist_mult 太小，主体点不足 100 个")
    body = means[sel]
    lo = np.percentile(body, 0.5, axis=0) - 1e-6
    hi = np.percentile(body, 99.5, axis=0) + 1e-6
    extent = hi - lo
    if voxel_size <= 0:
        voxel_size = float(extent.max() / 200.0)
    # 平面/近平面场景某方向厚度可能为 0，保证每个方向至少几层体素
    extent = np.maximum(extent, voxel_size * 4.0)
    origin = lo - margin_mult * extent  # 外扩 margin
    grid_extent = hi + margin_mult * extent - origin  # = extent * (1 + 2*margin)
    dims = np.maximum(1, np.ceil(grid_extent / voxel_size).astype(np.int64))
    while dims.prod() > max_voxels:
        voxel_size *= 1.2
        dims = np.maximum(1, np.ceil(grid_extent / voxel_size).astype(np.int64))
    nx, ny, nz = (int(d) for d in dims)
    n_vox = nx * ny * nz
    print(f"[gs2obj] TSDF 网格 {nx}x{ny}x{nz}（{n_vox/1e6:.1f}M 体素，"
          f"voxel={voxel_size:.5g}，trunc={voxel_size*trunc_mult:.5g}）")
    tsdf = np.zeros(n_vox, np.float32)
    weights = np.zeros(n_vox, np.float32)

    # 体素中心坐标（按行主序，C 顺序）
    gx = origin[0] + (np.arange(nx) + 0.5) * voxel_size
    gy = origin[1] + (np.arange(ny) + 0.5) * voxel_size
    gz = origin[2] + (np.arange(nz) + 0.5) * voxel_size
    gz, gy, gx = np.meshgrid(gz, gy, gx, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    trunc = float(voxel_size * trunc_mult)

    n_views = 0
    for rgb, depth, alpha, view in render_views(
        splats,
        views,
        device=device,
        batch_size=batch_size,
        render_width=render_width,
        near_plane=near_plane,
    ):
        _fuse_view(
            tsdf, weights, centers, depth, alpha, view, trunc, alpha_thresh, near_plane
        )
        n_views += 1
        if n_views % 50 == 0:
            print(f"[gs2obj] 已融合 {n_views} 视图")

    # 归一化 + 未观测区置 1
    wsum = np.maximum(weights, 1e-6)
    tsdf /= wsum
    tsdf[weights < 1e-6] = 1.0
    tsdf = tsdf.reshape(nx, ny, nz)
    print(f"[gs2obj] TSDF 融合完成（{n_views} 视图，观测体素 "
          f"{(weights > 1e-6).mean()*100:.1f}%）")
    if save_tsdf:
        np.savez(
            save_tsdf,
            tsdf=tsdf,
            origin=origin,
            voxel_size=float(voxel_size),
            dims=dims,
        )
        print(f"[gs2obj] TSDF 已保存 → {save_tsdf}")
    return tsdf, weights.reshape(nx, ny, nz), origin, float(voxel_size), (nx, ny, nz)


def _fuse_view(
    tsdf, weights, centers, depth, alpha, view, trunc, alpha_thresh, near_plane
):
    """把单视图深度融合进 TSDF（sdf = depth - z_cam，表面前为正）。"""
    c2w = view["c2w"].astype(np.float64)
    K = view["K"].astype(np.float64)
    W, H = depth.shape[1], depth.shape[0]
    R = c2w[:3, :3].T  # world → camera 旋转
    t = c2w[:3, 3]
    cam = (centers - t) @ R.T  # (V,3)
    z = cam[:, 2]
    valid = z > near_plane
    if not valid.any():
        return
    uv = (K @ cam[valid].T).T
    zz = uv[:, 2:3]
    uv = uv[:, :2] / np.maximum(zz, 1e-10)
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < W - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H - 1)
    )
    if not inside.any():
        return
    u, v = uv[inside].T
    zz = z[valid][inside]
    d_samp = ndimage.map_coordinates(depth[..., 0], [v, u], order=1, mode="constant", cval=0.0)
    a_samp = ndimage.map_coordinates(alpha[..., 0], [v, u], order=1, mode="constant", cval=0.0)
    sdf = d_samp - zz
    w = a_samp * (a_samp >= alpha_thresh)
    keep = (w > 0) & (d_samp > 0)
    if not keep.any():
        return
    tsdf_val = np.clip(sdf[keep] / trunc, -1.0, 1.0)
    ww = w[keep].astype(np.float32)
    idx = np.flatnonzero(valid)[inside][keep].astype(np.int64)
    # bincount 等价于 add.at（重复索引累加）但快得多
    tsdf += np.bincount(idx, weights=tsdf_val * ww, minlength=len(tsdf)).astype(
        np.float32
    )
    weights += np.bincount(idx, weights=ww, minlength=len(weights)).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Marching Cubes → 网格 → OBJ
# ---------------------------------------------------------------------------


def marching_cubes_mesh(tsdf, origin, voxel_size, keep_largest=False, strip_dist=0.0):
    """TSDF 体素场 → trimesh 网格（零等值面）。"""
    verts, faces, normals, _ = marching_cubes(
        tsdf, level=0.0, spacing=(voxel_size, voxel_size, voxel_size)
    )
    verts = verts + origin
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    if strip_dist > 0:
        v = mesh.vertices
        grid_max = origin + np.asarray(tsdf.shape) * voxel_size
        near_min = np.abs(v - origin).min(axis=1) < strip_dist
        near_max = np.abs(v - grid_max).min(axis=1) < strip_dist
        boundary = near_min | near_max
        mesh.update_faces(~boundary[mesh.faces].all(axis=1))
        mesh.remove_unreferenced_vertices()
    if keep_largest:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda p: len(p.faces))
    print(f"[gs2obj] 网格：{len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面，"
          f"水密={mesh.is_watertight}")
    return mesh


# ---------------------------------------------------------------------------
# 7. 主流程（库接口 + CLI）
# ---------------------------------------------------------------------------


def gs_to_obj(
    ply,
    data_dir,
    out_dir,
    normalize="ours",
    factor=1,
    face_size=512,
    max_images=0,
    batch_size=8,
    render_width=0,
    voxel_size=0.0,
    trunc_mult=4.0,
    alpha_thresh=0.5,
    near_plane=0.01,
    max_voxels=30_000_000,
    save_tsdf="",
    grid_max_dist_mult=8.0,
    save_depth=False,
    save_points=False,
    keep_largest=False,
    strip_dist=0.0,
    device="cuda:0",
    seed=42,
):
    """splat.ply + sparse → OBJ（+ 可选点云/深度图）。返回输出路径 dict。"""
    os.makedirs(out_dir, exist_ok=True)
    splats = load_splats(ply, device=device)
    cam_info = build_cameras(
        data_dir,
        normalize=normalize,
        factor=factor,
        face_size=face_size,
        max_images=max_images,
        seed=seed,
    )
    views = cam_info["views"]
    if len(views) == 0:
        raise ValueError("没有可渲染的相机（检查 --max-images / sparse）")

    tsdf, weights, origin, vs, dims = tsdf_fusion(
        splats,
        views,
        device=device,
        batch_size=batch_size,
        render_width=render_width,
        voxel_size=voxel_size,
        trunc_mult=trunc_mult,
        alpha_thresh=alpha_thresh,
        near_plane=near_plane,
        max_voxels=max_voxels,
        save_tsdf=save_tsdf,
        grid_max_dist_mult=grid_max_dist_mult,
    )
    mesh = marching_cubes_mesh(
        tsdf, origin, vs, keep_largest=keep_largest, strip_dist=strip_dist
    )

    obj_path = os.path.join(out_dir, "mesh.obj")
    mesh.export(obj_path)
    ply_mesh = os.path.join(out_dir, "mesh.ply")
    mesh.export(ply_mesh)
    print(f"[gs2obj] 已导出 {obj_path} / {ply_mesh}")

    if save_depth or save_points:
        # 第二次渲染，用于导出点云/深度图（可选调试，成本高）
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
        for i, (rgb, depth, alpha, view) in enumerate(
            render_views(
                splats,
                views,
                device=device,
                batch_size=batch_size,
                render_width=render_width,
                near_plane=near_plane,
            )
        ):
            if save_depth:
                np.savez(
                    os.path.join(out_dir, "depth", f"view_{i:04d}.npz"),
                    depth=depth,
                    alpha=alpha,
                    rgb=rgb,
                    c2w=view["c2w"],
                    K=view["K"],
                )
            if save_points:
                xyz, col = unproject_depth(
                    depth, alpha, rgb, view["K"], view["c2w"], alpha_thresh
                )
                pc = trimesh.PointCloud(xyz, colors=(col * 255).astype(np.uint8))
                pc.export(os.path.join(out_dir, "points", f"pc_{i:04d}.ply"))
    return dict(obj=obj_path, ply=ply_mesh)


def selftest():
    """合成平面场景自检：渲染 → 反投影 → TSDF → MC → 平面网格。"""
    import tempfile

    dev = "cuda:0"
    # 20x20 平面高斯，z=0
    xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1).astype(np.float32)
    n = len(pts)
    sh0 = (np.stack([pts[:, 2], pts[:, 1], pts[:, 0]], 1) * 0.2 + 0.5)
    sh0 = ((sh0 - 0.5) / 0.28209479177387814).astype(np.float32)
    splats = {
        "means": torch.from_numpy(pts).float().to(dev),
        "scales": torch.full((n, 3), 0.08, dtype=torch.float32, device=dev),
        "quats": torch.zeros(n, 4, dtype=torch.float32, device=dev),
        "opacities": torch.full((n,), 0.9, dtype=torch.float32, device=dev),
        "colors": torch.from_numpy(sh0[:, None, :]).float().to(dev),
        "sh_degree": 0,
        "n": n,
        "means_np": pts,
    }
    splats["quats"][:, 0] = 1.0
    # 相机：z=2 朝 -z 看平面（COLMAP z 向前）
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = [0, 0, 2.0]
    K = np.array([[500.0, 0, 256], [0, 500.0, 256], [0, 0, 1]], np.float64)
    views = [dict(c2w=c2w, K=K, W=512, H=512)]
    tsdf, _, origin, vs, _ = tsdf_fusion(
        splats, views, device=dev, batch_size=1, voxel_size=0.05, alpha_thresh=0.3
    )
    mesh = marching_cubes_mesh(tsdf, origin, vs)
    assert len(mesh.vertices) > 50, f"网格顶点太少: {len(mesh.vertices)}"
    assert abs(mesh.vertices[:, 2].mean()) < 0.3, f"平面 z 偏差过大: {mesh.vertices[:, 2].mean()}"
    print(f"[selftest] PASS：{len(mesh.vertices)} 顶点，平面 z 均值 "
          f"{mesh.vertices[:, 2].mean():.3f}")
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="3DGS → OBJ（多视角深度渲染 + TSDF 融合 + Marching Cubes）"
    )
    p.add_argument("--ply", default="", help="输入 splat.ply")
    p.add_argument("--data-dir", default="", help="COLMAP sparse 所在数据集目录")
    p.add_argument("--out-dir", default="", help="输出目录（默认 <ply 同目录>/mesh）")
    p.add_argument("--normalize", choices=["ours", "gsplat"], default="ours",
                   help="相机归一化：ours=本项目 train.py；gsplat=gsplat simple_trainer")
    p.add_argument("--factor", type=int, default=1,
                   help="图片下采样倍数（gsplat 数据集的 images_2 对应 2）")
    p.add_argument("--face-size", type=int, default=512,
                   help="全景切面边长（渲染深度用）")
    p.add_argument("--max-images", type=int, default=0,
                   help="最多用 N 张图（0=全部；全景每张算 6 面）")
    p.add_argument("--batch-size", type=int, default=8, help="每批渲染相机数")
    p.add_argument("--render-size", type=int, default=0,
                   help="渲染宽度（0=相机原始尺寸，小值省显存/时间）")
    p.add_argument("--voxel-size", type=float, default=0.0,
                   help="TSDF 体素尺寸（0=按场景 bbox/200 自动）")
    p.add_argument("--trunc-mult", type=float, default=4.0,
                   help="TSDF 截断距离 = 体素 × 倍数")
    p.add_argument("--alpha-thresh", type=float, default=0.5,
                   help="深度可靠像素的最低 alpha")
    p.add_argument("--near-plane", type=float, default=0.05,
                   help="渲染近裁剪面（相机穿行场景时调大，跳过贴身高斯）")
    p.add_argument("--max-voxels", type=int, default=30_000_000,
                   help="体素数上限（超出自动调粗体素）")
    p.add_argument("--save-tsdf", default="",
                   help="保存 TSDF 体积 npz（调试用）")
    p.add_argument("--grid-max-dist-mult", type=float, default=8.0,
                   help="网格范围=中位距离×倍数（只重建主体，忽略飞点/远景）")
    p.add_argument("--save-depth", action="store_true", help="额外保存深度图 npz")
    p.add_argument("--save-points", action="store_true", help="额外保存彩色点云 PLY")
    p.add_argument("--keep-largest", action="store_true",
                   help="只保留最大连通分量（去掉边界壳/小碎片）")
    p.add_argument("--strip-dist", type=float, default=0.0,
                   help="剥离贴网格边界的面（>0 时生效，单位与场景一致）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true", help="合成数据自检")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.ply or not args.data_dir:
        raise SystemExit("[gs2obj] 需要 --ply 和 --data-dir")
    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ply)), "mesh")
    gs_to_obj(
        args.ply,
        args.data_dir,
        out_dir,
        normalize=args.normalize,
        factor=args.factor,
        face_size=args.face_size,
        max_images=args.max_images,
        batch_size=args.batch_size,
        render_width=args.render_size,
        voxel_size=args.voxel_size,
        trunc_mult=args.trunc_mult,
        alpha_thresh=args.alpha_thresh,
        near_plane=args.near_plane,
        max_voxels=args.max_voxels,
        save_tsdf=args.save_tsdf,
        grid_max_dist_mult=args.grid_max_dist_mult,
        save_depth=args.save_depth,
        save_points=args.save_points,
        keep_largest=args.keep_largest,
        strip_dist=args.strip_dist,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
