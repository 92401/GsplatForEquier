# -*- coding: utf-8 -*-
"""去除 3DGS 训练产物中的漂浮物/噪点（PointNuker 核心算法的内嵌实现）。

PointNuker（https://github.com/Dymensium/PointNuker，MIT）是一个训练后清理
3DGS 漂浮物的 GUI 工具，核心思路：
  1. 对高斯中心点做 DBSCAN 聚类，保留最大簇（主场景），扔掉零星小簇/噪声点；
  2. 配合半径离群点去除、统计离群点去除、AABB 裁剪等清理步骤；
  3. GS-safe 保存：只按行删除，PLY 里所有高斯属性（x/y/z、SH、opacity、
     scale、rot 等）原样保留。

本模块把上述核心算法集成到项目里（不安装 open3d / PointNuker 本体），
用 sklearn 的 DBSCAN / NearestNeighbors 实现等价逻辑，可作为库导入：

    from clean_floaters import clean_ply, dbscan_main_cluster
    mask = dbscan_main_cluster(xyz, eps=0.5, min_points=20)

也可作为命令行工具：

    python clean_floaters.py --ply outputs/run/splat.ply \
        --out outputs/run/splat_clean.ply \
        --keep-main-cluster --auto-eps
    python clean_floaters.py --selftest
"""

import argparse
import os
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement, PlyListProperty
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------------------
# PLY 读写（GS-safe：只按行删，保留全部属性与 dtype）
# ---------------------------------------------------------------------------


def load_ply(path):
    """读取 PLY，返回 (data, props, text, byte_order)。

    data  : dict {属性名: np.ndarray}，行数与高斯数一致
    props : [(属性名, dtype), ...]，用于原样重建 PLY
    """
    plydata = PlyData.read(path)
    if "vertex" not in plydata:
        raise ValueError(f"[clean] {path} 里没有 vertex 元素")
    vertex = plydata["vertex"]
    props = []
    for p in vertex.properties:
        if isinstance(p, PlyListProperty):
            raise ValueError(
                f"[clean] 遇到 list 属性 {p.name}，本工具只支持标量属性的 GS PLY"
            )
        props.append((p.name, p.val_dtype))
    data = {name: vertex[name] for name, _ in props}
    return data, props, plydata.text, plydata.byte_order


def save_ply(path, data, props, mask, text=False, byte_order="<"):
    """按 mask 保留行，把 PLY 写回磁盘（属性集合/顺序/dtype 与输入一致）。"""
    keep = np.flatnonzero(mask)
    dtype = np.dtype(props)
    rec = np.empty(len(keep), dtype=dtype)
    for name, _ in props:
        rec[name] = np.asarray(data[name])[keep]
    element = PlyElement.describe(rec, "vertex")
    PlyData([element], text=text, byte_order=byte_order).write(path)


# ---------------------------------------------------------------------------
# 各清理步骤（均返回 bool mask，True=保留）
# ---------------------------------------------------------------------------


def drop_nonfinite(xyz):
    """剔除坐标含 NaN/Inf 的高斯（gsplat 导出时已滤过，这里兜底）。"""
    return np.isfinite(xyz).all(axis=1)


def opacity_filter(opacities, min_opacity):
    """按不透明度过滤：sigmoid(opacity) < min_opacity 的点删除。

    PLY 里存的是 logit 空间的原始 opacity，先还原成概率再比较。
    训练尾期大量低不透明度透明高斯常以“半透明雾”形式漂浮，此步可直接清掉。
    """
    return sigmoid(opacities) >= min_opacity


def radius_outlier_removal(xyz, radius, min_k):
    """半径离群点去除（等价 open3d remove_radius_outlier）。

    统计每个点半径 radius 球内的邻居数（含自身，与 open3d 一致），
    少于 min_k 个的点删除。适合去掉孤立的小团噪点。
    """
    n = len(xyz)
    if n < min_k:
        return np.zeros(n, dtype=bool)
    nn = NearestNeighbors(radius=radius, algorithm="kd_tree", n_jobs=1).fit(xyz)
    counts = nn.radius_neighbors(xyz, return_distance=False)
    return np.array([len(c) for c in counts], dtype=np.int64) >= min_k


def statistical_outlier_removal(xyz, k, n_sigmas):
    """统计离群点去除（等价 open3d remove_statistical_outlier）。

    每个点到其 k 个最近邻的平均距离，超过
    (全局平均 + n_sigmas × 全局标准差) 的点删除。
    """
    n = len(xyz)
    if n < 3:
        return np.ones(n, dtype=bool)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree", n_jobs=1).fit(xyz)
    dists, _ = nn.kneighbors(xyz)
    mean_d = dists[:, 1:].mean(axis=1)
    thr = mean_d.mean() + n_sigmas * mean_d.std()
    return mean_d <= thr


def aabb_crop(xyz, lo, hi):
    """AABB 裁剪：三个轴都在 [lo, hi] 内的点保留。"""
    keep = np.ones(len(xyz), dtype=bool)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if a is not None:
            keep &= xyz[:, i] >= a
        if b is not None:
            keep &= xyz[:, i] <= b
    return keep


def dbscan_main_cluster(
    xyz,
    eps,
    min_points=20,
    max_n=1_000_000,
    assign_dist=None,
    seed=42,
):
    """DBSCAN 聚类，只保留最大簇（PointNuker 的 Cluster Finder）。

    eps       : 聚类半径（与场景坐标同一单位）
    min_points: 成为一簇所需的最少点数（DBSCAN min_samples）
    max_n     : 超过该点数时先随机抽样做聚类、再按距离回贴，避免内存/时间爆炸
    assign_dist: 抽样模式下，全量点离“主簇抽样点”的距离阈值（默认=eps）
    """
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=bool)
    labels = None
    if n <= max_n:
        labels = DBSCAN(eps=eps, min_samples=min_points, n_jobs=1).fit_predict(xyz)
    else:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n, max_n, replace=False)
        sub = xyz[sub_idx]
        sub_labels = DBSCAN(eps=eps, min_samples=min_points, n_jobs=1).fit_predict(sub)
        counts = np.bincount(sub_labels[sub_labels >= 0])
        if counts.size == 0:
            return np.zeros(n, dtype=bool)
        main = int(np.argmax(counts))
        main_pts = sub[sub_labels == main]
        dists, _ = (
            NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1)
            .fit(main_pts)
            .kneighbors(xyz)
        )
        return dists[:, 0] <= (assign_dist if assign_dist else eps)

    counts = np.bincount(labels[labels >= 0])
    if counts.size == 0:
        # 全部是噪声，没有任何簇：保守起见什么都不删
        return np.ones(n, dtype=bool)
    main = int(np.argmax(counts))
    return labels == main


def suggest_eps(xyz, k=16, q=0.95, max_n=200_000, seed=42):
    """自动建议 DBSCAN 的 eps：k-NN 平均距离的 q 分位数（抽样估计）。

    等价于 PointNuker 的 Parameter Assistant 思路：
    先用小样本算每个点到 k 个最近邻的平均距离，再取高分位点当聚类半径。
    """
    n = len(xyz)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= max_n else rng.choice(n, max_n, replace=False)
    sub = xyz[idx]
    kk = min(k + 1, len(sub))
    nn = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree", n_jobs=1).fit(sub)
    dists, _ = nn.kneighbors(sub)
    mean_d = dists[:, 1:].mean(axis=1)
    return float(np.quantile(mean_d, q))


# ---------------------------------------------------------------------------
# 主流程（库接口）
# ---------------------------------------------------------------------------


def clean_ply(
    src,
    dst,
    *,
    min_opacity=0.0,
    radius=None,
    min_k=16,
    stat_k=20,
    n_sigmas=1.5,
    aabb_lo=None,
    aabb_hi=None,
    keep_main_cluster=False,
    eps=None,
    min_points=20,
    auto_eps=False,
    auto_eps_k=16,
    auto_eps_q=0.95,
    dbscan_max_n=1_000_000,
    assign_dist=None,
    seed=42,
    dry_run=False,
    verbose=True,
):
    """执行完整清理管线，返回 (keep_mask, stats)。

    src/dst : 输入/输出 PLY 路径（dry_run 时不写 dst）
    清理顺序：非有限点 → 低不透明度 → 半径离群 → 统计离群 → AABB → 最大簇。
    stats   : [(步骤名, 删除数, 剩余数), ...]
    """
    data, props, text, byte_order = load_ply(src)
    n0 = len(data["x"])
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

    mask = np.ones(n0, dtype=bool)
    stats = []

    def apply_step(name, m):
        nonlocal mask
        removed = int((mask & ~m).sum())
        mask &= m
        stats.append((name, removed, int(mask.sum())))
        if verbose:
            print(f"[clean] {name}: 删除 {removed}（剩余 {mask.sum()}/{n0}）")

    apply_step("非有限点", drop_nonfinite(xyz))

    if min_opacity > 0 and "opacity" in data:
        apply_step(f"不透明度<{min_opacity}", opacity_filter(data["opacity"], min_opacity))

    cur = xyz[mask]
    if radius is not None and radius > 0:
        m = np.ones(mask.sum(), dtype=bool)
        if len(cur):
            m = radius_outlier_removal(cur, radius, min_k)
        full = np.zeros(n0, dtype=bool)
        full[mask] = m
        apply_step(f"半径离群(r={radius},k>={min_k})", full)
        cur = xyz[mask]

    if stat_k > 0:
        m = np.ones(mask.sum(), dtype=bool)
        if len(cur):
            m = statistical_outlier_removal(cur, stat_k, n_sigmas)
        full = np.zeros(n0, dtype=bool)
        full[mask] = m
        apply_step(f"统计离群(k={stat_k},σ={n_sigmas})", full)
        cur = xyz[mask]

    if aabb_lo is not None or aabb_hi is not None:
        lo = aabb_lo if aabb_lo is not None else [None, None, None]
        hi = aabb_hi if aabb_hi is not None else [None, None, None]
        m = np.ones(mask.sum(), dtype=bool)
        if len(cur):
            m = aabb_crop(cur, lo, hi)
        full = np.zeros(n0, dtype=bool)
        full[mask] = m
        apply_step("AABB 裁剪", full)
        cur = xyz[mask]

    if keep_main_cluster:
        if eps is None:
            if auto_eps:
                eps = suggest_eps(cur, k=auto_eps_k, q=auto_eps_q, seed=seed)
                if verbose:
                    print(f"[clean] 自动 eps = {eps:.4g}")
            else:
                raise ValueError("--keep-main-cluster 需要 --eps 或 --auto-eps")
        m = np.ones(mask.sum(), dtype=bool)
        if len(cur):
            m = dbscan_main_cluster(
                cur,
                eps,
                min_points=min_points,
                max_n=dbscan_max_n,
                assign_dist=assign_dist,
                seed=seed,
            )
        full = np.zeros(n0, dtype=bool)
        full[mask] = m
        apply_step(f"DBSCAN 最大簇(eps={eps},min_pts={min_points})", full)

    n_keep = int(mask.sum())
    ratio = n_keep / n0 if n0 else 0
    if n_keep == 0:
        raise RuntimeError("[clean] 全部点都被删掉了，请放宽参数（如加大 eps / 减小 min_points）")
    if ratio < 0.01 and verbose:
        print(f"[clean] 警告：只剩 {ratio:.1%} 的点，请检查参数是否过狠")

    if not dry_run:
        save_ply(dst, data, props, mask, text=text, byte_order=byte_order)
        if verbose:
            print(f"[clean] 已保存 {n_keep}/{n0} 个高斯 → {dst}")
    elif verbose:
        print(f"[clean] dry-run：保留 {n_keep}/{n0}（{ratio:.1%}）")
    return mask, stats


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_vec3(s):
    """解析 'x,y,z'，允许单个分量写成 none（表示该轴不限制）。"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"需要 x,y,z 三个分量，收到：{s!r}")
    out = []
    for p in parts:
        out.append(None if p.lower() in ("none", "") else float(p))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="清理 3DGS 训练产物中的漂浮物（PointNuker 核心算法内嵌版）"
    )
    p.add_argument("--ply", default="", help="输入 splat.ply（gsplat 导出）")
    p.add_argument("--out", default="", help="输出 PLY 路径（默认 <ply 目录>/splat_clean.ply）")
    p.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    # 步骤开关
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="删除 sigmoid(opacity)<该值 的高斯（logit 存盘，自动还原）")
    p.add_argument("--radius-outlier", action="store_true",
                   help="半径离群点去除（球内邻居数<min-k 的点删除）")
    p.add_argument("--radius", type=float, default=0.02,
                   help="半径离群点去除的球半径（与场景坐标同单位）")
    p.add_argument("--min-k", type=int, default=16,
                   help="半径球内最少点数（含自身，open3d 语义）")
    p.add_argument("--stat-outlier", action="store_true",
                   help="统计离群点去除（k-NN 平均距离超阈值删除）")
    p.add_argument("--stat-k", type=int, default=20, help="统计离群点的最近邻数")
    p.add_argument("--n-sigmas", type=float, default=1.5,
                   help="统计离群点的标准差倍数阈值")
    p.add_argument("--aabb-min", type=parse_vec3, default=None,
                   help="AABB 最小角 x,y,z（none=不限制；负值请用 = 形式，"
                        "如 --aabb-min=-5,-5,-5）")
    p.add_argument("--aabb-max", type=parse_vec3, default=None,
                   help="AABB 最大角 x,y,z（none=不限制；负值请用 = 形式）")
    p.add_argument("--keep-main-cluster", action="store_true",
                   help="DBSCAN 聚类并只保留最大簇（PointNuker 主功能）")
    p.add_argument("--eps", type=float, default=None,
                   help="DBSCAN 聚类半径（与场景坐标同单位）")
    p.add_argument("--auto-eps", action="store_true",
                   help="自动建议 eps：k-NN 平均距离的高分位数")
    p.add_argument("--auto-eps-k", type=int, default=16)
    p.add_argument("--auto-eps-q", type=float, default=0.95)
    p.add_argument("--min-points", type=int, default=20,
                   help="DBSCAN 成为簇的最少点数")
    p.add_argument("--dbscan-max-n", type=int, default=1_000_000,
                   help="超过该点数先抽样聚类再按距离回贴（省内存）")
    p.add_argument("--assign-dist", type=float, default=None,
                   help="抽样模式下回贴距离阈值（默认=eps）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="跑合成数据自检并退出")
    return p.parse_args(argv)


def selftest():
    """合成数据自检：主簇 + 远处漂浮物，验证清理与 GS-safe 保存。"""
    rng = np.random.default_rng(7)
    n_main, n_far, n_far2 = 2000, 120, 30
    xyz = np.vstack([
        rng.normal(0, 0.3, (n_main, 3)),
        rng.uniform(8, 12, (n_far, 3)),
        rng.uniform(30, 40, (n_far2, 3)),
    ])
    n = len(xyz)
    opacity = np.log(1.0 / np.clip(rng.uniform(0.05, 0.95, n), 1e-6, 1.0) - 1.0)
    data = {
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "f_dc_0": rng.normal(0, 1, n).astype(np.float32),
        "f_dc_1": rng.normal(0, 1, n).astype(np.float32),
        "f_dc_2": rng.normal(0, 1, n).astype(np.float32),
        "f_rest_0": rng.normal(0, 1, n).astype(np.float32),
        "f_rest_1": rng.normal(0, 1, n).astype(np.float32),
        "f_rest_2": rng.normal(0, 1, n).astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scale_0": rng.normal(-2, 0.5, n).astype(np.float32),
        "scale_1": rng.normal(-2, 0.5, n).astype(np.float32),
        "scale_2": rng.normal(-2, 0.5, n).astype(np.float32),
        "rot_0": rng.normal(0, 1, n).astype(np.float32),
        "rot_1": rng.normal(0, 1, n).astype(np.float32),
        "rot_2": rng.normal(0, 1, n).astype(np.float32),
        "rot_3": rng.normal(0, 1, n).astype(np.float32),
    }
    props = [(k, v.dtype) for k, v in data.items()]
    tmpdir = tempfile.mkdtemp(prefix="pointnuker_selftest_")
    src = os.path.join(tmpdir, "src.ply")
    dst = os.path.join(tmpdir, "dst.ply")
    save_ply(src, data, props, np.ones(n, dtype=bool))

    mask, stats = clean_ply(
        src, dst, keep_main_cluster=True, eps=0.5, min_points=20, verbose=False
    )
    kept = int(mask.sum())
    assert kept == n_main, f"主簇保留数不对：{kept} != {n_main}"
    kept_xyz = xyz[mask]
    assert kept_xyz.max(axis=0)[0] < 2, "漂浮物没有被清掉"

    # GS-safe：属性名、dtype、保留行的值必须与输入完全一致
    data2, props2, _, _ = load_ply(dst)
    assert [a for a, _ in props2] == [a for a, _ in props], "属性集合变了"
    assert [np.dtype(b) for _, b in props2] == [np.dtype(b) for _, b in props], "dtype 变了"
    for name, _ in props:
        np.testing.assert_array_equal(np.asarray(data2[name]), np.asarray(data[name])[mask],
                                      err_msg=f"{name} 值不一致")
    print(f"[selftest] PASS：{kept}/{n} 保留，属性无损（{len(props)} 个属性）")
    return True


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.ply:
        raise SystemExit("[clean] 需要 --ply 指定输入 splat.ply")
    if not args.keep_main_cluster and not (args.radius_outlier or args.stat_outlier
                                           or args.aabb_min or args.aabb_max
                                           or args.min_opacity > 0):
        print("[clean] 警告：没有启用任何清理步骤，仅做 NaN 过滤")
    if not args.out:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.ply)),
                                "splat_clean.ply")
    clean_ply(
        args.ply,
        args.out,
        min_opacity=args.min_opacity,
        radius=args.radius if args.radius_outlier else None,
        min_k=args.min_k,
        stat_k=args.stat_k if args.stat_outlier else 0,
        n_sigmas=args.n_sigmas,
        aabb_lo=args.aabb_min,
        aabb_hi=args.aabb_max,
        keep_main_cluster=args.keep_main_cluster,
        eps=args.eps,
        min_points=args.min_points,
        auto_eps=args.auto_eps,
        auto_eps_k=args.auto_eps_k,
        auto_eps_q=args.auto_eps_q,
        dbscan_max_n=args.dbscan_max_n,
        assign_dist=args.assign_dist,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
