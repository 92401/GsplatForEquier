"""伪地理坐标数据生成器。

把一份 COLMAP 空三结果（局部坐标）绑定到一个真实的 UTM/地理坐标系上，
生成：
  1) rtk_traj.csv   模拟 RTK/卫惯输出的轨迹（时间戳+经纬度+高程+姿态+四元数）
  2) enu_traj.csv   ENU 轨迹（调试用）
  3) transform.json 相似变换参数（局部坐标 -> ENU）
  4) sparse/0/      变换到 ENU 坐标系的 COLMAP sparse（可直接被 load_sparse 读取）

用途：在真实 RTK 盒子数据到位之前，用这套"伪地理坐标"开发和验证
坐标转换、地理配准、带地理坐标的 PLY/OBJ 导出等下游模块。

用法:
  python gen_geo_data.py --data-dir D:\\gaussian_splatting\\spirula ^
      --out-dir outputs\\geo --lat 30.6599 --lon 104.0633 --alt 500

说明：--scale 为"局部单位 -> 米"的尺度（auto=相邻相机中位步长 1.2m）；
位姿与点云按同一相似变换 p_enu = s*R@p + t 映射，RTK 轨迹与 sparse/0
严格自洽（生成后脚本会做一致性校验）。
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np

import geo
from data import build_camtoworlds, load_sparse, qvec_to_rotmat


# ---------------------------------------------------------------------------
# COLMAP 二进制写出（保持与 data.py 读取格式一致）
# ---------------------------------------------------------------------------

def _read_points3d_with_tracks(path: str):
    """读取 points3D.bin，含每条 track 的 (image_id, point2D_idx)。"""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    count = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    ids, xyz, rgb, err, tracks = [], [], [], [], []
    for _ in range(count):
        ids.append(struct.unpack_from("<Q", data, pos)[0])
        pos += 8
        xyz.append(struct.unpack_from("<3d", data, pos))
        pos += 24
        rgb.append(struct.unpack_from("<3B", data, pos))
        pos += 3
        err.append(struct.unpack_from("<d", data, pos)[0])
        pos += 8
        track_len = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        tr = []
        for _ in range(track_len):
            tr.append(struct.unpack_from("<II", data, pos))
            pos += 8
        tracks.append(tr)
    return (np.array(ids, np.uint64), np.array(xyz, np.float64),
            np.array(rgb, np.uint8), np.array(err, np.float64), tracks)


def write_cameras_bin(path: str, cameras: list):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for c in cameras:
            f.write(struct.pack("<IiQQ", c["id"], c["model"],
                                c["width"], c["height"]))
            f.write(struct.pack("<%dd" % len(c["params"]), *c["params"]))


def write_images_bin(path: str, images: list):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<I", im["image_id"]))
            f.write(struct.pack("<4d", *im["qvec"]))
            f.write(struct.pack("<3d", *im["tvec"]))
            f.write(struct.pack("<I", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im["xys"])))
            for xy, pid in zip(im["xys"], im["point3D_ids"]):
                f.write(struct.pack("<2d", *xy))
                f.write(struct.pack("<q", pid))


def write_points3d_bin(path: str, ids, xyz, rgb, err, tracks):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(ids)))
        for i in range(len(ids)):
            f.write(struct.pack("<Q", ids[i]))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(struct.pack("<3B", *rgb[i]))
            f.write(struct.pack("<d", err[i]))
            f.write(struct.pack("<Q", len(tracks[i])))
            for image_id, point2d_idx in tracks[i]:
                f.write(struct.pack("<II", image_id, point2d_idx))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=r"D:\gaussian_splatting\spirula",
                    help="COLMAP 数据集目录（含 sparse/0）")
    ap.add_argument("--out-dir", default=r"D:\gaussian_splatting\pano_gsplat\outputs\geo")
    ap.add_argument("--lat", type=float, default=30.6599,
                    help="参考原点纬度（度），默认成都")
    ap.add_argument("--lon", type=float, default=104.0633,
                    help="参考原点经度（度）")
    ap.add_argument("--alt", type=float, default=500.0,
                    help="参考原点高程（米）")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="局部单位 -> 米 的尺度；0=auto（相邻相机中位步长 1.2m）")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="模拟采集频率（帧/秒，决定 RTK 时间戳间隔）")
    ap.add_argument("--start-time", type=float, default=1782892800.0,
                    help="第一帧模拟时间戳（Unix 秒，默认 2026-07-01 00:00:00 UTC）")
    args = ap.parse_args()

    sparse = load_sparse(args.data_dir)
    cameras = sparse["cameras"]
    images = sparse["images"]
    points_xyz = sparse["points_xyz"]
    c2w = build_camtoworlds(images)          # 原始空三局部坐标
    centers = c2w[:, :3, 3]
    n = len(images)

    # 尺度：auto = 相邻相机中位步长 -> 1.2m
    if args.scale > 0:
        scale = args.scale
    else:
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        median_step = float(np.median(steps)) if len(steps) else 1.0
        scale = 1.2 / median_step if median_step > 1e-9 else 1.0
        print(f"[geo] auto scale: median step {median_step:.5f} -> "
              f"scale {scale:.4f} m/unit")

    # 相似变换：p_enu = s * R @ p + t，R=I，原点 = 第一台相机
    r_mat = np.eye(3)
    t_vec = -scale * centers[0]

    # 相机位姿 -> ENU + 地理坐标
    enu_c = geo.apply_similarity(centers, scale, r_mat, t_vec)
    geo_rows = []
    for i in range(n):
        lat, lon, h = geo.enu_to_geodetic(*enu_c[i], args.lat, args.lon, args.alt)
        r_cam_enu = r_mat @ c2w[i, :3, :3]
        yaw, pitch, roll = geo.rotation_to_ypr(r_cam_enu)
        qw, qx, qy, qz = geo.rotmat_to_qvec(r_cam_enu)
        geo_rows.append(dict(
            idx=i, name=images[i]["name"],
            timestamp=args.start_time + i / args.fps,
            e=enu_c[i, 0], n=enu_c[i, 1], u=enu_c[i, 2],
            lat=lat, lon=lon, alt=h,
            yaw=yaw, pitch=pitch, roll=roll,
            qw=qw, qx=qx, qy=qy, qz=qz,
        ))

    # 输出目录
    os.makedirs(os.path.join(args.out_dir, "sparse", "0"), exist_ok=True)

    # 1) rtk_traj.csv
    rtk_path = os.path.join(args.out_dir, "rtk_traj.csv")
    with open(rtk_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["timestamp", "lat", "lon", "alt", "yaw", "pitch", "roll",
                    "qw", "qx", "qy", "qz", "image"])
        for g in geo_rows:
            w.writerow([f"{g['timestamp']:.3f}", f"{g['lat']:.9f}",
                        f"{g['lon']:.9f}", f"{g['alt']:.3f}",
                        f"{g['yaw']:.4f}", f"{g['pitch']:.4f}",
                        f"{g['roll']:.4f}", f"{g['qw']:.8f}",
                        f"{g['qx']:.8f}", f"{g['qy']:.8f}",
                        f"{g['qz']:.8f}", g["name"]])

    # 2) enu_traj.csv
    enu_path = os.path.join(args.out_dir, "enu_traj.csv")
    with open(enu_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["idx", "e", "n", "u", "image"])
        for g in geo_rows:
            w.writerow([g["idx"], f"{g['e']:.4f}", f"{g['n']:.4f}",
                        f"{g['u']:.4f}", g["name"]])

    # 3) transform.json
    zone, east0, north0, northern = geo.latlon_to_utm(args.lat, args.lon)
    tf = dict(
        scale=float(scale),
        R=r_mat.tolist(),
        t=t_vec.tolist(),
        ref_lat=args.lat, ref_lon=args.lon, ref_alt=args.alt,
        ref_utm_zone=zone, ref_utm_easting=east0, ref_utm_northing=north0,
        ref_utm_northern=bool(northern),
        n_images=n,
        source=os.path.abspath(args.data_dir),
        note="p_enu = s*R@p_local + t；ENU 原点 = 参考经纬度/高程",
    )
    with open(os.path.join(args.out_dir, "transform.json"), "w",
              encoding="utf-8") as f:
        json.dump(tf, f, indent=2, ensure_ascii=False)

    # 4) sparse/0：位姿与点云变换到 ENU
    c2w_geo = geo.transform_c2w_similarity(c2w, scale, r_mat, t_vec)
    new_images = []
    for i, im in enumerate(images):
        r_new = c2w_geo[i, :3, :3]
        t_new = c2w_geo[i, :3, 3]
        new_images.append(dict(
            image_id=im["image_id"],
            qvec=tuple(geo.rotmat_to_qvec(r_new)),
            tvec=tuple(-r_new @ t_new),
            camera_id=im["camera_id"],
            name=im["name"],
            xys=im["xys"],
            point3D_ids=im["point3D_ids"],
        ))
    pts_ids, pts_xyz, pts_rgb, pts_err, pts_tracks = _read_points3d_with_tracks(
        os.path.join(args.data_dir, "sparse", "0", "points3D.bin"))
    pts_xyz_geo = geo.apply_similarity(pts_xyz, scale, r_mat, t_vec)

    write_cameras_bin(os.path.join(args.out_dir, "sparse", "0", "cameras.bin"),
                      cameras)
    write_images_bin(os.path.join(args.out_dir, "sparse", "0", "images.bin"),
                     new_images)
    write_points3d_bin(os.path.join(args.out_dir, "sparse", "0",
                                    "points3D.bin"),
                       pts_ids, pts_xyz_geo, pts_rgb, pts_err, pts_tracks)

    # 一致性校验：sparse/0 的相机中心 == RTK/ENU 轨迹（重读后对比）
    sparse2 = load_sparse(args.out_dir)
    c2w2 = build_camtoworlds(sparse2["images"])
    err = np.abs(c2w2[:, :3, 3] - enu_c).max()
    if err > 1e-5:
        raise RuntimeError(f"sparse/0 与 ENU 轨迹不一致: {err:.2e}")

    # 汇总
    total = float(np.linalg.norm(np.diff(enu_c, axis=0), axis=1).sum())
    lats = [g["lat"] for g in geo_rows]
    lons = [g["lon"] for g in geo_rows]
    alts = [g["alt"] for g in geo_rows]
    print(f"[geo] {n} 张影像，轨迹总长 {total:.1f} m")
    print(f"[geo] 经度 {min(lons):.7f}..{max(lons):.7f}, "
          f"纬度 {min(lats):.7f}..{max(lats):.7f}, "
          f"高程 {min(alts):.1f}..{max(alts):.1f} m")
    print(f"[geo] 校验通过（sparse/0 与 RTK 轨迹一致，最大误差 {err:.2e} m）")
    print(f"[geo] 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
