"""地理坐标转换库（零第三方依赖）。

提供 WGS84 大地坐标 <-> ECEF <-> ENU、UTM 投影、3D 相似变换拟合/应用、
旋转矩阵 <-> 航向/俯仰/横滚 转换，以及 COLMAP 位姿的相似变换。
用于"空三局部坐标 <-> RTK/地理坐标"链路的开发和自检。

用法（自检）:
  python geo.py --selftest
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# WGS84 椭球参数
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)
UTM_K0 = 0.9996


# ---------------------------------------------------------------------------
# WGS84 <-> ECEF <-> ENU
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float):
    """WGS84 大地坐标(度,米) -> ECEF 地心地固坐标 (x,y,z) 米。"""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    x = (n + h_m) * math.cos(lat) * math.cos(lon)
    y = (n + h_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h_m) * math.sin(lat)
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z, iters: int = 8):
    """ECEF -> WGS84 大地坐标 (lat_deg, lon_deg, h_m)，迭代求解（收敛到亚毫米）。"""
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    h = 0.0
    for _ in range(iters):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        h = (p / math.cos(lat) - n) if p > 1e-8 else (abs(z) - WGS84_A * (1 - WGS84_E2))
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon), h


def _enu_rot(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """ENU 旋转矩阵：enu = R @ (ecef - ecef0)。"""
    lat = math.radians(lat0_deg)
    lon = math.radians(lon0_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def geodetic_to_enu(lat_deg, lon_deg, h_m, lat0_deg, lon0_deg, h0_m):
    """WGS84 -> 以 (lat0,lon0,h0) 为原点的 ENU（东/北/上，米）。"""
    p = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    p0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    return _enu_rot(lat0_deg, lon0_deg) @ (p - p0)


def enu_to_geodetic(e, n, u, lat0_deg, lon0_deg, h0_m):
    """ENU（米）-> WGS84。"""
    p0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    p = p0 + _enu_rot(lat0_deg, lon0_deg).T @ np.array([e, n, u])
    return ecef_to_geodetic(p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# UTM（Snyder 横轴墨卡托级数，WGS84，精度优于亚米级，足够工程使用）
# ---------------------------------------------------------------------------

def _meridian_arc(phi: float) -> float:
    """子午线弧长（从赤道到纬度 phi，米）。"""
    e2 = WGS84_E2
    return WGS84_A * (
        (1.0 - e2 / 4.0 - 3.0 * e2 ** 2 / 64.0 - 5.0 * e2 ** 3 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e2 ** 2 / 32.0 + 45.0 * e2 ** 3 / 1024.0)
        * math.sin(2.0 * phi)
        + (15.0 * e2 ** 2 / 256.0 + 45.0 * e2 ** 3 / 1024.0)
        * math.sin(4.0 * phi)
        - (35.0 * e2 ** 3 / 3072.0) * math.sin(6.0 * phi)
    )


def utm_zone(lon_deg: float) -> int:
    """经度 -> UTM 带号（1-60）。"""
    return int(math.floor((lon_deg + 180.0) / 6.0)) + 1


def latlon_to_utm(lat_deg: float, lon_deg: float):
    """WGS84 -> (zone, easting, northing, northern)，单位为米。

    返回带号供反算使用；northern=False 表示南半球（northing 含 10000km 假北）。
    """
    zone = utm_zone(lon_deg)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    e2, ep2 = WGS84_E2, WGS84_EP2
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = ep2 * math.cos(lat) ** 2
    a = (lon - lon0) * math.cos(lat)
    m = _meridian_arc(lat)
    x = UTM_K0 * n * (
        a + (1.0 - t + c) * a ** 3 / 6.0
        + (5.0 - 18.0 * t + t ** 2 + 72.0 * c - 58.0 * ep2) * a ** 5 / 120.0
    )
    y = UTM_K0 * (
        m + n * math.tan(lat) * (
            a ** 2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c ** 2) * a ** 4 / 24.0
            + (61.0 - 58.0 * t + t ** 2 + 600.0 * c - 330.0 * ep2)
            * a ** 6 / 720.0
        )
    )
    northern = lat_deg >= 0.0
    easting = 500000.0 + x
    northing = y if northern else y + 10000000.0
    return zone, easting, northing, northern


def utm_to_latlon(zone: int, easting: float, northing: float,
                  northern: bool = True):
    """UTM -> WGS84 (lat_deg, lon_deg)。"""
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    e2, ep2 = WGS84_E2, WGS84_EP2
    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0
    m = y / UTM_K0
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))
    mu = m / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e2 ** 2 / 64.0
                         - 5.0 * e2 ** 3 / 256.0))
    phi1 = (mu
            + (3.0 * e1 / 2.0 - 27.0 * e1 ** 3 / 32.0) * math.sin(2.0 * mu)
            + (21.0 * e1 ** 2 / 16.0 - 55.0 * e1 ** 4 / 32.0)
            * math.sin(4.0 * mu)
            + (151.0 * e1 ** 3 / 96.0) * math.sin(6.0 * mu)
            + (1097.0 * e1 ** 4 / 512.0) * math.sin(8.0 * mu))
    n1 = WGS84_A / math.sqrt(1.0 - e2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = ep2 * math.cos(phi1) ** 2
    r1 = WGS84_A * (1.0 - e2) / (1.0 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * UTM_K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 ** 2 - 9.0 * ep2)
        * d ** 4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 ** 2 - 252.0 * ep2
           - 3.0 * c1 ** 2) * d ** 6 / 720.0
    )
    lon = lon0 + (
        d - (1.0 + 2.0 * t1 + c1) * d ** 3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 ** 2 + 8.0 * ep2
           + 24.0 * t1 ** 2) * d ** 5 / 120.0
    ) / math.cos(phi1)
    return math.degrees(lat), math.degrees(lon)


# ---------------------------------------------------------------------------
# 旋转 <-> 航向/俯仰/横滚（ENU 约定）
# ---------------------------------------------------------------------------

def rotation_to_ypr(R: np.ndarray):
    """相机->世界旋转矩阵 -> (yaw, pitch, roll) 度。

    约定（ENU 世界系，相机 +Z 为光轴、+Y 朝下）：
      yaw   = 光轴方位角，自北顺时针（0-360）
      pitch = 光轴仰角，上仰为正
      roll  = 绕光轴旋转，光轴方向看逆时针为正
    """
    f = R[:, 2]
    yaw = math.degrees(math.atan2(f[0], f[1])) % 360.0
    pitch = math.degrees(math.atan2(f[2], math.hypot(f[0], f[1])))
    up_world = np.array([0.0, 0.0, 1.0])
    right0 = np.cross(f, up_world)
    if np.linalg.norm(right0) < 1e-9:     # 光轴竖直时横滚无定义，取 0
        return yaw, pitch, 0.0
    right0 = right0 / np.linalg.norm(right0)
    up_lvl = np.cross(right0, f)          # 相机水平时的"上"方向（ENU 左手系）
    cam_down = R[:, 1]                    # 相机 +Y 即"下"
    roll = math.degrees(math.atan2(
        float(np.dot(cam_down, right0)), -float(np.dot(cam_down, up_lvl))
    ))
    return yaw, pitch, roll


def ypr_to_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """rotation_to_ypr 的逆：航向/俯仰/横滚 -> 相机->世界旋转矩阵。"""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    f = np.array([
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
    ])
    up_world = np.array([0.0, 0.0, 1.0])
    right0 = np.cross(f, up_world)
    if np.linalg.norm(right0) < 1e-9:
        right0 = np.array([1.0, 0.0, 0.0])
    else:
        right0 = right0 / np.linalg.norm(right0)
    up_lvl = np.cross(right0, f)
    # 横滚绕光轴旋转：右轴与下轴一起转，保证三轴正交（ENU 左手系）
    right = math.cos(roll) * right0 + math.sin(roll) * up_lvl
    cam_down = math.sin(roll) * right0 - math.cos(roll) * up_lvl
    return np.column_stack([right, cam_down, f])


# ---------------------------------------------------------------------------
# 3D 相似变换（Umeyama/Kabsch + 尺度）
# ---------------------------------------------------------------------------

def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """最小二乘拟合 3D 相似变换：dst ~= s * R @ src + t。

    Args:
        src: [N,3] 源点（局部坐标）
        dst: [N,3] 目标点（地理/ENU 坐标）
    Returns:
        (scale, R, t, rms)：尺度、旋转、平移、拟合残差 RMS（米）。
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 3:
        raise ValueError("fit_similarity needs >= 3 points")
    ms = src.mean(axis=0)
    md = dst.mean(axis=0)
    src_c = src - ms
    dst_c = dst - md
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    dd = np.diag([1.0, 1.0, d])
    r = u @ dd @ vt
    var_src = (src_c ** 2).sum() / n
    scale = float(np.sum(s * dd) / var_src) if var_src > 1e-12 else 1.0
    t = md - scale * (r @ ms)
    resid = scale * (r @ src.T).T + t - dst
    rms = float(np.sqrt((resid ** 2).sum() / n))
    return scale, r, t, rms


def apply_similarity(points: np.ndarray, scale: float, R: np.ndarray,
                     t: np.ndarray) -> np.ndarray:
    """应用相似变换：p' = s * R @ p + t。支持 [N,3] 或 [3]。"""
    pts = np.asarray(points, float)
    return scale * (R @ pts.T).T + t


def transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray,
                             t: np.ndarray) -> np.ndarray:
    """把 COLMAP 相机->世界矩阵 [N,4,4] 按世界系相似变换更新。

    世界点 p' = s*R@p + t 时：R' = R @ R_cw，t' = s*R@t_cw + t。
    """
    c2w = np.asarray(c2w, float)
    out = c2w.copy()
    out[:, :3, :3] = R @ c2w[:, :3, :3]
    out[:, :3, 3] = scale * (R @ c2w[:, :3, 3].T).T + t
    return out


def qvec_to_rotmat(q) -> np.ndarray:
    """四元数 (w,x,y,z) -> 旋转矩阵（与 data.py 一致，避免循环依赖）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_qvec(R: np.ndarray):
    """旋转矩阵 -> 四元数 (w,x,y,z)（Shepperd 方法，数值稳定）。"""
    r = np.asarray(R, float)
    t = np.trace(r)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)
    ok = True

    # 1) ECEF 往返
    lat0, lon0, h0 = 30.6599, 104.0633, 500.0
    ecef = geodetic_to_ecef(lat0, lon0, h0)
    lat1, lon1, h1 = ecef_to_geodetic(*ecef)
    err = max(abs(lat1 - lat0), abs(lon1 - lon0), abs(h1 - h0))
    ok &= err < 1e-7
    print(f"[geo] ECEF 往返误差: lat/lon {err:.2e} deg, h {h1-h0:.2e} m")

    # 2) ENU 往返
    e, n, u = 123.4, -56.7, 89.0
    lat2, lon2, h2 = enu_to_geodetic(e, n, u, lat0, lon0, h0)
    e2, n2, u2 = geodetic_to_enu(lat2, lon2, h2, lat0, lon0, h0)
    err = max(abs(e2 - e), abs(n2 - n), abs(u2 - u))
    ok &= err < 1e-6
    print(f"[geo] ENU 往返误差: {err:.2e} m")

    # 3) UTM 已知点（40.5N, 74.5W = 18 带中央经线 -75° 以东 0.5°）
    #    easting 应在 500km 假东以东约 42km，northing 约 4484.1km
    zone, east, north, northern = latlon_to_utm(40.5, -74.5)
    ok &= zone == 18 and northern
    ok &= 542000.0 < east < 543000.0 and 4483000.0 < north < 4484000.0
    print(f"[geo] UTM 已知点: zone {zone} E={east:.1f} N={north:.1f} "
          f"(期望 E≈542.4km, N≈4483.4km)")

    # 4) UTM 往返（多个点）
    max_utm = 0.0
    for _ in range(50):
        lat = rng.uniform(-80, 84)
        lon = rng.uniform(-180, 180)
        z, e, n_, north_ = latlon_to_utm(lat, lon)
        la, lo = utm_to_latlon(z, e, n_, north_)
        max_utm = max(max_utm, math.hypot(la - lat, lo - lon))
    ok &= max_utm < 1e-7
    print(f"[geo] UTM 往返最大误差: {max_utm:.2e} deg")

    # 5) ypr 往返
    R0 = ypr_to_rotation(123.4, 5.6, -7.8)
    yaw, pitch, roll = rotation_to_ypr(R0)
    R1 = ypr_to_rotation(yaw, pitch, roll)
    ok &= np.abs(R1 - R0).max() < 1e-9
    print(f"[geo] ypr 往返: yaw {yaw:.3f} pitch {pitch:.3f} roll {roll:.3f}, "
          f"R 最大误差 {np.abs(R1 - R0).max():.1e}")

    # 6) 相似变换拟合恢复
    scale_true = 1.25
    r_true = ypr_to_rotation(30, -10, 5)
    t_true = np.array([100.0, -50.0, 20.0])
    src = rng.normal(size=(100, 3)) * 10.0
    dst = apply_similarity(src, scale_true, r_true, t_true)
    s2, r2, t2, rms = fit_similarity(src, dst)
    ok &= rms < 1e-9
    ok &= abs(s2 - scale_true) < 1e-9 and np.abs(r2 - r_true).max() < 1e-9
    print(f"[geo] 相似变换恢复: s {s2:.6f} (真值 {scale_true}), "
          f"R 误差 {np.abs(r2 - r_true).max():.1e}, RMS {rms:.1e}")

    # 7) c2w 相似变换一致性（与逐点变换对比）
    c2w = np.tile(np.eye(4), (3, 1, 1))
    c2w[:, :3, 3] = rng.normal(size=(3, 3))
    c2w2 = transform_c2w_similarity(c2w, scale_true, r_true, t_true)
    for i in range(3):
        p = c2w[i, :3, 3]
        p2 = apply_similarity(p, scale_true, r_true, t_true)
        ok &= np.abs(c2w2[i, :3, 3] - p2).max() < 1e-9
    print("[geo] c2w 相似变换一致")

    print("[geo] self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
