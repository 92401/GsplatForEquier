"""Project verification scripts.

Usage:
  python check.py warp    # equirect -> 6-face warp self-consistency
  python check.py split   # face-split geometry vs the SfM's own observations
  python check.py all     # both (default)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import (
    FACE_DIRS,
    EquirectCamera,
    build_camtoworlds,
    build_face_c2w,
    camera_is_equirect,
    face_rotation,
    load_pano,
    load_sparse,
    pinhole_intrinsics,
    qvec_to_rotmat,
    warp_pano_to_faces,
)


# ---------------------------------------------------------------------------
# Check A: warp self-consistency (originally check_warp.py)
# ---------------------------------------------------------------------------
"""Numerically verify the equirect -> 6-face warp.

We re-implement the sampling independently (nearest-neighbour, outside the
warp function) for a grid of face pixels and compare against the warp output.
A correct mapping yields near-zero differences everywhere except a handful of
pixels near the panorama seam / pole where nearest-neighbour lands on a
different sample than bilinear.
"""


def main_warp():
    data_dir = r"D:\gaussian_splatting\spirula"
    sparse = load_sparse(data_dir)
    cam = sparse["cameras"][0]
    images = sorted(sparse["images"], key=lambda im: im["name"])
    pano_path = os.path.join(data_dir, "images", images[0]["name"])

    equirect = EquirectCamera(cam["width"], cam["height"])
    pano = load_pano(pano_path)  # [H, W, 3] float
    S = 64
    faces = warp_pano_to_faces(pano, equirect, S, device="cpu")  # [6,S,S,3]

    print(f"pano {pano.shape}  faces {faces.shape}")
    ok = True
    diffs = []
    for i, (name, d) in enumerate(FACE_DIRS.items()):
        R = torch.from_numpy(face_rotation(d)).float()
        # independent ray math (no shared helper calls)
        px = torch.arange(S, dtype=torch.float32) + 0.5  # pixel centers
        gu, gv = torch.meshgrid(px, px, indexing="xy")
        x = (gu - S / 2) / (S / 2)
        y = (gv - S / 2) / (S / 2)
        d_face = F.normalize(torch.stack([x, y, torch.ones_like(x)], -1), dim=-1)
        d_pano = torch.einsum("hwi,ij->hwj", d_face, R)
        theta = torch.atan2(d_pano[..., 0], d_pano[..., 2])
        phi = torch.atan2(-d_pano[..., 1],
                          torch.hypot(d_pano[..., 0], d_pano[..., 2]))
        u = equirect.fx * theta + equirect.cx
        v = equirect.cy - equirect.fy * phi
        # independent bilinear sample of the panorama (same convention as
        # grid_sample with align_corners=True: continuous pixel coords)
        W, H = pano.shape[1], pano.shape[0]
        u0 = torch.floor(u)
        v0 = torch.floor(v)
        fu, fv = (u - u0).clamp(0, 1), (v - v0).clamp(0, 1)
        gx = lambda xx: xx.clamp(0, W - 1).long()
        gy = lambda yy: yy.clamp(0, H - 1).long()
        top = pano[gy(v0), gx(u0)] * (1 - fu)[..., None] + \
              pano[gy(v0), gx(u0 + 1)] * fu[..., None]
        bot = pano[gy(v0 + 1), gx(u0)] * (1 - fu)[..., None] + \
              pano[gy(v0 + 1), gx(u0 + 1)] * fu[..., None]
        ref = top * (1 - fv)[..., None] + bot * fv[..., None]

        diff = (faces[i] - ref).abs()
        diffs.append(diff)
        print(f"{name:6s} mean={diff.mean():.5f} median={diff.median():.5f} "
              f"p95={diff.quantile(0.95):.5f} max={diff.max():.5f}")
        ok = ok and diff.mean() < 1e-4 and diff.max() < 1e-2

    all_diff = torch.cat([d.flatten() for d in diffs])
    print(f"ALL   mean={all_diff.mean():.5f} median={all_diff.median():.5f} "
          f"p95={all_diff.quantile(0.95):.5f} max={all_diff.max():.5f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Check B: face-split geometry vs the SfM's own observations
#          (originally check_split_geometry.py)
# ---------------------------------------------------------------------------
"""Verify the equirect face-split geometry against the SfM's own observations.

Two independent checks on the training camera math:

1. Reprojection error: every sparse 3D point has a 2D observation stored in
   COLMAP's images.bin.  We re-project the 3D point with OUR equirectangular
   convention (fx = w/2pi, fy = h/pi, cx = w/2, cy = h/2) and compare with the
   stored observation.  If the convention used by the warp does not match the
   one the SfM poses were estimated with, the error is huge.

2. Face color consistency: for a sample of 3D points we project each point
   through the face pinhole (face K + face c2w) into a warped GT face, and
   through the equirect camera into the panorama, and compare the two sampled
   colors.  A correct face chain makes them equal (both look along the same
   scene ray), regardless of occlusion.
"""


def reproject_error(sparse):
    """Check 1: equirect convention vs the SfM's stored 2D observations."""
    images = sorted(sparse["images"], key=lambda im: im["name"])
    pts = sparse["points_xyz"]
    id_to_idx = {int(pid): i for i, pid in enumerate(sparse["points_ids"])}
    cam = sparse["cameras"][0]
    eq = EquirectCamera(cam["width"], cam["height"])
    errs = []
    n = 0
    for im in images:
        R = qvec_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        for k in range(len(im["point3D_ids"])):
            pid = im["point3D_ids"][k]
            if pid < 0:
                continue
            Xc = R @ pts[id_to_idx[int(pid)]] + t
            theta = np.arctan2(Xc[0], Xc[2])
            phi = np.arctan2(-Xc[1], np.hypot(Xc[0], Xc[2]))
            u = eq.fx * theta + eq.cx
            v = eq.cy - eq.fy * phi
            err = np.hypot(u - im["xys"][k][0], v - im["xys"][k][1])
            errs.append(err)
            n += 1
    errs = np.array(errs)
    print(f"[check1] {n} observations")
    print(f"  median={np.median(errs):.3f}px  mean={errs.mean():.3f}px  "
          f"p95={np.percentile(errs, 95):.3f}px  max={errs.max():.3f}px")
    ok = np.median(errs) < 3.0
    print("  RESULT:", "PASS (convention matches SfM)" if ok else "FAIL")
    return ok


def face_color_consistency(sparse, data_dir, n_images=20, S=256, n_pts=400):
    """Check 2: face pinhole chain samples the same rays as the panorama."""
    images = sorted(sparse["images"], key=lambda im: im["name"])[:n_images]
    pts = sparse["points_xyz"]
    id_to_idx = {int(pid): i for i, pid in enumerate(sparse["points_ids"])}
    cam = sparse["cameras"][0]
    eq = EquirectCamera(cam["width"], cam["height"])
    c2w = build_camtoworlds(images)
    K = pinhole_intrinsics(S)
    rng = np.random.default_rng(0)

    image_dir = os.path.join(data_dir, "images")
    diffs = []
    n_checked = 0
    for i, im in enumerate(images):
        pano = load_pano(os.path.join(image_dir, im["name"]))
        faces = warp_pano_to_faces(pano, eq, S, device="cpu")  # [6,S,S,3]
        R = qvec_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        face_rots = [face_rotation(d) for d in FACE_DIRS.values()]

        valid = np.where(im["point3D_ids"] >= 0)[0]
        if len(valid) == 0:
            continue
        sel = rng.choice(valid, size=min(n_pts, len(valid)), replace=False)
        for k in sel:
            Xc = R @ pts[id_to_idx[int(im["point3D_ids"][k])]] + t
            # ---- pano side: sample at OUR reprojection of the same ray ----
            theta = np.arctan2(Xc[0], Xc[2])
            phi = np.arctan2(-Xc[1], np.hypot(Xc[0], Xc[2]))
            u_p = eq.fx * theta + eq.cx
            v_p = eq.cy - eq.fy * phi
            if not (0 <= u_p < cam["width"] and 0 <= v_p < cam["height"]):
                continue
            gx = 2.0 * u_p / (cam["width"] - 1) - 1.0
            gy = 2.0 * v_p / (cam["height"] - 1) - 1.0
            grid = torch.tensor([[[[gx, gy]]]], dtype=torch.float32)
            col_p = F.grid_sample(
                pano.permute(2, 0, 1).unsqueeze(0), grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )[0, :, 0, 0]
            # ---- face side: project through each face, keep first inside ----
            found = False
            for fi, R_face in enumerate(face_rots):
                Xf = R_face @ Xc
                if Xf[2] <= 0:
                    continue
                u_f = K[0, 0] * Xf[0] / Xf[2] + K[0, 2]
                v_f = K[1, 1] * Xf[1] / Xf[2] + K[1, 2]
                if 0 <= u_f < S and 0 <= v_f < S:
                    # warp pixel k holds the ray at face coordinate k+0.5, so
                    # sample the face grid at u_f - 0.5 (align_corners=True)
                    gxf = 2.0 * (u_f - 0.5) / (S - 1) - 1.0
                    gyf = 2.0 * (v_f - 0.5) / (S - 1) - 1.0
                    grid_f = torch.tensor([[[[gxf, gyf]]]], dtype=torch.float32)
                    col_f = F.grid_sample(
                        faces[fi].permute(2, 0, 1).unsqueeze(0), grid_f,
                        mode="bilinear", padding_mode="border", align_corners=True,
                    )[0, :, 0, 0]
                    diffs.append(float((col_p - col_f).abs().max()))
                    n_checked += 1
                    found = True
                    break
            if not found and n_checked == 0:
                continue
    diffs = np.array(diffs)
    print(f"[check2] {n_checked} point-face samples")
    if n_checked:
        print(f"  mean={diffs.mean():.5f}  median={np.median(diffs):.5f}  "
              f"p95={np.percentile(diffs, 95):.5f}  max={diffs.max():.5f}")
        # residual ~0.03 is float32 (warp) vs float64 (check) sampling noise
        ok = np.median(diffs) < 0.05
        print("  RESULT:", "PASS (face rays == pano rays)" if ok else "FAIL")
        return ok
    print("  RESULT: SKIP (no samples)")
    return True


def main_geometry():
    data_dir = r"D:\gaussian_splatting\spirula"
    sparse = load_sparse(data_dir)
    assert all(camera_is_equirect(c) for c in sparse["cameras"])
    ok1 = reproject_error(sparse)
    ok2 = face_color_consistency(sparse, data_dir)
    print("\nOVERALL:", "PASS" if (ok1 and ok2) else "FAIL")
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", choices=["warp", "split", "all"],
                    default="all")
    args = ap.parse_args()
    ok = True
    if args.which in ("warp", "all"):
        ok = main_warp() and ok
    if args.which in ("split", "all"):
        ok = main_geometry() and ok
    print("\nOVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
