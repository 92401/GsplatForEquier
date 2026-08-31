"""Diagnose a trained run: splat statistics + camera trajectory baseline.

Usage:
  python diag.py <ckpt.pt> [--data-dir D:\\gaussian_splatting\\spirula]

Prints distribution of Gaussian scales/opacities/positions and the camera
trajectory geometry, to tell apart "training didn't converge" from
"the capture has almost no parallax".
"""

import argparse

import numpy as np
import torch

from data import build_camtoworlds, load_sparse


def fmt(x):
    return ", ".join(f"{v:.6g}" for v in x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data-dir", default=r"D:\gaussian_splatting\spirula")
    args = ap.parse_args()

    # ---- splat stats ------------------------------------------------------
    sd = torch.load(args.ckpt, map_location="cpu")
    n = len(sd["means"])
    scales = torch.exp(sd["scales"])          # world units
    opac = torch.sigmoid(sd["opacities"])
    means = sd["means"]

    print(f"num_gaussians = {n}")
    print("exp(scales):  "
          f"mean={scales.mean():.5f} median={scales.median():.5f} "
          f"p95={scales.quantile(0.95):.5f} max={scales.max():.5f}")
    print("opacity:      "
          f"mean={opac.mean():.5f} median={opac.median():.5f} "
          f"frac<0.01={100*(opac<0.01).float().mean():.1f}% "
          f"frac<0.1={100*(opac<0.1).float().mean():.1f}% "
          f"frac>0.5={100*(opac>0.5).float().mean():.1f}%")
    print(f"means bbox:   min={fmt(means.min(0).values)}")
    print(f"              max={fmt(means.max(0).values)}")
    extent = (means.max(0).values - means.min(0).values)
    print(f"              extent={fmt(extent)}")
    r = means.norm(dim=-1)
    print(f"means radius: median={r.median():.4f} p95={r.quantile(0.95):.4f} "
          f"max={r.max():.4f}")

    # ---- camera trajectory ------------------------------------------------
    sparse = load_sparse(args.data_dir)
    images = sorted(sparse["images"], key=lambda im: im["name"])
    c2w = build_camtoworlds(images)
    centers = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    path = float(steps.sum())
    print(f"\ncameras: {len(centers)}")
    print(f"trajectory: total_path={path:.3f} mean_step={steps.mean():.5f} "
          f"median_step={np.median(steps):.5f} max_step={steps.max():.5f}")
    print(f"            zero_step_frac="
          f"{100*np.mean(steps < 1e-6):.1f}%")
    from_center = np.linalg.norm(centers - centers.mean(0), axis=1)
    print(f"cam radius:  mean={from_center.mean():.4f} "
          f"max={from_center.max():.4f}")
    n_share = sum(1 for i in range(1, len(centers))
                  if np.linalg.norm(centers[i] - centers[i - 1]) < 0.02)
    print(f"near-duplicate consecutive cameras (<0.02 units): {n_share}")


if __name__ == "__main__":
    main()
