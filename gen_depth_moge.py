"""用 MoGe-2 为全景数据生成稠密深度（每张全景 6 个面各一张深度图）。

用法:
  python gen_depth_moge.py --data-dir D:\\gaussian_splatting\\spirula \
      --face-size 512 --out-dir D:\\gaussian_splatting\\spirula\\depths

输出: <out-dir>/<图片名>.pt，内容是 [6, S, S] float32 的度量深度（米）。
训练时用 --depth-dir 指向该目录即可启用稠密深度监督；
脚本会把深度按每面与 COLMAP 稀疏深度的中位比例对齐到归一化场景尺度。
"""

import argparse
import os

import numpy as np
import torch

from data import EquirectCamera, load_sparse, load_pano, warp_pano_to_faces


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=r"D:\gaussian_splatting\spirula")
    p.add_argument("--face-size", type=int, default=512,
                   help="深度图分辨率（建议 512；训练时会自动缩放到训练面尺寸）")
    p.add_argument("--out-dir", default="",
                   help="输出目录（默认 <data-dir>/depths）")
    p.add_argument("--model", default="Ruicheng/moge-2-vitb-normal",
                   help="HF 模型名，可选 vitb/vitl")
    p.add_argument("--max-images", type=int, default=0,
                   help="只处理前 N 张（0=全部）")
    p.add_argument("--num-tokens", type=int, default=1800,
                   help="MoGe 基础 token 数（越大越精细越慢）")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="fp16 推理")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    cfg = parse_args()
    from moge.model import import_model_class_by_version

    out_dir = cfg.out_dir or os.path.join(cfg.data_dir, "depths")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    print("[moge] loading model", cfg.model)
    model = import_model_class_by_version("v2").from_pretrained(cfg.model)
    model = model.to(device).eval()
    if cfg.fp16:
        model.half()

    sparse = load_sparse(cfg.data_dir)
    cam = sparse["cameras"][0]
    equirect = EquirectCamera(cam["width"], cam["height"])
    images = sorted(sparse["images"], key=lambda im: im["name"])
    if cfg.max_images > 0:
        images = images[: cfg.max_images]
    image_dir = os.path.join(cfg.data_dir, "images")

    S = cfg.face_size
    for im in images:
        name = im["name"]
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".pt")
        if os.path.exists(out_path):
            print(f"[moge] skip {name} (exists)")
            continue
        pano = load_pano(os.path.join(image_dir, name))
        faces = warp_pano_to_faces(pano, equirect, S, device=cfg.device)  # [6,S,S,3]
        depths = np.zeros((6, S, S), np.float32)
        for f in range(6):
            img = faces[f].permute(2, 0, 1).contiguous()  # [3,S,S] float[0,1]
            out = model.infer(
                img,
                num_tokens=cfg.num_tokens,
                fov_x=90.0,          # 面是 90° FOV，已知焦距可提升精度
                use_fp16=cfg.fp16,
            )
            d = out["depth"].float().cpu().numpy()  # [S,S] 米
            depths[f] = d
            print(f"[moge] {name} face {f}: depth range "
                  f"{np.nanmin(d):.3f} ~ {np.nanmax(d):.3f} m")
        torch.save(torch.from_numpy(depths), out_path)
        print(f"[moge] saved {out_path}")

    print("[moge] done")


if __name__ == "__main__":
    main()
