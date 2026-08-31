# GsplatForEquier（pano_gsplat）— 等距柱状/透视 3DGS 训练器

基于 gsplat 渲染内核开发的 3D 高斯泼溅（3DGS）训练器，适配**分幅（等距柱状
全景）和不分幅（透视）两类数据。技术路线参考 Spirula Studio 的 360
方案：等距柱状相机 → 6 面透视切分 → 稀疏点初始化 → 训练，同时支持普通透视
相机数据集。

## 1. 项目简介

输入：COLMAP 稀疏重建结果（相机位姿 + 稀疏点云）+ 对应图像。

输出：训练好的高斯模型（`ckpt_final.pt` 检查点 + `splat.ply` 可查看模型）。

项目把“360 影像 → 高斯模型”的全流程（数据读取、全景切面、高斯初始化、训练、
增密、评测、诊断）收敛在一个可命令行调用的 Python 项目里，所有功能均可通过
命令行参数或配置文件开启/关闭。

## 2. 功能列表

| 功能 | 说明 |
|---|---|
| 等距柱状全景训练 | 每张 360 影像切成 6 个 90° 针孔面（`warp_pano_to_faces`），与 Spirula 同路线 |
| 透视相机训练 | COLMAP model 0-4（PINHOLE 族）直接用原图 + 相机内参，不切面 |
| 双增密策略 | `DefaultStrategy`（Inria clone/split）与 `MCMCStrategy`（MCMC 增密） |
| AbsGS | 用绝对 2D 梯度做增密，恢复细碎细节（配合 default 策略） |
| 显存优化 | `--packed` 打包光栅化 + `--sparse-grad` 稀疏梯度，省显存 |
| 验证集评测 | 留出验证图，周期计算 PSNR/SSIM/LPIPS，写 `metrics.json` |
| PPISP 光度校正 | 可学习的曝光/晕影/颜色/CRF 校正，两种绑定（per_view / hybrid） |
| bilagrid 曝光校正 | 逐视角 bilateral grid 曝光/白平衡校正 |
| 深度监督 | COLMAP 稀疏深度 + MoGe-2 稠密深度（视差 L1） |
| 位姿优化 | 逐视角 6D 位姿增量，全景模式 6 面共享（rig 感知，不破坏刚体关系） |
| 随机背景 | 每步随机背景色，抑制透明漂浮物 |
| 粗到细 | 先低分辨率面热身，再切完整分辨率，省显存加速收敛 |
| 面缓存 | 重采样后的面缓存到磁盘（uint8 .pt），重复实验省时间 |
| 配置文件 | JSON/YAML 配置加载，命令行参数优先，支持导出生效配置 |
## 3. 技术路线

```text
sparse/0（COLMAP，EQUIRECTANGULAR model 17 或 PINHOLE 族）
  │  data.load_sparse()
  ▼
全景：每张 → 6 个 90° pinhole 面（warp_pano_to_faces，像素中心约定）
透视：每张直接用（K=1，无切面，自动缩放 K 适配图片尺寸）
  ▼
稀疏点 → 初始高斯（位置=点坐标，颜色→SH0，尺度=4-NN 距离）
  ▼
gsplat 光栅化（pinhole）+ L1/SSIM 损失 + Adam（按 6 路并行监督缩放超参）
  ▼
DefaultStrategy（clone/split）或 MCMCStrategy 增密
  ▼
ckpt_final.pt + splat.ply
```

## 4. 目录结构（模块职责）

```text
pano_gsplat/
├── train.py           训练入口：参数/配置解析 + 高斯初始化 + 优化器/策略
│                      + 训练主循环 + 验证评测 + 保存
├── data.py            数据层：COLMAP 二进制读取、相机数学、全景切面、
│                      图像加载与归一化、面缓存
├── pose.py            位姿优化模块（rig 感知：全景 6 面共享一个位姿增量）
├── check.py           验证入口：切面自洽性 / 与 SfM 观测一致性
├── diag.py            诊断入口：高斯尺度/不透明度/位置分布 + 相机轨迹基线
├── gen_depth_moge.py  MoGe-2 稠密深度生成（每张全景 6 个面各一张深度 .pt）
├── lib_bilagrid.py    第三方：3D bilateral grid（Apache-2.0，gsplat examples 原样拷贝）
├── configs/           配置示例（example.yaml）
├── requirements.txt   依赖清单
└── README.md          本文档
```

## 5. 环境配置

### 5.1 基础环境

建议直接使用现有的 `gsplat` conda 环境：

```powershell
...\.conda\envs\gsplat\python.exe
```

需要安装的依赖（见 `requirements.txt`）：
### 5.2 可选依赖
**MoGe-2（稠密深度监督）**：Windows 可装版本，固定到 2025-11-02 的提交
（主分支已是 MoGe-3，依赖 Triton 在 Windows 上不便）：

```powershell
git clone https://github.com/microsoft/MoGe.git your_path...\MoGe
git -C ...\MoGe checkout 07444410f1e33f402353b99d6ccd26bd31e469e8
...\.conda\envs\gsplat\python.exe -m pip install -e your_path...\MoGe
```

**PPISP（光度校正）**：nv-tlabs/ppisp（Apache-2.0），需要编译 CUDA 扩展。
Windows 编译要点：VS2022（MSVC 14.44）的 STL 拒绝 CUDA<12.4，而 torch cu118
又禁止 CUDA 12.x 的 nvcc，因此必须用 VS2019（MSVC 14.29）+ CUDA 11.8 编译：

```powershell
git clone https://github.com/nv-tlabs/ppisp.git your_path...\ppisp
# 用 VS2019 的 vcvars64 环境 + DISTUTILS_USE_SDK=1 执行：
#   python setup.py bdist_wheel
# 然后安装生成的 wheel（--no-build-isolation）
...\.conda\envs\gsplat\python.exe -m pip install --force-reinstall --no-deps `
  your_path...\ppisp\dist\ppisp-1.2.1-cp310-cp310-win_amd64.whl
```

## 6. 数据要求

两类 COLMAP 稀疏模型均可直接使用：

**A. 等距柱状全景**（每张切成 6 个 90° 面）：

```text
<data_dir>/
  images/      等距柱状图（2:1，文件名与 sparse 一致）
  sparse/0/    cameras.bin / images.bin / points3D.bin
               相机必须是 EQUIRECTANGULAR（COLMAP model 17，参数即宽高）
```

**B. 透视相机**（K=1 不切面，直接用原图 + 相机自身内参）：

```text
<data_dir>/
  images/      透视图像（文件名与 sparse 一致）
  sparse/0/    相机为 COLMAP model 0-4（SIMPLE_PINHOLE/PINHOLE/
               SIMPLE_RADIAL/RADIAL/OPENCV），畸变忽略；
               图片尺寸与标定尺寸不一致时自动缩放 K
```

## 7. 正式训练

全部 354 张全景、1024 面、3 万步、MCMC 增密（建议开面缓存）：

```powershell
python train.py `
  --data-dir D:\gaussian_splatting\spirula `
  --face-size 1024 --steps 30000 --strategy mcmc `
  --face-cache D:\gaussian_splatting\pano_gsplat\face_cache `
  --out-dir \outputs
```

透视数据示例：

```powershell
python train.py `
  --data-dir D:\gaussian_splatting\GGPS-data\datasets\FTP\gsplat_ftp `
  --steps 30000 --out-dir \outputs
```

## 8. 配置文件

支持 JSON 或 YAML，键 = 参数名（连字符键 `data-dir` 等价于 `data_dir`），
**命令行显式参数优先于配置文件**。示例见 `configs/example.yaml`：

```powershell
# 用配置跑（示例配置里已含 354 张全景的正式训练参数）
python train.py --config configs\example.yaml

# 覆盖个别参数（比如先小规模测试）
python train.py `
  --config configs\example.yaml --max-images 4 --face-size 256 --steps 300

# 导出生效配置，方便保存/复用
\python train.py `
  --config configs\example.yaml --dump-config outputs\my_run.json
```

## 9. 验证与评测

训练时用 `--test-every N` 每 N 张留 1 张做验证集，`--eval-every` 周期评测，
结果写入 `outputs/<run>/metrics.json`（PSNR/SSIM/LPIPS）：

```powershell
python train.py `
  --config configs\example.yaml --test-every 8 --eval-every 2000 `
  --save-eval-images
```

`--save-eval-images` 会额外保存验证视图的 GT|渲染对比图
（`eval_<step>_v<idx>.png`）。

## 10. 工具脚本

```powershell
# 切面自洽性校验（warp 重投影 / SfM 观测一致性），改动切面逻辑后必跑
python check.py all

# 模型诊断：高斯尺度/不透明度/位置分布 + 相机轨迹基线统计
python diag.py outputs\full\ckpt_final.pt

# MoGe 稠密深度生成（每张全景 6 个面各一张度量深度 .pt）
python gen_depth_moge.py `
  --data-dir D:\gaussian_splatting\spirula --face-size 512 `
  --out-dir D:\gaussian_splatting\spirula\depths
```

## 11. 参数详解

### 11.1 基础参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--config` | 空 | JSON/YAML 配置文件，提供默认值（命令行显式参数优先） |
| `--dump-config` | 空 | 把生效配置导出为 JSON 并退出 |
| `--data-dir` | `D:\gaussian_splatting\spirula` | 数据集目录（含 images/ 和 sparse/0） |
| `--image-dir` | 空 | 覆盖图片目录（默认 `<data-dir>/images`，可用 images_2/4 等） |
| `--out-dir` | `outputs\run` | 输出目录（ckpt/ply/metrics/对比图） |
| `--max-images` | 0 | 只用前 N 张图（0=全部），调试用 |
| `--face-size` | 512 | 每面边长（90° FOV，焦距 = S/2）。3840×1920 全景建议 1024 |
| `--batch-size` | 1 | 每步全景数（每张展开成 6 个面，即每步 6×batch 个面） |
| `--steps` | 30000 | 总训练步数 |
| `--sh-degree` | 3 | 球谐阶数（0/1/2/3）：越高颜色越丰富，每高斯参数越多 |
| `--strategy` | default | 增密策略：`default`=Inria clone/split，`mcmc`=MCMC 增密 |
| `--seed` | 42 | 随机种子（复现实验用） |
| `--device` | cuda:0 | 训练设备 |

### 11.2 渲染与损失

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--antialiased` | 关 | gsplat 的 Mip-Splatting 补偿因子（更抗锯齿，PSNR/SSIM 略降） |
| `--ssim-lambda` | 0.2 | 损失中 SSIM 占比：loss = (1-λ)·L1 + λ·(1-SSIM) |
| `--near-plane` | 0.01 | 近裁剪面（归一化场景尺度下） |
| `--far-plane` | 1e10 | 远裁剪面 |

### 12.3 初始化与正则

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--init-opacity` | 自动 | 初始不透明度（MCMC 自动 0.5，default 自动 0.1） |
| `--init-scale` | 自动 | 初始尺度乘子（MCMC 自动 0.1，default 自动 1.0） |
| `--scale-reg` | 自动 | 尺度正则权重（MCMC 自动 0.01，default 自动 0） |
| `--opacity-reg` | 自动 | 不透明度正则权重（MCMC 自动 0.01，default 自动 0） |

### 12.4 学习率

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--means-lr` | 1.6e-4 | 高斯中心学习率（按场景尺度缩放） |
| `--scales-lr` | 5e-3 | 尺度（log 空间）学习率 |
| `--opacities-lr` | 5e-2 | 不透明度（logit 空间）学习率 |
| `--quats-lr` | 1e-3 | 旋转四元数学习率 |
| `--sh0-lr` | 2.5e-3 | 球谐 DC 系数学习率 |
| `--shn-lr` | 1.25e-4 | 球谐高阶系数学习率（sh0 的 1/20） |

说明：全景模式每步 6 面并行监督，优化器的 Adam 超参（lr/eps/betas）会按
`sqrt(batch_size × 6)` 自动放大，保证 6 路并行监督下稳定收敛。

### 11.5 增密与效率

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--absgrad` | 关 | AbsGS：用绝对 2D 梯度增密（配合 default 策略） |
| `--packed` | 关 | 打包光栅化：把“tile×槽位”稠密表改成扁平+偏移表，省掉空槽 |
| `--sparse-grad` | 关 | 稀疏梯度：只给可见高斯保留梯度（需配合 `--packed`） |

### 11.6 评测

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--test-every` | 0 | 每 N 张图留 1 张做验证集（0=全部训练） |
| `--eval-every` | 0 | 每 N 步跑一次验证评测（0=只在最后） |
| `--eval-max-images` | 20 | 每次评测最多渲染的验证图数 |
| `--save-eval-images` | 关 | 保存验证 GT\|渲染对比图 |

### 11.7 粗到细

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--coarse-face-size` | 0 | 第一阶段面的边长（0=关闭粗到细） |
| `--coarse-steps` | 0 | 多少步后从粗分辨率切到完整分辨率 |

### 11.8 光度校正

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--bilagrid` | 关 | 逐视角 bilateral grid 曝光/白平衡校正 |
| `--bilagrid-shape` | 16,16,8 | bilagrid 网格 X,Y,W |
| `--ppisp` | 关 | PPISP 逐视角光度校正（曝光/晕影/颜色/CRF） |
| `--ppisp-mode` | per_view | `per_view`=每个 post-split 视角独立 slot（Spirula）；`hybrid`=曝光/颜色按全景共享、晕影/CRF 按面方向 |
| `--ppisp-lr` | 0.002 | PPISP 主参数学习率 |
| `--ppisp-reg-scale` | 1.0 | PPISP 正则损失权重 |

### 11.9 深度监督

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--depth-supervision-weight` | 0 | 深度监督权重（>0 启用；用 COLMAP 稀疏观测深度） |
| `--depth-dir` | 空 | MoGe 稠密深度目录（`gen_depth_moge.py` 输出），启用逐像素深度监督 |

### 11.10 位姿 / 背景 / 缓存

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--pose-opt` | 关 | 逐视角位姿优化（全景模式 6 面共享一个位姿增量，rig 感知） |
| `--pose-opt-lr` | 1e-5 | 位姿优化学习率 |
| `--random-bkgd` | 关 | 每步随机背景色，抑制透明漂浮物 |
| `--face-cache` | 空 | 把重采样后的面缓存到磁盘（uint8 .pt），重跑省时间 |
| `--save-every` | 0 | 每 N 步保存 ckpt+ply（0=只在最后） |
| `--preview-every` | 0 | 每 N 步保存 GT\|渲染对比图 |

## 12. 关键概念说明

- **分幅（face split）**：等距柱状图直接光栅化会有极点畸变，项目按 Spirula
  方案切成 6 个 90° 针孔面，每个面用普通 pinhole 相机渲染，6 面共享同一全景
  位姿（rig 关系）。
- **球谐（SH）**：高斯颜色用球谐系数表示，`--sh-degree` 越高视角相关颜色越
  精细，但参数量按 (k+1)²×3 增长（0 阶=3、1 阶=12、2 阶=27、3 阶=48 个 float）。
- **增密策略**：`default` 按 2D 梯度阈值复制小高斯/分裂大高斯并剪掉低不透明度
  高斯（原版 3DGS）；`mcmc` 用 MCMC 采样的方式来增密，配合更小的初始尺度和
  正则使用，细节更丰富。
- **packed/sparse-grad**：packed 把光栅化的内存布局从稠密 tile 表变成扁平表；
  sparse-grad 让反向只更新“本步被光栅化到的”高斯，两者配合能显著降低显存。
- **PPISP 两种绑定**：`per_view` 每张面独立一套光度参数（自由度大、易过拟合）；
  `hybrid` 曝光/白平衡按全景共享、晕影/CRF 按面方向（物理更合理、参数更稳）。
## 14. 修复记录（重要）

1. **像素中心约定**：GT 重采样改为像素中心（u = i + 0.5），与 gsplat 光栅化
   和 Spirula 的 WarpFace 一致；早期角落约定导致 GT 与渲染错位半像素。
2. **MCMC 超参**：MCMC 自动套用配套超参（init_scale=0.1、init_opacity=0.5、
   scale_reg=0.01、opacity_reg=0.01），否则噪声项与尺度形成正反馈，高斯飞走。
3. **分幅旋转方向**：`warp_pano_to_faces` 的面→全景射线旋转方向写反，导致
   left/right/up/down 四个面 GT 左右/上下对调；修复后同配置 2000 步 PSNR
   从 19.9 → 25.5 dB。由 `check.py split` 守护。
4. **深度监督量纲**：统一用归一化后的位姿/点云计算深度，MoGe 稠密深度也按
   面与稀疏深度中位比例对齐到同一场景尺度。

## 15. 已知限制

- 无掩膜（SAM/几何模板）、无法线监督、无浏览器 viewer；
- 8GB 显存下大场景增密后显存吃紧，可先用 packed+sparse-grad 缓解
  （主机卸载实验见 `host-offload` 分支）。
