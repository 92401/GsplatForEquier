# pano_gsplat — 等距柱状/透视 3DGS 训练器（gsplat 内核）

一个"360 影像 → 高斯模型"训练项目，用 gsplat 做渲染内核，
参考 Spirula Studio 的 360 技术路线（等距柱状相机 → 6 面透视切分 →
稀疏点初始化 → 训练），同时支持普通透视相机数据集。

## 技术路线

```
sparse/0（COLMAP，EQUIRECTANGULAR model 17 或 PINHOLE）
  │  data.load_sparse()
  ▼
全景：每张 → 6 个 90° pinhole 面（warp_pano_to_faces）
透视：每张直接用（K=1，无切面）
  ▼
稀疏点 → 初始高斯（位置=点坐标，颜色→SH0，尺度=4-NN 距离，不透明度 0.1/0.5）
  ▼
gsplat 光栅化（pinhole）+ L1/SSIM + Adam
  ▼
DefaultStrategy（clone/split）或 MCMCStrategy 增密
  ▼
ckpt_final.pt + splat.ply
```

## 目录结构（模块职责）

```
pano_gsplat/
├── train.py         训练入口：命令行参数解析（parse_args）+ 配置解析
│                    + 高斯初始化（make_splats）+ 优化器/策略（make_optimizers、
│                    make_strategy）+ 训练主循环（main）+ 保存（save_ply/save_preview）
├── data.py          数据层：COLMAP 二进制读取（read_cameras/images/points3d_bin、
│                    load_sparse）+ 相机数学（EquirectCamera、camera_K、
│                    qvec_to_rotmat、build_camtoworlds）+ 全景切面（FACE_DIRS、
│                    face_rotation、build_face_c2w、pinhole_intrinsics、
│                    warp_pano_to_faces）+ 图像加载与归一化（load_pano、
│                    normalize_world、get_faces）
├── check.py         验证入口：`python check.py warp`（切面自洽性）/
│                    `split`（与 SfM 观测的重投影/射线一致性）/ `all`
├── diag.py          诊断入口：`python diag.py <ckpt.pt>`，输出高斯尺度/
│                    不透明度/位置分布 + 相机轨迹基线统计
├── pose.py          位姿优化模块（rig 感知：全景 6 面共享一个位姿增量；
│                    CameraOptModule/rotation_6d_to_matrix 来自 gsplat examples）
├── gen_depth_moge.py  MoGe-2 稠密深度生成（每张全景 6 个面各一张度量深度 .pt）
├── lib_bilagrid.py  第三方：3D bilateral grid（Apache-2.0，gsplat examples 原样拷贝）
├── requirements.txt 依赖清单
└── README.md        本文档
```

## 依赖

建议直接使用已有的 gsplat conda 环境：

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe
```

需要：gsplat（1.5.x）、torch、numpy、scikit-learn、imageio、`fused-ssim`。

**可选（稠密深度监督）**：MoGe-2（Windows 可装版本）：

```powershell
git clone https://github.com/microsoft/MoGe.git D:\gaussian_splatting\tools\MoGe
git -C D:\gaussian_splatting\tools\MoGe checkout 07444410f1e33f402353b99d6ccd26bd31e469e8
C:\Users\syk\.conda\envs\gsplat\python.exe -m pip install -e D:\gaussian_splatting\tools\MoGe
```

（MoGe 主分支已是 MoGe-3，依赖 Triton 在 Windows 上不便，故固定到 2025-11-02 的 MoGe-2 提交）

**可选（PPISP 光度校正）**：nv-tlabs/ppisp（Apache-2.0），需要 CUDA 扩展编译。
本机的编译路径（VS2022 14.44 的 STL 拒绝 CUDA<12.4，而 torch cu118 又禁止
CUDA 12.x 的 nvcc，因此必须用 VS2019 的 MSVC 14.29 编译）：

```powershell
git clone https://github.com/nv-tlabs/ppisp.git D:\gaussian_splatting\tools\ppisp
# 构建脚本会强制 setuptools 使用 VS2019 的 vcvarsall 并打 wheel
C:\Users\syk\.conda\envs\gsplat\python.exe D:\gaussian_splatting\tools\build_ppisp_win.py
C:\Users\syk\.conda\envs\gsplat\python.exe -m pip install --force-reinstall --no-deps `
  D:\gaussian_splatting\tools\ppisp\dist\ppisp-1.2.1-cp310-cp310-win_amd64.whl
```

## 数据要求

支持两类 COLMAP 稀疏模型：

**A. 等距柱状全景**（每张切成 6 个 90° 面）：

```
<data_dir>/
  images/      等距柱状图（2:1，文件名与 sparse 一致）
  sparse/0/    cameras.bin / images.bin / points3D.bin，相机必须是
               EQUIRECTANGULAR（COLMAP model 17，参数即宽高）
```

**B. 透视相机**（K=1 不切面，直接用原图 + 相机自身内参）：

```
<data_dir>/
  images/      透视图像（文件名与 sparse 一致）
  sparse/0/    相机为 COLMAP model 0-4（SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL/
               RADIAL/OPENCV），畸变忽略；图片尺寸与标定尺寸不一致时自动缩放 K
```

示例：`D:\gaussian_splatting\spirula`（354 张全景）和
`D:\gaussian_splatting\GGPS-data\datasets\FTP\gsplat_ftp`（2124 张透视 rig 图）
都可以直接跑。

## 使用

冒烟测试（1 张全景、128 面、30 步）：

```powershell
cd D:\gaussian_splatting\pano_gsplat
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --data-dir D:\gaussian_splatting\spirula `
  --max-images 1 --face-size 128 --steps 30 --out-dir outputs\smoke
```

正式训练（全部 354 张、1024 面、3 万步，MCMC，建议开面缓存）：

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --data-dir D:\gaussian_splatting\spirula `
  --face-size 1024 --steps 30000 --strategy mcmc `
  --face-cache D:\gaussian_splatting\pano_gsplat\face_cache `
  --out-dir D:\gaussian_splatting\pano_gsplat\outputs\full
```

透视数据（rig 图）示例：

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --data-dir D:\gaussian_splatting\GGPS-data\datasets\FTP\gsplat_ftp `
  --steps 30000 --out-dir D:\gaussian_splatting\pano_gsplat\outputs\persp
```

验证与诊断：

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe check.py all
C:\Users\syk\.conda\envs\gsplat\python.exe diag.py outputs\full\ckpt_final.pt
```

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-dir` | `D:\gaussian_splatting\spirula` | 数据集目录（含 images/ 和 sparse/0） |
| `--config` | 空 | JSON/YAML 配置文件，提供默认值（命令行显式参数优先） |
| `--dump-config` | 空 | 把生效配置导出为 JSON 并退出 |
| `--image-dir` | 空 | 覆盖图片目录（默认 `<data-dir>/images`，可用 images_2/4 等） |
| `--face-size` | 512 | 每面边长（90° FOV，焦距=S/2）。3840×1920 全景建议 1024 |
| `--strategy` | default | `default`=Inria clone/split，`mcmc`=MCMC 增密 |
| `--antialiased` | 关 | gsplat 的 Mip-Splatting 补偿因子（更抗锯齿，PSNR/SSIM 略降） |
| `--init-scale` / `--init-opacity` | 自动 | MCMC 自动用 0.1/0.5，其余用 1.0/0.1，可手动覆盖 |
| `--scale-reg` / `--opacity-reg` | 自动 | MCMC 自动用 0.01/0.01，其余默认 0 |
| `--batch-size` | 1 | 每步全景数（每张展开成 6 个面） |
| `--max-images` | 0 | 只用前 N 张（0=全部），调试用 |
| `--face-cache` | 空 | 把重采样后的面缓存到磁盘（uint8），重跑省时间 |
| `--save-every` / `--preview-every` | 0 | 每 N 步保存 ckpt+ply / GT\|渲染对比图 |
| `--absgrad` | 关 | AbsGS：用绝对 2D 梯度增密（配合 default 策略） |
| `--packed` | 关 | packed 光栅化（省显存，略慢） |
| `--sparse-grad` | 关 | 稀疏梯度（自动开启 --packed，需 SparseAdam） |
| `--test-every` | 0 | 每 N 张图留 1 张做验证集（0=全部训练） |
| `--eval-every` | 0 | 每 N 步跑一次验证评测（0=只在最后） |
| `--eval-max-images` | 20 | 每次评测最多渲染的验证图数 |
| `--save-eval-images` | 关 | 保存验证 GT\|渲染对比图 |
| `--coarse-face-size` / `--coarse-steps` | 0 / 0 | 粗到细：先小面热身 N 步再切到完整尺寸 |
| `--bilagrid` | 关 | 逐视角 bilateral grid 曝光/白平衡校正 |
| `--bilagrid-shape` | 16,16,8 | bilagrid 网格 X,Y,W |
| `--ppisp` | 关 | PPISP 逐视角光度校正（曝光/晕影/颜色/CRF，Spirula 语义：每个 post-split 视角一个 slot） |
| `--ppisp-mode` | per_view | `per_view`=Spirula 每视角一个 slot；`hybrid`=曝光/颜色按全景共享、晕影/CRF 按面方向 |
| `--ppisp-lr` | 0.002 | PPISP 主参数学习率 |
| `--ppisp-reg-scale` | 1.0 | PPISP 正则损失权重 |
| `--depth-supervision-weight` | 0 | 深度监督权重（当前用 COLMAP 稀疏观测深度） |
| `--depth-dir` | 空 | MoGe 稠密深度目录（gen_depth_moge.py 输出），启用逐像素深度监督 |
| `--pose-opt` / `--pose-opt-lr` | 关 / 1e-5 | 位姿优化（rig 感知：全景 6 面共享一个位姿增量） |
| `--random-bkgd` | 关 | 每步随机背景色，抑制透明漂浮物 |
| `--offload` | none | 主机卸载：`none`=全 GPU；`naive`=参数/Adam 常驻内存，每步整批传 GPU 并搬回梯度（v1 仅支持 default 策略，暂不支持 packed/absgrad） |
| `--bench-gaussians` | 0 | 把点云平铺膨胀到指定高斯数，用于显存/内存压力测试（配合小步数使用） |

## 新增功能（均已测试）

| 功能 | 说明                                                                                                                                    | 测试结果（短训） |
|---|-----------------------------------------------------------------------------------------------------------------------------------------|---|
| AbsGS + packed + sparse_grad | 绝对梯度增密；packed 把"tile×槽位"的稠密表改成扁平+偏移表，省掉空槽；sparse-grad 把"全体高斯的梯度"改成"仅可见高斯的稀疏梯度"，省掉零值 | 60 步 loss 0.28→0.06，显存 0.76 GiB |
| 验证集评测 | `--test-every` 留出验证图，`--eval-every` 周期渲染并写 metrics.json（PSNR/SSIM/LPIPS）                                                  | 90 步 PSNR 21.1→22.3，LPIPS 0.31→0.18 |
| 粗到细 | 先低分辨率面热身再切高分辨率                                                                                                            | 128→256 切换正常，loss 持续下降 |
| bilagrid | 逐视角曝光/白平衡校正（每面一个网格）外观解耦升级                                                                                       | 60 步 loss 0.28→0.055，SSIM 0.864 |
| 深度监督 | 渲染期望深度 ED，在 COLMAP 观测像素处做视差 L1；MoGe 稠密深度可复用同一接口                                                             | 训练正常，RGB 质量与基线持平 |
| MoGe 稠密深度 | `gen_depth_moge.py` 生成每面度量深度，训练时按面与稀疏深度中位比例对齐到场景尺度，逐像素视差 L1                                         | 2 图 40 步训练正常 |
| 位姿优化 | 每张全景/每张图一个 6D 位姿增量（全景模式 6 面共享，不破坏 rig）                                                                        | 40 步 loss 0.28→0.07 |
| 随机背景 | 每步随机背景色                                                                                                                          | 与位姿优化同测通过 |
| 组合运行 | 上述功能同时开启                                                                                                                        | 12 图 60 步 + 评测，无冲突 |

评测写入 `outputs/<run>/metrics.json`，用 `--save-eval-images` 可保存验证视图对比图。

### PPISP 光度校正

用于解决多视角采集的光照不一致（自动曝光/白平衡漂移、晕影、色偏、相机响应
差异）。两种绑定方式，用 `--ppisp-mode` 切换：

- `per_view`（默认，Spirula 语义）：每个 post-split 视角（全景=每张 6 个面、
  透视=每张图）一个独立 slot，`num_cameras = num_frames = 训练视角数`
  （等价于 Spirula `use_ppisp=true`、`ppisp_param_type=no_crf` 的近似，
  nv-tlabs 版没有关 CRF 的开关，靠正则压住）。
- `hybrid`（物理优先）：曝光/白平衡按**全景**共享（`num_frames = 全景数`），
  晕影/CRF 按**面方向**（`num_cameras = 6`，所有全景共享；透视数据退化为
  `num_cameras = 1`）。自由度从 1860 组降到 310 组曝光/颜色 + 6 组晕影/CRF，
  参数更稳、更符合 360 相机“同一时刻同一曝光”的物理链路。

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --config configs\example.yaml --ppisp --ppisp-mode hybrid --steps 2000 `
  --out-dir D:\gaussian_splatting\pano_gsplat\outputs\ppisp_hybrid
```

应用顺序：光栅化 → **PPISP** → bilagrid → 随机背景 → loss
（对应 Spirula `apply_ppisp_before_bilagrid=true` 的默认行为）。

注意事项：

- 控制器默认关闭（`use_controller=False`）：每视角一个 CNN 控制器在 1860 个
  slot 下不现实；新视角推理因此退化为恒等校正，验证集数字衡量的是纯场景质量。
- 训练输出额外保存 `ppisp_final.pt`（含曝光/晕影/颜色/CRF 参数），
  训练视角评测可对比“原始渲染 vs PPISP 校正后”的 PSNR 差：

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe D:\gaussian_splatting\tools\eval_train_ppisp.py `
  --ckpt outputs\ppisp_on\ckpt_final.pt --ppisp-state outputs\ppisp_on\ppisp_final.pt
```

**2000 步 spirula 三组对比**（同 seed、同配置）：

| 指标 | 基线（无 PPISP） | per_view | hybrid |
|---|---|---|---|
| 验证集 PSNR | 24.39 | 22.73 | 23.29 |
| 验证集 SSIM | 0.781 | 0.768 | 0.777 |
| 验证集 LPIPS | 0.299 | 0.315 | 0.299 |
| 训练视角 PSNR 校正增益 | — | +0.40（含 -1.40 负样本） | +0.65（全部为正） |
| 学飞程度（color/CRF max） | — | 1.52 / 3.41 | 0.29 / 0.33 |

结论：hybrid 明显优于 per_view（验证集 +0.56 dB、训练校正一致为正、参数不
学飞），但仍比基线低 1.10 dB——该数据光照一致性较好，PPISP 的光度自由度对
场景是净负担；在有强自动曝光/白平衡漂移的数据上才值得开。

### 主机卸载（naive，实验性）

把高斯参数和 Adam 状态常驻 CPU 内存，每步整批拷贝到 GPU 前向/反向、
梯度搬回 CPU 更新，用显存换规模。设计参考 CLM-GS（ASPLOS 2026）的
`naive_offload`，数值与全 GPU 路径等价（同 seed 小规模 loss 逐位一致）。

```powershell
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --config configs\example.yaml --steps 30000 --strategy default `
  --offload naive --sh-degree 2
```

约束（v1）：只支持 `--strategy default`；不支持 `--packed`/`--absgrad`
（`--sparse-grad` 会被自动关闭）。`--offload clm`（保留关键属性在 GPU、
SH 分页）尚未实现。

压力测试（`--bench-gaussians N` 把点云平铺膨胀到 N 个高斯，
`--face-size 512 --sh-degree 2`，6 面同时渲染）：

| 高斯数 | GPU 峰值 | 结论 |
|---|---|---|
| 1M | 2.4 GiB | 轻松 |
| 2M | 4.8 GiB | 正常 |
| 3M | 7.1 GiB | 8GB 卡的实际舒适上限 |
| 5M | 11.8 GiB | 跑通但已溢出到共享内存，极慢 |
| 8M/10M | — | OOM（瓶颈是 SH 反向按 6 相机各一份系数缓冲） |

降 `--sh-degree 0` 可到 10M（峰值约 12 GiB，仍溢出共享内存）。要真正在
8GB 内跑 10M，需再做逐面渲染（C=1）和稀疏梯度，是下一步的优化方向。

### 配置文件

支持 JSON 或 YAML，键 = 参数名（连字符键 `data-dir` 等价于 `data_dir`），
命令行显式参数优先于配置文件。示例见 `configs/example.yaml`：

```powershell
# 用配置跑（示例配置里已含 354 张全景的正式训练参数）
C:\Users\syk\.conda\envs\gsplat\python.exe train.py --config configs\example.yaml

# 覆盖个别参数（比如先小规模测试）
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --config configs\example.yaml --max-images 4 --face-size 256 --steps 300

# 导出生效配置，方便保存/复用
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --config configs\example.yaml --dump-config outputs\my_run.json
```

### MoGe 稠密深度使用流程

```powershell
# 1. 生成深度（每张全景 6 个面，建议 --face-size 512；首次运行自动下载权重）
C:\Users\syk\.conda\envs\gsplat\python.exe gen_depth_moge.py `
  --data-dir D:\gaussian_splatting\spirula --face-size 512 `
  --out-dir D:\gaussian_splatting\spirula\depths

# 2. 训练时启用稠密深度监督
C:\Users\syk\.conda\envs\gsplat\python.exe train.py `
  --data-dir D:\gaussian_splatting\spirula --face-size 1024 --steps 30000 `
  --strategy mcmc --depth-dir D:\gaussian_splatting\spirula\depths `
  --depth-supervision-weight 0.1
```

## 修复记录（重要）

1. **像素中心约定**：GT 重采样改为像素中心（u = i + 0.5），与 gsplat
   光栅化和 Spirula 的 WarpFace 一致；早期角落约定导致 GT 与渲染错位半像素。
2. **MCMC 超参**：MCMC 自动套用配套超参（init_scale=0.1、init_opacity=0.5、
   scale_reg=0.01、opacity_reg=0.01）。MCMC 的噪声项正比于 `lr × noise_lr × 尺度²`，
   没有正则压尺度会进入正反馈：尺度变大 → 噪声放大 → 高斯飞走。
3. **分幅旋转方向**：`warp_pano_to_faces` 的面→全景射线旋转方向写反
   （`R.T` 与 `R` 的 einsum 方向），导致 left/right/up/down 四个面的 GT
   左右/上下对调。修复后同配置 2000 步 PSNR 从 19.9 → 25.5 dB。
   由 `check.py split` 的"重投影误差 + 面射线颜色一致性"守护。
4. **深度监督量纲**：稀疏深度 GT 原先用 COLMAP 原始坐标，而训练场景是
   归一化坐标，两者尺度不一致；现统一用归一化后的位姿/点云计算深度，
   MoGe 稠密深度也按面与稀疏深度中位比例对齐到同一场景尺度。

## 已知限制

- 无掩膜（SAM/几何模板）、无法线监督、无浏览器 viewer
- 透视模式忽略镜头畸变（model 2-4 只取内参）；不支持 FISHEYE（model 5）
- 世界归一化用"相机中心居中 + 等距缩放"，未做主轴对齐
- 深度监督需要先运行 `gen_depth_moge.py` 或使用 COLMAP 稀疏深度（默认）
- 位姿优化默认关闭（弱基线场景建议小学习率），intrinsics 仍固定

## 许可

本项目代码为独立实现（参考 Spirula Studio 公开的算法思路，未复制其源码），
基于 Apache-2.0 的 gsplat。公司商用请保留 gsplat 版权声明。
