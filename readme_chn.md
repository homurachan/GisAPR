# GisAPR 210

**GPU-accelerated in situ Atomic Perturbation Refinement**

[English manual](README.md)

GisAPR 以 cryo-EM 颗粒图像为目标，对 PDB 中选定链的刚体位置与取向进行 refinement。程序对该链的姿态变化进行采样，生成模型密度，通过优化后的 v606 搜索引擎搜索颗粒取向，再根据颗粒评分比较各个构象。其余链保持固定，提供周围结构背景。

本手册对应 **210 版及修正后的指数 CC 评分脚本**。如果你使用的是最初发布的 210 压缩包，请先用单独提供的修正版替换 `new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py`，再使用下文介绍的指数评分模式。最初压缩包中的这个分支使用的是直接 CC 求和。具体行为及保留的拟合失败回退方式见[评分](#评分)。

截图展示的是较早的 GUI 布局。下文的操作说明和参数解释以 210 版为准；旧截图中的注释不代表当前程序的完整行为。请将 `README.md`、`readme_chn.md` 和提供的 `Pictures/` 目录放在一起，通常放在 GisAPR 安装目录中。

## 目录

- [210版的主要变化](#210版的主要变化)
- [安装](#安装)
- [输入文件与准备](#输入文件与准备)
- [快速开始](#快速开始)
- [使用GUI](#使用gui)
- [命令行refinement](#命令行refinement)
- [继续运行](#继续运行)
- [使用现有MRC进行一次搜索](#使用现有mrc进行一次搜索)
- [Geometric restraint](#geometric-restraint)
- [评分](#评分)
- [输出文件与BEST_FIT导出](#输出文件与best_fit导出)
- [GPU运行与工作目录](#gpu运行与工作目录)
- [故障排查与保留限制](#故障排查与保留限制)
- [程序文件、验证与致谢](#程序文件验证与致谢)

## 210版的主要变化

- PDB 读写采用固定列宽的纯文本处理，不再需要 BioPython、`xpdb.py` 或 CuPy。
- 每个选定 GPU 对应一个 permanent worker。模型密度生成、geometric restraint 和优化后的搜索共用这个进程，因此每个 worker 只需导入一次 PyTorch，无需对每个采样构象重复导入。
- 可以跳过 geometric restraint。启用时，它仍然只用于拒绝不合格构象，不会向通过检查的构象图像评分中添加任何数值。
- 整个 refinement 流程和 GUI 均支持不提供 FSC 文件。
- PSO 可以设置六维 pivot point。PSO、Pattern Search 和 Simplex 均支持通过 checkpoint 继续运行。
- Pixel-size search 使用相同的计算流程，默认跳过 geometric restraint。
- 正常结束时自动汇总本次运行的评分，并将获胜模型复制为 `*_BEST_FIT.pdb`。

候选构象的 MRC 文件，以及原有的搜索和评分调试文件，仍会写入磁盘。优化后的搜索仍然读取 MRC 模型；移除 CuPy 并不意味着移除模型 MRC 文件。

## 安装

使用 **Python 3.10 或更高版本**。对于正式的大规模运行，预期环境为 Linux 和 NVIDIA GPU。CPU 模式可用于小规模检查。以下命令使用 Bash 语法，不适用于 Windows Command Prompt 或 PowerShell。

首先解压源代码包，并设置安装目录的绝对路径：

```bash
export GISAPR_DIR=/absolute/path/to/GisAPR-210
python -m venv "$GISAPR_DIR/.venv"
source "$GISAPR_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
```

通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装适合当前 CUDA 环境的 PyTorch，然后安装程序依赖：

```bash
python -m pip install -r "$GISAPR_DIR/requirements.txt"
```

Refinement 的依赖包括 NumPy、SciPy、PyTorch、mrcfile、einops、Matplotlib、scikit-learn 和 joblib。可选的颗粒预处理脚本还需要 `starfile`；当前 `requirements.txt` 未列出这个依赖：

```bash
python -m pip install starfile
```

GUI 需要 Tkinter 和图形显示环境。Tkinter 由操作系统或 Python 发行版提供，不通过 `pip` 安装；Debian/Ubuntu 中相应的系统软件包通常名为 `python3-tk`。命令行程序不需要图形显示环境。

检查当前 Python 环境：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
```

210 版保留了历史文件名 `GUI_v209.py` 和 `CTF_cupy.py`。后者实际使用 NumPy，不需要 CuPy。Refinement 流程不会导入 RELION、EMAN 或 Chimera/ChimeraX；这些程序可用于上游 alignment、reconstruction 或模型准备。

## 输入文件与准备

### 必需文件

| 输入 | 用途 |
| --- | --- |
| 初始 PDB | 包含要移动的链及周围保持固定的链。 |
| Particle STAR 及其引用的 MRC/MRCS 图像 | 提供颗粒图像、近似取向，以及存在时的 defocus 信息。输入图像应完成 whitening，并适当居中。 |
| 取向采样 STAR | 提供模型投影方向。请明确生成此文件；程序中的历史默认文件名并不对应随包提供的可直接使用数据集。 |
| FSC 曲线，可选 | 提供径向 Fourier 权重。不提供时使用值为 1 的 FSC 权重。 |

每次只 refine 一条链。如果希望将一条链中的一部分作为独立刚体进行 refinement，请先在 PDB 中为这一部分分配单独的 chain identifier。`--rotate_chain` 必须对应输入模型中实际存在的链。

### 坐标系与PDB约定

PDB 的固定部分应已与颗粒 alignment 对应。进行 local search 时，应先通过上游 alignment 或 reconstruction 获得可用的颗粒取向，并将模型拟合到对应的 map 中。

密度生成器将 PDB 坐标放入 volume 的方式为：

```text
voxel_position = coordinate_in_angstrom / apix_PDB + boxsize // 2
```

因此，**PDB 坐标 (0, 0, 0) Å 对应生成 box 的中心**。Refinement 不会自动将初始 PDB 重新居中。请确保模型与居中的颗粒图像使用相同的物理坐标系；只修改 MRC header 的 origin 不能代替实际坐标检查。

PDB 坐标读取自第 31–54 列，chain identifier 读取自第 22 列，element 优先读取第 77–78 列，缺失时根据 atom name 判断。Atom serial 和 residue number 字段作为文本保留。变换后的完整 PDB 保留固定链和其他记录。

输入中应只包含一个准备使用的 `MODEL`：坐标变换和链拆分使用第一个 model，而密度与几何读取器可以读取所有 model。保留的密度 kernel 通常使用 `ATOM` 记录，支持 H、C、N、O、P 和 S；其他元素在此 kernel 中不贡献密度。坐标变换可以保留 `HETATM` 记录，但这不代表这些记录会参与默认的密度计算。

### Particle STAR字段

优化后的搜索通过标签查找 STAR loop，因此无需为该搜索将文件第一行改为 `# relion 30001`。

| STAR 标签 | 优化后搜索中的行为 |
| --- | --- |
| `_rlnImageName` | 必需。`1@particles.mrcs` 这类 stack 引用使用从 1 开始的图像序号。 |
| `_rlnAngleRot`, `_rlnAngleTilt` | 必需的颗粒取向字段。 |
| `_rlnAnglePsi` | 可选；缺失时默认为零。 |
| `_rlnDefocusU`, `_rlnDefocusV`, `_rlnDefocusAngle` | 只有三个字段同时存在时才使用 CTF。Defocus 单位为 Å，角度单位为度。 |
| `_rlnOriginX/Y`, `_rlnOriginX/YAngst` | 搜索不会应用这些位移。请在 refinement 前完成需要的图像平移。 |
| Optics 中的 pixel size、voltage 和 Cs | 不用于配置搜索；请通过 CLI/GUI 提供实际值。 |

图像的相对路径以**工作目录**为基准解析，而不是以 STAR 文件所在目录为基准。请使用相对于工作目录有效的路径，或使用绝对路径。整个数据集使用统一提供的 particle pixel size、voltage 和 Cs；程序不会应用每个 optics group 的独立设置。保留的 CTF 实现使用 amplitude contrast 0.1，不读取 STAR 中的 phase-shift 或 amplitude-contrast 字段。

### 颗粒图像的居中与whitening

搜索会对 Fourier 数组归一化，并对模型模板进行 whitening，但不会为输入颗粒执行所需的径向振幅 whitening。请使用已准备好的图像，或运行包内的预处理脚本。

对于包含 `data_particles` 和 `data_optics` 的 RELION STAR，可以应用已存储的位移，并按需裁剪图像：

```bash
python "$GISAPR_DIR/read_star_shift_crop_and_generate_new_star_v5.py" \
  --star_name particles.star \
  --output_root_name particles_whiten \
  --newboxsize 256 \
  --batchsize 5000 \
  --dowhitening
```

这里的 `256` 是实空间输出 box size 的示例值。请选择不大于输入图像尺寸的正偶数。该脚本读取 `rlnOriginXAngst`、`rlnOriginYAngst`、`rlnImageName`，以及 optics 第一行的 `rlnImagePixelSize`。它将位移取整为整数 pixel，用零填充暴露的边缘，进行中心裁剪、whitening、拟合平面去除和归一化，最后将存储的 origin 设为零。

如果图像已经居中，可以保留原始 box size 并跳过平移：

```bash
python "$GISAPR_DIR/read_star_shift_crop_and_generate_new_star_v5.py" \
  --star_name particles.star \
  --output_root_name particles_whiten \
  --newboxsize 0 \
  --batchsize 5000 \
  --dowhitening \
  --doSkipShifting
```

**本版本中，`--newboxsize 0` 应与 `--doSkipShifting` 一起使用。** 此预处理脚本未作修改；在跳过平移的同时指定正的 crop size 会触发变量未定义错误。以上两条命令是两种可选的准备方式，不应对输出再次进行 whitening。

输出包括 `particles_whiten.star`、`particles_whiten_0001.mrcs` 及后续编号的 stacks。如果主机 RAM 不足，可以减小 `--batchsize`。脚本中的 `--doOnlyMakeStar` 只改写引用关系，不会生成图像 stack 或执行 whitening。

### 生成取向列表

生成约 3° 的等面积方向采样：

```bash
python "$GISAPR_DIR/generate_healpix_order_and_relion_star.py" \
  --o c1_3deg.star --useEQPS --EQPSangleDegree 3.0 --apix 1.58
```

必须添加 `--useEQPS`；只设置 `--EQPSangleDegree` 不会选择 EQPS 采样。也可以生成 HEALPix 网格：

```bash
python "$GISAPR_DIR/generate_healpix_order_and_relion_star.py" \
  --o c1_healpix4.star --healpixOrder 4 --apix 1.58
```

如果适合当前模型和数据集，可使用支持 symmetry 的生成器去除等价投影方向：

```bash
python "$GISAPR_DIR/generate_healpix_order_and_relion_star_symgroup_v2.py" \
  --o d7_3deg.star --useEQPS --EQPSangleDegree 3.0 --sym D7 --apix 1.58
```

支持的 symmetry 名称包括 `C1`、`Cn`、`Dn`、`tet`、`oct`、`I2` 和 `I3`；`I`、`ico`、`icos` 均为 `I2` 的别名。这个功能减少投影方向，但不会对 PDB 扰动施加对称性。请使用适合实际搜索模型的 symmetry；单个亚基移动后，模型的对称性可能发生变化。

生成器的 `--discardPositiveRot` 会移除部分方向，生成通用的全方向列表时不需要此选项。搜索中的 in-plane sampling 由 `--psiStep` 控制，独立于生成器写入的 Psi 值。无论 angle STAR 中写入了什么 optics metadata，都应在 refinement 命令中设置显微镜参数。

### Pixel size与box size

| 参数 | 含义 |
| --- | --- |
| `--apix` / Particle Apix | Particle pixel size，单位 Å/pixel。 |
| `--apix_PDB` / PDB Apix | 将 PDB 栅格化为模型 MRC 时使用的采样间隔；PDB 坐标本身仍以 Å 为单位。通常首先设为与 particle pixel size 相同。 |
| `--boxsize` / Boxsize | 实际颗粒实空间 box size，也是生成模型的 box size。 |
| `--newboxsize` / Search boxsize in pixel | 搜索使用的中心 Fourier crop size，与预处理脚本中的实空间裁剪选项不同。 |

Box size 应为正偶数。对于常用 wrapper 及其默认 CCG crop，应满足 `32 <= newboxsize <= boxsize`。Fourier cropping 保持视野不变，使实际搜索 pixel size 变为 `boxsize / newboxsize * apix`。

改变 `apix_PDB` 会改变模型相对于图像的尺度，比例约为 `apix / apix_PDB`，但不会改写 PDB 坐标。如果需要校准这个相对尺度，请使用 [pixel-size search](#pixel-size-grid-search)。候选密度生成使用的 resolution 参数为 `2 * apix_PDB`。

### 可选FSC

通常不使用 FSC 时，直接省略 `--fsc_file`，或将 GUI 中的对应字段留空。此时搜索将 FSC 权重初始化为 1。

如需应用曲线，在优化器中使用 `--fsc_file curve.fsc`，在单次搜索 wrapper 中使用 `--FSC curve.fsc`。读取器要求无 header、以空白分隔的数值行，将**第二列**依次作为 radial-shell 权重，并忽略第一列。未提供的更高半径 shell 权重为零。

如果在 GUI 中提供曲线并希望使用其数值，请取消勾选 **Do ignore FSC**。CLI 中对应的选项为优化器的 `--do_ignoreFSC` 和单次搜索 wrapper 的 `--ignoreFSC`。这些 ignore 选项仍然打开提供的文件，保留文件所覆盖的径向范围，只将读取的权重替换为 1。如果完全不需要曲线，应移除文件名，而不是指定不存在的文件再添加 ignore 选项。

## 快速开始

每个独立运行应使用单独的工作目录。以下示例假设 `model.pdb`、`particles_whiten.star`、其引用的 stacks 和 `c1_3deg.star` 均可从该目录访问。请根据数据修改示例中的 box size、pixel size、mask radius、显微镜参数和 chain ID。

```bash
cd /absolute/path/to/your/run_directory
export GISAPR_DIR=/absolute/path/to/GisAPR-210

python "$GISAPR_DIR/test_op_Particle_Swarm_optimization_refine_rot_trans_ver622.py" \
  --PDB_NAME model.pdb \
  --STAR_NAME particles_whiten.star \
  --rotate_chain A \
  --ang c1_3deg.star \
  --output_name_root pso_ \
  --boxsize 256 --newboxsize 160 \
  --apix 1.58 --apix_PDB 1.58 \
  --voltage 300 --cs 2.7 --maskRadius 110 \
  --do_local_search --local_stepsize 20 --psiStep 3 --kk 3 \
  --gpuid 0 --max_workers 1 --SplitParticles 1 \
  --do_run_CC --do_simple_sum \
  --skip_geometric_restraint \
  --PSO_num_particles 3 --PSO_iterations 30 \
  --PSO_Bounds '(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)' \
  --PSO_pivot_point '[0,0,0,0,0,0]'
```

本示例使用颗粒取向的 local search、修正评分脚本中的指数 CC 汇总，不使用 FSC，也不进行几何拒绝。如果需要 geometric restraint，请移除 `--skip_geometric_restraint`，并按照下文设置合适的限制。30 次 PSO 迭代只是示例计算预算，不保证收敛。

正常结束后，检查打印的最大 score、`*_BEST_FIT.pdb` 和 `pso_BEST_FIT.json`。原始 score 越大越好；优化器日志记录的是 score 的负值，因此越小越好。

## 使用GUI

**从数据工作目录**启动 GUI：

```bash
python "$GISAPR_DIR/GUI_v209.py"
```

210 版保留了这个脚本的原始文件名。它根据自身安装目录定位包内的 refinement 和搜索脚本；相对数据路径仍以工作目录为基准。

![较早界面中的 GUI 导航；210 版还包含 Refine Once Parameters。](Pictures/gisapr_GUI_1.png)

### 1. Input Files

填写 particle STAR、初始 PDB、选定链、取向列表、box size、particle/PDB pixel size、voltage 和 Cs。FSC 为可选项。历史的 subunit-mass 字段为兼容而保留，不会改变当前计算流程中通过检查构象的图像评分。

切换页面前，请点击本页的 **Submit**。

![Input Files：210 版中的 FSC 为可选输入。](Pictures/gisapr_GUI_2.png)

### 2. Search Parameters

选择搜索 box size、mask radius、in-plane step、isSPA `n`（`kk`）和 local-search range。如果颗粒取向可作为合理起点，保持 local search 开启。GUI 中的 **Do run CC** 和 **Do simple sum** 复选框选择[评分](#评分)一节介绍的评分模式。

GUI 默认值与 CLI 不同：默认勾选 local search、CC 模式和忽略 FSC，而不勾选 simple sum。若需要修正后的指数 CC 模式，应同时勾选 **Do simple sum** 和 **Do run CC**。填入 FSC 文件名不会自动取消 **Do ignore FSC**。

常规 refinement 页面不再提供 **Mask Soft Edge in pixel**；该流程使用默认值 6 pixels。单独的 Refine Once 页面仍可设置 mask edge。保留的 GPU-projection 复选框属于兼容选项；优化后的流程已经通过 PyTorch 生成模型投影。

点击 **Submit**。

![较早界面中的 Search Parameters。旧的 mask-edge 和 GPU 显存注释不能用于定义 210 版行为。](Pictures/gisapr_GUI_3.png)

### 3. Refine Parameters

设置 output root、GPU IDs、几何阈值或 **Skip geometric restraint**，并选择算法。GPU IDs 可写为 `0` 或 `0:1`；历史显示值 `0:0:0` 表示重复指定同一设备。

210 版在此页提供四种算法：

| 算法 | 主要设置 |
| --- | --- |
| Grid Search | 六维 bounds，以及角度和平移的采样间隔。 |
| Simplex | 六维 bounds、最大迭代次数、`xatol` 和 `fatol`。旧的指定/随机 initial simplex GUI 字段已移除。 |
| Particle Swarm Optimization（PSO） | 六维 bounds、粒子群规模、迭代次数、惯性权重和六维 pivot。 |
| Pattern search | 六维 bounds、可选初始点、初始步长、容差、步长扩大/缩小因子和最大迭代次数。 |

GPU-worker 字段控制 Grid、PSO 和 Pattern Search 的候选构象任务分配。Simplex 按顺序计算目标函数。旧的 CPU-worker 字段只会传递给 Grid，不控制重写后计算流程中的 worker 创建。Pixel-size grid search 可通过 CLI 使用。

设置好算法后，点击 **Submit**。

![较早界面中的公共 refinement 设置和 Simplex 控件。](Pictures/gisapr_GUI_4.png)

![Grid Search 控件：角度和平移的采样间隔。](Pictures/gisapr_GUI_41.png)

![较早界面中的 PSO 控件。210 版的 pivot 是姿态参数空间中的六维吸引点。](Pictures/gisapr_GUI_42.png)

### 4. Continue Run Parameters，按需使用

在 Refine Parameters 中选择 PSO、Pattern search 或 Simplex。在 Continue Run Parameters 中勾选 **Do continue run**，选择匹配的 checkpoint，并填写 **Continue Run this more rounds**。两个页面均需 Submit。对于可因收敛而提前停止的方法，追加轮数表示运行上限。

开始新运行时，取消勾选 **Do continue run** 并重新 Submit。Grid Search 不支持此 checkpoint 工作流程。

![较早的继续运行页面。210 版支持 PSO、Pattern Search 和 Simplex。](Pictures/gisapr_GUI_5.png)

### 5. 关闭GUI并运行打印的命令

GUI 只生成命令，**不会**直接启动 refinement。提交所有必需页面后，关闭窗口，将终端中打印的命令复制出来，并在相同的数据工作目录和 Python 环境中运行。

配置保存在工作目录中的 `input_params.json`、`search_params.json`、`refine_params.json` 和 `continue_params.json`。切换页面或关闭窗口不会保存尚未 Submit 的改动。如果缺少必需页面的设置，210 版会打印 `Submit parameters on these pages before closing: Input, Search, Refine`，列出仍需提交的页面。

![历史终端示例：必需的参数页面未提交时，命令生成失败。](Pictures/gisapr_GUI_6.png)

![历史终端中的命令生成示例。请使用当前生成的命令，不要使用截图中的 Windows 路径。](Pictures/gisapr_GUI_7.png)

### Refine Once Parameters

这是 210 版新增的页面，用于将现有模型 MRC 与颗粒进行搜索。填写 **Particle Star file**、**Model MRC file** 和 **Angle list**，设置搜索参数并 Submit。FSC 为可选项。生成的命令使用 `wrap_to_search_v2.py` 和 local search。此页不优化 PDB 链，也不生成 BEST_FIT PDB。

**关闭 GUI 时，已保存且非空的 `once_params.json` 优先于常规 refinement 页面。** 如果需要恢复常规 refinement 命令生成，请在启动 GUI 前将这个文件移开：

```bash
mv once_params.json once_params.saved.json
```

仅在 `once_params.json` 存在时运行此命令；如有需要，另选一个备份文件名。提供的截图中没有新的 Refine Once 或 Pattern Search 页面。

## 命令行refinement

### 共用设置与参数单位

所有姿态优化器均按以下顺序处理六个参数：

```text
(rot, tilt, psi, tx, ty, tz)
```

前三个参数虽然沿用了 `rot/tilt/psi` 的历史名称，但实际是**以度为单位的 rotation-vector 分量**。PDB 坐标变换使用 `Rotation.from_rotvec`，不是 RELION ZYZ Euler rotation。后三个参数是以 Å 为单位的平移。默认情况下，选定链首先绕其自身坐标的算术平均中心旋转，再进行平移。这个物理旋转中心与 PSO pivot 不同。

| 共用选项 | 含义 / 常用 CLI 默认值 |
| --- | --- |
| `--PDB_NAME`, `--STAR_NAME`, `--rotate_chain` | 必需的模型、颗粒和待移动链。 |
| `--ang` | 模型取向 STAR。请明确指定生成的文件。 |
| `--output_name_root` | 候选文件的前缀，可包含目录。每次独立运行使用不同前缀。 |
| `--boxsize`, `--newboxsize` | 实空间 box 和搜索 Fourier box；姿态优化器默认分别为 256 和 160。 |
| `--apix`, `--apix_PDB` | Particle 和 model sampling，单位 Å/pixel；默认均为 1.58。 |
| `--voltage`, `--cs` | 单位分别为 kV 和 mm；默认分别为 300 和 2.7。 |
| `--maskRadius` | 实空间 mask radius，以原始 particle pixel 为单位；默认 110。请按 box 和颗粒大小调整。 |
| `--psiStep` | In-plane 角度采样步长，单位为度；优化器默认 3。 |
| `--kk` | isSPA 权重参数，在 GUI 中显示为 `isSPA n`；优化器默认 3。 |
| `--do_local_search` | 将模型方向限制在给定颗粒取向附近。不加此选项时，CLI 在提供的方向列表中进行全局搜索。 |
| `--local_stepsize` | Local angular range，单位为度；虽然历史名称为“stepsize”，但它不是角度列表的采样间隔。默认 30。 |
| `--transRange` | 兼容参数，默认 0。优化后的引擎通过 CCG peak 获取图像平移；此选项不是 PDB 平移的 bounds。 |
| `--fsc_file`, `--do_ignoreFSC` | 可选曲线及其 ignore 开关，见 FSC 一节。 |
| `--do_run_CC`, `--do_simple_sum` | 评分模式开关。四种姿态优化器 CLI 中均默认关闭。 |
| `--skip_geometric_restraint` | 关闭几何计算和拒绝检查。姿态 refinement 中通常默认关闭此跳过选项。 |
| `--gpuid` | 以冒号分隔的逻辑 CUDA IDs，默认 `0`；小规模检查可使用 `cpu`。 |
| `--SplitParticles` | STAR 分块数，默认 1。 |
| `--doSplitDiffGpu` | 将分块分配到不同设备；见后面的 GPU 运行说明。 |
| `--yflip` | 对生成密度应用历史的 EMAN 风格 Y flip。仅在坐标约定需要时使用。 |

算法默认值和 GUI 默认值并不完全一致。请明确指定影响实验的设置。每个脚本均支持 `--help`，可查看完整参数列表。

以下示例先在工作目录中定义一次 Bash 数组：

```bash
COMMON=(
  --PDB_NAME model.pdb
  --STAR_NAME particles_whiten.star
  --rotate_chain A
  --ang c1_3deg.star
  --boxsize 256 --newboxsize 160
  --apix 1.58 --apix_PDB 1.58
  --voltage 300 --cs 2.7 --maskRadius 110
  --do_local_search --local_stepsize 20 --psiStep 3 --kk 3
  --gpuid 0 --SplitParticles 1
  --skip_geometric_restraint
)
```

以下算法是可选方案，不是必须依次执行的步骤。每次独立优化均应使用自己的目录，并在该目录中设置正确的输入路径和 `COMMON`。

### PSO

```bash
python "$GISAPR_DIR/test_op_Particle_Swarm_optimization_refine_rot_trans_ver622.py" \
  "${COMMON[@]}" --output_name_root pso_ \
  --do_run_CC --do_simple_sum --max_workers 1 \
  --PSO_num_particles 3 --PSO_iterations 30 \
  --PSO_Bounds '(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)' \
  --PSO_pivot_point '[0,0,0,0,0,0]'
```

`--PSO_num_particles` 表示粒子群中的成员数，不是 cryo-EM 图像数。Pivot 是六维吸引点，与姿态参数使用相同单位。默认的零 pivot 指向初始模型姿态。改变它不会改变原子旋转的物理中心。PSO 保留原有的软边界行为，因此采样位置可能略微超出名义 bounds。

可选控制包括 `--PSO_wmax`/`--PSO_wmin`，默认 0.9/0.4，以及 `--PSO_add_noise_velocities`；噪声强度和衰减分别由 `--PSO_noise_strength` 和 `--PSO_noise_decay_per_round` 控制。Surrogate guidance 需要通过 `--PSO_use_surrogate` 显式开启，其训练计划和权重由 `--PSO_surrogate_*` 系列选项控制。如果所选方案不包含此功能，保持关闭即可。

### Pattern Search

```bash
python "$GISAPR_DIR/test_op_pattern_search_try_multithreading_v3.py" \
  "${COMMON[@]}" --output_name_root pattern_ \
  --do_run_CC --do_simple_sum --max_workers 1 \
  --Pattern_Bounds '(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)' \
  --Pattern_given_initial --Pattern_initial '[0,0,0,0,0,0]' \
  --Pattern_stepsize 1 --Pattern_tol 0.01 \
  --Pattern_expand 1.2 --Pattern_shrink 0.5 --Pattern_max_iter 500
```

只有添加 `--Pattern_given_initial` 时，初始点字符串才会生效。Pattern Search 对邻近姿态进行搜索，并根据是否改善来调整步长。标量步长分别作用于每个坐标，使用该坐标对应的度或 Å 单位。当步长达到容差时，程序可以在最大迭代次数之前停止。

### Simplex

```bash
python "$GISAPR_DIR/test_op_Downhill_simplex_optimization_refine_rot_trans_rnd_init_v72.py" \
  "${COMMON[@]}" --output_name_root simplex_ \
  --do_run_CC --do_simple_sum \
  --Simplex_Bounds '(-15,15),(-15,15),(-15,15),(-20,20),(-20,20),(-20,20)' \
  --Simplex_maxiter 100 --Simplex_xatol 0.2 --Simplex_fatol 1.0
```

Simplex 在 bounds 内建立初始 simplex，然后通过 Nelder–Mead 风格的过程优化。`xatol` 控制参数的分散程度，`fatol` 控制目标函数值的分散程度；请根据实际评分尺度设置容差。当前目标函数按顺序计算，即使提供多个 GPU IDs 也是如此。姿态维数应保持为六。

### 姿态Grid Search

```bash
python "$GISAPR_DIR/test_op_GridSearch_refine_rot_trans_v321.py" \
  "${COMMON[@]}" --output_name_root grid_ \
  --do_run_CC --do_simple_sum --max_workers_GPU 1 \
  --Grid_Bounds '(-5,5),(-5,5),(-5,5),(-5,5),(-5,5),(-5,5)' \
  --Grid_Rotation_Stepsize 5 --Grid_Translation_Stepsize 5
```

此示例每一维包含三个采样点，因此共有 729 个候选姿态。Grid 的计算量随六维采样点数的乘积增长。角度步长以度为单位，平移步长以 Å 为单位。Grid Search 不提供迭代 checkpoint 继续运行功能。

### Pixel-size Grid Search

```bash
python "$GISAPR_DIR/test_op_PixelSize_GridSearch.py" \
  "${COMMON[@]}" --output_name_root pixelsize_ \
  --Pixelsize_Bounds '(-0.05,0.05)' \
  --Pixelsize_stepsize 0.01 --max_workers_GPU 1
```

这里的 bounds 是**加到 `--apix_PDB` 上的偏移量**，单位 Å/pixel。当 `apix_PDB=1.58` 时，示例搜索范围约为 1.53–1.63 Å/pixel。Particle `--apix` 保持不变。适配器使用零姿态，只改变模型栅格化的采样间隔，因此复制出的 PDB 本身不包含获胜的 sampling 值；还需保留输出标签和运行设置。

可提供任意实际存在的 chain ID。Pixel-size search 内部使用 CC 加 simple-sum 模式，**不接受** `--do_run_CC` 或 `--do_simple_sum`；不要向此命令添加这两个选项。默认跳过 geometric restraint。如果需要启用，应从 `COMMON` 中移除 skip 选项，添加 `--enable_geometric_restraint`，并设置有实际意义的 overlap/distance 限制，因为它继承的阈值默认值极其宽松。Pixel-size search 仅通过 CLI 提供，不支持迭代 checkpoint 继续运行。

## 继续运行

Checkpoint 写入工作目录。继续运行时应使用**相同的方法、初始 PDB、图像、链、评分脚本/模式、搜索设置和 output root**。重复原始命令，并追加相应选项：

| 方法 | Checkpoint 文件名 | 最多追加 20 轮时添加的选项 |
| --- | --- | --- |
| PSO | `my_pso_state_round_N.npz` | `--PSO_continue --PSO_continue_file my_pso_state_round_30.npz --PSO_continue_more_rounds 20` |
| Pattern Search | `my_pattern_state_round_N.npz` | `--Pattern_continue --Pattern_continue_file my_pattern_state_round_30.npz --Pattern_continue_more_rounds 20` |
| Simplex | `my_simplex_state_round_N.npz` | `--Simplex_continue --Simplex_continue_file my_simplex_state_round_30.npz --Simplex_continue_more_rounds 20` |

请将示例 checkpoint 替换为实际存在的文件。三种方法也接受等价的通用别名 `--continue_run`、`--continue_file` 和 `--continue_more_rounds`。无需同时提供两种形式。

PSO 恢复粒子群位置、速度、最佳状态、历史记录和随机状态。Pattern Search 恢复当前点、步长和停止设置。Simplex 恢复全部 simplex 顶点及其评分，而不是仅从最佳顶点重新开始。三种方法的轮数对应不同操作；Simplex checkpoint 的第一轮包含初始 simplex 的评估。

追加轮数并不意味着忽略收敛条件。如果恢复的状态已收敛，Pattern Search 或 Simplex 可能立即停止。PSO 的可选参数 `--PSO_continue_reset_velocities` 会同时扰动速度和位置，常规继续运行时不要添加。显式提供的 PSO pivot 会覆盖保存的 pivot。

不要在同一次继续运行中混用线性求和和修正后的指数评分。如果改变评分定义，应开始新的运行，因为 checkpoint 含有此前计算的目标函数值。本说明不表示 checkpoint 与更早、不同实现的 Simplex 版本兼容。

## 使用现有MRC进行一次搜索

可通过 `wrap_to_search_v2.py`，使用现有模型 MRC 和取向列表对颗粒进行搜索。此流程不执行 PDB 姿态优化或 geometric restraint。

如有需要，可先明确生成模型 MRC：

```bash
python "$GISAPR_DIR/pdb2mrc_gpu_ver_fp32_v3.py" \
  --i model.pdb --o model.mrc \
  --box 256 --apix 1.58 --res 3.16 --gpuid 0
```

此命令保持 PDB 坐标系。独立密度生成器还提供 `--center`，但仅在确实希望改变该坐标系时使用。如果在 CPU 上生成密度，使用 `--device cpu`。

运行搜索：

```bash
python "$GISAPR_DIR/wrap_to_search_v2.py" \
  --script "$GISAPR_DIR/test1_with_isspa_weight_varingKK_search_translation_also_v606_torch_optimized_standalone.py" \
  --i c1_3deg.star --ang c1_3deg.star \
  --mrc model.mrc --p particles_whiten.star --o search_once.txt \
  --oriboxsize 256 --newboxsize 160 --apix 1.58 \
  --voltage 300 --cs 2.7 --maskRadius 110 --maskEdge 6 \
  --psiStep 3 --kk 3 --transRange 0 \
  --doLocalSearch --localRange 20 \
  --gpuid 0 --SplitParticles 1
```

优化后的引擎为兼容保留 `--i`，实际从 `--ang` 读取方向；在此 wrapper 中，两者均填写 angle list。模型 MRC 必须具有匹配的 box size 和坐标系。搜索 sampling 由明确提供的参数决定，因此不能只依赖 MRC header 来设置 pixel size。

此示例省略 FSC。如需使用曲线，添加 `--FSC curve.fsc`。若希望通过 CLI 搜索列表中的所有方向，省略 `--doLocalSearch` 和 `--localRange`。GUI 的 Refine Once 始终生成 local-search 命令。

该 wrapper 写出原始搜索结果。如需单独汇总评分，请创建包含 `search_once.txt` 的 index，并按下一节运行评分脚本。此 wrapper 不会自动生成优化器 checkpoint 或 BEST_FIT PDB。

## Geometric restraint

启用 restraint 时，每个候选构象先与固定 mainbody 进行几何检查，再进入后续密度生成、搜索和评分。计算内容为二值 mask 重叠的 voxel 数，以及 selected-chain mask 和 mainbody mask 之间的最小距离。

拒绝条件为：

```python
overlap >= MAXIUM_ALLOWED_overlapped_pixels or distance >= MAX_MinDistance_Allowed
```

距离单位为 Å。虽然历史选项名中写的是“pixels”，overlap 实际表示 voxel 数。等于阈值也会被拒绝。常规计算流程使用独立的 geometry grid：box size 为 256，pixel size 为 1.5 Å，density threshold 为 1.0；它不是经过裁剪的 search grid。

| 工作流程 | 默认最大 overlap | 默认允许的最大最小距离 |
| --- | --- | --- |
| PSO、Pattern Search、Simplex CLI | 300 | 30 Å |
| 姿态 Grid Search CLI | 130 | 20 Å |
| GUI 生成的姿态 refinement | 未修改时为 150 | 未修改时为 30 Å |
| Pixel-size search | 默认关闭 | 默认关闭 |

例如，可添加以下选项明确设置阈值：

```text
--MAXIUM_ALLOWED_overlapped_pixels 300 --MAX_MinDistance_Allowed 30
```

请按模型和实验方案选择阈值；示例值不是通用几何标准。被拒绝的构象返回预设的优化器目标值 `overlap + MAX_MinDistance_Allowed`，并跳过图像搜索。通过检查的构象只返回**图像评分的负值**。Overlap 和 distance 不会添加到通过检查的评分中。历史参数 `Geometric_restrain_Scaling_Factor` 和 `chain_MASS_in_residues` 也不会改变通过检查后的 score。

使用 `--skip_geometric_restraint` 或 GUI 复选框可以跳过整个几何计算及两项拒绝条件。启用时，permanent worker 重用固定几何信息，无需为每个采样重复执行 chain/mainbody 密度文件流程。已进行检查的姿态仍会保留几何文本诊断结果。

## 评分

优化后的搜索写出每个颗粒的结果，`new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py` 负责汇总。CC 模式中，脚本定义 `c_i = raw_CC_i / 1024`。Z-score 模式中，仅保留小于 9999 的数值。

| 优化器选项 | 独立评分脚本参数 | 拟合成功时的汇总方式 |
| --- | --- | --- |
| `--do_run_CC --do_simple_sum` | `1 1` | 修正后的模式：`sum(exp(c_i))`。 |
| `--do_run_CC` | `1 0` | 基于峰的 CC 统计量。 |
| `--do_simple_sum` | `0 1` | 保留 Z-score 的直接求和。 |
| 两个选项均不提供 | `0 0` | 基于峰的 Z-score 统计量。 |

对于基于峰的统计量，脚本取两个拟合 Gaussian 均值中较大的一个 `mu`，再计算 `count(x > mu) * mu + sum(x[x > mu])`。拟合参数和峰检测诊断结果会打印到终端；此脚本不会保存绘图文件。

“Simple sum”沿用历史名称。即使选择该模式，脚本仍会先尝试 Gaussian 拟合，再将拟合成功时的汇总结果替换为指定的求和方式。**如果拟合失败，保留的回退逻辑会直接对选定的 CC 或 Z-score 值求和。** 该回退分支不会使用指数 CC 求和。输入为空时返回零。

要确认 CC 脚本已修正，请检查 CC 拟合成功分支中 `if do_simple_sum > 0` 下是否包含：

```python
total_estimated_sum_cc = np.sum(np.exp(CC_data))
```

其他位置的 `np.sum(CC_data)`，包括拟合失败的回退分支，均为有意保留。指数求和与直接求和的数值不同，不应混在同一组结果中。比较构象时，应保持颗粒集合和评分模式不变。

如需独立调试，让 `index.txt` 每行包含一个原始搜索文件名，然后运行：

```bash
python "$GISAPR_DIR/new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py" \
  index.txt 1 1 score.txt
```

运行目录应能正确解析 index 中的文件名。每条结果占两行：第一行为搜索文件名，第二行为汇总 score，随后是诊断数值字段。**第二行的第一个数值字段**是排序使用的 score。Score 越大越好；优化器最小化其负值。

## 输出文件与BEST_FIT导出

候选文件名保留现有姿态编码，例如：

```text
rotN1p2_tiltN11p7_psi4p2deg_trans2p1_8p1_N4p8ANG
```

`N` 表示负号，`p` 表示小数点。文件名中的姿态字段有意四舍五入到一位小数。编码中的平移单位为 Å；`rot/tilt/psi` 文件名标签仍表示前述 rotation-vector 分量。

| 文件 | 内容 |
| --- | --- |
| `<root><chain>_<pose>.pdb` | 选定链完成变换后的完整模型。 |
| 候选 `.mrc` | 优化搜索读取的模型密度；这些文件可能较大。 |
| `search_v606opt_*.txt` | 每个颗粒的搜索结果，保留原有格式。文件名中的历史文本 `orient3degree` 不一定代表实际 angle-list 间隔。 |
| `index_for_result_*.txt` | 独立评分脚本的输入 index。 |
| `ReSuLt_*.txt` | 两行一组的文件名/汇总评分记录。 |
| `*_GeometricRestrain_Result.txt` | 已执行 restraint 的姿态对应的几何诊断结果。 |
| `RUN01_*_generate_models_from_pdb.sh`, `RUN01_*_search_script.sh` | 用于调试和重跑的命令记录。 |
| 终端输出中的 Gaussian 诊断结果 | 独立评分脚本的拟合参数、检测到的峰和汇总值。 |
| `my_*_state_round_N.npz` | PSO、Pattern Search 或 Simplex 的 checkpoint 状态。 |

五种 `test_op` 流程均在**正常结束**时调用结果导出器。它收集与本次 output root 和 chain 匹配的评分文件，选择最大的原始 score，找到对应的原始候选 PDB，并复制为：

```text
<winning_candidate_pdb_stem>_BEST_FIT.pdb
```

同时在输出目录写入 `result.log` 和 `<output_root_basename>BEST_FIT.json`。JSON 记录 score、搜索/评分文件名、源 PDB、复制后的 PDB 和结果数量。相同 root 下此前运行的匹配评分也会参与选择，以支持继续运行。独立实验应使用新的 root 和工作目录。

在导出完成前，请保留候选 PDB 和 `ReSuLt_*.txt` 文件。被 geometric restraint 拒绝的候选没有完整的图像评分，不参与此选择。如果没有已完成的评分，不会生成 BEST_FIT PDB。强制终止可能导致导出尚未执行；检查现有结果后，可以手动运行导出器：

```bash
python "$GISAPR_DIR/gisapr_results.py" --output_name_root pso_
```

此独立命令按给定前缀选择结果，因此应使用目标运行准确且独立的 root。

原来的手动评分检查流程仍然可用。在仅含目标运行评分文件的目录中执行：

```bash
cat ReSuLt_*.txt > result.log
python "$GISAPR_DIR/read_Result_print_max.py" --i result.log
```

该脚本打印最大 score 及其对应搜索文件名。自动导出还会完成对应 PDB 的查找和复制，因此优化器正常结束后无需手动按 pose token 搜索文件。

## GPU运行与工作目录

单 GPU 使用 `--gpuid 0`；两个可见 GPU 使用 `--gpuid 0:1`。ID 是应用 `CUDA_VISIBLE_DEVICES` 后的逻辑编号。`0:0:0` 这样的重复 ID 共用一个 worker，不代表三个 GPU。

| 方法 | 候选任务分配选项 |
| --- | --- |
| PSO、Pattern Search | `--max_workers` |
| 姿态 Grid、Pixel-size Grid | `--max_workers_GPU` |
| Simplex | 目标函数仍然按顺序计算。 |

例如，在 PSO 或 Pattern Search 中使用 `--gpuid 0:1 --max_workers 2 --SplitParticles 1`，可并行计算不同候选构象。Grid 方法使用 `--max_workers_GPU 2`。每个设备串行处理分配给自身的请求。

Particle splitting 是另一层并行方式：`--SplitParticles N` 将 STAR 分成若干块，`--doSplitDiffGpu` 将这些块分配到可用 ID 对应的设备上。如果不加此开关，所有块都在候选构象分配到的 GPU 上执行。如果启用不同 GPU 分块开关，但只提供一个数字 GPU ID，wrapper 会根据分块数将其扩展为连续 ID。如需控制允许使用的设备，请明确提供 `0:1` 等列表。不要请求作业分配之外的 GPU。

Worker 依次启动，并在整个优化器运行期间保持存活。这避免了每次 sampling 重复导入 PyTorch，也避免了 Python 环境位于 NFS 时反复支付启动开销。第一个 worker 的启动仍然需要时间。Geometry、density generation 和 search 共用这些 worker；Gaussian 评分脚本仍然是独立的诊断步骤。

小规模 CPU 检查时，将优化器/wrapper 的 `--gpuid 0` 替换为 `--gpuid cpu`。CPU 运行不能替代实际生产 GPU 任务的性能测量。GPU 显存需求取决于 box size、取向数量和任务规模；旧截图中的 48 GB 注释不是普遍要求。

输入路径、GUI JSON 文件和 checkpoint/log 文件名均与工作目录有关。候选文件输出到 `--output_name_root` 指定的目录，但 checkpoint 和优化器日志仍使用工作目录。独立运行目录还可防止某次运行的 `result.log` 或 GUI 配置覆盖另一次运行的文件。

## 故障排查与保留限制

| 现象 | 检查 / 处理 |
| --- | --- |
| 缺少 angle STAR | 生成列表并使用 `--ang` 指定，不要依赖历史默认文件名。 |
| STAR 有效但找不到 image stack | 以工作目录为基准检查 `_rlnImageName` 路径。 |
| 不需要 FSC 却出现 FSC 错误 | 移除 FSC 文件名。Ignore 选项仍然会打开提供的文件。 |
| GUI 没有打印 refinement 命令 | 关闭前 Submit Input、Search 和 Refine 页面。 |
| GUI 意外打印 Refine Once 命令 | 将已保存的 `once_params.json` 移开。 |
| GUI 无法打开 | 使用带 Tkinter 和图形显示环境的 Python，或改用 CLI。 |
| 颗粒预处理出现 import 错误 | 在当前环境中安装 `starfile`。 |
| 预处理出现 `shifted_image` 未定义 | `--newboxsize 0` 与 `--doSkipShifting` 配合使用，或使用应用平移的预处理方式。 |
| EQPS 列表比预期稀疏 | 添加 `--useEQPS`；只指定 `--EQPSangleDegree` 不会选择 EQPS。 |
| 多数候选跳过图像搜索 | 检查几何诊断、输入坐标系和阈值；如果实验方案不需要 restraint，使用 skip 选项。 |
| Score 尺度与预期不同 | 检查评分修正是否安装、评分模式选项、颗粒集合，以及 Gaussian 拟合是否进入直接求和回退分支。 |
| 没有 BEST_FIT PDB | 检查是否有已完成且匹配的评分、候选 PDB 是否保留；如果程序在 finalization 前停止，可运行导出器。 |
| 继续运行立即停止 | Pattern Search 或 Simplex 可能已满足保存的收敛条件。 |
| GPU 显存不足 | 减小任务维度或并行候选数量；检查请求的设备是否与已分配 GPU 一致。 |

`read_search_txt_pick_good_v2p4.py` 作为历史的颗粒筛选脚本保留。其 STAR-header 解析和 particle-index 处理需先与当前搜索输出核对，才能使用导出的 selection。它不属于已验证的自动 refinement/BEST_FIT 流程；请确认选中和排除的颗粒能够完整对应目标输入集合。本次文档更新不修改该脚本或颗粒预处理代码。

## 程序文件、验证与致谢

| 文件 | 作用 |
| --- | --- |
| `GUI_v209.py` | 210 版 GUI 和命令生成。 |
| `test_op_*.py` | 上述各优化器入口。 |
| `func.py`, `func_for_PixelSize_search.py` | 共用的构象评估和 pixel-size 适配。 |
| `gisapr_runtime.py`, `gisapr_worker.py` | Permanent worker 生命周期和请求处理。 |
| `pdb_text.py` | 纯文本 PDB 坐标处理。 |
| `pdb2mrc_gpu_ver_fp32_v3.py` | PyTorch 模型密度生成；保留历史文件名。 |
| `func_check_boundary_for_testing_v8.py` | Geometric restraint 实现。 |
| `wrap_to_search_v2.py` | MRC/angle-list 搜索 wrapper 和颗粒分块。 |
| `test1_with_isspa_weight_varingKK_search_translation_also_v606_torch_optimized_standalone.py` | 优化后的颗粒搜索引擎。 |
| `new_method_to_fit_the_2nd_Gaussian_PEAK_v4.py` | 独立的评分汇总和 Gaussian 诊断。 |
| `optimizer_checkpoint.py` | 共用的继续运行支持。 |
| `gisapr_results.py`, `read_Result_print_max.py` | 自动 BEST_FIT 导出和手动评分检查。 |

源代码包包含 `tests/` 和 `VALIDATION.md`。如需运行包内检查，在选定环境中安装 pytest 后执行：

```bash
python -m pip install pytest
cd "$GISAPR_DIR"
python -m pytest -q tests
```

检查范围见 `VALIDATION.md`。CPU 检查不能证明生产 CUDA 性能、NFS 启动速度或新数据集上的科学准确性。部分与原版比较的检查可能需要提供的 209 参考文件。

项目保留原始包中的以下实现致谢：

- Central Fourier-slice extraction 改编自 [libtilt](https://github.com/teamtomo/libtilt)。
- Equal-sphere partitioning 移植自 [EqualSpherePartition 的 Fortran 实现](https://github.com/GongZheng-Justin/EqualSpherePartition/blob/main/ACM_EqualSphere.f90)。
- HEALPix 和 whitening 工具受到 RELION 启发。
- 模型密度生成受到 EMAN `pdb2mrc` 启发。

包内 `LICENSE` 为 GNU GPL version 3 许可证。重新分发代码时，请查阅随包许可证和保留的源代码声明。
