# 零样本视觉文档检索 —— 项目计划

> **项目名称**：ZeroShotVDR  
> **方法基础**：ColPali（Late Interaction + VLM）  
> **数据集**：MMLongBench  
> **硬件环境**：Windows 10 原生 + NVIDIA RTX 4060 Laptop（8 GB 显存）+ Conda + uv  
> **时间跨度**：2025.5.8 – 2025.6.9

---

## 目录

- [一、推荐项目结构](#一推荐项目结构)
- [二、环境配置指导（Win10 原生 + RTX 4060 Laptop + Conda + uv）](#二环境配置指导win10-原生--rtx-4060-laptop--conda--uv)
  - [2.1 概览：分层管理](#21-概览分层管理)
  - [2.2 NVIDIA 驱动与 CUDA](#22-nvidia-驱动与-cuda)
  - [2.3 Conda 安装与 Python 环境](#23-conda-安装与-python-环境)
  - [2.4 uv 安装与 pyproject.toml 配置](#24-uv-安装与-pyprojecttoml-配置)
  - [2.5 PDF 渲染器（pypdfium2）](#25-pdf-渲染器pypdfium2)
  - [2.6 模型权重与数据集下载](#26-模型权重与数据集下载)
  - [2.7 环境验证脚本](#27-环境验证脚本)
- [三、分阶段实施步骤](#三分阶段实施步骤)
  - [Phase 1：文献调研与环境准备（5.8 – 5.12）](#phase-1文献调研与环境准备58--512)
  - [Phase 2：基础系统实现（5.13 – 5.22）](#phase-2基础系统实现513--522)
  - [Phase 3：基础评测与调优（5.23 – 5.25）](#phase-3基础评测与调优523--525)
  - [Phase 4：进阶方法研究与实现（5.26 – 6.2）](#phase-4进阶方法研究与实现526--62)
  - [Phase 5：报告撰写与答辩准备（6.3 – 6.9）](#phase-5报告撰写与答辩准备63--69)
- [四、核心模块接口设计](#四核心模块接口设计)
  - [4.0 数据契约（contracts.py）](#40-数据契约contractspy)
  - [4.1 数据接入层（data/）](#41-数据接入层data)
  - [4.2 索引层（indexing/）](#42-索引层indexing)
  - [4.3 检索层（retrieval/）](#43-检索层retrieval)
  - [4.4 评测层（evaluation/）](#44-评测层evaluation)
  - [4.5 设计决策说明](#45-设计决策说明)
- [五、关键风险与注意事项](#五关键风险与注意事项)
  - [5.1 RTX 4060 Laptop 显存限制](#51-rtx-4060-laptop-显存限制)
  - [5.2 Windows 原生环境兼容性](#52-windows-原生环境兼容性)
  - [5.3 HuggingFace 访问问题](#53-huggingface-访问问题)
  - [5.4 ColPali MaxSim 计算效率](#54-colpali-maxsim-计算效率)
- [六、里程碑与交付检查清单](#六里程碑与交付检查清单)

---

## 一、推荐项目结构

> **v1 修订说明**：模块从 `src/*.py` 扁平结构改为 `src/zeroshot_vdr/` 包内分层结构，
> 以匹配当前仓库的实际包布局，并为 Phase 4 进阶方法预留扩展空间。
> 核心设计遵循五层架构：数据接入 → 索引 → 检索 → 评测 → 配置支撑。

```
ZeroShotVDR/
├── data/                          # 数据目录（不纳入版本控制）
│   ├── MMLongBench/               # 原始数据集
│   │   ├── raw/                   # Hugging Face 下载的 tar.gz 与解压目录
│   │   │   ├── 0_mmlb_data.tar.gz
│   │   │   ├── 1_vrag_image.tar.gz
│   │   │   ├── mmlb_data/
│   │   │   └── mmlb_image/
│   │   └── subsets/               # 可选：抽样子集/任务子集
│   └── processed/                 # 预处理产出
│       ├── images/                # 按页生成的图像（PNG / JPEG）
│       │   └── {doc_id}/
│       │       ├── page_001.png
│       │       └── page_002.png
│       ├── corpus_meta.json       # 页面语料元信息（统一契约）
│       └── index/                 # 离线索引
│           ├── pages/             # 逐页 embedding 文件（每页独立 .pt）
│           │   ├── {page_id}.pt
│           │   └── ...
│           ├── index_meta.json    # 索引元信息：模型名称、维度、时间戳、页数
│           └── page_ids.json      # embedding<->页面 映射表（稳定 ID 契约）
│
├── src/zeroshot_vdr/              # 核心包（editable install）
│   ├── __init__.py
│   ├── contracts.py               # 数据契约：Page, Query, RetrievalResult 等 dataclass
│   ├── config.py                  # 配置加载与管理
│   ├── utils.py                   # 日志、计时、路径等通用工具
│   ├── data/                      # 数据接入层
│   │   ├── __init__.py
│   │   ├── corpus.py              # 页面语料构建（支持 DocumentQA、PDF 等多来源）
│   │   └── adapters.py            # 数据集适配器（将不同数据格式转为统一契约）
│   ├── indexing/                  # 索引层
│   │   ├── __init__.py
│   │   ├── encoder.py             # ColPali 页面编码器
│   │   └── store.py               # 索引持久化与加载（支持多视图）
│   ├── retrieval/                 # 检索执行层
│   │   ├── __init__.py
│   │   ├── encoder.py             # 查询编码器
│   │   ├── scoring.py             # MaxSim 等相似度计算
│   │   └── pipeline.py            # 检索流水线编排（候选召回 → 精排 → 结果组装）
│   ├── evaluation/                # 评测层
│   │   ├── __init__.py
│   │   ├── metrics.py             # 指标实现（与数据集解耦）
│   │   └── ground_truth.py        # Ground truth 加载与适配
│   └── advanced/                  # 进阶改进方法
│       ├── __init__.py
│       ├── patch_pruner.py        # 候选方向 A：自适应索引压缩
│       └── two_stage.py           # 候选方向 B：两阶段粗精检索
│
├── config/
│   └── default.yaml               # 全局配置文件（数据/模型/索引/检索/评测参数）

├── .cache/
│   └── huggingface/               # 项目内 Hugging Face 缓存（模型/数据都放这里）
│
├── scripts/                       # 一键执行脚本
│   ├── run_corpus_build.bat
│   ├── run_index.bat
│   ├── run_retrieval.bat
│   └── run_eval.bat
│
├── outputs/                       # 检索结果 & 评测报告（不纳入版本控制）
│   ├── retrieval_results/
│   │   └── results_top{k}.json    # 各 k 值的检索结果
│   └── eval_reports/
│       └── metrics_summary.csv    # 评测指标汇总
│
├── docs/                          # 文档
│   ├── NJUProject_VDR.md          # 课程任务说明
│   ├── Proposal_VDR.md            # 开题报告
│   ├── Project_Plan.md            # 本文件：项目计划
│   └── revision/                  # 修订记录
│       └── core_module_revision_v1.md
│
├── pyproject.toml                 # uv 原生项目配置
├── uv.lock                        # 依赖锁定文件
├── .python-version                # uv 读取，固定 Python 3.10
├── sitecustomize.py               # PEFT MoE 兼容性补丁（Python 启动时自动加载）
├── .gitignore
├── LICENSE
└── README.md
```

### .gitignore 建议

```gitignore
# Data & outputs
data/
outputs/

# Python
__pycache__/
*.pyc
.venv/

# IDE
.vscode/
.idea/

# Models (HuggingFace cache)
models/
.cache/

# OS
Thumbs.db
Desktop.ini
```

---

## 二、环境配置指导（Win10 原生 + RTX 4060 Laptop + Conda + uv）

### 2.1 概览：分层管理

| 层             | 工具                | 职责                            |
| -------------- | ------------------- | ------------------------------- |
| Python 运行时  | **Conda**           | 安装 Python 3.10，创建隔离环境  |
| 非 Python 依赖 | **无（纯 Python）** | pypdfium2 零外部依赖，CUDA 驱动 |
| Python 包      | **uv**              | 所有 PyPI 包安装、锁定、更新    |
| 项目元数据     | **pyproject.toml**  | uv 原生格式，声明依赖与构建配置 |

> **为什么不用 Conda 装 Python 包？**  
> Conda 的 CUDA toolkit 配套包在 Windows 上可能出现版本冲突或需本地编译。  
> PyTorch 官方在 PyPI 上提供了 Windows + CUDA 12.4 的预编译 wheel，uv 直接下载，零编译。驱动 595.79 支持最高 CUDA 13.2，向下兼容。

---

### 2.2 NVIDIA 驱动与 CUDA

**RTX 4060 Laptop 要求**：

1. **NVIDIA 驱动程序**：>= 546.x（Game Ready 或 Studio 均可）
   - 下载：https://www.nvidia.com/download/
   - 验证：`nvidia-smi` 应显示 CUDA Version: 12.x 或更高

2. **无需单独安装 CUDA Toolkit**  
   PyTorch wheel 自带 CUDA runtime（12.4），无需系统级 CUDA Toolkit。当前驱动 595.79（支持最高 CUDA 13.2）完全兼容。

3. **验证驱动与 CUDA 支持**：
   ```powershell
   nvidia-smi
   # 应看到 Driver Version >= 546, CUDA Version: 12.x 或 13.x
   ```

---

### 2.3 Conda 安装与 Python 环境

1. **安装 Miniconda**（推荐，轻量）：
   - 下载 Windows 64-bit：https://docs.conda.io/en/latest/miniconda.html
   - 安装时勾选 "Add Miniconda3 to my PATH environment variable"

2. **仅创建 Python 环境（不装任何 Python 包）**：

   ```powershell
   # 创建空白环境（仅 Python 3.10）
   conda create -n zeroshotvdr python=3.10 -y

    # 先激活 Conda 基础环境
   conda activate zeroshotvdr
   ```

3. **创建并激活项目虚拟环境**：

   ```powershell
    # 在项目根目录执行，uv 会创建 .venv
   uv sync

   # 日常进入项目时，先激活 conda，再激活项目 .venv
   conda activate zeroshotvdr
   .\.venv\Scripts\Activate.ps1
   ```

   **为什么这样就能最小步骤使用？**
   - `uv sync` 会安装第三方依赖，并将当前仓库以 editable install 的形式安装到 `.venv`。
   - editable install 的作用是让 `src/` 下的项目代码可直接被当前虚拟环境导入。
   - 本项目的兼容性补丁位于仓库根目录的 `sitecustomize.py`。当你在**项目根目录**启动 `.venv` 里的 `python` 时，Python 会通过标准启动流程自动导入它。
   - 这就是为什么后续直接运行 `python` 即可加载 ColPali，而不需要手动再执行任何补丁代码。
   - 约束条件也很明确：请在项目根目录中启动 `python`，并优先使用 `.venv\Scripts\python.exe` 或已激活的项目 `.venv`。

4. **验证**：
   ```powershell
   python --version   # Python 3.10.x
    where python       # 顶部应优先指向 ZeroShotVDR\.venv\Scripts\python.exe
   ```

---

### 2.4 uv 安装与 pyproject.toml 配置

#### 安装 uv

```powershell
# 方式一：pip（在 conda 环境中）
pip install uv

# 方式二：PowerShell 一键安装（全局）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 创建 pyproject.toml

在项目根目录创建 `pyproject.toml`（以下为完整内容）：

```toml
[project]
name = "zeroshot-vdr"
version = "0.1.0"
description = "Zero-Shot Visual Document Retrieval with ColPali"
readme = "README.md"
requires-python = ">=3.10,<3.12"
license = { file = "LICENSE" }
authors = [
    { name = "Member A" },
    { name = "Member B" },
]

dependencies = [
    # Core ML
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    # Transformers & VLMs
    "transformers>=5.3.0,<5.6.0",
    "colpali-engine>=0.3.0",
    "einops>=0.7.0",
    "peft>=0.18.0,<0.19.0",
    # Document processing
    "Pillow>=10.0.0",
    "pypdfium2>=4.0.0",
    # Data & Metrics
    "datasets>=2.18.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
    # Utilities
    "tqdm>=4.65.0",
    "rich>=13.0.0",
    # Analysis & notebooks
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
    "pandas>=2.0.0",
    "ipywidgets>=8.0.0",
]

[tool.uv]
# 默认索引保持 PyPI，torch/torchvision 通过 [tool.uv.sources] 走 CUDA 索引
# 若网络不畅，可取消下一行注释启用清华镜像：
extra-index-url = ["https://mirror.nju.edu.cn/pypi/web/simple"]

[tool.uv.sources]
# 明确指定 PyTorch 来源，防止误装 CPU 版本
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

#### 安装依赖

```powershell
# 在项目根目录执行（确保 conda activate zeroshotvdr 已生效）
uv sync

# uv 将：
# 1. 解析 pyproject.toml 中的依赖
# 2. 从 PyTorch CUDA 索引下载 GPU 版本 torch
# 3. 安装 colpali-engine 及其 VLM 依赖
# 4. 生成 uv.lock 锁定精确版本
```

`uv sync` 默认会安装 notebook、绘图和分析所需依赖，无需额外参数。

**关于“最小步骤即可使用”的补充说明**：

- 当前 ColPali 相关依赖组合需要一个仓库级兼容补丁，补丁文件位于项目根目录的 `sitecustomize.py`。
- 该文件不是通过额外脚本手动执行，而是依赖 Python 的标准启动机制自动导入。
- 因此，推荐的实际使用方式是：在项目根目录执行 `uv sync`，激活 `.venv`，然后直接运行 `python`。
- 若在仓库外目录中启动 Python，或误用了 Conda 环境的解释器而不是 `.venv` 解释器，这个自动补丁可能不会生效。

---

### 2.5 PDF 渲染器（pypdfium2）

**我们选择 `pypdfium2` 作为唯一的 PDF 渲染方案**，理由：

- ✅ **纯 Python 包**：通过 uv/pip 直接安装，零系统级依赖
- ✅ **跨平台一致**：Windows / Linux / macOS 行为完全一致
- ✅ **高性能**：基于 PDFium（Chromium 的 PDF 引擎），渲染质量高
- ❌ ~~pdf2image + poppler~~：Windows 需手动安装 poppler 二进制并配置 PATH，增加环境搭建复杂度

**使用示例**：

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("document.pdf")
page = pdf[0]                    # 获取第一页（索引从 0 开始）
bitmap = page.render(scale=2.0)  # scale=2.0 即 2x 分辨率（约 144 DPI）
pil_image = bitmap.to_pil()      # 转为 PIL Image（可直接送入 ColPali）
```

**统一渲染参数**（在 `config/default.yaml` 中配置）：

```yaml
# ===== 数据配置 =====
data:
  root_dir: "data/MMLongBench/raw"
  subsets: ["docqa"] # Phase 2 主评测集，后续可扩展
  page_id_template: "{subset}/{doc_id}/p{page_idx}"

# ===== 预处理/渲染配置 =====
rendering:
  backend: pypdfium2 # 固定使用 pypdfium2
  scale: 2.0 # 渲染缩放因子
  target_size: [672, 672] # 送入 ColPali 前 resize 到的目标尺寸

# ===== 模型配置 =====
model:
  repo: "vidore/colpali-v1.3"
  base_repo: "vidore/colpaligemma-3b-pt-448-base"
  device: "cuda:0"
  dtype: "bfloat16"

# ===== 索引配置 =====
index:
  dir: "data/processed/index"
  batch_size: 4 # 编码时 GPU batch size（适配 8GB 显存）
  storage_dtype: "float16" # 落盘精度
  per_page_files: true # 是否每页独立存储（v1 修订后固定为 true）

# ===== 检索配置 =====
retrieval:
  top_k_values: [1, 3, 5, 10]
  score_batch_size: 64 # MaxSim 时每批处理页面数
  candidate_strategy: "full" # baseline=full; Phase 4 可切换为 "mean_pool_top50"

# ===== 评测配置 =====
evaluation:
  k_values: [1, 3, 5, 10]
  output_dir: "outputs/eval_reports"

# ===== 路径配置 =====
paths:
  hf_home: ".cache/huggingface"
  hf_hub_cache: ".cache/huggingface/hub"
  hf_datasets_cache: ".cache/huggingface/datasets"
  image_output: "data/processed/images"
  corpus_meta: "data/processed/corpus_meta.json"
```

---

### 2.6 模型权重与数据集下载

这一节建议改成“项目内本地缓存 + 显式下载”。

原方案里有两个容易卡住的点：

- 直接把 `from_pretrained(..., device_map='cuda')` 当作“预下载脚本”，会在下载前就尝试占用 GPU，不利于排错。
- 对 MMLongBench 使用 `load_dataset(...)` 不是官方推荐下载方式；该仓库主体是多个大型 `tar.gz` 文件，直接 `streaming=True` 后再 `list(ds)` 也会把流式读取优势抵消掉。

#### 统一放到项目目录下

以下 PowerShell 环境变量会把 Hugging Face 的模型与数据缓存都放到当前项目内：

```powershell
New-Item -ItemType Directory -Force -Path .cache\huggingface, data\MMLongBench\raw | Out-Null

$env:HF_HOME = "$PWD\.cache\huggingface"
$env:HF_HUB_CACHE = "$PWD\.cache\huggingface\hub"
$env:HF_DATASETS_CACHE = "$PWD\.cache\huggingface\datasets"

# 网络不稳时再启用镜像；PowerShell 中不要用 set
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

#### ColPali-v1.3 模型权重

需要注意，`vidore/colpali-v1.3` 不是完整模型本体，而是依赖 `vidore/colpaligemma-3b-pt-448-base` 的 LoRA adapter。

- `vidore/colpali-v1.3`：约 114 MB
- `vidore/colpaligemma-3b-pt-448-base`：约 5.87 GB

建议先把这两个仓库都下载到项目内缓存，而不是一上来直接跑 GPU 加载：

```powershell
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vidore/colpaligemma-3b-pt-448-base', cache_dir='.cache/huggingface/hub', resume_download=True); snapshot_download(repo_id='vidore/colpali-v1.3', cache_dir='.cache/huggingface/hub', resume_download=True); print('ColPali base + adapter downloaded into project-local cache.')"
```

下载完成后，模型文件会位于项目目录下的 `.cache/huggingface/`，不再落到用户目录。

若只想做最小化联通性验证，可先下载小文件而不是完整权重：

```powershell
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vidore/colpali-v1.3', allow_patterns=['adapter_config.json', 'README.md'], local_dir='models/colpali-v1.3-probe', cache_dir='.cache/huggingface/hub'); print('Probe files downloaded.')"
```

真正运行模型时，仍然可以直接使用仓库名：

```python
from colpali_engine.models import ColPali
import torch

model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()
```

因为前面已经把 `HF_HOME` 指到了项目内，所以实际读取的仍是项目内缓存。

#### MMLongBench 数据集

MMLongBench 官方 README 推荐的是下载各个 `tar.gz` 文件并在本地解压，而不是用 `load_dataset()` 直接拉全量资产。

建议分两步进行：

1. 先做 smoke test，只下载元数据包与一个任务子集。
2. 管线跑通后，再补齐全量图像包。

```powershell
# 第一步：只下载最小可验证集合
uv run hf download ZhaoweiWang/MMLongBench 0_mmlb_data.tar.gz --local-dir data/MMLongBench/raw --repo-type dataset
uv run hf download ZhaoweiWang/MMLongBench 1_vrag_image.tar.gz --local-dir data/MMLongBench/raw --repo-type dataset

# 解压到项目内
tar -xzf data/MMLongBench/raw/0_mmlb_data.tar.gz -C data/MMLongBench/raw
tar -xzf data/MMLongBench/raw/1_vrag_image.tar.gz -C data/MMLongBench/raw
```

全量下载时，再补齐其余任务包：

```powershell
foreach ($file in @(
    '2_vh_image.tar.gz',
    '2_mm-niah_image.tar.gz',
    '3_icl_image.tar.gz',
    '4_summ_image.tar.gz',
    '5_docqa_image.tar.gz'
)) {
    uv run hf download ZhaoweiWang/MMLongBench $file --local-dir data/MMLongBench/raw --repo-type dataset
}
```

如果只是检查数据组织或字段，不要把整个流式数据集转成列表。可以只读取少量样本：

```powershell
uv run python -c "from datasets import load_dataset; ds = load_dataset('ZhaoweiWang/MMLongBench', split='test', streaming=True); first = next(iter(ds)); print(first.keys())"
```

**数据集目录结构（预期）**：

```
data/MMLongBench/
├── raw/
│   ├── 0_mmlb_data.tar.gz
│   ├── 1_vrag_image.tar.gz
│   ├── mmlb_data/
│   └── mmlb_image/
└── processed/
```

---

### 2.7 环境验证脚本

项目提供两个验证脚本，均位于 `scripts/` 目录下。脚本的**唯一权威版本**以实际文件为准，下方仅作功能说明。

#### `scripts/check_env.py` —— 基础环境验证

一键检测以下项是否就绪：

| 检查项            | 内容                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------- |
| Python 版本       | 必须为 3.10.x                                                                               |
| 轻量依赖导入      | numpy, yaml, tqdm, rich, PIL, pypdfium2, datasets（先于 torch 加载，避免 pyarrow DLL 冲突） |
| CUDA & 深度学习栈 | CUDA 可用性、GPU 信息、显存容量；transformers, colpali_engine, einops 导入                  |
| HF 缓存路径       | 输出项目内 `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE` 配置                            |

运行方式：从项目根目录执行 `python scripts/check_env.py`，预期全部项打印 `[PASS]`。

> ⚠️ 脚本以 `HF_HUB_OFFLINE=1` 运行，仅验证包导入，不连接 HuggingFace Hub，避免网络超时。

#### `scripts/test_model_load.py` —— 模型加载与推理验证

在基础环境验证通过后，进一步验证 ColPali-v1.3 模型能否在 GPU 上正常加载并完成前向推理：

| 验证项             | 内容                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| sitecustomize 补丁 | 确认 PEFT MoE 兼容性补丁已生效（解决 transformers v5 + PaliGemma 的 `KeyError: 'llava'` 问题） |
| 模型加载           | `ColPali.from_pretrained("vidore/colpali-v1.3", device_map="cuda:0")`                          |
| 图像编码           | 输入 PIL Image → 输出 patch embeddings `[n_patches, dim]`                                      |
| 文本/查询编码      | 输入查询文本 → 输出 token embeddings `[n_tokens, dim]`                                         |
| 显存报告           | 输出当前 CUDA 显存分配/缓存情况                                                                |

运行方式：从项目根目录执行 `python scripts/test_model_load.py`，预期输出"模型验证全部通过"。

---

## 三、分阶段实施步骤

### Phase 1：文献调研与环境准备（5.8 – 5.12）

**目标**：建立对任务的全面理解，搭好可运行的开发环境。

#### Step 1.1 论文阅读清单

| 优先级 | 论文                                                                                | 关键内容                             | 建议时间 |
| ------ | ----------------------------------------------------------------------------------- | ------------------------------------ | -------- |
| ★★★    | ColPali: Efficient Document Retrieval with Vision Language Models                   | 整体方法、模型架构、Late Interaction | 2h       |
| ★★☆    | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction | Late Interaction 机制原理、MaxSim    | 1.5h     |
| ★☆☆    | MMLongBench 论文 / 文档                                                             | 数据集构成、标注规范、评测指标       | 1h       |

#### Step 1.2 环境搭建

- [ ] 安装 Miniconda，创建 `zeroshotvdr` 环境（Python 3.10）
- [ ] 安装 uv，完成 `uv sync`（依赖安装并创建 `.venv`）
- [ ] 进入项目时按顺序执行 `conda activate zeroshotvdr` 和 `.\.venv\Scripts\Activate.ps1`
- [ ] 将 ColPali base + adapter 下载到项目内 `.cache/huggingface/`
- [ ] 将 MMLongBench 的 `0_mmlb_data.tar.gz` 和一个任务图像包下载到 `data/MMLongBench/raw/`
- [ ] 验证项目内模型缓存与数据目录可读取
- [ ] 运行 `python scripts/check_env.py`，全部项通过

#### Step 1.3 数据集探索

- [ ] 统计 MMLongBench 数据规模：重点确认 DocumentQA 子集的文档数、页面数、查询数
- [ ] 理解 DocumentQA 标注格式：`page_list`（页面图像路径列表）、`ans_page_list`（答案所在页面列表）和 `answer` 字段如何映射为页级检索 ground truth
- [ ] 抽样查看 DocumentQA 页面图像（版式类型：纯文/表格/图表/混合）
- [ ] 确认各 length 档位（K4/K8/K16/K32/K64/K128）的页面规模差异
- [ ] 确认训练集 / 测试集划分（MMLongBench 是否仅提供测试集）

#### Phase 1 产出

- [ ] 可运行环境（验证脚本全 PASS）
- [ ] 数据集统计笔记（记录在实验报告中）

---

### Phase 2：基础系统实现（5.13 – 5.22）

**目标**：实现完整的 ColPali-based 页级检索管线，以 DocumentQA 子集为主评测集跑通端到端流程。优先打通"数据集适配 → 语料构建 → 索引 → 检索 → 评测"闭环，PDF 渲染作为辅助数据源后补。

> ⚠️ **开发前置须知**：
>
> 1. 所有同时使用 `datasets` 与 `torch` 的模块，必须在文件顶部
>    `import datasets`（或 `import pandas`）**先于** `import torch`，否则会触发
>    pyarrow C++ DLL 冲突导致进程崩溃。详见 5.2.1 节。
> 2. 建议将数据读取层与模型层分属不同模块文件，从物理上规避导入顺序问题：
>    `data/` 层依赖 `datasets`，`indexing/` 和 `retrieval/` 层依赖 `torch`，
>    跨层调用通过函数传参而非顶层 import 混合。

#### Step 2.1 数据接入与语料构建（成员 A） —— 3 天

**文件**：`src/zeroshot_vdr/data/corpus.py`, `src/zeroshot_vdr/data/adapters.py`, `src/zeroshot_vdr/contracts.py`

| 子步骤 | 内容                                                                                                                                                         |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2.1.1  | 定义数据契约 `contracts.py`：`Page`（page_id, doc_id, page_idx, image_path）、`Query`（query_id, text）、`RelevanceJudgment`（query_id, page_id, relevance） |
| 2.1.2  | 实现 `DocumentQAAdapter`：从 MMLongBench DocumentQA 子集读取 `page_list` + `ans_page_list`，转为统一 `Page` / `Query` / `RelevanceJudgment` 契约             |
| 2.1.3  | 实现 `PageCorpus` 类：聚合所有适配器产出的页面，分配稳定的 `page_id`（格式：`{subset}/{doc_id}/p{page_idx}`），输出 `corpus_meta.json`                       |
| 2.1.4  | 预留 `PDFAdapter`：基于 pypdfium2 的 PDF→图像 渲染路径，供后续补入非 DocumentQA 数据源                                                                       |

**预期 API**：

```python
# src/zeroshot_vdr/contracts.py
@dataclass
class Page:
    page_id: str       # 稳定唯一标识，如 "docqa/longdocurl_K4/p3"
    doc_id: str
    page_idx: int
    image_path: str
    source_subset: str # 数据来源（docqa / icl / niah / ...）

@dataclass
class Query:
    query_id: str
    text: str
    source_subset: str

# src/zeroshot_vdr/data/corpus.py
class PageCorpus:
    def __init__(self, config): ...
    def build_from_adapters(self) -> list[Page]: ...
    def save_metadata(self, path: str) -> str: ...
    @classmethod
    def load_metadata(cls, path: str) -> list[Page]: ...
```

#### Step 2.2 索引构建（成员 A） —— 4 天

**文件**：`src/zeroshot_vdr/indexing/encoder.py`, `src/zeroshot_vdr/indexing/store.py`

| 子步骤 | 内容                                                                                                                                                                |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.2.1  | 加载 ColPali-v1.3 模型（`colpali_engine.models.ColPali`）                                                                                                           |
| 2.2.2  | 实现 `PageEncoder`：逐页编码图像 → patch embeddings `[n_patches, dim]`                                                                                              |
| 2.2.3  | 处理显存友好的 batching：每次加载 N 张图像（建议 batch_size=2~4 for 8GB VRAM）                                                                                      |
| 2.2.4  | **索引存储**：每页独立存储为 `{page_id}.pt`（shape `[n_patches, dim]`），而非单一巨型张量。此设计支持：（a）真增量追加，（b）后续 patch pruning 后各页 patch 数不同 |
| 2.2.5  | `IndexStore` 类统一管理索引的构建、加载、增量更新，产出 `page_ids.json`（page_id → 文件路径映射）和 `index_meta.json`                                               |
| 2.2.6  | 记录索引构建耗时与存储大小；支持断点续建                                                                                                                            |

**预期 API**：

```python
# src/zeroshot_vdr/indexing/encoder.py
class PageEncoder:
    def __init__(self, model, batch_size: int = 4, dtype=torch.float16): ...
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor: ...
    def encode_corpus(self, pages: list[Page], store: "IndexStore") -> None: ...

# src/zeroshot_vdr/indexing/store.py
class IndexStore:
    def __init__(self, index_dir: str): ...
    def write_page(self, page_id: str, embedding: torch.Tensor) -> None: ...
    def read_page(self, page_id: str) -> torch.Tensor: ...
    def read_all(self, page_ids: list[str] | None = None) -> tuple[torch.Tensor, list[str]]: ...
    def list_page_ids(self) -> list[str]: ...
    @property
    def stats(self) -> dict: ...
```

#### Step 2.3 查询编码与检索（成员 B） —— 4 天

**文件**：`src/zeroshot_vdr/retrieval/encoder.py`, `src/zeroshot_vdr/retrieval/scoring.py`, `src/zeroshot_vdr/retrieval/pipeline.py`

| 子步骤 | 内容                                                                                                                                                                               |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.3.1  | 使用 ColPali 的文本编码器对查询进行编码（query tokens → embeddings `[n_tokens, dim]`）                                                                                             |
| 2.3.2  | 实现 **MaxSim 相似度**模块：对查询中每个 token，找到页面 patch 中最高相似度，求和。独立为 `scoring.py`，便于后续替换或扩展打分函数                                                 |
| 2.3.3  | **显存优化**：逐页或分批计算，避免构建 `[n_queries, n_pages, n_tokens, n_patches]` 全相似度矩阵                                                                                    |
| 2.3.4  | 实现 `RetrievalPipeline`：编排"查询编码 → 候选召回 → 精排打分 → Top-k 结果组装"四个环节。Baseline 中候选召回=全量（等价于直接打分），但流水线结构已为 Phase 4 两阶段检索预留扩展点 |
| 2.3.5  | Top-k 排序（k = 1, 3, 5, 10），返回 `[RetrievalResult(page_id, score, rank)]`                                                                                                      |
| 2.3.6  | 记录单次查询平均延迟                                                                                                                                                               |

**MaxSim 公式**：

```
Score(Q, P) = Σ_i max_j Sim(q_i, p_j)

其中：
- Q = {q_1, ..., q_m} 为查询的 m 个 token embeddings
- P = {p_1, ..., p_n} 为页面的 n 个 patch embeddings
- Sim 通常为余弦相似度或点积
```

**预期 API**：

```python
# src/zeroshot_vdr/retrieval/scoring.py
def maxsim_score(query_emb: torch.Tensor, page_emb: torch.Tensor) -> torch.Tensor:
    """返回标量 score"""
    ...

def batched_maxsim(query_emb: torch.Tensor, pages_emb: torch.Tensor) -> torch.Tensor:
    """返回 [batch_size] scores"""
    ...

# src/zeroshot_vdr/retrieval/pipeline.py
class RetrievalPipeline:
    def __init__(self, model, index_store: IndexStore, config: dict): ...
    def encode_query(self, query: str) -> torch.Tensor: ...
    def retrieve(self, query: str | torch.Tensor, top_k: int = 10) -> list[RetrievalResult]: ...
    def retrieve_batch(self, queries: list[str], top_k: int = 10) -> list[list[RetrievalResult]]: ...
```

#### Step 2.4 评测系统（成员 B） —— 2 天

**文件**：`src/zeroshot_vdr/evaluation/metrics.py`, `src/zeroshot_vdr/evaluation/ground_truth.py`

| 子步骤 | 内容                                                                                                                                                                          |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.4.1  | `ground_truth.py`：加载 MMLongBench 测试集标注，转为统一格式 `{query_id: set[page_id]}`。适配逻辑与指标计算分离，后续新增子集只需增加适配器                                   |
| 2.4.2  | `metrics.py`：实现 4 项指标计算——**Recall@k**、**Precision@k**、**MRR**、**nDCG@k**。指标函数接受 `(retrieved_page_ids, relevant_page_ids, k)` 的标准化输入，与具体数据集解耦 |
| 2.4.3  | 编写批量评测脚本：遍历测试查询 → 检索 → 对比 ground truth                                                                                                                     |
| 2.4.4  | 输出结果：CSV 汇总表 + JSON 详细结果                                                                                                                                          |

**指标定义**：

| 指标        | 公式 / 说明                                            |
| ----------- | ------------------------------------------------------ |
| Recall@k    | (检索到的相关页面数) / (总相关页面数)                  |
| Precision@k | (检索到的相关页面数) / k                               |
| MRR         | Mean Reciprocal Rank：相关页面首次出现的倒数排名的均值 |
| nDCG@k      | Normalized Discounted Cumulative Gain at k             |

**预期 API**：

```python
# src/zeroshot_vdr/evaluation/metrics.py
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def mrr(retrieved: list[str], relevant: set[str]) -> float: ...
def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...

# src/zeroshot_vdr/evaluation/ground_truth.py
class GroundTruthLoader:
    def __init__(self, config): ...
    def load(self, subset: str | None = None) -> dict[str, set[str]]: ...
    # 返回 {query_id: {relevant_page_id, ...}}
```

#### Phase 2 产出

- [ ] `src/zeroshot_vdr/contracts.py`：数据契约定义
- [ ] `src/zeroshot_vdr/data/`：DocumentQA 适配器 + 语料构建可用
- [ ] `src/zeroshot_vdr/indexing/`：可离线构建 & 加载索引，每页独立存储
- [ ] `src/zeroshot_vdr/retrieval/`：流水线式检索，查询→Top-k 结果端到端
- [ ] `src/zeroshot_vdr/evaluation/`：四项指标计算，与数据集解耦
- [ ] `config/default.yaml`：全局配置（数据/模型/索引/检索/评测参数全覆盖）
- [ ] `scripts/` 下各 bat 脚本可用

---

### Phase 3：基础评测与调优（5.23 – 5.25）

**目标**：完成基础系统的全面评测，分析瓶颈与失败案例。

#### Step 3.1 全量评测

- [ ] 在 MMLongBench 测试集上运行完整检索
- [ ] 输出 k=1,3,5,10 下的 Recall, Precision, MRR, nDCG
- [ ] 记录效率指标：索引构建时间、索引文件大小、单次检索延迟

#### Step 3.2 结果分析

- [ ] 绘制 k vs 指标曲线（Recall@k, Precision@k, nDCG@k）
- [ ] 识别 Bad Cases：Recall@10 < 1.0 的查询，人工分析失败原因
- [ ] 分类失败类型（如：跨页图表、视觉密集版面、查询歧义等）

#### Step 3.3 方向决策

- [ ] 基于失败分析，确定 Phase 4 的改进方向（方向 A 或 B，或二者结合）
- [ ] 撰写 Milestone 报告（5.28 截止）

#### Phase 3 产出

- [ ] `outputs/eval_reports/metrics_summary.csv`
- [ ] Bad Cases 分析笔记
- [ ] Milestone 报告草稿

---

### Phase 4：进阶方法研究与实现（5.26 – 6.2）

**目标**：设计并实现一种原创的改进方法，通过实验验证有效性。

#### 候选方向 A：查询感知的自适应索引压缩

| 步骤 | 内容                                                   |
| ---- | ------------------------------------------------------ |
| 4A.1 | 实现 Patch 信息量评分器（信息熵 / 与均值池化的偏离度） |
| 4A.2 | 对每页保留 top-m 个高信息量 patch embeddings           |
| 4A.3 | 实验中对比不同 m 值下的 检索性能 vs 存储开销           |
| 4A.4 | 目标：Recall@5 下降 <= 2%，索引缩减 >= 50%             |

**文件**：`src/zeroshot_vdr/advanced/patch_pruner.py`

#### 候选方向 B：两阶段粗精检索

| 步骤 | 内容                                                           |
| ---- | -------------------------------------------------------------- |
| 4B.1 | 第一阶段：页面全局表示（patch mean pooling）快速粗筛 -> top-50 |
| 4B.2 | 第二阶段：在候选集上 full MaxSim 精排 -> top-k                 |
| 4B.3 | 对比不同候选集规模下的速度与精度                               |
| 4B.4 | 目标：精度无损（Recall@5 下降 <= 1%），检索延迟降低 >= 3x      |

**文件**：`src/zeroshot_vdr/advanced/two_stage.py`

#### 实验设计（两方向通用）

- [ ] Baseline（ColPali）vs. 改进方法的 4 项指标对比
- [ ] 效率对比（检索延迟、索引大小、显存占用）
- [ ] 消融实验（验证改进中各组件贡献）

#### Phase 4 产出

- [ ] `src/zeroshot_vdr/advanced/` 目录下改进代码
- [ ] 对比实验表格（指标 + 效率）
- [ ] 消融实验结果
- [ ] 实验中发现的任何 insight（写入报告）

---

### Phase 5：报告撰写与答辩准备（6.3 – 6.9）

**目标**：完成高质量的实验报告和答辩 PPT，整理代码并提交。

#### Step 5.1 实验报告（NeurIPS 模板，英文，正文 8-9 页）

| 章节         | 负责   | 内容要点                           |
| ------------ | ------ | ---------------------------------- |
| Introduction | 共同   | 任务背景、挑战、本项目贡献         |
| Related Work | 共同   | ColPali、ColBERT、VLM 文档检索综述 |
| Method       | 成员 A | 基础系统架构图 + 改进方法伪代码    |
| Experiments  | 成员 B | 实验设置、主表结果、消融/效率对比  |
| Analysis     | 成员 B | Bad Case 分析、局限性与未来方向    |
| Conclusion   | 共同   | 总结 + 展望                        |

#### Step 5.2 答辩 PPT

- 方法设计思路（图示化）
- 实验亮点（一页摘要对比表）
- 创新性阐述（与已有工作的差异化）

#### Step 5.3 代码整理

- [ ] 统一代码风格（docstring、type hints）
- [ ] 删除调试/死代码
- [ ] 编写 `README.md`（项目说明、快速开始、目录说明）
- [ ] 确认 `uv sync` 一键复现环境
- [ ] 打包提交

#### Phase 5 产出

- [ ] 实验报告 PDF
- [ ] 答辩 PPT
- [ ] 最终代码包
- [ ] README.md

---

## 四、核心模块接口设计

> **v1 修订说明**：本章根据 `docs/revision/core_module_revision_v1.md` 进行了重构。
> 核心变化：（1）预处理层从"PDF 渲染"转向"语料构建 + 数据适配"；
> （2）索引存储从单一巨型张量改为每页独立文件，支持变长 patch 数；
> （3）检索层从单一 Retriever 类改为分环节流水线；
> （4）评测层将指标计算与数据集适配解耦；
> （5）新增显式的数据契约层，统一页面/查询/结果标识体系。

---

### 4.0 数据契约（`contracts.py`）

所有模块间传递的核心对象统一使用以下 dataclass，确保 page_id / query_id 在索引、检索、评测全链路中一致。

```python
"""
数据契约：定义系统中跨模块传递的核心数据结构。
所有 page_id / query_id 均为稳定字符串，贯穿索引→检索→评测全链路。
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Page:
    """页面语料中的单页。"""
    page_id: str           # 稳定唯一标识，格式: {subset}/{doc_id}/p{page_idx}
    doc_id: str            # 所属文档标识
    page_idx: int          # 文档内页码（0-based）
    image_path: str        # 页面图像文件路径
    source_subset: str     # 数据来源子集（docqa / icl / niah / summ / vrag）


@dataclass
class Query:
    """单条检索查询。"""
    query_id: str          # 稳定唯一标识
    text: str              # 查询文本
    source_subset: str     # 所属数据子集


@dataclass
class RetrievalResult:
    """单条检索命中结果。"""
    page_id: str           # 命中页面的稳定 ID
    score: float           # 相似度分数
    rank: int              # 排名（1-based）


@dataclass
class RelevanceJudgment:
    """单条标注：某查询对某页面的相关性。"""
    query_id: str
    page_id: str
    relevance: int         # 0/1 或分级相关度
```

---

### 4.1 数据接入层（`data/`）

**设计原则**：将不同来源的数据组织方式（DocumentQA 的 page_list、PDF 文件等）统一转为 `Page` / `Query` / `RelevanceJudgment` 契约。数据来源差异封装在适配器内部，不传播到索引、检索和评测层。

```python
"""
数据接入层：从 MMLongBench 等数据源构建统一的页面语料与查询集。
优先支持 DocumentQA 子集（page_list + ans_page_list），
同时预留 PDF 渲染适配器供扩展。
"""

from pathlib import Path
from typing import Iterator
from zeroshot_vdr.contracts import Page, Query, RelevanceJudgment


class BaseAdapter:
    """数据集适配器基类：将不同数据格式转为统一契约。"""

    def iter_pages(self) -> Iterator[Page]: ...
    def iter_queries(self) -> Iterator[Query]: ...
    def iter_judgments(self) -> Iterator[RelevanceJudgment]: ...


class DocumentQAAdapter(BaseAdapter):
    """
    MMLongBench DocumentQA 子集适配器。

    输入：mmlb_data/documentQA/ 下的 JSONL 文件（含 page_list + ans_page_list）。
    输出：Page（page_id = "docqa/{filename}/{doc_id}/p{idx}"）、
          Query、RelevanceJudgment。
    """

    def __init__(self, data_dir: str, subset_filter: str | None = None): ...
    def iter_pages(self) -> Iterator[Page]: ...
    def iter_queries(self) -> Iterator[Query]: ...
    def iter_judgments(self) -> Iterator[RelevanceJudgment]: ...


class PDFAdapter(BaseAdapter):
    """
    PDF 页面渲染适配器（基于 pypdfium2）。
    用于从原始 PDF 文件构建页面语料（非 DocumentQA 路径时使用）。

    Parameters
    ----------
    pdf_dir : str
    output_dir : str
    target_size : tuple[int, int]
    scale : float
    """

    def __init__(self, pdf_dir: str, output_dir: str,
                 target_size: tuple[int, int] = (672, 672),
                 scale: float = 2.0): ...
    def iter_pages(self) -> Iterator[Page]: ...


class PageCorpus:
    """
    页面语料聚合器。

    职责：
    1. 聚合多个 Adapter 产出的 Page
    2. 为每个 Page 分配稳定的 page_id
    3. 持久化 corpus_meta.json
    """

    def __init__(self, config: dict): ...

    def build(self, adapters: list[BaseAdapter]) -> list[Page]:
        """从多个适配器构建统一页面语料。"""
        ...

    def save_metadata(self, path: str | None = None) -> str:
        """保存 corpus_meta.json。返回文件路径。"""
        ...

    @classmethod
    def load_metadata(cls, path: str) -> list[Page]:
        """加载 corpus_meta.json 还原 Page 列表。"""
        ...
```

---

### 4.2 索引层（`indexing/`）

**设计原则**：

1. 编码（encoder）与存储（store）在抽象上分离，便于后续替换编码器或存储后端。
2. 每页独立存储为一个 `.pt` 文件（`{page_id}.pt`），而非单一巨型张量。这样天然支持：
   - 增量追加（新页直接写入新文件）
   - Patch pruning 后各页 patch 数不同的场景
   - 按需加载（两阶段检索中粗筛阶段可只加载均值池化视图）

```python
"""
索引层：使用 ColPali 编码页面图像，并以逐页独立文件形式持久化。
"""

import torch
from PIL import Image
from zeroshot_vdr.contracts import Page


class PageEncoder:
    """
    ColPali 页面编码器。

    Parameters
    ----------
    model : ColPali
    batch_size : int
    dtype : torch.dtype
    device : str
    """

    def __init__(self, model, batch_size: int = 4,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda:0"): ...

    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """编码单张页面图像 → [n_patches, dim]"""
        ...

    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """编码一批图像 → [batch, n_patches, dim]"""
        ...

    def encode_corpus(self, pages: list[Page],
                      store: "IndexStore",
                      show_progress: bool = True) -> None:
        """遍历页面语料，逐批编码并写入索引存储。"""
        ...


class IndexStore:
    """
    索引持久化存储。

    存储布局：
        {index_dir}/
        ├── pages/
        │   ├── {page_id}.pt    # torch.Tensor [n_patches, dim]
        │   └── ...
        ├── page_ids.json       # [page_id, ...] 有序列表
        └── index_meta.json     # 模型名、维度、时间戳、总页数

    支持的操作：写入、读取（单页/批量/全量）、增量追加、统计。
    """

    def __init__(self, index_dir: str): ...

    # -- 写入 --
    def write_page(self, page_id: str, embedding: torch.Tensor) -> None: ...
    def write_batch(self, page_ids: list[str],
                    embeddings: torch.Tensor) -> None: ...

    # -- 读取 --
    def read_page(self, page_id: str) -> torch.Tensor: ...
    def read_batch(self, page_ids: list[str]) -> torch.Tensor:
        """返回 [len(page_ids), n_patches, dim]（需各页 patch 数相同）"""
        ...
    def read_all(self) -> tuple[torch.Tensor, list[str]]:
        """返回 (stacked_tensor, page_id_list)。
        仅当全量页面 patch 数相同时可用；变长场景请逐页读取。"""
        ...

    def list_page_ids(self) -> list[str]: ...

    # -- 视图 --
    def get_mean_pooled_view(self) -> tuple[torch.Tensor, list[str]]:
        """返回全量页面的均值池化向量 [n_pages, dim]，供两阶段粗筛使用。"""
        ...

    # -- 元信息 --
    @property
    def stats(self) -> dict:
        """{num_pages, dim, total_size_mb, build_time_sec, ...}"""
        ...
```

---

### 4.3 检索层（`retrieval/`）

**设计原则**：检索不是单一步骤，而是"查询编码 → 候选召回 → 精排打分 → 结果组装"的流水线。Baseline 中候选召回=全量（等价于直接全量 MaxSim），但流水线结构已为 Phase 4 两阶段检索预留扩展点。

```python
"""
检索层：查询编码 → 候选召回 → MaxSim 精排 → Top-k 结果组装。
"""

import torch
from zeroshot_vdr.contracts import Page, RetrievalResult
from zeroshot_vdr.indexing.store import IndexStore


class QueryEncoder:
    """ColPali 查询编码器。"""

    def __init__(self, model): ...
    def encode(self, query: str) -> torch.Tensor:
        """文本查询 → [n_tokens, dim]"""
        ...


# -- 打分函数（scoring.py） --

def maxsim_score(query_emb: torch.Tensor, page_emb: torch.Tensor) -> torch.Tensor:
    """
    MaxSim 相似度（单页）。

    Parameters
    ----------
    query_emb : [n_tokens, dim]
    page_emb : [n_patches, dim]

    Returns
    -------
    torch.Tensor : 标量 score
    """
    ...

def batched_maxsim(query_emb: torch.Tensor,
                   pages_emb: torch.Tensor) -> torch.Tensor:
    """
    批量 MaxSim。

    Parameters
    ----------
    query_emb : [n_tokens, dim]
    pages_emb : [batch_size, n_patches, dim]

    Returns
    -------
    torch.Tensor : [batch_size] scores
    """
    ...


# -- 检索流水线（pipeline.py） --

class RetrievalPipeline:
    """
    检索流水线编排器。

    流程：
    1. encode_query(query)      → query embedding
    2. generate_candidates()    → 候选 page_id 列表（baseline = 全量）
    3. score_candidates()       → 对候选逐批 MaxSim 打分
    4. assemble_results(top_k)  → 排序、截断、封装为 RetrievalResult 列表

    Phase 4 中可通过替换 generate_candidates() 策略接入两阶段检索。
    """

    def __init__(self, model, index_store: IndexStore,
                 query_encoder: QueryEncoder | None = None,
                 config: dict | None = None): ...

    def encode_query(self, query: str) -> torch.Tensor: ...

    def retrieve(self, query: str | torch.Tensor,
                 top_k: int = 10,
                 candidate_ids: list[str] | None = None,
                 score_batch_size: int = 64) -> list[RetrievalResult]:
        """
        检索 Top-k 相关页面。

        Parameters
        ----------
        query : str | torch.Tensor
        top_k : int
        candidate_ids : list[str] | None
            候选页面列表；为 None 时默认全量检索。
        score_batch_size : int
            逐批计算 MaxSim 的页面 batch 大小。

        Returns
        -------
        list[RetrievalResult]
        """
        ...

    def retrieve_batch(self, queries: list[str],
                       top_k: int = 10,
                       **kwargs) -> list[list[RetrievalResult]]: ...

    def generate_candidates(self, query_emb: torch.Tensor,
                            top_n: int | None = None) -> list[str]:
        """
        候选召回阶段。
        Baseline：返回全量 page_ids。
        Phase 4：可替换为均值池化粗筛。
        """
        ...

    def score_candidates(self, query_emb: torch.Tensor,
                         candidate_ids: list[str],
                         batch_size: int = 64) -> torch.Tensor:
        """对候选集逐批 MaxSim 打分，返回 [n_candidates] scores。"""
        ...
```

---

### 4.4 评测层（`evaluation/`）

**设计原则**：

1. 指标计算函数接受标准化输入 `(retrieved_page_ids, relevant_page_ids, k)`，与具体数据集解耦。
2. Ground truth 的加载和格式转换在独立模块中完成，新增评测子集只需增加适配逻辑。

```python
"""
评测层：指标计算与 ground truth 加载。
"""

import pandas as pd
from zeroshot_vdr.contracts import RetrievalResult


# -- 指标计算（metrics.py） --

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k = |retrieved[:k] ∩ relevant| / |relevant|"""
    ...

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k = |retrieved[:k] ∩ relevant| / k"""
    ...

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank：首个相关页面的倒数排名"""
    ...

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG@k"""
    ...

def compute_all_metrics(
    retrieval_results: dict[str, list[RetrievalResult]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    """
    批量计算全部指标。

    Parameters
    ----------
    retrieval_results : {query_id: [RetrievalResult, ...]}
    ground_truth : {query_id: {relevant_page_id, ...}}
    k_values : list[int]

    Returns
    -------
    pd.DataFrame : columns=['k', 'Recall', 'Precision', 'MRR', 'nDCG']
    """
    ...


# -- Ground Truth 加载（ground_truth.py） --

class GroundTruthLoader:
    """
    Ground truth 加载与格式转换。

    职责：
    1. 从 MMLongBench 标注数据中提取 (query_id, page_id) 对
    2. 转为统一的 {query_id: set[page_id]} 格式
    3. page_id 使用与 PageCorpus 一致的命名规则
    """

    def __init__(self, config: dict): ...

    def load(self, subset: str | None = None) -> dict[str, set[str]]:
        """
        加载 ground truth。

        Parameters
        ----------
        subset : str | None
            限定子集（docqa/icl/niah/summ/vrag）；None 表示全部。

        Returns
        -------
        dict[str, set[str]] : {query_id: {relevant_page_id, ...}}
        """
        ...

    @staticmethod
    def build_page_id(subset: str, doc_id: str, page_idx: int) -> str:
        """构造与 PageCorpus 一致的 page_id。"""
        ...
```

---

### 4.5 设计决策说明

以下说明为 v1 修订中几个关键架构决策的原理，供后续实现和 Phase 4 扩展时参考。

#### 4.5.1 为什么需要稳定 page_id 契约？

| 问题         | 原方案                                                             | 修订后                                                    |
| ------------ | ------------------------------------------------------------------ | --------------------------------------------------------- |
| 页面标识方式 | 临时拼接 `{doc_id}_{page_idx}`                                     | `Page.page_id` 为规范格式 `{subset}/{doc_id}/p{page_idx}` |
| ID 一致性    | Retriever 返回 `doc_id, page_idx`，Evaluator 期望 `page_id` 字符串 | 全链路使用同一 `page_id` 字符串                           |
| 多子集扩展   | 不同子集的 doc_id 可能冲突                                         | `page_id` 含 subset 前缀，天然隔离                        |

**核心原则**：所有模块间传递的页面标识和查询标识必须使用稳定、可追溯的字符串，禁止在各模块间临时拼接或拆解。

#### 4.5.2 为什么索引采用逐页独立存储？

| 考量                      | 单一巨型张量 (`embeddings.pt`) | 逐页独立文件 (`{page_id}.pt`)           |
| ------------------------- | ------------------------------ | --------------------------------------- |
| 增量追加                  | 需重写整份索引                 | 直接写入新文件                          |
| Patch pruning（Phase 4A） | 各页 patch 数不同时无法 stack  | 天然支持变长                            |
| 按需加载                  | 必须全量加载或手动切片         | 按 page_id 精确读取                     |
| 两阶段粗筛（Phase 4B）    | 需额外维护全局均值视图         | `get_mean_pooled_view()` 实时计算或缓存 |

#### 4.5.3 为什么检索要分层而不是单一步骤？

原 `ColPaliRetriever` 将查询编码、全量 MaxSim、Top-k 排序揉在一个类中。分层后：

```
encode_query → generate_candidates → score_candidates → assemble_results
```

- **Baseline**：`generate_candidates` 返回全量 page_ids，等价于原行为。
- **两阶段检索（Phase 4B）**：只需替换 `generate_candidates` 为均值池化粗筛，其余环节复用。
- **索引压缩（Phase 4A）**：只需在 `score_candidates` 中处理变长 patch 的张量。

#### 4.5.4 模块边界与 Windows 导入约束

为确保 Windows 下 `datasets`（→ pyarrow）与 `torch` 不产生 DLL 冲突，模块在设计上遵循以下边界：

| 层            | 主要依赖                       | 避免顶层导入                                              |
| ------------- | ------------------------------ | --------------------------------------------------------- |
| `data/`       | `datasets`, `PIL`, `pypdfium2` | `torch`                                                   |
| `indexing/`   | `torch`, `PIL`, `transformers` | `datasets`                                                |
| `retrieval/`  | `torch`, `transformers`        | `datasets`                                                |
| `evaluation/` | `pandas`（→ pyarrow）, `numpy` | `torch`（指标函数接受 Python list/set，避免 tensor 操作） |

若某模块必须同时使用二者，应在模块顶部**先 `import datasets` / `import pandas`，再 `import torch`**，详见 5.2.1 节。

---

## 五、关键风险与注意事项

### 5.1 RTX 4060 Laptop 显存限制

> **Warning: 8 GB 显存是最大约束。**

**缓解措施**：

| 场景        | 策略                                           |
| ----------- | ---------------------------------------------- |
| 图像编码    | `batch_size=4`（或更小），使用 `torch.float16` |
| 索引加载    | 使用 `mmap_mode=True` 加载 `.pt`，避免全量加载 |
| MaxSim 计算 | 分批计算（page batch），避免构建完整相似度矩阵 |
| 进阶方法    | 候选方向 A（压缩索引）天然减低显存需求         |

**显存预算估算**（ColPali-v1.3）：

- 模型参数：~2-3 GB（bfloat16）
- 单张图像编码：~1 GB（672x672 输入，含中间激活）
- MaxSim 计算：可控（逐批释放）

### 5.2 Windows 原生环境兼容性

| 组件           | Windows 兼容性 | 说明                                    |
| -------------- | -------------- | --------------------------------------- |
| PyTorch        | OK             | 官方预编译 wheel                        |
| transformers   | OK             | 纯 Python                               |
| colpali-engine | OK             | 纯 Python，依赖 transformers            |
| pypdfium2      | OK             | 纯 Python，零系统依赖，Windows 首选方案 |
| uv             | OK             | Windows 原生支持                        |
| datasets       | 需注意         | 见下方 pyarrow DLL 冲突                 |

#### 5.2.1 pyarrow 与 torch 的 C++ DLL 冲突（重要）

**现象**：在已导入 `torch` 的进程中 `import datasets`，Python 进程直接崩溃（Windows access violation），无 Python traceback。

**根因**：`datasets` → `pandas` → `pyarrow` 的 C++ 原生扩展与 `torch` 的 CUDA 运行时库存在 DLL 符号冲突。`pyarrow` 24.0.0 与 `torch` 2.6.0 搭配时，若 torch 先初始化，pyarrow 在加载其 C 扩展（`pyarrow/__init__.py` → `pyarrow/dataset.py`）时触发内存访问违例。

**解决方案**：

```python
# 正确顺序：先导入 datasets/pandas/pyarrow，再导入 torch
import datasets   # 或 from datasets import ...
import torch

# 错误顺序：先导入 torch 再导入 datasets 会导致崩溃
import torch
import datasets   # ← access violation!
```

**模块边界隔离策略**（根本性规避）：

| 层            | 主要依赖                       | 应避免的顶层导入                        |
| ------------- | ------------------------------ | --------------------------------------- |
| `data/`       | `datasets`, `PIL`, `pypdfium2` | `torch`                                 |
| `indexing/`   | `torch`, `PIL`, `transformers` | `datasets`                              |
| `retrieval/`  | `torch`, `transformers`        | `datasets`                              |
| `evaluation/` | `pandas`, `numpy`              | `torch`（指标函数使用 Python list/set） |

通过将数据读取层与模型层分属不同模块文件，从物理上规避了在同一文件顶部混合导入的风险。跨层数据传递通过函数参数（dataclass / Python list）完成，而非顶层 import 混合。

### 5.3 HuggingFace 访问问题

若无法直接访问 huggingface.co：

```powershell
# PowerShell 中设置镜像环境变量（在当前终端生效）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:HF_HUB_CACHE = "$PWD\.cache\huggingface\hub"
$env:HF_DATASETS_CACHE = "$PWD\.cache\huggingface\datasets"

# 或在代码中设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

若用 `transformers` 下载屡次超时：

```python
# 下载策略配置
from huggingface_hub import snapshot_download
snapshot_download(
    "vidore/colpali-v1.3",
    cache_dir=".cache/huggingface/hub",  # 放到项目目录下
    resume_download=True,
)
```

### 5.4 ColPali MaxSim 计算效率

MaxSim 计算复杂度为 O(m x n x d)，其中 m = 查询 token 数，n = 页面 patch 数，d = 维度。

**高效实现要点**：

1. 使用 `torch.einsum` 或 `torch.bmm` 进行批量矩阵乘法
2. 避免 Python for 循环逐页计算
3. 预归一化 embeddings（余弦相似度 -> 点积）
4. 对大索引使用分块计算 + 内存映射

---

## 六、里程碑与交付检查清单

### Milestone 1：环境与数据就绪（5.12）

- [ ] Conda 环境 `zeroshotvdr` 创建成功
- [ ] `uv sync` 无报错完成
- [ ] `python scripts/check_env.py` 全部 PASS
- [ ] ColPali-v1.3 权重下载至本地
- [ ] MMLongBench 数据集可读取，统计信息已记录

### Milestone 2：基础管线跑通（5.22）

- [ ] `src/zeroshot_vdr/data/`：DocumentQA 适配器 + 语料构建可用
- [ ] `src/zeroshot_vdr/indexing/`：可离线构建 & 加载索引（逐页独立存储）
- [ ] `src/zeroshot_vdr/retrieval/`：流水线式检索，查询→Top-k 结果端到端
- [ ] `src/zeroshot_vdr/evaluation/`：四项指标计算可用，与数据集解耦
- [ ] 端到端管线在 DocumentQA 10 条查询子集上输出合理结果

### Milestone 3：基础评测完成 + Milestone 报告（5.28）

- [ ] 全量测试集评测结果（4 指标 x 4 k 值）
- [ ] Bad Cases 分析完成
- [ ] 改进方向已确定
- [ ] Milestone 报告提交

### Milestone 4：进阶方法完成（6.2）

- [ ] 改进方法代码实现完毕
- [ ] Baseline vs. 改进方法对比实验完成
- [ ] 消融实验完成
- [ ] 效率对比数据齐全

### Milestone 5：最终提交（6.9）

- [ ] 实验报告 PDF（NeurIPS 模板，8-9 页）
- [ ] 答辩 PPT
- [ ] 代码包整理（README、注释、死代码清理）
- [ ] 最终提交

---

> **参考文献**
>
> - Faysse, M. et al. _ColPali: Efficient Document Retrieval with Vision Language Models_. arXiv, 2024.
> - Khattab, O. and Zaharia, M. _ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT_. SIGIR, 2020.
> - MMLongBench Dataset: https://huggingface.co/datasets/ZhaoweiWang/MMLongBench
> - ColPali Model: https://huggingface.co/vidore/colpali-v1.3
