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
  - [4.1 preprocess.py —— 文档预处理](#41-preprocesspy--文档预处理)
  - [4.2 indexer.py —— 索引构建](#42-indexerpy--索引构建)
  - [4.3 retriever.py —— 查询编码与检索](#43-retrieverpy--查询编码与检索)
  - [4.4 evaluator.py —— 评测系统](#44-evaluatorpy--评测系统)
- [五、关键风险与注意事项](#五关键风险与注意事项)
  - [5.1 RTX 4060 Laptop 显存限制](#51-rtx-4060-laptop-显存限制)
  - [5.2 Windows 原生环境兼容性](#52-windows-原生环境兼容性)
  - [5.3 HuggingFace 访问问题](#53-huggingface-访问问题)
  - [5.4 ColPali MaxSim 计算效率](#54-colpali-maxsim-计算效率)
- [六、里程碑与交付检查清单](#六里程碑与交付检查清单)

---

## 一、推荐项目结构

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
│       ├── metadata.json          # 页面元信息：doc_id, page_idx, image_path
│       └── index/                 # 离线索引
│           ├── embeddings.pt      # patch-level embeddings（.pt 或 .npy）
│           ├── index_meta.json    # 索引元信息：模型名称、维度、时间戳
│           └── page_ids.json      # embedding<->页面 映射表
│
├── src/                           # 核心源码
│   ├── __init__.py
│   ├── preprocess.py              # PDF → 图像转换 + metadata 构建
│   ├── indexer.py                 # ColPali 编码 + 索引构建/存储/加载
│   ├── retriever.py               # 查询编码 + MaxSim 相似度 + Top-k 排序
│   ├── evaluator.py               # Recall@k, Precision@k, MRR, nDCG@k
│   ├── utils.py                   # 配置加载、日志、计时等通用工具
│   └── advanced/                  # 进阶改进方法
│       ├── __init__.py
│       ├── patch_pruner.py        # 候选方向 A：自适应索引压缩
│       └── two_stage.py           # 候选方向 B：两阶段粗精检索
│
├── config/
│   └── default.yaml               # 全局配置文件（模型路径、参数等）

├── .cache/
│   └── huggingface/               # 项目内 Hugging Face 缓存（模型/数据都放这里）
│
├── scripts/                       # 一键执行脚本（Windows .bat）
│   ├── run_preprocess.bat
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
│   └── Project_Plan.md            # 本文件：项目计划
│
├── pyproject.toml                 # uv 原生项目配置（替代 requirements.txt）
├── uv.lock                        # 依赖锁定文件（uv sync 自动生成）
├── .python-version                # uv 读取，固定 Python 3.10
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
rendering:
  backend: pypdfium2 # 固定使用 pypdfium2
  scale: 2.0 # 渲染缩放因子
  target_size: [672, 672] # 送入 ColPali 前 resize 到的目标尺寸
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

创建 `scripts/check_env.py`，一键检测环境是否正确配置。

> ⚠️ **导入顺序约束**：Windows 上 `pyarrow`（datasets/pandas 的依赖）与 `torch` 存在 C++ DLL 冲突，必须先 import datasets 再 import torch，详见 5.2.1 节。

```python
"""环境验证脚本。运行：python scripts/check_env.py

.. note::
   导入顺序很重要：pyarrow（datasets/pandas 的依赖）与 torch 在 Windows 上存在
   C++ DLL 冲突（access violation）。必须先导入 datasets/pandas/pyarrow，再导入
   torch，否则 pyarrow 初始化时会崩溃。
"""
import os
import sys
from pathlib import Path
from importlib import import_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
# 环境检查脚本仅验证包可导入，无需连接 HuggingFace Hub
# datasets / huggingface_hub 在 import 阶段可能发起网络请求，导致超时卡死
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def check_python():
    """检查 Python 版本是否为 3.10.x"""
    v = sys.version_info
    assert v.major == 3 and v.minor == 10, f"需 Python 3.10.x，当前 {v.major}.{v.minor}"
    print(f"[PASS] Python {v.major}.{v.minor}.{v.micro}")


def check_imports_lightweight():
    """先导入不依赖 torch/CUDA 的轻量包（避免 pyarrow DLL 冲突）"""
    pkgs = [
        "numpy", "yaml", "tqdm", "rich",
        "PIL", "pypdfium2", "datasets",
    ]
    for pkg in pkgs:
        try:
            import_module(pkg)
            print(f"[PASS] {pkg}")
        except Exception as e:
            print(f"[FAIL] {pkg}: {e}")


def check_torch_and_ml():
    """在轻量包加载之后再导入 torch 及 ML 栈"""
    import torch  # noqa: F811  -- 延迟导入以避免 pyarrow DLL 冲突
    assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动与 PyTorch 安装"
    print(f"[PASS] CUDA {torch.version.cuda} 可用")
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"       显存: {gb:.1f} GB")

    ml_pkgs = ["transformers", "colpali_engine", "einops"]
    for pkg in ml_pkgs:
        try:
            import_module(pkg)
            print(f"[PASS] {pkg}")
        except Exception as e:
            print(f"[FAIL] {pkg}: {e}")


def check_hf_cache():
    """输出 HuggingFace 缓存路径配置"""
    print(f"[PASS] HF_HOME = {HF_HOME}")
    print(f"[PASS] HF_HUB_CACHE = {HF_HOME / 'hub'}")
    print(f"[PASS] HF_DATASETS_CACHE = {HF_HOME / 'datasets'}")


if __name__ == "__main__":
    print("=" * 50)
    print("ZeroShotVDR 环境检查")
    print("=" * 50)
    check_python()
    check_imports_lightweight()
    check_torch_and_ml()
    check_hf_cache()
    print("=" * 50)
    print("检查完成，全部通过。")
```

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

- [ ] 统计 MMLongBench 数据规模（PDF 数量、总页数、查询数）
- [ ] 抽样查看 PDF 页面（版式类型：纯文/表格/图表/混合）
- [ ] 理解标注格式（ground truth 如何标记相关页面）
- [ ] 确认 MMLongBench 子任务能否直接映射为页级检索；若不能，需在 Phase 1 明确重构标注或更换数据集
- [ ] 确认训练集 / 测试集划分

#### Phase 1 产出

- [ ] 可运行环境（验证脚本全 PASS）
- [ ] 数据集统计笔记（记录在实验报告中）

---

### Phase 2：基础系统实现（5.13 – 5.22）

**目标**：实现完整的 ColPali-based 页级检索管线，跑通端到端流程。

> ⚠️ **开发前置须知**：所有同时使用 `datasets` 与 `torch` 的模块，必须在文件顶部
> `import datasets`（或 `import pandas`）**先于** `import torch`，否则会触发
> pyarrow C++ DLL 冲突导致进程崩溃。详见 5.2.1 节。

#### Step 2.1 文档预处理（成员 A） —— 3 天

**文件**：`src/preprocess.py`

| 子步骤 | 内容                                                                   |
| ------ | ---------------------------------------------------------------------- |
| 2.1.1  | 实现 `PDFProcessor` 类：读取 PDF，调用 pypdfium2 将每页渲染为图像      |
| 2.1.2  | 统一图像分辨率（建议 448x448 或 672x672，对齐 ColPali 输入）           |
| 2.1.3  | 构建 `metadata.json`：记录 doc_id、page_idx、image_path、original_size |
| 2.1.4  | 实现批量处理脚本：遍历 PDF 目录，支持断点续传（跳过已处理）            |

**预期 API**：

```python
# src/preprocess.py
class PDFProcessor:
    def __init__(self, pdf_dir: str, output_dir: str):
        ...
    def process_all(self) -> dict:  # 返回 metadata
        ...
    def process_one(self, pdf_path: str) -> list[dict]:
        ...
```

#### Step 2.2 索引构建（成员 A） —— 4 天

**文件**：`src/indexer.py`

| 子步骤 | 内容                                                                                        |
| ------ | ------------------------------------------------------------------------------------------- |
| 2.2.1  | 加载 ColPali-v1.3 模型（`colpali_engine.models.ColPali`）                                   |
| 2.2.2  | 实现 `PageEncoder`：逐页编码图像 -> patch embeddings                                        |
| 2.2.3  | 处理显存友好的 batching：每次加载 N 张图像（建议 batch_size=4~8 for 8GB VRAM）              |
| 2.2.4  | 设计索引存储格式：`embeddings.pt`（shape: [total_pages, n_patches, dim]） + `page_ids.json` |
| 2.2.5  | 支持增量索引：检测新增/变更页面，仅编码新增部分                                             |
| 2.2.6  | 记录索引构建耗时与存储大小                                                                  |

**预期 API**：

```python
# src/indexer.py
class IndexBuilder:
    def __init__(self, model, metadata: dict, index_dir: str):
        ...
    def build_index(self, batch_size: int = 4) -> None:
        ...
    def load_index(self) -> tuple[torch.Tensor, list[dict]]:
        ...
```

#### Step 2.3 查询编码与检索（成员 B） —— 4 天

**文件**：`src/retriever.py`

| 子步骤 | 内容                                                                                               |
| ------ | -------------------------------------------------------------------------------------------------- |
| 2.3.1  | 使用 ColPali 的文本编码器对查询进行编码（query tokens -> embeddings）                              |
| 2.3.2  | 实现 **MaxSim 相似度**：对查询中每个 token，找到页面 patch 中最高相似度，求和                      |
| 2.3.3  | **显存优化**：避免构建 `[n_queries, n_pages, n_tokens, n_patches]` 全相似度矩阵，改为逐页/分批计算 |
| 2.3.4  | 实现 Top-k 排序（k = 1, 3, 5, 10），返回 `[(doc_id, page_idx, score)]`                             |
| 2.3.5  | 记录单次查询平均延迟                                                                               |

**MaxSim 公式**：

```
Score(Q, P) = Sigma_i max_j Sim(q_i, p_j)

其中：
- Q = {q_1, ..., q_m} 为查询的 m 个 token embeddings
- P = {p_1, ..., p_n} 为页面的 n 个 patch embeddings
- Sim 通常为余弦相似度或点积
```

**预期 API**：

```python
# src/retriever.py
class ColPaliRetriever:
    def __init__(self, model):
        ...
    def encode_query(self, query: str) -> torch.Tensor:
        ...
    def retrieve(self, query_emb: torch.Tensor,
                 index_emb: torch.Tensor,
                 top_k: int = 10) -> list[dict]:
        ...
```

#### Step 2.4 评测系统（成员 B） —— 2 天

**文件**：`src/evaluator.py`

| 子步骤 | 内容                                                                  |
| ------ | --------------------------------------------------------------------- |
| 2.4.1  | 加载 MMLongBench 测试集 ground truth                                  |
| 2.4.2  | 实现 4 项指标计算：**Recall@k**、**Precision@k**、**MRR**、**nDCG@k** |
| 2.4.3  | 编写批量评测脚本：遍历测试查询 -> 检索 -> 对比 ground truth           |
| 2.4.4  | 输出结果：CSV 汇总表 + JSON 详细结果                                  |

**指标定义**：

| 指标        | 公式 / 说明                                            |
| ----------- | ------------------------------------------------------ |
| Recall@k    | (检索到的相关页面数) / (总相关页面数)                  |
| Precision@k | (检索到的相关页面数) / k                               |
| MRR         | Mean Reciprocal Rank：相关页面首次出现的倒数排名的均值 |
| nDCG@k      | Normalized Discounted Cumulative Gain at k             |

**预期 API**：

```python
# src/evaluator.py
class Evaluator:
    def __init__(self, ground_truth: dict):
        ...
    def compute_metrics(self, results: dict, k_values: list[int]) -> pd.DataFrame:
        ...
```

#### Phase 2 产出

- [ ] `src/preprocess.py`：可运行的 PDF->图像 转换脚本
- [ ] `src/indexer.py`：可离线构建 & 加载索引
- [ ] `src/retriever.py`：查询->Top-k 结果端到端
- [ ] `src/evaluator.py`：四项指标计算
- [ ] `config/default.yaml`：全局配置
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

**文件**：`src/advanced/patch_pruner.py`

#### 候选方向 B：两阶段粗精检索

| 步骤 | 内容                                                           |
| ---- | -------------------------------------------------------------- |
| 4B.1 | 第一阶段：页面全局表示（patch mean pooling）快速粗筛 -> top-50 |
| 4B.2 | 第二阶段：在候选集上 full MaxSim 精排 -> top-k                 |
| 4B.3 | 对比不同候选集规模下的速度与精度                               |
| 4B.4 | 目标：精度无损（Recall@5 下降 <= 1%），检索延迟降低 >= 3x      |

**文件**：`src/advanced/two_stage.py`

#### 实验设计（两方向通用）

- [ ] Baseline（ColPali）vs. 改进方法的 4 项指标对比
- [ ] 效率对比（检索延迟、索引大小、显存占用）
- [ ] 消融实验（验证改进中各组件贡献）

#### Phase 4 产出

- [ ] `src/advanced/` 目录下改进代码
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

### 4.1 preprocess.py —— 文档预处理

```python
"""
文档预处理：PDF -> 按页图像 + metadata。
使用 pypdfium2 作为 PDF 渲染引擎（纯 Python，零系统依赖）。
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class PageMeta:
    """单页元信息"""
    doc_id: str
    page_idx: int
    image_path: str
    original_width: int
    original_height: int
    rendered_width: int
    rendered_height: int


class PDFProcessor:
    """
    PDF 页面渲染器（基于 pypdfium2）。

    Parameters
    ----------
    pdf_dir : str
        存放 PDF 的根目录。
    output_dir : str
        图像输出目录。
    target_size : tuple[int, int], optional
        送入模型前的目标尺寸 (width, height)。默认 (672, 672)。
    scale : float, optional
        pypdfium2 渲染时的缩放因子。默认 2.0（约 144 DPI）。
    """

    def __init__(
        self,
        pdf_dir: str,
        output_dir: str,
        target_size: tuple[int, int] = (672, 672),
        scale: float = 2.0,
    ) -> None: ...

    def process_all(self, force: bool = False) -> list[PageMeta]:
        """
        处理所有 PDF。

        Parameters
        ----------
        force : bool
            True 时强制重新处理已存在的页面。

        Returns
        -------
        list[PageMeta]
            全部页面的元信息列表。
        """
        ...

    def process_one(self, pdf_path: str) -> list[PageMeta]:
        """处理单个 PDF 的所有页面。"""
        ...

    def save_metadata(self, path: Optional[str] = None) -> str:
        """将 metadata 保存为 JSON。返回文件路径。"""
        ...
```

### 4.2 indexer.py —— 索引构建

```python
"""
索引构建：使用 ColPali 编码页面图像，存储为离线索引。
"""

import torch
from pathlib import Path
from typing import Iterator

class IndexBuilder:
    """
    ColPali 页面索引构建器。

    Parameters
    ----------
    model : ColPali
        已加载的 ColPali 模型实例。
    metadata : list[PageMeta]
        preprocess 产出的页面元信息列表。
    index_dir : str
        索引存储目录。
    batch_size : int, optional
        GPU 推理批次大小（默认 4，适配 8GB 显存）。
    dtype : torch.dtype, optional
        embedding 存储精度（默认 float16）。
    """

    def __init__(
        self,
        model,
        metadata: list,
        index_dir: str,
        batch_size: int = 4,
        dtype: torch.dtype = torch.float16,
    ) -> None: ...

    def build_index(self) -> None:
        """
        构建全量索引。
        产出：
        - {index_dir}/embeddings.pt  (torch.Tensor: [n_pages, n_patches, dim])
        - {index_dir}/page_ids.json  (list[dict]: doc_id, page_idx, idx)
        - {index_dir}/index_meta.json
        """
        ...

    def load_index(self) -> tuple[torch.Tensor, list[dict]]:
        """
        加载已构建的索引。

        Returns
        -------
        tuple[torch.Tensor, list[dict]]
            (embeddings tensor, page_id mapping)
        """
        ...

    def incremental_update(self, new_metadata: list) -> int:
        """
        增量更新索引：仅编码新增页面。
        返回新增页面数。
        """
        ...

    @property
    def index_stats(self) -> dict:
        """索引统计：总页数、维度、文件大小、构建耗时。"""
        ...
```

### 4.3 retriever.py —— 查询编码与检索

```python
"""
查询编码与检索：文本查询 -> MaxSim 相似度 -> Top-k 页面。
"""

import torch

class ColPaliRetriever:
    """
    ColPali 检索器。

    Parameters
    ----------
    model : ColPali
        已加载的 ColPali 模型实例。
    index_embeddings : torch.Tensor, optional
        预加载的页面索引 [n_pages, n_patches, dim]。
        若为 None，需在 retrieve() 时传入。
    page_ids : list[dict], optional
        页面 ID 映射表。
    """

    def __init__(
        self,
        model,
        index_embeddings: torch.Tensor | None = None,
        page_ids: list[dict] | None = None,
    ) -> None: ...

    def encode_query(self, query: str) -> torch.Tensor:
        """
        将文本查询编码为 token embeddings。

        Returns
        -------
        torch.Tensor
            Query embedding [n_tokens, dim]。
        """
        ...

    def retrieve(
        self,
        query: str | torch.Tensor,
        top_k: int = 10,
        index_embeddings: torch.Tensor | None = None,
        batch_size: int = 64,
    ) -> list[dict]:
        """
        检索 Top-k 相关页面。

        Parameters
        ----------
        query : str | torch.Tensor
            文本查询或预编码的 query embedding。
        top_k : int
            返回结果数量。
        index_embeddings : torch.Tensor, optional
            覆盖实例级别的索引（用于两阶段检索）。
        batch_size : int
            逐批计算 MaxSim 的页面 batch 大小。

        Returns
        -------
        list[dict]
            [{doc_id, page_idx, score, rank}, ...] 按 score 降序。
        """
        ...

    def _maxsim(
        self,
        query_emb: torch.Tensor,
        page_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        MaxSim 相似度计算（单批）。

        Parameters
        ----------
        query_emb : [n_tokens, dim]
        page_emb : [batch_size, n_patches, dim]

        Returns
        -------
        torch.Tensor
            scores [batch_size]。
        """
        ...
```

### 4.4 evaluator.py —— 评测系统

```python
"""
评测系统：Recall@k, Precision@k, MRR, nDCG@k。
"""

import pandas as pd

class Evaluator:
    """
    检索评测器。

    Parameters
    ----------
    ground_truth : dict
        MMLongBench 测试集标注。格式：
        {query_id: [relevant_page_id, ...], ...}
    k_values : list[int], optional
        评测的 k 值列表。默认 [1, 3, 5, 10]。
    """

    def __init__(
        self,
        ground_truth: dict,
        k_values: list[int] = [1, 3, 5, 10],
    ) -> None: ...

    def evaluate(
        self,
        retrieval_results: dict,
    ) -> pd.DataFrame:
        """
        计算全部指标。

        Parameters
        ----------
        retrieval_results : dict
            检索结果。格式：
            {query_id: [(doc_id, page_idx, score), ...], ...}

        Returns
        -------
        pd.DataFrame
            columns: ['k', 'Recall', 'Precision', 'MRR', 'nDCG']
            带 avg 汇总行。
        """
        ...

    def recall_at_k(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> float: ...

    def precision_at_k(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> float: ...

    def mrr(
        self, retrieved: list[str], relevant: set[str]
    ) -> float: ...

    def ndcg_at_k(
        self, retrieved: list[str], relevant: set[str], k: int
    ) -> float: ...
```

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
| datasets       | ⚠️ 需注意      | 见下方 pyarrow DLL 冲突                 |

#### 5.2.1 pyarrow 与 torch 的 C++ DLL 冲突（重要）

**现象**：在已导入 `torch` 的进程中 `import datasets`，Python 进程直接崩溃（Windows access violation），无 Python traceback。

**根因**：`datasets` → `pandas` → `pyarrow` 的 C++ 原生扩展与 `torch` 的 CUDA 运行时库存在 DLL 符号冲突。`pyarrow` 24.0.0 与 `torch` 2.6.0 搭配时，若 torch 先初始化，pyarrow 在加载其 C 扩展（`pyarrow/__init__.py` → `pyarrow/dataset.py`）时触发内存访问违例。

**解决方案**：

```python
# ✅ 正确顺序：先导入 datasets/pandas/pyarrow，再导入 torch
import datasets   # 或 from datasets import ...
import torch

# ❌ 错误顺序：先导入 torch 再导入 datasets 会导致崩溃
import torch
import datasets   # ← access violation!
```

**约束**：项目中所有会同时用到 `datasets` 和 `torch` 的模块，都必须在文件顶部先 `import datasets`（或 `import pandas`），再 `import torch`。这一约束已体现在 `scripts/check_env.py` 的导入顺序中。

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

- [ ] `src/preprocess.py` 可处理全部 PDF
- [ ] `src/indexer.py` 可构建并持久化索引
- [ ] `src/retriever.py` 可对单条查询返回 Top-k
- [ ] `src/evaluator.py` 可计算 4 项指标
- [ ] 端到端管线在 10 条查询子集上输出合理结果

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
