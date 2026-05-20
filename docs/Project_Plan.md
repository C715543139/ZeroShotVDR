# 零样本视觉文档检索 —— 项目计划

> **项目名称**：ZeroShotVDR  
> **方法基础**：ColPali（Late Interaction + VLM）  
> **数据集**：MMLongBench  
> **硬件环境**：Ubuntu + 2x NVIDIA RTX 3090（24 GB 显存）+ Conda + uv  
> **时间跨度**：2025.5.8 – 2025.6.9

> **v9 同步说明（2026-05-20）**：本计划已按当前 Phase 4 完成状态同步。`src/zeroshot_vdr/advanced/` 已落地 `two_stage.py`、`neighbors.py`、`profiling.py` 与 `mean_pool_cache.py`；脚本目录已重组为 `scripts/run/` 与 `scripts/command/`，根目录新增 `main.py` 作为统一入口。Phase 4 valid-only 全量实验矩阵与 cache 版本均已跑通，当前推荐方法为 `adaptive_neighbors + mean-pool cache`，在 14,385 条有效标注查询上达到 Recall@10 = 0.8523、nDCG@10 = 0.6325、平均延迟约 0.060 s/query；Phase 5 的最终论文、答辩 PPT 与整理提交仍待完成。

---

## 目录

- [一、推荐项目结构](#一推荐项目结构)
  │ ├── core_module_revision_v1.md
  │ ├── core_module_revision_v2.md
  │ ├── core_module_revision_v3.md
  │ ├── core_module_revision_v4.md
  │ └── core_module_revision_v5.md
- [二、环境配置指导（Linux + Conda + uv）](#二环境配置指导linux--conda--uv)
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
  - [5.1 显存限制与批次大小选择](#51-显存限制与批次大小选择)
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
│   │   │   ├── 0_mmlb_data.tar.gz       # 元数据（Phase 1 必选）
│   │   │   ├── 5_docqa_image.tar.gz      # DocumentQA 图像（Phase 2 baseline 必选）
│   │   │   ├── 1_vrag_image.tar.gz       # 扩展任务资源（后续可选下载）
│   │   │   ├── ...                       # 其他任务图像包
│   │   │   ├── mmlb_data/
│   │   │   └── mmlb_image/
│   │   └── subsets/               # 可选：抽样子集/任务子集
│   └── processed/                 # 预处理产出
│       ├── images/                # 按页生成的图像（PNG / JPEG）
│       │   └── {doc_id}/
│       │       ├── page_001.png
│       │       └── page_002.png
│       ├── corpus_meta.json       # 页面语料元信息（统一契约）
│       └── {index_name}/          # 离线索引（当前稳定 full run 使用 index_stable_page_ids）
│           ├── pages/             # 逐页 embedding 文件（每页独立 .pt）
│           │   ├── {page_id}.pt
│           │   └── ...
│           ├── index_meta.json    # 索引元信息：模型名称、维度、时间戳、页数
│           └── page_ids.json      # 有序 JSON 数组 [page_id, ...]（稳定 ID 契约）
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
│   └── advanced/                  # Phase 4 两阶段检索与分析模块
│       ├── __init__.py
│       ├── two_stage.py           # TwoStageRetriever 与自适应候选逻辑
│       ├── neighbors.py           # page_id 解析与邻页扩展
│       ├── mean_pool_cache.py     # MeanPoolCache 与批量构建逻辑
│       └── profiling.py           # Trace / profiling 数据结构
│
├── config/
│   └── default.yaml               # 全局配置文件（数据/模型/索引/检索/评测参数）

├── .cache/
│   └── huggingface/               # 项目内 Hugging Face 缓存（模型/数据都放这里）
│
├── scripts/                       # 运行、分析与辅助命令
│   ├── run/
│   │   ├── run_step3_eval.py      # Step 3.1 评测脚本
│   │   ├── run_step3_analysis.py  # Step 3.2 分析脚本
│   │   ├── run_phase4_eval.py     # Phase 4 单次评测入口
│   │   └── run_phase4_full.sh     # Phase 4 批量全量运行脚本
│   ├── command/
│   │   ├── env.sh                 # bash 激活入口
│   │   ├── check_env.py           # 基础环境验证
│   │   ├── check_phase4_progress.sh
│   │   └── run_step3_clean.py     # Step 3 清理脚本
│   ├── analyze_phase4_trace.py    # Phase 4 trace / slice 分析
│   ├── backfill_step3_phase4_schema.py
│   └── test_model_load.py         # ColPali 模型加载验证
│
├── outputs/                       # 检索结果 & 评测报告（不纳入版本控制）
│   ├── retrieval_results/
│   │   └── results_top{k}.json    # 各 k 值的检索结果
│   └── eval_reports/
│       └── {run_name}/
│           ├── summary.json
│           ├── slice_metrics.csv
│           ├── bucket_metrics.csv
│           ├── phase4_trace.jsonl
│           ├── trace_summary.json
│           ├── metrics_summary.csv
│           ├── metrics_overall.csv
│           ├── metrics_by_subtask.csv
│           ├── metrics_by_length.csv
│           ├── retrieval_details.json
│           └── run_summary.json
│
├── docs/                          # 文档
│   ├── NJUProject_VDR.md          # 课程任务说明
│   ├── Proposal_VDR.md            # 开题报告
│   ├── Project_Plan.md            # 本文件：项目计划
│   ├── Milestone_Report_Phase3.md # 阶段三里程碑报告
│   ├── Milestone_Report_Phase4.md # 阶段四里程碑报告
│   └── revision/                  # 修订记录
│       ├── core_module_revision_v1.md
│       ├── core_module_revision_v2.md
│       ├── core_module_revision_v3.md
│       ├── core_module_revision_v4.md
│       ├── core_module_revision_v5.md
│       ├── core_module_revision_v6.md
│       ├── core_module_revision_v7.md
│       ├── project_plan_revision_v1.md
│       ├── project_plan_revision_v2.md
│       ├── milestone_report_phase4_revision_v1.md
│       ├── step3_eval_revision_v1.md
│       └── step3_direction_revision_v1.md
│
├── main.py                        # 统一命令入口（step3 / phase4 / command）
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

### 1.1 命名分层与概念体系

> **v2 修订说明**：为避免全文中 `DocumentQA` / `documentQA` / `docqa` 等命名混用，
> 以及任务类别、子任务、长度档位、文档标识、页面标识之间概念不清的问题，
> 本节建立一套统一的命名分层体系。全文所有模块、配置、脚本、结果文件均遵循此体系。

MMLongBench 数据集及本项目的概念分为以下五个层级，由粗到细：

| 层级        | 英文名        | 说明                             | 本项目中的值（Phase 2）                                    |
| ----------- | ------------- | -------------------------------- | ---------------------------------------------------------- |
| L1 任务族   | `task_family` | MMLongBench 中的大类任务         | `docqa`, `icl`, `niah`, `summ`, `vrag`                     |
| L2 子任务   | `subtask`     | 任务族下的具体数据集变体         | `longdocurl`, `mmlongdoc`, `slidevqa`, `text_mmlongdoc` 等 |
| L3 长度档位 | `length`      | 上下文长度 K 值                  | `K4`, `K8`, `K16`, `K32`, `K64`, `K128`                    |
| L4 文档     | `doc_id`      | 单个文档的唯一标识（内部归一化） | Page 侧由原始图片路径解析 source doc id；Query 侧保留 `doc_name` 归一化值 |
| L5 页面     | `page_idx`    | 文档内的页码（0-based）          | Page 侧由原始图片路径解析 source page idx                  |

**全文统一命名规则：**

- 在代码、配置、文件名中，涉及任务族的标识符**一律使用小写短名**：`docqa`（而非 `DocumentQA` 或 `documentQA`）
- 子任务名保留原始命名中的小写形式：`longdocurl`, `mmlongdoc`, `slidevqa` 等
- 长度档位统一写为 `K{数字}` 格式：`K4`, `K8`, ..., `K128`
- `page_id` 的稳定格式为：`{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}`
  - 例：`docqa/longdocurl_K4/abc123/p0`
- `query_id` 的稳定格式为：`{task_family}/{subtask}_{length}/{query_index}`
  - 例：`docqa/longdocurl_K4/q001`
- 当前实现中，`Page` 侧的 `doc_id/page_idx` 不再直接取自样本级 `doc_name + enumerate(page_list)`，而是先从原始图片路径解析 source doc / source page，再填入上述 `page_id` 字符串格式
- `Query` 除 `doc_id` 外还显式携带 `candidate_page_ids`，主协议优先使用该字段决定 sample-specific candidate scope；仅在其为空时才回退到 `query.doc_id` 对应的文档内候选页

**命名分层对后续设计的意义：**

1. 配置中的 `subsets` 字段使用 L1 任务族名（如 `["docqa"]`），长度档位和子任务过滤通过额外参数控制
2. page_id 自带层级前缀，天然隔离不同任务族/子任务/档位之间的命名冲突
3. 评测结果可按 `task_family` / `subtask` / `length` 任一粒度分组汇总，无需事后解析 ID 字符串

**当前实现中的标识语义约束**（v8 同步）：

> 需要区分 Page 侧与 Query 侧两套语义：Page 侧的稳定 `doc_id/page_idx` 从原始图片路径解析，用于构造全局稳定 `page_id`；Query 侧的 `doc_id` 仍保留为样本级 `doc_name` 归一化值，主要用于追踪与回退逻辑。禁止再用 sample 内 `enumerate(page_list)` 直接构造稳定页面身份。

---

## 二、环境配置指导（Linux + Conda + uv）

> **2026-05-20 同步说明**：当前主评测与 full run 均在 Ubuntu + 2x RTX 3090 上完成。统一先在项目根目录执行 `source ./scripts/command/env.sh`，随后优先通过 `python main.py ...` 调用 Step 3 / Phase 4 / command 子命令；若需直连脚本，则使用 `scripts/run/` 与 `scripts/command/` 下的路径。

### 2.1 概览：分层管理

| 层             | 工具                | 职责                            |
| -------------- | ------------------- | ------------------------------- |
| Python 运行时  | **Conda**           | 安装 Python 3.10，创建隔离环境  |
| 非 Python 依赖 | **无（纯 Python）** | pypdfium2 零外部依赖，CUDA 驱动 |
| Python 包      | **uv**              | 所有 PyPI 包安装、锁定、更新    |
| 项目元数据     | **pyproject.toml**  | uv 原生格式，声明依赖与构建配置 |

> **为什么不用 Conda 装 Python 包？**  
> PyTorch 官方在 PyPI 上提供了 CUDA 12.4 的预编译 wheel，uv 直接下载，无需本地编译。驱动 595.79 支持最高 CUDA 13.2，向下兼容。

---

### 2.2 NVIDIA 驱动与 CUDA

**环境要求**：

1. **NVIDIA 驱动程序**：>= 546.x
   - 验证：`nvidia-smi` 应显示 CUDA Version: 12.x 或更高

2. **无需单独安装 CUDA Toolkit**  
   PyTorch wheel 自带 CUDA runtime（12.4），无需系统级 CUDA Toolkit。当前驱动 595.79（支持最高 CUDA 13.2）完全兼容。

3. **验证驱动与 CUDA 支持**：
   ```bash
   nvidia-smi
   # 应看到 Driver Version >= 546, CUDA Version: 12.x 或 13.x
   ```

---

### 2.3 Conda 安装与 Python 环境

1. **安装 Miniconda**（推荐，轻量）：
   - 下载 Linux 64-bit：https://docs.conda.io/en/latest/miniconda.html

2. **仅创建 Python 环境（不装任何 Python 包）**：

   ```bash
   # 创建空白环境（仅 Python 3.10）
   conda create -n zeroshotvdr python=3.10 -y

   # 激活环境
   conda activate zeroshotvdr
   ```

3. **创建并激活项目虚拟环境**：

   ```bash
   # 在项目根目录执行，uv 会创建 .venv
   uv sync

   # 日常进入项目时，先激活 conda，再激活项目 .venv
   conda activate zeroshotvdr
   source .venv/bin/activate
   ```

   或直接使用项目提供的统一入口：

   ```bash
   source scripts/command/env.sh
   ```

   **为什么这样就能最小步骤使用？**
   - `uv sync` 会安装第三方依赖，并将当前仓库以 editable install 的形式安装到 `.venv`。
   - editable install 的作用是让 `src/` 下的项目代码可直接被当前虚拟环境导入。
   - 本项目的兼容性补丁位于仓库根目录的 `sitecustomize.py`。当你在**项目根目录**启动 `.venv` 里的 `python` 时，Python 会通过标准启动流程自动导入它。
   - 约束条件也很明确：请在项目根目录中启动 `python`，并优先使用 `.venv/bin/python` 或已激活的项目 `.venv`。

4. **验证**：
   ```bash
   python --version   # Python 3.10.x
   which python       # 应指向 ZeroShotVDR/.venv/bin/python
   ```

---

### 2.4 uv 安装与 pyproject.toml 配置

#### 安装 uv

```bash
# 在 conda 环境中
pip install uv
```

#### 安装依赖

```bash
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
- ✅ **跨平台一致**：Linux / macOS 行为完全一致
- ✅ **高性能**：基于 PDFium（Chromium 的 PDF 引擎），渲染质量高

**使用示例**：

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("document.pdf")
page = pdf[0]                    # 获取第一页（索引从 0 开始）
bitmap = page.render(scale=2.0)  # scale=2.0 即 2x 分辨率（约 144 DPI）
pil_image = bitmap.to_pil()      # 转为 PIL Image（可直接送入 ColPali）
```

---

### 2.6 模型权重与数据集下载

这一节建议改成“项目内本地缓存 + 显式下载”。

原方案里有两个容易卡住的点：

- 直接把 `from_pretrained(..., device_map='cuda')` 当作“预下载脚本”，会在下载前就尝试占用 GPU，不利于排错。
- 对 MMLongBench 使用 `load_dataset(...)` 不是官方推荐下载方式；该仓库主体是多个大型 `tar.gz` 文件，直接 `streaming=True` 后再 `list(ds)` 也会把流式读取优势抵消掉。

#### 统一放到项目目录下

以下环境变量会把 Hugging Face 的模型与数据缓存都放到当前项目内：

```bash
mkdir -p .cache/huggingface data/MMLongBench/raw

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"

# 网络不稳时再启用镜像
export HF_ENDPOINT="https://hf-mirror.com"
```

#### ColPali-v1.3 模型权重

需要注意，`vidore/colpali-v1.3` 不是完整模型本体，而是依赖 `vidore/colpaligemma-3b-pt-448-base` 的 LoRA adapter。

- `vidore/colpali-v1.3`：约 114 MB
- `vidore/colpaligemma-3b-pt-448-base`：约 5.87 GB

建议先把这两个仓库都下载到项目内缓存，而不是一上来直接跑 GPU 加载：

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vidore/colpaligemma-3b-pt-448-base', cache_dir='.cache/huggingface/hub', resume_download=True); snapshot_download(repo_id='vidore/colpali-v1.3', cache_dir='.cache/huggingface/hub', resume_download=True); print('ColPali base + adapter downloaded into project-local cache.')"
```

下载完成后，模型文件会位于项目目录下的 `.cache/huggingface/`，不再落到用户目录。

若只想做最小化联通性验证，可先下载小文件而不是完整权重：

```bash
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

1. 先做 smoke test，只下载元数据包与 DocumentQA 图像包。
2. 管线跑通后，再补齐全量图像包。

```bash
# 第一步：只下载最小可验证集合（聚焦 Phase 2 主评测集 DocumentQA）
uv run hf download ZhaoweiWang/MMLongBench 0_mmlb_data.tar.gz --local-dir data/MMLongBench/raw --repo-type dataset
uv run hf download ZhaoweiWang/MMLongBench 5_docqa_image.tar.gz --local-dir data/MMLongBench/raw --repo-type dataset

# 解压到项目内
tar -xzf data/MMLongBench/raw/0_mmlb_data.tar.gz -C data/MMLongBench/raw
tar -xzf data/MMLongBench/raw/5_docqa_image.tar.gz -C data/MMLongBench/raw
```

全量下载时，再补齐其余任务包：

```bash
for file in \
    '2_vh_image.tar.gz' \
    '2_mm-niah_image.tar.gz' \
    '3_icl_image.tar.gz' \
    '4_summ_image.tar.gz' \
    '5_docqa_image.tar.gz'; do
    uv run hf download ZhaoweiWang/MMLongBench "$file" --local-dir data/MMLongBench/raw --repo-type dataset
done
```

如果只是检查数据组织或字段，不要把整个流式数据集转成列表。可以只读取少量样本：

```bash
uv run python -c "from datasets import load_dataset; ds = load_dataset('ZhaoweiWang/MMLongBench', split='test', streaming=True); first = next(iter(ds)); print(first.keys())"
```

**数据集目录结构（预期）**：

```
data/MMLongBench/
├── raw/
│   ├── 0_mmlb_data.tar.gz          # 元数据（Phase 1 必选）
│   ├── 5_docqa_image.tar.gz        # DocumentQA 图像（Phase 2 baseline 必选）
│   ├── 1_vrag_image.tar.gz         # 扩展任务资源（后续可选下载）
│   ├── ...
│   ├── mmlb_data/
│   └── mmlb_image/
└── processed/
```

---

### 2.7 环境验证与运行脚本

项目当前脚本已按职责重组：

- `scripts/run/`：实验运行脚本
- `scripts/command/`：环境、检查、清理与进度命令
- `main.py`：统一入口，优先调度项目 `.venv` Python

对日常使用者而言，**推荐优先使用 `main.py`**；只有在调试脚本实现本身时，才直接调用 `scripts/run/` 或 `scripts/command/` 下的文件。

#### `scripts/command/env.sh` —— 环境激活入口

```bash
source scripts/command/env.sh
```

当前脚本会定位项目根目录并激活项目 `.venv`；这是所有 full run、smoke run 与环境检查前的统一入口。

#### `main.py` —— 统一命令入口

用法：

```bash
python main.py <group> <command> [args...]
```

当前支持的三组命令为：

| 分组      | 常用命令                    | 作用 |
| --------- | --------------------------- | ---- |
| `step3`   | `eval`, `analysis`          | Step 3 评测与分析 |
| `phase4`  | `eval`, `full`             | Phase 4 单次评测与批量运行 |
| `command` | `check-env`, `step3-clean`, `phase4-progress`, `trace-analyze`, `backfill-step3-phase4-schema`, `test-model-load` | 辅助命令 |

示例：

```bash
python main.py command check-env
python main.py step3 eval --run-name step3_docqa_full_dual3090_stable_page_ids --index-dir data/processed/index_stable_page_ids
python main.py phase4 eval --run-name smoke_fixed64 --method fixed_topn --coarse-top-n 64 --max-queries 50 --valid-only
python main.py command phase4-progress --watch
```

若需要把 `--help` 原样转发给子脚本，可使用：

```bash
python main.py phase4 eval -- --help
```

#### 环境与模型验证脚本

| 脚本 | 作用 | 推荐调用方式 |
| ---- | ---- | ------------ |
| `scripts/command/check_env.py` | 检查 Python 版本、关键依赖、CUDA、HF 缓存路径 | `python main.py command check-env` |
| `scripts/test_model_load.py` | 验证 ColPali-v1.3 可在 GPU 上完成图像与文本前向 | `python main.py command test-model-load` |

其中 `check_env.py` 以 offline 模式运行，不连接 HuggingFace Hub，仅验证本地环境是否就绪。

#### `scripts/run/` —— 核心评测脚本

| 脚本 | 作用 | 当前说明 |
| ---- | ---- | -------- |
| `scripts/run/run_step3_eval.py` | Step 3.1 页级检索评测 | 支持 `--stats-only`、smoke、full run、断点恢复与稳定 index 路径 |
| `scripts/run/run_step3_analysis.py` | Step 3.2 结果分析 | 输出趋势、bad case 与方向决策辅助产物 |
| `scripts/run/run_phase4_eval.py` | Phase 4 单次评测入口 | 支持 `fixed_topn`、`adaptive`、`adaptive_neighbors`、valid-only、trace、cache |
| `scripts/run/run_phase4_full.sh` | Phase 4 批量全量运行脚本 | 顺序执行主消融全量任务 |

推荐示例：

```bash
source scripts/command/env.sh
python main.py step3 analysis --run-dir outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids
python main.py phase4 eval --run-name phase4_adaptive_neighbors_cache_full_20260520 --method adaptive_neighbors --neighbor-window 1 --neighbor-seed-n 8 --valid-only --trace-enabled --use-mean-pool-cache true --mean-pool-cache-dir outputs/cache/mean_pool_full_20260520_rerun --index-dir data/processed/index_stable_page_ids
```

#### `scripts/command/` 与辅助分析脚本

| 脚本 | 作用 |
| ---- | ---- |
| `scripts/command/run_step3_clean.py` | 预览或清理 Step 3 结果目录与索引页 |
| `scripts/command/check_phase4_progress.sh` | 轮询 Phase 4 长任务进度 |
| `scripts/analyze_phase4_trace.py` | 汇总 `phase4_trace.jsonl`，生成 `slice_metrics.csv` 与桶统计 |
| `scripts/backfill_step3_phase4_schema.py` | 为阶段三 stable run 回填 Phase 4 兼容 schema 产物 |

示例：

```bash
python main.py command step3-clean --run-names smoke_eval_longdocurl_K4_q5 --clean-outputs --clean-index --dry-run
python main.py command trace-analyze --trace outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/phase4_trace.jsonl --out outputs/eval_reports/phase4_adaptive_neighbors_cache_full_20260520/slice_metrics.csv
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
- [ ] 进入项目时执行 `source scripts/command/env.sh`（或手动 `conda activate zeroshotvdr && source .venv/bin/activate`）
- [ ] 将 ColPali base + adapter 下载到项目内 `.cache/huggingface/`
- [ ] **明确下载 DocumentQA 所需资源**：`0_mmlb_data.tar.gz`（元数据）+ `5_docqa_image.tar.gz`（DocumentQA 页面图像），解压到 `data/MMLongBench/raw/`
- [ ] 验证项目内模型缓存与数据目录可读取
- [ ] 运行 `python main.py command check-env`，全部项通过

> **v2 对齐说明**：Phase 2 的主闭环依赖 DocumentQA 子集，因此 Phase 1 的前置资源准备明确围绕 DocumentQA 展开（下载 `5_docqa_image.tar.gz` 而非泛化的"一个任务图像包"）。其余任务族（icl/niah/summ/vrag）的图像包可在 Phase 2 管线跑通后按需补充下载。

#### Step 1.3 数据集探索

- [ ] 统计 MMLongBench 数据规模：重点确认 DocumentQA 子集的文档数、页面数、查询数
- [ ] 理解 DocumentQA 标注格式：`page_list`（页面图像路径列表）、`ans_page_list`（答案所在页面列表）和 `answer` 字段如何映射为页级检索 ground truth
- [ ] 抽样查看 DocumentQA 页面图像（版式类型：纯文/表格/图表/混合）
- [ ] 确认各 length 档位（K4/K8/K16/K32/K64/K128）的页面规模差异
- [ ] 确认训练集 / 测试集划分（MMLongBench 是否仅提供测试集）

#### Phase 1 产出

- [ ] 可运行环境（验证脚本全 PASS）
- [ ] DocumentQA 数据集统计笔记：各子任务（longdocurl/mmlongdoc/slidevqa）× 各档位（K4-K128）的文档数、页面数、查询数
- [ ] 确认 `0_mmlb_data.tar.gz` 和 `5_docqa_image.tar.gz` 已解压至项目内，数据路径可正常读取

---

### Phase 2：基础系统实现（5.13 – 5.22）

**目标**：实现完整的 ColPali-based 页级检索管线，以 DocumentQA 子集为主评测集跑通端到端流程。优先打通"数据集适配 → 语料构建 → 索引 → 检索 → 评测"闭环，PDF 渲染作为辅助数据源后补。

> ⚠️ **开发前置须知**：
>
> 1. 所有同时使用 `datasets` 与 `torch` 的模块，必须在文件顶部
>    `import datasets`（或 `import pandas`）**先于** `import torch`，否则会触发
>    pyarrow 与 torch 的 C++ 原生库加载顺序冲突导致进程崩溃。详见 5.2.1 节。
> 2. 建议将数据读取层与模型层分属不同模块文件，从物理上规避导入顺序问题：
>    `data/` 层依赖 `datasets`，`indexing/` 和 `retrieval/` 层依赖 `torch`，
>    跨层调用通过函数传参而非顶层 import 混合。

#### Step 2.1 数据接入与语料构建（成员 A） —— 3 天

**文件**：`src/zeroshot_vdr/data/corpus.py`, `src/zeroshot_vdr/data/adapters.py`, `src/zeroshot_vdr/contracts.py`

| 子步骤 | 内容                                                                                                                                                                                                                                                                                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1.1  | 定义数据契约 `contracts.py`：`Page`（page_id, doc_id, raw_doc_name, task_family, subtask, length, page_idx, image_path）、`Query`（query_id, text, doc_id, raw_doc_name, task_family, subtask, length, candidate_page_ids）、`RetrievalResult`（query_id, page_id, score, rank）、`RelevanceJudgment`（query_id, page_id, relevance）以及 ID 构造辅助函数 `normalize_doc_id()`, `build_page_id()`, `build_query_id()`, `build_page_id_from_image()` |
| 2.1.2  | 实现 `DocumentQAAdapter`：从 MMLongBench DocumentQA 子集读取 `doc_name`、`page_list` 与 `ans_page_list`；Query 侧保留 `normalize_doc_id(doc_name)` 作为样本级文档标识，同时根据 `page_list` 构建 `candidate_page_ids`；Page 侧则通过原始图片路径恢复稳定 source doc / source page 后生成 `page_id` |
| 2.1.3  | 实现 `PageCorpus` 类：聚合所有适配器产出的页面，输出 `corpus_meta.json`；当前稳定页面身份仍沿用 `"{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}"` 字符串格式，但 `doc_id/page_idx` 的语义已切换为图片路径解析得到的 source doc / source page |
| 2.1.4  | 预留 `PDFAdapter`：基于 pypdfium2 的 PDF→图像 渲染路径，供后续补入非 DocumentQA 数据源                                                                                                                                                                                                                                                                                                            |

**预期 API**：

```python
# src/zeroshot_vdr/contracts.py
@dataclass
class Page:
    page_id: str
    doc_id: str
    raw_doc_name: str | None
    task_family: str
    subtask: str
    length: str
    page_idx: int
    image_path: str

@dataclass
class Query:
    query_id: str
    text: str
    doc_id: str
    raw_doc_name: str | None
    task_family: str
    subtask: str
    length: str
    candidate_page_ids: tuple[str, ...]

# 全链路唯一合法的 ID 构造入口
def normalize_doc_id(raw_doc_name: str) -> str: ...
def build_page_id(task_family, subtask, length, doc_id, page_idx) -> str: ...
def build_query_id(task_family, subtask, length, query_index) -> str: ...
def build_page_id_from_image(task_family, subtask, length, image_rel_path,
                             fallback_page_idx=None) -> str: ...

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

| 子步骤 | 内容                                                                                                                                                                                                                                |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.2.1  | 加载 ColPali-v1.3 模型（`colpali_engine.models.ColPali`）                                                                                                                                                                           |
| 2.2.2  | 实现 `PageEncoder`：逐页编码图像 → patch embeddings `[n_patches, dim]`                                                                                                                                                              |
| 2.2.3  | 处理显存友好的 batching：每次加载 N 张图像（建议 batch_size=2~4 for 8GB VRAM）                                                                                                                                                      |
| 2.2.4  | **索引存储**：每页独立存储为 `{safe_page_id}.pt`（shape `[n_patches, dim]`，其中 `safe_page_id` 由 `page_id` 中的 `/` 替换为 `_` 得到），而非单一巨型张量。此设计支持：（a）真增量追加，（b）后续 patch pruning 后各页 patch 数不同 |
| 2.2.5  | `IndexStore` 类统一管理索引的构建、加载、增量更新，产出 `page_ids.json`（有序 JSON 数组 `[page_id, ...]`，记录全局页面顺序）和 `index_meta.json`（由 `save_meta()` 显式写入）                                                       |
| 2.2.6  | 记录索引构建耗时与存储大小；支持断点续建（`encode_corpus(resume=True)` 默认跳过已索引页面，重复调用安全且幂等）                                                                                                                     |

**预期 API**：

```python
# src/zeroshot_vdr/indexing/encoder.py
class PageEncoder:
    """PageEncoder 的生产构造入口为 from_pretrained()，直接构造主要用于测试。"""
    def __init__(self, model, processor,
                 batch_size: int = 4,
                 dtype: torch.dtype | None = None,   # None=不转换模型精度
                 device: str = "cuda:0",
                 storage_dtype: torch.dtype = torch.float16): ...
    @classmethod
    def from_pretrained(cls, model_repo="vidore/colpali-v1.3",
                        base_repo="vidore/colpaligemma-3b-pt-448-base",
                        device="cuda:0", dtype=None, batch_size=4,
                        storage_dtype=torch.float16) -> "PageEncoder": ...
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor: ...
    def encode_corpus(self, pages: list[Page], store: "IndexStore",
                      show_progress: bool = True,
                      resume: bool = True) -> None: ...  # resume=True 支持断点续建

# src/zeroshot_vdr/indexing/store.py
class IndexStore:
    def __init__(self, index_dir: str): ...
    # 核心接口
    def write_page(self, page_id: str, embedding: torch.Tensor) -> None: ...
    def write_batch(self, page_ids: list[str], embeddings: torch.Tensor) -> None: ...
    def read_page(self, page_id: str) -> torch.Tensor: ...
    def iter_pages(self, page_ids: list[str] | None = None): ...
    def list_page_ids(self, doc_id: str | None = None) -> list[str]: ...
    def get_mean_pooled_view(self, page_ids: list[str] | None = None
                             ) -> tuple[torch.Tensor, list[str]]: ...
    # 便利函数（仅 patch 数一致时可用）
    def read_stacked(self, page_ids: list[str]) -> tuple[torch.Tensor, list[str]]: ...
    # 元信息（需显式调用；write_page 不会自动创建 index_meta.json）
    def save_meta(self, model_name: str, dim: int) -> None: ...
    def load_meta(self) -> dict: ...
    @property
    def stats(self) -> dict: ...
```

#### Step 2.3 查询编码与检索（成员 B） —— 4 天

**文件**：`src/zeroshot_vdr/retrieval/encoder.py`, `src/zeroshot_vdr/retrieval/scoring.py`, `src/zeroshot_vdr/retrieval/pipeline.py`

| 子步骤 | 内容                                                                                                                                                                                                                                                                                                                                        |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.3.1  | 使用 ColPali 的文本编码器对查询进行编码（query tokens → embeddings `[n_tokens, dim]`）。当前实现中的 `QueryEncoder` 需要 `model + processor`，并额外提供 `from_pretrained()` / `from_page_encoder()` / `encode_batch()` 作为便利构造与批量接口。                                                                                            |
| 2.3.2  | 实现 **MaxSim 相似度**模块：对查询中每个 token，找到页面 patch 中最高相似度，求和。当前默认 `Sim` 为 **L2 归一化后的点积**（等价于余弦相似度），独立为 `scoring.py`，便于后续替换或扩展打分函数。                                                                                                                                           |
| 2.3.3  | **显存优化**：逐页或分批计算，避免构建 `[n_queries, n_pages, n_tokens, n_patches]` 全相似度矩阵                                                                                                                                                                                                                                             |
| 2.3.4  | 实现 `RetrievalPipeline`：编排"查询编码 → 候选召回 → 精排打分 → Top-k 结果组装"四个环节。**当前 baseline 默认行为**：若 `Query` 自带 `candidate_page_ids`，候选召回优先直接返回该 sample-specific 候选集；仅当 `candidate_page_ids` 为空且 `candidate_ids is None` 时，才回退到与 `query.doc_id` 对应的文档内页面集合。若显式传入 `candidate_ids=[]`，则直接返回空结果。 |
| 2.3.5  | Top-k 排序（k = 1, 3, 5, 10），返回 `[RetrievalResult(query_id, page_id, score, rank)]`                                                                                                                                                                                                                                                     |
| 2.3.6  | 记录查询耗时。当前实现会在 `retrieve()` 内统计**单次查询耗时**并输出 debug 日志；并未在检索层维护公共的“平均延迟”累积状态，后续若需平均值，应在评测/实验层聚合。                                                                                                                                                                            |

**MaxSim 公式**：

```
Score(Q, P) = Σ_i max_j Sim(q_i, p_j)

其中：
- Q = {q_1, ..., q_m} 为查询的 m 个 token embeddings
- P = {p_1, ..., p_n} 为页面的 n 个 patch embeddings
- 当前实现默认使用 **L2 归一化后的点积**（等价于余弦相似度）；同时保留 `norm` 参数以允许关闭归一化
```

**预期 API**：

```python
# src/zeroshot_vdr/retrieval/scoring.py
def maxsim_score(query_emb: torch.Tensor,
                 page_emb: torch.Tensor,
                 norm: bool = True) -> torch.Tensor:
    """返回标量 score"""
    ...

def batched_maxsim(query_emb: torch.Tensor,
                   pages_emb: torch.Tensor,
                   norm: bool = True) -> torch.Tensor:
    """返回 [batch_size] scores"""
    ...

def batched_maxsim_variable(query_emb: torch.Tensor,
                            pages_list: list[torch.Tensor],
                            norm: bool = True) -> torch.Tensor:
    """变长 patch 回退路径：逐页返回 scores"""
    ...

# src/zeroshot_vdr/retrieval/pipeline.py
class RetrievalPipeline:
    def __init__(self, model, index_store: IndexStore,
                 processor=None,
                 query_encoder: QueryEncoder | None = None,
                 config: dict | None = None): ...
    def encode_query(self, query_text: str) -> torch.Tensor: ...
    def retrieve(self, query: Query, top_k: int = 10,
                 candidate_ids: list[str] | None = None,
                 score_batch_size: int | None = None) -> list[RetrievalResult]: ...
    def retrieve_text(self, text: str, candidate_ids: list[str],
                      top_k: int = 10) -> list[RetrievalResult]: ...  # 便利包装
    def retrieve_batch(self, queries: list[Query], top_k: int = 10,
                       **kwargs
                       ) -> list[list[RetrievalResult]]: ...
    def generate_candidates(self, query: Query,
                            query_emb: torch.Tensor | None = None,
                            top_n: int | None = None) -> list[str]: ...
    def score_candidates(self, query_emb: torch.Tensor,
                         candidate_ids: list[str],
                         batch_size: int | None = None
                         ) -> tuple[torch.Tensor, list[str]]: ...
```

#### Step 2.4 评测系统（成员 B） —— 2 天

**文件**：`src/zeroshot_vdr/evaluation/metrics.py`, `src/zeroshot_vdr/evaluation/ground_truth.py`

| 子步骤 | 内容                                                                                                                                                                                                                         |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.4.1  | `ground_truth.py`：当前实现通过 `DocumentQAAdapter.build_ground_truth()` 加载 DocumentQA 标注，并转为统一格式 `{query_id: set[page_id]}`。过滤入口为 `subtasks` / `lengths` / `task_family`，适配逻辑与指标计算分离          |
| 2.4.2  | `metrics.py`：实现 4 项指标计算——**Recall@k**、**Precision@k**、**MRR**、**nDCG@k**。原子指标函数接受 `(retrieved_page_ids, relevant_page_ids, k)` 的标准化输入；批量聚合函数当前接受 `{query_id: [retrieved_page_id, ...]}` |
| 2.4.3  | 编写批量评测脚本：遍历测试查询 → 检索 → 对比 ground truth                                                                                                                                                                    |
| 2.4.4  | 输出结果：CSV 汇总表 + JSON 详细结果                                                                                                                                                                                         |

**指标定义**：

| 指标        | 公式 / 说明                                            |
| ----------- | ------------------------------------------------------ |
| Recall@k    | (检索到的相关页面数) / (总相关页面数)                  |
| Precision@k | (检索到的相关页面数) / k                               |
| MRR         | Mean Reciprocal Rank：相关页面首次出现的倒数排名的均值 |
| nDCG@k      | Normalized Discounted Cumulative Gain at k             |

> **v7 修订说明**：本节已根据 `src/zeroshot_vdr/evaluation/metrics.py` 与
> `src/zeroshot_vdr/evaluation/ground_truth.py` 的真实实现同步更新。主要差异：
> （1）`compute_all_metrics()` 当前接受 `dict[str, list[str]]`，而非
> `dict[str, list[RetrievalResult]]`；（2）`GroundTruthLoader.load()` 当前使用
> `subtasks` / `lengths` / `task_family` 过滤，而非 `subset` 单参数；
> （3）当前实现未在 `GroundTruthLoader` 中暴露 `build_page_id()` /
> `build_query_id()` 静态方法，而是直接复用 `DocumentQAAdapter` 的 ground-truth
> 构建逻辑；（4）当前实现额外提供 `load_by_subtask()`、`load_by_length()` 与
> `compute_metrics_by_group()` 便利接口。

**当前实现 API**：

```python
# src/zeroshot_vdr/evaluation/metrics.py
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def mrr(retrieved: list[str], relevant: set[str]) -> float: ...
def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...

def compute_all_metrics(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> pd.DataFrame: ...
# 返回列当前为 ['k', 'Recall', 'Precision', 'MRR', 'nDCG', 'n_queries']

def compute_metrics_by_group(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    group_fn,
    k_values: list[int] | None = None,
) -> pd.DataFrame: ...

# src/zeroshot_vdr/evaluation/ground_truth.py
class GroundTruthLoader:
    def __init__(self, config: dict | None = None): ...
    def load(
        self,
        subtasks: list[str] | None = None,
        lengths: list[str] | None = None,
        task_family: str = "docqa",
    ) -> dict[str, set[str]]: ...
    def load_by_subtask(
        self,
        subtask: str,
        lengths: list[str] | None = None,
    ) -> dict[str, set[str]]: ...
    def load_by_length(
        self,
        length: str,
        subtasks: list[str] | None = None,
    ) -> dict[str, set[str]]: ...
    @property
    def config(self) -> dict: ...
    # 返回 {query_id: {relevant_page_id, ...}}
```

#### Phase 2 产出

- [ ] `src/zeroshot_vdr/contracts.py`：数据契约定义
- [ ] `src/zeroshot_vdr/data/`：DocumentQA 适配器 + 语料构建可用
- [ ] `src/zeroshot_vdr/indexing/`：可离线构建 & 加载索引，每页独立存储
- [ ] `src/zeroshot_vdr/retrieval/`：流水线式检索，查询→Top-k 结果端到端
- [ ] `src/zeroshot_vdr/evaluation/`：四项指标计算，与数据集解耦
- [ ] `config/default.yaml`：全局配置（数据/模型/索引/检索/评测参数全覆盖）
- [ ] `scripts/` 下环境激活、验证与评测脚本可用

---

### Phase 3：基础评测与调优（5.23 – 5.25）

**目标**：完成基础系统的全面评测，分析瓶颈与失败案例。

> **评测协议**（详见 4.5.5）：当前 stable baseline 采用**query-scoped page retrieval**，
> 即每个查询优先只在其 `candidate_page_ids` 对应的 sample-specific 页面集合内排序；
> 仅在该字段缺失时才回退到 `query.doc_id` 对应的文档内页面集合。主汇报口径不再以 K32 为主，
> 而是以全部长度档位的完整披露 + 14,385 条有效标注查询的主比较口径并行呈现。

#### Step 3.1 全量评测

- [x] 在 DocumentQA 三个子任务（longdocurl/mmlongdoc/slidevqa）× 所有长度档位（K4-K128）上运行 stable baseline 全量评测
- [x] 输出 k=1,3,5,10 下的 Recall, Precision, MRR, nDCG（分子任务、分档位）
- [x] 记录效率指标：索引构建时间、索引文件大小、单次检索延迟
- [x] 建立“完整披露口径 + 有效标注主比较口径”双轨汇报方式

**当前实现状态（2026-05-16 同步）**：

- Step 3.1 已有可执行脚本：`scripts/run/run_step3_eval.py`
- Linux 主平台统一先执行 `source ./scripts/command/env.sh`，随后优先通过 `python main.py step3 eval ...` 调用评测入口
- Step 3 清理入口已补齐：`scripts/command/run_step3_clean.py`
- 当前 stable 协议已从“仅按 `query.doc_id` 默认候选”升级为“优先按 `Query.candidate_page_ids` 决定样本级候选范围”
- 支持 `--stats-only`、`--max-queries`、`--query-offset`、`--skip-index-build`，适合先做局部验证
- 清理脚本支持 `--run-names`、`--subtasks`、`--lengths` 与 `--all`，默认 dry-run 预览，适合在重跑前清理旧结果与局部索引
- 当前输出会按 run_name 写入 `outputs/eval_reports/{run_name}/`，其中 `metrics_summary.csv` 为汇总表，`run_summary.json` 记录范围统计与性能指标
- 当前稳定版 full run 目录为 `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/`
- 历史目录 `outputs/eval_reports/step3_docqa_full_dual3090/` 仅可作为 provisional baseline 对照，不再作为主结论依据

**当前稳定运行实测（Ubuntu + 2x RTX 3090）**：

- 评测范围：15,577 queries / 3,653 docs / 87,922 pages
- 索引补建时间：6,689.0 s（约 1h51m29s）
- 检索时间：1,102.9 s（约 18m23s）
- 平均延迟：0.071 s/query
- P95 延迟：0.138 s/query
- 当前有效全量索引目录：`data/processed/index_stable_page_ids/`，体量约 88.6 GB

#### Step 3.2 结果分析

- [x] 绘制 k vs 指标曲线（Recall@k, Precision@k, nDCG@k）
- [x] 识别 Bad Cases：Recall@10 < 1.0 的查询，并单独剥离 `no_ground_truth`
- [x] 分类失败类型（如：miss_top10、multi_page_partial、neighbor-page confusion）
- [x] 输出分析目录：`outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/`
- [x] 当前分析结论：1,192 条查询缺失或带无效页级标注；14,385 条为有效标注查询；真实 bad case 2,636 条，bad-case rate = 18.32%

#### Step 3.3 方向决策

- [x] 基于失败分析，确定 Phase 4 主线为方向 B：查询自适应两阶段粗精检索
- [x] 保留方向 A 作为补充实验 / 未来工作
- [x] 完成方向决策文档：`docs/revision/step3_direction_revision_v1.md`
- [x] 完成里程碑报告：`docs/Milestone_Report_Phase3.md`

#### Phase 3 产出

- [x] `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/`（稳定语义 full run 结果）
- [x] `outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids/analysis/`（Step 3.2 分析结果）
- [x] `docs/revision/step3_direction_revision_v1.md`
- [x] `docs/Milestone_Report_Phase3.md`

---

### Phase 4：进阶方法研究与实现（5.26 – 6.2）

**目标**：在不破坏 Phase 3 stable baseline 的前提下，实现 query-adaptive two-stage coarse-to-fine retrieval，并完成 valid-only 全量消融与结果分析。

> **2026-05-20 同步说明**：Phase 4 主线已经完成。`src/zeroshot_vdr/advanced/` 已落地，`scripts/run/run_phase4_eval.py` 与 `main.py` 已提供统一实验入口，valid-only 全量实验矩阵与 cache 版本均已跑通；当前推荐方法为 `adaptive_neighbors + mean-pool cache`。

#### 补充方向 A：查询感知的自适应索引压缩（暂缓）

方向 A 在当前迭代中未进入正式实现，主要原因是 hardest slices 仍强依赖细粒度 patch 证据，激进压缩更容易直接伤害质量。该方向保留为后续补充实验或未来工作。

#### 主线方向 B：查询自适应的两阶段粗精检索（已完成）

| Stage | 实际落地内容 | 状态 |
| ---- | ------------ | ---- |
| Stage 0 | 复现 Phase 3 baseline，固定 valid-only 主比较口径 | [x] |
| Stage 1 | 新增 Phase 4 配置与 `advanced/` 目录 | [x] |
| Stage 2 | 实现 page_id 解析与邻页工具函数 | [x] |
| Stage 3 | 实现 adaptive top-N 选择逻辑 | [x] |
| Stage 4 | 实现 fixed top-N `TwoStageRetriever` 最小版本 | [x] |
| Stage 5 | 接入 `MeanPoolCache`，并修复 full cache 构建 OOM 问题 | [x] |
| Stage 6 | 将 adaptive top-N 接入运行时 | [x] |
| Stage 7 | 将 neighbor expansion 接入运行时 | [x] |
| Stage 8 | 落地 `scripts/run/run_phase4_eval.py` 评测入口 | [x] |
| Stage 9 | 落地 per-query trace、`slice_metrics.csv` 与 `bucket_metrics.csv` | [x] |
| Stage 10 | 完成 full valid-only 消融与里程碑报告 | [x] |

#### 当前落地位置

- 两阶段检索主实现：`src/zeroshot_vdr/advanced/two_stage.py`
- 邻页扩展与 page_id 解析：`src/zeroshot_vdr/advanced/neighbors.py`
- Mean-pool cache：`src/zeroshot_vdr/advanced/mean_pool_cache.py`
- Trace / profiling：`src/zeroshot_vdr/advanced/profiling.py`
- Phase 4 评测入口：`scripts/run/run_phase4_eval.py`
- Trace 汇总脚本：`scripts/analyze_phase4_trace.py`
- 统一入口：`main.py`
- 单元测试：`tests/phase4/test_neighbors.py`、`tests/phase4/test_adaptive.py`、`tests/phase4/test_two_stage.py`、`tests/phase4/test_mean_pool_cache.py`

#### 主结果汇总（14,385 valid-only queries）

| Method | Recall@10 | nDCG@10 | Avg Latency (s) | P95 Latency (s) | Avg Rerank Candidates | 备注 |
| ------ | --------: | ------: | --------------: | --------------: | --------------------: | ---- |
| Phase 3 Full MaxSim | 0.8517 | 0.6325 | 0.0716 | 0.1384 | 32.7 | valid-only baseline |
| Fixed Top-32 | 0.8482 | 0.6308 | 0.0794 | 0.1434 | 19.0 | 更快候选数，但有小幅掉点 |
| Fixed Top-64 | 0.8513 | 0.6325 | 0.0889 | 0.1800 | 26.8 | 质量接近 baseline |
| Fixed Top-128 | 0.8517 | 0.6326 | 0.0907 | 0.1989 | 32.2 | 基本回到 baseline 质量，但时延更差 |
| Adaptive | 0.8482 | 0.6308 | 0.0790 | 0.1417 | 19.0 | 自适应候选规模与 Top-32 接近 |
| Adaptive + Neighbor | 0.8523 | 0.6325 | 0.0796 | 0.1437 | 19.8 | 无 cache 情况下最佳总体 Recall |
| Fixed Top-64 + Cache | 0.8513 | 0.6325 | 0.0650 | 0.1178 | 26.8 | 与无 cache 同质，明显更快 |
| Adaptive + Cache | 0.8482 | 0.6308 | 0.0592 | 0.0847 | 19.0 | 与无 cache 同质，粗筛开销大幅下降 |
| Adaptive + Neighbor + Cache | 0.8523 | 0.6325 | 0.0600 | 0.0858 | 19.8 | 当前推荐方法 |

#### 当前结论

- Phase 4 满足既定硬性质量标准：推荐方法的 Recall@10 没有低于 Phase 3 baseline 0.005 以内阈值，nDCG@10 与 baseline 基本持平。
- 推荐方法为 `adaptive_neighbors + mean-pool cache`：它在总体上略高于 Phase 3 baseline 的 Recall@10，同时把平均延迟从约 0.0716 s/query 降到约 0.0600 s/query。
- 改善最明显的切片集中在 `slidevqa/K64` 与 `slidevqa/K128`；`longdocurl/K128` 和 `mmlongdoc/K128` 上存在轻微质量回落，但换来了显著的时延与 rerank 候选规模下降。
- Full `MeanPoolCache` 已可稳定构建并复用：当前 87,090 页的 cache 目录为 `outputs/cache/mean_pool_full_20260520_rerun`，体量约 26 MB，而原 patch index 约 87 GB。

#### Phase 4 产出

- [x] `src/zeroshot_vdr/advanced/` 目录下改进代码
- [x] `scripts/run/run_phase4_eval.py`、`scripts/analyze_phase4_trace.py` 与 `main.py`
- [x] valid-only 全量主表、slice 表、bucket 表与 per-query trace
- [x] cache 与 no-cache 对比实验
- [x] `docs/Milestone_Report_Phase4.md`

---

### Phase 5：报告撰写与答辩准备（6.3 – 6.9）

**目标**：基于已经完成的 Phase 3 / Phase 4 结果，收口最终论文、答辩材料与可提交代码包。

> **2026-05-20 同步说明**：Phase 4 的代码、实验矩阵和里程碑报告已经齐备，Phase 5 的重点已从“补实验”切换为“整理主表、提炼结论、撰写正式报告和答辩材料”。

#### Step 5.1 实验报告（NeurIPS 模板，英文，正文 8-9 页）

| 章节         | 负责   | 内容要点                           |
| ------------ | ------ | ---------------------------------- |
| Introduction | 共同   | 任务背景、长文档页级检索难点、本项目贡献 |
| Related Work | 共同   | ColPali、ColBERT、VLM 文档检索综述 |
| Method       | 成员 A | Stable baseline、two-stage retrieval、mean-pool cache、neighbor expansion |
| Experiments  | 成员 B | Phase 3 baseline、Phase 4 主表、cache 对比、slice 结果 |
| Analysis     | 成员 B | slidevqa 长上下文收益、K128 trade-off、局限性与未来工作 |
| Conclusion   | 共同   | 质量-效率结论与后续方向 |

#### Step 5.2 答辩 PPT

- 方法设计思路（图示化）
- 实验亮点（一页摘要对比表）
- 创新性阐述（与已有工作的差异化）

#### Step 5.3 代码整理

- [x] 完成脚本目录重组（`scripts/run/`、`scripts/command/`）
- [x] 新增 `main.py` 统一入口
- [ ] 统一剩余 docstring、type hints 与帮助文本
- [ ] 编写或补齐最终 `README.md` 的快速开始与复现实验指引
- [ ] 复核 `uv sync` / `.venv` 环境的一键复现说明
- [ ] 最终清理调试产物并打包提交

#### Phase 5 产出

- [x] `docs/Milestone_Report_Phase4.md`
- [ ] 实验报告 PDF
- [ ] 答辩 PPT
- [ ] 最终代码包
- [ ] README.md

---

## 四、核心模块接口设计

> **修订汇总说明**：本章综合吸收 `docs/revision/core_module_revision_v1.md` 至 `docs/revision/core_module_revision_v5.md` 的逐轮修订结果。
> 核心变化：（1）预处理层从"PDF 渲染"转向"语料构建 + 数据适配"；
> （2）索引存储从单一巨型张量改为每页独立文件，支持变长 patch 数；
> （3）检索层从单一 Retriever 类改为分环节流水线；
> （4）评测层将指标计算与数据集适配解耦；
> （5）新增显式的数据契约层，统一页面/查询/结果标识体系；
> （6）根据 Step 2.2 实际实现同步更新索引层接口（PageEncoder processor 参数、from_pretrained 工厂方法、encode_corpus resume 参数、page_ids.json 数组格式、save_meta 显式触发、get_mean_pooled_view 返回 tuple、write_batch 接口）；
> （7）根据 stable redesign 同步更新页面身份与评测协议说明（`candidate_page_ids`、source-image-derived page_id、有效标注主比较口径）。

---

### 4.0 数据契约（`contracts.py`）

所有模块间传递的核心对象统一使用以下 dataclass，确保 page_id / query_id 在索引、检索、评测全链路中一致。

> **v8 同步说明**：
>
> 1. `source_subset` 字段已不足以承载文档内检索、跨子任务评测和多档位对比的需求，
>    替换为 `task_family` + `subtask` + `length` 的显式分层字段。
> 2. `Query` 仍保留 `doc_id`，但当前主协议优先使用 `candidate_page_ids` 决定 sample-specific 候选范围；
>    `doc_id` 主要用于追踪和在 `candidate_page_ids` 缺失时的回退逻辑（详见 4.5.5）。
> 3. 新增 `raw_doc_name` 字段保留原始文档来源标识，便于调试和数据核对。
> 4. `RetrievalResult` 新增 `query_id` 字段，使单条结果可独立追溯其来源查询。
> 5. `Page.page_id` 的字符串格式未变，但其 `doc_id/page_idx` 语义已切换为从原始图片路径解析得到的 source doc / source page；
>    `contracts.py` 当前新增 `normalize_image_rel_path()`、`extract_source_doc_id()`、`extract_source_page_idx()`、`build_page_id_from_image()` 等 helper。

**DocumentQA 的标识语义（当前实现）**：

- Page 侧：从 `page_list` 中的原始图片路径解析 `source_doc_id` 与 `source_page_idx`，再构造稳定 `page_id`
- Query 侧：保留 `normalize_doc_id(doc_name)` 作为样本级文档标识，并显式携带 `candidate_page_ids`
- 因此，当前实现中“Page 的 `doc_id`”与“Query 的 `doc_id`”不再要求语义完全相同；真正的检索候选范围由 `candidate_page_ids` 决定

```python
"""
数据契约：定义系统中跨模块传递的核心数据结构。
所有 page_id / query_id 均为稳定字符串，贯穿索引→检索→评测全链路。
"""

from dataclasses import dataclass, field


@dataclass
class Page:
    """页面语料中的单页。"""
    page_id: str           # 稳定唯一标识，格式: {task_family}/{subtask}_{length}/{doc_id}/p{page_idx}
    doc_id: str            # 由原始图片路径解析出的稳定 source doc id
    raw_doc_name: str | None  # 原始数据中的文档来源字段（doc_name），用于调试和回溯
    task_family: str       # L1 任务族（docqa / icl / niah / summ / vrag）
    subtask: str           # L2 子任务（longdocurl / mmlongdoc / slidevqa / ...）
    length: str            # L3 长度档位（K4 / K8 / K16 / K32 / K64 / K128）
    page_idx: int          # 由原始图片路径解析出的 source page idx（0-based）
    image_path: str        # 页面图像文件路径


@dataclass
class Query:
    """单条检索查询。

    当前 baseline 主协议优先使用 candidate_page_ids 做 sample-specific 候选约束；
    doc_id 仍保留用于追踪和缺省回退逻辑。纯文本字符串查询仅作为便利接口存在，
    不作为 baseline 主协议。
    """
    query_id: str          # 稳定唯一标识，格式: {task_family}/{subtask}_{length}/{query_index}
    text: str              # 查询文本
    doc_id: str            # 样本级文档标识（回退逻辑 / 调试用）
    raw_doc_name: str | None  # 原始数据中的文档来源字段
    task_family: str       # L1 任务族
    subtask: str           # L2 子任务
    length: str            # L3 长度档位
    candidate_page_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RetrievalResult:
    """单条检索命中结果。"""
    query_id: str          # 来源查询的稳定 ID（便于结果追溯）
    page_id: str           # 命中页面的稳定 ID
    score: float           # 相似度分数
    rank: int              # 排名（1-based）


@dataclass
class RelevanceJudgment:
    """单条标注：某查询对某页面的相关性。"""
    query_id: str
    page_id: str
    relevance: int         # 0/1 或分级相关度


# ---- ID 构造辅助函数（v3 新增） ----

def normalize_doc_id(raw_doc_name: str) -> str:
    """将原始 doc_name 归一化为内部稳定 doc_id。

    规则由实现阶段根据实际数据特征确定（如去空格、统一大小写、
    移除特殊字符等）。所有 Adapter 和 Corpus 必须通过此函数获得 doc_id，
    禁止各自临时构造。
    """
    ...


def build_page_id(
    task_family: str,
    subtask: str,
    length: str,
    doc_id: str,
    page_idx: int,
) -> str:
    """构造稳定的 page_id。

    格式：{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}
    这是全链路唯一合法的 page_id 构造入口。
    """
    ...


def build_query_id(
    task_family: str,
    subtask: str,
    length: str,
    query_index: int,
) -> str:
    """构造稳定的 query_id。

    格式：{task_family}/{subtask}_{length}/q{query_index:0>3d}
    这是全链路唯一合法的 query_id 构造入口。
    """
    ...


def normalize_image_rel_path(image_rel_path: str) -> str: ...


def extract_source_doc_id(image_rel_path: str) -> str: ...


def extract_source_page_idx(
    image_rel_path: str,
    fallback_page_idx: int | None = None,
) -> int: ...


def build_page_id_from_image(
    task_family: str,
    subtask: str,
    length: str,
    image_rel_path: str,
    fallback_page_idx: int | None = None,
) -> str: ...
```

**ID 构造约束**（v8 同步）：

`build_page_id()` / `build_query_id()` 仍是最终字符串格式的合法构造入口；但对 DocumentQA 而言，Page 侧必须先通过原始图片路径恢复 source doc / source page，再调用 `build_page_id()` 或 `build_page_id_from_image()`。`DocumentQAAdapter`、`GroundTruthLoader` 和结果落盘模块不得再以 sample 内页序直接拼接稳定页面身份。

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
        Query 侧保留 normalize_doc_id(doc_name) 作为样本级 doc_id；
        Page 侧则通过原始图片路径恢复稳定 source doc / source page 后构造 page_id。
        输出：Page（基于图片路径构造稳定 page_id）、
            Query（携带 doc_id 与 candidate_page_ids）、RelevanceJudgment。
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

> **v5 修订说明**：本节已根据 Step 2.2 的实际实现代码（`IndexStore` / `PageEncoder`）
> 同步更新。主要变更：（1）PageEncoder 构造函数新增必填 `processor` 参数、独立
> `storage_dtype` 落盘精度、`from_pretrained()` 工厂方法；
> （2）`encode_corpus()` 新增 `resume` 断点续建参数；
> （3）`page_ids.json` 格式确认为有序 JSON 数组而非映射字典；
> （4）`index_meta.json` 由显式 `save_meta()` 创建，非隐式触发；
> （5）新增 `write_batch`、`save_meta`、`load_meta` 接口；
> （6）批量读取接口返回类型统一为 `tuple[Tensor, list[str]]`。

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

    processor 与 model 分离是有意设计，便于独立替换和测试。
    生产场景下推荐使用 from_pretrained() 工厂方法构造，
    内部封装了 sitecustomize 补丁验证和模型/处理器的加载。

    Parameters
    ----------
    model : ColPali
        ColPali 模型实例
    processor : ColPaliProcessor
        ColPali 图像处理器（与 model 匹配）
    batch_size : int
        GPU 编码批次大小（适配 8GB 显存建议 2~4）
    dtype : torch.dtype | None
        推理精度；None 表示不额外转换模型精度
    device : str
        推理设备
    storage_dtype : torch.dtype
        落盘精度（float16 约节省 50% 存储），独立于推理 dtype
    """

    def __init__(self, model, processor,
                 batch_size: int = 4,
                 dtype: torch.dtype | None = None,
                 device: str = "cuda:0",
                 storage_dtype: torch.dtype = torch.float16): ...

    @classmethod
    def from_pretrained(
        cls,
        model_repo: str = "vidore/colpali-v1.3",
        base_repo: str = "vidore/colpaligemma-3b-pt-448-base",
        device: str = "cuda:0",
        dtype: torch.dtype | None = None,
        batch_size: int = 4,
        storage_dtype: torch.dtype = torch.float16,
    ) -> "PageEncoder":
        """
        生产场景下的推荐构造入口。
        内部封装 sitecustomize 补丁验证和模型/处理器的加载。
        直接构造 PageEncoder(model, processor, ...) 主要用于测试和离线场景。
        """
        ...

    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """编码单张页面图像 → [n_patches, dim]"""
        ...

    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """编码一批图像 → [batch, n_patches, dim]"""
        ...

    def encode_corpus(self, pages: list[Page],
                      store: "IndexStore",
                      show_progress: bool = True,
                      resume: bool = True) -> None:
        """
        遍历页面语料，逐批编码并写入索引存储。

        resume=True 为默认行为，跳过 store 中已存在的 page_id，
        支持断点续建。重复调用 encode_corpus 是安全的，不会重复写入。
        """
        ...


class IndexStore:
    """
    索引持久化存储。

    存储布局：
        {index_dir}/
        ├── pages/
        │   └── {safe_page_id}.pt    # safe_page_id = page_id 中 '/' 替换为 '_'
        ├── page_ids.json             # 有序 JSON 数组 [page_id, ...]，记录全局页面顺序
        └── index_meta.json           # 模型名、维度、时间戳、页数（由 save_meta() 显式创建）

    IndexStore 的稳定读取语义为按页读取与按页面列表迭代读取。
    任何需要返回全量 stacked tensor 的接口均视为 baseline 便利函数，
    而非变长 patch 场景下的通用接口。

    批量读取接口（read_stacked、get_mean_pooled_view）统一返回
    ``tuple[Tensor, list[str]]``，保证张量行与页面 ID 顺序可追溯。
    """

    def __init__(self, index_dir: str): ...

    # -- 核心接口：写入 --
    def write_page(self, page_id: str, embedding: torch.Tensor) -> None:
        """写入单页 embedding → pages/{safe_page_id}.pt。"""
        ...

    def write_batch(self, page_ids: list[str],
                    embeddings: torch.Tensor) -> None:
        """
        批量写入 [batch, n_patches, dim] 的 embeddings，
        等价于逐页调用 write_page()。encode_corpus() 内部使用此路径。
        """
        ...

    # -- 核心接口：读取 --
    def read_page(self, page_id: str) -> torch.Tensor:
        """读取单页 embedding → [n_patches, dim]。"""
        ...

    def iter_pages(self, page_ids: list[str] | None = None):
        """按页迭代读取，兼容变长 patch 场景。

        Parameters
        ----------
        page_ids : list[str] | None
            要读取的页面列表；None 表示全部页面。

        Yields
        ------
        tuple[str, torch.Tensor] : (page_id, embedding [n_patches, dim])
        """
        ...

    def list_page_ids(self, doc_id: str | None = None) -> list[str]:
        """列出索引中的页面 ID。

        Parameters
        ----------
        doc_id : str | None
            限定文档；None 表示全部页面。
        """
        ...

    # -- 稳定视图接口 --
    def get_mean_pooled_view(self, page_ids: list[str] | None = None
                             ) -> tuple[torch.Tensor, list[str]]:
        """
        返回 (pooled_tensor [n_pages, dim], page_ids)。
        pooled_tensor[i] 是 page_ids[i] 对应页面的 patch embeddings 的均值向量。
        """
        ...

    # -- baseline 便利函数（非核心抽象，变长 patch 场景不可用） --
    def read_stacked(self, page_ids: list[str]) -> tuple[torch.Tensor, list[str]]:
        """返回 (stacked_tensor [n_pages, n_patches, dim], page_id_list)。

        仅在所有页面 patch 数一致时使用的便利函数。
        变长 patch 场景请使用 iter_pages()。
        """
        ...

    # -- 元信息（显式写入，非自动触发） --
    def save_meta(self, model_name: str, dim: int) -> None:
        """保存索引元信息到 index_meta.json。

        .. note::
           write_page() 不会自动创建 index_meta.json。
           完整的编码流程应以 store.save_meta(model_name, dim) 收尾。
        """
        ...

    def load_meta(self) -> dict:
        """加载索引元信息；若 index_meta.json 不存在则返回默认值。"""
        ...

    @property
    def stats(self) -> dict:
        """
        返回 {num_pages, dim, total_size_mb, storage_dir, created_at}。
        num_pages 来自实时加载 page_ids.json，不依赖 index_meta.json 快照。
        """
        ...
```

---

### 4.3 检索层（`retrieval/`）

**设计原则**：检索不是单一步骤，而是"查询编码 → 候选召回 → 精排打分 → 结果组装"的流水线。当前 stable baseline 中，候选召回优先采用 Query 显式携带的 sample-specific `candidate_page_ids`；若该字段为空，才回退为文档内全量页面。流水线结构已为 Phase 4 两阶段检索预留扩展点。

```python
"""
检索层：查询编码 → 候选召回 → MaxSim 精排 → Top-k 结果组装。
"""

import torch
from zeroshot_vdr.contracts import Page, RetrievalResult
from zeroshot_vdr.indexing.store import IndexStore


class QueryEncoder:
    """ColPali 查询编码器。"""

    def __init__(self, model, processor, device: str = "cuda:0"): ...
    @classmethod
    def from_pretrained(cls, model_repo: str = "vidore/colpali-v1.3",
                        base_repo: str = "vidore/colpaligemma-3b-pt-448-base",
                        device: str = "cuda:0",
                        dtype: torch.dtype | None = None) -> "QueryEncoder": ...
    @classmethod
    def from_page_encoder(cls, page_encoder) -> "QueryEncoder": ...
    def encode(self, query: str) -> torch.Tensor:
        """文本查询 → [n_tokens, dim]"""
        ...
    def encode_batch(self, queries: list[str]) -> torch.Tensor:
        """批量文本查询 → [batch, n_tokens, dim]"""
        ...


# -- 打分函数（scoring.py） --

def maxsim_score(query_emb: torch.Tensor,
                 page_emb: torch.Tensor,
                 norm: bool = True) -> torch.Tensor:
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
                   pages_emb: torch.Tensor,
                   norm: bool = True) -> torch.Tensor:
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

def batched_maxsim_variable(query_emb: torch.Tensor,
                            pages_list: list[torch.Tensor],
                            norm: bool = True) -> torch.Tensor:
    """
    变长 patch 页面列表的回退打分路径。

    Parameters
    ----------
    query_emb : [n_tokens, dim]
    pages_list : list[[n_patches_i, dim]]

    Returns
    -------
    torch.Tensor : [len(pages_list)] scores
    """
    ...


# -- 检索流水线（pipeline.py） --

class RetrievalPipeline:
    """
    检索流水线编排器。

    流程：
    1. encode_query(query.text)  → query embedding
    2. generate_candidates(query) → 候选 page_id 列表
    3. score_candidates()         → 对候选逐批 MaxSim 打分
    4. assemble_results(top_k)    → 排序、截断、封装为 RetrievalResult 列表

    Baseline 模式下，retrieve() 接收 Query 对象而不是纯文本字符串。
    若 Query 自带 candidate_page_ids，则 generate_candidates() 优先直接返回该集合；
    仅当 candidate_page_ids 为空且 candidate_ids is None 时，才回退到
    query.doc_id 对应的全部页面。若调用方显式传入 candidate_ids=[]，
    retrieve() 将直接返回空结果，而不会回退到默认候选生成。
    仅在显式启用 global retrieval 实验配置时，候选范围才允许扩展为全局语料。

    Phase 4 中可通过替换 generate_candidates() 策略接入两阶段检索。
    """

    def __init__(self, model, index_store: IndexStore,
                 processor=None,
                 query_encoder: QueryEncoder | None = None,
                 config: dict | None = None): ...

    def encode_query(self, query_text: str) -> torch.Tensor: ...

    def retrieve(self, query: Query,
                 top_k: int = 10,
                 candidate_ids: list[str] | None = None,
                 score_batch_size: int | None = None) -> list[RetrievalResult]:
        """
        检索 Top-k 相关页面（baseline 主协议）。

        Parameters
        ----------
        query : Query
            检索查询对象，当前可携带 candidate_page_ids。
        top_k : int
        candidate_ids : list[str] | None
            候选页面列表；为 None 时优先使用 query.candidate_page_ids，
            否则回退到 query.doc_id 对应的文档内页面集合。
            若显式传入空列表，则视为“空候选集”，返回空结果。
        score_batch_size : int | None
            逐批计算 MaxSim 的页面 batch 大小；None 表示使用配置默认值。

        Returns
        -------
        list[RetrievalResult]
        """
        ...

    def retrieve_text(self, text: str,
                      candidate_ids: list[str],
                      top_k: int = 10) -> list[RetrievalResult]:
        """
        纯文本查询的便利包装接口。

        此接口不承担 baseline 默认协议。调用方必须显式提供 candidate_ids。
        内部构造临时 Query 对象（当前实现中 `query_id="adhoc/q000"`，
        其余文档级元信息留空）后委托给 retrieve()。
        """
        ...

    def retrieve_batch(self, queries: list[Query],
                       top_k: int = 10,
                       **kwargs) -> list[list[RetrievalResult]]: ...

    def generate_candidates(self, query: Query,
                            query_emb: torch.Tensor | None = None,
                            top_n: int | None = None) -> list[str]:
        """
        候选召回阶段。

        Baseline：优先返回 Query 自带的 `candidate_page_ids`；若为空，则回退到 query.doc_id 对应的文档内页面。
        Phase 4：可替换为均值池化粗筛。
        仅在显式启用全局检索配置时，候选范围才允许跳出文档内约束。
        """
        ...

    def score_candidates(self, query_emb: torch.Tensor,
                         candidate_ids: list[str],
                         batch_size: int | None = None
                         ) -> tuple[torch.Tensor, list[str]]:
        """对候选集逐批 MaxSim 打分，返回 `(scores, scored_page_ids)`。"""
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
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    批量计算全部指标。

    Parameters
    ----------
    retrieval_results : {query_id: [retrieved_page_id, ...]}
    ground_truth : {query_id: {relevant_page_id, ...}}
    k_values : list[int] | None

    Returns
    -------
    pd.DataFrame : columns=['k', 'Recall', 'Precision', 'MRR', 'nDCG', 'n_queries']
    """
    ...

def compute_metrics_by_group(
    retrieval_results: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    group_fn,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """按自定义分组维度（如子任务/长度档位）计算指标。"""
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

    def __init__(self, config: dict | None = None): ...

    def load(
        self,
        subtasks: list[str] | None = None,
        lengths: list[str] | None = None,
        task_family: str = "docqa",
    ) -> dict[str, set[str]]:
        """
        加载 ground truth。

        Parameters
        ----------
        subtasks : list[str] | None
            限定子任务列表；None 时使用配置中的 `data.subtasks`。
        lengths : list[str] | None
            限定长度档位；None 时使用配置中的 `data.length`，若仍为空则回退到
            `['K4', 'K8', 'K16', 'K32', 'K64', 'K128']`。
        task_family : str
            当前实现固定使用 `"docqa"`。

        Returns
        -------
        dict[str, set[str]] : {query_id: {relevant_page_id, ...}}
        """
        ...

    def load_by_subtask(
        self,
        subtask: str,
        lengths: list[str] | None = None,
    ) -> dict[str, set[str]]:
        """加载单个子任务的 ground truth。"""
        ...

    def load_by_length(
        self,
        length: str,
        subtasks: list[str] | None = None,
    ) -> dict[str, set[str]]:
        """加载单个长度档位的 ground truth。"""
        ...

    @property
    def config(self) -> dict:
        """返回当前使用的配置字典（只读）。"""
        ...
```

---

### 4.5 设计决策说明

以下说明为 v1 修订中几个关键架构决策的原理，供后续实现和 Phase 4 扩展时参考。

#### 4.5.1 为什么需要稳定 page_id 契约？

| 问题         | 原方案                                                             | 修订后                                                                            |
| ------------ | ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| 页面标识方式 | 临时拼接 `{doc_id}_{page_idx}`                                     | `Page.page_id` 为规范格式 `{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}` |
| ID 一致性    | Retriever 返回 `doc_id, page_idx`，Evaluator 期望 `page_id` 字符串 | 全链路使用同一 `page_id` 字符串                                                   |
| 多子集扩展   | 不同子集的 doc_id 可能冲突                                         | `page_id` 含 task_family + subtask + length 前缀，天然隔离                        |

**核心原则**：所有模块间传递的页面标识和查询标识必须使用稳定、可追溯的字符串，禁止在各模块间临时拼接或拆解。

**ID 契约的适用边界**（v2 补充）：

上述 page_id 格式的设计意图不限于当前 DocumentQA baseline，而是覆盖以下场景：

| 场景                                                     | page_id 是否仍然成立 | 说明                                                              |
| -------------------------------------------------------- | -------------------- | ----------------------------------------------------------------- |
| 不同长度档位（K4/K8/.../K128）                           | ✅ 成立              | 长度包含在 page_id 中，同一文档的不同档位版本天然区分             |
| DocumentQA 内不同子任务（longdocurl/mmlongdoc/slidevqa） | ✅ 成立              | subtask 字段隔离，避免同名 doc_id 冲突                            |
| 扩展到其他任务族（icl/niah/summ/vrag）                   | ✅ 成立              | task_family 字段隔离，且各任务族的数据结构差异由对应 Adapter 封装 |
| 未来加入 PDF 渲染的额外文档                              | ✅ 成立              | 可使用专用 task_family（如 `pdf`）或归入现有族                    |

**query_id 采用相同的层级前缀规则**（`{task_family}/{subtask}_{length}/{query_index}`），确保即使不同子任务中出现相同的查询序号也不会冲突。

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

- **Baseline**：`generate_candidates` 优先返回当前 Query 的 `candidate_page_ids`；仅在其为空时，才回退到 `query.doc_id` 对应文档内的全部 page_ids。
- **两阶段检索（Phase 4B）**：只需替换 `generate_candidates` 为均值池化粗筛，其余环节复用。
- **索引压缩（Phase 4A）**：只需在 `score_candidates` 中处理变长 patch 的张量。

#### 4.5.4 模块边界与导入顺序约束

为避免 `datasets`（→ pyarrow）与 `torch` 的 C++ 原生库加载冲突，模块在设计上遵循以下边界：

| 层            | 主要依赖                       | 避免顶层导入                                              |
| ------------- | ------------------------------ | --------------------------------------------------------- |
| `data/`       | `datasets`, `PIL`, `pypdfium2` | `torch`                                                   |
| `indexing/`   | `torch`, `PIL`, `transformers` | `datasets`                                                |
| `retrieval/`  | `torch`, `transformers`        | `datasets`                                                |
| `evaluation/` | `pandas`（→ pyarrow）, `numpy` | `torch`（指标函数接受 Python list/set，避免 tensor 操作） |

若某模块必须同时使用二者，应在模块顶部**先 `import datasets` / `import pandas`，再 `import torch`**。

#### 4.5.5 评测协议：检索范围与结果汇报（v2 新增）

DocumentQA 的数据形式是"单个查询对应单个长文档内的页面集合"，这引出一个关键设计决定：检索的候选范围。

**当前 stable baseline 的检索范围**：

| 选项                          | 描述                               | 本项目选择                |
| ----------------------------- | ---------------------------------- | ------------------------- |
| sample-specific 候选检索      | 每个查询仅在其 `candidate_page_ids` 对应页面内排序 | ✅ **当前 stable baseline 采用** |
| 文档内回退检索（per-document） | 当 `candidate_page_ids` 缺失时，回退到 `query.doc_id` 对应页面集合 | 回退逻辑 |
| 全局语料检索（global corpus） | 所有文档的所有页面混合为一个语料库 | 留作 Phase 4 可选扩展对比 |

选择当前协议的理由：

1. DocumentQA 的 `page_list` 天然给出了样本级候选页面范围，当前 stable redesign 已据此构建 `candidate_page_ids`。
2. 长上下文场景下，旧的 sample 内 `doc_id/page_idx` 语义并不稳定；只有回到原始图片路径恢复 source doc / source page，评测结论才可信。
3. 跨文档检索会引入额外的文档级区分问题，而当前数据集并不提供可直接比较的跨文档页级相关性标注。

**结果汇报粒度**：

| 维度                                        | 汇报方式                                                    |
| ------------------------------------------- | ----------------------------------------------------------- |
| 子任务（longdocurl / mmlongdoc / slidevqa） | **分别汇报**各项指标，同时提供总体汇总                      |
| 长度档位（K4/K8/K16/K32/K64/K128）          | 全档位完整披露，并输出跨档位趋势图                          |
| 主比较口径                                  | 仅统计 14,385 条有效页级标注查询                            |
| 完整披露口径                                | 保留全部 15,577 条查询结果，作为附录级披露                  |

**注**：Phase 3 完成后，主汇报口径已从“以 K32 为主档位”调整为“以有效标注查询子集为主比较口径”。原因是当前指标实现会把空 relevant set 记为完美召回 / 完美 MRR，若不先剥离 `no_ground_truth` 查询，主表会被系统性抬高。

**主评测子任务范围**（v3 固定）：

> Phase 2-3 的主评测范围固定为 DocumentQA 中的 `longdocurl`、`mmlongdoc`、`slidevqa` 三个子任务。`text_mmlongdoc` 不纳入主评测表，除非后续确认其图像页面与标注协议可与当前页级检索设置严格对齐。此约束直接影响数据加载、结果统计和报告主表。

**Baseline 检索协议的接口层落实**（v8 同步）：

> Baseline 模式下，`RetrievalPipeline.retrieve()` 接收 `Query` 对象（非纯文本字符串）。若 `Query.candidate_page_ids` 非空，`generate_candidates()` 会优先直接返回该集合；仅当该字段为空且 `candidate_ids is None` 时，默认候选范围才会回退到与 `query.doc_id` 对应的页面集合，而不是全局语料集合；若显式传入 `candidate_ids=[]`，当前实现会直接返回空结果。只有在显式启用全局检索实验配置时，系统才允许跳出当前样本/文档候选范围。`retrieve_text()` 仅作为便利包装接口存在，不承担 baseline 默认协议，调用方必须显式提供 `candidate_ids`。

#### 4.5.6 索引存储策略的阶段性权衡（v2 新增）

逐页独立存储方案在为增量追加和变长 patch 提供便利的同时，存在以下需要留意的权衡：

| 考量             | 说明                                                                             | 缓解措施                                                     |
| ---------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 小文件 I/O 开销  | DocumentQA K4/K8 档位页面数较少（~数十页），影响可忽略；K128 可能达数千页        | 批量读取时使用 `torch.load` 的并行加载；必要时可缓存合并视图 |
| 跨平台迁移       | 逐页文件便于 rsync 增量同步，反而优于巨型单文件                         | 无额外风险                                                   |
| 未来更大规模     | 若页面数超 50K（超出当前 MMLongBench 总规模），可转为分片存储（shard）           | 当前阶段不需要                                               |

**结论**：逐页独立存储是当前阶段（Phase 2-4）的优先策略。若后续实验显示 I/O 成为瓶颈（预期不会，因为 GPU 推理远慢于磁盘读取），可引入可选的 shard 合并模式，不影响现有接口。

#### 4.5.7 核心抽象与便利实现的区分（v2 新增）

当前设计中，部分接口描述偏向 baseline 便利性，部分则是必须长期成立的设计不变量。为降低 Phase 4 扩展时的理解成本，以下对此进行明确区分：

**必须长期成立的核心抽象**（Phase 4 不应破坏）：

| 抽象                                                     | 理由                                             |
| -------------------------------------------------------- | ------------------------------------------------ |
| `Page` / `Query` / `RetrievalResult` 的 dataclass 契约   | 跨模块数据交换的"语言"，修改会影响全链路         |
| `BaseAdapter` → `PageCorpus` 的语料构建流程              | 新增数据源只需增加 Adapter，不应改动 Corpus 逻辑 |
| `IndexStore.write_page / read_page` 的按页读写语义       | 索引的原子操作接口，上层逻辑依赖其稳定性         |
| `RetrievalPipeline` 的四阶段流水线结构                   | Phase 4 通过替换阶段策略扩展，流水线本身不变     |
| 指标函数的纯函数签名 `(retrieved, relevant, k) -> float` | 与数据集解耦的评测基础                           |

**当前为 baseline 提供的便利实现**（Phase 4 可替换或增强）：

| 便利实现                                                    | Phase 4 的可能变化                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `IndexStore.read_stacked()` 返回全量 stacked tensor         | 变长 patch 时不可用，需改用 `iter_pages()`；不影响核心抽象                                                |
| `RetrievalPipeline.generate_candidates()` 默认 query-scoped 候选，缺失时回退到文档内候选 | Phase 4B 替换为均值池化粗筛；全局检索需显式配置                                                           |
| `scoring.batched_maxsim()` 假设同 batch 内各页 patch 数相同 | 当前实现已提供 `batched_maxsim_variable()` 作为逐页回退路径；Phase 4A 后也可继续演进为 padding / 分桶策略 |
| `RetrievalPipeline.retrieve_text()` 纯文本便利包装          | 不承担 baseline 默认协议；调用方需显式提供 candidate_ids                                                  |

这一区分的关键作用：当 Phase 4 需要改动某处时，可以先判断该处属于"核心抽象"还是"便利实现"，避免在核心抽象上做破坏性修改。

#### 4.5.8 配置层语义分层（v2 新增）

当前 `config/default.yaml` 中的配置项混合了不同语义层面的信息。为提高实验复现和切换的便利性，建议在实现时按以下层次组织配置：

| 配置层           | 语义                                     | 变更频率         | 示例                                               |
| ---------------- | ---------------------------------------- | ---------------- | -------------------------------------------------- |
| **项目级常量**   | 不随实验变化的固定信息                   | 几乎不变         | HF 缓存路径、模型 repo 名、page_id 模板            |
| **环境复现参数** | 保证他人可复现运行的关键设置             | 极少变化         | device、dtype、target_size                         |
| **实验切换参数** | 不同实验间需要对比的变量                 | 频繁变化         | subsets、length 档位、top_k 值、candidate_strategy |
| **策略选择参数** | 当前阶段的设计选择，后续可能成为实验变量 | Phase 切换时变化 | per_page_files、score_batch_size                   |

**建议实践**：在 `config.py` 中为上述四层分别提供加载函数，使得运行脚本可以通过命令行参数或环境变量覆盖"实验切换参数"和"策略选择参数"，而无需修改 YAML 文件本身。项目级常量和环境复现参数应保持稳定。

---

## 五、关键风险与注意事项

### 5.1 显存限制与批次大小选择

> **8 GB 显存环境下需注意批次大小控制。**

**缓解措施**：

| 场景        | 策略                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 图像编码    | `batch_size=4`（或更小），使用 `torch.float16`                                                                                                   |
| 索引加载    | 按页加载或按批加载控制内存占用，不依赖 `mmap_mode`。若后续引入分片化的 `npy`、`safetensors` 或其他支持内存映射的存储格式，再单独评估 memmap 方案 |
| MaxSim 计算 | 分批计算（page batch），避免构建完整相似度矩阵                                                                                                   |
| 进阶方法    | 候选方向 A（压缩索引）天然减低显存需求                                                                                                           |

**显存预算估算**（ColPali-v1.3）：

- 模型参数：~2-3 GB（bfloat16）
- 单张图像编码：~1 GB（672x672 输入，含中间激活）
- MaxSim 计算：可控（逐批释放）

### 5.2 依赖库兼容性

| 组件           | 兼容性 | 说明                                    |
| -------------- | ------ | --------------------------------------- |
| PyTorch        | OK     | 官方预编译 wheel                        |
| transformers   | OK     | 纯 Python                               |
| colpali-engine | OK     | 纯 Python，依赖 transformers            |
| pypdfium2      | OK     | 纯 Python，零系统依赖                   |
| uv             | OK     | 跨平台支持                              |
| datasets       | 需注意 | 见下方 pyarrow 加载顺序约束             |

#### 5.2.1 pyarrow 与 torch 的 C++ 原生库加载顺序约束

**现象**：在已导入 `torch` 的进程中 `import datasets`，Python 进程可能崩溃，无 Python traceback。

**根因**：`datasets` → `pandas` → `pyarrow` 的 C++ 原生扩展与 `torch` 的 CUDA 运行时库存在符号冲突。若 torch 先初始化，pyarrow 在加载其 C 扩展时可能触发内存访问违例。

**解决方案**：

```python
# 正确顺序：先导入 datasets/pandas/pyarrow，再导入 torch
import datasets   # 或 from datasets import ...
import torch

# 错误顺序：先导入 torch 再导入 datasets 可能导致崩溃
import torch
import datasets   # ← 可能崩溃!
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

```bash
# 在当前终端设置镜像环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$PWD/.cache/huggingface/datasets"

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

- [x] Conda 环境 `zeroshotvdr` 创建成功
- [x] `uv sync` 无报错完成
- [x] `python main.py command check-env` 全部 PASS
- [x] ColPali-v1.3 权重下载至本地
- [x] DocumentQA 数据（`0_mmlb_data.tar.gz` + `5_docqa_image.tar.gz`）已下载解压，统计信息已记录

### Milestone 2：基础管线跑通（5.22）

- [x] `src/zeroshot_vdr/data/`：DocumentQA 适配器 + 语料构建可用
- [x] `src/zeroshot_vdr/indexing/`：可离线构建 & 加载索引（逐页独立存储）
- [x] `src/zeroshot_vdr/retrieval/`：流水线式检索，查询→Top-k 结果端到端
- [x] `src/zeroshot_vdr/evaluation/`：四项指标计算可用，与数据集解耦
- [x] 端到端管线在 DocumentQA 10 条查询子集上输出合理结果

### Milestone 3：基础评测完成 + Milestone 报告（5.28）

- [x] DocumentQA 三子任务 × 全部长度档位的 stable baseline 全量评测结果
- [x] 跨档位趋势图（K4-K128）和跨子任务对比分析
- [x] Bad Cases 分析完成，并剥离 `no_ground_truth` 样本
- [x] 改进方向已确定
- [x] `docs/Milestone_Report_Phase3.md` 已提交

### Milestone 4：进阶方法完成（6.2）

- [x] 改进方法代码实现完毕
- [x] Baseline vs. 改进方法对比实验完成
- [x] 消融实验完成
- [x] 效率对比数据齐全
- [x] `docs/Milestone_Report_Phase4.md` 已提交

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
