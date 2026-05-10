# MMLongBench 数据集探索笔记

> **项目**：ZeroShotVDR  
> **日期**：2026-05-08（初稿），2026-05-10（Step 2.1 实现后更新）  
> **数据来源**：[ZhaoweiWang/MMLongBench](https://huggingface.co/datasets/ZhaoweiWang/MMLongBench)（NeurIPS 2025 Spotlight）  
> **论文参考**：`docs/paper/MMLongBench.pdf`

---

## 一、数据集概览

MMLongBench 是首个面向长上下文视觉语言模型（LCVLM）的综合性评测基准，覆盖 **5 大类任务**、**13,331 条样本**。每条样本在 **5 种标准化输入长度**（8K–128K tokens）上均提供对应版本，通过跨模态 tokenization 方案（视觉 patch + 文本 token）统一度量上下文长度。

| 属性     | 值                                                  |
| -------- | --------------------------------------------------- |
| 总样本数 | 13,331                                              |
| 任务类别 | 5 类（VRAG / NIAH / ICL / Summ / DocumentQA）       |
| 长度档位 | K4, K8, K16, K32, K64, K128（对应 ~4K–128K tokens） |
| 图像类型 | 自然图像、合成图像、文档页面截图、PDF 渲染页        |
| 数据划分 | 仅 **test** 集（无 train/val 划分）；评测基准       |

---

## 二、任务类别与子任务详情

### 2.1 各任务类别总览

| 类别              | 英文名                                | 子任务数 | 核心能力考查                    |
| ----------------- | ------------------------------------- | -------- | ------------------------------- |
| **Visual RAG**    | Visual Retrieval-Augmented Generation | 2        | 基于图像实体的知识检索与问答    |
| **NIAH**          | Needle-In-A-Haystack                  | 9        | 长上下文中定位特定视觉/文本信息 |
| **Many-Shot ICL** | In-Context Learning                   | 4        | 多样本上下文学习（图像分类）    |
| **Summarization** | Long-Document Summarization           | 2        | 长文档页面级摘要                |
| **DocumentQA**    | Long-Document VQA                     | 3        | 长文档页面级问答                |

---

### 2.2 Visual RAG（视觉检索增强生成）

**数据格式**：JSONL，每条样本包含查询图像、问题、答案及大量干扰上下文 chunk。

**子任务**：

| 子任务   | 文件前缀   | K128 样本数 | 上下文 chunk 数 | 长度范围     |
| -------- | ---------- | ----------- | --------------- | ------------ |
| InfoSeek | `infoseek` | 3,384       | ~1,058 ctxs     | ~131K tokens |
| ViQuAE   | `viquae`   | 6,954       | ~1,078 ctxs     | ~131K tokens |

**样本字段**（infoseek 为例）：
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 样本唯一标识 |
| `question` | str | 文本查询问题 |
| `image` | str | 查询图像相对路径（如 `infoseek/oven_04953036.jpg`） |
| `answer` | list[str] | 答案列表（多答案标注） |
| `entity_id` | str | Wikidata 实体 ID |
| `entity_text` | str | 实体名称 |
| `positive_ctxs` | list[dict] | 正相关上下文 chunk（含 `doc_id`, `text`, `title`） |
| `ctxs` | list[dict] | 全部上下文 chunk（含干扰项） |
| `length` | int | 输入 token 总数 |

**对本项目的适用性**：✅ **高度相关**。VRAG 是 MMLongBench 中最接近本项目"视觉文档检索"定位的任务。ColPali 的 Late Interaction 机制天然适合此类"在大量文本 chunk 中基于图像查询找到相关页面/片段"的场景。`positive_ctxs` 提供了明确的相关性标注，可直接用于评测 Recall@k。

---

### 2.3 NIAH（大海捞针）

**数据格式**：JSONL，每条样本包含一个"针"（目标信息）嵌入大量干扰内容中，要求模型定位或推理。

**子任务**：

| 子任务                  | 文件                                 | K128 样本数 | 类型              | 上下文特征                   |
| ----------------------- | ------------------------------------ | ----------- | ----------------- | ---------------------------- |
| Counting-Image          | `counting-image_test`                | 1,767       | 图像计数          | ~147 张图像 + 文本上下文     |
| Counting-Text           | `counting-text_test`                 | 1,767       | 文本计数          | ~147 张图像 + 大量文本       |
| Reasoning-Image         | `reasoning-image_test`               | 3,474       | 图像推理          | ~149 张图像，含拼图/视觉推理 |
| Reasoning-Text          | `reasoning-text_test`                | 1,737       | 文本推理          | ~148 张图像 + 文本逻辑推理   |
| Retrieval-Image         | `retrieval-image_test`               | 3,600       | 图像检索          | ~152 张图像，找特定子图      |
| Retrieval-Text          | `retrieval-text_test`                | 3,600       | 文本检索          | ~152 张图像 + 文本中找信息   |
| Text-Haystack Retrieval | `text-haystack_retrieval-image_test` | 2,814       | 文本干草堆+图检索 | 大量文本 + 少量图像          |
| VH Multi                | `vh_multi_test_1000`                 | 3,000       | 多图像视觉干草堆  | ~365 张图像                  |
| VH Single               | `vh_single_test_1000`                | 6,000       | 单图像视觉干草堆  | ~366 张图像                  |

**样本字段**（以 counting-image 为例）：
| 字段 | 说明 |
|------|------|
| `id` | 样本 ID |
| `question` | 任务描述（含 `<image>` 占位符） |
| `answer` | 答案（列表） |
| `positive_ctxs` | 正相关上下文（含 `type`, `text`, `image`, `has_ans`, `nid`） |
| `ctxs` | 全部上下文 |
| `image_list` | 文档图像路径列表 |
| `needle_image_list` | "针"图像路径 |
| `category` | 子类别（如 `count-image`） |

**对本项目的适用性**：⚠️ **部分相关**。NIAH 的 retrieval-image/text 子任务与文档检索有一定关联，但任务设计更偏向于"从大量无关内容中定位特定信息"，而非标准文档检索的"给定查询找相关页面"。VH（Visual Haystack）子任务可能适合评测视觉检索能力。

---

### 2.4 Many-Shot ICL（多样本上下文学习）

**数据格式**：JSON（非 JSONL），每个文件包含 1 条"配置"，内含 exemplar 列表和 test 示例。

**子任务**：

| 子任务   | 数据集来源       | K128 测试样本数 | Exemplar 配置              | 类别数 |
| -------- | ---------------- | --------------- | -------------------------- | ------ |
| cars196  | Stanford Cars    | 458             | 4 个 exemplar 列表 x 10 类 | 196 类 |
| food101  | Food-101         | 500             | 4 个 exemplar 列表 x 10 类 | 101 类 |
| inat2021 | iNaturalist 2021 | 待确认          | 类似结构                   | 多类   |
| sun397   | SUN397           | 待确认          | 类似结构                   | 397 类 |

**样本字段**：
| 字段 | 说明 |
|------|------|
| `exemplar_list` | 4 组 exemplar 列表，每组含 10 张不同版本的每类图像（50 张/类 x 10 类 x 4 组 = ~2,000 张） |
| `all_token_count` | 各 exemplar 组的 token 计数 |
| `test_example` | 测试样本列表（含 `id`, `name`, `image`, `answer`） |

**对本项目的适用性**：❌ **不直接适用**。ICL 任务本质是图像分类（给定若干示例图 -> 识别测试图类别），不涉及文档页面检索。ColPali 的 Late Interaction 方法无法直接用于此类任务。

---

### 2.5 Summarization（长文档摘要）

**数据格式**：JSONL，每条样本包含一个长文档（~51 页渲染图像）及其参考摘要。

**子任务**：

| 子任务      | 文件     | K128 样本数 | 文档来源            | 页数/文档 |
| ----------- | -------- | ----------- | ------------------- | --------- |
| GovReport   | `gov`    | 241         | 美国政府报告（GAO） | ~51 页    |
| MultiLexSum | `lexsum` | 146         | 法律案例摘要        | ~51 页    |

**样本字段**：
| 字段 | 说明 |
|------|------|
| `id` | 文档 ID |
| `url` | 原始文档 URL |
| `image_list` | 文档页面图像路径列表（~51 页） |
| `summary` | 参考摘要（gov 为结构化列表，lexsum 为文本字符串） |
| `length` | 输入 token 总数 |

**附件**：`gov_claims.jsonl` / `lexsum_claims.jsonl` 提供预提取的原子声明（atomic claims），用于 GPT-4o 评测。

**对本项目的适用性**：⚠️ **间接相关**。摘要任务需要理解长文档内容，ColPali 的页面级检索可作为摘要系统的前置模块（检索相关页面 -> 生成摘要）。但 MMLongBench 的评估方式（原子声明匹配）与检索评测（Recall@k）不同，需要额外适配。

---

### 2.6 Long-Document VQA（长文档问答）

**数据格式**：JSONL，每条样本包含一个长文档的多页图像和自然语言问答对。

**子任务**：

| 子任务     | 文件         | K128 样本数 | 文档来源      | 页数/文档 | 答案来源                   |
| ---------- | ------------ | ----------- | ------------- | --------- | -------------------------- |
| LongDocURL | `longdocurl` | 1,153       | 网页/PDF 文档 | ~51 页    | Text, Image, Chart, Table  |
| MMLongDoc  | `mmlongdoc`  | 961         | 研究报告等    | ~51 页    | Text, Chart, Table, Layout |
| SlideVQA   | `slidevqa`   | 1,064       | 演示幻灯片    | ~130 页   | Text, Figure               |

**样本字段**：
| 字段 | 说明 |
|------|------|
| `id` | 样本 ID |
| `doc_name` | 文档名称 |
| `question` | 自然语言问题 |
| `answer` | 标准答案（字符串） |
| `ans_page_list` | 答案所在页面编号列表 |
| `answer_sources` | 答案来源类型（Text / Chart / Table / Image / Layout） |
| `answer_format` | 答案格式（String / Number） |
| `page_list` | 文档页面图像路径列表 |

**对本项目的适用性**：✅ **高度相关**。DocumentQA 是本项目最直接的评测场景：

- `ans_page_list` 标注了答案所在的具体页面，可直接作为页级检索的 ground truth
- 三个子任务覆盖了不同类型的文档（URL 网页、研究报告、幻灯片），能全面评估检索能力
- `answer_sources` 标记了答案的类型（文本/图表/表格），可分析不同模态信息的检索难度
- 每个 query 关联的文档有 51–130 页，适合评测页面级检索的 Recall@k

---

## 三、数据统计总表

### 3.1 K128 档位各子任务样本数

| 任务类别       | 子任务                  | K128 样本数         | 文件大小 |
| -------------- | ----------------------- | ------------------- | -------- |
| **VRAG**       | infoseek                | 3,384               | ~1.7 GB  |
|                | viquae                  | 6,954               | ~3.5 GB  |
| **NIAH**       | counting-image          | 1,767               | ~764 MB  |
|                | counting-text           | 1,767               | ~768 MB  |
|                | reasoning-image         | 3,474               | ~1.5 GB  |
|                | reasoning-text          | 1,737               | ~755 MB  |
|                | retrieval-image         | 3,600               | ~1.5 GB  |
|                | retrieval-text          | 3,600               | ~1.5 GB  |
|                | text-haystack retrieval | 2,814               | ~1.5 GB  |
|                | vh_multi                | 3,000               | ~36 MB   |
|                | vh_single               | 6,000               | ~71 MB   |
| **ICL**        | cars196                 | 458 (test examples) | ~430 KB  |
|                | food101                 | 500 (test examples) | ~420 KB  |
|                | inat2021                | 待确认              | —        |
|                | sun397                  | 待确认              | —        |
| **Summ**       | gov                     | 241                 | ~1.7 MB  |
|                | lexsum                  | 146                 | ~598 KB  |
| **DocumentQA** | longdocurl              | 1,153               | ~2.9 MB  |
|                | mmlongdoc               | 961                 | ~5.3 MB  |
|                | slidevqa                | 1,064               | ~17.3 MB |

### 3.2 长度档位覆盖

每个子任务均有 **6 个长度档位**：`K4`, `K8`, `K16`, `K32`, `K64`, `K128`（即相同样本数 x 6 份数据文件），对应约 4K–128K tokens 的上下文长度。文件命名规则为 `{task_name}_K{length}_dep{depth}.jsonl`。

---

## 四、与本项目的适配分析

### 4.1 可直接映射为页级检索的任务

| 任务           | 关键字段               | 作为 Ground Truth 的方式                     |
| -------------- | ---------------------- | -------------------------------------------- |
| **DocumentQA** | `ans_page_list`        | 答案所在页面编号列表 → 相关页面集合          |
| **Summ**       | `image_list`（文档页） | 需额外判断哪些页面与摘要相关（原子声明映射） |
| **VRAG**       | `positive_ctxs`        | 正相关 chunk → 关联的文档页面/片段           |

### 4.2 需要适配的任务

| 任务     | 问题                           | 可能方案                          |
| -------- | ------------------------------ | --------------------------------- |
| **NIAH** | 标注的是"针"位置而非页面相关性 | 将 `positive_ctxs` 映射为相关页面 |
| **ICL**  | 图像分类任务，非检索           | **不建议用于本项目评测**          |

### 4.3 推荐评测方案

**Phase 2-3 基准评测**：优先使用 **DocumentQA**（longdocurl + mmlongdoc + slidevqa）作为主评测集，因为：

1. `ans_page_list` 明确标注了相关页面
2. 覆盖了网页、研究报告、幻灯片三种典型文档类型
3. 每个 query 对应 51–130 页的文档，能有效评测检索排序质量
4. 三种子任务共计 3,178 条查询，样本量充足

**Phase 4 扩展评测**：可纳入 VRAG 和 Summ 任务进行跨场景泛化测试。

---

## 五、图像数据存储

### 5.1 已下载图像包

| 图像包                 | 内容                               | 本地路径                                                                                                     | 规模                          |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| `1_vrag_image.tar.gz`  | VRAG 任务图像（infoseek + viquae） | `data/MMLongBench/raw/mmlb_image/infoseek/`（2,232 文件）, `viquae/`（3,317 文件）                           | 5,549 图像文件                |
| `5_docqa_image.tar.gz` | DocumentQA 任务图像                | `data/MMLongBench/raw/mmlb_image/longdocurl/`（396 文档目录）, `mmlongbench-doc/`（135）, `slideVQA/`（293） | 824 文档目录，~40,000+ 页图像 |

> **已下载**：`0_mmlb_data.tar.gz`（元数据）、`1_vrag_image.tar.gz`（VRAG 图像）、`5_docqa_image.tar.gz`（DocumentQA 图像）。
>
> **尚未下载的图像包**：
>
> - `2_vh_image.tar.gz` / `2_mm-niah_image.tar.gz`（NIAH）
> - `3_icl_image.tar.gz`（ICL）
> - `4_summ_image.tar.gz`（Summ）

### 5.2 图像路径约定

数据文件中引用的图像路径均为**相对于 `mmlb_image/` 目录**的相对路径。例如：

- VRAG: `infoseek/oven_04953036.jpg`
- DocumentQA — longdocurl: `longdocurl/{doc_id}/{doc_id}_page{N}.jpg`（页码 N 为 1-based）
- DocumentQA — mmlongdoc: `mmlongbench-doc/{hash}/{hash}_page{N}.jpg`
- DocumentQA — slidevqa: `slideVQA/{doc_dir}/{slide_name}-{N}-1024.jpg`（页码 N 为 slide 编号，1-based）
- Summ: `gov-report/bg_pdf_img/gao-18-107/gao-18-107_page0.jpg`

> **页码提取说明**：longdocurl 和 mmlongdoc 的图像文件名含标准 `_page{N}` 后缀；slidevqa 的图像文件名不含 `_page{N}`，而是以 `-{slide_num}-{resolution}.jpg` 格式嵌入 slide 编号。数据接入层（`DocumentQAAdapter._extract_page_number()`）统一处理两种模式，将 `ans_page_list` 中的页码映射到 `page_idx`（0-based）。

---

## 六、数据划分确认

| 问题                         | 结论                                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 是否有 train/val/test 划分？ | **仅 test 集**。MMLongBench 是一个评测基准（benchmark），不提供训练数据。                                         |
| 是否有官方评测脚本？         | 官方 GitHub 仓库提供评测代码。Summ 任务提供预提取原子声明用于 GPT-4o 评测。                                       |
| 零样本设定是否合适？         | ✅ 是。本项目使用 ColPali（预训练 VLM），不涉及在 MMLongBench 上训练，符合零样本检索（Zero-Shot VDR）的方法定位。 |

---

## 七、关键发现与后续行动

### 7.1 关键发现

1. **MMLongBench 是纯评测基准**：无训练集，无验证集，所有数据均为 test split。这正好符合本项目"零样本"的方法设计。

2. **DocumentQA 是最适配的子集**：`ans_page_list` 提供了精确的页面级 ground truth，3 个子任务共计 3,178 条查询，适合作为 Phase 2-3 的核心评测集。

3. **VRAG 与"NIAH retrieval"系列也可复用**：虽然数据结构不同（chunk 级 vs 页面级），但 positive_ctxs 可作为检索目标标注。

4. **ICL 任务不适用于本项目**：本质是图像分类而非文档检索，不纳入评测范围。

5. **图像数据状态**：目前已有 VRAG（`1_vrag_image.tar.gz`）和 DocumentQA（`5_docqa_image.tar.gz`）的图像包。NIAH、ICL、Summ 的图像包尚未下载，Phase 2 主评测集 DocumentQA 不受影响。

---

## 参考文献

- Wang, Z. et al. _MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly_. NeurIPS 2025 Spotlight.
- Faysse, M. et al. _ColPali: Efficient Document Retrieval with Vision Language Models_. arXiv, 2024.
- Khattab, O. and Zaharia, M. _ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT_. SIGIR, 2020.
