# ZeroShotVDR

ZeroShotVDR 是一个面向 **零样本视觉文档检索**（Zero-Shot Visual Document Retrieval, VDR）的课程项目。项目基于 **ColPali-v1.3**，在 **MMLongBench DocumentQA** 上实现页级检索、评测与查询自适应两阶段检索改进。

给定文本查询，系统在视觉丰富长文档的候选页面中返回最相关页面。相比纯 OCR 检索，本项目保留页面图像中的版式、表格、图表和视觉区域信息；相比直接对所有候选页执行完整 ColPali MaxSim，最终方法在保持检索质量的同时降低推理延迟。

## 项目概览

本项目包含三部分：

- **稳定 ColPali baseline**：完成 MMLongBench DocumentQA 数据适配、页面索引、查询编码、MaxSim 检索和 Recall / Precision / MRR / nDCG 评测。
- **评测协议修正**：使用从原始图像路径恢复的稳定 `page_id`，并在主表中仅统计 14,385 条具有有效页级标注的 valid-only queries。
- **进阶方法**：实现 `Adaptive + Neighbor + MeanPoolCache`，即 mean-pool 粗检索、自适应候选选择、邻页扩展和完整 MaxSim 精排。

关键文档：

- 课程任务说明：[docs/NJUProject_VDR.md](docs/NJUProject_VDR.md)
- 项目计划与环境细节：[docs/Project_Plan.md](docs/Project_Plan.md)
- 阶段四结果报告：[docs/Milestone_Report_Phase4.md](docs/Milestone_Report_Phase4.md)
- 最终论文源码：[report/main.tex](report/main.tex)
- 中文论文对照：[docs/ACL_Report_CN.md](docs/ACL_Report_CN.md)

## 结果摘要

主比较口径：MMLongBench DocumentQA 三个子任务（`longdocurl`、`mmlongdoc`、`slidevqa`）六个长度档位（K4-K128），共 14,385 条 valid-only queries。

| Method                              | Recall@10 | nDCG@10 |    Avg Latency |    P95 Latency | Avg Rerank Candidates |
| ----------------------------------- | --------: | ------: | -------------: | -------------: | --------------------: |
| Phase 3 Full MaxSim                 |    0.8517 |  0.6325 | 0.0716 s/query | 0.1384 s/query |                  32.7 |
| Adaptive + Neighbor + MeanPoolCache |    0.8523 |  0.6325 | 0.0600 s/query | 0.0858 s/query |                  19.8 |

K128 长候选集上，最终方法将 P95 latency 从 189.2 ms 降到 97.4 ms，并将平均 full MaxSim rerank 页数从 81.1 降到 34.8，同时 Recall@10 从 0.6818 提升到 0.6882。

结论：最终方法在保持 baseline 质量的前提下显著改善长候选集场景的质量-效率权衡。

## 项目主要结构

```text
ZeroShotVDR/
├── config/                 # 默认配置
├── data/                   # 本地数据与索引
├── docs/                   # 项目说明、计划与阶段报告
├── outputs/                # 评测输出、trace、metrics、cache 等
├── report/                 # ACL 风格最终报告 LaTeX 源文件
├── scripts/
│   ├── command/            # 环境检查、清理、进度查看等辅助命令
│   └── run/                # Step 3 / Phase 4 评测入口
├── src/zeroshot_vdr/
│   ├── data/               # MMLongBench DocumentQA 数据适配
│   ├── indexing/           # ColPali 页面编码与索引存储
│   ├── retrieval/          # 查询编码、MaxSim 打分和检索流水线
│   ├── evaluation/         # ground truth 与指标计算
│   └── advanced/           # 两阶段检索、邻页扩展、mean-pool cache
├── tests/                  # 单元测试与回归测试
├── main.py                 # 统一命令入口
├── pyproject.toml          # uv 项目配置
└── README.md
```

## 环境要求

推荐运行环境：

- Linux / Ubuntu
- Python 3.10
- Conda + uv
- NVIDIA GPU，推荐 CUDA 可用环境
- 项目 full run 使用过 2x RTX 3090 24GB

依赖由 `pyproject.toml` 和 `uv.lock` 管理。更完整的环境说明见 [docs/Project_Plan.md](docs/Project_Plan.md) 的“环境配置指导”部分。

## 配置引导

在项目根目录执行：

```bash
conda create -n zeroshotvdr python=3.10 -y
conda activate zeroshotvdr
uv sync
source .venv/bin/activate
```

日常进入项目时可直接使用统一环境入口：

```bash
source scripts/command/env.sh
```

验证环境：

```bash
python main.py command check-env
```

验证 ColPali 模型加载：

```bash
python main.py command test-model-load
```

数据集与模型权重较大，请按 [docs/Project_Plan.md](docs/Project_Plan.md) 中的数据和 HuggingFace 缓存说明准备。当前稳定索引默认位于：

```text
data/processed/index_stable_page_ids/
```

## 运行指南

查看统一入口帮助：

```bash
python main.py --help
```

### Step 3 baseline 评测

运行稳定 ColPali baseline：

```bash
python main.py step3 eval \
  --run-name step3_docqa_full_dual3090_stable_page_ids \
  --index-dir data/processed/index_stable_page_ids
```

分析 Step 3 输出：

```bash
python main.py step3 analysis \
  --run-dir outputs/eval_reports/step3_docqa_full_dual3090_stable_page_ids
```

### Phase 4 单次评测

运行最终推荐方法：

```bash
python main.py phase4 eval \
  --run-name phase4_adaptive_neighbors_cache_full_20260520 \
  --method adaptive_neighbors \
  --neighbor-window 1 \
  --neighbor-seed-n 8 \
  --valid-only \
  --trace-enabled \
  --use-mean-pool-cache true \
  --mean-pool-cache-dir outputs/cache/mean_pool_full_20260520_rerun \
  --index-dir data/processed/index_stable_page_ids
```

快速 smoke run 示例：

```bash
python main.py phase4 eval \
  --run-name smoke_fixed64 \
  --method fixed_topn \
  --coarse-top-n 64 \
  --max-queries 50 \
  --valid-only
```

### Phase 4 全量矩阵

```bash
python main.py phase4 full
```

查看长任务进度：

```bash
python main.py command phase4-progress --watch
```

## 测试

运行全部测试：

```bash
pytest -q
```

只运行 Phase 4 相关测试：

```bash
pytest -q tests/phase4
```

## 输出位置

主要输出目录：

- `outputs/eval_reports/`：评测结果、metrics、trace、slice / bucket 分析
- `outputs/cache/`：mean-pool cache
- `report/`：最终 ACL 风格报告源码和图表

`data/`、`outputs/`、`.cache/`、`.venv/` 等目录通常包含本地大文件或环境缓存，不应作为源码提交内容。
