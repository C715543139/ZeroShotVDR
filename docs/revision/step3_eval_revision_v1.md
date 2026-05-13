# ZeroShotVDR Step 3.1 修订记录 v1

## 1. 说明

本轮修订不再聚焦 Step 2 的核心模块 API，而是转向 **Phase 3 / Step 3.1 的评测编排层**。目标有三个：

1. 将当前仓库中已经落地的 Step 3.1 评测脚本同步回文档
2. 记录评测脚本实现过程中暴露出的两个真实缺陷及其修复
3. 基于本机实测与全量统计，对“在当前 Windows + RTX 4060 Laptop 8GB 设备上跑完全量主评测”给出可执行的时间估算

本修订的关注对象包括：

- `scripts/run_step3_eval.py`
- `scripts/run_step3_clean.py`
- `scripts/env.ps1`
- `src/zeroshot_vdr/indexing/store.py`
- `src/zeroshot_vdr/retrieval/pipeline.py`
- `src/zeroshot_vdr/retrieval/scoring.py`
- `docs/Project_Plan.md` 中与脚本、输出和 Step 3.1 相关的段落

---

## 2. 本轮实际变更

### 2.1 新增 Step 3.1 评测脚本

新增 `scripts/run_step3_eval.py`，当前脚本已经支持：

- DocumentQA 主评测子集（`longdocurl` / `mmlongdoc` / `slidevqa`）
- 文档内页级检索协议
- 项目内 Hugging Face 缓存目录自动设置
- 强制 offline 模式：`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
- `--stats-only` 纯统计模式
- `--max-queries` / `--query-offset` 局部 smoke test
- 缺页时自动增量补建索引
- 结果按 `outputs/eval_reports/{run_name}/` 分目录落盘

当前输出文件包括：

- `metrics_summary.csv`
- `metrics_overall.csv`
- `metrics_by_subtask.csv`
- `metrics_by_length.csv`
- `retrieval_details.json`
- `run_summary.json`

---

### 2.2 新增 Step 3 清理脚本

补充 `scripts/run_step3_clean.py` 作为 Step 3 的清理入口，负责处理两类产物：

- `outputs/eval_reports/{run_name}/` 下的评测结果目录
- `data/processed/index/pages/` 下按 `subtask × length` 匹配的页面 embedding 文件

当前脚本支持：

- 通过 `--run-names` 精确定位评测目录
- 通过 `--subtasks` / `--lengths` 做范围清理
- 通过 `--all` 一次清理全部 Step 3 产物
- 默认 dry-run 预览，传入 `--yes` 后才执行真实删除

这补齐了 Step 3 的整个生命周期：

- `run_step3_eval.py` 负责生成与落盘
- `run_step3_clean.py` 负责回收旧结果与局部索引，便于重新试跑

---

### 2.3 `env.ps1` 成为评测前统一入口

根据实际运行经验，Step 3.1 之前必须先激活项目环境。因此本轮将 `scripts/env.ps1` 固定为统一入口：

```powershell
. .\scripts\env.ps1
```

当前脚本行为：

- 若 PowerShell 已初始化 conda shell hook，则激活 `zeroshotvdr`
- 若未初始化，则输出 warning，但仍继续激活项目 `.venv`

这能保证评测命令至少使用正确的项目解释器，并使 `sitecustomize.py` 在项目根目录下自动生效。

---

### 2.4 本轮顺手修复的两个真实缺陷

#### 缺陷 A：候选页过滤不能只按 `doc_id`

原实现中，`IndexStore.list_page_ids()` 与 `RetrievalPipeline.generate_candidates()` 默认只按 `doc_id` 过滤候选页。该逻辑在 Step 3.1 全量评测中会产生一个真实问题：

- 同一文档会在 `K4/K8/K16/.../K128` 多个长度档位中重复出现
- 若只按 `doc_id` 过滤，`K32` 查询可能拿到 `K64` 或 `K128` 的页面
- 这会直接污染文档内排序与最终指标

本轮修复后，候选过滤已提升为同时约束：

- `task_family`
- `subtask`
- `length`
- `doc_id`

并补了跨长度同文档测试覆盖。

#### 缺陷 B：Query / Page embedding dtype 不一致会导致 MaxSim 崩溃

真实评测时，查询向量来自 `bfloat16` 推理，而页面索引默认以 `float16` 落盘。原来的 `batched_maxsim()` 没有做 dtype 对齐，导致第一次真实 smoke test 直接报错：

```text
expected m1 and m2 to have the same dtype, but got: BFloat16 != Half
```

本轮已在 `retrieval/scoring.py` 中加入 dtype 对齐逻辑，并补充 mixed precision 测试。

---

## 3. 已完成的本机验证

### 3.1 smoke test 结果

已完成：

```powershell
. .\scripts\env.ps1
python scripts/run_step3_eval.py --subtasks longdocurl --lengths K4 --max-queries 5 --page-batch-size 1 --score-batch-size 8 --run-name smoke_eval_longdocurl_K4_q5
```

结果目录：

- `outputs/eval_reports/smoke_eval_longdocurl_K4_q5/`

关键结果：

- 评测范围：5 queries，4 docs，20 pages
- 平均候选页数：4.4 pages/query
- 平均检索延迟：约 0.184 s/query
- Recall@1 = 0.4
- Recall@3 = 1.0
- MRR = 0.7

这证明 Step 3.1 从“脚本入口 → 数据筛选 → 查询编码 → 文档内排序 → 指标输出”已可在本机端到端跑通。

---

### 3.2 全量范围统计

已完成：

```powershell
. .\scripts\env.ps1
python scripts/run_step3_eval.py --stats-only --run-name full_scope_stats
```

当前全量主评测范围统计为：

- 15,577 queries
- 3,653 docs
- 117,724 pages
- 当前已有局部索引 20 pages，因此尚需编码 117,704 pages
- 平均候选页数约 33.33 pages/query

结果文件位于：

- `outputs/eval_reports/full_scope_stats/run_summary.json`

---

## 4. 本机全量时间估算

### 4.1 估算依据

估算使用两组观测：

1. `longdocurl/K4` smoke test：
   - 平均延迟约 0.184 s/query
   - 平均候选页数 4.4

2. `slidevqa/K32` 代表性样本（5 queries，71 pages，两份文档）完整基准：
   - 候选规模与全量平均更接近（本次样本平均 36.4 pages/query）
   - 索引补建耗时 240.45 s，折合约 3.39 s/page
   - 检索阶段平均延迟约 0.156 s/query

之所以采用 `slidevqa/K32` 作为编码估算基准，是因为：

- 全量主评测平均候选页数为 33.33
- `slidevqa/K32` 的平均候选页数约 33.16，最接近全量平均负载
- `slidevqa` 页面也占全量页面数的较大比例，对总时长更敏感

---

### 4.2 保守时间估算

在当前本机、并采用已验证稳定的保守参数：

```text
page_batch_size = 1
score_batch_size = 8
device = cuda:0
```

可给出如下估算：

- **索引补建**：117,704 pages × 3.39 s/page ≈ **110.7 小时**
- **全量检索**：15,577 queries × 0.156 s/query ≈ **0.7 小时**
- **模型加载、结果写盘、波动余量**：约 **0.5–1 小时**

因此，**本机完整 Step 3.1 全量评测的现实估算为 110–115 小时，约 4.6–4.8 天**。

---

### 4.3 如何理解这个估算

这个估算是“保守且可落地”的，而不是“理论最优”的：

- 它基于已经在本机跑通的离线脚本与弱设备安全参数
- 它优先保证**能跑完**，而不是优先追求最高吞吐
- 若后续确认 `page_batch_size=2` 或 `4` 在本机长期稳定不 OOM，则总时间可能下降

但在未完成对应整轮实测前，这类更激进配置只应作为“潜在优化空间”，不应覆盖主估算。

---

## 5. 对 `Project_Plan.md` 的回填项

本轮已同步更新以下内容：

1. `scripts/` 结构从早期 bat 草案同步为当前真实脚本布局，并补充 `run_step3_clean.py`
2. `outputs/eval_reports/` 的输出结构改为按 `run_name` 分目录
3. 2.7 节新增 `env.ps1`、`run_step3_eval.py` 与 `run_step3_clean.py` 的说明
4. Phase 3 / Step 3.1 增加当前实现状态与本机全量时间估算
5. Phase 3 产出补充 `run_summary.json`

这意味着：后续若继续推进 Phase 3，不需要再从零解释“脚本怎么跑”或“为什么本机不适合直接全量”，而可以直接引用本修订记录与计划文档中的同步说明。
