"""评测层：指标计算与 ground truth 加载。

指标计算与具体数据集解耦，接受标准化的 Python list/set 输入。
Ground truth 的加载与格式转换在独立模块中完成。
"""

from zeroshot_vdr.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    compute_all_metrics,
)
from zeroshot_vdr.evaluation.ground_truth import GroundTruthLoader

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "compute_all_metrics",
    "GroundTruthLoader",
]
