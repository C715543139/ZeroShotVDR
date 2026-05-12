"""
Ground truth 加载与格式转换。

从 MMLongBench 标注数据中提取 (query_id, page_id) 对，
转为统一的 ``{query_id: set[page_id]}`` 格式。

适配逻辑（DocumentQAAdapter）与指标计算分离，
新增评测子集只需增加适配器，无需改动指标模块。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GroundTruthLoader:
    """Ground truth 加载与格式转换。

    封装 ``DocumentQAAdapter`` 的 ``build_ground_truth()``，
    提供按子集过滤、按任务族选择的统一入口。

    Parameters
    ----------
    config : dict | None
        全局配置字典；为 None 时从 ``config/default.yaml`` 加载
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            from zeroshot_vdr.config import load_config

            config = load_config()

        self._config = config
        self._data_config = config.get("data", {})
        self._gt_cache: dict[str, dict[str, set[str]]] = {}

    # ------------------------------------------------------------------
    # 主加载接口
    # ------------------------------------------------------------------

    def load(
        self,
        subtasks: list[str] | None = None,
        lengths: list[str] | None = None,
        task_family: str = "docqa",
    ) -> dict[str, set[str]]:
        """加载 ground truth。

        Parameters
        ----------
        subtasks : list[str] | None
            限定子任务列表；None 使用配置中的 subtasks。
            例：``["longdocurl", "mmlongdoc"]``
        lengths : list[str] | None
            限定长度档位；None 使用配置中的 length（或全部档位）。
            例：``["K32"]``
        task_family : str
            任务族名，固定为 ``"docqa"``

        Returns
        -------
        dict[str, set[str]]
            ``{query_id: {relevant_page_id, ...}}``
        """
        # 解析默认值
        if subtasks is None:
            subtasks = self._data_config.get(
                "subtasks", ["longdocurl", "mmlongdoc", "slidevqa"]
            )
        if lengths is None:
            cfg_length = self._data_config.get("length")
            if cfg_length:
                lengths = [cfg_length] if isinstance(cfg_length, str) else cfg_length
            else:
                lengths = ["K4", "K8", "K16", "K32", "K64", "K128"]

        # 缓存键
        cache_key = f"{task_family}_{'-'.join(sorted(subtasks))}_{'-'.join(sorted(lengths))}"
        if cache_key in self._gt_cache:
            return self._gt_cache[cache_key]

        # 通过 DocumentQAAdapter 加载
        from zeroshot_vdr.data.adapters import DocumentQAAdapter

        data_dir = self._data_config.get("root_dir", "data/MMLongBench/raw")

        adapter = DocumentQAAdapter(
            data_dir=data_dir,
            subtasks=subtasks,
            lengths=lengths,
        )

        gt = adapter.build_ground_truth()

        logger.info(
            "Ground truth 加载完成: %d 查询 (%s × %s)",
            len(gt), subtasks, lengths,
        )

        self._gt_cache[cache_key] = gt
        return gt

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def load_by_subtask(
        self,
        subtask: str,
        lengths: list[str] | None = None,
    ) -> dict[str, set[str]]:
        """加载单个子任务的 ground truth。

        Parameters
        ----------
        subtask : str
            子任务名（如 "longdocurl"）
        lengths : list[str] | None

        Returns
        -------
        dict[str, set[str]]
        """
        return self.load(subtasks=[subtask], lengths=lengths)

    def load_by_length(
        self,
        length: str,
        subtasks: list[str] | None = None,
    ) -> dict[str, set[str]]:
        """加载单个长度档位的 ground truth。

        Parameters
        ----------
        length : str
            长度档位（如 "K32"）
        subtasks : list[str] | None

        Returns
        -------
        dict[str, set[str]]
        """
        return self.load(subtasks=subtasks, lengths=[length])

    @property
    def config(self) -> dict:
        """返回当前使用的配置字典（只读）。"""
        return self._config
