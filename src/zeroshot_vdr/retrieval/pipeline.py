"""
检索流水线：编排查询编码 → 候选召回 → MaxSim 精排 → Top-k 结果组装。

Baseline 采用文档内检索协议（per-document ranking），
流水线四阶段结构为 Phase 4 两阶段检索预留扩展点。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from zeroshot_vdr.contracts import Query, RetrievalResult
from zeroshot_vdr.indexing.store import IndexStore
from zeroshot_vdr.retrieval.encoder import QueryEncoder
from zeroshot_vdr.retrieval.scoring import batched_maxsim

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """检索流水线编排器。

    流程：
    1. ``encode_query(query.text)``  → query embedding
    2. ``generate_candidates(query)`` → 候选 page_id 列表
    3. ``score_candidates()``         → 对候选逐批 MaxSim 打分
    4. ``assemble_results(top_k)``    → 排序、截断、封装为 RetrievalResult 列表

    Baseline 模式下，``retrieve(Query)`` 接收 Query 对象而不是纯文本字符串。
    若 candidate_ids 为空，则 ``generate_candidates()`` 默认使用
    ``query.doc_id`` 对应的文档内页面集合（文档内检索协议）。

    Parameters
    ----------
    model :
        ColPali 模型实例，或 PageEncoder / QueryEncoder（自动提取底层模型）。
    index_store : IndexStore
        索引存储实例
    processor :
        ColPali 处理器；仅在 model 为裸 ColPali 模型时需要。
        若 model 是 PageEncoder，processor 会自动提取，无需显式传入。
    query_encoder : QueryEncoder | None
        显式提供的查询编码器；优先级最高。
    config : dict | None
        检索配置字典；为 None 时从 config/default.yaml 加载
    """

    def __init__(
        self,
        model,
        index_store: IndexStore,
        processor=None,
        query_encoder: QueryEncoder | None = None,
        config: dict | None = None,
    ):
        self._index_store = index_store

        # ---- 解析 model / processor 参数，构造 QueryEncoder ----
        # 优先级: 显式 query_encoder > model 为 QueryEncoder > model 为 PageEncoder
        #          > model + processor > 报错
        if query_encoder is not None:
            self._query_encoder = query_encoder
            self._model = query_encoder.model
        elif isinstance(model, QueryEncoder):
            self._query_encoder = model
            self._model = model.model
        elif hasattr(model, "_model") and hasattr(model, "_processor"):
            # PageEncoder: 有 _model 和 _processor 属性
            self._model = model._model
            self._query_encoder = QueryEncoder(
                model=model._model,
                processor=model._processor,
                device=getattr(model, "_device", "cuda:0"),
            )
        elif processor is not None:
            # 直接传入 ColPali 模型 + 独立 processor
            self._model = model
            self._query_encoder = QueryEncoder(
                model=model,
                processor=processor,
                device=getattr(model, "device", None)
                or next(model.parameters()).device.type,
            )
        else:
            raise TypeError(
                "无法构造 QueryEncoder：请满足以下条件之一：\n"
                "  (1) 传入 query_encoder= 参数\n"
                "  (2) 将 model= 设为 PageEncoder 实例（自动提取处理器）\n"
                "  (3) 同时传入 model=ColPali 和 processor=ColPaliProcessor"
            )

        # ---- 配置 ----
        if config is None:
            from zeroshot_vdr.config import get_retrieval_config
            config = get_retrieval_config()

        self._config = config
        self._score_batch_size = config.get("score_batch_size", 64)
        self._candidate_strategy = config.get("candidate_strategy", "full")

    # ------------------------------------------------------------------
    # 查询编码
    # ------------------------------------------------------------------

    def encode_query(self, query_text: str) -> torch.Tensor:
        """编码查询文本 → ``[n_tokens, dim]``。

        Parameters
        ----------
        query_text : str
            自然语言查询文本

        Returns
        -------
        torch.Tensor
            shape ``[n_tokens, dim]``
        """
        return self._query_encoder.encode(query_text)

    # ------------------------------------------------------------------
    # 候选召回
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        query: Query,
        query_emb: torch.Tensor | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """候选召回阶段。

        Baseline（candidate_strategy="full"）：
        返回与 ``query.doc_id`` 对应的全部页面（文档内候选）。

        Phase 4（candidate_strategy="mean_pool_topN"）：
        使用均值池化向量粗筛 top-n 候选（通过 top_n 参数控制）。

        Parameters
        ----------
        query : Query
            检索查询对象
        query_emb : torch.Tensor | None
            查询 embedding（Phase 4 粗筛使用）
        top_n : int | None
            候选数量限制；None 表示全部返回

        Returns
        -------
        list[str]
            候选 page_id 列表
        """
        strategy = self._candidate_strategy

        if strategy == "full":
            # Baseline: 文档内全部页面
            candidates = self._index_store.list_page_ids(doc_id=query.doc_id)
            logger.debug(
                "候选召回 (full): doc_id=%s → %d 候选页面",
                query.doc_id, len(candidates),
            )
            return candidates

        elif strategy == "mean_pool_topN":
            # Phase 4: 均值池化粗筛
            return self._generate_candidates_mean_pool(
                query, query_emb, top_n
            )

        else:
            raise ValueError(f"未知的候选策略: {strategy}")

    def _generate_candidates_mean_pool(
        self,
        query: Query,
        query_emb: torch.Tensor | None,
        top_n: int | None = None,
    ) -> list[str]:
        """均值池化粗筛（Phase 4 预留）。"""
        if query_emb is None:
            raise ValueError("mean_pool 策略需要 query_emb 参数")

        if top_n is None:
            top_n = 50  # 默认 top-50

        # 获取文档内页面的均值池化视图
        doc_page_ids = self._index_store.list_page_ids(doc_id=query.doc_id)
        pooled, pids = self._index_store.get_mean_pooled_view(doc_page_ids)

        if pooled.numel() == 0:
            return []

        # 计算查询均值向量与页面均值向量的点积
        query_mean = query_emb.mean(dim=0)  # [dim]
        query_mean = query_mean / query_mean.norm(p=2).clamp(min=1e-12)
        pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

        scores = pooled @ query_mean  # [n_pages]
        top_indices = scores.topk(min(top_n, len(pids))).indices

        return [pids[i] for i in top_indices.tolist()]

    # ------------------------------------------------------------------
    # 精排打分
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        query_emb: torch.Tensor,
        candidate_ids: list[str],
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """对候选集逐批 MaxSim 打分。

        按 batch_size 分批从 IndexStore 加载页面 embeddings，
        逐批计算 batched_maxsim，避免同时加载所有页面导致 OOM。

        Parameters
        ----------
        query_emb : torch.Tensor
            shape ``[n_tokens, dim]``
        candidate_ids : list[str]
            候选 page_id 列表
        batch_size : int | None
            每批处理的页面数；None 使用配置中的 score_batch_size

        Returns
        -------
        tuple[torch.Tensor, list[str]]
            ``(scores [n_candidates], scored_page_ids)``
            scores 与 scored_page_ids 一一对应
        """
        if batch_size is None:
            batch_size = self._score_batch_size

        if not candidate_ids:
            return torch.empty((0,)), []

        all_scores: list[torch.Tensor] = []
        all_pids: list[str] = []

        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i : i + batch_size]

            # 从索引加载 embeddings
            try:
                pages_tensor, valid_ids = self._index_store.read_stacked(batch_ids)
                # read_stacked 成功 → 所有页 patch 数一致
                if pages_tensor.numel() == 0:
                    continue
                batch_scores = batched_maxsim(query_emb, pages_tensor)

            except Exception as e:
                logger.warning(
                    "批量读取失败 (candidates %d-%d): %s，回退到逐页模式",
                    i, i + len(batch_ids), e,
                )
                # 回退：逐页加载（可能 patch 数不同）
                tensor_list, valid_ids = self._load_pages_variable(batch_ids)
                if not tensor_list:
                    continue

                # 检查 patch 数是否一致
                shapes = {t.shape[0] for t in tensor_list}
                if len(shapes) == 1:
                    pages_tensor = torch.stack(tensor_list)
                    batch_scores = batched_maxsim(query_emb, pages_tensor)
                else:
                    logger.debug(
                        "候选页面 patch 数不一致 (sizes=%s)，使用逐页 MaxSim",
                        shapes,
                    )
                    from zeroshot_vdr.retrieval.scoring import batched_maxsim_variable
                    batch_scores = batched_maxsim_variable(
                        query_emb, tensor_list
                    )

            all_scores.append(batch_scores)
            all_pids.extend(valid_ids)

        if not all_scores:
            return torch.empty((0,)), []

        return torch.cat(all_scores), all_pids

    def _load_pages_variable(
        self, page_ids: list[str]
    ) -> tuple[list[torch.Tensor], list[str]]:
        """逐页加载 embeddings（变长 patch 回退路径）。

        与 ``read_stacked`` 不同，此方法返回原本的 tensor 列表，
        由上层 ``score_candidates`` 根据 patch 数是否一致选择打分方式。

        Returns
        -------
        tuple[list[torch.Tensor], list[str]]
            ``(tensor_list, valid_page_ids)``，每个 tensor shape ``[n_patches_i, dim]``
        """
        tensors: list[torch.Tensor] = []
        valid_ids: list[str] = []

        for pid, emb in self._index_store.iter_pages(page_ids):
            valid_ids.append(pid)
            tensors.append(emb)

        return tensors, valid_ids

    # ------------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_results(
        query_id: str,
        scores: torch.Tensor,
        page_ids: list[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        """排序、截断、封装为 RetrievalResult 列表。

        Parameters
        ----------
        query_id : str
        scores : torch.Tensor
            shape ``[n_candidates]``
        page_ids : list[str]
        top_k : int

        Returns
        -------
        list[RetrievalResult]
            按 score 降序排列，最多 top_k 条
        """
        if scores.numel() == 0:
            return []

        # 降序排序
        sorted_indices = scores.argsort(descending=True)
        top_k = min(top_k, len(sorted_indices))

        results: list[RetrievalResult] = []
        for rank_0, idx in enumerate(sorted_indices[:top_k].tolist()):
            results.append(
                RetrievalResult(
                    query_id=query_id,
                    page_id=page_ids[idx],
                    score=float(scores[idx].item()),
                    rank=rank_0 + 1,  # 1-based
                )
            )

        return results

    # ------------------------------------------------------------------
    # 主检索接口
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Query,
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
        score_batch_size: int | None = None,
    ) -> list[RetrievalResult]:
        """检索 Top-k 相关页面（baseline 主协议）。

        Parameters
        ----------
        query : Query
            检索查询对象，必须携带 doc_id。
        top_k : int
            返回结果数
        candidate_ids : list[str] | None
            候选页面列表；为 None 时默认使用 query.doc_id 对应的文档内页面集合。
        score_batch_size : int | None
            逐批计算 MaxSim 的页面 batch 大小；None 使用配置默认值。

        Returns
        -------
        list[RetrievalResult]
            按 score 降序排列，最多 top_k 条
        """
        t_start = time.perf_counter()

        # ---- 阶段 1: 查询编码 ----
        query_emb = self.encode_query(query.text)

        # ---- 阶段 2: 候选召回 ----
        if candidate_ids is None:
            candidate_ids = self.generate_candidates(query, query_emb)

        if not candidate_ids:
            logger.warning("查询 %s 无候选页面 (doc_id=%s)", query.query_id, query.doc_id)
            return []

        # ---- 阶段 3: 精排打分 ----
        scores, scored_ids = self.score_candidates(
            query_emb, candidate_ids, batch_size=score_batch_size
        )

        # ---- 阶段 4: 结果组装 ----
        results = self._assemble_results(query.query_id, scores, scored_ids, top_k)

        elapsed = time.perf_counter() - t_start
        logger.debug(
            "检索完成: query=%s candidates=%d top_k=%d elapsed=%.3fs",
            query.query_id, len(candidate_ids), top_k, elapsed,
        )

        return results

    def retrieve_text(
        self,
        text: str,
        candidate_ids: list[str],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """纯文本查询的便利包装接口。

        此接口不承担 baseline 默认协议。
        调用方必须显式提供 candidate_ids。

        Parameters
        ----------
        text : str
            查询文本
        candidate_ids : list[str]
            候选页面 ID 列表（必须显式提供）
        top_k : int
            返回结果数

        Returns
        -------
        list[RetrievalResult]
        """
        # 构造临时 Query（无完整元信息）
        temp_query = Query(
            query_id="adhoc/q000",
            text=text,
            doc_id="",
            raw_doc_name=None,
            task_family="",
            subtask="",
            length="",
        )
        return self.retrieve(
            query=temp_query,
            top_k=top_k,
            candidate_ids=candidate_ids,
        )

    def retrieve_batch(
        self,
        queries: list[Query],
        top_k: int = 10,
        **kwargs,
    ) -> list[list[RetrievalResult]]:
        """批量检索。

        Parameters
        ----------
        queries : list[Query]
            查询列表
        top_k : int
            返回结果数
        **kwargs
            传递给 retrieve() 的额外参数

        Returns
        -------
        list[list[RetrievalResult]]
            每个查询的结果列表
        """
        results: list[list[RetrievalResult]] = []
        for query in queries:
            results.append(self.retrieve(query, top_k=top_k, **kwargs))
        return results

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def index_store(self) -> IndexStore:
        return self._index_store

    @property
    def query_encoder(self) -> QueryEncoder:
        return self._query_encoder
