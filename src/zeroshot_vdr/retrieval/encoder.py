"""
ColPali 查询编码器：将文本查询编码为 token-level embeddings。

使用与 PageEncoder 共享的 ColPali 模型，通过 process_queries 处理文本输入，
输出 ``[n_tokens, dim]`` 的 token embeddings 供 MaxSim 检索使用。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class QueryEncoder:
    """ColPali 查询编码器。

    将自然语言查询文本编码为多向量表示（Late Interaction），
    每个查询输出 ``[n_tokens, dim]`` 的 token embeddings。

    QueryEncoder 与 PageEncoder 共享同一个 ColPali 模型实例，
    仅在使用不同的 processor 方法（process_queries vs process_images）。

    Parameters
    ----------
    model :
        ColPali 模型实例（``colpali_engine.models.ColPali``），
        通常与 PageEncoder 共享同一实例
    processor :
        ColPali 处理器实例（``ColPaliProcessor``），
        用于处理查询文本
    device : str
        推理设备
    """

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda:0",
    ):
        self._model = model
        self._processor = processor
        self._device = device

        self._model.eval()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_repo: str = "vidore/colpali-v1.3",
        base_repo: str = "vidore/colpaligemma-3b-pt-448-base",
        device: str = "cuda:0",
        dtype: torch.dtype | None = None,
    ) -> "QueryEncoder":
        """从 HuggingFace 仓库加载模型和处理器。

        .. important::
           调用前需确保 ``sitecustomize.py`` 补丁已生效。

        Parameters
        ----------
        model_repo : str
            ColPali LoRA adapter 仓库名
        base_repo : str
            PaliGemma 基础模型仓库名
        device : str
            推理设备
        dtype : torch.dtype | None
            推理精度；None 使用 bfloat16

        Returns
        -------
        QueryEncoder
        """
        if dtype is None:
            dtype = torch.bfloat16

        from zeroshot_vdr.indexing.encoder import _verify_peft_patch
        _verify_peft_patch()

        from colpali_engine.models import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import (
            ColPaliProcessor,
        )

        logger.info("加载 ColPali 模型: %s (base: %s)", model_repo, base_repo)
        model = ColPali.from_pretrained(
            model_repo,
            torch_dtype=dtype,
            device_map=device,
        ).eval()

        logger.info("加载 ColPali 处理器: %s", base_repo)
        processor = ColPaliProcessor.from_pretrained(base_repo)

        return cls(model=model, processor=processor, device=device)

    @classmethod
    def from_page_encoder(cls, page_encoder) -> "QueryEncoder":
        """从已有的 PageEncoder 实例共享模型和处理器。

        这是推荐的生产构造方式，避免重复加载模型权重。

        Parameters
        ----------
        page_encoder : PageEncoder
            已初始化的页面编码器实例

        Returns
        -------
        QueryEncoder
            共享同一模型和处理器的查询编码器
        """
        return cls(
            model=page_encoder._model,
            processor=page_encoder._processor,
            device=page_encoder._device,
        )

    # ------------------------------------------------------------------
    # 编码接口
    # ------------------------------------------------------------------

    def encode(self, query: str) -> torch.Tensor:
        """编码单条文本查询 → ``[n_tokens, dim]``。

        Parameters
        ----------
        query : str
            自然语言查询文本

        Returns
        -------
        torch.Tensor
            shape ``[n_tokens, dim]``
        """
        embeddings = self.encode_batch([query])
        return embeddings[0]

    def encode_batch(self, queries: list[str]) -> torch.Tensor:
        """编码一批查询 → ``[batch, n_tokens, dim]``。

        Parameters
        ----------
        queries : list[str]
            查询文本列表

        Returns
        -------
        torch.Tensor
            shape ``[batch, n_tokens, dim]``
        """
        if not queries:
            return torch.empty((0,))

        batch = self._processor.process_queries(queries).to(self._device)

        with torch.no_grad():
            embeddings = self._model.forward(**batch)

        return embeddings.cpu()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def model(self):
        return self._model
