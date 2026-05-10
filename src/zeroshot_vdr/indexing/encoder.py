"""
ColPali 页面编码器：将页面图像编码为 patch embeddings。

使用 ColPali-v1.3 模型（LoRA adapter on PaliGemma-3B），
输出形状 ``[n_patches, dim]`` 的 Late Interaction 表示。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from zeroshot_vdr.contracts import Page
from zeroshot_vdr.indexing.store import IndexStore

logger = logging.getLogger(__name__)


def _verify_peft_patch() -> None:
    """验证 sitecustomize PEFT MoE 兼容补丁是否已生效。

    若补丁未生效，在 Windows 上加载 ColPali 时会触发
    ``KeyError: 'llava'``，进程直接崩溃。

    Raises
    ------
    RuntimeError
        若补丁未检测到，给出明确的修复指引
    """
    try:
        import transformers.integrations.peft as transformers_peft
    except Exception:
        return  # transformers 未安装，静默通过

    patched_fn = getattr(transformers_peft, "_convert_peft_config_moe", None)
    if patched_fn is None:
        return

    if getattr(patched_fn, "__name__", "") == "_patched_convert_peft_config_moe":
        logger.debug("sitecustomize PEFT MoE 补丁已生效")
        return

    raise RuntimeError(
        "sitecustomize PEFT MoE 补丁未生效！\n"
        "请确保从项目根目录使用 .venv 中的 Python 启动：\n"
        "  conda activate zeroshotvdr\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python your_script.py\n"
        "或在脚本顶部显式添加：\n"
        "  import sitecustomize  # 必须在 import torch 之前\n"
    )


class PageEncoder:
    """ColPali 页面编码器。

    使用 ColPali VLM 将页面图像编码为多向量表示（Late Interaction），
    每页输出 ``[n_patches, dim]`` 的 patch embeddings。

    Parameters
    ----------
    model :
        ColPali 模型实例（``colpali_engine.models.ColPali``）
    processor :
        ColPali 处理器实例（``ColPaliProcessor``）
    batch_size : int
        GPU 编码批次大小（适配 8GB 显存建议 2~4）
    dtype : torch.dtype
        推理精度
    device : str
        推理设备
    storage_dtype : torch.dtype
        落盘精度（float16 约节省 50% 存储）
    """

    def __init__(
        self,
        model,
        processor,
        batch_size: int = 4,
        dtype: torch.dtype | None = None,
        device: str = "cuda:0",
        storage_dtype: torch.dtype = torch.float16,
    ):
        self._model = model
        self._processor = processor
        self._batch_size = batch_size
        self._device = device
        self._storage_dtype = storage_dtype

        if dtype is not None:
            self._model = self._model.to(dtype)

        self._model.eval()

    # ------------------------------------------------------------------
    # 模型工厂
    # ------------------------------------------------------------------

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
        """从 HuggingFace 仓库加载模型和处理器。

        .. important::
           调用前需确保 ``sitecustomize.py`` 补丁已生效：
           - 从项目根目录使用 ``.venv\\Scripts\\python.exe`` 启动 Python
           - 或先 ``conda activate zeroshotvdr && .\\.venv\\Scripts\\Activate.ps1``
           - 或在脚本顶部显式 ``import sitecustomize``

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
        batch_size : int
            编码批次大小
        storage_dtype : torch.dtype
            落盘精度

        Returns
        -------
        PageEncoder
        """
        if dtype is None:
            dtype = torch.bfloat16

        # ---- 验证 sitecustomize PEFT MoE 补丁 ----
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

        return cls(
            model=model,
            processor=processor,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            storage_dtype=storage_dtype,
        )

    # ------------------------------------------------------------------
    # 编码接口
    # ------------------------------------------------------------------

    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """编码单张页面图像 → ``[n_patches, dim]``。

        Parameters
        ----------
        image : PIL.Image.Image
            页面图像（任意尺寸，处理器会统一 resize）

        Returns
        -------
        torch.Tensor
            shape ``[n_patches, dim]``
        """
        embeddings = self.encode_batch([image])
        return embeddings[0]

    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """编码一批图像 → ``[batch, n_patches, dim]``。

        Parameters
        ----------
        images : list[PIL.Image.Image]

        Returns
        -------
        torch.Tensor
            shape ``[batch, n_patches, dim]``
        """
        if not images:
            return torch.empty((0,))

        batch = self._processor.process_images(images).to(self._device)

        with torch.no_grad():
            embeddings = self._model.forward(**batch)

        return embeddings.cpu()

    # ------------------------------------------------------------------
    # 语料编码（主要编排入口）
    # ------------------------------------------------------------------

    def encode_corpus(
        self,
        pages: list[Page],
        store: IndexStore,
        show_progress: bool = True,
        resume: bool = True,
    ) -> None:
        """遍历页面语料，逐批编码并写入索引存储。

        支持断点续建：若 ``resume=True``，跳过已在索引中的页面。

        Parameters
        ----------
        pages : list[Page]
            待编码的页面列表
        store : IndexStore
            索引存储实例
        show_progress : bool
            是否显示进度条
        resume : bool
            是否跳过已索引的页面（断点续建）
        """
        # 确定待编码页面
        if resume:
            indexed = set(store.list_page_ids())
            remaining = [p for p in pages if p.page_id not in indexed]
            skipped = len(pages) - len(remaining)
            if skipped > 0:
                logger.info(
                    "断点续建: %d/%d 页面已索引，跳过", skipped, len(pages)
                )
        else:
            remaining = list(pages)

        if not remaining:
            logger.info("所有页面均已索引，无需编码")
            return

        # 分批编码
        total = len(remaining)
        batch_size = self._batch_size

        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="编码页面",
                unit="batch",
                total=(total + batch_size - 1) // batch_size,
            )

        for start in iterator:
            batch_pages = remaining[start : start + batch_size]

            # 加载图像
            images: list[Image.Image] = []
            valid_pages: list[Page] = []
            for page in batch_pages:
                try:
                    img = Image.open(page.image_path).convert("RGB")
                    images.append(img)
                    valid_pages.append(page)
                except Exception as e:
                    logger.warning(
                        "无法加载图像 %s: %s", page.image_path, e
                    )
                    continue

            if not images:
                continue

            # 编码
            try:
                embeddings = self.encode_batch(images)
            except Exception as e:
                logger.error("批次编码失败 (pages %d-%d): %s", start, start + batch_size, e)
                raise

            # 转为落盘精度
            if self._storage_dtype != embeddings.dtype:
                embeddings = embeddings.to(self._storage_dtype)

            # 写入索引
            page_ids = [p.page_id for p in valid_pages]
            store.write_batch(page_ids, embeddings)

        logger.info("语料编码完成: %d 页面已写入索引", total)

    # ------------------------------------------------------------------
    # 便利属性
    # ------------------------------------------------------------------

    @property
    def model(self):
        """ColPali 模型实例。"""
        return self._model

    @property
    def processor(self):
        """ColPali 处理器实例。"""
        return self._processor

    @property
    def batch_size(self) -> int:
        """编码批次大小。"""
        return self._batch_size
