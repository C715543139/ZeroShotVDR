"""本地研究环境的启动兼容性补丁。

此仓库依赖于 colpali-engine 所需的 transformers v5 + PEFT 组合。某些已发布的
transformers 版本会将 PaliGemma/LLaVA 类模型路由到 MoE PEFT 转换路径，即使没
有定义 MoE 目标映射，这会在加载 ColPali 适配器时引发 ``KeyError: 'llava'``。

当此仓库根目录位于 ``sys.path`` 中时，Python 会在启动时自动导入 ``sitecustomize``。
在此处打补丁可以使交互式 `python` 会话和 `uv run python` 行为保持一致，而无需编辑
`.venv` 中的文件。
"""

from __future__ import annotations


def _patch_transformers_peft_llava_conversion() -> None:
    try:
        import transformers.integrations.peft as transformers_peft
    except Exception:
        return

    original = getattr(transformers_peft, "_convert_peft_config_moe", None)
    model_mapping = getattr(transformers_peft, "_MODEL_TO_CONVERSION_PATTERN", None)
    moe_mapping = getattr(transformers_peft, "_MOE_TARGET_MODULE_MAPPING", None)

    if original is None or model_mapping is None or moe_mapping is None:
        return

    if getattr(original, "__name__", "") == "_patched_convert_peft_config_moe":
        return

    def _patched_convert_peft_config_moe(peft_config, model_type: str):
        base_model_type = model_mapping.get(model_type)
        if base_model_type is None:
            return peft_config

        # 某些视觉模型系列在通用检查点转换映射中列出，
        # 但没有 MoE 特定的目标映射。
        if moe_mapping.get(base_model_type) is None:
            return peft_config

        return original(peft_config, model_type)

    transformers_peft._convert_peft_config_moe = _patched_convert_peft_config_moe


_patch_transformers_peft_llava_conversion()