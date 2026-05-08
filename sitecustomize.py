"""Startup compatibility patches for the local research environment.

This repository relies on a transformers v5 + PEFT combination required by
colpali-engine. Some released transformers versions route PaliGemma/LLaVA-like
models through a MoE PEFT conversion path even though no MoE target mapping is
defined, which raises ``KeyError: 'llava'`` while loading the ColPali adapter.

Python imports ``sitecustomize`` automatically on startup when this repository
root is on ``sys.path``. Patching here keeps interactive `python` sessions and
`uv run python` behavior aligned without editing files inside `.venv`.
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

        # Some visual model families are listed in the generic checkpoint
        # conversion mapping but have no MoE-specific target mapping.
        if moe_mapping.get(base_model_type) is None:
            return peft_config

        return original(peft_config, model_type)

    transformers_peft._convert_peft_config_moe = _patched_convert_peft_config_moe


_patch_transformers_peft_llava_conversion()