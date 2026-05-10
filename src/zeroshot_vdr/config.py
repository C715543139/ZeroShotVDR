"""
配置加载与管理模块。

遵循 4.5.8 节语义分层：项目级常量 / 环境复现参数 / 实验切换参数 / 策略选择参数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from zeroshot_vdr.utils import get_project_root

# 默认配置文件路径
_DEFAULT_CONFIG = get_project_root() / "config" / "default.yaml"

# 缓存：避免重复加载
_config_cache: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 配置文件。

    Parameters
    ----------
    path : str | Path | None
        配置文件路径；为 None 时使用 config/default.yaml

    Returns
    -------
    dict
        配置字典（所有键均为小写）
    """
    global _config_cache
    if path is None and _config_cache is not None:
        return _config_cache

    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 设置 HF 缓存环境变量
    _setup_hf_env(config)

    if path is None:
        _config_cache = config

    return config


def _setup_hf_env(config: dict) -> None:
    """根据配置设置 HuggingFace 缓存环境变量（项目内缓存）。"""
    paths = config.get("paths", {})
    project_root = get_project_root()

    hf_home = paths.get("hf_home", ".cache/huggingface")
    os.environ.setdefault("HF_HOME", str(project_root / hf_home))

    hf_hub = paths.get("hf_hub_cache", ".cache/huggingface/hub")
    os.environ.setdefault("HF_HUB_CACHE", str(project_root / hf_hub))

    hf_datasets = paths.get("hf_datasets_cache", ".cache/huggingface/datasets")
    os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / hf_datasets))


def get_data_config(config: dict | None = None) -> dict:
    """获取数据配置段。"""
    cfg = config if config is not None else load_config()
    return cfg.get("data", {})


def get_model_config(config: dict | None = None) -> dict:
    """获取模型配置段。"""
    cfg = config if config is not None else load_config()
    return cfg.get("model", {})


def get_index_config(config: dict | None = None) -> dict:
    """获取索引配置段。"""
    cfg = config if config is not None else load_config()
    return cfg.get("index", {})


def get_retrieval_config(config: dict | None = None) -> dict:
    """获取检索配置段。"""
    cfg = config if config is not None else load_config()
    return cfg.get("retrieval", {})


def get_evaluation_config(config: dict | None = None) -> dict:
    """获取评测配置段。"""
    cfg = config if config is not None else load_config()
    return cfg.get("evaluation", {})
