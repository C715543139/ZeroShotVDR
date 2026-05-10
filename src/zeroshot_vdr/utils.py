"""
通用工具模块：日志、计时、路径处理等。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 项目根目录（从本文件位置推导）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """返回项目根目录的绝对路径。"""
    return _PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    """将相对路径（基于项目根目录）解析为绝对路径。

    若输入已是绝对路径则直接返回。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def setup_logging(name: str = "zeroshot_vdr", level: int = logging.INFO) -> logging.Logger:
    """创建并配置项目级 logger。

    Parameters
    ----------
    name : str
        Logger 名称
    level : int
        日志级别

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer(name: str = "operation", logger: logging.Logger | None = None) -> Iterator[float]:
    """上下文管理器：计时并输出耗时。

    Usage::

        with timer("index build", logger=log) as elapsed:
            do_work()
        # 自动打印 "index build 完成，耗时 12.34s"
        print(f"实际耗时: {elapsed:.2f}s")
    """
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    msg = f"{name} 完成，耗时 {elapsed:.2f}s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def format_duration(seconds: float) -> str:
    """将秒数格式化为易读字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h{int(m)}m{s:.0f}s"
