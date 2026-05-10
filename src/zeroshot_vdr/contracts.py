"""
数据契约：定义系统中跨模块传递的核心数据结构。

所有 page_id / query_id 均为稳定字符串，贯穿索引→检索→评测全链路。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


# ============================================================================
# 核心数据结构
# ============================================================================


@dataclass
class Page:
    """页面语料中的单页。"""

    page_id: str  # 稳定唯一标识: {task_family}/{subtask}_{length}/{doc_id}/p{page_idx}
    doc_id: str  # 所属文档标识（由 raw_doc_name 归一化得到）
    raw_doc_name: str | None  # 原始数据中的文档来源字段
    task_family: str  # L1 任务族: docqa / icl / niah / summ / vrag
    subtask: str  # L2 子任务: longdocurl / mmlongdoc / slidevqa / ...
    length: str  # L3 长度档位: K4 / K8 / K16 / K32 / K64 / K128
    page_idx: int  # 文档内页码（0-based）
    image_path: str  # 页面图像文件路径（绝对路径或相对于项目根目录）


@dataclass
class Query:
    """单条检索查询。

    由于 baseline 采用文档内检索，Query 必须显式携带所属文档标识 doc_id，
    以便默认候选集合能够约束在同一文档页面内。
    """

    query_id: str  # 稳定唯一标识: {task_family}/{subtask}_{length}/q{query_index:0>3d}
    text: str  # 查询文本
    doc_id: str  # 所属文档标识（决定默认候选范围）
    raw_doc_name: str | None  # 原始数据中的文档来源字段
    task_family: str  # L1 任务族
    subtask: str  # L2 子任务
    length: str  # L3 长度档位


@dataclass
class RetrievalResult:
    """单条检索命中结果。"""

    query_id: str  # 来源查询的稳定 ID
    page_id: str  # 命中页面的稳定 ID
    score: float  # 相似度分数
    rank: int  # 排名（1-based）


@dataclass
class RelevanceJudgment:
    """单条标注：某查询对某页面的相关性。"""

    query_id: str
    page_id: str
    relevance: int  # 0/1 或分级相关度


# ============================================================================
# ID 构造辅助函数（全链路唯一入口）
# ============================================================================


def normalize_doc_id(raw_doc_name: str) -> str:
    """将原始 doc_name 归一化为内部稳定 doc_id。

    规则：
    1. 去除首尾空白字符
    2. 将路径分隔符（反斜杠/正斜杠）替换为下划线
    3. 将连续空白替换为单个下划线

    所有 Adapter 和 Corpus 必须通过此函数获得 doc_id，
    禁止在不同模块中各自临时构造。

    Parameters
    ----------
    raw_doc_name : str
        原始数据中的文档标识字段（如 "4088173", "mmlongbench-doc/abc123"）

    Returns
    -------
    str
        归一化后的稳定 doc_id
    """
    doc_id = raw_doc_name.strip()
    # 替换路径分隔符为下划线
    doc_id = doc_id.replace("\\", "_").replace("/", "_")
    # 将连续空白/下划线折叠为单个下划线
    doc_id = re.sub(r"[_\s]+", "_", doc_id)
    return doc_id


def build_page_id(
    task_family: str,
    subtask: str,
    length: str,
    doc_id: str,
    page_idx: int,
) -> str:
    """构造稳定的 page_id。

    格式：{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}

    这是全链路唯一合法的 page_id 构造入口。

    Parameters
    ----------
    task_family : str
        L1 任务族名（如 "docqa"）
    subtask : str
        L2 子任务名（如 "longdocurl"）
    length : str
        L3 长度档位（如 "K4"）
    doc_id : str
        归一化后的文档 ID
    page_idx : int
        文档内页码（0-based）

    Returns
    -------
    str
        稳定的 page_id 字符串
    """
    return f"{task_family}/{subtask}_{length}/{doc_id}/p{page_idx}"


def build_query_id(
    task_family: str,
    subtask: str,
    length: str,
    query_index: int,
) -> str:
    """构造稳定的 query_id。

    格式：{task_family}/{subtask}_{length}/q{query_index:0>3d}

    这是全链路唯一合法的 query_id 构造入口。

    Parameters
    ----------
    task_family : str
        L1 任务族名（如 "docqa"）
    subtask : str
        L2 子任务名（如 "longdocurl"）
    length : str
        L3 长度档位（如 "K4"）
    query_index : int
        该子任务×档位下的查询序号

    Returns
    -------
    str
        稳定的 query_id 字符串
    """
    return f"{task_family}/{subtask}_{length}/q{query_index:0>3d}"
