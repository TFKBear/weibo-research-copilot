"""评论来源后端。

为什么要有一个「后端」概念：
评论是整条链路里最容易失效的一环 —— 微博同时存在移动端 hotflow 与网页端
buildComments 两套评论接口，参数、Referer、响应结构都不同，且会各自独立变化。
把「从哪里拿评论」显式建模，才有三件事：

1. 自动回退（一个空了就换另一个，而不是把空当成"没有评论"）；
2. 可追溯（研究日志记录到底是哪个后端供给的数据）；
3. 可诊断（``spikes/04_probe_comments.py`` 能逐个试并打印原始响应）。

⚠ 这些接口形态是**待验证**的：第 1 步实测发现移动端 hotflow 对某些帖子
返回 ``ok=1`` 但无数据。因此本模块的所有后端都会在第一次使用时被验证，
并且由调用方把"哪个后端成功"写进日志，而不是默认相信某一个。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .endpoints import MOBILE_BASE, Request, comments_request

#: 网页端评论接口。微博主站自己用的就是它，``flow`` 控制排序。
WEB_COMMENTS_URL = "https://weibo.com/ajax/statuses/buildComments"

#: 单页条数。接口上限一般为 20。
PAGE_SIZE = 20


Builder = Callable[..., Request]


@dataclass(frozen=True, slots=True)
class CommentBackend:
    """一个评论来源。"""

    key: str
    label: str
    note: str
    builder: Builder
    supports_sort: bool = False

    def request(
        self,
        post_mid: str,
        *,
        max_id: int = 0,
        max_id_type: int = 0,
        sort: str = "hot",
    ) -> Request:
        return self.builder(
            post_mid, max_id=max_id, max_id_type=max_id_type, sort=sort
        )


# --------------------------------------------------------------------------- #
# 1) 移动端 hotflow
# --------------------------------------------------------------------------- #
def _mobile_hotflow(
    post_mid: str, *, max_id: int, max_id_type: int, sort: str
) -> Request:
    """移动端热门评论流。不提供排序参数（拿到的就是平台的"热门"排序）。"""
    return comments_request(post_mid, max_id=max_id, max_id_type=max_id_type)


def _mobile_hotflow_reload(
    post_mid: str, *, max_id: int, max_id_type: int, sort: str
) -> Request:
    """移动端 hotflow + ``is_reload``。

    前端"刷新评论"时会带这个参数；部分帖子只有在带它时才返回数据。
    """
    base = comments_request(post_mid, max_id=max_id, max_id_type=max_id_type)
    params = dict(base.params)
    params["is_reload"] = 1
    return replace(base, params=params, label=base.label + " (is_reload)")


# --------------------------------------------------------------------------- #
# 2) 网页端 buildComments
# --------------------------------------------------------------------------- #
def _web_build_comments(
    post_mid: str, *, max_id: int, max_id_type: int, sort: str
) -> Request:
    """网页端评论接口。

    ``flow``：0 = 按热度，1 = 按时间。这是本项目唯一确认能表达"评论排序"的参数，
    对研究很有价值 —— "按热度取前 N 条"与"按时间取前 N 条"是两个完全不同的样本。
    """
    flow = 0 if sort == "hot" else 1
    return Request(
        url=WEB_COMMENTS_URL,
        params={
            "is_reload": 1,
            "id": post_mid,
            "is_show_bulletin": 2,
            "count": PAGE_SIZE,
            "flow": flow,
            "max_id": max_id,
            "max_id_type": max_id_type,
        },
        label=f"网页评论 mid={post_mid} flow={flow} max_id={max_id}",
        referer="https://weibo.com/",
    )


# --------------------------------------------------------------------------- #
MOBILE_HOTFLOW = CommentBackend(
    key="mobile_hotflow",
    label="移动端 hotflow",
    note="m.weibo.cn/comments/hotflow，主流抓取方案，按平台热门排序，无排序参数",
    builder=_mobile_hotflow,
)

MOBILE_HOTFLOW_RELOAD = CommentBackend(
    key="mobile_hotflow_reload",
    label="移动端 hotflow (is_reload)",
    note="同上但带 is_reload=1，部分帖子只有带此参数才返回数据",
    builder=_mobile_hotflow_reload,
)

WEB_BUILD_COMMENTS = CommentBackend(
    key="web_build_comments",
    label="网页端 buildComments",
    note="weibo.com/ajax/statuses/buildComments，支持 flow 控制按热度/按时间",
    builder=_web_build_comments,
    supports_sort=True,
)

#: ``auto`` 模式的尝试顺序。
#:
#: 移动端放第一位：它的字段更贴近移动端页面，且不需要依赖 weibo.com 的
#: 桌面会话；网页端作为第二道保险，同时提供排序参数。
AUTO_BACKENDS: tuple[CommentBackend, ...] = (
    MOBILE_HOTFLOW,
    WEB_BUILD_COMMENTS,
    MOBILE_HOTFLOW_RELOAD,
)

#: 探测脚本用的完整清单（含对照组）。
ALL_BACKENDS: tuple[CommentBackend, ...] = (
    MOBILE_HOTFLOW,
    MOBILE_HOTFLOW_RELOAD,
    WEB_BUILD_COMMENTS,
)

BACKENDS_BY_KEY: dict[str, CommentBackend] = {b.key: b for b in ALL_BACKENDS}


def get_backend(key: str) -> CommentBackend:
    try:
        return BACKENDS_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(
            f"未知评论后端 {key!r}；可用：{', '.join(sorted(BACKENDS_BY_KEY))}"
        ) from exc


def mobile_detail_url(post_mid: str) -> str:
    """帖子详情页地址，用于诊断时人工打开对照。"""
    return f"{MOBILE_BASE}/detail/{post_mid}"
