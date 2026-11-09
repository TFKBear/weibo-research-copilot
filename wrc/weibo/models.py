"""采集到的结构化对象。

这里刻意只保留「研究中真的会用到的字段」，而不是复制微博接口的全部结构。
理由（产品宪法第 1 条）：可追溯要求我们能说清每个字段是怎么来的，
字段越少越容易在论文中交代；额外字段按需在分析阶段派生。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TimePrecision(str, Enum):
    """时间戳精度。

    微博检索结果的时间经常只给到「日」（例如 ``05-12``），
    这在做时间序列分析时是一个真实的系统性误差来源，
    因此必须随数据一起记录，而不是悄悄补成 00:00:00。
    """

    SECOND = "second"
    MINUTE = "minute"
    DAY = "day"
    UNKNOWN = "unknown"


class CommentLevel(int, Enum):
    """评论层级。"""

    PRIMARY = 1  # 主楼评论
    SECONDARY = 2  # 楼中楼


@dataclass(slots=True)
class WeiboPost:
    """一条微博帖子。"""

    mid: str
    user_id: str
    screen_name: str
    text_raw: str
    created_at: datetime | None
    created_at_raw: str
    created_at_precision: TimePrecision
    reposts_count: int = 0
    comments_count: int = 0
    attitudes_count: int = 0
    bid: str | None = None
    is_long_text: bool = False
    source: str = "m.weibo.cn"

    @property
    def engagement(self) -> int:
        """互动总量，用于「按热度」排序与取样分层。"""
        return self.reposts_count + self.comments_count + self.attitudes_count

    def preview(self, width: int = 60) -> str:
        text = " ".join(self.text_raw.split())
        if len(text) > width:
            text = text[: width - 1] + "…"
        return text


@dataclass(slots=True)
class WeiboComment:
    """一条评论。"""

    cid: str
    post_mid: str
    user_id: str
    screen_name: str
    text_raw: str
    created_at: datetime | None
    created_at_raw: str
    created_at_precision: TimePrecision
    level: CommentLevel = CommentLevel.PRIMARY
    parent_cid: str | None = None
    like_count: int = 0
    reply_count: int = 0
    followers_count: int | None = None
    """评论者粉丝数（接口不总会返回）。

    保留它是为了模块 B2 的「清洗前后粉丝数分布对比」——
    清洗是否引入了系统性偏差，只能靠这个分布来回答。
    """
    source: str = "m.weibo.cn"


@dataclass(slots=True)
class CommentPage:
    """评论接口的一页结果。"""

    comments: list[WeiboComment] = field(default_factory=list)
    next_max_id: int | None = None
    next_max_id_type: int | None = None
    total_number: int | None = None

    @property
    def has_more(self) -> bool:
        return bool(self.next_max_id)


@dataclass(slots=True)
class SearchPage:
    """tag 检索的一页结果。"""

    posts: list[WeiboPost] = field(default_factory=list)
    page: int = 1
    has_more: bool = True
