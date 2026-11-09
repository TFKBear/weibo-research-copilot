"""按微博平台特性定制的采集算法：时间窗、排序、翻页与停止条件。

为什么单独一个模块，而不是塞进 ``collect.py``：

微博评论区有三种**互不兼容**的读取方式，选错一种，样本代表性和研究问题就对不上：

1. **热门流**（移动端现成可用）——按热度返回，**时间不单调**。
   这意味着「翻到某天就停」是不可能的：你无法从"这条比窗口更早"推出
   "后面也不会有窗口内的评论"。热度流的停止条件只能是条数、页数或预算。
2. **按时间流**（仅网页端支持 ``sort=time``）——时间单调递减，
   于是「翻到早于窗口起点的评论就停」是**可靠**的提前停止条件，
   而且能说清"窗口已被完整覆盖"。
3. **二级评论**（楼中楼）——递归抓取，量级爆炸，默认关闭。

把这些判断集中在 ``plan_comment_collection()`` 里，好处是：
- 停止条件由"能不能证明窗口已覆盖"决定，而不是由"翻够多少页"决定；
- 每一种降级（排序没生效、时间不单调）都会变成一句**能写进论文的说明**，
  而不是悄悄变换抽样方式；
- 同一套判断 CLI 与界面共用，不会出现"界面按时间、命令行按热度"这种分叉。

平台事实（实测与接口文档，改动需重测）：

- 移动端 ``comments/hotflow``：每页约 10 条，``max_id`` 递推，**忽略 sort 参数**；
- 网页端 ``comments/hotflow``：每页约 20 条，支持 ``sort=hot|time``；
- 评论区 ``total_number`` 是平台侧的**总数**（含折叠评论），与实际可取条数不等；
- 检索页 ``containerid`` 的 ``type=1`` 是综合 Tab，其余 Tab 未验证，不臆测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Iterable, Sequence

#: 微博时间一律按北京时间理解。评论对象里的 ``created_at`` 是**带时区**的
#: （``parse_created_at`` 会挂上 CST），而界面上填的时间窗是朴素 datetime——
#: 直接比较会抛 ``TypeError: can't compare offset-naive and offset-aware``。
#: 统一在这里剥掉时区再比：全项目都是北京时间，剥掉不会引入歧义。
CST = timezone(timedelta(hours=8), "CST")


def _naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(CST).replace(tzinfo=None) if value.tzinfo else value

#: 各后端单页条数（实测）。用于估算页数与请求量，宁可按最小页估。
PAGE_SIZE_HINTS: dict[str, int] = {
    "mobile_hotflow": 10,
    "web_hotflow": 20,
}
DEFAULT_PAGE_SIZE = 10

#: 支持的排序口径。``hot`` = 平台热门流；``time`` = 按时间倒序。
SORT_LABELS = {"hot": "热度优先", "time": "时间优先"}


# --------------------------------------------------------------------------- #
# 时间窗
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TimeWindow:
    """评论时间窗，闭区间 ``[since, until]``。

    设计要点：**时间未知的评论不静默丢弃**。
    接口对"刚刚/今天"等相对时间解析失败时 ``created_at`` 会是 ``None``，
    这类评论通常是最新的、恰恰是研究者最想看的。悄悄丢掉它们会系统性
    偏向旧样本，所以这里选择保留，并单独计数（``unknown_time``），
    让研究者自己决定要不要在分析阶段剔除。
    """

    since: datetime | None = None
    until: datetime | None = None

    @property
    def active(self) -> bool:
        return self.since is not None or self.until is not None

    def describe(self, fmt: str = "%Y-%m-%d %H:%M") -> str:
        if not self.active:
            return "不限时间"
        left = self.since.strftime(fmt) if self.since else "不限"
        right = self.until.strftime(fmt) if self.until else "不限"
        return f"{left} ～ {right}"

    def accepts(self, comment: Any) -> bool:
        """该评论是否落在窗口内。

        日粒度（``05-12`` 这类）按"日期与窗口有交集"判定：
        把日粒度硬当成 00:00 会让窗口最后一天的评论被整片漏掉。
        """
        when = _naive(getattr(comment, "created_at", None))
        if when is None:
            return True  # 时间未知：保留，另行计数
        precision = getattr(comment, "created_at_precision", None)
        day_level = precision is not None and getattr(precision, "value", "") == "day"
        since, until = _naive(self.since), _naive(self.until)
        if day_level:
            day: date = when.date()
            if since is not None and day < since.date():
                return False
            if until is not None and day > until.date():
                return False
            return True
        if since is not None and when < since:
            return False
        if until is not None and when > until:
            return False
        return True

    def older_than_start(self, when: datetime | None) -> bool:
        """``when`` 是否已早于窗口起点（仅对**时间单调递减**的流有意义）。"""
        when, since = _naive(when), _naive(self.since)
        if when is None or since is None:
            return False
        return when < since

    def newer_than_end(self, when: datetime | None) -> bool:
        when, until = _naive(when), _naive(self.until)
        if when is None or until is None:
            return False
        return when > until

    def to_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "active": self.active,
            "label": self.describe(),
        }


# --------------------------------------------------------------------------- #
# 采集方案
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CollectionPlan:
    """一篇帖子的评论采集方案：目标、能否提前停止、页数上限、以及要披露的话。"""

    target: int
    window: TimeWindow
    sort: str
    sort_applied: bool
    monotonic_time: bool
    can_stop_early: bool
    page_cap: int
    page_size: int = DEFAULT_PAGE_SIZE
    clamped_by_budget: bool = False
    """页数上限是否被剩余请求预算压低。用来说明"为什么提前停了"。"""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "window": self.window.to_dict(),
            "sort": self.sort,
            "sort_applied": self.sort_applied,
            "monotonic_time": self.monotonic_time,
            "can_stop_early": self.can_stop_early,
            "page_cap": self.page_cap,
            "page_size": self.page_size,
            "clamped_by_budget": self.clamped_by_budget,
            "notes": list(self.notes),
        }


def plan_comment_collection(
    *,
    target: int,
    window: TimeWindow | None = None,
    sort: str = "hot",
    sort_applied: bool = True,
    max_scan_pages: int | None = None,
    remaining_budget: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> CollectionPlan:
    """生成采集方案。纯函数：同样的输入永远给出同样的停止条件。

    Args:
        target: **窗口内**的目标条数。注意语义：不是"取 target 条再筛窗口"，
            那样窗口越大、越容易只剩几条可用。
        window: 时间窗；未设置时退化为"取前 target 条"。
        sort: ``hot`` / ``time``。
        sort_applied: 后端是否真的接受了排序参数（移动端会忽略）。
        max_scan_pages: 扫描页数上限；缺省按目标估算。
        remaining_budget: 剩余请求预算，用于把页数上限压到不会撞预算的程度。
        page_size: 单页条数，用于估算。

    Returns:
        ``CollectionPlan``，其中 ``notes`` 是**直接可以抄进论文方法部分**的说明。
    """
    window = window or TimeWindow()
    page_size = max(1, int(page_size))
    effective_sort = sort if sort_applied else "platform"
    monotonic = effective_sort == "time"

    notes: list[str] = []
    if not sort_applied and sort != "platform":
        notes.append(
            f"当前评论后端忽略排序参数，实际返回的是平台默认流（热门流），"
            f"「{SORT_LABELS.get(sort, sort)}」没有生效。"
        )

    if not window.active:
        # 无窗口：与旧行为一致，按条数翻页
        pages = max(3, (int(target) // page_size) + 3)
        if max_scan_pages is not None:
            pages = max(1, int(max_scan_pages))
        clamped = False
        if remaining_budget is not None and remaining_budget < pages:
            pages = max(1, int(remaining_budget))
            clamped = True
        return CollectionPlan(
            target=int(target) or 1,
            window=window,
            sort=effective_sort,
            sort_applied=sort_applied,
            monotonic_time=monotonic,
            can_stop_early=False,
            page_cap=pages,
            page_size=page_size,
            clamped_by_budget=clamped,
            notes=notes,
        )

    # 有窗口：停止条件取决于时间是否单调
    can_stop_early = monotonic and window.since is not None
    if can_stop_early:
        notes.append(
            "按时间倒序且时间单调，翻到早于窗口起点的评论即停止，"
            "因此可以宣称窗口已被完整覆盖。"
        )
    else:
        notes.append(
            "热门流的时间不单调，无法从「这条比窗口更早」推出「后面没有窗口内的评论」，"
            "所以只能翻到窗口内满目标条数、或触及页数上限。"
            "若因页数上限停止，窗口内可能仍有未采集到的评论。"
        )

    # 窗口模式：扫描页数不能只按目标算——窗口外的评论也要占页。
    # 缺省给 5 倍余量，并受预算硬约束。
    if max_scan_pages is not None:
        pages = max(1, int(max_scan_pages))
    else:
        base = max(3, (int(target) // page_size) + 3)
        pages = base * 5 if not can_stop_early else base * 5
    clamped = False
    if remaining_budget is not None and remaining_budget < pages:
        pages = max(1, int(remaining_budget))
        clamped = True
        notes.append(
            f"剩余请求预算只够 {pages} 页，本次会按预算提前停止；"
            "窗口内可能仍有未采集到的评论。"
        )

    if window.since is not None and window.until is not None and window.since > window.until:
        notes.append("窗口起点晚于终点，不会采到任何评论——请检查日期是否填反。")

    return CollectionPlan(
        target=int(target) or 1,
        window=window,
        sort=effective_sort,
        sort_applied=sort_applied,
        monotonic_time=monotonic,
        can_stop_early=can_stop_early,
        page_cap=pages,
        page_size=page_size,
        clamped_by_budget=clamped,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# 窗口覆盖度
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class WindowCoverage:
    """一次采集实际覆盖到的时间范围——比"采了多少条"更能说明样本代表性。"""

    window: TimeWindow
    in_window: int = 0
    out_of_window: int = 0
    unknown_time: int = 0
    covered_from: datetime | None = None
    covered_to: datetime | None = None
    complete: bool = False
    """是否可宣称"窗口已完整覆盖"。只有按时间单调流、并停在窗口起点时才为 True。"""

    def observe(self, comment: Any, *, in_window: bool) -> None:
        if in_window:
            self.in_window += 1
            when = getattr(comment, "created_at", None)
            if when is None:
                self.unknown_time += 1
                return
            if self.covered_from is None or when < self.covered_from:
                self.covered_from = when
            if self.covered_to is None or when > self.covered_to:
                self.covered_to = when
        else:
            self.out_of_window += 1

    def render(self) -> str:
        if not self.window.active:
            return f"未设时间窗；窗口内 {self.in_window} 条（等于全部采集量）"
        parts = [
            f"窗口 {self.window.describe()}",
            f"窗口内 {self.in_window} 条",
            f"窗口外跳过 {self.out_of_window} 条",
        ]
        if self.unknown_time:
            parts.append(f"时间未知 {self.unknown_time} 条（已保留，未计入覆盖范围）")
        if self.covered_from and self.covered_to:
            parts.append(
                f"实际覆盖 {self.covered_from:%Y-%m-%d %H:%M} ～ "
                f"{self.covered_to:%Y-%m-%d %H:%M}"
            )
        parts.append("窗口已完整覆盖" if self.complete else "窗口未完整覆盖（见停止原因）")
        return "；".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "in_window": self.in_window,
            "out_of_window": self.out_of_window,
            "unknown_time": self.unknown_time,
            "covered_from": self.covered_from.isoformat() if self.covered_from else None,
            "covered_to": self.covered_to.isoformat() if self.covered_to else None,
            "complete": self.complete,
            "text": self.render(),
        }


# --------------------------------------------------------------------------- #
# 输入解析（界面与命令行共用）
# --------------------------------------------------------------------------- #
TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d", "%m-%d %H:%M", "%m-%d")


def parse_window_time(text: str, *, end_of_day: bool = False, now: datetime | None = None) -> datetime | None:
    """把界面/命令行里的文本解析成时间点；空文本返回 ``None``。

    规则与界面提示一致，不做"聪明猜测"：
    - ``2026-09-01`` → 当天 00:00（作为终点时取 23:59:59）
    - ``2026-09-01 12:30`` → 精确到分
    - ``09-01`` → 视为**今年**，跨年时取最接近现在的年份（避免把 12 月的昨天算成明年）
    """
    raw = (text or "").strip()
    if not raw:
        return None
    now = now or datetime.now()
    # 参数里存的是 ISO 时间（带 T 和时区），读回来必须能解析；
    # 只填日期的简写仍走下面的"友好格式"分支，以保留 end_of_day 语义。
    if "T" in raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=CST)
    for fmt in TIME_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=now.year)
            # 若解析结果比现在晚半年以上，几乎一定是跨年（如 1 月填 12-31）
            if parsed - now > timedelta(days=180):
                parsed = parsed.replace(year=now.year - 1)
        if "%H" not in fmt and end_of_day:
            parsed = datetime.combine(parsed.date(), dt_time(23, 59, 59))
        return parsed.replace(tzinfo=CST)  # 统一按北京时间理解
    raise ValueError(
        f"看不懂这个时间「{raw}」。请用 2026-09-01 或 2026-09-01 12:30 这样的格式。"
    )


def recent_days(days: int, *, now: datetime | None = None, end_of_day: bool = True) -> tuple[datetime, datetime]:
    """「最近 N 天」快捷窗口：含今天在内的 N 个自然日。"""
    now = now or datetime.now()
    days = max(1, int(days))
    start = datetime.combine((now - timedelta(days=days - 1)).date(), dt_time(0, 0))
    end = datetime.combine(now.date(), dt_time(23, 59, 59)) if end_of_day else now
    return start.replace(tzinfo=CST), end.replace(tzinfo=CST)


def page_oldest(page_comments: Iterable[Any]) -> datetime | None:
    """一页里最早的时间（忽略时间未知的条目）。"""
    values: Sequence[datetime] = [
        c.created_at for c in page_comments if getattr(c, "created_at", None) is not None
    ]
    return min(values) if values else None


def page_newest(page_comments: Iterable[Any]) -> datetime | None:
    values: Sequence[datetime] = [
        c.created_at for c in page_comments if getattr(c, "created_at", None) is not None
    ]
    return max(values) if values else None
