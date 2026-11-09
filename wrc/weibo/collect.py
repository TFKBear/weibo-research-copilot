"""采集协议层：把「检索 / 评论 / 二级评论」封装成带频率纪律的操作。

这一层是限速、退避、预算的唯一执行点。任何绕过它直接调 ``fetcher`` 的代码
都会破坏频率纪律，因此所有采集函数都必须经过 ``WeiboClient.get_json```。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ..errors import CommentSourceEmpty, ConfigProblem, EmptyResult, ErrorKind, WrcError
from ..fetching import Fetcher
from .comment_backends import AUTO_BACKENDS, get_backend
from .endpoints import (
    Request,
    config_request,
    search_request,
    subcomments_request,
)
from .models import (
    CommentLevel,
    WeiboComment,
    WeiboPost,
    SearchPage,
)
from .parsing import (
    is_logged_in,
    parse_comments_page,
    parse_search_page,
)
from .ratelimit import RateLimiter
from .window import (
    DEFAULT_PAGE_SIZE,
    PAGE_SIZE_HINTS,
    TimeWindow,
    WindowCoverage,
    page_oldest,
    plan_comment_collection,
)


@dataclass(slots=True)
class CommentFetchResult:
    """一次评论采集的结果与「为什么停下来」。"""

    comments: list[WeiboComment] = field(default_factory=list)
    pages_fetched: int = 0
    total_number: int | None = None
    stopped_reason: str = "unknown"
    """停止原因。必须记录进研究日志：它决定了样本的代表性。"""

    backend: str = ""
    """实际供给数据的评论后端。换后端等于换抽样方式，必须可回查。"""

    sort: str = "hot"
    sort_applied: bool = False
    """排序是否真的被接口接受（移动端后端会忽略排序参数）。"""

    window: dict[str, Any] = field(default_factory=dict)
    """时间窗覆盖度（``WindowCoverage.to_dict()``）。用来说明样本代表性。"""

    plan: dict[str, Any] = field(default_factory=dict)
    """本次采用的采集方案（停止条件与理由），可原样写进方法部分。"""

    attempts: list[tuple[str, str]] = field(default_factory=list)
    """各后端的尝试结果，用于诊断「为什么最终用了/没用某个后端」。"""

    error_kind: str = ""
    error_message: str = ""

    @property
    def count(self) -> int:
        return len(self.comments)


class WeiboClient:
    """带频率纪律的微博采集客户端。"""

    def __init__(
        self,
        fetcher: Fetcher,
        limiter: RateLimiter,
        *,
        max_retries: int = 2,
        sleep: Callable[[float], Any] = asyncio.sleep,
        log: Any | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._limiter = limiter
        self._max_retries = max_retries
        self._sleep = sleep
        self._log = log
        self._notify = notify or (lambda _m: None)

    # ------------------------------------------------------------------ #
    @property
    def budget_used(self) -> int:
        return self._limiter.budget.used

    @property
    def budget_limit(self) -> int:
        return self._limiter.budget.limit

    async def get_json(self, request: Request) -> Any:
        """唯一的出站入口：限速 → 预算 → 重试/退避。"""
        attempt = 0
        while True:
            await self._limiter.wait_turn()  # 可能抛 BudgetExceeded
            if self._limiter.budget.should_warn():
                self._notify(
                    f"本次采集已使用 {self._limiter.budget.used}/"
                    f"{self._limiter.budget.limit} 次请求，接近安全阈值。"
                    "建议缩小范围或换时段，而不是提高上限。"
                )
            try:
                return await self._fetcher.get_json(request)
            except WrcError as exc:
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                seconds = await self._limiter.sleep_backoff(attempt)
                self._record(
                    "backoff",
                    label=request.label,
                    kind=exc.kind.value,
                    attempt=attempt + 1,
                    waited_s=seconds,
                )
                self._notify(
                    f"{exc.what_happened} 已暂停 {seconds:.0f} 秒后重试"
                    f"（第 {attempt + 1}/{self._max_retries} 次）。"
                )
                attempt += 1

    # ------------------------------------------------------------------ #
    async def check_login(self) -> bool:
        payload = await self.get_json(config_request())
        return is_logged_in(payload)

    async def search_page(
        self, tag: str, *, page: int = 1, tab_type: int = 1, mode: str = "topic"
    ) -> SearchPage:
        request = search_request(tag, page=page, tab_type=tab_type, mode=mode)
        payload = await self.get_json(request)
        result = parse_search_page(payload, page=page)
        self._record(
            "search_page",
            tag=tag,
            page=page,
            posts=len(result.posts),
            tab_type=tab_type,
            mode=mode,
        )
        return result

    async def search_posts(
        self,
        tag: str,
        *,
        limit: int = 20,
        max_pages: int = 3,
        order: str = "engagement",
        tab_type: int = 1,
        mode: str = "topic",
    ) -> list[WeiboPost]:
        """检索 tag 下的帖子。

        Args:
            order: ``platform`` 按接口返回顺序（相关度）；``engagement`` 按互动总量
                降序（**本地重排**）；``time`` 按发帖时间降序（**本地重排**）。
            mode: ``topic`` 话题聚合 / ``keyword`` 关键词全文检索。

        ⚠ 诚实性要求：``engagement`` / ``time`` 都是我们在本地对
        「相关度排序的前 N 条」再排序，**不等于**微博站内「热门/实时」的真实排序。
        研究日志会记录这一点，任何面向用户的说明也不得模糊它。
        """
        collected: list[WeiboPost] = []
        seen: set[str] = set()
        page = 1
        while len(collected) < limit and page <= max_pages:
            result = await self.search_page(tag, page=page, tab_type=tab_type, mode=mode)
            new = [post for post in result.posts if post.mid not in seen]
            if not new:
                break
            for post in new:
                seen.add(post.mid)
                collected.append(post)
            if not result.has_more:
                break
            page += 1

        if not collected:
            raise EmptyResult(detail=f"tag={tag}, pages_tried={max_pages}")

        ranked = _rank_posts(collected, order)
        self._record(
            "search_ranked",
            tag=tag,
            order=order,
            mode=mode,
            order_is_local_rerank=order in {"engagement", "time"},
            candidates=len(collected),
            returned=min(limit, len(ranked)),
        )
        return ranked[:limit]

    # ------------------------------------------------------------------ #
    async def collect_comments(
        self,
        post_mid: str,
        *,
        target: int = 50,
        include_subcomments: bool = False,
        subcomment_target: int = 10,
        backend: str = "auto",
        sort: str = "hot",
        expected_total: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_scan_pages: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> CommentFetchResult:
        """采集一篇帖子的评论。

        评论区有三件事必须诚实对待：

        1. **来源是一种抽样。** 不同后端的排序不同（热门流 / 按时间），
           拿到的样本因此不同。实际用的后端与排序都会写进研究日志。
        2. **「空」与「坏」必须分开。** 平台显示该帖有 N 条评论、接口却返回空，
           这是故障而不是结论。第 1 步实测就踩到了这个坑：接口返回 ``ok=1``
           但 ``data`` 为空，旧代码把它当成"平台没有更多评论"，
           于是研究会在空样本上继续跑清洗与情感分析，日志里还看不出异常。
           现在这种情况直接抛 ``CommentSourceEmpty``。
        3. **时间窗不是"多筛一步"。** 设了窗口后 ``target`` 的语义是
           **窗口内**要凑够多少条：采满即停，采不满就如实报告覆盖度。
           若先取 target 条再筛窗口，窗口越大越可能只剩几条可用，
           而研究者会以为自己采到了 target 条。

        Args:
            target: 目标评论数。设了时间窗时指**窗口内**的条数。
            include_subcomments: 是否顺带抓二级评论，默认关闭。
            backend: ``auto`` 依次尝试各后端；也可指定具体后端名用于复现。
            sort: ``hot`` / ``time``，仅网页端后端支持，移动端会忽略并如实记录。
            expected_total: 平台显示的评论总数，用于区分"空"与"坏"。
            since / until: 时间窗（闭区间）；``None`` 表示不限。
            max_scan_pages: 扫描页数上限；缺省按窗口与目标估算，并受预算约束。
        """
        if target <= 0:
            raise ValueError("target 必须为正整数")
        window = TimeWindow(since=since, until=until)
        if window.since and window.until and window.since > window.until:
            raise ConfigProblem(
                what_happened="时间窗的起点晚于终点，采不到任何评论。",
                detail=f"起点 {window.since:%Y-%m-%d %H:%M}，终点 {window.until:%Y-%m-%d %H:%M}",
            )

        candidates = AUTO_BACKENDS if backend == "auto" else (get_backend(backend),)
        if backend == "auto" and sort == "time":
            candidates = tuple(sorted(candidates, key=lambda b: not b.supports_sort))
        attempts: list[tuple[str, str]] = []
        chosen = None
        first_page = None

        for candidate in candidates:
            request = candidate.request(post_mid, max_id=0, max_id_type=0, sort=sort)
            try:
                payload = await self.get_json(request)
                page = parse_comments_page(payload, post_mid=post_mid)
            except WrcError as exc:
                attempts.append((candidate.key, f"{exc.kind.value}：{exc.what_happened}"))
                self._record(
                    "comments_backend_failed",
                    post_mid=post_mid,
                    backend=candidate.key,
                    kind=exc.kind.value,
                )
                if backend != "auto" or exc.kind not in {ErrorKind.API, ErrorKind.PARSE}:
                    raise
                continue

            if page.comments:
                chosen, first_page = candidate, page
                self._record(
                    "comments_backend_selected",
                    post_mid=post_mid,
                    backend=candidate.key,
                    sort=sort,
                    sort_applied=candidate.supports_sort,
                )
                break

            shape = _payload_shape(payload)
            attempts.append((candidate.key, f"空数据（{shape}）"))
            self._record(
                "comments_backend_empty",
                post_mid=post_mid,
                backend=candidate.key,
                shape=shape,
                total_number=page.total_number,
            )
            if backend != "auto":
                chosen, first_page = candidate, page
                break

        if chosen is None or first_page is None or not first_page.comments:
            if (expected_total is not None and expected_total > 0) or any(
                not outcome.startswith("空数据") for _, outcome in attempts
            ):
                raise CommentSourceEmpty(
                    post_mid=post_mid,
                    expected_total=expected_total,
                    attempts=tuple(attempts),
                )
            # 平台侧确实没有评论（或未提供总数且各后端一致为空）
            self._record(
                "comments_none",
                post_mid=post_mid,
                expected_total=expected_total,
                attempts=[name for name, _ in attempts],
            )
            return CommentFetchResult(
                stopped_reason="post_has_no_comments",
                backend=chosen.key if chosen else "",
                attempts=attempts,
            )

        result = CommentFetchResult(
            backend=chosen.key,
            attempts=attempts,
            sort=sort,
            sort_applied=chosen.supports_sort,
        )
        # 采集方案由 window 模块统一决定：能不能提前停、页数上限多少。
        # 放在"选定后端之后"是因为排序是否生效取决于后端。
        plan = plan_comment_collection(
            target=target,
            window=window,
            sort=sort,
            sort_applied=chosen.supports_sort,
            max_scan_pages=max_scan_pages,
            remaining_budget=self._limiter.budget.remaining,
            page_size=PAGE_SIZE_HINTS.get(chosen.key, DEFAULT_PAGE_SIZE),
        )
        result.plan = plan.to_dict()
        coverage = WindowCoverage(window=window)
        self._record(
            "collection_plan",
            post_mid=post_mid,
            backend=chosen.key,
            target=target,
            sort=sort,
            sort_applied=chosen.supports_sort,
            can_stop_early=plan.can_stop_early,
            page_cap=plan.page_cap,
            window=window.to_dict(),
        )

        seen_cids: set[str] = set()
        seen_cursors: set[tuple[int, int]] = set()
        max_pages = plan.page_cap
        page = first_page
        page_index = 0
        current_cursor = 0

        while True:
            page_index += 1
            result.pages_fetched = page_index
            if page.total_number is not None:
                result.total_number = page.total_number

            fresh = []
            for comment in page.comments:
                if comment.cid in seen_cids:
                    continue
                seen_cids.add(comment.cid)
                fresh.append(comment)
            for comment in fresh:
                accepted = coverage.window.accepts(comment)
                if accepted:
                    if len(result.comments) >= target:
                        continue
                    result.comments.append(comment)
                coverage.observe(comment, in_window=accepted)

            self._record(
                "comments_page",
                post_mid=post_mid,
                backend=chosen.key,
                page=page_index,
                got=len(page.comments),
                fresh=len(fresh),
                accumulated=len(result.comments),
                out_of_window=coverage.out_of_window,
                oldest=(page_oldest(fresh).isoformat() if page_oldest(fresh) else None),
                cursor=current_cursor,
            )

            if len(result.comments) >= target:
                result.stopped_reason = "reached_target"
                break

            if should_stop is not None and should_stop():
                result.stopped_reason = "user_stopped"
                break

            # 只有"时间单调递减"的流才能这样停：本页最早的一条已早于窗口起点，
            # 后面的页只会更早。热门流不能这么判断，否则会漏掉窗口内的评论。
            # 这个判断要放在 has_more 之前：两件事同时成立时，
            # "越过窗口起点"才是更有解释力的停止原因。
            if plan.can_stop_early and coverage.window.older_than_start(page_oldest(fresh)):
                result.stopped_reason = "reached_window_start"
                coverage.complete = True
                break

            if not page.has_more:
                result.stopped_reason = "no_more_from_platform"
                break

            cursor = int(page.next_max_id or 0)
            if cursor <= 0:
                result.stopped_reason = "empty_cursor"
                break
            max_id_type = int(page.next_max_id_type or 0)
            cursor_key = (cursor, max_id_type)
            if cursor_key in seen_cursors:
                # 接口退化时会重复返回同一个 cursor，继续翻只会空转并增加风控风险
                result.stopped_reason = "cursor_stalled"
                self._record("cursor_stalled", post_mid=post_mid, cursor=cursor)
                break
            if page_index >= max_pages:
                result.stopped_reason = (
                    "budget_near_limit" if plan.clamped_by_budget else "page_cap"
                )
                break
            seen_cursors.add(cursor_key)

            current_cursor = cursor
            request = chosen.request(
                post_mid, max_id=cursor, max_id_type=max_id_type, sort=sort
            )
            try:
                payload = await self.get_json(request)
                page = parse_comments_page(payload, post_mid=post_mid)
            except WrcError as exc:
                # 后续页失败不能丢弃此前成功页，也不能宣称已覆盖完整评论区。
                result.stopped_reason = "pagination_error"
                result.error_kind = exc.kind.value
                result.error_message = exc.what_happened
                self._record("comments_pagination_failed", post_mid=post_mid,
                             backend=chosen.key, cursor=cursor, kind=exc.kind.value,
                             retained=len(result.comments), detail=exc.detail)
                break

            if not page.comments and len(result.comments) < target:
                # 翻页中途突然空页：不是"翻完了"，而是这一页没给数据。
                # 保留已采数据，但如实记录，避免把残缺样本当成完整样本。
                result.stopped_reason = "empty_page_mid_stream"
                self._record(
                    "comments_pagination_empty",
                    post_mid=post_mid,
                    backend=chosen.key,
                    cursor=cursor,
                )
                break

        # 平台已经翻到底：只要时间单调，就等同于"窗口内的都看到了"
        if plan.monotonic_time and result.stopped_reason in {
            "no_more_from_platform", "empty_cursor"
        }:
            coverage.complete = True

        result.comments = result.comments[:target]
        result.window = coverage.to_dict()

        if include_subcomments:
            if not result.error_kind and result.stopped_reason != "user_stopped":
                try:
                    await self._attach_subcomments(
                        result, subcomment_target=subcomment_target, should_stop=should_stop
                    )
                except WrcError as exc:
                    result.stopped_reason = "subcomments_error"
                    result.error_kind = exc.kind.value
                    result.error_message = exc.what_happened

        self._record(
            "comments_done",
            post_mid=post_mid,
            target=target,
            collected=len(result.comments),
            stopped_reason=result.stopped_reason,
            error_kind=result.error_kind,
            error_message=result.error_message,
            backend=result.backend,
            sort=sort,
            sort_applied=result.sort_applied,
            window=result.window,
            sample_bias=(
                "热门评论流（高互动偏置）"
                if sort == "hot" or not result.sort_applied
                else "按时间（近端偏置）"
            ),
        )
        return result

    async def _attach_subcomments(
        self, result: CommentFetchResult, *, subcomment_target: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """为有跟评的一级评论补抓二级评论。

        二级评论会让数据量呈数量级增长，且每多抓一条就多一次风控暴露，
        因此这里采取保守策略：只对已有回复的一级评论抓前若干条。
        """
        parents = [c for c in list(result.comments)
                   if c.level == CommentLevel.PRIMARY and c.reply_count > 0]
        if not parents:
            self._record("subcomments_skipped", reason="no_parent_with_replies")
            return

        collected: list[WeiboComment] = []
        for parent in parents:
            want = min(subcomment_target, parent.reply_count)
            cursor, cursor_type = 0, 0
            got = 0
            seen: set[str] = set()
            cursors: set[tuple[int, int]] = set()
            while got < want:
                if should_stop is not None and should_stop():
                    result.stopped_reason = "user_stopped"
                    return
                if self._limiter.budget.remaining <= 0:
                    self._record("subcomments_budget_stop", parent_cid=parent.cid)
                    result.stopped_reason = "subcomments_error"
                    result.error_kind = "budget"
                    result.error_message = "二级评论采集达到请求上限，已保留此前数据。"
                    return
                if (cursor, cursor_type) in cursors:
                    result.stopped_reason = "subcomments_cursor_stalled"
                    return
                cursors.add((cursor, cursor_type))
                request = subcomments_request(
                    parent.cid, max_id=cursor, max_id_type=cursor_type
                )
                payload = await self.get_json(request)
                page = parse_comments_page(
                    payload,
                    post_mid=parent.post_mid,
                    level=CommentLevel.SECONDARY,
                    parent_cid=parent.cid,
                )
                fresh = []
                for comment in page.comments:
                    if comment.cid not in seen:
                        seen.add(comment.cid)
                        fresh.append(comment)
                if not fresh:
                    if page.has_more:
                        result.stopped_reason = "subcomments_cursor_stalled"
                    break
                collected.extend(fresh[: want - got])
                result.comments.extend(fresh[: want - got])
                got += len(fresh[: want - got])
                if not page.has_more or not page.next_max_id:
                    break
                cursor = int(page.next_max_id)
                if page.next_max_id_type is not None:
                    cursor_type = int(page.next_max_id_type)

        self._record(
            "subcomments_done",
            parents=len(parents),
            collected=len(collected),
        )

    # ------------------------------------------------------------------ #
    def _record(self, event: str, **fields: Any) -> None:
        if self._log is not None:
            try:
                self._log.record(event, **fields)
            except Exception:  # pragma: no cover
                pass


def _rank_posts(posts: list[WeiboPost], order: str) -> list[WeiboPost]:
    if order == "platform":
        return list(posts)
    if order == "engagement":
        return sorted(posts, key=lambda p: p.engagement, reverse=True)
    if order == "time":
        return sorted(
            posts,
            key=lambda p: (p.created_at is not None, p.created_at),
            reverse=True,
        )
    raise ValueError(f"未知排序方式: {order}")


def _payload_shape(payload: Any) -> str:
    """描述一个"成功的空响应"长什么样。

    诊断"为什么没数据"时，最有用的信息往往不是错误信息，而是
    ``ok=1`` 但结构不对时那个结构本身。只提取类型与键名，不打印正文。
    """
    if not isinstance(payload, dict):
        return f"顶层类型={type(payload).__name__}"
    parts = [f"ok={payload.get('ok')!r}"]
    message = payload.get("msg") or payload.get("message")
    if message:
        parts.append(f"msg={str(message)[:60]!r}")
    data = payload.get("data")
    if isinstance(data, list):
        parts.append(f"data=list(len={len(data)})")
    elif isinstance(data, dict):
        parts.append(f"data.keys={sorted(data)[:8]}")
    else:
        parts.append(f"data={type(data).__name__}")
    return "，".join(parts)
