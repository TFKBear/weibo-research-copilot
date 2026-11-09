"""评论时间窗采集（按微博平台特性定制的采集算法）。

测试要守住的几条判断：

1. **窗口内目标**：设了时间窗后，`target` 指"窗口内凑够多少条"，
   而不是"先取 target 条再筛窗口"——后者会让窗口越大、可用样本越少。
2. **能提前停 vs 不能提前停**：只有时间单调递减的流（网页端 sort=time）
   才能在越过窗口起点时停止。热门流不能，否则会漏掉窗口内的评论，
   而研究者会以为采全了。
3. **停不下来要说出来**：`coverage.complete` 与 `plan.notes` 必须能直接
   抄进论文，说明"窗口覆盖到哪、为什么停"。
4. **时间未知的评论不静默丢弃**：解析失败的"刚刚/今天"是最新的评论，
   丢掉会让样本系统性偏向旧数据。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from wrc.errors import ConfigProblem
from wrc.fetching import MockFetcher
from wrc.pipeline import CollectOptions
from wrc.weibo.collect import WeiboClient
from wrc.weibo.ratelimit import RateLimiter, RequestBudget
from wrc.weibo.window import (
    CST,
    TimeWindow,
    WindowCoverage,
    page_newest,
    page_oldest,
    parse_window_time,
    plan_comment_collection,
    recent_days,
)

SINCE = datetime(2026, 9, 15, 0, 0)
UNTIL = datetime(2026, 9, 30, 23, 59)


async def _no_sleep(_seconds: float) -> None:
    return None


def make_client(router, *, budget: int = 1000) -> WeiboClient:
    limiter = RateLimiter(
        delay_range=(0.0, 0.0),
        budget=RequestBudget(limit=budget),
        sleep=_no_sleep,
        rng=None,
    )
    return WeiboClient(MockFetcher(router), limiter, max_retries=0)


# --------------------------------------------------------------------------- #
# 报文构造
# --------------------------------------------------------------------------- #
def _web_comment(cid: str, created: str, text: str = "评论") -> dict:
    return {
        "id": cid,
        "text": text,
        "created_at": created,
        "likes_count": 1,
        "total_number": 0,
        "user": {"id": f"u-{cid}", "screen_name": "某人", "followers_count": 100},
    }


def _mobile_payload(items: list[dict], *, max_id: int | None, total: int = 99) -> dict:
    data: dict = {"data": items, "total_number": total, "max_id_type": 0}
    if max_id is not None:
        data["max_id"] = max_id
    return {"ok": 1, "data": data}


def _web_payload(items: list[dict], *, max_id: int | None) -> dict:
    body: dict = {"ok": 1, "data": items}
    if max_id is not None:
        body["max_id"] = max_id
    return body


def _mobile_pages_router(pages: list[list[dict]]):
    """按 max_id 顺序返回移动端分页报文。"""

    def router(request):
        cursor = int(request.params.get("max_id") or 0)
        index = 0 if cursor == 0 else cursor - 1
        if index >= len(pages):
            return _mobile_payload([], max_id=None)
        next_id = index + 2 if index + 1 < len(pages) else None
        return _mobile_payload(pages[index], max_id=next_id)

    return router


def _web_pages_router(pages: list[list[dict]]):
    def router(request):
        cursor = int(request.params.get("max_id") or 0)
        index = 0 if cursor == 0 else cursor - 1
        if index >= len(pages):
            return _web_payload([], max_id=None)
        next_id = index + 2 if index + 1 < len(pages) else None
        return _web_payload(pages[index], max_id=next_id)

    return router


# --------------------------------------------------------------------------- #
# TimeWindow
# --------------------------------------------------------------------------- #
def test_time_window_accepts_inside_and_rejects_outside():
    window = TimeWindow(since=SINCE, until=UNTIL)
    inside = SimpleNamespace(created_at=datetime(2026, 9, 20, 8, 0), created_at_precision=None)
    before = SimpleNamespace(created_at=datetime(2026, 9, 1, 8, 0), created_at_precision=None)
    after = SimpleNamespace(created_at=datetime(2026, 10, 2, 8, 0), created_at_precision=None)

    assert window.accepts(inside) is True
    assert window.accepts(before) is False
    assert window.accepts(after) is False


def test_time_window_keeps_unknown_time_comments():
    """时间解析失败不能静默丢——那是最新评论，丢了会让样本偏旧。"""
    window = TimeWindow(since=SINCE, until=UNTIL)
    unknown = SimpleNamespace(created_at=None, created_at_precision=None)
    assert window.accepts(unknown) is True


def test_time_window_day_precision_uses_date_overlap():
    """日粒度（05-12 这种）不能按 00:00 卡，否则窗口最后一天会被整片漏掉。"""
    window = TimeWindow(
        since=datetime(2026, 9, 15, 14, 0), until=datetime(2026, 9, 15, 18, 0)
    )
    day_level = SimpleNamespace(
        created_at=datetime(2026, 9, 15, 0, 0),
        created_at_precision=SimpleNamespace(value="day"),
    )
    assert window.accepts(day_level) is True


def test_inactive_window_accepts_everything():
    window = TimeWindow()
    assert window.active is False
    assert window.accepts(SimpleNamespace(created_at=datetime(2000, 1, 1))) is True
    assert window.describe() == "不限时间"


def test_window_describe_is_readable():
    window = TimeWindow(since=SINCE, until=UNTIL)
    text = window.describe()
    assert "2026-09-15 00:00" in text and "2026-09-30 23:59" in text


# --------------------------------------------------------------------------- #
# 采集方案
# --------------------------------------------------------------------------- #
def test_plan_hot_flow_cannot_stop_early_and_says_so():
    plan = plan_comment_collection(
        target=50, window=TimeWindow(since=SINCE, until=UNTIL), sort="hot",
        sort_applied=False,
    )
    assert plan.can_stop_early is False
    assert plan.monotonic_time is False
    joined = " ".join(plan.notes)
    assert "忽略排序参数" in joined
    assert "热门流的时间不单调" in joined
    assert "未采集到的评论" in joined


def test_plan_time_flow_can_stop_early_and_promises_coverage():
    plan = plan_comment_collection(
        target=50, window=TimeWindow(since=SINCE, until=UNTIL), sort="time",
        sort_applied=True,
    )
    assert plan.can_stop_early is True
    assert plan.monotonic_time is True
    assert any("完整覆盖" in note for note in plan.notes)


def test_plan_clamps_pages_by_remaining_budget():
    plan = plan_comment_collection(
        target=200, window=TimeWindow(since=SINCE), sort="hot", max_scan_pages=40,
        remaining_budget=5,
    )
    assert plan.page_cap == 5
    assert plan.clamped_by_budget is True
    assert any("预算只够" in note for note in plan.notes)


def test_plan_without_window_matches_old_page_estimate():
    plan = plan_comment_collection(target=50, sort="hot", page_size=10)
    assert plan.can_stop_early is False
    assert plan.page_cap == max(3, 50 // 10 + 3)


# --------------------------------------------------------------------------- #
# 采集：时间单调的流可以提前停止
# --------------------------------------------------------------------------- #
def test_time_ordered_flow_stops_when_passing_window_start():
    pages = [
        [
            _web_comment("c1", "2026-09-20 10:00:00"),
            _web_comment("c2", "2026-09-18 10:00:00"),
            _web_comment("c3", "2026-09-16 10:00:00"),
        ],
        [
            _web_comment("c4", "2026-09-10 10:00:00"),
            _web_comment("c5", "2026-09-08 10:00:00"),
        ],
    ]
    client = make_client(_web_pages_router(pages))

    async def scenario():
        return await client.collect_comments(
            "m1", target=50, backend="web_build_comments", sort="time",
            since=SINCE, until=UNTIL,
        )

    result = asyncio.run(scenario())
    assert result.count == 3
    assert result.stopped_reason == "reached_window_start"
    assert result.window["in_window"] == 3
    assert result.window["out_of_window"] == 2
    assert result.window["complete"] is True
    assert result.pages_fetched == 2


def test_hot_flow_cannot_stop_early_and_reports_incomplete_coverage():
    """热门流：窗口外的评论夹在窗口内之间，不能因为看到旧的就停。"""
    pages = [
        [
            _mobile_comment("c1", "2026-09-20 10:00:00"),
            _mobile_comment("c2", "2026-09-01 10:00:00"),  # 窗口外，但不能据此停止
        ],
        [
            _mobile_comment("c3", "2026-09-22 10:00:00"),
            _mobile_comment("c4", "2026-08-01 10:00:00"),  # 又是窗口外
        ],
    ]
    client = make_client(_mobile_pages_router(pages))

    async def scenario():
        return await client.collect_comments(
            "m1", target=2, backend="mobile_hotflow", sort="time",  # 排序会被忽略
            since=SINCE, until=UNTIL,
        )

    result = asyncio.run(scenario())
    assert result.count == 2
    assert result.stopped_reason == "reached_target"
    assert result.sort_applied is False
    # 关键：不能因为"看到 9-01 的评论"就宣布窗口采完了
    assert result.window["complete"] is False
    assert result.window["out_of_window"] == 2
    assert result.pages_fetched == 2


def _mobile_comment(cid: str, created: str, text: str = "评论") -> dict:
    return {
        "id": cid,
        "text": text,
        "created_at": created,
        "like_counts": 3,
        "total_number": 0,
        "user": {"id": f"u-{cid}", "screen_name": "某人", "followers_count": 66},
    }


def test_window_keeps_unknown_time_and_counts_it():
    items = [
        _mobile_comment("c1", "2026-09-20 10:00:00"),
        {"id": "c2", "text": "没有时间", "user": {"id": "u2"}},
    ]
    client = make_client(lambda request: _mobile_payload(items, max_id=None))

    async def scenario():
        return await client.collect_comments(
            "m1", target=10, backend="mobile_hotflow", since=SINCE, until=UNTIL,
        )

    result = asyncio.run(scenario())
    assert result.count == 2, "时间未知的评论必须保留"
    assert result.window["unknown_time"] == 1


def test_window_excludes_comments_after_until():
    items = [
        _mobile_comment("c1", "2026-09-20 10:00:00"),
        _mobile_comment("c2", "2026-10-05 10:00:00"),
    ]
    client = make_client(lambda request: _mobile_payload(items, max_id=None))

    async def scenario():
        return await client.collect_comments(
            "m1", target=10, backend="mobile_hotflow", since=SINCE, until=UNTIL,
        )

    result = asyncio.run(scenario())
    assert result.count == 1
    assert result.window["out_of_window"] == 1


def test_reversed_window_is_rejected_with_explanation():
    client = make_client(lambda request: _mobile_payload([], max_id=None))

    async def scenario():
        return await client.collect_comments(
            "m1", target=5, since=UNTIL, until=SINCE,
        )

    with pytest.raises(ConfigProblem) as excinfo:
        asyncio.run(scenario())
    assert "起点晚于终点" in str(excinfo.value)


def test_plan_is_recorded_on_result():
    items = [_mobile_comment("c1", "2026-09-20 10:00:00")]
    client = make_client(lambda request: _mobile_payload(items, max_id=None))

    async def scenario():
        return await client.collect_comments(
            "m1", target=5, backend="mobile_hotflow", since=SINCE, until=UNTIL,
        )

    result = asyncio.run(scenario())
    assert result.plan["window"]["active"] is True
    assert result.plan["sort_applied"] is False
    assert result.plan["notes"], "采集口径说明不能为空"


# --------------------------------------------------------------------------- #
# 输入解析
# --------------------------------------------------------------------------- #
def test_parse_window_time_date_only():
    assert parse_window_time("2026-09-01") == datetime(2026, 9, 1, 0, 0, tzinfo=CST)
    assert parse_window_time("2026-09-01", end_of_day=True) == datetime(
        2026, 9, 1, 23, 59, 59, tzinfo=CST
    )


def test_parse_window_time_with_minutes():
    assert parse_window_time("2026-09-01 14:30") == datetime(2026, 9, 1, 14, 30, tzinfo=CST)
    assert parse_window_time("2026/09/01 14:30") == datetime(2026, 9, 1, 14, 30, tzinfo=CST)


def test_parse_window_time_empty_is_none():
    assert parse_window_time("") is None
    assert parse_window_time("   ") is None


def test_parse_window_time_rejects_garbage_with_hint():
    with pytest.raises(ValueError) as excinfo:
        parse_window_time("上周三")
    assert "2026-09-01" in str(excinfo.value)


def test_parse_window_time_infers_year_for_month_day():
    now = datetime(2026, 10, 26, 12, 0)
    parsed = parse_window_time("09-20", now=now)
    assert parsed is not None and parsed.year == 2026
    # 1 月填 12-31 应理解为去年
    parsed_ny = parse_window_time("12-31", now=datetime(2026, 1, 2, 12, 0))
    assert parsed_ny is not None and parsed_ny.year == 2025


def test_recent_days_includes_today():
    now = datetime(2026, 10, 26, 12, 0)
    since, until = recent_days(7, now=now)
    assert since.date() == datetime(2026, 10, 20).date()
    assert (until - since).days == 6


# --------------------------------------------------------------------------- #
# 覆盖度渲染
# --------------------------------------------------------------------------- #
def test_coverage_render_mentions_incomplete_window():
    coverage = WindowCoverage(window=TimeWindow(since=SINCE, until=UNTIL))
    coverage.observe(SimpleNamespace(created_at=datetime(2026, 9, 20)), in_window=True)
    coverage.observe(SimpleNamespace(created_at=datetime(2026, 9, 1)), in_window=False)
    text = coverage.render()
    assert "窗口内 1 条" in text
    assert "窗口外跳过 1 条" in text
    assert "未完整覆盖" in text


def test_page_oldest_ignores_unknown_time():
    page = [
        SimpleNamespace(created_at=None),
        SimpleNamespace(created_at=datetime(2026, 9, 18)),
        SimpleNamespace(created_at=datetime(2026, 9, 20)),
    ]
    assert page_oldest(page) == datetime(2026, 9, 18)
    assert page_newest(page) == datetime(2026, 9, 20)


# --------------------------------------------------------------------------- #
# 参数落库
# --------------------------------------------------------------------------- #
def test_collect_options_carry_window():
    opts = CollectOptions(tag="#t#", since=SINCE, until=UNTIL)
    assert opts.since == SINCE
    assert opts.until == UNTIL
