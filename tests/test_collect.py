"""采集协议层测试：用合成接口数据跑通「检索 → 翻页 → 目标量」全流程。

用 MockFetcher 而不是打真实接口，原因有两个：
- 微博接口不稳定，测试不能依赖它；
- 翻页边界（cursor 停滞、目标量不足）靠真实数据很难稳定复现，
  而这些正是最容易写错、后果最严重（无限空转 / 少采）的地方。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wrc.errors import BudgetExceeded, CommentSourceEmpty, EmptyResult, RateLimited
from wrc.fetching import MockFetcher
from wrc.mockdata import build_scenario
from wrc.weibo.collect import WeiboClient, _rank_posts
from wrc.weibo.endpoints import Request
from wrc.weibo.ratelimit import RateLimiter, RequestBudget


async def _no_sleep(_seconds: float) -> None:
    return None


def make_client(router, *, budget: int = 1000, max_retries: int = 0) -> WeiboClient:
    limiter = RateLimiter(
        delay_range=(0.0, 0.0),
        budget=RequestBudget(limit=budget),
        sleep=_no_sleep,
        rng=None,
    )
    return WeiboClient(MockFetcher(router), limiter, max_retries=max_retries)


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def test_search_posts_returns_ranked_candidates():
    client = make_client(build_scenario(posts=4, comments_per_post=6, seed=3))

    async def scenario():
        return await client.search_posts("延迟退休", limit=4, order="engagement")

    posts = asyncio.run(scenario())
    assert len(posts) == 4
    engagements = [post.engagement for post in posts]
    assert engagements == sorted(engagements, reverse=True)
    assert all(post.mid for post in posts)


def test_search_posts_raises_empty_result_when_nothing_found():
    def empty_router(request: Request) -> Any:
        return {"ok": 1, "data": {"cards": []}}

    client = make_client(empty_router)

    async def scenario():
        return await client.search_posts("空话题", limit=5)

    with pytest.raises(EmptyResult):
        asyncio.run(scenario())


def test_rank_posts_time_order_puts_known_dates_first():
    from datetime import datetime

    from wrc.weibo.models import TimePrecision, WeiboPost

    def post(mid: str, day: int) -> WeiboPost:
        return WeiboPost(
            mid=mid,
            user_id="u",
            screen_name="s",
            text_raw="t",
            created_at=datetime(2025, 5, day, tzinfo=None),
            created_at_raw=f"05-{day:02d}",
            created_at_precision=TimePrecision.DAY,
        )

    ranked = _rank_posts([post("a", 10), post("b", 20), post("c", 15)], "time")
    assert [item.mid for item in ranked] == ["b", "c", "a"]


# --------------------------------------------------------------------------- #
# 评论
# --------------------------------------------------------------------------- #
def test_collect_comments_reaches_target_and_dedupes():
    client = make_client(build_scenario(posts=2, comments_per_post=45, seed=5))

    async def scenario():
        posts = await client.search_posts("延迟退休", limit=2)
        return await client.collect_comments(posts[0].mid, target=30)

    result = asyncio.run(scenario())
    assert result.count == 30
    assert result.stopped_reason == "reached_target"
    assert result.pages_fetched >= 2
    cids = [comment.cid for comment in result.comments]
    assert len(cids) == len(set(cids))
    assert result.total_number is not None
    assert result.total_number >= result.count


def test_collect_comments_stops_when_platform_runs_out():
    client = make_client(build_scenario(posts=1, comments_per_post=25, seed=9))

    async def scenario():
        posts = await client.search_posts("延迟退休", limit=1)
        return await client.collect_comments(posts[0].mid, target=500)

    result = asyncio.run(scenario())
    assert result.count == 25
    assert result.stopped_reason == "no_more_from_platform"


def test_collect_comments_detects_stalled_cursor():
    """cursor 原地不动时必须主动停手，否则会无限翻页并放大风控风险。"""

    def stalled_router(request: Request) -> Any:
        if "container/getIndex" in request.url:
            return {
                "ok": 1,
                "data": {
                    "cards": [
                        {
                            "card_type": 9,
                            "mblog": {
                                "mid": "m1",
                                "text": "帖子",
                                "created_at": "05-12",
                                "user": {"id": "u1"},
                            },
                        }
                    ]
                },
            }
        return {
            "ok": 1,
            "data": {
                "data": [
                    {
                        "id": "c1",
                        "text": "评论",
                        "created_at": "今天 10:00",
                        "user": {"id": "u2"},
                    }
                ],
                "max_id": 777,
                "max_id_type": 0,
            },
        }

    client = make_client(stalled_router)

    async def scenario():
        posts = await client.search_posts("话题", limit=1)
        return await client.collect_comments(posts[0].mid, target=50)

    result = asyncio.run(scenario())
    assert result.stopped_reason == "cursor_stalled"
    assert result.count == 1


def test_budget_pressure_stops_with_partial_data_instead_of_raising():
    """预算不足时**提前停止并保留已采数据**，而不是抛异常。

    旧行为：页数上限不受预算约束，翻到预算耗尽时由限流器抛 ``BudgetExceeded``。
    这看起来"把问题暴露出来了"，实际后果更糟——``pipeline`` 捕获 WrcError 后
    会跳过整篇帖子，那一篇已采到的评论全部不落库。
    现在的规则：预算压低页数上限，停止原因记为 ``budget_near_limit``，
    已采评论照常返回，研究日志里写明"是预算让它停的"。
    """
    client = make_client(
        build_scenario(posts=1, comments_per_post=80, seed=2), budget=3
    )

    async def scenario():
        posts = await client.search_posts("延迟退休", limit=1)
        return await client.collect_comments(posts[0].mid, target=50)

    result = asyncio.run(scenario())
    assert result.stopped_reason == "budget_near_limit"
    assert result.count >= 1, "已采到的评论不能因为预算不足被丢掉"
    assert result.plan["clamped_by_budget"] is True


# --------------------------------------------------------------------------- #
# 评论后端切换（第 1 步实机故障的回归测试）
# --------------------------------------------------------------------------- #
_WEB_COMMENT = {
    "id": "w1",
    "text": "来自网页端的评论",
    "created_at": "2025-05-12 10:00:00",
    "likes_count": 4,
    "total_number": 0,
    "user": {"id": "u9", "followers_count": 50},
}


def _comment_router(*, mobile_payload, web_payload):
    def router(request: Request) -> Any:
        if "comments/hotflow" in request.url:
            return mobile_payload
        if "statuses/buildComments" in request.url:
            return web_payload
        raise AssertionError(f"未预期的请求：{request.url}")

    return router


def test_collect_comments_falls_back_to_web_backend():
    """移动端返回 ok=1 但空数据时，必须换后端而不是报"没有评论"。"""
    client = make_client(
        _comment_router(
            mobile_payload={"ok": 1, "data": {"data": [], "max_id": 0, "max_id_type": 0}},
            web_payload={"ok": 1, "data": [_WEB_COMMENT], "max_id": 0, "max_id_type": 0},
        )
    )

    async def scenario():
        return await client.collect_comments("m1", target=10, expected_total=679)

    result = asyncio.run(scenario())
    assert result.backend == "web_build_comments"
    assert result.count == 1
    assert result.comments[0].cid == "w1"
    assert result.stopped_reason == "no_more_from_platform"


def test_collect_comments_raises_when_all_backends_empty_but_post_has_comments():
    """核心回归：空数据 + 平台显示有评论 = 故障，绝不当成空结果继续。"""
    empty = {"ok": 1, "data": {"data": [], "max_id": 0, "max_id_type": 0}}
    client = make_client(
        _comment_router(mobile_payload=empty, web_payload={"ok": 1, "data": []})
    )

    async def scenario():
        return await client.collect_comments("m1", target=10, expected_total=679)

    with pytest.raises(CommentSourceEmpty) as excinfo:
        asyncio.run(scenario())
    message = excinfo.value.what_happened
    assert "m1" in message
    assert "679" in message


def test_collect_comments_accepts_genuinely_empty_post():
    """平台显示 0 条评论时，空结果就是正确答案，不应报错。"""
    empty = {"ok": 1, "data": {"data": [], "max_id": 0, "max_id_type": 0}}
    client = make_client(
        _comment_router(mobile_payload=empty, web_payload={"ok": 1, "data": []})
    )

    async def scenario():
        return await client.collect_comments("m1", target=10, expected_total=0)

    result = asyncio.run(scenario())
    assert result.count == 0
    assert result.stopped_reason == "post_has_no_comments"


def test_retry_then_success_on_rate_limit():
    """第一次请求命中 432，退避后成功 —— 退避逻辑必须真的重试。"""

    state = {"calls": 0}

    def flaky_router(request: Request) -> Any:
        if "container/getIndex" in request.url:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RateLimited(status=432)
            return {
                "ok": 1,
                "data": {
                    "cards": [
                        {
                            "card_type": 9,
                            "mblog": {
                                "mid": "m1",
                                "text": "帖子",
                                "created_at": "05-12",
                                "user": {"id": "u1"},
                            },
                        }
                    ]
                },
            }
        return {"ok": 1, "data": {"data": [], "max_id": 0}}

    limiter = RateLimiter(
        delay_range=(0.0, 0.0),
        budget=RequestBudget(limit=50),
        sleep=_no_sleep,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )
    client = WeiboClient(MockFetcher(flaky_router), limiter, max_retries=1)

    async def scenario():
        return await client.search_posts("话题", limit=1)

    posts = asyncio.run(scenario())
    assert [post.mid for post in posts] == ["m1"]
    assert state["calls"] == 2
