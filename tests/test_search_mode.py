"""检索口径：话题聚合 vs 关键词全文。

这两者拿到的**不是同一批帖子**：
- 话题聚合 ``q=#话题#``：只有主动给帖子打上该话题的人；
- 关键词全文 ``q=关键词``：正文里提到该词就算，混着"顺口提一句"的帖子。

混为一谈的后果不是"稍微不准"，而是研究结论站不住：
你在报告里写"该话题的参与者如何如何"，样本里却有一半只是路过提及的人。
所以口径必须显式可选、落库、并在报告里写出来。
"""

from __future__ import annotations

import pytest

from wrc.errors import ConfigProblem
from wrc.pipeline import CollectOptions
from wrc.weibo.endpoints import (
    SEARCH_MODES,
    search_container_id,
    search_query,
    search_request,
)


def test_topic_mode_wraps_hashes():
    assert search_query("延迟退休", "topic") == "#延迟退休#"
    assert search_query("#延迟退休#", "topic") == "#延迟退休#"


def test_keyword_mode_keeps_bare_word():
    assert search_query("延迟退休", "keyword") == "延迟退休"
    # 用户习惯性带了井号也不该把井号带进全文检索
    assert search_query("#延迟退休#", "keyword") == "延迟退休"


def test_keyword_mode_rejects_empty():
    with pytest.raises(ConfigProblem):
        search_query("  ##  ", "keyword")


def test_container_id_differs_between_modes():
    topic = search_container_id("延迟退休", mode="topic")
    keyword = search_container_id("延迟退休", mode="keyword")
    assert "q=#延迟退休#" in topic
    assert keyword.endswith("q=延迟退休")
    assert topic != keyword


def test_search_request_label_says_which_mode():
    topic = search_request("延迟退休", mode="topic")
    keyword = search_request("延迟退休", mode="keyword")
    assert "全文" not in topic.label
    assert "全文" in keyword.label


def test_both_modes_are_documented():
    assert set(SEARCH_MODES) == {"topic", "keyword"}


def test_collect_options_default_is_topic():
    opts = CollectOptions(tag="#t#")
    assert opts.search_mode == "topic"


def test_cli_search_mode_flag_reaches_options():
    from wrc.cli import _collect_options, build_parser

    args = build_parser().parse_args(
        ["run", "--tag", "延迟退休", "--select", "1", "--search-mode", "keyword"]
    )
    assert _collect_options(args).search_mode == "keyword"


def test_search_passes_mode_to_client():
    """口径必须真的进到请求里，不能只落在参数对象上。"""
    import asyncio

    from wrc.fetching import MockFetcher
    from wrc.weibo.collect import WeiboClient
    from wrc.weibo.ratelimit import RateLimiter, RequestBudget

    seen: list[str] = []

    def router(request):
        seen.append(str(request.params["containerid"]))
        return {
            "ok": 1,
            "data": {
                "cards": [
                    {
                        "card_type": 9,
                        "mblog": {
                            "id": "m1", "bid": "b1", "text": "帖子",
                            "created_at": "2026-09-20 10:00",
                            "user": {"id": "u1", "screen_name": "某人"},
                            "reposts_count": 1, "comments_count": 2, "attitudes_count": 3,
                        },
                    }
                ]
            },
        }

    async def no_sleep(_seconds):
        return None

    client = WeiboClient(
        MockFetcher(router),
        RateLimiter(delay_range=(0.0, 0.0), budget=RequestBudget(limit=10),
                    sleep=no_sleep, rng=None),
        max_retries=0,
    )

    async def scenario():
        return await client.search_posts("延迟退休", limit=1, mode="keyword")

    asyncio.run(scenario())
    assert seen and "q=延迟退休" in seen[0] and "#延迟退休#" not in seen[0]
