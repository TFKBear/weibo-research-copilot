"""话题推荐测试（模块 E2）。

核心约束：**量级必须来自实测探测，不能来自模型**。
所以这里重点测两件事：
1. LLM 只被要求提标签名与理由，解析层不会把任何数字当成量级；
2. 量级描述完全由探测结果决定。
"""

from __future__ import annotations

from wrc.assistant.topic import (
    ProbeResult,
    TagCandidate,
    parse_tags,
    render_candidates,
)


def test_parse_tags_dict_shape():
    payload = {"tags": [{"tag": "#延迟退休#", "reason": "长期标签"}]}
    assert parse_tags(payload) == [("延迟退休", "长期标签")]


def test_parse_tags_accepts_plain_strings():
    assert parse_tags({"tags": ["小米SU7", " 汽车 "]} ) == [
        ("小米SU7", ""), ("汽车", "")
    ]


def test_parse_tags_dedupes_and_drops_empty():
    payload = {"tags": [{"tag": "A"}, {"tag": "A"}, {"tag": ""}, {"nope": 1}]}
    assert parse_tags(payload) == [("A", "")]


def test_parse_tags_ignores_model_supplied_counts():
    """模型即使自作主张给出数量，也不该被当成量级使用。"""
    payload = {"tags": [{"tag": "延迟退休", "reason": "x", "volume": 99999}]}
    tags = parse_tags(payload)
    assert tags == [("延迟退休", "x")]
    assert "99999" not in str(tags)


def test_probe_level_is_measured_not_guessed():
    assert ProbeResult(tag="a", ok=False).level == "无法探测"
    assert ProbeResult(tag="a", ok=True, posts=0).level == "未检索到帖子"
    assert ProbeResult(tag="a", ok=True, posts=3).level == "讨论很少"
    assert ProbeResult(tag="a", ok=True, posts=10).level == "有一定讨论"
    assert ProbeResult(tag="a", ok=True, posts=25).level == "讨论活跃"


def test_probe_describe_mentions_pages_and_engagement():
    probe = ProbeResult(
        tag="a", ok=True, posts=18, pages=2, median_engagement=640,
        latest_post_at="2026-10-24",
    )
    text = probe.describe()
    assert "18" in text and "2" in text and "640" in text and "2026-10-24" in text


def test_candidate_render_shows_reason_and_measurement():
    candidate = TagCandidate(
        tag="延迟退休",
        reason="政策讨论的长期标签",
        probe=ProbeResult(tag="延迟退休", ok=True, posts=12, pages=2,
                          samples=["关于延迟退休的讨论"]),
    )
    text = candidate.render(1)
    assert "延迟退休" in text
    assert "政策讨论的长期标签" in text
    assert "实测" in text
    assert "有一定讨论" in text


def test_candidate_without_probe_says_undetected():
    assert "未探测" in TagCandidate(tag="x", reason="y").render(1)


def test_render_candidates_states_the_caveat():
    """必须说明"实测"不等于话题总量——否则读者会误读。"""
    text = render_candidates([
        TagCandidate("a", "r", ProbeResult(tag="a", ok=True, posts=5))
    ])
    assert "不等于话题总量" in text
    assert "不是模型估计" in text
