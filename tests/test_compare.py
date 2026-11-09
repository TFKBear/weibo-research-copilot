"""跨话题对比的测试。

这个模块最容易犯的错不是算错数，而是**允许不该比的两次采集被比**。
所以测试几乎全在可比性判断上：mock 与 live、上限不同、档位不同、参数未记录，
每一种都必须被拦下来，而且要在文字里说清楚"为什么不能比"。
"""

from __future__ import annotations

from wrc.analyze.compare import (
    RunProfile,
    build_comparison,
    profile_from_rows,
)


def _profile(tag, *, mode="live", tier="moderate", per_post=50, posts=2,
             comments=60, kept=55, sentiment=None, terms=None):
    return RunProfile(
        run_id=f"run-{tag}",
        tag=tag,
        mode=mode,
        n_posts=posts,
        n_comments=comments,
        n_kept=kept,
        retention=kept / comments if comments else 0.0,
        sentiment=sentiment if sentiment is not None else {},
        sentiment_n=kept if sentiment is not None else 0,
        top_terms=terms or [],
        params={
            "mode": mode,
            "clean_tier": tier,
            "comments_per_post": per_post,
            "post_limit": posts,
            "comment_backend": "auto",
            "comment_sort": "hot",
            "include_subcomments": False,
        },
    )


def test_identical_params_are_comparable():
    result = build_comparison([
        _profile("甲", sentiment={"positive": 0.3, "neutral": 0.3, "negative": 0.4}),
        _profile("乙", sentiment={"positive": 0.5, "neutral": 0.2, "negative": 0.3}),
    ])
    assert result.comparable is True
    assert result.differences == []
    assert result.verdict() == "可直接对比"


def test_mock_and_live_never_comparable():
    result = build_comparison([
        _profile("真数据"),
        _profile("样例数据", mode="mock"),
    ])
    assert result.comparable is False
    assert any("mock" in warning for warning in result.warnings)


def test_different_per_post_limit_blocks_volume_comparison():
    result = build_comparison([
        _profile("甲", per_post=30),
        _profile("乙", per_post=100),
    ])
    assert result.volume_comparable is False
    assert result.share_comparable is True
    assert "仅占比可对比" in result.verdict()
    assert any("每帖评论上限" in item for item in result.differences)
    assert any("数量差异首先反映的是采集参数差异" in w for w in result.warnings)


def test_different_clean_tier_blocks_share_comparison():
    result = build_comparison([
        _profile("甲", tier="light"),
        _profile("乙", tier="heavy"),
    ])
    assert result.volume_comparable is True
    assert result.share_comparable is False
    text = result.render()
    # 占比列必须显示为「—」，不能偷偷给出数字
    assert "仅绝对量可对比" in text
    assert "清洗差异" in text


def test_unrecorded_param_is_not_assumed_equal():
    first = _profile("甲")
    second = _profile("乙")
    second.params.pop("comment_backend")
    result = build_comparison([first, second])
    assert result.share_comparable is False
    assert any("未记录" in item for item in result.differences)
    assert any("未记录不等于相同" in item for item in result.differences)


def test_same_tag_runs_get_stability_warning():
    result = build_comparison([_profile("同一话题"), _profile("同一话题")])
    assert any("采集稳定性" in warning for warning in result.warnings)


def test_small_samples_are_flagged():
    result = build_comparison([
        _profile("甲", comments=20, kept=18),
        _profile("乙", comments=20, kept=16),
    ])
    assert any("不足 30 条" in warning for warning in result.warnings)


def test_missing_sentiment_is_flagged_not_invented():
    result = build_comparison([_profile("甲"), _profile("乙")])
    assert any("还没有情感分析结果" in warning for warning in result.warnings)
    assert "—" in result.render()


def test_fewer_than_two_profiles_degrades():
    result = build_comparison([])
    assert result.profiles == []
    assert "没有可对比的采集" in result.render()


def test_render_never_claims_significance():
    result = build_comparison([
        _profile("甲", sentiment={"positive": 0.3, "neutral": 0.3, "negative": 0.4}),
        _profile("乙", sentiment={"positive": 0.6, "neutral": 0.1, "negative": 0.3}),
    ])
    text = result.render()
    assert "不包含显著性检验" in text
    assert "不是话题全量" in text


def test_profile_from_rows_computes_shares_and_terms():
    rows = [
        {"cid": "1", "cleaned_text": "小米 汽车 保值", "sentiment": "negative"},
        {"cid": "2", "cleaned_text": "小米 汽车 保值", "sentiment": "positive"},
        {"cid": "3", "cleaned_text": "小米 用料 不错", "sentiment": "positive"},
    ]
    profile = profile_from_rows(
        run_id="r1", tag="#测试#",
        params={"mode": "live", "clean_tier": "moderate"},
        rows=rows, n_posts=2, all_rows_count=4,
    )
    assert profile.n_comments == 4
    assert profile.n_kept == 3
    assert profile.retention == 0.75
    assert profile.sentiment["positive"] == 2 / 3
    assert profile.sentiment_n == 3
    terms = dict(profile.top_terms)
    assert terms.get("小米") == 3
    assert terms.get("保值") == 2


def test_profile_without_sentiment_has_empty_shares():
    profile = profile_from_rows(
        run_id="r1", tag="#测试#", params={},
        rows=[{"cid": "1", "cleaned_text": "内容"}],
    )
    assert profile.sentiment == {}
    assert profile.sentiment_n == 0


def test_to_dict_is_serializable():
    result = build_comparison([
        _profile("甲", sentiment={"positive": 0.5, "neutral": 0.2, "negative": 0.3}),
        _profile("乙", sentiment={"positive": 0.2, "neutral": 0.3, "negative": 0.5}),
    ])
    payload = result.to_dict()
    assert payload["module"] == "compare"
    assert payload["verdict"] == "可直接对比"
    assert len(payload["profiles"]) == 2


# --------------------------------------------------------------------------- #
# 时间窗与检索口径（本轮新增：这两条比"上限不同"更致命）
# --------------------------------------------------------------------------- #
def _with_params(profile, **extra):
    profile.params.update(extra)
    return profile


def test_different_comment_windows_are_not_comparable():
    """窗口不同 = 覆盖的时段不同，占比和绝对量都不能直接比。"""
    a = _with_params(_profile("甲"), comment_window_label="2026-09-01 00:00 ～ 2026-09-15 23:59")
    b = _with_params(_profile("乙"), comment_window_label="2026-09-16 00:00 ～ 2026-09-30 23:59")
    result = build_comparison([a, b])

    assert result.comparable is False
    assert result.volume_comparable is False
    assert result.share_comparable is False
    assert any("时间窗" in item for item in result.differences)


def test_same_window_label_still_comparable():
    label = "2026-09-01 00:00 ～ 2026-09-30 23:59"
    a = _with_params(_profile("甲"), comment_window_label=label)
    b = _with_params(_profile("乙"), comment_window_label=label)
    assert build_comparison([a, b]).comparable is True


def test_topic_vs_keyword_is_not_share_comparable():
    """话题聚合与关键词全文是两种人群，占比不能直接比，但绝对量口径一致时可以比。"""
    a = _with_params(_profile("甲"), search_mode="topic")
    b = _with_params(_profile("乙"), search_mode="keyword")
    result = build_comparison([a, b])

    assert result.share_comparable is False
    assert result.volume_comparable is True
    assert result.verdict() == "仅绝对量可对比（占比不可比）"


def test_profile_from_rows_flattens_window_dict():
    """时间窗是 dict，不可哈希；必须压成 label 才能参与"集合大小"比较。"""
    rows = [{"cid": "1", "cleaned_text": "内容", "sentiment": "negative"}]
    profile = profile_from_rows(
        run_id="r1", tag="#测试#",
        params={
            "mode": "live",
            "comment_window": {"active": True, "label": "2026-09-01 00:00 ～ 2026-09-30 23:59"},
        },
        rows=rows,
    )
    assert profile.params["comment_window_label"].startswith("2026-09-01")
    # 老数据没有这个字段时不能崩，要给出"不限时间"这个明确说法
    legacy = profile_from_rows(run_id="r2", tag="#测试#", params={}, rows=rows)
    assert legacy.params["comment_window_label"] == "不限时间"
    assert legacy.params["search_mode"] == "topic"


def test_compare_tolerates_nested_window_dicts_side_by_side():
    """两次采集的窗口写法不同（一个是 dict、一个是压平后的 label）也不能抛异常。"""
    a = profile_from_rows(
        run_id="r1", tag="#甲#",
        params={"mode": "live", "comment_window": {"active": False}},
        rows=[{"cid": "1", "cleaned_text": "内容"}],
    )
    b = _with_params(_profile("乙"), comment_window_label="不限时间")
    assert a.params["comment_window_label"] == "不限时间"
    result = build_comparison([a, b])
    # 这两次采集的其它参数确实不同，不能断言"可比"；
    # 这里只盯一件事：时间窗没有被误报成差异，也没有因为 dict 不可哈希而崩。
    assert not any("时间窗" in item for item in result.differences)
