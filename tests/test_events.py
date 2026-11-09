"""事件前后对比（event study）测试。

这个模块最容易出的错不是算错数，而是**把"数据峰值日"当成"事件日"再宣称因果**。
所以规则被写成：事件日期只能由研究者显式给；两侧样本不足只报数；
p 值必须标注假设是否满足，且不替研究者决定要不要用。
"""

from __future__ import annotations

import pytest

from wrc.analyze.events import (
    DEFAULT_PRE_DAYS,
    MIN_WINDOW_N,
    EventSpec,
    analyse_event,
    analyse_events,
    events_from_params,
    parse_event_arg,
    two_proportion_test,
)


def _rows(day: str, count: int, *, stance="oppose", sentiment="negative") -> list[dict]:
    return [
        {
            "cid": f"{day}-{index}",
            "created_at": f"{day}T10:00:00+08:00",
            "created_at_precision": "second",
            "stance": stance,
            "sentiment": sentiment,
        }
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
def test_window_boundaries_are_pre_half_open_post_inclusive():
    """前段是 [event - pre, event)，后段是 [event, event + post]。"""
    rows = (
        _rows("2025-01-01", 40)          # 前段
        + _rows("2025-01-08", 40)        # 事件当天 → 后段
        + _rows("2025-01-15", 40)        # 后段最后一天
        + _rows("2025-01-16", 40)        # 窗口外
    )
    result = analyse_event(
        rows, event=EventSpec(date="2025-01-08", pre_days=7, post_days=7),
        metric="stance",
    )
    assert result.pre.n == 40
    assert result.post.n == 80
    assert result.decidable is True


def test_small_window_refuses_to_give_a_delta():
    rows = _rows("2025-01-01", 10) + _rows("2025-01-08", 10)
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="stance")
    assert result.decidable is False
    assert result.deltas == {}
    assert any("不足" in warning for warning in result.warnings)
    assert "只报数不给结论" in result.render()


def test_valid_window_reports_delta_and_test():
    rows = _rows("2025-01-01", 60, stance="support") + _rows("2025-01-08", 60, stance="oppose")
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="stance")
    assert result.decidable is True
    assert result.pre.shares["support"] == pytest.approx(1.0)
    assert result.post.shares["oppose"] == pytest.approx(1.0)
    assert result.deltas["oppose"] == pytest.approx(1.0)
    assert result.tests["oppose"]["p_value"] < 0.05


def test_sentiment_metric_works_too():
    rows = _rows("2025-01-01", 40, sentiment="neutral") + _rows("2025-01-08", 40, sentiment="negative")
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="sentiment")
    assert result.deltas["negative"] == pytest.approx(1.0)


def test_empty_side_is_flagged():
    rows = _rows("2025-01-01", 50)
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="stance")
    assert result.post.n == 0
    assert any("完全没有数据" in warning for warning in result.warnings)


def test_volume_imbalance_is_warned():
    """前后评论量差 3 倍以上时提醒：占比变化可能来自讨论量激增。"""
    rows = _rows("2025-01-01", 31) + _rows("2025-01-08", 120)
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="stance")
    assert any("评论量相差很大" in warning for warning in result.warnings)


def test_invalid_date_format_raises():
    with pytest.raises(ValueError):
        analyse_event([], event=EventSpec(date="2025/01/08"), metric="stance")


# --------------------------------------------------------------------------- #
def test_two_proportion_test_basic():
    result = two_proportion_test(10, 100, 30, 100)
    assert result is not None
    assert result["z"] > 0
    assert result["p_value"] < 0.01
    assert result["assumptions_ok"] is True


def test_two_proportion_test_flags_low_expected_counts():
    result = two_proportion_test(1, 40, 2, 40)
    assert result is not None
    assert result["assumptions_ok"] is False


def test_two_proportion_test_handles_degenerate_input():
    assert two_proportion_test(0, 0, 5, 10) is None


# --------------------------------------------------------------------------- #
def test_parse_event_arg_variants():
    assert parse_event_arg("2025-01-08").pre_days == DEFAULT_PRE_DAYS
    spec = parse_event_arg("2025-01-08:政策公布:3:5")
    assert spec.date == "2025-01-08"
    assert spec.label == "政策公布"
    assert spec.pre_days == 3
    assert spec.post_days == 5


def test_parse_event_arg_rejects_bad_date():
    with pytest.raises(ValueError):
        parse_event_arg("2025年1月8日")


def test_events_roundtrip_through_params():
    specs = [EventSpec(date="2025-01-08", label="政策公布", pre_days=3, post_days=5)]
    payload = {"events": [spec.to_dict() for spec in specs]}
    restored = events_from_params(payload)
    assert len(restored) == 1
    assert restored[0].date == "2025-01-08"
    assert restored[0].post_days == 5


def test_events_from_empty_params_is_empty():
    assert events_from_params({}) == []


def test_analyse_events_runs_both_metrics():
    rows = _rows("2025-01-01", 40) + _rows("2025-01-08", 40)
    results = analyse_events(rows, events=[EventSpec(date="2025-01-08")])
    assert {item.metric for item in results} == {"stance", "sentiment"}


def test_render_states_no_causality():
    rows = _rows("2025-01-01", 40) + _rows("2025-01-08", 40)
    result = analyse_event(rows, event=EventSpec(date="2025-01-08"), metric="stance")
    text = result.render()
    assert "不是因果推断" in text
    assert "窗口" in text


def test_to_dict_is_serializable():
    rows = _rows("2025-01-01", 40) + _rows("2025-01-08", 40)
    payload = analyse_event(
        rows, event=EventSpec(date="2025-01-08"), metric="stance"
    ).to_dict()
    assert payload["module"] == "event"
    assert payload["event"]["date"] == "2025-01-08"
    assert payload["pre"]["n"] == 40


def test_min_window_constant_is_documented_threshold():
    assert MIN_WINDOW_N >= 20


# --------------------------------------------------------------------------- #
# 图表上的事件竖线
# --------------------------------------------------------------------------- #
def test_event_positions_snap_to_first_bucket_not_before():
    """事件日不在桶上时，线画在"第一个不早于事件日的桶"。

    画早了会让读者以为变化发生在事件之前，所以宁可画晚一格。
    """
    from wrc.analyze.charts import _event_positions

    buckets = ["2025-01-01", "2025-01-03", "2025-01-05"]
    positions = _event_positions(buckets, [EventSpec(date="2025-01-04")])
    assert positions == [(2.0, "2025-01-04")]  # 01-05 而不是 01-03


def test_event_positions_exact_match_stays_on_bucket():
    from wrc.analyze.charts import _event_positions

    buckets = ["2025-01-01", "2025-01-03"]
    assert _event_positions(buckets, [EventSpec(date="2025-01-03")]) == [(1.0, "2025-01-03")]


def test_event_positions_out_of_range_is_skipped():
    from wrc.analyze.charts import _event_positions

    buckets = ["2025-01-01", "2025-01-03"]
    # 事件晚于所有桶 → 不画（而不是画在最后一格误导读者）
    assert _event_positions(buckets, [EventSpec(date="2025-02-01")]) == []


# --------------------------------------------------------------------------- #
# 话语转折点扫描（不是事件研究）
# --------------------------------------------------------------------------- #
def test_scan_changepoints_ranks_by_focal_delta():
    from wrc.analyze.events import scan_changepoints

    rows = (
        _rows("2025-01-01", 20, stance="support")
        + _rows("2025-01-02", 20, stance="support")
        + _rows("2025-01-03", 20, stance="support")
        + _rows("2025-01-08", 20, stance="oppose")
        + _rows("2025-01-09", 20, stance="oppose")
        + _rows("2025-01-10", 20, stance="oppose")
    )
    candidates = scan_changepoints(rows, metric="stance", top=3)
    assert candidates
    # 2025-01-08 正好是"支持→反对"的分界，应排第一
    assert candidates[0].date == "2025-01-08"
    assert candidates[0].deltas["oppose"] == pytest.approx(1.0)


def test_scan_changepoints_skips_days_without_enough_data():
    from wrc.analyze.events import scan_changepoints

    rows = _rows("2025-01-01", 5) + _rows("2025-01-08", 5)
    assert scan_changepoints(rows, metric="stance") == []


def test_scan_changepoints_needs_several_days():
    from wrc.analyze.events import scan_changepoints

    assert scan_changepoints(_rows("2025-01-01", 60), metric="stance") == []


def test_changepoint_render_states_it_is_not_an_event_study():
    from wrc.analyze.events import render_changepoints, scan_changepoints

    rows = (
        _rows("2025-01-01", 20, stance="support")
        + _rows("2025-01-02", 20, stance="support")
        + _rows("2025-01-03", 20, stance="support")
        + _rows("2025-01-08", 20, stance="oppose")
        + _rows("2025-01-09", 20, stance="oppose")
        + _rows("2025-01-10", 20, stance="oppose")
    )
    text = render_changepoints(scan_changepoints(rows, metric="stance"))
    assert "不是**事件研究" in text or "不是事件研究" in text.replace("**", "")
    assert "用结果定义原因" in text


def test_resolve_events_empty_list_clears_saved():
    """``--no-events`` 必须真的能清掉已保存的事件。

    早期把 None 和 [] 都当假值，导致 --no-events 根本不起作用。
    """
    import tempfile
    from pathlib import Path

    from wrc.analyze.service import resolve_events
    from wrc.store import Store

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "wrc.sqlite3")
        store.init_schema()
        store.create_run(run_id="r1", tag="#t#", order_mode="engagement", params={})
        spec = EventSpec(date="2025-01-08", label="测试")
        assert len(resolve_events(store, "r1", [spec])) == 1
        assert len(resolve_events(store, "r1", None)) == 1  # 读回已保存
        assert resolve_events(store, "r1", []) == []       # 显式清空
        assert resolve_events(store, "r1", None) == []     # 清空后确实没有了
