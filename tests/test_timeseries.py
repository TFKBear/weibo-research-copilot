"""时间演变分析的测试。

这个模块的风险不是崩溃，而是**产出一个看起来很有说服力的错结论**。
所以测试的重点是它该有的克制：
- 时间只精确到日时，不许按小时聚合（虚假精度）；
- 时段样本太小时，占比不给数字；
- 前后样本不足时，明确说"不做趋势判断"而不是硬算一个差值；
- 只有一个时段时，不许说"演变"。
"""

from __future__ import annotations

import pytest

from wrc.analyze.timeseries import (
    DAY,
    HOUR,
    MIN_BUCKET_N,
    MIN_TREND_N,
    RELIABLE_TREND_N,
    build_timeseries,
)


def _rows(days: dict[str, int], *, sentiment="negative", precision="second"):
    """按 {日期: 条数} 生成行；时间统一给 10:00:00。"""
    rows = []
    index = 0
    for day, count in days.items():
        for _ in range(count):
            index += 1
            stamp = f"{day}T10:00:00+08:00" if precision != "day" else day
            rows.append({
                "cid": f"c{index}",
                "created_at": stamp,
                "created_at_precision": precision,
                "sentiment": sentiment,
            })
    return rows


def _stance_rows(days: dict[str, dict[str, int]]) -> list[dict]:
    """按 {日期: {立场: 条数}} 生成带立场标签的行。"""
    rows = []
    index = 0
    for day, spec in days.items():
        for stance, count in spec.items():
            for _ in range(count):
                index += 1
                rows.append({
                    "cid": f"s{index}",
                    "created_at": f"{day}T10:00:00+08:00",
                    "created_at_precision": "second",
                    "sentiment": "negative",
                    "stance": stance,
                })
    return rows


def test_buckets_and_volume_are_counted_by_day():
    rows = _rows({"2025-01-01": 3, "2025-01-02": 2})
    result = build_timeseries(rows)
    assert result.granularity == DAY
    assert result.buckets == ["2025-01-01", "2025-01-02"]
    assert result.volume == [3, 2]
    assert result.n_total == 5


def test_day_precision_forces_daily_even_if_hour_requested():
    """只精确到日的时间戳按小时分桶 = 把 00:00 当成"凌晨爆发"，属于造假。"""
    rows = _rows({"2025-01-01": 3, "2025-01-02": 3}, precision="day")
    result = build_timeseries(rows, granularity=HOUR)
    assert result.granularity == DAY
    assert result.precision_note and "强制降级" in result.precision_note


def test_hour_granularity_when_span_is_short_and_sample_large():
    rows = []
    for hour in range(24):
        for index in range(3):
            rows.append({
                "cid": f"c{hour}-{index}",
                "created_at": f"2025-01-01T{hour:02d}:15:00+08:00",
                "created_at_precision": "second",
                "sentiment": "negative",
            })
    result = build_timeseries(rows)
    assert result.granularity == HOUR
    assert len(result.buckets) == 24


def test_small_buckets_are_marked_and_share_hidden_in_text():
    rows = _rows({"2025-01-01": 20, "2025-01-02": 1})
    result = build_timeseries(rows)
    assert result.low_n_buckets == ["2025-01-02"]
    text = result.render()
    assert "样本过小" in text
    # 低样本时段的占比不打印数字
    assert "100%" not in text or "—" in text


def test_trend_not_decided_when_halves_too_small():
    """两侧各 2 条：连放宽后的门槛（5）都不到，仍然拒绝判断。"""
    rows = _rows({"2025-01-01": 2, "2025-01-02": 2})
    result = build_timeseries(rows)
    assert result.shift["decidable"] is False
    assert "不做趋势判断" in result.render()
    assert result.shift["early_n"] + result.shift["late_n"] == 4
    assert min(result.shift["early_n"], result.shift["late_n"]) < MIN_TREND_N


def test_trend_decided_but_flagged_when_below_reliable_threshold():
    """样本 10 vs 10：门槛（5）过了，但没到可信线（30），必须标注"仅供参考"。

    这是研究者看过实际输出后定的口径：宁可给一条带警示的线索，
    也不要在小样本上装作没有信息。
    """
    rows = _rows({"2025-01-01": 10, "2025-01-02": 10})
    result = build_timeseries(rows)
    assert result.shift["decidable"] is True
    assert min(result.shift["early_n"], result.shift["late_n"]) < RELIABLE_TREND_N
    text = result.render()
    assert "仅供参考" in text
    assert any("只能当线索" in warning for warning in result.warnings)


def test_trend_detected_when_both_halves_large_enough():
    rows = []
    for index in range(40):
        rows.append({
            "cid": f"a{index}", "created_at": "2025-01-01T10:00:00+08:00",
            "created_at_precision": "second", "sentiment": "negative",
        })
    for index in range(40):
        rows.append({
            "cid": f"b{index}", "created_at": "2025-01-02T10:00:00+08:00",
            "created_at_precision": "second", "sentiment": "positive",
        })
    result = build_timeseries(rows)
    assert result.shift["decidable"] is True
    assert result.shift["deltas"]["positive"] == 1.0
    text = result.render()
    assert "+100%" in text or "+100" in text
    # 必须提醒这只是描述性对比
    assert "统计检验" in text


def test_single_bucket_says_no_evolution():
    result = build_timeseries(_rows({"2025-01-01": 9}))
    assert any("看不出时间变化" in warning for warning in result.warnings)


def test_unparsable_times_are_counted_not_faked():
    rows = [
        {"cid": "a", "created_at": "昨天下午", "created_at_precision": "unknown"},
        {"cid": "b", "created_at": "", "created_at_precision": "unknown"},
        {"cid": "c", "created_at": "2025-01-01T10:00:00+08:00",
         "created_at_precision": "second", "sentiment": "positive"},
    ]
    result = build_timeseries(rows)
    assert result.n_unparsed == 2
    assert result.n_total == 1


def test_no_time_data_degrades_gracefully():
    result = build_timeseries([{"cid": "a", "created_at": None}])
    assert result.buckets == []
    assert "无法做时间演变" in result.render()


def test_concentration_warning_on_peak_day():
    rows = _rows({"2025-01-01": 30, "2025-01-02": 2, "2025-01-03": 2,
                  "2025-01-04": 2, "2025-01-05": 2})
    result = build_timeseries(rows)
    assert any("高度集中" in warning for warning in result.warnings)


def test_render_always_states_the_sampling_caveat():
    result = build_timeseries(_rows({"2025-01-01": 6, "2025-01-02": 6}))
    text = result.render()
    assert "热门评论流" in text
    assert "不是平台全量舆论的演化" in text


def test_cluster_share_computed_when_clusters_present():
    rows = []
    for index in range(10):
        rows.append({
            "cid": f"a{index}", "created_at": "2025-01-01T10:00:00+08:00",
            "created_at_precision": "second", "sentiment": "negative",
            "cluster_discourse": 0 if index < 7 else 1,
        })
    result = build_timeseries(rows)
    assert "簇0" in result.cluster_share
    assert result.cluster_share["簇0"][0] == 0.7


def test_to_dict_is_serializable_and_keeps_method():
    result = build_timeseries(_rows({"2025-01-01": 6, "2025-01-02": 6}))
    payload = result.to_dict()
    assert payload["module"] == "timeseries"
    assert payload["method"]["min_bucket_n"] == MIN_BUCKET_N
    assert isinstance(payload["volume"], list)


# --------------------------------------------------------------------------- #
# 立场维度随时间变化
# --------------------------------------------------------------------------- #
def test_stance_share_is_computed_per_bucket():
    rows = _stance_rows({
        "2025-01-01": {"oppose": 6, "support": 4},
        "2025-01-02": {"oppose": 9, "support": 1},
    })
    result = build_timeseries(rows)
    assert result.stance_share["oppose"][0] == pytest.approx(0.6)
    assert result.stance_share["oppose"][1] == pytest.approx(0.9)
    assert result.stance_share["support"][1] == pytest.approx(0.1)
    assert result.stance_counts["oppose"] == [6, 9]


def test_irrelevant_share_is_tracked_because_it_dilutes():
    """「无关」占比上升会机械稀释支持率/反对率，所以必须单独可见。"""
    rows = _stance_rows({
        "2025-01-01": {"oppose": 8, "irrelevant": 2},
        "2025-01-02": {"oppose": 4, "irrelevant": 6},
    })
    result = build_timeseries(rows)
    assert result.stance_share["irrelevant"] == [pytest.approx(0.2), pytest.approx(0.6)]
    text = result.render()
    assert "无关" in text


def test_small_bucket_shows_dash_in_stance_table():
    rows = _stance_rows({
        "2025-01-01": {"oppose": 8},
        "2025-01-02": {"oppose": 2},   # < MIN_BUCKET_N
    })
    result = build_timeseries(rows)
    text = result.render()
    stance_part = text[text.find("立场占比随时间"):]
    assert "—" in stance_part


def test_stance_trend_judgement_uses_same_thresholds():
    rows = _stance_rows({
        "2025-01-01": {"oppose": 10, "support": 10},
        "2025-01-02": {"oppose": 18, "support": 2},
    })
    result = build_timeseries(rows)
    assert "立场趋势判断" in result.render()
    assert result.shift_stance["deltas"]["oppose"] == pytest.approx(0.4)


def test_no_stance_means_no_stance_section():
    rows = _rows({"2025-01-01": 6, "2025-01-02": 6})
    result = build_timeseries(rows)
    assert result.stance_share == {}
    assert "立场占比随时间" not in result.render()


def test_to_dict_includes_stance_series():
    rows = _stance_rows({
        "2025-01-01": {"oppose": 6, "support": 4},
        "2025-01-02": {"oppose": 9, "support": 1},
    })
    payload = build_timeseries(rows).to_dict()
    assert "stance_share" in payload
    assert "shift_stance" in payload
    assert payload["stance_share"]["oppose"] == [0.6, 0.9]
