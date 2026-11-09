"""采集口径（报告里的「采集口径与样本边界」小节）。

这套东西存在的理由：论文方法部分必须能回答"样本怎么来的"。
而最容易出错、也最容易糊弄的地方是**覆盖度**——

研究者在界面上填了 9-01～9-30，如果程序在报告里照抄这个区间，
等于给自己发了一张假通行证：实际可能只采到 9-20 就停了。
所以覆盖度一律由**已采数据反算**，而不是照抄计划值。
"""

from __future__ import annotations

import json
from datetime import datetime

from wrc.analyze.collection import (
    load_params,
    render_collection,
    summarise_collection,
)


def _rows(*specs):
    return [
        {"cid": f"c{index}", "post_mid": "m1", "created_at": created}
        for index, created in enumerate(specs)
    ]


PARAMS = {
    "tag": "#延迟退休#",
    "order_mode": "engagement",
    "post_limit": 20,
    "comments_per_post": 50,
    "comment_sort": "time",
    "comment_backend": "auto",
    "max_requests": 120,
    "include_subcomments": False,
    "comment_window": {
        "since": "2026-09-01T00:00:00+08:00",
        "until": "2026-09-30T23:59:59+08:00",
        "active": True,
        "label": "2026-09-01 00:00 ～ 2026-09-30 23:59",
    },
}


def test_summary_reports_covered_range_from_data_not_plan():
    """关键：覆盖度用实际数据反算。计划的窗口是 9-01～9-30，
    但数据只到 9-20，报告就必须写 9-20。"""
    rows = _rows(
        "2026-09-02T09:14:00+08:00",
        "2026-09-20T22:40:00+08:00",
    )
    summary = summarise_collection(params=PARAMS, rows=rows)
    assert summary["covered_from"].startswith("2026-09-02")
    assert summary["covered_to"].startswith("2026-09-20")
    assert summary["window_active"] is True


def test_summary_flags_sort_not_applied():
    log_events = [
        {"event": "comments_done", "run_id": "r", "sort_applied": False, "backend": "mobile_hotflow"},
    ]
    summary = summarise_collection(params=PARAMS, rows=_rows("2026-09-02"), log_events=log_events)
    assert summary["sort_applied"] is False
    text = render_collection(summary)
    assert "未生效" in text


def test_hot_sort_without_param_is_not_warned_about():
    """排序选的就是热度、后端没有排序参数——这不算"失效"，不该出现警告符号。"""
    params = dict(PARAMS)
    params["comment_sort"] = "hot"
    log_events = [{"event": "comments_done", "sort_applied": False, "backend": "mobile_hotflow"}]
    text = render_collection(
        summarise_collection(params=params, rows=_rows("2026-09-02"), log_events=log_events)
    )
    assert "未生效" not in text
    assert "按热度返回" in text


def test_summary_marks_sort_applied_when_all_posts_agree():
    log_events = [
        {"event": "comments_done", "sort_applied": True, "backend": "web_build_comments"},
        {"event": "comments_done", "sort_applied": True, "backend": "web_build_comments"},
    ]
    summary = summarise_collection(params=PARAMS, rows=_rows("2026-09-02"), log_events=log_events)
    assert summary["sort_applied"] is True
    assert "已生效" in render_collection(summary)


def test_summary_is_silent_when_no_log():
    """日志不在时不能瞎猜"已生效"，只能不表态。"""
    summary = summarise_collection(params=PARAMS, rows=_rows("2026-09-02"))
    assert summary["sort_applied"] is None
    assert "已生效" not in render_collection(summary)


def test_summary_mentions_per_post_overrides():
    params = dict(PARAMS)
    params["per_post_targets"] = {"m1": 300, "m2": 50}
    text = render_collection(summarise_collection(params=params, rows=_rows("2026-09-02")))
    assert "逐帖覆盖" in text
    assert "300" in text


def test_summary_without_window_says_so():
    params = dict(PARAMS)
    params["comment_window"] = {"active": False, "label": "不限时间"}
    text = render_collection(summarise_collection(params=params, rows=_rows("2026-09-02")))
    # 统一说法见 analyze.collection.WINDOW_NONE_TEXT
    assert "不限时间" in text
    assert "采该帖可见的最新评论" in text


def test_render_includes_backend_and_requests():
    log_events = [
        {"event": "comments_done", "sort_applied": False, "backend": "mobile_hotflow"},
        {"event": "collect_done", "requests": 37},
    ]
    text = render_collection(
        summarise_collection(params=PARAMS, rows=_rows("2026-09-02"), log_events=log_events)
    )
    assert "mobile_hotflow" in text
    assert "37" in text


def test_load_params_tolerates_bad_json():
    assert load_params({}) == {}
    assert load_params({"params_json": "not json"}) == {}
    assert load_params({"params_json": json.dumps({"tag": "#t#"})})["tag"] == "#t#"


def test_load_params_fills_topic_and_order_from_columns():
    """话题与排序是表列而不是参数；不合并会在报告里留下「—」。"""
    row = {
        "tag": "#延迟退休#",
        "order_mode": "engagement",
        "params_json": json.dumps({"comments_per_post": 100}),
    }
    params = load_params(row)
    assert params["tag"] == "#延迟退休#"
    assert params["order_mode"] == "engagement"
    assert params["comments_per_post"] == 100


def test_selected_posts_counts_distinct_posts():
    rows = [
        {"cid": "c1", "post_mid": "m1", "created_at": "2026-09-02T10:00:00+08:00"},
        {"cid": "c2", "post_mid": "m1", "created_at": "2026-09-02T11:00:00+08:00"},
        {"cid": "c3", "post_mid": "m2", "created_at": "2026-09-02T12:00:00+08:00"},
    ]
    summary = summarise_collection(params=PARAMS, rows=rows)
    assert summary["selected_posts"] == 2
    assert summary["kept"] == 3


def test_report_places_collection_section_before_cleaning():
    """采集口径是方法前提，必须排在清洗之前，读者先知道样本怎么来的。"""
    from wrc.analyze.report import build_analysis_report

    text = build_analysis_report(
        run={"run_id": "r", "tag": "#t#"},
        cleaning={"tier": "moderate", "pipeline_version": "v", "total_input": 10,
                  "total_kept": 9, "retention_rate": 0.9},
        collection_text="采集口径\n" + "=" * 20 + "\n检索话题：#t#",
    )
    assert "## 采集口径与样本边界" in text
    assert text.index("采集口径与样本边界") < text.index("## 一、数据清洗")


def test_summary_survives_rows_without_timestamps():
    rows = [{"cid": "c1", "post_mid": "m1", "created_at": None}]
    summary = summarise_collection(params=PARAMS, rows=rows)
    assert summary["covered_from"] is None
    assert summary["kept"] == 1
    assert "实际覆盖：" not in render_collection(summary)


def test_datetime_parse_handles_plain_and_iso():
    rows = _rows("2026-09-02", "2026-09-03T10:00:00+08:00")
    summary = summarise_collection(params=PARAMS, rows=rows)
    assert summary["covered_from"] == "2026-09-02T00:00:00"
    assert datetime.fromisoformat(summary["covered_to"]).day == 3


def test_method_card_carries_collection_provenance():
    """方法卡片的「一、数据」必须写清口径——它是论文方法段最容易抄的一段。"""
    from wrc.analyze.method_card import build_method_card

    text = build_method_card(
        module="stance",
        method={"module": "stance"},
        run={
            "run_id": "r1",
            "tag": "#延迟退休#",
            "search_mode": "keyword",
            "comment_window": {"active": True, "label": "2026-09-01 00:00 ～ 2026-09-30 23:59"},
            "comment_sort": "time",
            "sort_applied": False,
            "covered_from": "2026-09-02T09:14:00",
            "covered_to": "2026-09-20T22:40:00",
        },
    )
    assert "关键词全文检索" in text
    assert "2026-09-01 00:00 ～ 2026-09-30 23:59" in text
    assert "未生效" in text
    assert "实际覆盖" in text


def test_method_card_omits_absent_collection_fields():
    from wrc.analyze.method_card import build_method_card

    text = build_method_card(module="stance", method={"module": "stance"},
                             run={"run_id": "r1", "tag": "#t#"})
    assert "检索口径" not in text
    assert "评论时间窗" not in text
