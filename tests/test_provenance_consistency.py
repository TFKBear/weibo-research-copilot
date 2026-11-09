"""口径一致性：同一批数据，报告 / 附录 / 方法卡片必须说同一套话。

为什么值得单独测：

这套工具里"口径"出现过四条不同的渲染路径——报告正文的采集口径小节、
可复现性附录的「二、采集」、方法卡片的「一、数据」、导出的 method_note。
四条路各写一遍，迟早分叉；而分叉的后果不是"显示不一致"这么轻：
论文里两处写的时间窗不同，审稿人只会得出一个结论——这份数据来源不可信。

所以这里不让它们各自推导，而是要求它们从**同一份 summary** 取值，
并用测试盯住"看起来一样的字"到底是不是同一句。
"""

from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from wrc.analyze.collection import render_collection, summarise_collection
from wrc.analyze.method_card import build_method_card
from wrc.reproducibility import build_appendix, collect_facts
from wrc.store import Store

CST = timezone(timedelta(hours=8))


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    return store

WINDOW = {
    "since": "2026-09-01T00:00:00+08:00",
    "until": "2026-09-30T23:59:59+08:00",
    "active": True,
    "label": "2026-09-01 00:00 ～ 2026-09-30 23:59",
}

PARAMS = {
    "mode": "live",
    "search_mode": "keyword",
    "comment_sort": "time",
    "comments_per_post": 80,
    "post_limit": 20,
    "comment_backend": "auto",
    "max_requests": 150,
    "include_subcomments": False,
    "comment_window": WINDOW,
}


def _seed(store: Store) -> str:
    from wrc.weibo.models import TimePrecision, WeiboComment

    run_id = "run-consistency"
    store.create_run(run_id=run_id, tag="#延迟退休#", order_mode="engagement", params=PARAMS)
    comments = [
        WeiboComment(
            cid=f"c{index}", post_mid="p1", user_id=f"u{index}", screen_name="某人",
            text_raw="这条政策对年轻人不公平",
            created_at=datetime(2026, 9, 20, 10 + index, 0, tzinfo=CST),
            created_at_raw="09-20 10:00", created_at_precision=TimePrecision.MINUTE,
            followers_count=500,
        )
        for index in range(3)
    ]
    store.save_comments(run_id, comments)

    from wrc.clean.basic import Cleaner
    from wrc.pipeline import CleanOptions

    options = CleanOptions(tier="moderate")
    result = Cleaner(
        tier="moderate", dedupe_scope="global", enabled_steps=options.resolve_steps()
    ).run(store.load_comments(run_id))
    store.save_clean(
        run_id, result.rows, clean_tier="moderate", pipeline_version=result.pipeline_version
    )
    return run_id


def _summary(store: Store, run_id: str):
    rows = store.load_analysis_rows(run_id)
    return summarise_collection(params=PARAMS, rows=rows)


def _appendix_text(store: Store, tmp_path, run_id: str) -> str:
    settings = SimpleNamespace(
        export_dir=tmp_path / "exports",
        data_dir=tmp_path,
        db_path=tmp_path / "wrc.sqlite3",
        ai_embedding_model_dir=tmp_path / "no-model",
        ai_sentiment_model_dir=tmp_path / "no-model",
    )
    return build_appendix(collect_facts(store, settings, run_id, with_hashes=False))


def _card_text(summary) -> str:
    return build_method_card(
        module="stance",
        method={"module": "stance"},
        run={
            "run_id": "run-consistency",
            "tag": "#延迟退休#",
            "search_mode": summary.get("search_mode"),
            "comment_window": summary.get("window"),
            "comment_sort": summary.get("sort"),
            "sort_applied": summary.get("sort_applied"),
            "covered_from": summary.get("covered_from"),
            "covered_to": summary.get("covered_to"),
        },
    )


def test_three_surfaces_agree_on_window_and_mode(store, tmp_path):
    run_id = _seed(store)
    summary = _summary(store, run_id)

    report_text = render_collection(summary)
    appendix_text = _appendix_text(store, tmp_path, run_id)
    card_text = _card_text(summary)

    # 时间窗：三处出现的是同一个 label
    assert WINDOW["label"] in report_text
    assert WINDOW["label"] in appendix_text
    assert WINDOW["label"] in card_text

    # 检索口径：三处都点明这是关键词全文，而不是话题聚合
    for text, name in ((report_text, "报告"), (appendix_text, "附录"), (card_text, "卡片")):
        assert "关键词" in text, f"{name}没写检索口径"

    # 实际覆盖：报告与卡片用同一组起止时间（附录同理，日期部分一致）
    assert summary["covered_from"] in report_text
    assert summary["covered_from"] in card_text
    assert summary["covered_from"][:10] in appendix_text


def test_report_and_card_agree_on_sort_not_applied(store, tmp_path):
    """按时间取却没用网页端后端时，两处都必须写"未生效"。"""
    run_id = _seed(store)
    summary = _summary(store, run_id)
    # 没有研究日志时 sort_applied 为 None：不能凭空说"已生效"，也不能说"未生效"
    assert summary["sort_applied"] is None
    assert "已生效" not in render_collection(summary)

    log_events = [{"event": "comments_done", "sort_applied": False, "backend": "mobile_hotflow"}]
    rows = store.load_analysis_rows(run_id)
    flagged = summarise_collection(params=PARAMS, rows=rows, log_events=log_events)
    assert "未生效" in render_collection(flagged)
    assert "未生效" in _card_text(flagged)


def test_coverage_is_derived_not_copied_from_plan(store, tmp_path):
    """数据只落在 9-20，报告就不能声称覆盖到 9-30——即使窗口是那么设的。"""
    run_id = _seed(store)
    summary = _summary(store, run_id)
    assert summary["covered_from"].startswith("2026-09-20")
    assert summary["covered_to"].startswith("2026-09-20")
    assert "2026-09-30" not in summary["covered_to"]


def test_wording_comes_from_one_place(store, tmp_path):
    """四处渲染不许各写一套字：口径文案只有 analyze.collection 一个来源。"""
    from wrc.analyze.collection import search_mode_text, window_text

    run_id = _seed(store)
    summary = _summary(store, run_id)
    report_text = render_collection(summary)
    appendix_text = _appendix_text(store, tmp_path, run_id)
    card_text = _card_text(summary)

    mode_phrase = search_mode_text(summary["search_mode"])
    window_phrase = window_text(summary["window"])

    # 报告与卡片逐字包含统一说法
    assert mode_phrase in report_text
    assert mode_phrase in card_text
    assert window_phrase in card_text
    # 附录也用同一句（它的行首是 "- 检索口径："）
    assert f"- 检索口径：{mode_phrase}" in appendix_text
    assert f"- 评论时间窗：{window_phrase}" in appendix_text

    # 报告里的时间窗说法也必须逐字一致
    assert f"评论时间窗：{window_phrase}" in report_text
