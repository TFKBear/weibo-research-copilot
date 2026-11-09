"""分析结果落库 / 导出接入 / 图表 / 报告重建 的测试。

这一批测的是"分析结果能不能真的用进论文"这条链路：
落库 → 导出成 CSV 列 → 报告可重建。判得准不准是另一回事（那要靠人工抽查）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wrc.analyze.report import build_report_from_summaries
from wrc.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    store.create_run(run_id="run-1", tag="#测试#", order_mode="engagement", params={})
    return store


def _seed_run(store: Store) -> None:
    from wrc.weibo.models import TimePrecision, WeiboComment

    comments = [
        WeiboComment(
            cid=f"c{i}", post_mid="p1", user_id=f"u{i}", screen_name="某人",
            text_raw=f"评论{i}", created_at=None, created_at_raw="今天 10:00",
            created_at_precision=TimePrecision.MINUTE,
        )
        for i in range(3)
    ]
    store.save_comments("run-1", comments)
    store.save_clean(
        "run-1",
        [
            {"cid": f"c{i}", "post_mid": "p1", "cleaned_text": f"评论{i}",
             "kept": True, "drop_reasons": []}
            for i in range(3)
        ],
        clean_tier="moderate",
        pipeline_version="basic-0.2.0",
    )


# --------------------------------------------------------------------------- #
def test_analysis_roundtrip(store):
    _seed_run(store)
    store.save_analysis_run(
        analysis_id="a1", run_id="run-1", module="sentiment",
        params={"limit": 3}, method={"llm_model": "m"},
        summary={"distribution": {"positive": 2}, "_rendered": "渲染文本"},
    )
    store.save_sentiment_items(
        "a1",
        [
            {"cid": "c0", "sentiment": "positive", "intensity": 0.8,
             "local_sentiment": "neutral", "local_score": 0.5, "in_sample": True},
            {"cid": "c1", "sentiment": "negative", "intensity": 0.6},
        ],
    )

    merged = store.load_comment_analysis("run-1")
    assert merged["c0"]["sentiment"] == "positive"
    assert merged["c0"]["in_sample"] is True
    assert merged["c1"]["sentiment"] == "negative"
    assert "c2" not in merged


def test_latest_analysis_wins_per_module(store):
    """同一模块跑两次时，导出只能反映最新一次。

    混两次结果会让 CSV 里同时存在两套模型标签，而方法卡片只能记录一套——
    那是不可复现的数据。
    """
    _seed_run(store)
    store.save_analysis_run(
        analysis_id="old", run_id="run-1", module="sentiment",
        params={}, method={}, summary={},
    )
    store.save_sentiment_items("old", [{"cid": "c0", "sentiment": "positive"}])

    store.save_analysis_run(
        analysis_id="new", run_id="run-1", module="sentiment",
        params={}, method={}, summary={},
    )
    store.save_sentiment_items("new", [{"cid": "c0", "sentiment": "negative"}])

    merged = store.load_comment_analysis("run-1")
    assert merged["c0"]["sentiment"] == "negative"


def test_latest_analysis_summaries_keep_latest(store):
    _seed_run(store)
    store.save_analysis_run(
        analysis_id="a1", run_id="run-1", module="sentiment",
        params={}, method={"v": 1}, summary={"_rendered": "第一版"},
    )
    store.save_analysis_run(
        analysis_id="a2", run_id="run-1", module="sentiment",
        params={}, method={"v": 2}, summary={"_rendered": "第二版"},
    )
    summaries = store.latest_analysis_summaries("run-1")
    assert summaries["sentiment"]["_rendered"] == "第二版"
    assert summaries["sentiment"]["method"]["v"] == 2


def test_analysis_columns_reach_export(store, tmp_path):
    from wrc.export import Exporter
    from wrc.privacy.anonymize import PseudoIdHasher
    from wrc.privacy.consent import confirm

    _seed_run(store)
    store.save_analysis_run(
        analysis_id="a1", run_id="run-1", module="sentiment",
        params={}, method={}, summary={},
    )
    store.save_sentiment_items(
        "a1", [{"cid": "c0", "sentiment": "positive", "intensity": 0.9}]
    )
    exporter = Exporter(
        store, hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_analysis("run-1", consent=confirm(True))
    file = manifest.files[0]
    header = file.path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert "sentiment" in header and "intensity" in header
    assert "cluster_discourse" in header
    body = file.path.read_text(encoding="utf-8-sig").splitlines()[1]
    assert "positive" in body


# --------------------------------------------------------------------------- #
def test_report_rebuilds_from_summaries():
    summaries = {
        "sentiment": {
            "_rendered": "情感渲染块", "method": {"llm_model": "deepseek-v4.1-flash"},
        },
        "cluster_discourse": {"_rendered": "聚类渲染块", "method": {}},
    }
    report = build_report_from_summaries(
        run={"run_id": "run-1", "tag": "#测试#"},
        cleaning={"tier": "moderate", "pipeline_version": "basic-0.2.0",
                  "total_input": 10, "total_kept": 9, "retention_rate": 0.9},
        summaries=summaries,
    )
    assert "情感渲染块" in report
    assert "聚类渲染块" in report
    assert "方法卡片 · 情感分析" in report
    assert "用户社群" not in report  # 没跑就不该出现空章节


def test_report_includes_chart_links(tmp_path):
    charts = [tmp_path / "sentiment_distribution.png"]
    charts[0].write_bytes(b"fake")
    report = build_report_from_summaries(
        run={"run_id": "r"},
        cleaning=None,
        summaries={"sentiment": {"_rendered": "x", "method": {}}},
        charts=charts,
        output_dir=tmp_path,
    )
    assert "![整体情感分布](sentiment_distribution.png)" in report


# --------------------------------------------------------------------------- #
def test_charts_render_all_produces_files(tmp_path):
    from wrc.analyze import charts

    sentiment = SimpleNamespace(
        distribution={"positive": 3, "neutral": 1, "negative": 2},
        per_post={"p1": {"positive": 3, "neutral": 1, "negative": 2}},
        daily={
            "2026-06-12": {"positive": 2, "negative": 1},
            "2026-06-13": {"positive": 1, "negative": 1},
        },
    )
    produced = charts.render_all(tmp_path, sentiment=sentiment)
    names = {path.name for path in produced}
    assert "sentiment_distribution.png" in names
    assert "sentiment_by_post.png" in names
    assert "sentiment_trend.png" in names
    assert all(path.stat().st_size > 0 for path in produced)


def test_charts_skip_trend_with_single_day(tmp_path):
    """只有一天时画"趋势"没有意义，应主动不出图而不是画一条直线。"""
    from wrc.analyze import charts

    sentiment = SimpleNamespace(
        distribution={"positive": 1},
        per_post={"p1": {"positive": 1}},
        daily={"2026-06-12": {"positive": 1}},
    )
    produced = charts.render_all(tmp_path, sentiment=sentiment)
    names = {path.name for path in produced}
    assert "sentiment_trend.png" not in names


# --------------------------------------------------------------------------- #
# 清洗影响图（需求 B2）
# --------------------------------------------------------------------------- #
def _raw_row(cid: str, text: str, followers: int | None) -> dict:
    return {"cid": cid, "text_raw": text, "followers_count": followers}


def _kept_row(cid: str, text: str, followers: int | None) -> dict:
    return {"cid": cid, "cleaned_text": text, "followers_count": followers}


def test_cleaning_comparison_draws_two_panels(tmp_path):
    from wrc.analyze import charts

    raw = [
        _raw_row(f"c{i}", "评论" * (i % 7 + 1), 100 * (i + 1))
        for i in range(40)
    ]
    kept = [row for index, row in enumerate(raw) if index % 5 != 0]
    kept = [_kept_row(row["cid"], row["text_raw"], row["followers_count"]) for row in kept]

    path = charts.cleaning_comparison(raw, kept, tmp_path / "cleaning_comparison.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_cleaning_comparison_returns_none_without_data(tmp_path):
    from wrc.analyze import charts

    assert charts.cleaning_comparison([], [], tmp_path / "x.png") is None
    # 只有清洗后、没有清洗前：无从对比，也必须安静返回 None
    assert charts.cleaning_comparison(
        [], [{"cid": "c1", "cleaned_text": "x", "followers_count": 1}],
        tmp_path / "x.png",
    ) is None


def test_cleaning_comparison_refuses_when_followers_missing(tmp_path):
    """接口不返回粉丝数时，图仍要画（长度那一格有效），不能整张图失败。"""
    from wrc.analyze import charts

    raw = [_raw_row(f"c{i}", "评论" * (i + 1), None) for i in range(10)]
    kept = [_kept_row(f"c{i}", "评论" * (i + 1), None) for i in range(8)]
    path = charts.cleaning_comparison(raw, kept, tmp_path / "cleaning_comparison.png")
    assert path is not None and path.exists()


def test_report_embeds_cleaning_chart(tmp_path):
    """报告第一节要引用清洗影响图，否则研究者根本看不到这张图。"""
    from wrc.analyze.report import build_analysis_report

    chart = tmp_path / "cleaning_comparison.png"
    chart.write_bytes(b"fake")
    report = build_analysis_report(
        run={"run_id": "r", "tag": "#t#"},
        cleaning={"tier": "moderate", "pipeline_version": "basic-0.2.0",
                  "total_input": 100, "total_kept": 90, "retention_rate": 0.9},
        charts=[chart],
        output_dir=tmp_path,
    )
    assert "![清洗前后分布对比](cleaning_comparison.png)" in report
    assert "系统性偏差" in report


def test_charts_failure_does_not_break_others(tmp_path):
    from wrc.analyze import charts

    broken = SimpleNamespace(distribution=None, per_post=None, daily=None)
    produced = charts.render_all(tmp_path, sentiment=broken)
    assert produced == []


# --------------------------------------------------------------------------- #
def test_merge_analysis_carries_local_channel_fields(store):
    """人工抽查表要并排显示「LLM 判定」与「本地判定」。

    `_merge_analysis` 曾经只带出 sentiment/intensity/cluster_discourse，
    把 local_sentiment / local_score 丢掉了——抽查表于是只剩一列可看。
    这条是那个 bug 的回归测试。
    """
    _seed_run(store)
    store.save_analysis_run(
        analysis_id="a1", run_id="run-1", module="sentiment",
        params={}, method={}, summary={"_rendered": "x"},
    )
    store.save_sentiment_items("a1", [
        {"cid": "c0", "sentiment": "positive", "intensity": 0.8,
         "local_sentiment": "negative", "local_score": 0.91, "in_sample": True},
        {"cid": "c1", "sentiment": "neutral", "intensity": 0.2},
    ])

    from wrc.analyze.service import _merge_analysis

    merged = _merge_analysis(store.load_analysis_rows("run-1"), store.load_comment_analysis("run-1"))
    by_cid = {row["cid"]: row for row in merged}
    assert by_cid["c0"]["local_sentiment"] == "negative"
    assert by_cid["c0"]["local_score"] == 0.91
    assert by_cid["c0"]["in_sample"] is True
    # 未抽样的行要保持"没有"，不能被填成 0 或空字符串
    assert by_cid["c1"]["local_sentiment"] is None
    assert by_cid["c1"]["in_sample"] is False
