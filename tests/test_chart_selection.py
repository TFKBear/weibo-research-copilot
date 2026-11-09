"""图表可选性：用户能选哪些图、选中的是否真的只画这些。

目标客户是社科学生，他们需要的不是"系统自动给一堆图"，而是
"我论文里要这张，别的先不要"。所以图表必须是可选清单，
并且**选了就必须照做**（包括"一张都不要"）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wrc.analyze.charts import (
    CHART_SPECS,
    activity_heatmap,
    chart_keys,
    render_all,
    term_frequency,
)

CST = timezone(timedelta(hours=8))


def _rows():
    base = datetime(2026, 9, 20, 9, 0, tzinfo=CST)
    rows = []
    for index in range(24):
        when = base + timedelta(hours=index)
        rows.append({
            "cid": f"c{index}",
            "created_at": when.isoformat(),
            "cleaned_text": "延迟退休 对年轻人 不公平 社保 工资",
        })
    return rows


def test_chart_registry_is_unique_and_labelled():
    keys = chart_keys()
    assert len(keys) == len(set(keys))
    assert len(keys) >= 12, "可选图表太少，满足不了「多种可视化形式」"
    assert all(_label for _k, _f, _label, _n in CHART_SPECS)


def test_include_selects_only_requested(tmp_path):
    produced = render_all(tmp_path, rows=_rows(), include=["term_frequency"])
    assert [path.name for path in produced] == ["term_frequency.png"]


def test_include_accepts_filename_too(tmp_path):
    """命令行写 --charts term_frequency 和写 .png 文件名都应认。"""
    produced = render_all(tmp_path, rows=_rows(), include=["activity_heatmap.png"])
    assert [path.name for path in produced] == ["activity_heatmap.png"]


def test_empty_include_means_no_charts(tmp_path):
    """明确"全不选"和"没指定"是两件事：前者一张都不画。"""
    produced = render_all(tmp_path, rows=_rows(), include=[])
    assert produced == []


def test_none_include_draws_everything_available(tmp_path):
    produced = render_all(tmp_path, rows=_rows(), include=None)
    names = {path.name for path in produced}
    assert {"activity_heatmap.png", "term_frequency.png"} <= names


def test_cleaning_chart_only_with_rows(tmp_path):
    """清洗对比需要前后两份数据；没给就安静跳过，而不是画一张空图。"""
    assert render_all(tmp_path, include=["cleaning_comparison"]) == []
    produced = render_all(
        tmp_path,
        cleaning_rows=(
            [{"cid": "c1", "text_raw": "原文较长一些", "followers_count": 10}],
            [{"cid": "c1", "cleaned_text": "原文较长一些", "followers_count": 10}],
        ),
        include=["cleaning_comparison"],
    )
    assert [path.name for path in produced] == ["cleaning_comparison.png"]


# --------------------------------------------------------------------------- #
# 两张新图
# --------------------------------------------------------------------------- #
def test_activity_heatmap_renders(tmp_path):
    path = activity_heatmap(_rows(), tmp_path / "heat.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_activity_heatmap_skips_without_timestamps(tmp_path):
    rows = [{"cid": "c1", "created_at": None, "cleaned_text": "内容"}]
    assert activity_heatmap(rows, tmp_path / "heat.png") is None


def test_term_frequency_renders_top_words(tmp_path):
    path = term_frequency(_rows(), tmp_path / "terms.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_term_frequency_skips_empty_text(tmp_path):
    assert term_frequency([{"cid": "c1", "cleaned_text": ""}], tmp_path / "t.png") is None


def test_report_references_new_charts(tmp_path):
    """画了却不在报告里引用，等于没画——研究者不会去翻目录找图。"""
    from wrc.analyze.report import build_analysis_report

    charts = []
    for name in ("activity_heatmap.png", "term_frequency.png"):
        path = tmp_path / name
        path.write_bytes(b"fake")
        charts.append(path)

    class _Shim:
        method = {"module": "cluster_discourse"}
        distribution: dict = {}
        per_post: dict = {}
        daily: dict = {}

        def render(self) -> str:
            return "内容"

    report = build_analysis_report(
        run={"run_id": "r1", "tag": "#t#"},
        cleaning={"tier": "moderate", "pipeline_version": "v", "total_input": 10,
                  "total_kept": 10, "retention_rate": 1.0},
        discourse=_Shim(),
        timeseries=_Shim(),
        charts=charts,
        output_dir=tmp_path,
    )
    assert "![高频词](term_frequency.png)" in report
    assert "![发布时段热力图](activity_heatmap.png)" in report


@pytest.mark.parametrize("key", ["sentiment_distribution", "timeseries_evolution"])
def test_missing_result_does_not_break_selection(tmp_path, key):
    """选了某张图但这批数据没有相应结果时，只跳过，不报错。"""
    produced = render_all(tmp_path, include=[key])
    assert produced == []
