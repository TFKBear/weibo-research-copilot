"""导出的文字产物：日志、方法说明、附录里的口径。

这些测试来自一次**真实运行**（会话 20261026T195346-6d31）暴露的问题：
- 导出的 `research_log.md` 只有表头，一行事件都没有；
- `method_note.md` 里「实际选中帖子数」写的是 `-`；
- 附录完全没提"20 篇帖子有 9 篇被平台拒绝"，读者会以为都采到了。

共同点：数据是对的，**说明数据怎么来的那段是空的或错的**。
对一个以可追溯为卖点的工具，这比算错一个数更严重。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from wrc.export import write_method_note
from wrc.pipeline import CleanOptions, CollectOptions, build_method_params
from wrc.research_log import ResearchLog
from wrc.weibo.window import CST, parse_window_time

CST_TZ = timezone(timedelta(hours=8))


# --------------------------------------------------------------------------- #
# 研究日志必须能从磁盘读回事件
# --------------------------------------------------------------------------- #
def test_write_markdown_reads_events_back_from_disk(tmp_path):
    path = tmp_path / "research_log.jsonl"
    writer = ResearchLog(path, "run-1")
    writer.record("run_start", tag="#延迟退休#")
    writer.record("comments_done", collected=50)
    writer.record("collect_done", comments=404)

    # 模拟"另一个进程导出"：全新对象，内存里没有任何事件
    fresh = ResearchLog(path, "run-1")
    target = fresh.write_markdown(tmp_path / "research_log.md")
    text = target.read_text(encoding="utf-8")

    assert text.count("| `run_start` |") == 1
    assert "comments_done" in text
    assert "404" in text


def test_write_markdown_ignores_other_runs(tmp_path):
    path = tmp_path / "research_log.jsonl"
    ResearchLog(path, "run-1").record("run_start", tag="#甲#")
    ResearchLog(path, "run-2").record("run_start", tag="#乙#")

    text = ResearchLog(path, "run-2").write_markdown(tmp_path / "out.md").read_text(
        encoding="utf-8"
    )
    assert "#乙#" in text
    assert "#甲#" not in text


# --------------------------------------------------------------------------- #
# 方法说明里的进样率
# --------------------------------------------------------------------------- #
def _opts(**kwargs) -> CollectOptions:
    base = dict(tag="#延迟退休#", order="engagement", post_limit=20, comments_per_post=50)
    base.update(kwargs)
    return CollectOptions(**base)


def test_method_params_report_selected_and_skipped_counts():
    params = build_method_params(
        collect_opts=_opts(),
        clean_opts=CleanOptions(tier="moderate"),
        attempted=20,
        succeeded=11,
        requests_used=83,
    )
    assert params["实际选中帖子数"] == "20 篇（成功 11 篇，跳过 9 篇）"
    assert params["本次实际请求数"] == 83


def test_method_params_without_counts_stay_blank():
    params = build_method_params(
        collect_opts=_opts(), clean_opts=CleanOptions(tier="moderate"),
        attempted=0, succeeded=0,
    )
    assert params["实际选中帖子数"] == "-"


def test_method_params_mark_keyword_mode():
    params = build_method_params(
        collect_opts=_opts(search_mode="keyword"), clean_opts=CleanOptions(tier="light"),
    )
    assert "关键词全文检索" in params["检索口径"]


# --------------------------------------------------------------------------- #
# 方法说明要能吃"没有 description 的清洗步骤"
# --------------------------------------------------------------------------- #
def test_method_note_tolerates_step_without_description(tmp_path):
    """附录/日志里的清洗步骤只有 label/removed/share；缺 description 不该崩。"""
    path = write_method_note(
        tmp_path / "method_note.md",
        run={"run_id": "r1", "tag": "#t#", "created_at": "2026-10-26T19:53:46"},
        params={"评论排序": "hot"},
        cleaning={
            "tier": "moderate",
            "pipeline_version": "basic-0.2.0",
            "total_input": 404,
            "total_kept": 375,
            "retention_rate": 0.928,
            "steps": [{"label": "疑似僵尸号（粉丝<5）", "removed": 29, "share": 0.071},
                      {"label": "精确去重", "removed": 4, "share": 0.01,
                       "description": "完全相同的内容"}],
        },
        manifest_rows={"analysis_comments.csv": 404},
    )
    text = path.read_text(encoding="utf-8")
    assert "疑似僵尸号" in text and "7.1%" in text
    assert "精确去重" in text and "完全相同的内容" in text


# --------------------------------------------------------------------------- #
# 时间窗：ISO 字符串要能读回
# --------------------------------------------------------------------------- #
def test_parse_window_time_accepts_stored_iso():
    """参数里存的是 ISO（带 T 和时区）；读不回来就没法重生成任何产物。"""
    value = parse_window_time("2026-01-01T00:00:00+08:00")
    assert value is not None
    assert value.utcoffset() == timedelta(hours=8)
    assert (value.year, value.month, value.day) == (2026, 1, 1)

    naive_iso = parse_window_time("2026-09-20T14:30:00")
    assert naive_iso == datetime(2026, 9, 20, 14, 30, tzinfo=CST)


def test_parse_window_time_iso_and_friendly_agree():
    """同一时刻的两种写法必须解析成同一个时间点。"""
    assert parse_window_time("2026-09-20T14:30:00+08:00") == parse_window_time(
        "2026-09-20 14:30"
    )


# --------------------------------------------------------------------------- #
# 附录：进样率与空值渲染
# --------------------------------------------------------------------------- #
def test_appendix_renders_skipped_posts_and_empty_dicts():
    from wrc.reproducibility import AppendixFacts, _value, build_appendix

    facts = AppendixFacts(
        run={"run_id": "r1", "tag": "#延迟退休#"},
        params={
            "mode": "live",
            "search_mode": "topic",
            "comment_window_text": "2026-01-01 00:00 ～ 2026-10-26 23:59",
            "covered_from": "2026-08-16T10:19:53",
            "covered_to": "2026-10-26T13:53:32",
            "skipped_posts_text": "20 篇中成功 11 篇，跳过 9 篇（接口返回暂无数据/帖子不可见）",
            "per_post_targets": {},
        },
        cleaning=None,
        summaries={},
        env={},
        generated_at="2026-10-26 20:00:00",
    )
    text = build_appendix(facts)
    assert "进样率：20 篇中成功 11 篇，跳过 9 篇" in text
    # 空 dict 不能渲染成空白单元格，那样看不出"是没设"还是"读丢了"
    assert _value({}) == "无"
    assert "| per_post_targets | 无 |" in text
