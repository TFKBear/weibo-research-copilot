"""从数据库重建报告与图表（不重跑任何分析）。

用途：
- 只补跑了一个模块时，把整份报告重建成完整的；
- 用户回来想改措辞/重出图，不必再花一次 LLM 的钱；
- P2「报告导出」的底座。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _SentimentShim:
    def __init__(self, summary: dict) -> None:
        self._summary = summary
        self.distribution = summary.get("distribution") or {}
        self.per_post = summary.get("per_post") or {}
        self.daily = summary.get("daily") or {}
        self.method = summary.get("method") or {}

    def render(self) -> str:
        return str(self._summary.get("_rendered") or "")


class _DiscourseShim:
    def __init__(self, summary: dict) -> None:
        self._summary = summary
        self.k = int(summary.get("k") or 0)
        self.sizes = {int(k): v for k, v in (summary.get("sizes") or {}).items()}
        self.silhouette = summary.get("silhouette")
        self.method = summary.get("method") or {}

    def render(self) -> str:
        return str(self._summary.get("_rendered") or "")


class _CommunityShim:
    def __init__(self, summary: dict) -> None:
        self._summary = summary
        self.n_users = int(summary.get("n_users") or 0)
        self.roles = {int(k): v for k, v in (summary.get("roles") or {}).items()}
        self.explained_variance = summary.get("explained_variance")
        self.method = summary.get("method") or {}
        self.coords = {
            user: tuple(point) for user, point in (summary.get("coords") or {}).items()
        }
        self.user_labels: dict[str, int] = {}
        for item in summary.get("assignments") or []:
            user = item.get("user_pseudo_id")
            if user:
                self.user_labels[str(user)] = int(item.get("cluster", -1))

    def render(self) -> str:
        return str(self._summary.get("_rendered") or "")


class _StanceShim:
    """把库里的立场摘要包装成能画图的对象。"""

    def __init__(self, summary: dict) -> None:
        self._summary = summary
        self.topic = str(summary.get("topic") or "")
        self.distribution = dict(summary.get("distribution") or {})
        self.method = summary.get("method") or {}

    def render(self) -> str:
        return str(self._summary.get("_rendered") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument(
        "--event", action="append", default=None, metavar="日期[:标签[:前N:后N]]",
        help=(
            "事件节点，可重复。例：--event 2026-09-29:政策公布:7:7。"
            "给了就会写回该会话的参数里，之后重建报告不用再传。"
        ),
    )
    parser.add_argument("--no-events", action="store_true", help="忽略已保存的事件")
    parser.add_argument(
        "--scan-changepoints", action="store_true",
        help=(
            "扫描话语转折点（不是事件研究）：逐日当分界比较前后窗口，"
            "按变动幅度排序，用来定位候选窗口。"
        ),
    )
    parser.add_argument(
        "--scan-metric", choices=("stance", "sentiment"), default="stance",
        help="转折点扫描用的指标（默认立场）",
    )
    parser.add_argument("--scan-top", type=int, default=5, help="列出前几个候选日")
    parser.add_argument(
        "--no-term-trends", action="store_true",
        help="不生成「术语随时间变化」小节（默认会生成）",
    )
    parser.add_argument(
        "--charts", default=None,
        help="只画这些图（逗号分隔；key 见 python tools/run_analysis.py --list-charts）",
    )
    args = parser.parse_args()

    from wrc.analyze import charts
    from wrc.analyze.report import write_report
    from wrc.analyze.report import build_report_from_summaries
    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()
    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2

    summaries = store.latest_analysis_summaries(run_id)
    if not summaries:
        print(f"会话 {run_id} 还没有任何分析结果。先跑 tools/run_analysis.py。")
        return 2

    run_row = store.get_run(run_id) or {}
    kept = store.load_analysis_rows(run_id)
    total = len(store.load_analysis_rows(run_id, only_kept=False))
    cleaning = {
        "tier": (kept[0].get("clean_tier") if kept else None) or "未记录",
        "pipeline_version": (kept[0].get("pipeline_version") if kept else None) or "未记录",
        "total_input": total,
        "total_kept": len(kept),
        "retention_rate": (len(kept) / total) if total else 0.0,
        "steps": [],
    }

    out_dir = settings.export_dir / run_id / "report"

    # 取数一次，后面的交叉表与时间演变都用它。
    comment_analysis = store.load_comment_analysis(run_id)

    # 时间演变：不花任何模型调用，只是把已有标签按时间重排，
    # 所以每次重建报告都顺手重算一遍，比读旧摘要更可靠。
    from wrc.analyze.service import _merge_analysis
    from wrc.analyze.timeseries import build_timeseries

    merged = _merge_analysis(kept, comment_analysis)
    timeseries_result = build_timeseries(merged) if merged else None

    # 回复结构：同样不花模型调用。二级评论没采时它自己会说"无数据"，不会假装有。
    from wrc.analyze.replies import build_reply_structure

    replies_result = build_reply_structure(kept) if kept else None
    stance_result = (
        _StanceShim(summaries["stance"]) if "stance" in summaries else None
    )

    from wrc.analyze.service import (
        build_discourse_stance_crosstab, build_stance_crosstab,
    )

    stance_table = build_stance_crosstab(store, run_id)
    cluster_stance_table = build_discourse_stance_crosstab(store, run_id)

    # 事件前后对比：事件日期来自 --event 或该会话已保存的参数
    from wrc.analyze.events import parse_event_arg
    from wrc.analyze.service import run_event_study

    event_specs = None
    if args.event:
        event_specs = [parse_event_arg(value) for value in args.event]
    elif args.no_events:
        event_specs = []
    event_results, _used_events = run_event_study(store, run_id, event_specs)

    term_text = ""
    if not args.no_term_trends and merged:
        from wrc.analyze.terms import term_trends

        trends = term_trends(merged)
        term_text = trends.render()
        if trends.rising:
            print("关注点上升最快：" + "、".join(
                f"{term}（{first:.0%}→{last:.0%}）"
                for term, first, last in trends.rising[:3]
            ))

    changepoint_text = ""
    if args.scan_changepoints and merged:
        from wrc.analyze.events import render_changepoints, scan_changepoints

        candidates = scan_changepoints(
            merged, metric=args.scan_metric, top=max(1, args.scan_top)
        )
        changepoint_text = render_changepoints(candidates, metric=args.scan_metric)
        if candidates:
            print("话语转折点候选中，变动最大的是：" + "、".join(
                item.date for item in candidates[:3]
            ))

    include = [item.strip() for item in args.charts.split(",")] if args.charts else None
    produced = charts.render_all(
        out_dir,
        sentiment=_SentimentShim(summaries["sentiment"]) if "sentiment" in summaries else None,
        stance=stance_result,
        stance_crosstab=stance_table,
        discourse=(_DiscourseShim(summaries["cluster_discourse"])
                   if "cluster_discourse" in summaries else None),
        community=(_CommunityShim(summaries["cluster_community"])
                   if "cluster_community" in summaries else None),
        timeseries=timeseries_result,
        replies=replies_result,
        events=[result.event for result in event_results],
        rows=kept,
        cleaning_rows=(store.load_comments(run_id), kept),
        include=include,
    )
    print(f"图表：{len(produced)} 张 → {out_dir}")
    for path in produced:
        print(f"  ✓ {path.name}")

    # 采集口径：参数来自 collection_runs，覆盖度由数据反算，排序是否生效看研究日志。
    from wrc.analyze.collection import (
        load_params, render_collection, summarise_collection,
    )
    from wrc.research_log import ResearchLog

    log_events: list[dict] = []
    try:
        log_path = settings.research_log_path
        if log_path.exists():
            log_events = [
                event for event in ResearchLog.read(log_path)
                if event.get("run_id") == run_id
            ]
    except Exception as exc:  # noqa: BLE001
        print(f"研究日志读取失败（采集口径将缺少排序生效信息）：{type(exc).__name__}: {exc}")

    summary = summarise_collection(
        params=load_params(run_row),
        rows=kept,
        log_events=log_events,
    )
    collection_text = render_collection(summary)

    # 交叉表：需要簇分配 + 情感倾向，两者都在 analysis_comments 里现成可取。
    # 放在这里而不是 build_report_from_summaries 内部，是因为报告层只该做排版，
    # 取数逻辑属于工具层。
    crosstab = None
    if "cluster_discourse" in summaries and "sentiment" in summaries:
        from wrc.analyze.crosstab import discourse_by_sentiment

        assignments: dict[str, int] = {}
        rows: list[dict] = []
        for cid, item in comment_analysis.items():
            cluster = item.get("cluster_discourse")
            if cluster is not None:
                assignments[cid] = int(cluster)
            rows.append({"cid": cid, "sentiment": item.get("sentiment")})
        if assignments:
            crosstab = discourse_by_sentiment(rows, assignments)

    content = build_report_from_summaries(
        run={
            "run_id": run_id,
            "tag": run_row.get("tag", ""),
            "created_at": run_row.get("created_at", ""),
            # 方法卡片要能写清"样本怎么来的"：把采集口径一起带进去
            "search_mode": summary.get("search_mode"),
            "comment_window": summary.get("window"),
            "comment_sort": summary.get("sort"),
            "sort_applied": summary.get("sort_applied"),
            "covered_from": summary.get("covered_from"),
            "covered_to": summary.get("covered_to"),
        },
        cleaning=cleaning,
        summaries=summaries,
        crosstab=crosstab,
        timeseries=timeseries_result,
        replies=replies_result,
        stance=stance_result,
        stance_crosstab=stance_table,
        discourse_stance=cluster_stance_table,
        event_results=event_results,
        changepoint_text=changepoint_text,
        term_text=term_text,
        collection_text=collection_text,
        charts=produced,
        output_dir=out_dir,
    )
    path = write_report(out_dir / "report.md", content)
    print(f"报告：{path}")

    # Word 版：给要交作业/送导师审的那一份。失败不应让已有 Markdown 白跑。
    try:
        from wrc.analyze.docx_export import export_report_docx

        docx_path = export_report_docx(content, out_dir)
        print(f"Word：{docx_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Word 导出失败（Markdown 已生成，可用 Word 粘贴）：{type(exc).__name__}: {exc}")

    print(f"包含模块：{', '.join(sorted(summaries))}")
    if crosstab is not None:
        print(f"交叉表：{len(crosstab.groups)} 个分组，"
              f"抑制 {len(crosstab.suppressed_cells)} 个单元格 / "
              f"{len(crosstab.suppressed_groups)} 个组")
    return 0


if __name__ == "__main__":
    sys.exit(main())
