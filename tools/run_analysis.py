"""一键跑完整分析：情感 + 聚类 + 图表 + 报告，并把结果落库、接入导出层。

这是"正式入口"的原型，GUI 的「开始分析」按钮走的就是同一套服务函数。

用法：
    python tools/run_analysis.py                  # 最近一次采集，全部跑
    python tools/run_analysis.py --limit 100      # 限制情感分析条数（控制成本）
    python tools/run_analysis.py --no-sentiment   # 只跑聚类
    python tools/run_analysis.py --no-cluster     # 只跑情感
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cleaning_summary(store, run_id: str, kept_rows: list[dict]) -> dict:
    """从库里的清洗结果拼出报告要用的清洗摘要。"""
    all_rows = store.load_analysis_rows(run_id, only_kept=False)
    total = len(all_rows)
    kept = len(kept_rows)
    tier = (kept_rows[0].get("clean_tier") if kept_rows else None) or "未记录"
    version = (kept_rows[0].get("pipeline_version") if kept_rows else None) or "未记录"
    return {
        "tier": tier,
        "pipeline_version": version,
        "total_input": total,
        "total_kept": kept,
        "retention_rate": (kept / total) if total else 0.0,
        "steps": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--limit", type=int, default=None, help="情感分析最多处理多少条")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument(
        "--no-stance", action="store_true",
        help="跳过立场分析（默认会跑：支持/反对 是研究者的主变量）",
    )
    parser.add_argument(
        "--topic", default=None,
        help="立场分析的对象；默认取本次会话的话题标签",
    )
    parser.add_argument("--k", type=int, default=None, help="强制聚类数")
    parser.add_argument(
        "--charts", default=None,
        help="只画这些图（逗号分隔的 key，见 --list-charts）；默认全画",
    )
    parser.add_argument(
        "--list-charts", action="store_true", help="列出可选图表 key 后退出"
    )
    parser.add_argument(
        "--event", action="append", default=None, metavar="日期[:标签[:前N:后N]]",
        help=(
            "事件节点，可重复。例：--event 2026-09-29:政策公布:7:7。"
            "事件日期必须由研究者指定——工具不会用'评论量最大的那天'替代。"
        ),
    )
    args = parser.parse_args()

    if args.list_charts:
        from wrc.analyze.charts import CHART_SPECS

        print("可选图表（--charts 用逗号分隔这些 key）：")
        for key, filename, label, _needs in CHART_SPECS:
            print(f"  {key:<24} {label:<22} → {filename}")
        return 0

    from wrc.analyze import charts
    from wrc.analyze.report import build_analysis_report, write_report
    from wrc.analyze.service import run_community, run_discourse, run_sentiment
    from wrc.config import get_settings
    from wrc.export import Exporter
    from wrc.llm.client import LlmClient
    from wrc.privacy.anonymize import PseudoIdHasher
    from wrc.privacy.consent import confirm
    from wrc.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init_schema()

    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2
    rows = store.load_analysis_rows(run_id)
    if not rows:
        print(f"会话 {run_id} 没有清洗后的数据。先跑清洗。")
        return 2

    run_row = store.get_run(run_id) or {}
    run_info = {
        "run_id": run_id,
        "tag": run_row.get("tag", ""),
        "created_at": run_row.get("created_at", ""),
    }
    cleaning = _cleaning_summary(store, run_id, rows)
    print(
        f"会话 {run_id}：{len(rows)} 条清洗后评论"
        f"（档位 {cleaning['tier']}，保留率 {cleaning['retention_rate']:.1%}）\n"
    )

    def emit(level: str, message: str) -> None:
        print(f"  [{level}] {message}")

    sentiment_result = None
    discourse_result = None
    community_result = None

    if not args.no_sentiment:
        client = LlmClient()
        ok, reason = client.available()
        if not ok:
            print(f"跳过情感分析：LLM 不可用（{reason}）")
        else:
            print(f"=== 情感分析（{client.config.model}）===")
            classifier = _load_local_sentiment(settings)
            try:
                sentiment_result, _ = run_sentiment(
                    run_id, store=store, settings=settings, llm_client=client,
                    local_classifier=classifier, limit=args.limit, emit=emit,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  情感分析失败：{type(exc).__name__}: {exc}")

    # 立场是独立维度：只要 LLM 可用就跑，默认开启（研究者的主变量）。
    stance_result = None
    if not args.no_stance:
        print("\n=== 立场分析 ===")
        stance_client = LlmClient()
        stance_ok, stance_reason = stance_client.available()
        if not stance_ok:
            print(f"  跳过立场分析：LLM 不可用（{stance_reason}）")
        else:
            try:
                from wrc.analyze.service import run_stance

                stance_result, _ = run_stance(
                    run_id, store=store, settings=settings,
                    llm_client=stance_client,
                    topic=args.topic or run_info.get("tag") or None,
                    emit=emit,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  立场分析失败：{type(exc).__name__}: {exc}")

    if not args.no_cluster:
        print("\n=== 话语聚类 ===")
        try:
            discourse_result, _ = run_discourse(
                run_id, store=store, settings=settings, k=args.k, emit=emit
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  话语聚类跳过：{type(exc).__name__}: {exc}")

        print("\n=== 用户社群聚类 ===")
        try:
            community_result, _ = run_community(
                run_id, store=store, settings=settings, k=args.k
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  用户社群聚类跳过：{type(exc).__name__}: {exc}")

    # P3：时间演变与回复结构都不调模型，只重排已有结果 / 读回复关系，所以总是跑。
    print("\n=== 时间演变 ===")
    timeseries_result = None
    try:
        from wrc.analyze.service import run_timeseries

        timeseries_result, _ = run_timeseries(
            run_id, store=store, settings=settings, emit=emit
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  时间演变跳过：{type(exc).__name__}: {exc}")

    print("\n=== 二级评论回复结构 ===")
    replies_result = None
    try:
        from wrc.analyze.service import run_replies

        replies_result, _ = run_replies(
            run_id, store=store, settings=settings, emit=emit
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  回复结构跳过：{type(exc).__name__}: {exc}")

    crosstab_result = None
    if sentiment_result is not None and discourse_result is not None:
        from wrc.analyze.crosstab import discourse_by_sentiment

        assignments = {
            str(item["cid"]): int(item["cluster"])
            for item in discourse_result.to_dict()["assignments"]
            if item.get("cluster") is not None
        }
        if assignments:
            crosstab_result = discourse_by_sentiment(
                store.load_analysis_rows(run_id), assignments
            )

    out_dir = settings.export_dir / run_id / "report"

    # 事件前后对比（需要研究者给事件日期，没给就整节省略）
    from wrc.analyze.events import parse_event_arg, render_all as render_events
    from wrc.analyze.service import run_event_study

    event_specs = (
        [parse_event_arg(value) for value in args.event] if args.event else None
    )
    event_results, used_events = run_event_study(store, run_id, event_specs)
    if event_results:
        print("\n=== 事件前后对比 ===")
        print(render_events(event_results))
    elif used_events:
        print("\n=== 事件前后对比 ===")
        print("已保存的事件没有可用的窗口数据，请检查日期是否落在采集范围内。")

    # 立场 × 情感 / 簇 × 立场：两个维度都有结果才有意义
    from wrc.analyze.service import (
        build_discourse_stance_crosstab, build_stance_crosstab,
    )
    from wrc.analyze.stance_crosstab import render as render_stance_table

    stance_table = build_stance_crosstab(store, run_id)
    cluster_stance_table = build_discourse_stance_crosstab(store, run_id)
    if stance_table is not None:
        print("\n=== 情感 × 立场 ===")
        print(render_stance_table(stance_table,
                                  title=f"情感 × 立场 交叉表（最小可展示样本 n≥{stance_table.min_n}）"))
    if cluster_stance_table is not None:
        print("\n=== 话语簇 × 立场 ===")
        print(render_stance_table(
            cluster_stance_table,
            title=f"话语簇 × 立场 交叉表（最小可展示样本 n≥{cluster_stance_table.min_n}）",
            group_label="话语簇",
        ))

    print(f"\n=== 图表 ===\n输出目录：{out_dir}")
    include = [item.strip() for item in args.charts.split(",")] if args.charts else None
    generated = charts.render_all(
        out_dir,
        sentiment=sentiment_result,
        stance=stance_result,
        stance_crosstab=stance_table,
        discourse=discourse_result,
        community=community_result,
        timeseries=timeseries_result,
        replies=replies_result,
        events=[result.event for result in event_results],
        rows=kept,
        cleaning_rows=(store.load_comments(run_id), kept),
        include=include,
    )
    for path in generated:
        print(f"  ✓ {path.name}")

    report = build_analysis_report(
        run=run_info,
        cleaning=cleaning,
        sentiment=sentiment_result,
        discourse=discourse_result,
        community=community_result,
        crosstab=crosstab_result,
        timeseries=timeseries_result,
        replies=replies_result,
        stance=stance_result,
        stance_crosstab=stance_table,
        discourse_stance=cluster_stance_table,
        event_results=event_results,
        charts=generated,
        output_dir=out_dir,
    )
    report_path = write_report(out_dir / "report.md", report)
    print(f"\n报告：{report_path}")

    try:
        from wrc.analyze.docx_export import export_report_docx

        print(f"Word：{export_report_docx(report, out_dir)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Word 导出失败（Markdown 已生成）：{type(exc).__name__}: {exc}")

    # 导出层：把分析列并进 analysis_comments.csv
    try:
        exporter = Exporter(
            store,
            hasher=PseudoIdHasher(settings.pseudo_salt_path),
            export_root=settings.export_dir,
        )
        manifest = exporter.export_analysis(
            run_id, consent=confirm(True, source="tool"), method={"分析": "已并入"}
        )
        for item in manifest.files:
            print(f"导出：{item.path.name}（{item.rows} 行）")
    except Exception as exc:  # noqa: BLE001
        print(f"导出层更新失败：{type(exc).__name__}: {exc}")

    return 0


def _load_local_sentiment(settings):
    from wrc.ai.status import detect_status

    status = detect_status(settings.ai_sentiment_model_dir)
    if not status.ready:
        print(f"  本地对照模型未就绪（{status.summary()}），只跑 LLM 通道")
        return None
    from wrc.analyze.classifier import OnnxSequenceClassifier

    return OnnxSequenceClassifier(
        status.onnx_path, status.tokenizer_path, model_name=settings.ai_sentiment_model_dir.name
    )


if __name__ == "__main__":
    sys.exit(main())
