"""对已经存在的采集会话重跑清洗与导出（不重新采集）。

存在的理由很具体：采集中途被打断（某篇帖子接口报错、你自己按了停止、
浏览器崩了），数据库里其实**已经有评论**，但清洗和导出从没跑过。
旧版本里这种数据等于废了——你只能重采一遍，白花请求。

现在可以把它捞回来：

```bash
python tools/run_clean.py --run 20261025T230437-b4f0
python tools/run_clean.py --run <会话ID> --tier light      # 换档位
python tools/run_clean.py --run <会话ID> --no-export       # 只清洗，不导出
```

⚠ 注意：重跑清洗会**覆盖**该会话已有的清洗结果。如果那个会话已经
跑过情感分析，分析结果会与新的清洗结果对不上——这时工具会提醒你。

用法里带 ``--list`` 可以看有哪些会话、各自采了多少条评论。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    parser.add_argument("--list", action="store_true", help="列出所有会话与评论数")
    parser.add_argument(
        "--tier", choices=("none", "light", "moderate", "heavy"), default=None,
        help="清洗档位；默认沿用该会话上次的档位",
    )
    parser.add_argument(
        "--dedupe", choices=("global", "post", "off"), default=None,
        help="去重范围；默认沿用采集参数",
    )
    parser.add_argument("--no-export", action="store_true", help="只清洗，不导出")
    args = parser.parse_args()

    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init_schema()

    if args.list:
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT r.run_id, r.tag, r.created_at, r.status, "
                "(SELECT COUNT(*) FROM raw_comments c WHERE c.run_id=r.run_id) AS comments, "
                "(SELECT COUNT(*) FROM clean_comments k WHERE k.run_id=r.run_id) AS cleaned "
                "FROM collection_runs r ORDER BY r.created_at DESC"
            ).fetchall()
        print(f"{'会话 ID':<24}{'评论':>6}{'已清洗':>8}  状态   话题")
        for row in rows:
            print(
                f"{row['run_id']:<24}{row['comments']:>6}{row['cleaned']:>8}  "
                f"{row['status']:<7}{row['tag']}"
            )
        return 0

    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2

    from wrc.pipeline import CleanOptions, ResearchPipeline
    from wrc.privacy.consent import confirm

    pipeline = ResearchPipeline(settings)
    comments = store.load_comments(run_id)
    if not comments:
        print(f"会话 {run_id} 里没有评论，无法清洗。先重新采集。")
        return 2

    run_row = store.get_run(run_id) or {}
    params = {}
    try:
        import json

        params = json.loads(run_row.get("params_json") or "{}")
    except Exception:  # noqa: BLE001
        params = {}

    tier = args.tier or params.get("clean_tier") or "moderate"
    dedupe = args.dedupe or params.get("dedupe_scope") or "global"
    print(f"会话 {run_id}（{run_row.get('tag', '')}）：{len(comments)} 条评论")
    print(f"清洗档位 {tier}；去重 {dedupe}")

    summaries = store.latest_analysis_summaries(run_id)
    if summaries:
        print(
            "⚠ 该会话已有分析结果（" + "、".join(sorted(summaries)) + "）。"
            "重跑清洗后，这些分析结果可能与新的清洗结果不一致，建议随后重跑分析。"
        )

    def emit(level: str, message: str) -> None:
        print(f"  [{level}] {message}")

    opts = CleanOptions(tier=tier, dedupe_scope=dedupe)
    report, _ai_summary = pipeline.clean(run_id, opts, sample_size=20)
    print()
    print(report.render())

    if args.no_export:
        print("已按要求跳过导出。")
        return 0

    try:
        from wrc.pipeline import CollectOptions

        collect_opts = CollectOptions(
            tag=str(run_row.get("tag") or ""),
            post_limit=int(params.get("post_limit") or 20),
            comments_per_post=int(params.get("comments_per_post") or 50),
            include_subcomments=bool(params.get("include_subcomments")),
            comment_backend=str(params.get("comment_backend") or "auto"),
            comment_sort=str(params.get("comment_sort") or "hot"),
            max_requests=int(params.get("max_requests") or 120),
        )
        manifest = pipeline.export(
            run_id,
            report=report,
            clean_opts=opts,
            collect_opts=collect_opts,
            consent=confirm(True, source="tool"),
        )
        out_dir = manifest.out_dir.parent
        print(f"\n导出目录：{out_dir}")
        for item in manifest.files:
            print(f"  · {item.path.name}（{item.rows} 行）")
        try:
            print(f"研究日志：{pipeline.write_research_log(run_id, out_dir)}")
            print(f"可复现附录：{pipeline.write_appendix(run_id, out_dir)}")
        except Exception as exc:  # noqa: BLE001
            print(f"日志/附录生成失败（不影响数据）：{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"导出失败：{type(exc).__name__}: {exc}")
        return 1

    print()
    print("下一步：python tools/run_analysis.py --run " + run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
