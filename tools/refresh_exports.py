"""重生成导出目录里的「文字类」产物（不重跑采集、不重跑分析）。

用途：编号说明、附录、研究日志这类东西会随代码修 bug 而变化——
例如真实运行里发现 `research_log.md` 只有表头没有一行事件
（导出发生在另一个进程，内存里没有事件），修好之后必须能把旧会话的产物补正，
而不是让研究者重采一遍。

会重写：`research_log.md`、`appendix.md`、`method_note.md`
不会动：`raw/`、`analysis/` 下的 CSV 与清单（数据文件保持原样）

用法：
    python tools/refresh_exports.py --run <run_id>
    python tools/refresh_exports.py --run <run_id> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _collect_options(params: dict, run: dict):
    from wrc.pipeline import CollectOptions
    from wrc.weibo.window import parse_window_time

    window = params.get("comment_window") or {}
    since = parse_window_time(str(window.get("since"))) if window.get("since") else None
    until = parse_window_time(str(window.get("until"))) if window.get("until") else None
    return CollectOptions(
        tag=str(run.get("tag") or ""),
        order=str(params.get("order_mode") or run.get("order_mode") or "engagement"),
        post_limit=int(params.get("post_limit") or 20),
        comments_per_post=int(params.get("comments_per_post") or 50),
        per_post_targets=dict(params.get("per_post_targets") or {}),
        include_subcomments=bool(params.get("include_subcomments")),
        subcomment_target=int(params.get("subcomment_target") or 10),
        comment_backend=str(params.get("comment_backend") or "auto"),
        comment_sort=str(params.get("comment_sort") or "hot"),
        max_requests=int(params.get("max_requests") or 120),
        since=since,
        until=until,
        search_mode=str(params.get("search_mode") or "topic"),
    )


def _clean_options(params: dict):
    from wrc.pipeline import CleanOptions

    return CleanOptions(
        tier=str(params.get("clean_tier") or "moderate"),
        dedupe_scope=str(params.get("dedupe_scope") or "global"),
        use_ai=bool(params.get("use_ai")),
        filter_bots=bool(params.get("filter_bots")),
    )


def _requests_used(settings, run_id: str) -> int:
    from wrc.research_log import ResearchLog

    path = Path(settings.research_log_path)
    if not path.exists():
        return 0
    for entry in ResearchLog.read(path):
        if entry.get("run_id") == run_id and entry.get("event") == "collect_done":
            try:
                return int(entry.get("requests") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def refresh(run_id: str, *, dry_run: bool = False) -> int:
    from wrc.config import get_settings
    from wrc.export import write_method_note
    from wrc.pipeline import build_method_params
    from wrc.reproducibility import cleaning_facts, write_for_run
    from wrc.research_log import ResearchLog
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()

    run = store.get_run(run_id)
    if run is None:
        print(f"找不到采集会话：{run_id}")
        return 2
    params = json.loads(run.get("params_json") or "{}")
    out_dir = Path(settings.export_dir) / run_id
    if not out_dir.exists():
        print(f"导出目录不存在：{out_dir}")
        return 2

    posts = store.load_posts(run_id)
    reached = {str(row.get("post_mid")) for row in store.load_comments(run_id)}
    collect_opts = _collect_options(params, run)
    clean_opts = _clean_options(params)
    method_params = build_method_params(
        collect_opts=collect_opts,
        clean_opts=clean_opts,
        requests_used=_requests_used(settings, run_id),
        attempted=len(posts),
        succeeded=len(reached),
    )

    cleaning = cleaning_facts(store, run_id)
    manifest_rows = {
        "raw_comments.csv": _count_rows(out_dir / "raw/raw_comments.csv"),
        "raw_posts.csv": _count_rows(out_dir / "raw/raw_posts.csv"),
        "analysis_comments.csv": _count_rows(out_dir / "analysis/analysis_comments.csv"),
    }
    manifest_rows = {key: value for key, value in manifest_rows.items() if value}

    if dry_run:
        print(f"[dry-run] 会话 {run_id}")
        print(f"  帖子：选中 {len(posts)} 篇，成功 {len(reached)} 篇")
        for key in ("实际选中帖子数", "评论时间窗", "检索口径", "本次实际请求数"):
            print(f"  {key}：{method_params.get(key)}")
        return 0

    log_path = ResearchLog(settings.research_log_path, run_id).write_markdown(
        out_dir / "research_log.md"
    )
    note_path = write_method_note(
        out_dir / "method_note.md",
        run={
            "run_id": run_id,
            "tag": run.get("tag", ""),
            "created_at": run.get("created_at", ""),
        },
        params=method_params,
        cleaning=cleaning,
        manifest_rows=manifest_rows,
    )
    appendix_path = write_for_run(store, settings, run_id, path=out_dir / "appendix.md")

    for path in (log_path, note_path, appendix_path):
        size = path.stat().st_size
        print(f"  ✓ {path.relative_to(out_dir.parent.parent)}  {size:,} 字节")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重生成导出目录里的文字类产物")
    parser.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要写入的口径")
    args = parser.parse_args(argv)

    run_id = args.run
    if run_id is None:
        from wrc.config import get_settings
        from wrc.store import Store

        store = Store(get_settings().db_path)
        store.init_schema()
        run_id = store.latest_run_id()
        if run_id is None:
            print("还没有采集记录。")
            return 2
    return refresh(run_id, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
