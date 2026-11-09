"""02 · 全链路技术验证（第 1 步的验收脚本）。

这条链路就是交付节奏第 1 步的验收标准：
   登录 → 搜 tag → 选帖 → 采评论 → 去重清洗 → 匿名化导出

它同时是一个可用的最小 CLI：
    离线自检：python spikes/02_chain.py --mock
    真实采集：python spikes/02_chain.py --tag 延迟退休 --select 1,2 --comments 50 --consent

设计上的两个刻意选择：
1. **必须显式选帖**（--select / --auto-select，或交互输入）。
   需求文档明确要求用户勾选目标帖子，而不是自动抓所有帖子；
   非交互环境下没有给选择参数就直接报错，避免误采。
2. **预算耗尽不静默失败**。触发请求上限时保留已采数据，
   继续走清洗与导出，并在日志里写明停止原因。
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import AsyncExitStack

import _common
from _common import (
    EXIT_OK,
    EXIT_USER_ERROR,
    add_common_arguments,
    ask_consent,
    console,
    info,
    note,
    warn,
)
from rich.table import Table

from wrc.weibo.comment_backends import BACKENDS_BY_KEY

ORDER_LABELS = {
    "engagement": "按互动总量（本地重排）",
    "time": "按发帖时间（本地重排）",
    "platform": "按接口返回顺序（平台相关度）",
}

COMMENT_SORT_LABELS = {
    "hot": "按热度（高互动评论优先）",
    "time": "按时间（最新评论优先）",
}


def render_posts(posts: list) -> None:
    table = Table(title="检索结果预览（请据此选择要抓评论的帖子）", title_justify="left")
    table.add_column("#", justify="right", style="bold")
    table.add_column("发帖时间", no_wrap=True)
    table.add_column("互动总量", justify="right")
    table.add_column("评论数", justify="right")
    table.add_column("内容摘要", overflow="fold")
    for index, post in enumerate(posts, start=1):
        when = (
            post.created_at.strftime("%Y-%m-%d %H:%M")
            if post.created_at
            else (post.created_at_raw or "未知")
        )
        table.add_row(
            str(index),
            when,
            f"{post.engagement:,}",
            f"{post.comments_count:,}",
            post.preview(44),
        )
    console().print(table)


def choose_posts(posts: list, args: argparse.Namespace) -> list:
    if args.select:
        try:
            indices = [int(part) for part in args.select.split(",") if part.strip()]
        except ValueError:
            raise SystemExit("--select 需要是逗号分隔的编号，例如 1,2,5")
        invalid = [i for i in indices if i < 1 or i > len(posts)]
        if invalid:
            raise SystemExit(f"--select 中的编号超出范围：{invalid}（可选 1–{len(posts)}）")
        return [posts[i - 1] for i in indices]

    if args.auto_select:
        count = max(1, min(args.auto_select, len(posts)))
        warn(f"按 --auto-select {count} 自动选择前 {count} 篇帖子。")
        return posts[:count]

    if not sys.stdin.isatty():
        raise SystemExit(
            "非交互环境必须显式指定帖子：--select 1,3 或 --auto-select 2"
        )
    raw = input("请输入要抓取评论的帖子编号（逗号分隔，例如 1,3）：").strip()
    try:
        indices = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise SystemExit("编号格式不正确，请使用逗号分隔的数字。")
    if not indices:
        raise SystemExit("没有选择任何帖子。")
    invalid = [i for i in indices if i < 1 or i > len(posts)]
    if invalid:
        raise SystemExit(f"编号超出范围：{invalid}")
    return [posts[i - 1] for i in indices]


async def run(args: argparse.Namespace) -> int:
    from wrc.clean.basic import Cleaner
    from wrc.clean.report import build_report
    from wrc.console import block, report_error
    from wrc.errors import AuthRequired, BudgetExceeded, CommentSourceEmpty, WrcError
    from wrc.export import Exporter, write_method_note, write_readme
    from wrc.fetching import MockFetcher
    from wrc.mockdata import build_scenario
    from wrc.privacy.anonymize import PseudoIdHasher
    from wrc.privacy.consent import confirm
    from wrc.research_log import ResearchLog, new_run_id
    from wrc.session import BrowserSession
    from wrc.store import Store
    from wrc.weibo.collect import WeiboClient
    from wrc.weibo.endpoints import normalize_tag
    from wrc.weibo.ratelimit import RateLimiter, RequestBudget

    settings = _common.make_settings(args) if not args.mock else check_mock_settings(args)
    settings.ensure_dirs()

    run_id = new_run_id()
    log = ResearchLog(settings.research_log_path, run_id, echo=args.verbose)
    store = Store(settings.db_path)
    store.init_schema()

    tag = normalize_tag(args.tag)
    mode = "mock" if args.mock else "live"
    _common.print_header(
        "02 · 全链路技术验证",
        f"{'离线合成数据' if args.mock else '真实微博'}：检索 → 选帖 → 评论 → 清洗 → 匿名化导出",
    )
    info(f"采集会话 ID：{run_id}")
    info(f"话题：{tag}    排序：{ORDER_LABELS[args.order]}")
    log.record(
        "run_start",
        mode=mode,
        tag=tag,
        order=args.order,
        order_label=ORDER_LABELS[args.order],
        posts_preview=args.posts,
        comments_target=args.comments,
        include_subcomments=args.subcomments,
        clean_tier=args.tier,
    )

    async with AsyncExitStack() as stack:
        if args.mock:
            scenario = build_scenario(
                tag=tag, posts=6, comments_per_post=max(args.comments, 24)
            )
            fetcher = MockFetcher(scenario)
            limiter = RateLimiter(
                delay_range=(0.0, 0.0),
                budget=RequestBudget(limit=10_000),
            )
            info("离线模式：使用合成数据，不发出任何网络请求。")
        else:
            session = await stack.enter_async_context(
                BrowserSession(settings, notify=note, log=log)
            )
            fetcher = session.fetcher
            limiter = RateLimiter(
                delay_range=settings.effective_delay_range(),
                budget=RequestBudget(
                    limit=settings.max_requests_per_run,
                    warn_ratio=settings.budget_warn_ratio,
                ),
                backoff_base_s=settings.backoff_base_s,
                backoff_max_s=settings.backoff_max_s,
            )

        client = WeiboClient(
            fetcher,
            limiter,
            max_retries=settings.max_retries,
            log=log,
            notify=warn,
        )

        # ---------------- 1. 登录态 ---------------- #
        console().print()
        console().print("步骤 1 / 6  校验登录态", markup=False)
        try:
            logged_in = await client.check_login()
        except WrcError as exc:
            if args.mock:
                raise
            report_error(exc)
            info("提示：先运行 python spikes/01_login.py 建立登录态。")
            return EXIT_USER_ERROR
        if not logged_in:
            report_error(AuthRequired(detail="check_login() 返回 false"))
            info("提示：先运行 python spikes/01_login.py 建立登录态。")
            return EXIT_USER_ERROR
        info("✓ 登录态有效")

        # ---------------- 2. 检索 ---------------- #
        console().print()
        console().print("步骤 2 / 6  检索话题下的帖子", markup=False)
        posts = await client.search_posts(
            tag, limit=args.posts, order=args.order, max_pages=args.max_pages
        )
        info(f"✓ 拿到 {len(posts)} 篇候选帖子")
        store.create_run(
            run_id=run_id,
            tag=tag,
            order_mode=args.order,
            params={
                "mode": mode,
                "posts_preview": args.posts,
                "comments_target_per_post": args.comments,
                "include_subcomments": args.subcomments,
                "clean_tier": args.tier,
                "delay_range_s": list(settings.effective_delay_range()),
                "max_requests": settings.max_requests_per_run,
            },
        )
        store.save_posts(run_id, posts)
        console().print()
        render_posts(posts)

        # ---------------- 3. 选帖 ---------------- #
        console().print()
        console().print("步骤 3 / 6  选择目标帖子", markup=False)
        selected = choose_posts(posts, args)
        for post in selected:
            info(f"→ mid={post.mid}  评论数={post.comments_count:,}  {post.preview(30)}")
        log.record(
            "posts_selected",
            count=len(selected),
            mids=[post.mid for post in selected],
        )

        # ---------------- 4. 采集评论 ---------------- #
        console().print()
        console().print("步骤 4 / 6  采集评论", markup=False)
        stopped_by_budget = False
        unreadable_posts: list[str] = []
        for index, post in enumerate(selected, start=1):
            info(f"({index}/{len(selected)}) {post.mid} 目标 {args.comments} 条 …")
            try:
                result = await client.collect_comments(
                    post.mid,
                    target=args.comments,
                    include_subcomments=args.subcomments,
                    backend=args.comment_backend,
                    sort=args.comment_sort,
                    expected_total=post.comments_count,
                )
            except BudgetExceeded as exc:
                report_error(exc)
                warn("已保留此前采到的全部数据，继续进入清洗与导出。")
                log.record("budget_exceeded", post_mid=post.mid,
                           used=client.budget_used, limit=client.budget_limit)
                stopped_by_budget = True
                break
            except CommentSourceEmpty as exc:
                # 绝不把"接口没给数据"当成"这个帖子没有评论"继续往下走
                report_error(exc)
                unreadable_posts.append(post.mid)
                log.record(
                    "comments_source_empty",
                    post_mid=post.mid,
                    expected_total=post.comments_count,
                )
                warn(f"已跳过该帖。建议运行：python spikes/04_probe_comments.py --mid {post.mid}")
                continue
            store.save_comments(run_id, result.comments)
            backend_note = result.backend or "未知后端"
            sort_note = (
                COMMENT_SORT_LABELS[result.sort]
                if result.sort_applied
                else "接口不支持排序（按平台默认热门流）"
            )
            info(
                f"    ✓ 采到 {result.count} 条（翻了 {result.pages_fetched} 页，"
                f"停止原因：{result.stopped_reason}）"
            )
            info(f"    数据来源：{backend_note}，排序：{sort_note}")
            log.record(
                "comments_collected",
                post_mid=post.mid,
                collected=result.count,
                pages=result.pages_fetched,
                stopped_reason=result.stopped_reason,
                platform_total=result.total_number,
                backend=result.backend,
                sort=result.sort,
                sort_applied=result.sort_applied,
            )

        counts = store.counts(run_id)
        if counts["raw_comments"] == 0:
            warn("没有采到任何评论，后续清洗与导出跳过。")
            if unreadable_posts:
                warn(
                    "这是采集中断，不是「这些帖子没有评论」。"
                    "请先运行评论接口诊断：python spikes/04_probe_comments.py "
                    f"--mid {unreadable_posts[0]}"
                )
            log.record("abort", reason="no_comments", unreadable=unreadable_posts)
            return EXIT_USER_ERROR
        info(f"✓ 本次共入库 {counts['raw_comments']} 条评论")

        # ---------------- 5. 清洗 ---------------- #
        console().print()
        console().print("步骤 5 / 6  清洗（可解释、可回退）", markup=False)
        source_rows = store.load_comments(run_id)
        cleaner = Cleaner(tier=args.tier)
        cleaner.dedupe_scope = args.dedupe_scope
        cleaner_result = cleaner.run(source_rows)
        report = build_report(
            cleaner_result, source_rows, sample_size=args.samples, seed=args.seed
        )
        store.save_clean(
            run_id,
            cleaner_result.rows,
            clean_tier=args.tier,
            pipeline_version=cleaner_result.pipeline_version,
        )
        console().print()
        block(report.render())
        log.record(
            "cleaning_done",
            tier=args.tier,
            pipeline_version=cleaner_result.pipeline_version,
            total_input=report.total_input,
            total_kept=report.total_kept,
            retention_rate=round(report.retention_rate, 4),
            steps={stat.key: stat.removed for stat in report.steps},
            sample_seed=report.seed,
        )

        # ---------------- 6. 导出 ---------------- #
        console().print()
        console().print("步骤 6 / 6  匿名化导出", markup=False)
        confirmed = ask_consent(args.consent)
        if not confirmed:
            warn("未确认数据使用声明，导出已取消。清洗结果已入库，可随时重新导出。")
            log.record("export_cancelled", reason="consent_not_given")
            return EXIT_USER_ERROR
        consent_record = confirm(True, source="cli_flag" if args.consent else "cli_prompt")

        mode_label = "离线合成数据（演示用，不可用于研究）" if args.mock else "真实采集"
        method_params = {
            "数据性质": mode_label,
            "检索排序": ORDER_LABELS[args.order],
            "候选帖子数": args.posts,
            "实际选中帖子数": len(selected),
            "每帖评论目标数": args.comments,
            "评论来源后端": args.comment_backend,
            "评论排序": (
                COMMENT_SORT_LABELS[args.comment_sort]
                if args.comment_sort == "time"
                else "按热度（平台热门流 / flow=0）"
            ),
            "二级评论": "已采集" if args.subcomments else "未采集",
            "请求间延迟": f"{settings.delay_min_s}–{settings.delay_max_s} 秒随机",
            "本次实际请求数": f"{client.budget_used} / {client.budget_limit}",
        }
        exporter = Exporter(
            store,
            hasher=PseudoIdHasher(settings.pseudo_salt_path),
            export_root=settings.export_dir,
        )
        manifests = [
            exporter.export_raw(run_id, consent=consent_record, method=method_params),
            exporter.export_analysis(run_id, consent=consent_record, method=method_params),
        ]
        for manifest in manifests:
            manifest.write()
            write_readme(manifest.out_dir / "README.txt", manifest=manifest)

        out_dir = manifests[0].out_dir.parent
        run_row = store.get_run(run_id) or {}
        method_path = write_method_note(
            out_dir / "method_note.md",
            run={
                "run_id": run_id,
                "tag": tag,
                "created_at": run_row.get("created_at", ""),
            },
            params=method_params,
            cleaning=report.to_dict(),
            manifest_rows={
                item.path.name: item.rows
                for manifest in manifests
                for item in manifest.files
            },
        )
        log.record(
            "export_done",
            out_dir=out_dir,
            files={item.path.name: item.rows
                   for manifest in manifests for item in manifest.files},
        )

        # ---------------- 收尾 ---------------- #
        store.finish_run(
            run_id,
            status="partial" if stopped_by_budget else "done",
            note=None if not stopped_by_budget else "budget_exceeded",
        )
        log_path = log.write_markdown(out_dir / "research_log.md")
        (out_dir / "research_log.jsonl").write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in log.events) + "\n",
            encoding="utf-8",
        )

        console().print()
        console().print("完成", markup=False)
        console().print(f"  采集会话   {run_id}", markup=False)
        console().print(f"  原始评论   {counts['raw_comments']} 条", markup=False)
        console().print(f"  清洗后保留 {report.total_kept} 条（{report.retention_rate:.1%}）",
                        markup=False)
        console().print(f"  实际请求   {client.budget_used} 次", markup=False)
        console().print(f"  导出目录   {out_dir}", markup=False)
        console().print(f"  方法说明   {method_path}", markup=False)
        console().print(f"  研究日志   {log_path}", markup=False)
        console().print()
        if args.mock:
            warn("本次为离线合成数据，仅用于验证管线是否可用，不能用于任何研究结论。")
        if stopped_by_budget:
            warn("本次因请求上限提前停止，样本量可能不足，请缩小范围后重新采集。")
        return EXIT_OK


def check_mock_settings(args: argparse.Namespace):
    """离线模式不碰网络，但配置校验仍需走一遍。"""
    from wrc.config import get_settings

    payload: dict[str, object] = {}
    if getattr(args, "delay_min", None):
        payload["delay_min_s"] = args.delay_min
    if getattr(args, "delay_max", None):
        payload["delay_max_s"] = args.delay_max
    return get_settings(**payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="全链路技术验证：检索 → 评论 → 清洗 → 导出")
    add_common_arguments(parser)
    parser.add_argument("--tag", default="延迟退休", help="话题标签，可不带井号")
    parser.add_argument(
        "--order",
        choices=("engagement", "time", "platform"),
        default="engagement",
        help="帖子排序（默认按互动总量，注意这是本地重排，不等于平台排序）",
    )
    parser.add_argument("--posts", type=int, default=20, help="预览的候选帖子数")
    parser.add_argument("--max-pages", type=int, default=3, help="检索最多翻几页")
    parser.add_argument("--select", default=None, help="要抓评论的帖子编号，如 1,3")
    parser.add_argument("--auto-select", type=int, default=None, help="自动选择前 N 篇")
    parser.add_argument("--comments", type=int, default=50, help="每帖评论目标数")
    parser.add_argument("--subcomments", action="store_true", help="是否采集二级评论（默认关闭）")
    parser.add_argument(
        "--comment-backend",
        choices=("auto", *sorted(BACKENDS_BY_KEY)),
        default="auto",
        help="评论来源后端；auto 会依次尝试，实测用 04_probe_comments.py 确认可用项",
    )
    parser.add_argument(
        "--comment-sort",
        choices=("hot", "time"),
        default="hot",
        help="评论排序：hot 按热度 / time 按时间（仅网页端后端支持，移动端会忽略并记录）",
    )
    parser.add_argument(
        "--tier", choices=("none", "light"), default="light", help="清洗档位（第 1 步仅 light/none）"
    )
    parser.add_argument(
        "--dedupe-scope", choices=("post", "global"), default="post",
        help="精确去重的范围：帖子内（默认）或全局",
    )
    parser.add_argument("--samples", type=int, default=20, help="清洗报告中抽查的样本数")
    parser.add_argument("--seed", type=int, default=20240101, help="抽查随机种子（保证可复现）")
    parser.add_argument("--consent", action="store_true", help="预先确认数据使用声明")
    parser.add_argument("--mock", action="store_true", help="使用离线合成数据，不访问网络")
    parser.add_argument("--verbose", action="store_true", help="实时打印研究日志事件")
    return parser


if __name__ == "__main__":
    sys.exit(_common.run_async(lambda: run(build_parser().parse_args())))
