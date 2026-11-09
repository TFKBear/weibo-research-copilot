"""正式命令行入口（P0 收口）。

`spikes/` 下那几个脚本是**验证记录**，保持原样，不随产品演进。
产品入口是这里：

    python -m wrc.cli doctor                  # 环境自检（登录、AI、浏览器）
    python -m wrc.cli login                   # 扫码登录
    python -m wrc.cli preview --tag 小米SU7    # 看看有哪些帖子
    python -m wrc.cli run --tag 小米SU7 --select 1,2 --comments 50 --consent

图形界面入口：`python -m wrc.gui`（或双击项目根目录的"启动博研助手.vbs"）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .clean.basic import TIER_META
from .config import PROJECT_ROOT, get_settings
from .console import banner, block, console, fail, info, note, ok, report_error, warn
from .errors import ApiError, WrcError
from .pipeline import CleanOptions, CollectOptions, ResearchPipeline
from .privacy.consent import confirm, render_statement

EXIT_OK = 0
EXIT_USER_ERROR = 2
EXIT_INTERRUPTED = 130

_ORDER_LABELS = {
    "engagement": "按互动总量（本地重排）",
    "time": "按发帖时间（本地重排）",
    "platform": "按接口返回顺序（平台相关度）",
}


# --------------------------------------------------------------------------- #
def _emit(level: str, message: str) -> None:
    if level == "ok":
        ok(message)
    elif level == "warn":
        warn(message)
    elif level == "error":
        fail(message)
    else:
        info(message)


def _run_async(main) -> int:
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        warn("已中断。已完成部分保存在 data/ 目录。")
        return EXIT_INTERRUPTED
    except WrcError as exc:
        report_error(exc)
        return EXIT_USER_ERROR
    except Exception as exc:  # noqa: BLE001
        report_error(exc)
        return EXIT_USER_ERROR


# --------------------------------------------------------------------------- #
def cmd_doctor(args: argparse.Namespace) -> int:
    from .ai.status import detect_status

    settings = get_settings()
    settings.ensure_dirs()
    banner("环境自检")

    console().print("路径", markup=False)
    console().print(f"  数据目录    {settings.data_dir}", markup=False)
    console().print(f"  导出目录    {settings.export_dir}", markup=False)
    console().print(f"  登录态文件  {settings.storage_state_path}"
                    f"（{'已存在' if settings.storage_state_path.exists() else '尚未创建'}）",
                    markup=False)

    console().print()
    console().print("AI（可选）", markup=False)
    status = detect_status(settings.ai_model_dir)
    console().print(f"  {status.summary()}", markup=False)
    console().print(f"  模型目录    {status.model_dir}", markup=False)
    for line in status.remediation():
        console().print(f"  · {line}", markup=False)

    console().print()
    console().print("清洗档位", markup=False)
    for key, meta in TIER_META.items():
        flag = "（需要 AI）" if meta["needs_ai"] else ""
        console().print(f"  {key:<9}{meta['label']}{flag} — {meta['summary']}", markup=False)
    console().print()

    info("检查登录态与浏览器 …")
    from .session import BrowserSession

    async def check() -> int:
        async with BrowserSession(settings, notify=note) as session:
            state = await session.describe_state()
            info(f"浏览器引擎：{state['engine']}；Cookie 数量：{state['cookie_count']}")
            try:
                logged = await session.check_login()
            except WrcError as exc:
                report_error(exc)
                return EXIT_USER_ERROR
            if logged:
                ok("登录态有效，可以开始采集。")
                return EXIT_OK
            warn("尚未登录或登录态已过期，运行：python -m wrc.cli login")
            return EXIT_USER_ERROR

    return _run_async(check)


def cmd_login(args: argparse.Namespace) -> int:
    from .session import BrowserSession

    settings = get_settings()
    settings.ensure_dirs()
    banner("登录")
    if args.cookie:
        raw = sys.stdin.read() if args.cookie == "-" else args.cookie
    else:
        raw = None

    async def flow() -> int:
        async with BrowserSession(settings, notify=note) as session:
            if raw:
                done = await session.import_cookie_string(raw)
            elif args.check:
                done = False
            else:
                done = await session.login_interactive()
            if args.check:
                done = await session.check_login()
            state = await session.describe_state()
            console().print()
            console().print(
                f"  是否已登录 {done}；Cookie {state['cookie_count']} 个；"
                f"引擎 {state['engine']}",
                markup=False,
            )
            return EXIT_OK if done else EXIT_USER_ERROR

    return _run_async(flow)


def cmd_ai(args: argparse.Namespace) -> int:
    from .ai.status import detect_status

    settings = get_settings()
    status = detect_status(settings.ai_model_dir)
    banner("AI 状态")
    console().print(f"  {status.summary()}", markup=False)
    console().print(f"  运行库      {status.runtime_detail}", markup=False)
    console().print(f"  模型目录    {status.model_dir}", markup=False)
    console().print(f"  权重文件    {status.onnx_path or '（未找到）'}", markup=False)
    console().print(f"  分词器      {status.tokenizer_path or '（未找到）'}", markup=False)
    if status.problems:
        console().print("  问题：", markup=False)
        for problem in status.problems:
            console().print(f"    - {problem}", markup=False)
    if not status.ready:
        console().print()
        console().print("  修复步骤：", markup=False)
        for step in status.remediation():
            console().print(f"    {step}", markup=False)
    return EXIT_OK if status.ready else EXIT_USER_ERROR


def cmd_preview(args: argparse.Namespace) -> int:
    pipeline = ResearchPipeline(get_settings(), emit=_emit)
    opts = _collect_options(args)
    banner(f"检索 {opts.normalised_tag()}")
    posts = _run_pipeline(pipeline.preview_posts(opts))
    if posts is None:
        return EXIT_USER_ERROR

    from rich.table import Table

    table = Table(show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("发帖时间", no_wrap=True)
    table.add_column("互动", justify="right")
    table.add_column("评论数", justify="right")
    table.add_column("摘要", overflow="fold")
    for index, post in enumerate(posts, start=1):
        table.add_row(
            str(index),
            post.created_at.strftime("%Y-%m-%d %H:%M") if post.created_at else "未知",
            f"{post.engagement:,}",
            f"{post.comments_count:,}",
            post.preview(40),
        )
    console().print(table)
    console().print()
    info("用 --select 1,3 指定要采集评论的帖子，例如：")
    console().print(
        f"  python -m wrc.cli run --tag {opts.tag} --select 1,2 "
        f"--comments {opts.comments_per_post} --consent",
        markup=False,
    )
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    pipeline = ResearchPipeline(settings, emit=_emit)
    opts = _collect_options(args)
    clean_opts = CleanOptions(
        tier=args.tier,
        dedupe_scope=args.dedupe,
        use_ai=args.ai,
        filter_bots=args.filter_bots,
    )

    banner(f"全流程：{opts.normalised_tag()}")
    info(f"排序 {_ORDER_LABELS[opts.order]}；每帖目标 {opts.comments_per_post} 条；"
         f"请求上限 {opts.max_requests}")
    info(f"清洗档位 {clean_opts.tier}；去重 {clean_opts.dedupe_scope}；"
         f"AI {'启用' if clean_opts.use_ai else '未启用'}")

    if clean_opts.use_ai:
        status = pipeline.ai_status()
        if not status.ready:
            warn(status.summary())
            for step in status.remediation():
                warn(f"  · {step}")
            return EXIT_USER_ERROR

    posts = _run_pipeline(pipeline.preview_posts(opts))
    if posts is None:
        return EXIT_USER_ERROR

    selected = _pick_posts(posts, args)
    if selected is None:
        return EXIT_USER_ERROR

    # 逐帖目标：--per-post 用「帖子编号=条数」表示，编号对应上面预览列表。
    # 这里才映射成 mid，因为编号是给人看的，mid 才是采集与复现要用的。
    if getattr(args, "per_post", None):
        mapping = _parse_per_post(args.per_post, posts)
        if mapping is None:
            return EXIT_USER_ERROR
        selected_mids = {post.mid for post in selected}
        opts.per_post_targets = {
            mid: count for mid, count in mapping.items() if mid in selected_mids
        }
        info(
            "逐帖目标："
            + "；".join(
                f"第 {index} 篇 = {count}"
                for index, count in sorted(_index_targets(args.per_post))
            )
            + f"（其余按每帖 {opts.comments_per_post} 条）"
        )

    outcome = _run_pipeline(
        _full_flow(pipeline, opts, clean_opts, selected, skip_consent=args.consent)
    )
    if outcome is None:
        return EXIT_USER_ERROR

    # 合并：跑完采集/清洗/导出后接着分析并出报告。
    # 真实运行里研究者采完就停了，拿到的是"有数据没报告"的半成品；
    # 所以默认一路跑到底，只想拿数据的人用 --no-analysis。
    if getattr(args, "no_analysis", False):
        info("已按 --no-analysis 跳过分析与报告；需要时再跑：")
        info(f"  python tools/run_analysis.py --run {outcome}")
        return EXIT_OK

    info("接着分析并生成报告（含情感 / 立场 / 聚类）…")
    code = _delegate_tool("tools/run_analysis.py", ["--run", outcome])
    if code != 0:
        warn("分析没跑完；数据与导出不受影响，可稍后单独重跑分析。")
        return EXIT_USER_ERROR
    code = _delegate_tool("tools/run_report.py", ["--run", outcome])
    if code != 0:
        warn("报告没生成；分析结果已入库，可稍后单独重跑报告。")
        return EXIT_USER_ERROR
    return EXIT_OK


async def _full_flow(pipeline, opts, clean_opts, selected, *, skip_consent: bool):
    run_id, comments, used = await pipeline.collect(opts, selected)
    _emit("ok", f"共入库 {comments} 条评论，用了 {used} 次请求")
    if comments == 0:
        raise ApiError(
            what_happened="没有采到任何评论，流程终止。",
            detail="collect_comments 全部返回空；请先运行 python -m wrc.cli doctor",
        )

    _emit("info", f"开始清洗（档位 {clean_opts.tier}）…")
    report, ai_summary = pipeline.clean(run_id, clean_opts)
    block(report.render())

    confirmed = skip_consent
    if not skip_consent:
        console().print(render_statement(), markup=False)
        console().print()
        if not sys.stdin.isatty():
            warn("非交互环境无法确认声明，请加 --consent。")
            return None
        confirmed = input("确认以上声明并继续导出？输入 y 确认：").strip().lower() in {
            "y", "yes", "是",
        }
    if not confirmed:
        warn("未确认数据使用声明，导出已取消。清洗结果已入库，可稍后重新导出。")
        return None

    consent = confirm(True, source="cli_flag" if skip_consent else "cli_prompt")
    manifest = pipeline.export(
        run_id,
        report=report,
        clean_opts=clean_opts,
        collect_opts=opts,
        consent=consent,
        ai_summary=ai_summary,
        requests_used=used,
    )
    out_dir = manifest.out_dir.parent
    log_path = pipeline.write_research_log(run_id, out_dir)
    appendix_path = pipeline.write_appendix(run_id, out_dir)
    console().print()
    console().print("完成", markup=False)
    console().print(f"  采集会话    {run_id}", markup=False)
    console().print(f"  原始评论    {report.total_input} 条", markup=False)
    console().print(f"  清洗后保留  {report.total_kept} 条（{report.retention_rate:.1%}）",
                    markup=False)
    console().print(f"  导出目录    {out_dir}", markup=False)
    console().print(f"  方法说明    {out_dir / 'method_note.md'}", markup=False)
    console().print(f"  研究日志    {log_path}", markup=False)
    console().print(f"  可复现附录  {appendix_path}", markup=False)
    return run_id


# --------------------------------------------------------------------------- #
def _run_pipeline(coro):
    try:
        return asyncio.run(coro)
    except WrcError as exc:
        report_error(exc)
        return None
    except KeyboardInterrupt:
        warn("已中断。")
        return None


# --------------------------------------------------------------------------- #
def cmd_ask(args: argparse.Namespace) -> int:
    from .assistant import ask, build_context, render_suggestions, suggest_next_steps
    from .llm.client import LlmClient
    from .store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()
    context = build_context(store, settings)

    banner("方法助手")
    console().print("当前状态：", markup=False)
    console().print(context.to_prompt(), markup=False)
    console().print()

    if args.suggest or not args.question:
        console().print(render_suggestions(suggest_next_steps(context)), markup=False)
        if not args.question:
            console().print()
            info('想提问就加一句，例如：python -m wrc.cli ask "我的样本量够做聚类吗？"')
        return EXIT_OK

    client = LlmClient()
    ok, reason = client.available()
    if not ok:
        warn(f"LLM 不可用（{reason}），只能给出规则建议。")
        console().print(render_suggestions(suggest_next_steps(context)), markup=False)
        return EXIT_USER_ERROR

    try:
        answer = ask(args.question, context, client=client)
    except WrcError as exc:
        report_error(exc)
        return EXIT_USER_ERROR
    console().print()
    console().print(answer, markup=False)
    return EXIT_OK


def _log_topic_recommendation(settings, description: str, candidates) -> None:
    """把话题推荐结果追加进研究日志。

    为什么值得记：论文的「样本选择」一节要交代"为什么用这个检索词"。
    如果不留痕，一周后只会剩下一句"当时随便试的"。
    日志写失败绝不能毁掉推荐结果（LLM 调用和探测请求都已经花掉了），所以整段吞异常。
    """
    from datetime import datetime

    from .research_log import ResearchLog

    try:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log = ResearchLog(settings.research_log_path, f"tags-{stamp}")
        log.record(
            "topic_recommend",
            description=description,
            candidates=[
                {
                    "tag": item.tag,
                    "reason": item.reason,
                    "posts": item.probe.posts if item.probe else None,
                    "median_engagement": (
                        item.probe.median_engagement if item.probe else None
                    ),
                    "latest_post_at": item.probe.latest_post_at if item.probe else None,
                    "error": item.probe.error if item.probe else None,
                }
                for item in candidates
            ],
        )
        info(f"推荐结果已记入研究日志：{settings.research_log_path}")
    except Exception:  # noqa: BLE001
        pass


def cmd_tags(args: argparse.Namespace) -> int:
    from .assistant.topic import recommend_tags, render_candidates
    from .llm.client import LlmClient

    settings = get_settings()
    settings.ensure_dirs()
    banner("话题推荐")
    info("先用模型提出候选标签，再逐个真实检索探测量级。")
    info("模型不会给出数量——所有量级数字都来自实际请求。")

    client = LlmClient()
    ok, reason = client.available()
    if not ok:
        warn(f"LLM 不可用（{reason}），无法生成候选标签。")
        return EXIT_USER_ERROR

    async def flow() -> int:
        candidates = await recommend_tags(
            args.description,
            client=client,
            settings=settings,
            limit=max(1, min(args.limit, 8)),
            emit=_emit,
        )
        if not candidates:
            warn("模型没有给出可用的标签。试试换个更具体的描述。")
            return EXIT_USER_ERROR
        console().print()
        console().print(render_candidates(candidates), markup=False)
        console().print()

        # 话题推荐也是研究决策的一部分：为什么选这个词、当时探测到多少，
        # 论文的「样本选择」一节要交代。不记进研究日志，一周后就说不清了。
        _log_topic_recommendation(settings, args.description, candidates)

        alive = [item for item in candidates if item.probe and item.probe.posts > 0]
        if alive:
            info(
                "下一步：选一个标签开始采集，例如\n"
                f'  python -m wrc.cli run --tag "{alive[0].tag}" --auto-select 2 '
                "--comments 50 --consent"
            )
        else:
            warn("所有候选都没有检索到内容，说明这些标签在当前环境下不活跃。")
        return EXIT_OK

    return _run_async(flow)


def cmd_analyze(args: argparse.Namespace) -> int:
    return _delegate_tool(
        "tools/run_analysis.py",
        [
            *(["--run", args.run] if args.run else []),
            *(["--limit", str(args.limit)] if args.limit else []),
            *(["--no-sentiment"] if args.no_sentiment else []),
            *(["--no-cluster"] if args.no_cluster else []),
            *(["--k", str(args.k)] if args.k else []),
        ],
    )


def cmd_report(args: argparse.Namespace) -> int:
    return _delegate_tool(
        "tools/run_report.py", [*(["--run", args.run] if args.run else [])]
    )


def cmd_compare(args: argparse.Namespace) -> int:
    return _delegate_tool(
        "tools/run_compare.py",
        [
            *(["--runs", args.runs] if args.runs else []),
            *(["--all"] if args.all else []),
            *(["--limit", str(args.limit)] if args.limit else []),
        ],
    )


def cmd_appendix(args: argparse.Namespace) -> int:
    """生成可复现性附录（研究日志正式版）。"""
    return _delegate_tool(
        "tools/run_appendix.py",
        [
            *(["--run", args.run] if args.run else []),
            *(["--all"] if args.all else []),
            *(["--no-hash"] if args.no_hash else []),
            *(["--show-paths"] if args.show_paths else []),
        ],
    )


def cmd_clean(args: argparse.Namespace) -> int:
    """对已有会话重跑清洗与导出（把半途失败的采集捞回来）。"""
    return _delegate_tool(
        "tools/run_clean.py",
        [
            *(["--run", args.run] if args.run else []),
            *(["--tier", args.tier] if args.tier else []),
            *(["--dedupe", args.dedupe] if args.dedupe else []),
            *(["--list"] if args.list else []),
            *(["--no-export"] if args.no_export else []),
        ],
    )


def cmd_reclean(args: argparse.Namespace) -> int:
    """用当前代码重跑清洗，与库中记录逐条比对（只读，不写入）。

    这是把「清洗可逐位复现」从口号变成可验证事实的入口。
    刻意**不提供写入模式**：一旦允许覆盖，研究者就可能用它
    "让数据看起来和上次一致"，那反而破坏可复现性。
    """
    from .reproducibility import verify_clean_replay
    from .store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init_schema()
    run_id = args.run or store.latest_run_id()
    if run_id is None:
        warn("还没有采集记录。")
        return EXIT_USER_ERROR

    banner("清洗重放校验")
    result = verify_clean_replay(
        store, run_id, tier=args.tier, dedupe_scope=args.dedupe
    )
    console().print()
    console().print(result.render(), markup=False)
    return EXIT_OK if result.consistent else EXIT_USER_ERROR


def _delegate_tool(script: str, extra: list[str]) -> int:
    """把分析/报告这类重活交给 tools/ 下的脚本。

    为什么不把逻辑搬进 cli.py：那些脚本要用 matplotlib、sklearn 等重依赖，
    而采集链路不该为了一个子命令就把它们全导入。转发进程最省事也最安全。
    """
    import subprocess

    target = PROJECT_ROOT / script
    if not target.exists():
        fail(f"找不到 {script}")
        return EXIT_USER_ERROR
    command = [sys.executable, str(target), *extra]
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(result.returncode)


def _collect_options(args: argparse.Namespace) -> CollectOptions:
    since, until = _window_args(args)
    return CollectOptions(
        tag=args.tag,
        order=args.order,
        post_limit=args.posts,
        search_mode=getattr(args, "search_mode", "topic"),
        comments_per_post=args.comments,
        include_subcomments=args.subcomments,
        subcomment_target=getattr(args, "subcomment_target", 10),
        comment_backend=args.comment_backend,
        comment_sort=args.comment_sort,
        max_requests=args.max_requests,
        since=since,
        until=until,
    )


def _window_args(args: argparse.Namespace):
    """解析 ``--since/--until``；格式错误直接报人能看懂的话。"""
    from .weibo.window import parse_window_time

    since = until = None
    for name, end_of_day in (("since", False), ("until", True)):
        raw = getattr(args, name, None)
        if not raw:
            continue
        try:
            value = parse_window_time(raw, end_of_day=end_of_day)
        except ValueError as exc:
            fail(f"--{name}：{exc}")
            raise SystemExit(EXIT_USER_ERROR) from exc
        if name == "since":
            since = value
        else:
            until = value
    if since and until and since > until:
        fail(
            f"时间窗起点晚于终点（{since:%Y-%m-%d %H:%M} > {until:%Y-%m-%d %H:%M}）。"
            "这样一条评论也采不到。"
        )
        raise SystemExit(EXIT_USER_ERROR)
    return since, until


def _index_targets(spec: str) -> list[tuple[int, int]]:
    """把 ``"1=200,3=50"`` 解析成 ``[(1, 200), (3, 50)]``；非法就抛 ValueError。"""
    pairs: list[tuple[int, int]] = []
    for part in (spec or "").split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"「{chunk}」缺少 = 号，应写成 1=200")
        left, right = chunk.split("=", 1)
        try:
            index = int(left.strip())
            count = int(right.strip())
        except ValueError as exc:
            raise ValueError(f"「{chunk}」里有非数字") from exc
        if index < 1:
            raise ValueError(f"帖子编号要从 1 开始，收到 {index}")
        if not 1 <= count <= 500:
            raise ValueError(f"目标条数要在 1–500 之间，收到 {count}")
        pairs.append((index, count))
    if not pairs:
        raise ValueError("没有解析到任何一项")
    return pairs


def _parse_per_post(spec: str, posts) -> dict[str, int] | None:
    """把编号写法映射成 ``{mid: 条数}``，编号对应预览列表的行号。"""
    try:
        pairs = _index_targets(spec)
    except ValueError as exc:
        fail(f"--per-post 无法解析：{exc}。写法示例：--per-post 1=200,3=50")
        return None
    mapping: dict[str, int] = {}
    total = len(posts)
    for index, count in pairs:
        if not 1 <= index <= total:
            fail(f"--per-post 里的编号 {index} 超出范围（可选 1–{total}）")
            return None
        mapping[posts[index - 1].mid] = count
    return mapping


def _pick_posts(posts, args: argparse.Namespace):
    if args.select:
        try:
            indices = [int(part) for part in args.select.split(",") if part.strip()]
        except ValueError:
            fail("--select 需要是逗号分隔的编号，例如 1,2")
            return None
        invalid = [i for i in indices if i < 1 or i > len(posts)]
        if invalid:
            fail(f"--select 编号超出范围：{invalid}（可选 1–{len(posts)}）")
            return None
        return [posts[i - 1] for i in indices]
    if args.auto_select:
        count = max(1, min(args.auto_select, len(posts)))
        warn(f"按 --auto-select 自动选择前 {count} 篇帖子。")
        return posts[:count]
    fail("请用 --select 1,3 或 --auto-select 2 明确指定要采集的帖子。")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wrc.cli", description="博研助手 · 微博舆论研究工具"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="环境自检（登录 / AI / 浏览器）")

    login = sub.add_parser("login", help="扫码登录并保存登录态")
    login.add_argument("--cookie", nargs="?", const="-", default=None,
                       help="降级路径：粘贴 Cookie 字符串，传 '-' 从标准输入读")
    login.add_argument("--check", action="store_true", help="只检查现有登录态")

    sub.add_parser("ai", help="查看 AI 模型状态与修复步骤")

    ask = sub.add_parser("ask", help="向方法助手提问（结合当前数据状态）")
    ask.add_argument("question", nargs="?", default=None, help="你的问题")
    ask.add_argument(
        "--suggest", action="store_true", help="不看 LLM，直接列出规则驱动的下一步建议"
    )

    analyze = sub.add_parser("analyze", help="分析：情感 + 聚类 + 图表 + 报告")
    analyze.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    analyze.add_argument("--limit", type=int, default=None, help="情感分析最多处理多少条")
    analyze.add_argument("--no-sentiment", action="store_true")
    analyze.add_argument("--no-cluster", action="store_true")
    analyze.add_argument("--k", type=int, default=None, help="强制聚类数")

    report = sub.add_parser("report", help="从数据库重建报告与图表（不重跑分析）")
    report.add_argument("--run", default=None)

    compare = sub.add_parser(
        "compare", help="跨话题对比（会先判断这些采集能不能比）"
    )
    compare.add_argument("--runs", default=None, help="逗号分隔的采集会话 ID")
    compare.add_argument("--all", action="store_true", help="包含 mock 会话")
    compare.add_argument("--limit", type=int, default=6, help="最多比较几次采集")

    appendix = sub.add_parser(
        "appendix", help="生成可复现性附录（环境 / 参数 / 校验和 / 复现等级）"
    )
    appendix.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    appendix.add_argument("--all", action="store_true", help="为所有会话各生成一份")
    appendix.add_argument(
        "--no-hash", action="store_true", help="跳过模型与文件指纹（更快）"
    )
    appendix.add_argument(
        "--show-paths", action="store_true",
        help="在附录里写入本机路径（默认不写）",
    )

    reclean = sub.add_parser(
        "reclean", help="用当前代码重跑清洗并逐条比对（只读，不改数据）"
    )
    reclean.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    reclean.add_argument("--tier", default=None, help="覆盖档位（默认用库中记录的）")
    reclean.add_argument(
        "--dedupe", default=None, choices=("global", "post", "off"),
        help="覆盖去重范围（默认用采集参数里记录的）",
    )

    clean = sub.add_parser(
        "clean", help="对已有会话重跑清洗与导出（不重新采集，可救回半途失败的采集）"
    )
    clean.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    clean.add_argument("--list", action="store_true", help="列出所有会话与评论数")
    clean.add_argument(
        "--tier", choices=("none", "light", "moderate", "heavy"), default=None,
        help="清洗档位；默认沿用上次",
    )
    clean.add_argument(
        "--dedupe", choices=("global", "post", "off"), default=None,
        help="去重范围；默认沿用采集参数",
    )
    clean.add_argument("--no-export", action="store_true", help="只清洗，不导出")

    tags = sub.add_parser("tags", help="话题推荐（每个候选都会真实检索探测量级）")
    tags.add_argument("description", help="描述你关心的事件/人物/议题")
    tags.add_argument("--limit", type=int, default=5, help="最多探测几个候选标签")

    preview = sub.add_parser("preview", help="检索话题下的候选帖子")
    _add_collect_args(preview)

    run = sub.add_parser("run", help="全流程：检索 → 采集 → 清洗 → 导出")
    _add_collect_args(run)
    run.add_argument("--select", default=None, help="要采集的帖子编号，如 1,3")
    run.add_argument("--auto-select", type=int, default=None, help="自动选择前 N 篇")
    run.add_argument(
        "--tier",
        choices=("none", "light", "moderate", "heavy"),
        default="moderate",
        help="清洗档位；heavy 需要 AI 模型",
    )
    run.add_argument(
        "--dedupe",
        choices=("global", "post", "off"),
        default="global",
        help="去重范围：global 跨帖严格去重（默认）/ post 仅帖内 / off 不去重",
    )
    run.add_argument("--ai", action="store_true", help="启用 AI 辅助清洗（需先准备模型）")
    run.add_argument(
        "--filter-bots",
        action="store_true",
        help="过滤疑似僵尸号（仅依据粉丝数 <5，未校验注册时间，可能误伤）",
    )
    run.add_argument("--consent", action="store_true", help="预先确认数据使用声明")
    run.add_argument(
        "--no-analysis", action="store_true",
        help="只采集/清洗/导出，不跑分析与报告（默认一路跑到底）",
    )

    for item in (parser,):
        item.set_defaults(handler=None)
    sub.choices["ask"].set_defaults(handler=cmd_ask)
    sub.choices["tags"].set_defaults(handler=cmd_tags)
    sub.choices["analyze"].set_defaults(handler=cmd_analyze)
    sub.choices["report"].set_defaults(handler=cmd_report)
    sub.choices["compare"].set_defaults(handler=cmd_compare)
    sub.choices["appendix"].set_defaults(handler=cmd_appendix)
    sub.choices["reclean"].set_defaults(handler=cmd_reclean)
    sub.choices["clean"].set_defaults(handler=cmd_clean)
    sub.choices["doctor"].set_defaults(handler=cmd_doctor)
    sub.choices["login"].set_defaults(handler=cmd_login)
    sub.choices["ai"].set_defaults(handler=cmd_ai)
    sub.choices["preview"].set_defaults(handler=cmd_preview)
    sub.choices["run"].set_defaults(handler=cmd_run)
    return parser

def _add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", required=True, help="话题标签，可不带井号")
    parser.add_argument(
        "--order",
        choices=("engagement", "time", "platform"),
        default="engagement",
        help="帖子排序（engagement/time 为本地重排，不等于平台排序）",
    )
    parser.add_argument("--posts", type=int, default=20, help="预览的候选帖子数")
    parser.add_argument(
        "--search-mode",
        choices=("topic", "keyword"),
        default="topic",
        help="检索口径：topic=话题聚合（q=#话题#，默认）；"
             "keyword=关键词全文检索（q=关键词，会混入仅提及该词的帖子）",
    )
    parser.add_argument("--comments", type=int, default=50, help="每帖评论目标数")
    parser.add_argument("--subcomments", action="store_true", help="是否采集二级评论")
    parser.add_argument(
        "--comment-backend",
        default="auto",
        help="评论来源后端：auto（默认）/ mobile_hotflow / web_build_comments / "
             "mobile_hotflow_reload",
    )
    parser.add_argument(
        "--comment-sort",
        choices=("hot", "time"),
        default="hot",
        help="评论排序（仅网页端后端支持；移动端会忽略并记录）",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="评论时间窗起点，如 2026-09-01 或 '2026-09-01 14:30'（含当天 00:00）",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="评论时间窗终点（只填日期时含当天 23:59）",
    )
    parser.add_argument(
        "--per-post",
        default=None,
        metavar="1=200,3=50",
        help="逐帖评论目标数（编号对应预览列表）；未列出的帖子用 --comments",
    )
    parser.add_argument(
        "--subcomment-target",
        type=int,
        default=10,
        help="每条一级评论最多补抓多少条二级评论（需 --subcomments）",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=120,
        help="单次采集请求硬上限（10–500）",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    limit = getattr(args, "max_requests", None)
    if limit is not None and (limit < 10 or limit > 500):
        console().print("--max-requests 需要在 10–500 之间。")
        return EXIT_USER_ERROR
    if args.handler is None:  # pragma: no cover - argparse 已保证
        parser.print_help()
        return EXIT_USER_ERROR
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
