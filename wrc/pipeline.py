"""研究流程编排层。

GUI 与 CLI 都只是这一层的两个壳。所有业务步骤（检索 / 采集 / 清洗 / 导出）
在这里定义一次，两边共用，避免"界面里能做的事命令行做不到"这类分叉。

约定：这一层不打印任何东西，只通过 ``emit(level, message)`` 把进度抛给外壳。
理由：GUI 要写日志窗口，CLI 要写终端，两者不能共用 print。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .clean.basic import Cleaner, TIER_META, TIER_STEPS
from .clean.report import CleaningReport, build_report
from .config import Settings
from .errors import (
    AIUnavailable, AuthRequired, ConfigProblem, EmptyResult, WrcError,
)
from .export import ExportManifest, Exporter, write_method_note, write_readme
from .privacy.anonymize import PseudoIdHasher
from .privacy.consent import ConsentRecord
from .research_log import ResearchLog, new_run_id
from .store import Store
from .weibo.collect import WeiboClient
from .weibo.endpoints import normalize_tag
from .weibo.models import WeiboPost
from .weibo.ratelimit import RateLimiter, RequestBudget
from .weibo.window import TimeWindow

Emit = Callable[[str, str], None]

#: 单帖目标评论数的上限。社科研究常要几百条/帖，所以放到 2000；
#: 但它只是"允许"，真实能采多少由帖子评论量与请求上限决定，工具不会假装采满。
MAX_COMMENTS_PER_POST = 2000

#: 估算请求数时假设的单页条数：移动端实测一次 10 条，按最小页来估，
#: 宁可估多（提前警告"会撞上限"），也不要估少（跑到一半被掐断还不解释）。
COMMENTS_PER_PAGE = 10


def estimate_comment_requests(targets: Sequence[int]) -> int:
    """按目标评论数粗估需要多少次评论请求（不含检索与二级评论）。

    只用于**采集前提醒**，不参与限流：真正的硬上限在 ``RequestBudget``。
    """
    total = 0
    for target in targets:
        pages = max(1, -(-int(target) // COMMENTS_PER_PAGE))
        total += pages + 1  # +1 是最后那次"确认没有更多"的请求
    return total


def _noop(level: str, message: str) -> None:  # pragma: no cover
    return None


def build_method_params(
    *,
    collect_opts: "CollectOptions",
    clean_opts: "CleanOptions",
    requests_used: int = 0,
    attempted: int = 0,
    succeeded: int = 0,
    ai_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """方法说明用的口径表。

    抽成独立函数是有原因的：`method_note.md` 现在既能导出时写，
    也能事后用 ``tools/refresh_exports.py`` 重生成。两处若各写一份，
    迟早出现"导出时的说明"和"重生成的说明"不一致——那正是可追溯性最怕的事。
    """
    skipped = max(0, attempted - succeeded)
    selected_text = (
        f"{attempted} 篇（成功 {succeeded} 篇"
        + (f"，跳过 {skipped} 篇" if skipped else "")
        + "）"
        if attempted
        else "-"
    )
    params: dict[str, Any] = {
        "检索排序": collect_opts.order,
        "检索口径": (
            "话题聚合（q=#话题#）"
            if collect_opts.search_mode == "topic"
            else "关键词全文检索（q=关键词，含仅提及该词的帖子）"
        ),
        "候选帖子数": collect_opts.post_limit,
        "实际选中帖子数": selected_text,
        "每帖评论目标数": collect_opts.comments_per_post,
        "逐帖目标覆盖": (
            "；".join(
                f"{mid}={n}" for mid, n in sorted(collect_opts.per_post_targets.items())
            )
            if collect_opts.per_post_targets
            else "无（全部按全局值）"
        ),
        "评论来源后端": collect_opts.comment_backend,
        "评论排序": collect_opts.comment_sort,
        "评论时间窗": TimeWindow(
            since=collect_opts.since, until=collect_opts.until
        ).describe(),
        "二级评论": "已采集" if collect_opts.include_subcomments else "未采集",
        "请求上限": collect_opts.max_requests,
        "本次实际请求数": requests_used or "-",
        "清洗档位": (
            f"{clean_opts.tier}"
            f"（{TIER_META.get(clean_opts.tier, {}).get('label', '')}）"
        ),
        "去重方式": {
            "global": "跨帖严格去重（全局）",
            "post": "仅帖内去重",
            "off": "未去重（用于研究重复话术）",
        }.get(clean_opts.dedupe_scope, clean_opts.dedupe_scope),
        "僵尸号过滤": (
            "已启用（仅依据粉丝数 <5，未校验注册时间）"
            if clean_opts.filter_bots
            else "未启用"
        ),
        "AI 辅助": "启用" if clean_opts.use_ai else "未启用",
    }
    if ai_summary:
        params["AI 模型"] = ai_summary.get("model_name", "-")
        params["AI 阈值"] = (
            f"营销 {ai_summary.get('marketing_threshold')} / "
            f"协同 {ai_summary.get('coordinated_threshold')}"
        )
    return params


@dataclass(slots=True)
class CollectOptions:
    """一次采集的全部参数。"""

    tag: str
    order: str = "engagement"
    post_limit: int = 20
    search_mode: str = "topic"
    """``topic`` = 话题聚合（q=#话题#）；``keyword`` = 关键词全文检索（q=关键词）。

    两者拿到的不是同一批帖子：话题聚合里是主动参与的人，全文检索里还混着
    "顺口提到一次"的帖子。混为一谈会让"我在研究话题参与者"这句话不成立。
    """
    comments_per_post: int = 50
    per_post_targets: dict[str, int] = field(default_factory=dict)
    """逐帖覆盖的目标评论数：``{post_mid: n}``；没写的帖子用 ``comments_per_post``。

    需求原文要求「每篇帖子的评论采集数量独立配置，而不是全局固定 1000/2000/5000」，
    理由写得很清楚：全局定值会让**小帖子过度采集、大帖子严重不足**，
    那是抽样偏差，不是采集参数问题。所以这里把覆盖能力放在采集模型里，
    而不是靠用户在两次采集之间反复改全局值。
    """
    include_subcomments: bool = False
    subcomment_target: int = 10
    """每条一级评论最多补抓多少条二级评论。默认 10：楼中楼会让请求量翻几倍。"""
    comment_backend: str = "auto"
    comment_sort: str = "hot"
    max_requests: int = 120
    delay_range: tuple[float, float] | None = None
    since: datetime | None = None
    until: datetime | None = None
    """评论时间窗（闭区间）。设了窗口后 ``comments_per_post``/逐帖目标
    指的是**窗口内**要凑够的条数——采满即停，采不满就如实报告覆盖度。"""

    def normalised_tag(self) -> str:
        return normalize_tag(self.tag)

    def target_for(self, post_mid: str) -> int:
        """该帖的目标评论数：逐帖覆盖优先，非法值回退全局值。"""
        raw = self.per_post_targets.get(str(post_mid))
        if raw is None:
            value = self.comments_per_post
        else:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = self.comments_per_post
        return max(1, min(MAX_COMMENTS_PER_POST, value))

    def targets_for(self, posts: Sequence[Any]) -> dict[str, int]:
        return {str(post.mid): self.target_for(str(post.mid)) for post in posts}


@dataclass(slots=True)
class CleanOptions:
    """一次清洗的全部参数。"""

    tier: str = "moderate"
    dedupe_scope: str = "global"
    """``global``：跨帖严格去重；``post``：仅帖内；``off``：完全不去重。

    用户明确要求：「要研究水军模板就不去重；勾了去重就严格去重」，
    所以默认是 global，而不是只做帖内。
    """
    use_ai: bool = False
    filter_bots: bool = False
    """是否过滤疑似僵尸号。

    默认关闭，因为它只有一半判据：需求要求"粉丝数 <5 且注册 <30 天"，
    但接口不返回注册时间。只按粉丝数删，会系统性删掉真实的低互动用户。
    """

    def resolve_steps(self) -> tuple[str, ...]:
        steps = list(TIER_STEPS.get(self.tier, ()))
        if self.dedupe_scope == "off":
            steps = [key for key in steps if key != "dedupe_exact"]
        if self.filter_bots and "drop_suspected_bots" not in steps:
            steps.append("drop_suspected_bots")
        return tuple(steps)


@dataclass(slots=True)
class RunOutcome:
    """一次完整流程的产出摘要，供界面展示与写日志。"""

    run_id: str
    tag: str
    posts: list[WeiboPost] = field(default_factory=list)
    comment_count: int = 0
    report: CleaningReport | None = None
    ai_summary: dict[str, Any] | None = None
    out_dir: Path | None = None
    method_path: Path | None = None
    files: list[tuple[str, int]] = field(default_factory=list)
    requests_used: int = 0


class ResearchPipeline:
    """编排一次研究流程。"""

    def __init__(
        self,
        settings: Settings,
        *,
        emit: Emit | None = None,
    ) -> None:
        self.settings = settings
        self._emit = emit or _noop
        self.settings.ensure_dirs()
        self.store = Store(settings.db_path)
        self.store.init_schema()

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    def ai_status(self):
        from .ai.status import detect_status

        return detect_status(self.settings.ai_embedding_model_dir)

    def _log(self, run_id: str) -> ResearchLog:
        return ResearchLog(self.settings.research_log_path, run_id)

    def _limiter(self, opts: CollectOptions) -> RateLimiter:
        delay = opts.delay_range or self.settings.effective_delay_range()
        return RateLimiter(
            delay_range=delay,
            budget=RequestBudget(
                limit=opts.max_requests,
                warn_ratio=self.settings.budget_warn_ratio,
            ),
            backoff_base_s=self.settings.backoff_base_s,
            backoff_max_s=self.settings.backoff_max_s,
        )

    def _client(self, session: Any, limiter: RateLimiter, log: ResearchLog) -> WeiboClient:
        return WeiboClient(
            session.fetcher,
            limiter,
            max_retries=self.settings.max_retries,
            log=log,
            notify=lambda message: self._emit("warn", message),
        )

    # ------------------------------------------------------------------ #
    # 1. 检索
    # ------------------------------------------------------------------ #
    async def preview_posts(self, opts: CollectOptions) -> list[WeiboPost]:
        """检索话题下的候选帖子（不落库、不采集评论）。"""
        from .session import BrowserSession

        tag = opts.normalised_tag()
        mode_label = "话题聚合" if opts.search_mode == "topic" else "关键词全文检索"
        self._emit("info", f"正在检索 {tag}（{mode_label}）…")
        async with BrowserSession(
            self.settings, notify=lambda m: self._emit("info", m)
        ) as session:
            limiter = self._limiter(opts)
            log = self._log("preview")
            client = self._client(session, limiter, log)
            if not await client.check_login():
                raise AuthRequired(detail="预览帖子前的登录态检查未通过")
            posts = await client.search_posts(
                tag, limit=opts.post_limit, order=opts.order, mode=opts.search_mode
            )
        self._emit("ok", f"拿到 {len(posts)} 篇候选帖子")
        return posts

    # ------------------------------------------------------------------ #
    # 2. 采集
    # ------------------------------------------------------------------ #
    async def collect(
        self,
        opts: CollectOptions,
        selected: Sequence[WeiboPost],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, int, int]:
        """采集选中帖子的评论。

        Args:
            should_stop: 帖子之间及评论分页之间询问；返回 True 则保留数据并收尾
                （已采到的数据照常保留并入库，不做"半途丢弃"）。

        Returns:
            (run_id, 入库评论数, 实际请求数)
        """
        from .session import BrowserSession

        if not selected:
            raise ConfigProblem(what_happened="没有选中任何帖子，无法开始采集。")

        window = TimeWindow(since=opts.since, until=opts.until)
        if window.since and window.until and window.since > window.until:
            raise ConfigProblem(
                what_happened="时间窗的起点晚于终点，这次采集一条也采不到。",
                detail=(
                    f"起点 {window.since:%Y-%m-%d %H:%M}，"
                    f"终点 {window.until:%Y-%m-%d %H:%M}。请检查是否把起止填反了。"
                ),
            )

        tag = opts.normalised_tag()
        run_id = new_run_id()
        log = self._log(run_id)
        targets = opts.targets_for(selected)
        log.record(
            "run_start",
            mode="gui",
            tag=tag,
            order=opts.order,
            posts=len(selected),
            comments_target=opts.comments_per_post,
            per_post_targets=targets,
            include_subcomments=opts.include_subcomments,
            subcomment_target=opts.subcomment_target,
            comment_backend=opts.comment_backend,
            comment_sort=opts.comment_sort,
            comment_window=window.to_dict(),
            max_requests=opts.max_requests,
        )
        self.store.create_run(
            run_id=run_id,
            tag=tag,
            order_mode=opts.order,
            params={
                "mode": "live",
                "post_limit": opts.post_limit,
                "search_mode": opts.search_mode,
                "comments_per_post": opts.comments_per_post,
                "per_post_targets": targets if opts.per_post_targets else {},
                "include_subcomments": opts.include_subcomments,
                "subcomment_target": opts.subcomment_target,
                "comment_backend": opts.comment_backend,
                "comment_sort": opts.comment_sort,
                "comment_window": window.to_dict(),
                "max_requests": opts.max_requests,
                "delay_range_s": list(opts.delay_range or self.settings.effective_delay_range()),
            },
        )
        self.store.save_posts(run_id, selected)

        # 采集前把"会不会撞上限"算清楚并说出来。
        # 用户设了逐帖目标之后，总和很容易超过请求上限——如果等到中途才被
        # RequestBudget 掐断，用户会以为程序坏了，而实际上是参数互相矛盾。
        planned = estimate_comment_requests(list(targets.values()))
        if window.active:
            planned = int(planned * 2.5)
            self._emit(
                "info",
                f"时间窗：{window.describe()}。窗口内目标 {sum(targets.values())} 条；"
                "窗口外的评论也要占翻页，所以实际请求会多于按条数的估算。",
            )
        if planned > opts.max_requests:
            self._emit(
                "warn",
                f"按当前目标数估算约需 {planned} 次请求，超过本次上限 {opts.max_requests} 次；"
                "采集会在到达上限时停在那一帖（已采数据保留）。"
                "建议降低部分帖子的目标数，或把上限调高。",
            )

        total_comments = 0
        skipped: list[dict[str, str]] = []
        incomplete = False
        halt_kinds = {"auth_required", "risk_blocked", "rate_limited", "budget", "network"}
        async with BrowserSession(
            self.settings, notify=lambda m: self._emit("info", m), log=log
        ) as session:
            limiter = self._limiter(opts)
            client = self._client(session, limiter, log)
            if not await client.check_login():
                raise AuthRequired(detail="采集开始前的登录态检查未通过")

            for index, post in enumerate(selected, start=1):
                if should_stop is not None and should_stop():
                    incomplete = True
                    self._emit("warn", "已按你的要求停止采集，已采到的数据会保留。")
                    log.record("stop_requested", at_post=post.mid)
                    break
                target = targets.get(str(post.mid), opts.comments_per_post)
                self._emit(
                    "info",
                    f"({index}/{len(selected)}) 采集 {post.mid}，目标 {target} 条 …",
                )
                # 单篇帖子的失败不该毁掉整轮采集。
                # 实测踩过：某帖接口返回 ok=0「暂无数据」，
                # 旧代码直接抛出，于是前面辛苦采到的 100 条连清洗都没跑就废了。
                # 现在的规则是：跳过这一篇、记进日志和研究日志、继续下一篇；
                # 只有"一篇都没采到"才算本次采集失败。
                try:
                    result = await client.collect_comments(
                        post.mid,
                        target=target,
                        include_subcomments=opts.include_subcomments,
                        subcomment_target=opts.subcomment_target,
                        backend=opts.comment_backend,
                        sort=opts.comment_sort,
                        expected_total=post.comments_count,
                        since=window.since,
                        until=window.until,
                        should_stop=should_stop,
                    )
                except WrcError as exc:
                    skipped.append({"post_mid": post.mid, "reason": exc.what_happened})
                    self._emit(
                        "warn",
                        f"  {post.mid} 跳过：{exc.what_happened}"
                        + ("（停止本轮）" if exc.kind.value in halt_kinds else "（继续下一篇）"),
                    )
                    log.record(
                        "post_skipped",
                        post_mid=post.mid,
                        kind=exc.kind.value,
                        what_happened=exc.what_happened,
                        detail=exc.detail,
                    )
                    if exc.kind.value in halt_kinds:
                        self._emit("warn", "采集已停止；此前成功的数据将保留。")
                        break
                    continue
                self.store.save_comments(run_id, result.comments)
                total_comments += len(result.comments)
                error_kind = getattr(result, "error_kind", "")
                if result.stopped_reason not in {
                    "reached_target", "no_more_from_platform", "reached_window_start",
                    "post_has_no_comments",
                }:
                    incomplete = True
                if error_kind:
                    self._emit("warn", f"  已保留 {result.count} 条；后续采集未完成："
                               f"{getattr(result, 'error_message', '')}")
                self._emit(
                    "ok",
                    f"  {post.mid} 采到 {result.count} 条（{result.backend}，"
                    f"停止原因：{result.stopped_reason}）",
                )
                window_info = getattr(result, "window", None) or {}
                if (window_info.get("window") or {}).get("active"):
                    # 窗口模式必须把"覆盖到哪"说出来：只报条数会让人误以为窗口采全了
                    self._emit(
                        "info" if window_info.get("complete") else "warn",
                        f"    {window_info.get('text', '')}",
                    )
                for note in (getattr(result, "plan", None) or {}).get("notes", []):
                    self._emit("info", f"    采集口径：{note}")
                if error_kind in halt_kinds:
                    self._emit("warn", "采集已停止；此前成功的数据将保留。")
                    break
            used = limiter.budget.used

        if total_comments == 0:
            # 一条都没有：这才算真的失败，并且要说清楚是"全被跳过"而不是"平台没内容"
            detail = "；".join(
                f"{item['post_mid']}：{item['reason']}" for item in skipped[:3]
            ) or "所有后端都返回空数据"
            raise EmptyResult(
                detail=f"共尝试 {len(selected)} 篇帖子，一条评论都没采到。{detail}",
                what_happened="这次采集没有拿到任何评论（每篇帖子都被跳过了）。",
            )

        self.store.finish_run(
            run_id, status="partial" if skipped or incomplete else "done",
            note=(f"{len(skipped)} 篇帖子被跳过；部分采集未完成"
                  if skipped or incomplete else None),
        )
        log.record("collect_done", run_id=run_id, comments=total_comments,
                   requests=used, skipped=len(skipped))
        return run_id, total_comments, used

    # ------------------------------------------------------------------ #
    # 3. 清洗
    # ------------------------------------------------------------------ #
    def clean(
        self,
        run_id: str,
        opts: CleanOptions,
        *,
        sample_size: int = 20,
        seed: int = 20240101,
    ) -> tuple[CleaningReport, dict[str, Any] | None]:
        """执行清洗并生成影响报告。

        Returns:
            (清洗报告, AI 判定摘要或 None)
        """
        rows = self.store.load_comments(run_id)
        if not rows:
            raise ConfigProblem(what_happened="本次采集没有评论，无法清洗。")

        ai_summary: dict[str, Any] | None = None
        flags: dict[str, list[str]] = {}

        if opts.use_ai:
            flags, ai_summary = self._run_ai(rows)

        cleaner = Cleaner(
            tier=opts.tier,
            dedupe_scope="global" if opts.dedupe_scope == "global" else "post",
            enabled_steps=opts.resolve_steps(),
        )
        result = cleaner.run(rows, external_flags=flags)
        report = build_report(result, rows, sample_size=sample_size, seed=seed)

        self.store.save_clean(
            run_id,
            result.rows,
            clean_tier=opts.tier,
            pipeline_version=result.pipeline_version,
        )
        # 清洗档位决定"占比能不能跨话题比"，所以必须回到采集参数里留痕，
        # 否则跨话题对比只能看到"未记录"。
        self.store.merge_run_params(run_id, {
            "clean_tier": opts.tier,
            "dedupe_scope": opts.dedupe_scope,
            "use_ai": opts.use_ai,
            "filter_bots": opts.filter_bots,
            "pipeline_version": result.pipeline_version,
        })
        self.store.record_event(
            run_id,
            "cleaning_done",
            {
                "tier": opts.tier,
                "dedupe_scope": opts.dedupe_scope,
                "use_ai": opts.use_ai,
                "kept": report.total_kept,
                "removed": report.total_removed,
            },
        )
        return report, ai_summary

    def _run_ai(
        self, rows: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """跑 AI 判定。模型不可用时**报错**，不静默降级。

        为什么报错而不是自动退回规则表：用户明确勾选了"启用 AI"，
        如果悄悄不跑 AI 却照常输出"清洗完成"，那份数据的清洗档位就与实际不符了。
        对研究来说，错误的档位标签比少洗几条更糟。
        """
        from .ai.analysis import analyse
        from .ai.embedder import build_embedder

        status = self.ai_status()
        if not status.ready:
            raise AIUnavailable(
                detail=(
                    f"runtime_ok={status.runtime_ok}；"
                    f"runtime={status.runtime_detail}；"
                    f"problems={'; '.join(status.problems) or '无'}"
                ),
                model_dir=str(self.settings.ai_model_dir),
            )

        self._emit("info", f"正在加载 AI 模型：{status.onnx_path.name} …")
        embedder = build_embedder(status.onnx_path, status.tokenizer_path)

        def progress(done: int, total: int) -> None:
            if done == total or done % 64 == 0:
                self._emit("info", f"  AI 编码进度 {done}/{total}")

        analysis = analyse(
            rows,
            embedder,
            marketing_threshold=self.settings.ai_marketing_threshold,
            coordinated_threshold=self.settings.ai_coordinated_threshold,
            min_cluster_size=self.settings.ai_min_cluster_size,
            on_progress=progress,
        )
        flags: dict[str, list[str]] = {}
        for row in rows:
            cid = str(row.get("cid"))
            reasons = analysis.flags_for(cid)
            if reasons:
                flags[cid] = reasons

        self._emit(
            "ok",
            f"AI 判定完成：营销 {len(analysis.marketing)} 条，"
            f"协同行为 {len(analysis.coordinated)} 条，用时 {analysis.elapsed_s}s",
        )
        return flags, analysis.to_dict()

    # ------------------------------------------------------------------ #
    # 4. 导出
    # ------------------------------------------------------------------ #
    def export(
        self,
        run_id: str,
        *,
        report: CleaningReport,
        clean_opts: CleanOptions,
        collect_opts: CollectOptions,
        consent: ConsentRecord,
        ai_summary: dict[str, Any] | None = None,
        requests_used: int = 0,
    ) -> ExportManifest:
        tag = collect_opts.normalised_tag()
        # 进样率必须写清：真实运行里 20 篇帖子有 9 篇被平台拒绝了，
        # 只写"选中 20 篇"会让论文读者以为这 20 篇都采到了。
        posts = self.store.load_posts(run_id)
        succeeded_posts = {
            str(row.get("post_mid")) for row in self.store.load_comments(run_id)
        }
        attempted = len(posts) or len(collect_opts.per_post_targets)
        method_params = build_method_params(
            collect_opts=collect_opts,
            clean_opts=clean_opts,
            requests_used=requests_used,
            attempted=attempted,
            succeeded=len(succeeded_posts),
            ai_summary=ai_summary,
        )

        exporter = Exporter(
            self.store,
            hasher=PseudoIdHasher(self.settings.pseudo_salt_path),
            export_root=self.settings.export_dir,
        )
        manifests = [
            exporter.export_raw(run_id, consent=consent, method=method_params),
            exporter.export_analysis(run_id, consent=consent, method=method_params),
        ]
        for manifest in manifests:
            manifest.write()
            write_readme(manifest.out_dir / "README.txt", manifest=manifest)

        out_dir = manifests[0].out_dir.parent
        run_row = self.store.get_run(run_id) or {}
        manifest_rows: dict[str, int] = {}
        for manifest in manifests:
            for item in manifest.files:
                manifest_rows[item.path.name] = item.rows
        method_path = write_method_note(
            out_dir / "method_note.md",
            run={
                "run_id": run_id,
                "tag": tag,
                "created_at": run_row.get("created_at", ""),
            },
            params=method_params,
            cleaning=report.to_dict(),
            manifest_rows=manifest_rows,
        )
        return manifests[0]

    # ------------------------------------------------------------------ #
    def write_research_log(self, run_id: str, out_dir: Path) -> Path:
        log = ResearchLog(self.settings.research_log_path, run_id)
        return log.write_markdown(out_dir / "research_log.md")

    def write_appendix(self, run_id: str, out_dir: Path) -> Path:
        """生成可复现性附录（P2「研究日志正式版」）。

        刻意做成"随时可重生成"：附录里的校验和会随文件变化，
        所以导出后、改动数据后都应重跑一次，而不是一次生成永久有效。
        """
        from .reproducibility import write_for_run

        return write_for_run(self.store, self.settings, run_id,
                             path=Path(out_dir) / "appendix.md")


__all__ = [
    "CleanOptions",
    "CollectOptions",
    "ResearchPipeline",
    "RunOutcome",
    "TIER_META",
    "WrcError",
]
