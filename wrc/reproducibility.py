"""可复现性附录（P2「研究日志正式版」）。

需求里这一项的表述是"研究日志正式版"，落到论文里其实就是**一份附录**：
审稿人或导师想复核你的分析，必须能在一个地方查到
「数据从哪来、怎么清洗、用什么模型、参数是什么、结果文件是哪一份」。

这一层和 `research_log.py` 的分工：

- `research_log.py` 记**事件流**（什么时刻发生了什么），给过程留痕；
- 本模块产**结构化附录**（一次会话的完整技术档案），给论文用。

设计上最要紧的一点是**可复现性分级**：不能笼统地写"本流程可复现"。
必须逐环节说清楚哪一段能逐位复现、哪一段不能、为什么。
把 LLM 标注写成"可复现"是这类工具最常见的不诚实——
服务端模型版本和采样随机性都不在你手里。
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import __version__
from .weibo.parsing import CST

#: 写进附录的关键依赖。装没装都要有一条记录，缺失时写"未安装"而不是省略。
TRACKED_DEPENDENCIES = (
    "playwright", "httpx", "pydantic", "jieba",
    "scikit-learn", "numpy", "matplotlib", "onnxruntime", "tokenizers",
    "python-docx",
)

#: 复现等级。
EXACT = "可逐位复现"
APPROX = "可用相同参数近似重采"
NONE = "不可复现"


@dataclass(slots=True)
class ReproducibilityItem:
    """一个环节的复现等级与依据。"""

    stage: str
    level: str
    basis: str
    requirement: str = ""

    def render(self) -> str:
        line = f"| {self.stage} | {self.level} | {self.basis} |"
        return line


@dataclass(slots=True)
class AppendixFacts:
    """附录所需的事实。刻意做成纯数据，便于测试与从别处装配。"""

    run: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    cleaning: dict[str, Any] | None = None
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] | None = None
    #: 本机绝对路径。默认**为空**——附录会流出到论文与公开材料，
    #: 带上本机目录结构既无学术价值，也可能暴露个人信息。需要时用开关打开。
    paths: dict[str, str] = field(default_factory=dict)
    generated_at: str = ""


# --------------------------------------------------------------------------- #
# 采集事实
# --------------------------------------------------------------------------- #
def environment_snapshot() -> dict[str, Any]:
    """记录运行环境。装了什么记什么，没装写"未安装"，不猜版本。"""
    deps: dict[str, str] = {}
    for name in TRACKED_DEPENDENCIES:
        try:
            from importlib.metadata import version

            deps[name] = version(name)
        except Exception:  # noqa: BLE001 - 包缺失或元数据损坏都算"未安装"
            deps[name] = "未安装"
    return {
        "tool_version": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "deps": deps,
        "cwd_note": "路径已省略：附录会流出到论文与公开材料，不写本机目录结构",
    }


def collect_files(export_dir: Path) -> list[dict[str, Any]]:
    """列出导出目录的数据文件与校验和。

    只收数据与文档文件（CSV/MD/JSON），不收 PNG 与 .docx——
    图片与 Word 是派生产物，重新生成即可，列进来只会让附录变长。

    也**不收附录自己**：附录要把别人的校验和写进正文，如果连自己也算，
    写入完成后校验和必然对不上——那是一个一眼看不出来、但确实是假的数字。
    """
    export_dir = Path(export_dir)
    if not export_dir.exists():
        return []
    allowed = {".csv", ".md", ".json", ".jsonl", ".txt"}
    items: list[dict[str, Any]] = []
    for path in sorted(export_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.stem.lower().startswith("appendix"):
            continue  # 自引用，见上
        items.append({
            "name": str(path.relative_to(export_dir)).replace("\\", "/"),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "sha256": sha256_of(path),
        })
    return items


def model_facts(model_dirs: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    """模型文件指纹。

    为什么一定要记模型文件的 SHA-256：聚类结果"可复现"的前提是
    **用的是同一个模型文件**。只说"用了 bge-small-zh"是不够的——
    量化版与全精度版会给出不同的向量，簇的划分也就不同。
    """
    items: list[dict[str, Any]] = []
    for label, directory in model_dirs:
        directory = Path(directory)
        if not directory.exists():
            items.append({"label": label, "dir": directory.name,
                          "state": "目录不存在", "files": []})
            continue
        rows: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".onnx", ".json"}:
                continue
            rows.append({
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "sha256": sha256_of(path) if path.suffix.lower() == ".onnx" else "",
            })
        items.append({
            "label": label,
            "dir": directory.name,
            "state": "就绪" if rows else "目录为空",
            "files": rows,
        })
    return items


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
def cleaning_facts(store: Any, run_id: str) -> dict[str, Any] | None:
    """从库里的清洗结果反推清洗概况。

    为什么不直接读当时的清洗报告：报告是给人看的，没有落库保存步骤明细；
    而 ``clean_comments`` 里每一行都带着 ``kept`` 与 ``drop_reasons``，
    自己统计一遍得到的是**同一份事实**，而且不依赖当时的进程还活着。
    """
    rows = store.load_analysis_rows(run_id, only_kept=False)
    if not rows:
        return None
    kept = [row for row in rows if row.get("kept")]
    total = len(rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if row.get("kept"):
            continue
        for reason in _split_reasons(row.get("drop_reasons")):
            reasons[reason] = reasons.get(reason, 0) + 1
    steps = [
        {
            "label": _reason_label(reason),
            "removed": count,
            "share": count / total if total else 0.0,
        }
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1])
    ]
    first = rows[0]
    return {
        "tier": first.get("clean_tier") or "未记录",
        "pipeline_version": first.get("pipeline_version") or "未记录",
        "total_input": total,
        "total_kept": len(kept),
        "retention_rate": (len(kept) / total) if total else 0.0,
        "steps": steps,
    }


def collect_facts(
    store: Any,
    settings: Any,
    run_id: str,
    *,
    with_hashes: bool = True,
    show_paths: bool = False,
) -> AppendixFacts:
    """从数据库与设置里装配附录事实。"""
    import json

    run = store.get_run(run_id) or {}
    try:
        params = json.loads(run.get("params_json") or "{}")
    except json.JSONDecodeError:
        params = {}
    params.setdefault("mode", "live")

    summaries = store.latest_analysis_summaries(run_id)
    cleaning = cleaning_facts(store, run_id)
    if cleaning:
        # 档位与管线版本以数据为准，参数表里缺失时补上（老会话没有记）
        params.setdefault("clean_tier", cleaning.get("tier"))
        params.setdefault("pipeline_version", cleaning.get("pipeline_version"))

    # 采集口径里最容易被漏掉、也最影响复现的三项：检索口径、时间窗、实际覆盖。
    # 覆盖度由数据反算而不是照抄计划——理由见 analyze/collection.py 的说明。
    try:
        from .analyze.collection import summarise_collection, window_text

        rows = store.load_analysis_rows(run_id)
        summary = summarise_collection(params=params, rows=rows)
        params.setdefault("search_mode", summary.get("search_mode"))
        # 口径的"唯一说法"由 analyze.collection 提供，附录不另写一套
        params.setdefault("comment_window_text", window_text(summary.get("window")))
        params.setdefault(
            "comment_window_label", (summary.get("window") or {}).get("label") or "不限时间"
        )
        params.setdefault("covered_from", summary.get("covered_from"))
        params.setdefault("covered_to", summary.get("covered_to"))

        # 进样率：选中的帖子有几篇真的采到了。真实运行里 20 篇有 9 篇被平台拒绝，
        # 只写"选中 20 篇"会让读者以为全采到了。
        try:
            posts = store.load_posts(run_id)
            reached = {str(row.get("post_mid")) for row in store.load_comments(run_id)}
            if posts:
                skipped = max(0, len(posts) - len(reached))
                params.setdefault(
                    "skipped_posts_text",
                    f"{len(posts)} 篇中成功 {len(reached)} 篇"
                    + (f"，跳过 {skipped} 篇（接口返回暂无数据/帖子不可见）" if skipped else ""),
                )
        except Exception:  # noqa: BLE001 - 进样率算不出来不该让附录失败
            pass
    except Exception:  # noqa: BLE001 - 附录少了这三行也不该写不出来
        params.setdefault("search_mode", params.get("search_mode"))
        params.setdefault("comment_window_label", "未记录")

    export_dir = Path(settings.export_dir) / run_id
    files = collect_files(export_dir) if with_hashes else []

    model_dirs = [
        ("句向量模型（话语聚类）", Path(settings.ai_embedding_model_dir)),
        ("情感分类模型（本地对照通道）", Path(settings.ai_sentiment_model_dir)),
    ]
    models = model_facts(model_dirs) if with_hashes else []

    llm: dict[str, Any] = {}
    sentiment = summaries.get("sentiment") or {}
    stats = (sentiment.get("method") or {}).get("llm_stats") or {}
    if stats:
        llm = {
            "base_url": stats.get("base_url"),
            "model": stats.get("model"),
            "calls": stats.get("calls"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
        }
    elif sentiment:
        method = sentiment.get("method") or {}
        llm = {"base_url": method.get("llm_base_url"), "model": method.get("llm_model")}

    return AppendixFacts(
        run=run,
        params=params,
        cleaning=cleaning,
        summaries=summaries,
        files=files,
        models=models,
        llm=llm,
        env=environment_snapshot(),
        replay=_safe_replay(store, run_id),
        paths=_paths_of(settings, run_id) if show_paths else {},
        generated_at=_now(),
    )


def _paths_of(settings: Any, run_id: str) -> dict[str, str]:
    """只在明确要求时收集本机路径（默认不写，理由见 AppendixFacts.paths）。"""
    items = {
        "数据目录": getattr(settings, "data_dir", ""),
        "导出目录": Path(getattr(settings, "export_dir", "")) / run_id,
        "数据库": getattr(settings, "db_path", ""),
        "句向量模型": getattr(settings, "ai_embedding_model_dir", ""),
        "情感模型": getattr(settings, "ai_sentiment_model_dir", ""),
    }
    return {name: str(value) for name, value in items.items() if value}


def _safe_replay(store: Any, run_id: str) -> dict[str, Any] | None:
    """重放校验失败不应让整份附录生不出来——附录还有别的内容要看。"""
    try:
        return verify_clean_replay(store, run_id).to_dict()
    except Exception:  # noqa: BLE001
        return {"replayable": False, "reason": "重放过程出现异常，已跳过（不影响附录其余部分）"}


def write_for_run(
    store: Any,
    settings: Any,
    run_id: str,
    *,
    path: Path | None = None,
    with_hashes: bool = True,
    show_paths: bool = False,
) -> Path:
    """生成并写出 ``appendix.md``，返回实际路径。"""
    facts = collect_facts(
        store, settings, run_id, with_hashes=with_hashes, show_paths=show_paths
    )
    target = Path(path) if path is not None else (
        Path(settings.export_dir) / run_id / "appendix.md"
    )
    return write_appendix(target, build_appendix(facts))


def _split_reasons(value: Any) -> list[str]:
    """``drop_reasons`` 在库里是 JSON 数组字符串，也可能是 ``a;b``。"""
    import json

    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return [item for item in text.split(";") if item]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


#: 移除原因的中文标签。没收录的代码直接原样输出，不臆造含义。
_REASON_ZH = {
    "dedupe_exact": "精确去重",
    "drop_empty": "清洗后为空",
    "drop_pure_emoji": "纯表情",
    "drop_pure_link": "纯链接",
    "drop_pure_mention": "纯 @ 提及",
    "drop_ad_keywords": "广告关键词",
    "drop_suspected_bots": "疑似僵尸号",
    "ai_marketing": "AI：营销模板",
    "ai_coordinated": "AI：协同行为",
}


def _reason_label(reason: str) -> str:
    """``drop_ad_keywords:加V私信`` → ``广告关键词（加V私信）``。

    保留冒号后的细节：清洗报告的价值就在于能看出"因为哪几个词被删"，
    把细节抹掉只剩类别，研究者就没法判断规则是否误杀。
    """
    base, _, detail = reason.partition(":")
    label = _REASON_ZH.get(base, base)
    return f"{label}（{detail}）" if detail else label


# --------------------------------------------------------------------------- #
# 清洗重放：把"可逐位复现"从一句声明变成可执行检查
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ReplayResult:
    """一次清洗重放的结果。"""

    run_id: str
    tier: str = ""
    dedupe_scope: str = ""
    scope_assumed: bool = False
    filter_bots: bool = False
    filter_bots_inferred: bool = False
    checked: int = 0
    replayable: bool = True
    reason: str = ""
    mismatched: list[dict[str, Any]] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return self.checked - len(self.mismatched)

    @property
    def consistent(self) -> bool:
        return self.replayable and not self.mismatched and self.checked > 0

    def render(self) -> str:
        lines = ["清洗重放校验", "=" * 62]
        if not self.replayable:
            lines.append(f"无法重放：{self.reason}")
            return "\n".join(lines)
        lines.append(f"- 会话：{self.run_id}")
        lines.append(f"- 重放档位：{self.tier}；去重范围：{self.dedupe_scope}"
                     + ("（原会话未记录，按默认推定）" if self.scope_assumed else ""))
        lines.append(
            f"- 僵尸号过滤：{'开启' if self.filter_bots else '关闭'}"
            + ("（原会话未记录，依据库中 drop_reasons 推定）"
               if self.filter_bots_inferred else "")
        )
        lines.append(f"- 逐条比对：{self.checked} 条，一致 {self.matched} 条，"
                     f"不一致 {len(self.mismatched)} 条")
        lines.append("")
        if self.consistent:
            lines.append(
                "结论：**重放结果与库中记录完全一致**。"
                "这条链路上的「可逐位复现」是可验证的，不是声明。"
            )
        else:
            lines.append("结论：**重放结果与库中记录不一致**，请勿在论文中声称清洗可逐位复现。")
            lines.append("")
            lines.append("前若干条差异：")
            for item in self.mismatched[:5]:
                lines.append(
                    f"  · cid={item['cid']}：库中 kept={item['stored_kept']}"
                    f"（{item['stored_reasons'] or '无原因'}），"
                    f"重放 kept={item['replay_kept']}（{item['replay_reasons'] or '无原因'}）"
                )
            if len(self.mismatched) > 5:
                lines.append(f"  … 另有 {len(self.mismatched) - 5} 条")
        lines.append("")
        lines.append(
            "说明：清洗步骤是固定规则表 + 固定执行顺序，不含随机性；"
            "若出现不一致，通常是管线版本已升级或去重范围不同，二者都会写进附录。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "dedupe_scope": self.dedupe_scope,
            "scope_assumed": self.scope_assumed,
            "filter_bots": self.filter_bots,
            "filter_bots_inferred": self.filter_bots_inferred,
            "checked": self.checked,
            "matched": self.matched,
            "mismatched": len(self.mismatched),
            "consistent": self.consistent,
            "replayable": self.replayable,
            "reason": self.reason,
        }


def verify_clean_replay(
    store: Any,
    run_id: str,
    *,
    tier: str | None = None,
    dedupe_scope: str | None = None,
) -> ReplayResult:
    """用当前代码对同一份原始数据重跑清洗，与库中记录逐条比对。

    这是把"清洗可逐位复现"从口号变成可验证事实的唯一方式。
    不写入任何数据——只读。
    """
    import json

    from .clean.basic import Cleaner
    from .pipeline import CleanOptions

    stored_rows = store.load_analysis_rows(run_id, only_kept=False)
    if not stored_rows:
        return ReplayResult(run_id=run_id, replayable=False,
                            reason="该会话没有清洗记录（clean_comments 为空）。")

    params: dict[str, Any] = {}
    run = store.get_run(run_id) or {}
    try:
        params = json.loads(run.get("params_json") or "{}")
    except json.JSONDecodeError:
        params = {}

    resolved_tier = tier or stored_rows[0].get("clean_tier") or "moderate"
    resolved_scope = dedupe_scope or params.get("dedupe_scope")
    scope_assumed = resolved_scope is None
    resolved_scope = resolved_scope or "global"

    stored_reasons: set[str] = set()
    for row in stored_rows:
        stored_reasons.update(_split_reasons(row.get("drop_reasons")))
    if any(reason.startswith("ai_") for reason in stored_reasons):
        return ReplayResult(
            run_id=run_id, tier=resolved_tier, dedupe_scope=resolved_scope,
            replayable=False,
            reason=(
                "该次清洗包含 AI 判定步骤（" +
                "、".join(sorted(r for r in stored_reasons if r.startswith("ai_"))) +
                "），重放需要重新调用模型，不能用规则重放。"
            ),
        )

    options = CleanOptions(tier=resolved_tier, dedupe_scope=resolved_scope)
    # 僵尸号过滤是"可选步骤"，没记进参数时无法从档位反推。
    # 但库里的 drop_reasons 里有 drop_suspected_bots，就证明当时确实开了——
    # 从自己的执行痕迹反推配置，是依据，不是猜测。
    if "filter_bots" in params:
        filter_bots = bool(params.get("filter_bots"))
        inferred = False
    else:
        filter_bots = any(r.startswith("drop_suspected_bots") for r in stored_reasons)
        inferred = filter_bots
    options.filter_bots = filter_bots

    cleaner = Cleaner(
        tier=resolved_tier,
        dedupe_scope="global" if resolved_scope == "global" else "post",
        enabled_steps=options.resolve_steps(),
    )
    raw_rows = store.load_comments(run_id)
    replay = cleaner.run(raw_rows)
    replayed = {str(row["cid"]): row for row in replay.rows}
    stored = {str(row["cid"]): row for row in stored_rows}

    mismatches: list[dict[str, Any]] = []
    for cid, stored_row in stored.items():
        replay_row = replayed.get(cid)
        if replay_row is None:
            continue
        replay_kept = bool(replay_row.get("kept"))
        if bool(stored_row.get("kept")) == replay_kept and (
            set(_split_reasons(stored_row.get("drop_reasons")))
            == set(_split_reasons(replay_row.get("drop_reasons")))
        ):
            continue
        mismatches.append({
            "cid": cid,
            "stored_kept": bool(stored_row.get("kept")),
            "replay_kept": replay_kept,
            "stored_reasons": "、".join(_split_reasons(stored_row.get("drop_reasons"))),
            "replay_reasons": "、".join(_split_reasons(replay_row.get("drop_reasons"))),
        })

    return ReplayResult(
        run_id=run_id,
        tier=resolved_tier,
        dedupe_scope=resolved_scope,
        scope_assumed=scope_assumed,
        filter_bots=filter_bots,
        filter_bots_inferred=inferred,
        checked=len(stored),
        mismatched=mismatches[:50],
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def build_appendix(facts: AppendixFacts) -> str:
    """把事实渲染成一份可直接放进论文附录的 Markdown。"""
    run = facts.run or {}
    env = facts.env or {}
    tag = run.get("tag") or "（未记录话题）"
    lines: list[str] = []
    lines.append(f"# 可复现性附录 · {tag}")
    lines.append("")
    lines.append(
        "> 本附录由博研助手自动生成，用于论文或课程报告的"
        "「研究方法 / 可复现性」部分。所有参数均为**实际运行值**，"
        "缺项写「未记录」而不是留空或补默认值。"
    )
    lines.append("")
    lines.append(f"- 采集会话：`{run.get('run_id') or '未记录'}`")
    lines.append(f"- 话题：{tag}")
    lines.append(f"- 采集时间：{run.get('created_at') or '未记录'}（北京时间）")
    lines.append(f"- 生成时间：{facts.generated_at or _now()}")
    lines.append(f"- 工具版本：`{env.get('tool_version') or __version__}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---------- 一、运行环境 ----------
    lines.append("## 一、运行环境")
    lines.append("")
    lines.append(f"- Python：{env.get('python') or '未记录'}")
    lines.append(f"- 操作系统：{env.get('platform') or '未记录'}")
    lines.append(f"- 架构：{env.get('machine') or '未记录'}")
    lines.append("")
    lines.append("| 依赖 | 版本 |")
    lines.append("| --- | --- |")
    for name, value in (env.get("deps") or {}).items():
        lines.append(f"| {name} | {value} |")
    lines.append("")
    if env.get("cwd_note"):
        lines.append(f"注：{env['cwd_note']}。")
        lines.append("")
    if facts.paths:
        lines.append("### 本机路径（已开启路径记录）")
        lines.append("")
        lines.append("| 项目 | 路径 |")
        lines.append("| --- | --- |")
        for name, value in facts.paths.items():
            lines.append(f"| {name} | `{value}` |")
        lines.append("")
        lines.append(
            "⚠ 这些是你的本机目录结构。对外提交附录前请确认是否可以公开；"
            "不需要时用 `--no-paths`（默认）重新生成即可。"
        )
        lines.append("")

    # ---------- 二、采集 ----------
    lines.append("## 二、采集")
    lines.append("")
    mode = facts.params.get("mode") or "live"
    lines.append(f"- 数据来源：微博（m.weibo.cn 公开接口）＋ 研究者本人登录态")
    lines.append(f"- 采集模式：`{mode}`" + ("（内置样例数据，**非真实数据**）"
                                          if mode == "mock" else ""))
    lines.append(f"- 帖子排序：{run.get('order_mode') or '未记录'}")
    from .analyze.collection import search_mode_text, window_text

    lines.append(f"- 检索口径：{search_mode_text(facts.params.get('search_mode'))}")
    lines.append(
        f"- 评论时间窗：{facts.params.get('comment_window_text') or window_text(None)}"
    )
    if facts.params.get("covered_from") and facts.params.get("covered_to"):
        lines.append(
            f"- 实际覆盖：{facts.params['covered_from']} ～ {facts.params['covered_to']}"
            "（由已采数据反算，不是计划值）"
        )
    if facts.params.get("skipped_posts_text"):
        lines.append(f"- 进样率：{facts.params['skipped_posts_text']}")
    if run.get("status"):
        lines.append(f"- 运行状态：{run['status']}")
    lines.append("")
    if facts.params:
        # 这三个是上面已经用小标题说过的派生值（由数据反算），
        # 再在参数表里重复一遍只会让人以为它们是采集时填的参数。
        derived = {
            "search_mode", "comment_window", "comment_window_label",
            "comment_window_text", "covered_from", "covered_to",
            "skipped_posts_text",
        }
        lines.append("| 参数 | 取值 |")
        lines.append("| --- | --- |")
        for key, value in facts.params.items():
            if key in derived:
                continue
            lines.append(f"| {key} | {_value(value)} |")
        lines.append("")
    sort_mode = str(facts.params.get("comment_sort") or "hot")
    if sort_mode == "time":
        bias_note = (
            "评论按时间倒序取自网页端接口；时间窗内越接近终点的评论越容易被采到，"
            "窗口早期评论可能采不足。"
        )
    else:
        bias_note = (
            "评论取自「热门评论流」，平台按热度而非时间返回，"
            "因此样本偏向高互动评论，不代表该帖全部评论。"
        )
    lines.append(
        "采集约束：请求之间插入随机延迟并设硬性请求上限；" + bias_note
    )
    lines.append("")

    # ---------- 三、清洗 ----------
    lines.append("## 三、清洗")
    lines.append("")
    cleaning = facts.cleaning
    if cleaning:
        rate = cleaning.get("retention_rate")
        lines.append(f"- 清洗档位：{cleaning.get('tier') or '未记录'}")
        lines.append(f"- 管线版本：`{cleaning.get('pipeline_version') or '未记录'}`")
        if cleaning.get("dedupe_scope"):
            lines.append(f"- 去重范围：{cleaning['dedupe_scope']}")
        if cleaning.get("total_input") is not None:
            rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "未记录"
            lines.append(
                f"- 样本：输入 {cleaning.get('total_input')} 条，"
                f"保留 {cleaning.get('total_kept')} 条（{rate_text}）"
            )
        steps = cleaning.get("steps") or []
        if steps:
            lines.append("")
            lines.append("| 步骤 | 移除 | 占输入 |")
            lines.append("| --- | ---: | ---: |")
            for step in steps:
                share = step.get("share")
                share_text = f"{share:.1%}" if isinstance(share, (int, float)) else "—"
                lines.append(
                    f"| {step.get('label', '')} | {step.get('removed', 0)} | {share_text} |"
                )
        lines.append("")
    else:
        lines.append("- 本次会话没有清洗记录（`clean_comments` 为空）。")
        lines.append("")

    # ---------- 四、分析 ----------
    lines.append("## 四、分析模块")
    lines.append("")
    if not facts.summaries:
        lines.append("- 本次会话还没有任何分析结果。")
        lines.append("")
    for module in ("sentiment", "cluster_discourse", "cluster_community",
                   "timeseries", "replies"):
        summary = facts.summaries.get(module)
        if not summary:
            continue
        lines.append(f"### {_MODULE_ZH.get(module, module)}")
        lines.append("")
        lines.append(f"- 分析 ID：`{summary.get('analysis_id') or '未记录'}`")
        lines.append(f"- 运行时间：{summary.get('created_at') or '未记录'}")
        method = summary.get("method") or {}
        flat = {key: value for key, value in method.items()
                if not isinstance(value, (dict, list))}
        nested = {key: value for key, value in method.items()
                  if isinstance(value, dict) and value}
        for key, value in flat.items():
            if key == "module":
                continue
            lines.append(f"- {key}：{_value(value)}")
        lines.append("")
        if nested:
            lines.append("| 子项 | 取值 |")
            lines.append("| --- | --- |")
            for key, payload in nested.items():
                for sub_key, sub_value in payload.items():
                    lines.append(f"| {key}.{sub_key} | {_value(sub_value)} |")
            lines.append("")

    # ---------- 五、LLM 用量 ----------
    if facts.llm:
        lines.append("## 五、大模型调用与用量")
        lines.append("")
        lines.append(f"- 服务地址：`{facts.llm.get('base_url') or '未记录'}`")
        lines.append(f"- 模型：`{facts.llm.get('model') or '未记录'}`")
        if facts.llm.get("calls") is not None:
            lines.append(f"- 调用次数：{facts.llm['calls']}")
        if facts.llm.get("prompt_tokens") is not None:
            lines.append(
                f"- token 用量：输入 {facts.llm.get('prompt_tokens')}，"
                f"输出 {facts.llm.get('completion_tokens')}"
            )
        lines.append("")
        lines.append(
            "⚠ 未记录采样温度等生成参数的部分，说明该次调用未显式设置，"
            "服务端默认值不由本工具控制。"
        )
        lines.append("")

    # ---------- 六、可复现性分级 ----------
    lines.append("## 六、可复现性分级")
    lines.append("")
    lines.append("> 逐环节说明：别人拿着这份附录，能复现到什么程度。")
    lines.append("")
    lines.append("| 环节 | 复现等级 | 依据 |")
    lines.append("| --- | --- | --- |")
    for item in reproducibility_items(facts):
        lines.append(item.render())
    lines.append("")
    lines.append(
        "**怎么读这张表**："
        f"「{EXACT}」= 相同输入在同一环境下会得到相同输出；"
        f"「{APPROX}」= 只能近似重跑；"
        f"「{NONE}」= 原理上不可能逐位复现，只能说明方法与参数。"
    )
    lines.append("")
    if facts.replay is not None:
        lines.append("### 本次清洗的重放校验")
        lines.append("")
        replay = facts.replay
        if not replay.get("replayable"):
            lines.append(f"- 未能重放：{replay.get('reason') or '原因未记录'}")
        elif replay.get("consistent"):
            lines.append(
                f"- 用当前代码对同一份原始数据重跑清洗，逐条比对 "
                f"{replay.get('checked')} 条，**全部一致**。"
                "也就是说上表「清洗：可逐位复现」这一行是**验证过的**，不是声明。"
            )
        else:
            lines.append(
                f"- ⚠ 重放 **不一致**：比对 {replay.get('checked')} 条，"
                f"不一致 {replay.get('mismatched')} 条。"
                "这种情况下不要在论文中声称清洗可逐位复现；"
                "常见原因是管线版本升级或去重范围不同。"
            )
        lines.append("")

    # ---------- 七、模型指纹 ----------
    if facts.models:
        lines.append("## 七、模型文件指纹")
        lines.append("")
        for item in facts.models:
            lines.append(f"### {item.get('label')}（`{item.get('dir')}`）")
            lines.append("")
            lines.append(f"- 状态：{item.get('state')}")
            files = item.get("files") or []
            if files:
                lines.append("")
                lines.append("| 文件 | 大小(MB) | SHA-256 |")
                lines.append("| --- | ---: | --- |")
                for row in files:
                    digest = row.get("sha256") or "（非权重文件，未计算）"
                    short = digest if digest.startswith("（") else digest[:16] + "…"
                    lines.append(f"| {row['name']} | {row['size_mb']} | `{short}` |")
            lines.append("")

    # ---------- 八、产出文件 ----------
    lines.append("## 八、产出文件与校验和")
    lines.append("")
    if facts.files:
        lines.append("| 文件 | 大小(KB) | SHA-256（前 16 位） |")
        lines.append("| --- | ---: | --- |")
        for item in facts.files:
            lines.append(
                f"| `{item['name']}` | {item.get('size_kb', '—')} | "
                f"`{str(item.get('sha256') or '')[:16]}` |"
            )
        lines.append("")
        lines.append(
            "校验和用于证明「论文里引用的数据就是这一份」。"
            "完整 SHA-256 见同目录 `manifest.json`。"
        )
    else:
        lines.append("- 没有找到导出文件。请先执行导出。")
    lines.append("")

    # ---------- 九、复核步骤 ----------
    lines.append("## 九、复核步骤（交给第三方）")
    lines.append("")
    for index, step in enumerate(_checklist(run), start=1):
        lines.append(f"{index}. {step}")
    lines.append("")

    # ---------- 十、已知局限 ----------
    lines.append("## 十、已知局限（务必在论文中披露）")
    lines.append("")
    for item in _limitations(facts):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("---")
    lines.append(
        f"本附录由博研助手 v{env.get('tool_version') or __version__} 于 "
        f"{facts.generated_at or _now()} 自动生成。"
        "参数为实际运行值；研究设计、抽样理由与伦理审批须由研究者自行撰写。"
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def reproducibility_items(facts: AppendixFacts) -> list[ReproducibilityItem]:
    """逐环节判定复现等级。这是本模块的核心判断。"""
    params = facts.params or {}
    items: list[ReproducibilityItem] = [
        ReproducibilityItem(
            stage="数据采集",
            level=APPROX,
            basis="平台内容随时间变化；同样参数重采得到的是**不同**样本",
            requirement="记录参数、采集时间与话题，才谈得上「近似重采」",
        ),
        ReproducibilityItem(
            stage="清洗",
            level=EXACT,
            basis=(
                f"固定规则表 + 固定执行顺序 + 固定随机种子（抽查抽样）；"
                f"管线版本 `{(facts.cleaning or {}).get('pipeline_version') or '未记录'}`"
            ),
            requirement="需要：同一原始数据 + 同一档位 + 同一管线版本",
        ),
    ]
    if "sentiment" in facts.summaries:
        items.append(ReproducibilityItem(
            stage="情感标注（LLM 通道）",
            level=NONE,
            basis=(
                "服务端模型版本与采样随机性不受本地控制；"
                "即使参数相同，服务端也可能已更新"
            ),
            requirement="只能复现方法与提示词版本，不能逐位复现标签；"
                        "需保留本次导出结果作为原始依据",
        ))
        items.append(ReproducibilityItem(
            stage="情感标注（本地模型通道）",
            level=EXACT,
            basis="ONNX 前向推理是确定性的；前提是模型文件字节一致（见第七节指纹）",
            requirement="需要：同一 ONNX 权重 + 同一分词器",
        ))
    if "cluster_discourse" in facts.summaries:
        items.append(ReproducibilityItem(
            stage="话语聚类",
            level=EXACT,
            basis="K-Means 固定 random_state；但**依赖句向量模型文件字节一致**",
            requirement="需要：同一 ONNX 权重（第七节有 SHA-256）+ 同一 K + 同一种子",
        ))
    if "cluster_community" in facts.summaries:
        items.append(ReproducibilityItem(
            stage="用户社群聚类",
            level=EXACT,
            basis="特征为确定性计算（计数/均值/log1p/标准化），聚类固定随机种子",
            requirement="需要：同一清洗结果 + 同一 K + 同一种子",
        ))
    if "timeseries" in facts.summaries:
        items.append(ReproducibilityItem(
            stage="时间演变",
            level=EXACT,
            basis="按时间戳分桶计数，无随机性",
            requirement="需要：同一原始数据（时间戳精度会影响分桶，已记录）",
        ))
    if "replies" in facts.summaries:
        items.append(ReproducibilityItem(
            stage="回复结构",
            level=EXACT,
            basis="按 parent_cid 重建树，无随机性",
            requirement="需要：同一原始数据（含二级评论）",
        ))
    items.append(ReproducibilityItem(
        stage="交叉表与小群体抑制",
        level=EXACT,
        basis="计数规则 + 固定阈值，无随机性",
        requirement=f"需要：同一分析结果 + 同一最小可展示样本（"
                    f"{params.get('min_n', 5)}）",
    ))
    items.append(ReproducibilityItem(
        stage="报告与图表",
        level=EXACT,
        basis="由上述结果排版生成，不引入新计算",
        requirement="需要：同一次分析的全部结果",
    ))
    return items


def _checklist(run: dict[str, Any]) -> list[str]:
    run_id = run.get("run_id") or "<会话ID>"
    return [
        "确认导出目录中的 `manifest.json` 与本附录的校验和一致",
        "用 `analysis_comments.csv` 对照论文表格，抽查 20 条原文与标签是否对应",
        "若要重跑清洗并核对：`python -m wrc.cli reclean --run " + run_id + "`"
        "（只校验、不写入）",
        f"重建报告（不重跑模型）：`python tools/run_report.py --run {run_id}`",
        "若声称「趋势变化」，请确认报告中的样本量提示，并补做统计检验",
    ]


def _limitations(facts: AppendixFacts) -> list[str]:
    items = [
        "评论来自平台「热门评论流」接口，偏向高互动评论，"
        "不代表该帖全部评论的分布；本附录不提供任何代表性修正。",
        "检索结果的排序与平台站内排序不完全一致；本地重排不等于平台排序。",
        "部分时间字段仅精确到「日」，已在数据中以 created_at_precision 标注；"
        "按小时聚合会被强制降级。",
        "所有自动标注（情感、聚类）均未经人工双编码校验，"
        "信度抽样只反映通道间一致性，不等于编码员间信度。",
        "本附录记录的是**方法与参数**，不是研究设计；"
        "抽样理由、伦理审批与研究问题须由研究者自行撰写。",
    ]
    if (facts.params or {}).get("mode") == "mock":
        items.insert(0, "⚠ 本次会话使用内置样例数据（mock），**不得**用于论文结论。")
    if "sentiment" in facts.summaries:
        items.append(
            "LLM 标注不可逐位复现；若论文需要严格复现，请以本次导出的标签为原始依据。"
        )
    return items


_MODULE_ZH = {
    "sentiment": "情感分析",
    "cluster_discourse": "话语聚类",
    "cluster_community": "用户社群聚类",
    "timeseries": "时间演变",
    "replies": "二级评论回复结构",
}


def _value(value: Any) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "、".join(_value(item) for item in value) if value else "无"
    if isinstance(value, dict):
        # 时间窗这类嵌套参数不能打成 Python repr：附录是给人读的
        return (
            "；".join(f"{key}={_value(item)}" for key, item in value.items())
            if value
            else "无"
        )
    return str(value)


def _now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def write_appendix(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
