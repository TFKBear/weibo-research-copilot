"""把一次采集的参数与结果整理成「采集口径」——报告里必须交代的那几行。

为什么值得单独成节：

研究者在论文方法部分必须回答四个问题：
1. 这个样本是怎么来的（检索词、排序、候选池、选中规则）？
2. 每一篇帖子采了多少（逐帖目标，而不是一个笼统的"N 条"）？
3. 有没有时间窗？窗口覆盖到了哪里？没覆盖到的部分怎么交代？
4. 实际拿了多少次请求、用的哪个后端、排序参数到底有没有生效？

这些信息分散在三处：采集参数（``collection_runs.params_json``）、
实际数据（``raw_comments.created_at``）、研究日志（``comments_done`` 事件）。
把它们合成一份口径，报告才有资格说"可追溯"。

一个刻意的取舍：**覆盖度用实际数据反算**，而不是照抄采集时的计划值。
计划里写"目标窗口 9-01～9-30"，实际可能只采到 9-20；
照抄计划就是给自己发一张假的通行证。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

#: 全项目统一按北京时间；跨时区数据在这里归一，避免 min/max 抛类型错误。
_CST = timezone(timedelta(hours=8), "CST")

SORT_ZH = {"hot": "热度优先（平台热门流）", "time": "按时间倒序"}

#: 检索口径的唯一说法。四处（报告 / 附录 / 方法卡片 / method_note）都必须用这里的字，
#: 否则会出现"报告说主动参与者、附录只说话题聚合"这类看似一致实则不同的表述。
SEARCH_MODE_ZH = {
    "topic": "话题聚合（q=#话题#），样本是主动参与该话题的帖子",
    "keyword": "关键词全文检索（q=关键词），样本含仅提及该词的帖子，不等于话题参与者",
}

#: 没设时间窗时的唯一说法。
WINDOW_NONE_TEXT = "不限时间（采该帖可见的最新评论）"


def search_mode_text(mode: str | None) -> str:
    return SEARCH_MODE_ZH.get(str(mode or "topic"), "未记录")


def window_text(window: dict[str, Any] | None) -> str:
    """时间窗的一句话说法（同一个窗口在任何地方都应显示成同一句）。"""
    window = window or {}
    if window.get("active"):
        return str(window.get("label") or "已设置（起止未记录）")
    return WINDOW_NONE_TEXT


def _parse_dt(value: Any) -> datetime | None:
    """解析时间并统一成北京时间朴素 datetime。

    库里既有带时区的 ISO（采集时写入 CST），也有朴素字符串（早期数据），
    混在一起取 min/max 会抛 "can't compare offset-naive and offset-aware"。
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_CST).replace(tzinfo=None)
    return parsed


def summarise_collection(
    *,
    params: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    log_events: Iterable[dict[str, Any]] = (),
    requests_used: int | None = None,
) -> dict[str, Any]:
    """生成采集口径摘要（纯函数，便于测试与复用）。

    Args:
        params: 该会话的采集参数（``params_json`` 解析后的 dict）。
        rows: 参与分析的评论行（需含 ``created_at``）。
        log_events: 研究日志事件；用来确认排序参数是否真的生效。
        requests_used: 实际请求数（日志里有就以日志为准）。
    """
    params = params or {}
    window = params.get("comment_window") or {}
    events = [
        event for event in log_events if event.get("event") in {"comments_done", "run_start"}
    ]
    done_events = [event for event in events if event.get("event") == "comments_done"]

    # 排序是否生效：日志里每篇帖子都记了 sort_applied；只要有一篇是 False，
    # 就说明"按时间排序"在这批数据里没有真正生效。
    applied_flags = [
        bool((event.get("payload") or event).get("sort_applied")) for event in done_events
    ]
    sort_applied = None if not applied_flags else all(applied_flags)

    # 覆盖度：用实际数据反算
    times = [t for t in (_parse_dt(row.get("created_at")) for row in rows) if t]
    covered_from = min(times) if times else None
    covered_to = max(times) if times else None

    backends = sorted({
        str((event.get("payload") or event).get("backend") or "")
        for event in done_events
    } - {""})

    request_count = requests_used
    if request_count is None:
        for event in log_events:
            if event.get("event") != "collect_done":
                continue
            payload = event.get("payload") or event
            if payload.get("requests") is not None:
                request_count = int(payload["requests"])

    per_post = params.get("per_post_targets") or {}
    targets = sorted({int(value) for value in per_post.values()}) if per_post else []

    return {
        "tag": params.get("tag") or "",
        "search_mode": params.get("search_mode") or "topic",
        "order": params.get("order_mode") or params.get("order") or "",
        "post_limit": params.get("post_limit"),
        "selected_posts": len({row.get("post_mid") for row in rows if row.get("post_mid")}),
        "comments_per_post": params.get("comments_per_post"),
        "per_post_targets": per_post,
        "target_spread": targets,
        "window": window,
        "window_active": bool(window.get("active")),
        "sort": params.get("comment_sort") or "hot",
        "sort_applied": sort_applied,
        "backend": params.get("comment_backend") or "auto",
        "backends_used": backends,
        "include_subcomments": bool(params.get("include_subcomments")),
        "max_requests": params.get("max_requests"),
        "requests_used": request_count,
        "kept": len(rows),
        "covered_from": covered_from.isoformat() if covered_from else None,
        "covered_to": covered_to.isoformat() if covered_to else None,
    }


def render_collection(summary: dict[str, Any]) -> str:
    """渲染成报告里的纯文本块（用等宽字体展示，方便直接抄进论文）。"""
    lines: list[str] = []
    lines.append("采集口径")
    lines.append("=" * 62)
    lines.append(f"检索话题：{summary.get('tag') or '—'}")
    lines.append(f"检索口径：{search_mode_text(summary.get('search_mode'))}")
    lines.append(
        f"帖子排序：{summary.get('order') or '—'}"
        "（engagement/time 为本地重排，platform 为接口顺序）"
    )
    if summary.get("post_limit"):
        lines.append(
            f"候选帖子：前 {summary['post_limit']} 篇；入样 {summary.get('selected_posts')} 篇"
            "（数据中实际出现的帖子；其余可能未选中，或被平台以「暂无数据」拒绝——"
            "真实运行里 20 篇候选中曾有 9 篇被拒）"
        )

    targets = summary.get("target_spread") or []
    if len(targets) > 1:
        lines.append(
            f"评论目标：全局 {summary.get('comments_per_post')} 条，"
            f"另有逐帖覆盖（共 {len(summary.get('per_post_targets') or {})} 篇，"
            f"取值 {', '.join(str(t) for t in targets)}）"
        )
    else:
        lines.append(f"评论目标：每帖 {summary.get('comments_per_post')} 条")

    window = summary.get("window") or {}
    lines.append(f"评论时间窗：{window_text(window)}")

    sort_text = SORT_ZH.get(summary.get("sort") or "hot", str(summary.get("sort")))
    if summary.get("sort_applied") is True:
        sort_text += "；已生效"
    elif summary.get("sort_applied") is False:
        if (summary.get("sort") or "hot") == "time":
            sort_text += "；⚠ 实际未生效（后端忽略排序参数），样本为平台默认流"
        else:
            sort_text += "；后端无排序参数，但平台默认流本身就是按热度返回"
    lines.append(f"评论排序：{sort_text}")

    backends = summary.get("backends_used") or []
    lines.append(
        f"评论后端：{summary.get('backend')}"
        + (f"（实际使用：{', '.join(backends)}）" if backends else "")
    )
    lines.append(
        "二级评论：" + ("已采集" if summary.get("include_subcomments") else "未采集（默认关闭）")
    )
    if summary.get("max_requests"):
        lines.append(
            f"请求：上限 {summary['max_requests']} 次"
            + (f"，实际 {summary['requests_used']} 次" if summary.get("requests_used") else "")
        )
    lines.append(f"参与分析：{summary.get('kept')} 条")
    if summary.get("covered_from") and summary.get("covered_to"):
        lines.append(f"实际覆盖：{summary['covered_from']} ～ {summary['covered_to']}")
    lines.append("")
    lines.append(
        "说明：时间窗与排序决定样本代表性。「实际覆盖」由已采数据反算，"
        "若短于设定的时间窗，说明窗口没有被采全，论文中应如实交代。"
    )
    return "\n".join(lines)


def load_params(run_row: dict[str, Any]) -> dict[str, Any]:
    """从 ``collection_runs`` 行里取出采集参数，并补上表列里的 topic/排序。

    话题与排序在 ``collection_runs`` 是独立列而不是参数 JSON 的一部分；
    不合并的话报告里会出现「检索话题：—」这种一眼就露馅的空白。
    """
    raw = run_row.get("params_json")
    params: dict[str, Any] = {}
    if raw:
        try:
            params = json.loads(raw) or {}
        except (TypeError, ValueError):
            params = {}
    if not params.get("tag") and run_row.get("tag"):
        params["tag"] = run_row["tag"]
    if not params.get("order_mode") and run_row.get("order_mode"):
        params["order_mode"] = run_row["order_mode"]
    return params
