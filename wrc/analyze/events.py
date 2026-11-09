"""事件节点对齐（event study）。

研究问题通常是"某件事发生之后，态度有没有变"。要回答它，需要三样东西：

1. **事件日期**（由研究者指定，工具不替他猜）；
2. **前后窗口**（默认各 7 天，可调）；
3. **可比的指标**（立场占比、情感占比）。

这个模块刻意做得很窄：它只算"前后两段的占比差 + 一个两比例 z 检验"，
不建模型、不做回归、不画趋势线。原因见下面第三条。

## 三条不可妥协的规则

**一、事件日期必须由研究者给。**
工具可以从数据里找出"评论量最大的那天"，但那是**数据特征**，不是**事件**。
把峰值日当事件日再宣称"事件效应"，是典型的事后归因。
所以 CLI 必须显式传 `--event`；没传就不做这一节。

**二、窗口两侧样本不足时，只报数不给结论。**
每侧至少 ``MIN_WINDOW_N`` 条才计算占比差；少于这个数只写"样本不足"。

**三、统计检验要标明假设，且不替你决定口径。**
这里给的是**两比例 z 检验**（正态近似），只在"每格期望频数 ≥5"时计算；
不满足就明说"不适合做该检验"。p 值只是参考——**用不用它、显著性水平取多少，
是研究设计与期刊口径问题，得研究者自己定**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from .timeseries import (
    SENTIMENT_ORDER, SENTIMENT_ZH, STANCE_ORDER, STANCE_ZH, _parse_time,
)

#: 每侧最少多少条才计算占比差。
MIN_WINDOW_N = 30
#: 两比例 z 检验的最低期望频数（正态近似的常规要求）。
MIN_EXPECTED_COUNT = 5
#: 默认窗口天数。
DEFAULT_PRE_DAYS = 7
DEFAULT_POST_DAYS = 7


@dataclass(slots=True)
class EventSpec:
    """一个事件：日期、可选标签、前后窗口天数。"""

    date: str
    label: str = ""
    pre_days: int = DEFAULT_PRE_DAYS
    post_days: int = DEFAULT_POST_DAYS

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "label": self.label,
            "pre_days": self.pre_days,
            "post_days": self.post_days,
        }

    def described(self) -> str:
        name = self.label or "事件"
        return f"{name}（{self.date}，前 {self.pre_days} 天 vs 后 {self.post_days} 天）"


@dataclass(slots=True)
class WindowStat:
    n: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "counts": dict(self.counts),
            "shares": {key: round(value, 4) for key, value in self.shares.items()},
        }


@dataclass(slots=True)
class EventResult:
    event: EventSpec
    metric: str
    pre: WindowStat = field(default_factory=WindowStat)
    post: WindowStat = field(default_factory=WindowStat)
    deltas: dict[str, float] = field(default_factory=dict)
    tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    decidable: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def metric_zh(self) -> str:
        return {"stance": "立场", "sentiment": "情感"}.get(self.metric, self.metric)

    @property
    def labels(self) -> tuple[str, ...]:
        return STANCE_ORDER if self.metric == "stance" else SENTIMENT_ORDER

    def render(self) -> str:
        zh = STANCE_ZH if self.metric == "stance" else SENTIMENT_ZH
        lines = [f"事件前后对比 · {self.event.described()}", "-" * 62]
        lines.append(
            f"{'':<8}{'前段 n':>8}{'后段 n':>8}"
            + "".join(f"{zh[label]:>9}" for label in self.labels if label in self.pre.counts
                      or label in self.post.counts)
        )
        lines.append(
            f"{'条数':<8}{self.pre.n:>8}{self.post.n:>8}"
        )
        for label in self.labels:
            if label not in self.pre.counts and label not in self.post.counts:
                continue
            pre_share = self.pre.shares.get(label)
            post_share = self.post.shares.get(label)
            delta = self.deltas.get(label)
            pre_text = f"{pre_share:.0%}" if pre_share is not None else "—"
            post_text = f"{post_share:.0%}" if post_share is not None else "—"
            delta_text = f"{delta:+.0%}" if delta is not None else "—"
            test = self.tests.get(label) or {}
            p_text = f"p={test['p_value']:.3f}" if test.get("p_value") is not None else ""
            lines.append(
                f"{zh[label]:<8}{pre_text:>8}{post_text:>8}{delta_text:>9}  {p_text}"
            )
        lines.append("")
        if not self.decidable:
            lines.append(
                f"⚠ 某一侧样本量不足 {MIN_WINDOW_N} 条，**只报数不给结论**；"
                "请扩大窗口或扩大采集。"
            )
        else:
            movers = [
                (label, delta) for label, delta in self.deltas.items()
                if abs(delta) >= 0.10
            ]
            if movers:
                movers.sort(key=lambda item: -abs(item[1]))
                text = "、".join(f"{zh.get(label, label)} {delta:+.0%}"
                                 for label, delta in movers)
                lines.append(f"前后变化（{self.metric_zh}）：{text}")
            else:
                lines.append(
                    f"前后变化（{self.metric_zh}）：各类别占比变动均 <10%，未观察到明显变化。"
                )
        for warning in self.warnings:
            lines.append(f"⚠ {warning}")
        lines.append("")
        lines.append(
            "说明：这是**描述性对比 + 一个两比例 z 检验**，不是因果推断。"
            "窗口边界的选取、事件日期的准确性、以及周边同时发生的其它事件，"
            "都会影响结果；论文中应说明窗口为何这样取。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "event",
            "event": self.event.to_dict(),
            "metric": self.metric,
            "pre": self.pre.to_dict(),
            "post": self.post.to_dict(),
            "deltas": {key: round(value, 4) for key, value in self.deltas.items()},
            "tests": self.tests,
            "decidable": self.decidable,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
def analyse_event(
    rows: Sequence[dict[str, Any]],
    *,
    event: EventSpec,
    metric: str = "stance",
) -> EventResult:
    """比较事件前后两个窗口内某个指标的分布。"""
    if metric not in ("stance", "sentiment"):
        raise ValueError(f"不支持的指标：{metric}")
    labels = STANCE_ORDER if metric == "stance" else SENTIMENT_ORDER

    try:
        event_day = datetime.strptime(event.date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"事件日期格式应为 YYYY-MM-DD：{event.date}") from exc

    pre_start = event_day - timedelta(days=event.pre_days)
    post_end = event_day + timedelta(days=event.post_days)

    pre_counts: dict[str, int] = {}
    post_counts: dict[str, int] = {}
    pre_n = post_n = 0
    for row in rows:
        stamp, _precision = _parse_time(row.get("created_at"))
        if stamp is None:
            continue
        day = stamp.date()
        label = str(row.get(metric) or "")
        if label not in labels:
            continue
        if pre_start <= day < event_day:
            pre_n += 1
            pre_counts[label] = pre_counts.get(label, 0) + 1
        elif event_day <= day <= post_end:
            post_n += 1
            post_counts[label] = post_counts.get(label, 0) + 1

    result = EventResult(
        event=event, metric=metric,
        pre=WindowStat(n=pre_n, counts=pre_counts),
        post=WindowStat(n=post_n, counts=post_counts),
    )
    result.decidable = pre_n >= MIN_WINDOW_N and post_n >= MIN_WINDOW_N

    if pre_n == 0 or post_n == 0:
        result.warnings.append(
            f"一侧完全没有数据（前 {pre_n} 条 / 后 {post_n} 条）："
            "该事件附近可能没有采集到评论，请先确认时间范围。"
        )
        return result

    if result.decidable:
        for label in labels:
            if label not in pre_counts and label not in post_counts:
                continue
            pre_share = pre_counts.get(label, 0) / pre_n
            post_share = post_counts.get(label, 0) / post_n
            result.pre.shares[label] = pre_share
            result.post.shares[label] = post_share
            result.deltas[label] = post_share - pre_share

            test = two_proportion_test(
                pre_counts.get(label, 0), pre_n,
                post_counts.get(label, 0), post_n,
            )
            if test is not None:
                result.tests[label] = test
                if not test["assumptions_ok"]:
                    result.warnings.append(
                        f"「{STANCE_ZH.get(label, SENTIMENT_ZH.get(label, label))}」"
                        "的期望频数偏低，z 检验的正态近似不可靠，p 值仅供参考。"
                    )
    else:
        result.warnings.append(
            f"前段 {pre_n} 条、后段 {post_n} 条，任一侧不足 {MIN_WINDOW_N} 条。"
        )

    if post_n and pre_n:
        ratio = post_n / pre_n
        if ratio >= 3 or ratio <= 1 / 3:
            result.warnings.append(
                f"两段评论量相差很大（{pre_n} vs {post_n}）："
                "占比差异可能部分来自「事件引发讨论量激增」，而非态度本身变化。"
            )
    return result


@dataclass(slots=True)
class ChangeCandidate:
    """一个候选转折点：把某天当作分界时的前后差异。"""

    date: str
    metric: str
    pre_n: int = 0
    post_n: int = 0
    deltas: dict[str, float] = field(default_factory=dict)
    shares_before: dict[str, float] = field(default_factory=dict)
    shares_after: dict[str, float] = field(default_factory=dict)

    @property
    def magnitude(self) -> float:
        return max((abs(value) for value in self.deltas.values()), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "metric": self.metric,
            "pre_n": self.pre_n,
            "post_n": self.post_n,
            "deltas": {key: round(value, 4) for key, value in self.deltas.items()},
        }


def scan_changepoints(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str = "stance",
    pre_days: int = DEFAULT_PRE_DAYS,
    post_days: int = DEFAULT_POST_DAYS,
    min_side: int = MIN_WINDOW_N,
    top: int = 5,
) -> list[ChangeCandidate]:
    """扫描"话语转折点"：逐日当作分界，比较前后窗口的占比差异。

    ⚠ 这不是事件研究。它回答的是「讨论结构在哪天前后变化最大」，
    不回答「那天发生了什么」。所以结果只能作为**定位候选窗口**的线索，
    事件日期仍须由研究者依据外部信息确定。
    """
    if metric not in ("stance", "sentiment"):
        raise ValueError(f"不支持的指标：{metric}")
    labels = STANCE_ORDER if metric == "stance" else SENTIMENT_ORDER

    by_day: dict[date, list[str]] = {}
    for row in rows:
        stamp, _precision = _parse_time(row.get("created_at"))
        value = str(row.get(metric) or "")
        if stamp is None or value not in labels:
            continue
        by_day.setdefault(stamp.date(), []).append(value)

    if len(by_day) < 3:
        return []

    days = sorted(by_day)
    candidates: list[ChangeCandidate] = []
    for day in days:
        pre = [value for key in days
               if day - timedelta(days=pre_days) <= key < day
               for value in by_day[key]]
        post = [value for key in days
                if day <= key <= day + timedelta(days=post_days)
                for value in by_day[key]]
        if len(pre) < min_side or len(post) < min_side:
            continue
        before = {label: pre.count(label) / len(pre) for label in labels}
        after = {label: post.count(label) / len(post) for label in labels}
        candidates.append(ChangeCandidate(
            date=day.isoformat(),
            metric=metric,
            pre_n=len(pre),
            post_n=len(post),
            deltas={label: after[label] - before[label] for label in labels},
            shares_before=before,
            shares_after=after,
        ))

    candidates.sort(
        key=lambda item: -abs(item.deltas.get(_focus_label(metric), 0.0))
    )
    return candidates[:top]


def _focus_label(metric: str) -> str:
    """扫描时按哪个类别排序：立场看「反对」，情感看「负面」。"""
    return "oppose" if metric == "stance" else "negative"


def render_changepoints(
    candidates: Sequence[ChangeCandidate], *, metric: str = "stance"
) -> str:
    if not candidates:
        return ""
    zh = STANCE_ZH if metric == "stance" else SENTIMENT_ZH
    focus = _focus_label(metric)
    lines = [
        "话语转折点扫描（**不是**事件研究）",
        "-" * 62,
        "逐日作为分界、比较前后窗口的占比差异，按变动幅度排序。",
        "它只能说明「讨论结构在哪天前后变化最大」，不能说明「那天发生了什么」。",
        "",
        f"{'候选日':<12}{'前段n':>7}{'后段n':>7}{zh[focus] + '前→后':>14}{'变动':>8}",
    ]
    for item in candidates:
        before = item.shares_before.get(focus, 0.0)
        after = item.shares_after.get(focus, 0.0)
        lines.append(
            f"{item.date:<12}{item.pre_n:>7}{item.post_n:>7}"
            f"{before:>7.0%}→{after:<6.0%}{item.deltas.get(focus, 0.0):>+8.0%}"
        )
    lines.append("")
    lines.append(
        "用法：把它当「去看哪段时间」的线索，再去查那几天有没有可核实的公共事件。"
        "**不要**直接把排名第一的日期写成「事件日」——那是用结果定义原因。"
    )
    return "\n".join(lines)


def two_proportion_test(
    x1: int, n1: int, x2: int, n2: int
) -> dict[str, Any] | None:
    """两比例 z 检验（双侧）。正态近似假设不满足时也返回结果，但标记出来。"""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    if pooled <= 0 or pooled >= 1:
        return {
            "p1": p1, "p2": p2, "z": None, "p_value": None,
            "assumptions_ok": False,
        }
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if standard_error == 0:
        return None
    z = (p2 - p1) / standard_error
    p_value = math.erfc(abs(z) / math.sqrt(2))  # 双侧
    expected = (n1 * pooled, n1 * (1 - pooled), n2 * pooled, n2 * (1 - pooled))
    return {
        "p1": round(p1, 4),
        "p2": round(p2, 4),
        "z": round(z, 3),
        "p_value": round(p_value, 4),
        "assumptions_ok": all(value >= MIN_EXPECTED_COUNT for value in expected),
    }


def analyse_events(
    rows: Sequence[dict[str, Any]],
    *,
    events: Sequence[EventSpec],
    metrics: Sequence[str] = ("stance", "sentiment"),
) -> list[EventResult]:
    """对每个事件、每个指标各算一份。"""
    results: list[EventResult] = []
    for event in events:
        for metric in metrics:
            results.append(analyse_event(rows, event=event, metric=metric))
    return results


def render_all(results: Sequence[EventResult]) -> str:
    if not results:
        return ""
    blocks = ["事件前后对比（event study）", "=" * 62,
              "事件日期由研究者指定；窗口两端都取了数据才有可能看出变化。", ""]
    for result in results:
        blocks.append(result.render())
        blocks.append("")
    return "\n".join(blocks)


def parse_event_arg(value: str) -> EventSpec:
    """解析命令行 ``--event`` 参数。

    支持三种写法：
    - ``2026-09-29``                 → 默认标签、默认窗口
    - ``2026-09-29:政策公布``         → 带标签
    - ``2026-09-29:政策公布:14:14``   → 带标签与前后窗口天数
    """
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts:
        raise ValueError("事件参数为空")
    spec = EventSpec(date=parts[0])
    if len(parts) >= 2:
        spec.label = parts[1]
    if len(parts) >= 3:
        spec.pre_days = int(parts[2])
    if len(parts) >= 4:
        spec.post_days = int(parts[3])
    # 早校验格式，避免跑到一半才炸
    datetime.strptime(spec.date, "%Y-%m-%d")
    return spec


def events_from_params(params: dict[str, Any]) -> list[EventSpec]:
    """从采集参数里读回之前保存的事件（便于只算一次、反复重建报告）。"""
    raw = params.get("events") or []
    specs: list[EventSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            specs.append(EventSpec(
                date=str(item.get("date")),
                label=str(item.get("label") or ""),
                pre_days=int(item.get("pre_days") or DEFAULT_PRE_DAYS),
                post_days=int(item.get("post_days") or DEFAULT_POST_DAYS),
            ))
        except (TypeError, ValueError):
            continue
    return specs
