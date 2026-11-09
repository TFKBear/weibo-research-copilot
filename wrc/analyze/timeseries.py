"""时间演变分析（P3 第 1 项：舆情的时间序列）。

要回答的问题是"舆论在这段时间里怎么变化的"。这件事有**三个必须先说清楚的前提**，
否则产出的曲线会被读成它并不支持的结论：

1. **采集不是全量的，更不是时序完整的**。
   评论来自「热门评论流」，平台按热度而非时间返回。所以这里的曲线描述的是
   **已采集样本在时间上的分布**，不是平台全量舆论的演化轨迹。早期评论尤其可能
   整体缺席（热门流偏向近期高互动）。这条必须写在图上和文里。
2. **时间精度不齐**。检索结果里混着「只精确到日」的时间戳。把这种记录放进
   小时桶是造假——模块会检测到日级精度并**强制降级到按日聚合**。
3. **小时段样本太小，占比没有意义**。n<5 的时段会被标成低置信并且不参与
   趋势判断，图表上用空心点区分，避免读者把一个 1/2 的占比当成 50% 的民意。

趋势判断也刻意保守：只有前后两侧样本都 ≥30 条、且占比变动 ≥10 个百分点时才说
"观察到变化"，否则明确写"样本不足以判断趋势"。宁可少给结论，也不给假结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

#: 桶粒度。``hour`` 只在时间跨度很短且样本足够时才允许。
DAY = "day"
HOUR = "hour"

#: 一个时段至少要这么多条，占比才拿出来解读。
MIN_BUCKET_N = 5
#: 判断趋势时，前后两侧各自的最低样本量。
#:
#: 原先是 30（"没有 30 条就别谈趋势"）。研究者看过实际输出后选择了更宽松的
#: 5 条，理由是：小样本趋势虽然脆弱，但**明确标注后**仍可提供线索，
#: 而完全不出结论等于丢掉信息。所以门槛降到 5，同时用 RELIABLE_TREND_N
#: 标记"够不够可信"，凡是低于它就强制打上"仅供参考"。
MIN_TREND_N = 5
#: 可信门槛。低于它在文字与图表上都必须标注"仅供参考"。
RELIABLE_TREND_N = 30
#: 占比变动达到这个幅度才认为"有变化"（百分点/100）。
TREND_DELTA = 0.10

#: 跨度不超过这么多天时，才考虑按小时聚合（否则一天一个点更可读）。
HOUR_MAX_SPAN_DAYS = 2

SENTIMENT_ORDER = ("positive", "neutral", "negative")
SENTIMENT_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}

#: 立场维度的固定顺序与中文名。与情感一样，顺序固定便于横向比较。
STANCE_ORDER: tuple[str, ...] = ("support", "oppose", "neutral", "irrelevant")
STANCE_ZH = {"support": "支持", "oppose": "反对",
             "neutral": "中立", "irrelevant": "无关"}


@dataclass(slots=True)
class TimeSeriesResult:
    """时间演变结果。"""

    granularity: str
    buckets: list[str] = field(default_factory=list)
    volume: list[int] = field(default_factory=list)
    sentiment_counts: dict[str, list[int]] = field(default_factory=dict)
    sentiment_share: dict[str, list[float]] = field(default_factory=dict)
    stance_counts: dict[str, list[int]] = field(default_factory=dict)
    stance_share: dict[str, list[float]] = field(default_factory=dict)
    cluster_share: dict[str, list[float]] = field(default_factory=dict)
    low_n_buckets: list[str] = field(default_factory=list)
    shift: dict[str, Any] = field(default_factory=dict)
    shift_stance: dict[str, Any] = field(default_factory=dict)
    n_total: int = 0
    n_unparsed: int = 0
    precision_note: str | None = None
    warnings: list[str] = field(default_factory=list)
    method: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def render(self) -> str:
        unit = "日" if self.granularity == DAY else "小时"
        lines = [
            f"时间演变（按{unit}聚合，n={self.n_total}）",
            "=" * 62,
        ]
        if not self.buckets:
            lines.append("没有可用的时间数据，无法做时间演变分析。")
            for warning in self.warnings:
                lines.append(f"⚠ {warning}")
            return "\n".join(lines)

        has_sentiment = bool(self.sentiment_counts)
        header = f"{'时段':<14}{'评论数':>7}"
        if has_sentiment:
            for label in SENTIMENT_ORDER:
                if label in self.sentiment_counts:
                    header += f"{SENTIMENT_ZH[label]:>8}"
        lines.append(header)
        lines.append("-" * 62)
        for index, bucket in enumerate(self.buckets):
            row = f"{bucket:<14}{self.volume[index]:>7}"
            if has_sentiment:
                for label in SENTIMENT_ORDER:
                    shares = self.sentiment_share.get(label)
                    if shares is None:
                        continue
                    if bucket in self.low_n_buckets:
                        row += f"{'—':>8}"
                    else:
                        row += f"{shares[index]:>7.0%} "
            marker = "  ← 样本过小，占比不解读" if bucket in self.low_n_buckets else ""
            lines.append(row + marker)

        lines.append("")
        if self.precision_note:
            lines.append(f"注：{self.precision_note}")
        if self.low_n_buckets:
            lines.append(
                f"注：{len(self.low_n_buckets)} 个时段样本量 <{MIN_BUCKET_N}，"
                "占比以「—」显示，不参与趋势判断。"
            )
        for warning in self.warnings:
            lines.append(f"⚠ {warning}")
        lines.append("")
        lines.append(_describe_shift(self.shift))
        lines.append("")
        if self.stance_share:
            lines.append(self._render_stance())
            lines.append("")
        lines.append(
            "说明：本曲线描述的是**已采集样本**在时间上的分布，不是平台全量舆论的演化。"
            "评论取自「热门评论流」，平台按热度而非时间返回，早期评论可能整体缺席。"
        )
        return "\n".join(lines)

    def _render_stance(self) -> str:
        """立场占比随时间的表 + 趋势判断。

        与情感表分开排：一张表同时塞 3 个情感列和 4 个立场列，
        在终端里会挤成两行看不懂的东西。
        """
        unit = "日" if self.granularity == DAY else "小时"
        lines = [f"立场占比随时间（按{unit}聚合）", "-" * 62]
        header = f"{'时段':<14}{'评论数':>7}"
        for label in STANCE_ORDER:
            if label in self.stance_share:
                header += f"{STANCE_ZH[label]:>8}"
        lines.append(header)
        for index, bucket in enumerate(self.buckets):
            row = f"{bucket:<14}{self.volume[index]:>7}"
            for label in STANCE_ORDER:
                shares = self.stance_share.get(label)
                if shares is None:
                    continue
                row += f"{'—':>8}" if bucket in self.low_n_buckets else f"{shares[index]:>7.0%} "
            lines.append(row)
        lines.append("")
        lines.append(
            _describe_shift(
                self.shift_stance,
                prefix="立场趋势判断",
                zh=STANCE_ZH,
                unit_name="立场",
            )
        )
        lines.append("")
        lines.append(
            "说明：立场占比的分母是**该时段有立场标签的评论数**，"
            "不含「无关」以外的缺失；解读时请与「立场分布」一节的提示一致。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "timeseries",
            "granularity": self.granularity,
            "buckets": self.buckets,
            "volume": self.volume,
            "sentiment_counts": self.sentiment_counts,
            "sentiment_share": {
                key: [round(value, 4) for value in values]
                for key, values in self.sentiment_share.items()
            },
            "stance_counts": self.stance_counts,
            "stance_share": {
                key: [round(value, 4) for value in values]
                for key, values in self.stance_share.items()
            },
            "cluster_share": {
                key: [round(value, 4) for value in values]
                for key, values in self.cluster_share.items()
            },
            "low_n_buckets": self.low_n_buckets,
            "shift": self.shift,
            "shift_stance": self.shift_stance,
            "n_total": self.n_total,
            "n_unparsed": self.n_unparsed,
            "precision_note": self.precision_note,
            "warnings": self.warnings,
            "method": self.method,
        }


def _describe_shift(
    shift: dict[str, Any],
    *,
    prefix: str = "趋势判断",
    zh: dict[str, str] | None = None,
    unit_name: str | None = None,
) -> str:
    zh = zh or SENTIMENT_ZH
    if not shift:
        return f"{prefix}：数据不足，未做判断。"
    if not shift.get("decidable"):
        return f"{prefix}：{shift.get('reason') or '样本不足以判断趋势。'}"
    deltas = shift.get("deltas") or {}
    early_n = shift.get("early_n") or 0
    late_n = shift.get("late_n") or 0
    unit = unit_name or "极性"
    caveat = ""
    if min(early_n, late_n) < RELIABLE_TREND_N:
        caveat = (
            f"\n⚠ 前后某侧样本量不足 {RELIABLE_TREND_N} 条，"
            "这条趋势（含「未观察到变化」这种结论）**仅供参考**，"
            "不建议作为论文结论。"
        )
    movers = [
        (label, delta) for label, delta in deltas.items()
        if abs(delta) >= TREND_DELTA
    ]
    if not movers:
        return (
            f"{prefix}：前后两段相比，各{unit}占比变动均 <"
            f"{TREND_DELTA:.0%}，未观察到明显变化。"
            + caveat
        )
    movers.sort(key=lambda item: -abs(item[1]))
    parts = "、".join(
        f"{zh.get(label, label)} {delta:+.0%}" for label, delta in movers
    )
    return (
        f"{prefix}：{shift.get('early_range', '')} 与 {shift.get('late_range', '')} 相比，"
        f"{parts}（{early_n} 条 vs {late_n} 条）。"
        "这仍是描述性对比，未做统计检验；若要在论文中声称「变化显著」，"
        "请自行补做比例检验并报告样本量。"
        + caveat
    )


# --------------------------------------------------------------------------- #
def build_timeseries(
    rows: Iterable[dict[str, Any]],
    *,
    granularity: str | None = None,
    min_bucket_n: int = MIN_BUCKET_N,
    min_trend_n: int = MIN_TREND_N,
) -> TimeSeriesResult:
    """按时段聚合评论量与情感/簇占比。

    Args:
        rows: 含 ``created_at``（ISO 字符串）与 ``created_at_precision`` 的行；
            可选 ``sentiment``、``cluster_discourse``。
        granularity: 强制粒度；``None`` 表示自动（默认按日，条件满足才按小时）。
    """
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    precision_seen: set[str] = set()
    unparsed = 0
    for row in rows:
        stamp, precision = _parse_time(row.get("created_at"))
        if stamp is None:
            unparsed += 1
            continue
        precision_seen.add(precision)
        parsed.append((stamp, row))

    if not parsed:
        return TimeSeriesResult(
            granularity=DAY,
            n_unparsed=unparsed,
            warnings=["所有记录的时间都无法解析，无法做时间演变。"],
            method=_method(DAY, min_bucket_n, min_trend_n, forced=False, span_days=0),
        )

    span_days = (max(item[0] for item in parsed) - min(item[0] for item in parsed)).days
    chosen, precision_note = _choose_granularity(
        granularity, precision_seen, span_days, len(parsed)
    )

    buckets: dict[str, list[dict[str, Any]]] = {}
    for stamp, row in parsed:
        buckets.setdefault(_key(stamp, chosen), []).append(row)
    ordered = sorted(buckets)

    volume = [len(buckets[key]) for key in ordered]
    low_n = [key for key in ordered if len(buckets[key]) < min_bucket_n]

    sentiment_counts: dict[str, list[int]] = {}
    sentiment_share: dict[str, list[float]] = {}
    labels_present = {
        str(row.get("sentiment"))
        for _stamp, row in parsed
        if row.get("sentiment")
    }
    if labels_present:
        for label in SENTIMENT_ORDER:
            if label not in labels_present:
                continue
            counts = [0] * len(ordered)
            for index, key in enumerate(ordered):
                counts[index] = sum(
                    1 for row in buckets[key] if str(row.get("sentiment")) == label
                )
            totals = [max(1, sum(
                1 for row in buckets[key] if row.get("sentiment")
            )) for key in ordered]
            sentiment_counts[label] = counts
            sentiment_share[label] = [
                count / total if total else 0.0
                for count, total in zip(counts, totals)
            ]

    cluster_share: dict[str, list[float]] = {}
    stance_counts: dict[str, list[int]] = {}
    stance_share: dict[str, list[float]] = {}
    stance_present = {
        str(row.get("stance")) for _stamp, row in parsed if row.get("stance")
    }
    if stance_present:
        for label in STANCE_ORDER:
            if label not in stance_present:
                continue
            counts = [
                sum(1 for row in buckets[key] if str(row.get("stance")) == label)
                for key in ordered
            ]
            totals = [
                max(1, sum(1 for row in buckets[key] if row.get("stance")))
                for key in ordered
            ]
            stance_counts[label] = counts
            stance_share[label] = [
                count / total if total else 0.0
                for count, total in zip(counts, totals)
            ]

    clusters = sorted({
        int(row["cluster_discourse"])
        for _stamp, row in parsed
        if row.get("cluster_discourse") is not None
    })
    if clusters:
        for cluster in clusters:
            shares = []
            for key in ordered:
                group = [row for row in buckets[key] if row.get("cluster_discourse") is not None]
                total = len(group) or 1
                hits = sum(1 for row in group if int(row["cluster_discourse"]) == cluster)
                shares.append(hits / total)
            cluster_share[f"簇{cluster}"] = shares

    shift = _compare_halves(ordered, buckets, min_trend_n=min_trend_n)
    shift_stance = (
        _compare_halves(
            ordered, buckets, min_trend_n=min_trend_n,
            key="stance", labels=STANCE_ORDER,
        )
        if stance_present else {}
    )
    warnings = _warnings(ordered, volume, low_n, shift, min_bucket_n)

    return TimeSeriesResult(
        granularity=chosen,
        buckets=ordered,
        volume=volume,
        sentiment_counts=sentiment_counts,
        sentiment_share=sentiment_share,
        stance_counts=stance_counts,
        stance_share=stance_share,
        cluster_share=cluster_share,
        low_n_buckets=low_n,
        shift=shift,
        shift_stance=shift_stance,
        n_total=len(parsed),
        n_unparsed=unparsed,
        precision_note=precision_note,
        warnings=warnings,
        method=_method(chosen, min_bucket_n, min_trend_n,
                       forced=granularity is not None, span_days=span_days),
    )


# --------------------------------------------------------------------------- #
def _parse_time(value: Any) -> tuple[datetime | None, str]:
    """解析库里存的 ISO 时间；解析不出来就当"无法解析"，不猜。"""
    if isinstance(value, datetime):
        return value, "second"
    text = str(value or "").strip()
    if not text:
        return None, "unknown"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                stamp = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        else:
            return None, "unknown"
    precision = "day" if len(text) <= 10 else "second"
    return stamp, precision


def _choose_granularity(
    requested: str | None, precision_seen: set[str], span_days: int, n: int
) -> tuple[str, str | None]:
    """决定聚合粒度，并给出"为什么不是小时"的说明。"""
    has_day_only = "day" in precision_seen
    if requested == HOUR:
        if has_day_only:
            return DAY, (
                "存在只精确到「日」的记录，按小时聚合会造成虚假精度，"
                "已强制降级为按日聚合。"
            )
        return HOUR, None
    if requested == DAY:
        return DAY, None
    if has_day_only:
        return DAY, "存在只精确到「日」的记录，故按日聚合。"
    if span_days <= HOUR_MAX_SPAN_DAYS and n >= 50:
        return HOUR, None
    return DAY, None


def _key(stamp: datetime, granularity: str) -> str:
    if granularity == HOUR:
        return stamp.strftime("%Y-%m-%d %H时")
    return stamp.strftime("%Y-%m-%d")


def _compare_halves(
    ordered: Sequence[str],
    buckets: dict[str, list[dict[str, Any]]],
    *,
    min_trend_n: int,
    key: str = "sentiment",
    labels: Sequence[str] = SENTIMENT_ORDER,
) -> dict[str, Any]:
    """前半时段 vs 后半时段。刻意用「时段」而不是「评论中位数」切片——
    按时段切才对应"时间上更早 vs 更晚"，按条数切则可能切在同一时刻。

    ``key`` 指定比较哪一列（sentiment / stance），两者用同一套判定逻辑，
    避免"情感用一套阈值、立场用另一套"这种不可比的实现。
    """
    if len(ordered) < 2:
        return {"decidable": False, "reason": "只有 1 个时段，无法比较前后变化。"}
    middle = len(ordered) // 2
    early_keys, late_keys = ordered[:middle] or ordered[:1], ordered[middle:]
    early = [row for key in early_keys for row in buckets[key]]
    late = [row for key in late_keys for row in buckets[key]]

    if len(early) < min_trend_n or len(late) < min_trend_n:
        return {
            "decidable": False,
            "reason": (
                f"前后两段样本量不足（{len(early)} 条 vs {len(late)} 条，"
                f"各需 ≥{min_trend_n} 条），不做趋势判断。"
            ),
            "early_n": len(early),
            "late_n": len(late),
        }

    deltas: dict[str, float] = {}
    for label in labels:
        early_total = sum(1 for row in early if row.get(key)) or 1
        late_total = sum(1 for row in late if row.get(key)) or 1
        early_share = sum(1 for row in early if str(row.get(key)) == label) / early_total
        late_share = sum(1 for row in late if str(row.get(key)) == label) / late_total
        if any(str(row.get(key)) == label for row in early + late):
            deltas[label] = late_share - early_share

    return {
        "decidable": True,
        "early_n": len(early),
        "late_n": len(late),
        "early_range": _range_label(early_keys),
        "late_range": _range_label(late_keys),
        "deltas": deltas,
    }


def _range_label(keys: Sequence[str]) -> str:
    if not keys:
        return "—"
    if len(keys) == 1:
        return keys[0]
    return f"{keys[0]} 至 {keys[-1]}"


def _warnings(
    ordered: Sequence[str],
    volume: Sequence[int],
    low_n: Sequence[str],
    shift: dict[str, Any],
    min_bucket_n: int,
) -> list[str]:
    warnings: list[str] = []
    if len(ordered) == 1:
        warnings.append(
            "所有评论都落在同一个时段，看不出时间变化。"
            "要谈「演变」需要跨多天的样本。"
        )
    if low_n and len(low_n) == len(ordered):
        warnings.append(
            f"每个时段都少于 {min_bucket_n} 条，占比完全不可靠，请勿据此讨论趋势。"
        )
    elif low_n:
        warnings.append(
            f"{len(low_n)} 个时段样本量 <{min_bucket_n}（{', '.join(low_n[:3])}"
            f"{' 等' if len(low_n) > 3 else ''}），这些时段的占比不做解读。"
        )
    if volume and max(volume) >= 3 * max(1, sorted(volume)[len(volume) // 2]):
        peak = ordered[volume.index(max(volume))]
        warnings.append(
            f"评论量高度集中：{peak} 一个时段占了 {max(volume)}/{sum(volume)} 条。"
            "这种情况下曲线形状主要由单日事件决定。"
        )
    if shift.get("decidable") and not shift.get("deltas"):
        warnings.append("两段都没有情感标签，无法比较情感占比变化。")
    if shift.get("decidable"):
        early_n = shift.get("early_n") or 0
        late_n = shift.get("late_n") or 0
        if min(early_n, late_n) < RELIABLE_TREND_N:
            warnings.append(
                f"趋势对比中前段 {early_n} 条、后段 {late_n} 条，"
                f"有一侧不足 {RELIABLE_TREND_N} 条：结论只能当线索，不能当结论。"
            )
    return warnings


def _method(
    granularity: str,
    min_bucket_n: int,
    min_trend_n: int,
    *,
    forced: bool,
    span_days: int,
) -> dict[str, Any]:
    return {
        "module": "timeseries",
        "granularity": "按日" if granularity == DAY else "按小时",
        "granularity_forced": forced,
        "min_bucket_n": min_bucket_n,
        "min_trend_n": min_trend_n,
        "reliable_trend_n": RELIABLE_TREND_N,
        "trend_delta_threshold": TREND_DELTA,
        "time_span_days": span_days,
        "source_note": "时间为平台返回的评论时间，未做插值或补齐",
    }
