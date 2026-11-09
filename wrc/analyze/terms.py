"""术语随时间的变化（"大家在谈什么"的定量线索）。

为什么需要它：立场占比告诉你"态度有没有变"，但回答不了"关注点有没有变"。
延迟退休这类长周期议题，讨论往往不是态度翻转，而是**关注点迁移**——
从"到底延不延"变成"我的社保怎么交""哪年能退"。

做法刻意很朴素：把时间按 N 天分箱，统计每个词在**该箱评论中的出现率**，
再按"首末箱差值"排出上升/下降的术语。

三条限制写在结果里，不让读者误读：

1. 这是**描述性**的，没有做显著性检验；样本小的箱波动很大。
2. 出现率是"提到该词的评论占比"，不是"该词出现次数"——
   后者会被刷屏式重复带偏。
3. 词表来自分词，**同义与缩写会漏**（"社保"与"社会保险"被当成两个词）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

from .cluster import tokenize
from .timeseries import _parse_time

DEFAULT_PERIOD_DAYS = 7
#: 一个词至少在多少个箱里出现，才参与"上升/下降"排名（避免只在最后一周出现的词霸榜）。
MIN_PERIODS_PRESENT = 2
#: 首箱与末箱各自的样本量下限；低于它就只报数不排名。
MIN_SIDE_N = 20


@dataclass(slots=True)
class TermTrendResult:
    period_days: int = DEFAULT_PERIOD_DAYS
    periods: list[str] = field(default_factory=list)
    totals: list[int] = field(default_factory=list)
    shares: dict[str, list[float]] = field(default_factory=dict)
    rising: list[tuple[str, float, float]] = field(default_factory=list)
    falling: list[tuple[str, float, float]] = field(default_factory=list)
    decidable: bool = True
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"术语随时间的变化（每 {self.period_days} 天一段）", "-" * 62]
        if not self.periods:
            lines.append("没有可用的时间数据。")
            return "\n".join(lines)
        lines.append("段次：" + " | ".join(
            f"{name}(n={total})" for name, total in zip(self.periods, self.totals)
        ))
        lines.append("")
        if not self.decidable:
            lines.append("⚠ 首段或末段样本量不足，只列出词表、不做升降排名。")
        if self.rising:
            lines.append("上升最明显（首段 → 末段的提到率）：")
            for term, first, last in self.rising:
                lines.append(f"  · {term:<8}{first:>6.0%} → {last:>6.0%}  {last - first:+.0%}")
        if self.falling:
            lines.append("下降最明显：")
            for term, first, last in self.falling:
                lines.append(f"  · {term:<8}{first:>6.0%} → {last:>6.0%}  {last - first:+.0%}")
        if not self.rising and not self.falling:
            lines.append("未发现明显的术语升降。")
        for warning in self.warnings:
            lines.append(f"⚠ {warning}")
        lines.append("")
        lines.append(
            "说明：这是描述性统计，未做显著性检验；出现率按「提到该词的评论占比」计，"
            "不是词频。同义词与缩写未合并，解读时请结合作品原文核对。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "terms",
            "period_days": self.period_days,
            "periods": self.periods,
            "totals": self.totals,
            "shares": {
                term: [round(value, 4) for value in values]
                for term, values in self.shares.items()
            },
            "rising": [[term, round(a, 4), round(b, 4)] for term, a, b in self.rising],
            "falling": [[term, round(a, 4), round(b, 4)] for term, a, b in self.falling],
            "decidable": self.decidable,
            "warnings": self.warnings,
        }


def _period_key(day: date, start: date, size: int) -> int:
    return (day - start).days // size


def term_trends(
    rows: Sequence[dict[str, Any]],
    *,
    period_days: int = DEFAULT_PERIOD_DAYS,
    terms: Sequence[str] | None = None,
    top: int = 6,
) -> TermTrendResult:
    """按时间分箱统计术语出现率，并给出升降排名。

    Args:
        terms: 只看这些词（用于验证具体假设）；省略则自动从数据里找升降最明显的词。
        top: 升降各列几个。
    """
    dated: list[tuple[date, str]] = []
    for row in rows:
        stamp, _precision = _parse_time(row.get("created_at"))
        text = str(row.get("cleaned_text") or row.get("text_raw") or "")
        if stamp is None or not text:
            continue
        dated.append((stamp.date(), text))
    if len(dated) < 2:
        return TermTrendResult(period_days=period_days, warnings=["时间数据不足。"])

    start = min(day for day, _ in dated)
    end = max(day for day, _ in dated)
    count = (end - start).days // period_days + 1

    totals = [0] * count
    hits: dict[str, list[int]] = {}
    corpus: list[list[str]] = [[] for _ in range(count)]
    for day, text in dated:
        index = min(_period_key(day, start, period_days), count - 1)
        totals[index] += 1
        tokens = tokenize(text)
        corpus[index].extend(set(tokens))
        for token in set(tokens):
            hits.setdefault(token, [0] * count)
            hits[token][index] += 1
        if terms:
            for term in terms:
                if term in text:
                    hits.setdefault(term, [0] * count)
                    hits[term][index] += 1

    labels = [
        (start + timedelta(days=period_days * i)).strftime("%m-%d")
        for i in range(count)
    ]

    def share(term: str) -> list[float]:
        values = hits.get(term) or [0] * count
        return [
            values[i] / totals[i] if totals[i] else 0.0 for i in range(count)
        ]

    shares = {term: share(term) for term in hits}
    # 只拿"够样本量"的段来比较首末：末尾常常是半个采集日（一两条例），
    # 用它当"末段"会让排名完全被噪声主导。
    valid = [index for index, total in enumerate(totals) if total >= MIN_SIDE_N]
    decidable = len(valid) >= 2

    rising: list[tuple[str, float, float]] = []
    falling: list[tuple[str, float, float]] = []
    if decidable:
        first_index, last_index = valid[0], valid[-1]
        candidates = []
        for term, values in hits.items():
            if len(term) < 2 and not term.isascii():
                continue
            present = sum(1 for value in values if value)
            if present < MIN_PERIODS_PRESENT:
                continue
            first, last = shares[term][first_index], shares[term][last_index]
            candidates.append((term, first, last, last - first))
        ups = sorted([item for item in candidates if item[3] > 0], key=lambda x: -x[3])
        downs = sorted([item for item in candidates if item[3] < 0], key=lambda x: x[3])
        rising = [(term, first, last) for term, first, last, _ in ups[:top]]
        falling = [(term, first, last) for term, first, last, _ in downs[:top]]

    warnings: list[str] = []
    if not decidable:
        warnings.append(
            f"只有 {len(valid)} 个时间段样本量达到 {MIN_SIDE_N} 条，"
            "不足以比较首末变化；只列出词表。"
        )
    if count < 3:
        warnings.append("只分成不到 3 段，谈不上「随时间变化」，建议扩大采集时间范围。")
    return TermTrendResult(
        period_days=period_days,
        periods=labels,
        totals=totals,
        shares=shares,
        rising=rising,
        falling=falling,
        decidable=decidable,
        warnings=warnings,
    )


def check_terms(
    rows: Sequence[dict[str, Any]],
    terms: Sequence[str],
    *,
    period_days: int = DEFAULT_PERIOD_DAYS,
) -> str:
    """针对具体假设查一组词的走向（如"计算器/弹性/传言"是否真的在涨）。"""
    result = term_trends(rows, period_days=period_days, terms=terms, top=0)
    lines = [f"指定术语的出现率（每 {period_days} 天一段）", "-" * 62]
    header = "术语".ljust(10) + "".join(f"{name:>8}" for name in result.periods)
    lines.append(header)
    lines.append(f"{'样本':<10}" + "".join(f"{total:>8}" for total in result.totals))
    for term in terms:
        values = result.shares.get(term) or [0.0] * len(result.periods)
        lines.append(f"{term:<10}" + "".join(f"{value:>7.0%} " for value in values))
    lines.append("")
    lines.append("说明：出现率＝提到该词的评论占比；未做显著性检验。")
    return "\n".join(lines)
