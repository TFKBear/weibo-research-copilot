"""清洗影响报告（模块 B2，强制输出、不可跳过）。

为什么必须是一个「报告」而不是一行日志：
社科研究里清洗是**规范性判断**，它决定了最终结论建立在什么样本上。
商业工具追求「一键干净」，本工具要求研究者必须能回答三个问题：

1. 每一步移除了多少？占比多少？
2. 被移除的到底是什么东西？（随机抽样 + 逐条可恢复）
3. 清洗是否引入了**系统性偏差**？（前后分布对比）

第 3 点尤其关键：如果清洗把「粉丝数低的新用户」整批删掉，
那么后续任何关于「公众意见」的结论都只代表「老用户的声音」——
这个偏差必须在清洗报告里就暴露出来，而不是等读者去质疑。
"""

from __future__ import annotations

import random
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 20240101


# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Distribution:
    """一组数值的摘要，用于前后对比。"""

    label: str
    n: int
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None

    def as_row(self) -> str:
        def fmt(value: float | None) -> str:
            if value is None:
                return "—"
            # 评论长度是个位数~几十，小数有意义；
            # 粉丝数动辄几十万，小数只是噪声，用千分位整数更好读。
            return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.2f}"

        cells = [fmt(self.mean), fmt(self.median), fmt(self.p10), fmt(self.p90)]
        # 列宽按本行最长的数字动态计算：均值可能是 885,258，也可能是 23.90，
        # 固定宽度会挤到下一列去。
        width = max(9, max(len(cell) for cell in cells) + 2)
        n_width = max(len(str(self.n)), 4)
        return (
            f"n={self.n:<{n_width}}"
            f"均值={cells[0]:<{width}}中位数={cells[1]:<{width}}"
            f"P10={cells[2]:<{width}}P90={cells[3]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
        }


@dataclass(slots=True)
class RemovalSample:
    """一条被移除的样本，供研究者判断是否「误杀」。"""

    cid: str
    post_mid: str
    text: str
    reasons: list[str]
    followers_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "post_mid": self.post_mid,
            "text": self.text,
            "reasons": self.reasons,
            "followers_count": self.followers_count,
        }


@dataclass(slots=True)
class StepStat:
    key: str
    label: str
    description: str
    removed: int
    share: float


@dataclass(slots=True)
class CleaningReport:
    """一次清洗的完整影响记录。"""

    tier: str
    pipeline_version: str
    total_input: int
    total_kept: int
    total_removed: int
    steps: list[StepStat] = field(default_factory=list)
    samples: list[RemovalSample] = field(default_factory=list)
    length_before: Distribution | None = None
    length_after: Distribution | None = None
    followers_before: Distribution | None = None
    followers_after: Distribution | None = None
    sample_size: int = DEFAULT_SAMPLE_SIZE
    seed: int = DEFAULT_SEED
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def retention_rate(self) -> float:
        if not self.total_input:
            return 0.0
        return self.total_kept / self.total_input

    def render(self) -> str:
        lines: list[str] = []
        lines.append("清洗影响报告")
        lines.append("=" * 62)
        lines.append(f"档位：{self.tier}    管线版本：{self.pipeline_version}")
        lines.append(
            f"输入 {self.total_input} 条 → 保留 {self.total_kept} 条"
            f"（{self.retention_rate:.1%}），移除 {self.total_removed} 条"
        )
        lines.append("")
        lines.append("各步骤移除数量：")
        if not self.steps:
            lines.append("  （本次未启用任何清洗步骤）")
        for stat in self.steps:
            lines.append(
                f"  - {_pad(stat.label, 12)}{stat.removed:>6} 条"
                f"  （占输入 {stat.share:.1%}）"
            )
        lines.append("")
        lines.append(f"被移除样本抽查（随机 {self.sample_size} 条，随机种子 {self.seed}）")
        if not self.samples:
            lines.append("  （无）")
        for index, sample in enumerate(self.samples, start=1):
            reasons = "/".join(sample.reasons)
            text = sample.text.replace("\n", " ⏎ ")
            lines.append(f"  {index:>2}. [{reasons}] {text}")
        if self.samples:
            lines.append(
                "      ↳ 如判定为误杀，可用该条 cid 调用 restore() 恢复："
            )
            lines.append(
                "        " + ", ".join(sample.cid for sample in self.samples[:3])
                + (" …" if len(self.samples) > 3 else "")
            )
        lines.append("")
        lines.append("清洗前后分布对比（判断是否引入系统性偏差）：")
        if self.length_before and self.length_after:
            lines.append("  评论长度（字符数）")
            lines.append("    清洗前  " + self.length_before.as_row())
            lines.append("    清洗后  " + self.length_after.as_row())
        if self.followers_before and self.followers_after:
            lines.append("  评论者粉丝数（接口返回者）")
            lines.append("    清洗前  " + self.followers_before.as_row())
            lines.append("    清洗后  " + self.followers_after.as_row())
        else:
            lines.append("  评论者粉丝数：接口未返回足够数据，无法对比。")
        if self.notes:
            lines.append("")
            lines.append("说明：")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "pipeline_version": self.pipeline_version,
            "total_input": self.total_input,
            "total_kept": self.total_kept,
            "total_removed": self.total_removed,
            "retention_rate": round(self.retention_rate, 4),
            "steps": [
                {
                    "key": stat.key,
                    "label": stat.label,
                    "description": stat.description,
                    "removed": stat.removed,
                    "share": round(stat.share, 4),
                }
                for stat in self.steps
            ],
            "samples": [sample.to_dict() for sample in self.samples],
            "sample_size": self.sample_size,
            "seed": self.seed,
            "length_before": self.length_before.to_dict() if self.length_before else None,
            "length_after": self.length_after.to_dict() if self.length_after else None,
            "followers_before": (
                self.followers_before.to_dict() if self.followers_before else None
            ),
            "followers_after": (
                self.followers_after.to_dict() if self.followers_after else None
            ),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
def build_report(
    cleaning_result: Any,
    source_rows: list[dict],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> CleaningReport:
    """把 ``CleaningResult`` 渲染成影响报告。

    Args:
        cleaning_result: ``wrc.clean.basic.CleaningResult``。
        source_rows: 与 ``cleaning_result.rows`` 同序的原始行，
            需要 ``text_raw``，可选 ``followers_count``。
        sample_size: 被移除样本的抽查条数（需求规定 20）。
        seed: 抽样随机种子。固定种子使「哪 20 条被看到」可复现，
            这本身就是可追溯性的一部分。
    """
    from .basic import STEPS

    specs = {spec.key: spec for spec in STEPS}
    total_input = cleaning_result.total_input
    kept_indices = [i for i, row in enumerate(cleaning_result.rows) if row["kept"]]
    removed_indices = [i for i, row in enumerate(cleaning_result.rows)
                       if not row["kept"]]

    steps: list[StepStat] = []
    for key in cleaning_result.active_steps:
        spec = specs.get(key)
        removed = cleaning_result.step_removed.get(key, 0)
        steps.append(
            StepStat(
                key=key,
                label=spec.label if spec else key,
                description=spec.description if spec else "",
                removed=removed,
                share=(removed / total_input) if total_input else 0.0,
            )
        )

    rng = random.Random(seed)
    picked = sorted(rng.sample(removed_indices, min(sample_size, len(removed_indices))))
    samples = [
        RemovalSample(
            cid=cleaning_result.rows[i]["cid"],
            post_mid=cleaning_result.rows[i]["post_mid"],
            text=str(source_rows[i].get("text_raw") or ""),
            reasons=list(cleaning_result.rows[i].get("drop_reasons") or []),
            followers_count=source_rows[i].get("followers_count"),
        )
        for i in picked
    ]

    length_before = _distribution(
        "清洗前", [len(str(source_rows[i].get("text_raw") or "")) for i in kept_indices]
        + [len(str(source_rows[i].get("text_raw") or "")) for i in removed_indices]
    )
    length_after = _distribution(
        "清洗后",
        [len(str(cleaning_result.rows[i]["cleaned_text"])) for i in kept_indices],
    )
    followers_before = _distribution(
        "清洗前", [source_rows[i].get("followers_count") for i in range(len(source_rows))]
    )
    followers_after = _distribution(
        "清洗后",
        [source_rows[i].get("followers_count") for i in kept_indices],
    )

    notes = [
        "HTML 标签剥离发生在解析层而非本清洗步骤中；表情以 token 形式保留（如 [doge]），"
        "因为表情在中文微博情感表达中承载独立信号。",
        "精确去重保留接口最先返回的那条记录；若需以互动量最高者为准，属于研究者的"
        "判断，工具不替你决定。",
    ]
    if cleaning_result.tier == "light":
        notes.append(
            "本次未执行广告词过滤、僵尸号过滤、营销/水军文案过滤；"
            "因此所得样本仍可能包含营销与协同行为内容，方法卡片会如实披露。"
        )

    return CleaningReport(
        tier=cleaning_result.tier,
        pipeline_version=cleaning_result.pipeline_version,
        total_input=total_input,
        total_kept=len(kept_indices),
        total_removed=len(removed_indices),
        steps=steps,
        samples=samples,
        length_before=length_before,
        length_after=length_after,
        followers_before=followers_before,
        followers_after=followers_after,
        sample_size=sample_size,
        seed=seed,
        notes=notes,
    )


def _distribution(label: str, values: Iterable[Any]) -> Distribution | None:
    """计算分布摘要；空集或全为非数值时返回 None。

    必须容忍脏值：分布统计是"最后一道报表"，它不该因为某个字段意外变成
    ``"2733.7万"`` 这样的字符串就整份崩掉 —— 那会连带把清洗决策一起丢掉。
    """
    numbers = sorted(v for v in (_numeric(x) for x in values) if v is not None)
    if len(numbers) < 2:
        return None
    return Distribution(
        label=label,
        n=len(numbers),
        mean=round(statistics.fmean(numbers), 2),
        median=round(statistics.median(numbers), 2),
        p10=round(_percentile(numbers, 0.10), 2),
        p90=round(_percentile(numbers, 0.90), 2),
    )


def _numeric(value: Any) -> float | None:
    """尽力把一个字段值理解为数字。无法理解时返回 None（该值不参与统计）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            from ..weibo.parsing import to_count

            parsed = to_count(text)
            return float(parsed) if parsed else None
    return None


def _percentile(sorted_values: list[float], q: float) -> float:
    """线性插值分位数；不引第三方依赖，口径与 numpy 默认一致。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _display_width(text: str) -> int:
    """按终端显示宽度计算：中日韩字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度补空格。

    用 ``f"{label:<12}"`` 对齐中英混排会错位：Python 数的是字符数，
    终端显示的是列数，一个汉字占两列。清洗报告是表格化文本，错位会很刺眼。
    """
    deficit = width - _display_width(text)
    return text + " " * max(0, deficit)
