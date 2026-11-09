"""聚类 × 属性 交叉表，强制小群体抑制（模块 C 的伦理要求）。

需求原文：若分析结果涉及聚类后的人口属性交叉展示，当某聚类样本量 <5 时，
自动抑制该交叉表并提示"样本量过小，不予展示"。

为什么这条不能只在 UI 上做：论文里的表格是从导出文件来的。
如果抑制逻辑只在界面上生效，研究者把 CSV 拉到 Excel 里自己交叉一下，
k-匿名就形同虚设。所以抑制必须发生在**生成表格数据的那一层**。

⚠ 现状说明（必须让研究者知道）：
微博评论接口不返回性别、年龄、学历等人口属性，本工具也不推断它们。
当前唯一可用的分组属性是「情感倾向」与「话语簇」——这两者不属于敏感属性，
被抑制只因为样本量本身没有解释力。真正的**人口属性交叉表**
需要研究者另行提供属性数据（例如问卷匹配），届时本模块的抑制规则同样适用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..privacy.anonymize import MIN_GROUP_SIZE, suppress_small_groups

#: 明确标注为"人口属性"的字段名。只有这些属性会触发伦理提示语，
#: 因为它们可能被用来反推个体。
DEMOGRAPHIC_KEYS = frozenset({"province", "gender", "age", "education", "occupation"})

#: 情感三极性的固定顺序（导出列顺序稳定，便于并排比较）。
SENTIMENT_ORDER = ("positive", "neutral", "negative")


def _text_cell(cells: dict[str, int | None], attribute: str) -> str:
    """文字表里的一个单元格。

    区分三件事：被抑制（``—``）、真的有 0 条（``0``）、有若干条（数字）。
    把"没有这个类别"和"样本太小被藏起来"画成同一个符号，会让读者
    以为工具藏了很多东西。
    """
    if attribute not in cells:
        return "0"
    value = cells[attribute]
    return "—" if value is None else str(value)


@dataclass(slots=True)
class CrosstabResult:
    groups: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    table: dict[str, dict[str, int | None]] = field(default_factory=dict)
    suppressed_cells: list[tuple[str, str, int]] = field(default_factory=list)
    suppressed_groups: list[tuple[str, int]] = field(default_factory=list)
    #: 各分组在被抑制**之前**的真实总数。
    #: 必须单独保存：抑制后单元格都是 None，从表里反推会算成 0，
    #: 于是"该组 5 条评论"会被写成"该组 0 条"——那是凭空造出来的数字。
    group_totals: dict[str, int] = field(default_factory=dict)
    min_n: int = MIN_GROUP_SIZE
    attribute_is_demographic: bool = False

    def render(self) -> str:
        lines = [f"交叉表（最小可展示样本 n≥{self.min_n}）", "=" * 62]
        if not self.table:
            lines.append("所有分组样本量过小，整表不予展示。")
            for group, size in self.suppressed_groups:
                lines.append(f"  · 「{group}」n={size} < {self.min_n}")
            return "\n".join(lines)

        header = "分组".ljust(12) + "".join(name.rjust(10) for name in self.attributes)
        lines.append(header)
        for group in self.groups:
            cells = self.table.get(group)
            if cells is None:
                lines.append(f"{group[:10]:<12}" + "（样本量过小，不予展示）")
                continue
            rendered = "".join(
                _text_cell(cells, attribute).rjust(10) for attribute in self.attributes
            )
            lines.append(f"{group[:10]:<12}{rendered}")
        if self.suppressed_cells:
            lines.append("")
            lines.append(
                f"注：{len(self.suppressed_cells)} 个单元格因样本量 <{self.min_n} 显示为「—」。"
            )
        if self.suppressed_groups:
            lines.append(
                f"注：{len(self.suppressed_groups)} 个分组整体样本量 <{self.min_n}，整行不予展示。"
            )
        if self.attribute_is_demographic:
            lines.append(
                "注：本表涉及人口属性，已按 k-匿名规则抑制小群体，避免反推个体。"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "attributes": self.attributes,
            "table": {k: dict(v) for k, v in self.table.items()},
            "suppressed_cells": [list(item) for item in self.suppressed_cells],
            "suppressed_groups": [list(item) for item in self.suppressed_groups],
            "group_totals": dict(self.group_totals),
            "min_n": self.min_n,
            "attribute_is_demographic": self.attribute_is_demographic,
        }

    def to_csv_rows(self) -> tuple[list[str], list[dict[str, Any]]]:
        """导出用的交叉表，**抑制在数据层就完成**。

        为什么必须导出而不是让研究者自己在 Excel 里交叉：论文里的表就是从
        CSV 来的。如果抑制只发生在界面或报告里，研究者把 CSV 拉到 Excel
        自己交叉一下就绕过去了，k-匿名形同虚设。

        被抑制的单元格写 ``—`` 而不是留空或写 0：
        - 留空在 Excel 里可能被当成缺失值自动跳过；
        - 写 0 是彻头彻尾的错误（那是"该类别没有评论"的意思）；
        - ``—`` 是数字列里非数字，Excel 不会拿它算平均，人也一眼看懂。

        同时要区分两种"没有数"：**被抑制的**（曾有很小的计数）和
        **真的为零的**（该组里根本没有这个类别）。后者照实写 0，
        否则读者会以为工具把什么都藏了。
        """
        columns = ["分组", "该组总样本", *self.attributes, "备注"]
        rows: list[dict[str, Any]] = []
        for group in self.groups:
            cells = self.table.get(group)
            row: dict[str, Any] = {"分组": group}
            total = self.group_totals.get(group)
            if cells is None:
                row["该组总样本"] = total if total is not None else "—"
                for attribute in self.attributes:
                    row[attribute] = "—"
                row["备注"] = f"整组样本量 <{self.min_n}，不予展示"
            else:
                row["该组总样本"] = total if total is not None else sum(
                    value for value in cells.values() if isinstance(value, int)
                )
                hidden = 0
                for attribute in self.attributes:
                    if attribute not in cells:
                        # 该组没有这个属性值的记录：是 0，不是被抑制
                        row[attribute] = 0
                    elif cells[attribute] is None:
                        row[attribute] = "—"
                        hidden += 1
                    else:
                        row[attribute] = cells[attribute]
                row["备注"] = (
                    f"{hidden} 个单元格样本量 <{self.min_n}，已抑制"
                    if hidden else ""
                )
            rows.append(row)
        return columns, rows


def build_crosstab(
    rows: Iterable[dict[str, Any]],
    *,
    group_of: Callable[[dict[str, Any]], str | None],
    attribute_of: Callable[[dict[str, Any]], str | None],
    attribute_name: str = "属性",
    attribute_order: Sequence[str] | None = None,
    group_order: Sequence[str] | None = None,
    min_n: int = MIN_GROUP_SIZE,
) -> CrosstabResult:
    """构造分组 × 属性 交叉表并做小群体抑制。

    Args:
        group_of: 从一行数据取出分组标签（如话语簇）。
        attribute_of: 从一行数据取出属性值（如情感倾向、省份）。
        attribute_order: 固定列顺序（可选）。给定后，即使某个取值在所有分组里
            都没出现，也会保留该列并显示 0——因为"这个类别一条都没有"本身
            就是信息，不该因为没被观测到就从表里消失。
        group_order: 固定行顺序（可选）。默认按组内样本量降序，
            但"支持/反对/中立/无关"这种有语义顺序的分组不该被大小重排。
        min_n: 最小可展示样本量，默认 5（模块 C 规定）。
    """
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        group = group_of(row)
        attribute = attribute_of(row)
        if group is None or attribute is None:
            continue
        counts.setdefault(str(group), {})
        counts[str(group)][str(attribute)] = counts[str(group)].get(str(attribute), 0) + 1

    suppression = suppress_small_groups(counts, min_n=min_n)
    attributes = (
        list(attribute_order)
        if attribute_order
        else sorted({name for cells in counts.values() for name in cells})
    )
    groups = (
        [name for name in group_order if name in counts]
        if group_order
        else sorted(counts, key=lambda name: -sum(counts[name].values()))
    )
    return CrosstabResult(
        groups=groups,
        attributes=attributes,
        table=suppression.table,
        suppressed_cells=suppression.suppressed_cells,
        suppressed_groups=suppression.suppressed_groups,
        group_totals={name: sum(cells.values()) for name, cells in counts.items()},
        min_n=min_n,
        attribute_is_demographic=attribute_name in DEMOGRAPHIC_KEYS,
    )


def discourse_by_sentiment(
    rows: Sequence[dict[str, Any]], assignments: dict[str, int], *, min_n: int = MIN_GROUP_SIZE
) -> CrosstabResult:
    """话语簇 × 情感倾向 的交叉表（最常见的组合，开箱可用）。"""

    def group_of(row: dict[str, Any]) -> str | None:
        cluster = assignments.get(str(row.get("cid")))
        return None if cluster is None else f"簇{cluster}"

    def attribute_of(row: dict[str, Any]) -> str | None:
        value = row.get("sentiment")
        return str(value) if value else None

    return build_crosstab(
        rows, group_of=group_of, attribute_of=attribute_of,
        attribute_name="sentiment",
        # 固定三个极性列：某极性一条都没有时显示 0，而不是整列消失。
        # 列顺序稳定也让两次分析的表格可以直接并排比较。
        attribute_order=SENTIMENT_ORDER,
        min_n=min_n,
    )
