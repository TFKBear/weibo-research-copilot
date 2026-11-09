"""情感 × 立场 交叉表。

为什么要这张表：它是**研究结论真正的落点**。

单看情感分布只能说"大家情绪很负面"；单看立场分布只能说"多数人反对"。
但研究者真正想写进论文的往往是那格**"情绪负面、立场支持"**——
也就是"认同政策目标，但担心执行方式"的那批人。这类人在两个维度
分开看时会被完全抹平：情感表里他们是负面，立场表里他们是支持。

实现上直接复用 `build_crosstab`，所以 k-匿名抑制规则与别处完全一致：
单元格 <5 条一律不展示。这里额外做两件事：

1. **行内占比**（某立场内部各情感的比例）——比绝对数更能说明结构；
2. 被抑制的单元格**不参与占比计算**，占比里直接显示「—」。
   把被抑制的格子当 0 算占比，是最隐蔽的一种造假。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..privacy.anonymize import MIN_GROUP_SIZE
from .crosstab import CrosstabResult, build_crosstab

#: 行（立场）固定顺序：语义顺序比样本量大小重要。
STANCE_ORDER: tuple[str, ...] = ("支持", "反对", "中立", "无关")
#: 列（情感）固定顺序。
SENTIMENT_ORDER: tuple[str, ...] = ("正面", "中性", "负面")

_SENTIMENT_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}
_STANCE_ZH = {"support": "支持", "oppose": "反对",
              "neutral": "中立", "irrelevant": "无关"}


def sentiment_by_stance(
    rows: Sequence[dict[str, Any]], *, min_n: int = MIN_GROUP_SIZE
) -> CrosstabResult:
    """立场 × 情感 交叉表（行=立场，列=情感）。"""

    def group_of(row: dict[str, Any]) -> str | None:
        return _STANCE_ZH.get(str(row.get("stance") or ""))

    def attribute_of(row: dict[str, Any]) -> str | None:
        return _SENTIMENT_ZH.get(str(row.get("sentiment") or ""))

    return build_crosstab(
        rows,
        group_of=group_of,
        attribute_of=attribute_of,
        attribute_name="sentiment",
        attribute_order=SENTIMENT_ORDER,
        group_order=STANCE_ORDER,
        min_n=min_n,
    )


def stance_by_discourse(
    rows: Sequence[dict[str, Any]],
    assignments: dict[str, int],
    *,
    min_n: int = MIN_GROUP_SIZE,
) -> CrosstabResult:
    """话语簇 × 立场 交叉表（行=簇，列=立场）。

    用来回答"不同话语阵营的态度结构是否不同"：
    簇 A 可能 8 成反对，簇 B 可能一半一半——这种差异在单看聚类或单看立场时都看不出来。
    """

    def group_of(row: dict[str, Any]) -> str | None:
        cluster = assignments.get(str(row.get("cid")))
        return None if cluster is None else f"簇{cluster}"

    def attribute_of(row: dict[str, Any]) -> str | None:
        return _STANCE_ZH.get(str(row.get("stance") or ""))

    clusters = sorted(set(assignments.values()))
    return build_crosstab(
        rows,
        group_of=group_of,
        attribute_of=attribute_of,
        attribute_name="stance",
        attribute_order=STANCE_ORDER,
        group_order=[f"簇{cluster}" for cluster in clusters],
        min_n=min_n,
    )


def row_shares(table: CrosstabResult) -> dict[str, dict[str, float | None]]:
    """行内占比。被抑制的单元格返回 None（显示为「—」），不按 0 计入分母。"""
    out: dict[str, dict[str, float | None]] = {}
    for group in table.groups:
        cells = table.table.get(group)
        if cells is None:
            continue
        visible_total = sum(
            value for value in cells.values() if isinstance(value, int)
        )
        shares: dict[str, float | None] = {}
        for attribute in table.attributes:
            if attribute not in cells:
                # 该组里这个类别一条都没有：占比是 0，不是"被抑制"
                shares[attribute] = 0.0 if visible_total else None
                continue
            value = cells[attribute]
            if value is None:
                shares[attribute] = None  # 被抑制：不给数字，也不进分母
            elif visible_total == 0:
                shares[attribute] = None
            else:
                shares[attribute] = value / visible_total
        out[group] = shares
    return out


def render(
    table: CrosstabResult,
    *,
    title: str | None = None,
    group_label: str = "立场",
) -> str:
    """渲染成"计数 + 行内占比"两行一组的文字表。

    ``title`` / ``group_label`` 可换：同一个渲染器要服务
    「立场 × 情感」与「簇 × 立场」两张表，各写一遍必然各错一遍。
    """
    lines = [
        title or f"情感 × 立场 交叉表（最小可展示样本 n≥{table.min_n}）",
        "=" * 62,
    ]
    if not table.table:
        lines.append("所有分组样本量过小，整表不予展示。")
        for group, size in table.suppressed_groups:
            lines.append(f"  · 「{group}」n={size} < {table.min_n}")
        return "\n".join(lines)

    shares = row_shares(table)
    header = group_label.ljust(10) + "".join(name.rjust(9) for name in table.attributes)
    lines.append(header + "   合计")
    lines.append("-" * 62)
    for group in table.groups:
        cells = table.table.get(group)
        total = table.group_totals.get(group, 0)
        if cells is None:
            lines.append(f"{group:<10}" + "（样本量过小，不予展示）")
            continue
        counts = "".join(
            ("—".rjust(9) if cells.get(name) is None else str(cells[name]).rjust(9))
            for name in table.attributes
        )
        lines.append(f"{group:<10}{counts}{total:>7}")
        share_cells = shares.get(group, {})
        rendered = "".join(
            ("—".rjust(9) if share_cells.get(name) is None
             else f"{share_cells[name]:.0%}".rjust(9))
            for name in table.attributes
        )
        lines.append(f"{'（行内占比）':<10}{rendered}")

    lines.append("")
    if table.suppressed_cells:
        lines.append(
            f"注：{len(table.suppressed_cells)} 个单元格样本量 <{table.min_n}，"
            "显示为「—」，且不计入行内占比的分母。"
        )
    if table.suppressed_groups:
        lines.append(
            f"注：{len(table.suppressed_groups)} 个立场分组整体 <{table.min_n}，整行不展示。"
        )

    callout = highlight(table)
    if callout:
        lines.append("")
        lines.append(callout)
    return "\n".join(lines)


def highlight(table: CrosstabResult) -> str:
    """点出最值得写进论文的那一格：立场支持但情感负面。

    这不是"自动结论"，只是把研究者最容易漏看的那一格摆到台面上；
    样本量不足时它**拒绝**给数字。
    """
    cells = table.table.get("支持")
    if cells is None:
        return ""
    value = cells.get("负面")
    if value is None:
        return (
            f"「支持 × 负面」这一格样本量 <{table.min_n}，不予解读——"
            "但这一格恰恰是「认同目标、担心执行」的关键证据，"
            "样本扩大后值得专门看。"
        )
    total = table.group_totals.get("支持", 0)
    share = value / total if total else 0.0
    return (
        f"值得注意：「支持」立场里有 {value} 条（{share:.0%}）情感判为负面——"
        "这类评论通常表达的是**认同政策目标、但担心执行方式**。"
        "只看情感分布或只看立场分布都会把它抹平。"
    )


def to_csv_rows(
    table: CrosstabResult,
    *,
    group_label: str = "立场",
    keep_support_negative_column: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """导出用：计数与行内占比各一列，被抑制的格子写「—」。

    ``keep_support_negative_column`` 只对「立场 × 情感」表有意义
    （那一列是"支持但负面"的关键占比）；「簇 × 立场」表不需要它。
    """
    columns = [group_label, "该组总数", *table.attributes]
    if keep_support_negative_column:
        columns.append("支持但负面")
    columns.append("备注")
    shares = row_shares(table)
    rows: list[dict[str, Any]] = []
    for group in table.groups:
        cells = table.table.get(group)
        row: dict[str, Any] = {group_label: group,
                               "该组总数": table.group_totals.get(group, 0)}
        if cells is None:
            for attribute in table.attributes:
                row[attribute] = "—"
            if keep_support_negative_column:
                row["支持但负面"] = "—"
            row["备注"] = f"整组样本量 <{table.min_n}，不予展示"
            rows.append(row)
            continue
        hidden = 0
        for attribute in table.attributes:
            if attribute not in cells:
                row[attribute] = "0"
            elif cells[attribute] is None:
                row[attribute] = "—"
                hidden += 1
            else:
                row[attribute] = cells[attribute]
        if keep_support_negative_column:
            share = shares.get(group, {}).get("负面")
            row["支持但负面"] = (
                f"{share:.1%}" if group == "支持" and share is not None else ""
            )
        row["备注"] = (f"{hidden} 个单元格样本量 <{table.min_n}，已抑制"
                       if hidden else "")
        rows.append(row)
    return columns, rows
