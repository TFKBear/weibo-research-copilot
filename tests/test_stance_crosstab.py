"""情感 × 立场 交叉表测试。

这张表是研究结论的落点（"认同目标但担心执行"的那批人），
所以测试重点在**占比算得对**、**被抑制的格子不参与分母**、
**样本不足时不硬给结论**。
"""

from __future__ import annotations

from wrc.analyze.stance_crosstab import (
    SENTIMENT_ORDER,
    STANCE_ORDER,
    highlight,
    render,
    row_shares,
    sentiment_by_stance,
    to_csv_rows,
)


def _rows(spec: dict[tuple[str, str], int]) -> list[dict]:
    """按 {(立场, 情感): 条数} 生成行。"""
    out = []
    index = 0
    for (stance, sentiment), count in spec.items():
        for _ in range(count):
            index += 1
            out.append({
                "cid": f"c{index}",
                "stance": stance,
                "sentiment": sentiment,
            })
    return out


def _base() -> list[dict]:
    return _rows({
        ("support", "positive"): 13,
        ("support", "neutral"): 28,
        ("support", "negative"): 19,
        ("oppose", "positive"): 15,
        ("oppose", "neutral"): 36,
        ("oppose", "negative"): 223,
        ("neutral", "positive"): 7,
        ("neutral", "neutral"): 53,
        ("neutral", "negative"): 24,
        ("irrelevant", "neutral"): 30,
        ("irrelevant", "negative"): 40,
    })


def test_group_and_attribute_order_are_semantic_not_by_size():
    table = sentiment_by_stance(_base())
    # 立场按 支持/反对/中立/无关 的语义顺序，不按样本量大小
    assert table.groups == list(STANCE_ORDER)
    assert table.attributes == list(SENTIMENT_ORDER)


def test_counts_match_input():
    table = sentiment_by_stance(_base())
    assert table.table["支持"]["负面"] == 19
    assert table.table["反对"]["负面"] == 223
    assert table.group_totals["支持"] == 60
    assert table.group_totals["反对"] == 274


def test_row_shares_are_within_group():
    table = sentiment_by_stance(_base())
    shares = row_shares(table)
    assert shares["反对"]["负面"] == 223 / 274
    assert abs(shares["支持"]["负面"] - 19 / 60) < 1e-9


def test_suppressed_cell_is_not_counted_in_denominator():
    """「无关 × 正面」只有 2 条（<5）→ 被抑制，且不进入行内占比分母。"""
    rows = _base() + _rows({("irrelevant", "positive"): 2})
    table = sentiment_by_stance(rows)
    shares = row_shares(table)
    assert table.table["无关"]["正面"] is None  # 被抑制
    assert shares["无关"]["正面"] is None  # 占比也显示 —
    # 分母只算可见的 30 + 40
    assert shares["无关"]["中性"] == 30 / 70


def test_small_group_is_dropped_entirely():
    """某个立场总共只有 3 条 → 整行不展示，且不计入任何统计。"""
    rows = _base() + [{"cid": "x1", "stance": "support", "sentiment": "negative"}]
    rows = [row for row in rows if row["stance"] != "support"]
    rows += _rows({("support", "negative"): 3})
    table = sentiment_by_stance(rows)
    assert "支持" not in table.table  # 整组被抑制
    assert "支持" in table.groups  # 但仍在行列表中，以便报告注明"不予展示"
    assert "（样本量过小，不予展示）" in render(table)


def test_render_contains_counts_and_shares():
    text = render(sentiment_by_stance(_base()))
    assert "行内占比" in text
    assert "223" in text
    assert "81%" in text  # 反对组负面占比


def test_highlight_names_the_key_cell():
    table = sentiment_by_stance(_base())
    text = highlight(table)
    assert "19" in text
    assert "32%" in text
    assert "认同政策目标" in text


def test_highlight_refuses_when_cell_suppressed():
    rows = _base()
    # 把「支持×负面」缩到 3 条（<5）→ 该格被抑制
    rows = [row for row in rows if not (
        row["stance"] == "support" and row["sentiment"] == "negative"
    )]
    rows += _rows({("support", "negative"): 3})
    table = sentiment_by_stance(rows)
    text = highlight(table)
    assert "不予解读" in text
    assert "样本扩大后" in text


def test_csv_has_the_key_share_column():
    columns, rows = to_csv_rows(sentiment_by_stance(_base()))
    assert "支持但负面" in columns
    support = next(row for row in rows if row["立场"] == "支持")
    assert support["支持但负面"] == "31.7%"
    oppose = next(row for row in rows if row["立场"] == "反对")
    assert oppose["支持但负面"] == ""


def test_csv_marks_suppressed_group():
    rows = _rows({("support", "positive"): 2})  # 整组 2 条 < 5
    table = sentiment_by_stance(rows)
    _columns, payload = to_csv_rows(table)
    assert payload[0]["备注"].startswith("整组")
    assert payload[0]["正面"] == "—"


def test_empty_input_is_safe():
    table = sentiment_by_stance([])
    assert table.groups == []
    assert "整表不予展示" in render(table)


def test_rows_missing_dimensions_are_skipped():
    rows = [
        {"cid": "c1", "stance": "support", "sentiment": None},
        {"cid": "c2", "stance": None, "sentiment": "negative"},
        {"cid": "c3", "stance": "support", "sentiment": "negative"},
    ]
    table = sentiment_by_stance(rows)
    assert table.group_totals == {"支持": 1}


# --------------------------------------------------------------------------- #
# 话语簇 × 立场
# --------------------------------------------------------------------------- #
def test_stance_by_discourse_groups_by_cluster():
    from wrc.analyze.stance_crosstab import stance_by_discourse

    rows = [
        {"cid": "a1", "stance": "oppose"}, {"cid": "a2", "stance": "support"},
        {"cid": "a3", "stance": "oppose"}, {"cid": "a4", "stance": "oppose"},
        {"cid": "a5", "stance": "neutral"}, {"cid": "a6", "stance": "oppose"},
    ]
    assignments = {"a1": 0, "a2": 0, "a3": 0, "a4": 1, "a5": 1, "a6": 1}
    # 每组只有 3 条，默认 min_n=5 会把表抑制掉；这里降阈值只为验分组逻辑
    table = stance_by_discourse(rows, assignments, min_n=2)
    assert table.groups == ["簇0", "簇1"]
    # 列是中文标签（与报告一致），不是原始英文枚举
    assert table.table["簇0"]["反对"] == 2
    assert table.table["簇1"]["反对"] == 2


def test_stance_by_discourse_marks_small_cluster():
    from wrc.analyze.stance_crosstab import stance_by_discourse

    rows = [{"cid": f"c{index}", "stance": "oppose"} for index in range(6)]
    assignments = {f"c{index}": (0 if index < 4 else 1) for index in range(6)}
    table = stance_by_discourse(rows, assignments)
    # 簇1 只有 2 条 → 整组抑制，但行仍保留用于说明
    assert table.table.get("簇1") is None
    assert "簇1" in table.groups


def test_shared_renderer_accepts_custom_labels():
    from wrc.analyze.stance_crosstab import render

    table = sentiment_by_stance(_base())
    text = render(table, title="自定义标题", group_label="话语簇")
    assert "自定义标题" in text
    assert "话语簇" in text
