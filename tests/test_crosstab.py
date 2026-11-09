"""交叉表与小群体抑制测试（模块 C 的合规要求）。"""

from __future__ import annotations

from wrc.analyze.crosstab import build_crosstab, discourse_by_sentiment


def _rows():
    rows = []
    # 簇 A：10 条，正面 8 / 负面 2（负面 <5 → 单元格抑制）
    for index in range(8):
        rows.append({"cid": f"a{index}", "cluster": "簇A", "sentiment": "positive"})
    for index in range(2):
        rows.append({"cid": f"an{index}", "cluster": "簇A", "sentiment": "negative"})
    # 簇 B：3 条（整组 <5 → 整行抑制）
    for index in range(3):
        rows.append({"cid": f"b{index}", "cluster": "簇B", "sentiment": "positive"})
    return rows


def test_small_cell_is_suppressed():
    result = build_crosstab(
        _rows(),
        group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"],
        attribute_name="sentiment",
    )
    assert result.table["簇A"]["positive"] == 8
    assert result.table["簇A"]["negative"] is None  # 2 < 5
    assert ("簇A", "negative", 2) in result.suppressed_cells


def test_small_group_is_dropped_entirely():
    result = build_crosstab(
        _rows(), group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"],
    )
    assert "簇B" not in result.table
    assert ("簇B", 3) in result.suppressed_groups
    assert "（样本量过小，不予展示）" in result.render()


def test_demographic_attribute_adds_ethics_note():
    rows = [{"cid": str(i), "cluster": "簇A", "province": "北京"} for i in range(6)]
    result = build_crosstab(
        rows, group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["province"], attribute_name="province",
    )
    assert result.attribute_is_demographic is True
    assert "k-匿名" in result.render()


def test_all_groups_tiny_renders_explanation():
    rows = [{"cid": "1", "cluster": "簇X", "sentiment": "negative"}]
    result = build_crosstab(
        rows, group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"],
    )
    text = result.render()
    assert "整表不予展示" in text


def test_discourse_by_sentiment_helper():
    rows = [
        {"cid": "c1", "sentiment": "positive"},
        {"cid": "c2", "sentiment": "negative"},
    ] + [{"cid": f"c{i}", "sentiment": "positive"} for i in range(3, 8)]
    assignments = {f"c{i}": 0 for i in range(1, 8)}
    result = discourse_by_sentiment(rows, assignments)
    assert result.table["簇0"]["positive"] == 6
    assert result.table["簇0"]["negative"] is None


# --------------------------------------------------------------------------- #
# 导出用交叉表：抑制必须在数据层完成
# --------------------------------------------------------------------------- #
def test_csv_marks_suppressed_cells_with_dash():
    result = build_crosstab(
        _rows(), group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"], attribute_name="sentiment",
    )
    _columns, rows = result.to_csv_rows()
    cluster_a = next(row for row in rows if row["分组"] == "簇A")
    assert cluster_a["positive"] == 8
    assert cluster_a["negative"] == "—"
    assert "已抑制" in cluster_a["备注"]


def test_csv_keeps_real_zeros_when_a_category_is_absent():
    """该组里根本没有这个类别 → 是 0，不是被抑制。两者不能混为一谈。"""
    rows = [{"cid": f"p{index}", "sentiment": "positive"} for index in range(8)]
    assignments = {f"p{index}": 0 for index in range(8)}
    result = discourse_by_sentiment(rows, assignments)
    _columns, payload = result.to_csv_rows()
    row = payload[0]
    # 列顺序固定为 positive/neutral/negative；没出现的极性是 0
    assert row["positive"] == 8
    assert row["neutral"] == 0
    assert row["negative"] == 0
    assert row["备注"] == ""


def test_csv_group_total_survives_suppression():
    """整组被抑制时，总数仍要如实给出——不能因为单元格都是 None 就算成 0。"""
    result = build_crosstab(
        _rows(), group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"], attribute_name="sentiment",
    )
    _columns, rows = result.to_csv_rows()
    cluster_b = next(row for row in rows if row["分组"] == "簇B")
    assert cluster_b["该组总样本"] == 3
    assert cluster_b["positive"] == "—"
    assert "整组" in cluster_b["备注"]


def test_csv_group_total_is_exact_even_when_all_cells_suppressed():
    """簇内每格都 <5 但整组 ≥5：总数必须是真实值，不能是 0。"""
    rows = []
    for index in range(3):
        rows.append({"cid": f"n{index}", "cluster": "簇C", "sentiment": "negative"})
    for index in range(2):
        rows.append({"cid": f"u{index}", "cluster": "簇C", "sentiment": "neutral"})
    result = build_crosstab(
        rows, group_of=lambda row: row["cluster"],
        attribute_of=lambda row: row["sentiment"], attribute_name="sentiment",
    )
    _columns, payload = result.to_csv_rows()
    row = payload[0]
    assert row["该组总样本"] == 5
    assert row["negative"] == "—" and row["neutral"] == "—"
