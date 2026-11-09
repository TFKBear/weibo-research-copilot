"""术语随时间变化测试。

价值在于区分「态度变了」和「关注点变了」——这两件事常被混为一谈。
风险同样是误读：末尾采集日只有一两条例时，任何词都会显示"暴涨/归零"，
所以必须只拿够样本量的段来比较首末。
"""

from __future__ import annotations

import pytest

from wrc.analyze.terms import (
    MIN_SIDE_N,
    check_terms,
    term_trends,
)


def _rows(day: str, texts: list[str]) -> list[dict]:
    return [
        {
            "cid": f"{day}-{index}",
            "created_at": f"{day}T10:00:00+08:00",
            "created_at_precision": "second",
            "cleaned_text": text,
        }
        for index, text in enumerate(texts)
    ]


def test_rising_term_is_detected():
    rows = (
        _rows("2025-01-01", ["社保问题"] * 2 + ["退休年龄"] * 18)
        + _rows("2025-01-10", ["社保问题"] * 18 + ["退休年龄"] * 2)
    )
    result = term_trends(rows, period_days=7)
    terms = [term for term, _first, _last in result.rising]
    assert "社保" in terms or "社保问题" in terms


def test_short_tail_period_is_excluded_from_comparison():
    """末尾只有 1 条时不能当"末段"——否则任何词都会显示暴跌。"""
    rows = (
        _rows("2025-01-01", ["社保"] * 30)
        + _rows("2025-01-08", ["社保"] * 30)
        + _rows("2025-02-20", ["别的"])  # 孤零零一条，落在很后面的段
    )
    result = term_trends(rows, period_days=7)
    # 末段样本量不足 MIN_SIDE_N，不参与排名
    assert result.totals[-1] < MIN_SIDE_N
    assert result.decidable is True  # 前面有两段够量
    assert all(term for term, _a, _b in result.rising + result.falling)


def test_too_few_valid_periods_refuses_ranking():
    rows = _rows("2025-01-01", ["社保"] * 5)
    result = term_trends(rows, period_days=7)
    assert result.decidable is False
    assert result.rising == [] and result.falling == []
    assert "不足" in result.render()


def test_render_states_no_significance_test():
    rows = (
        _rows("2025-01-01", ["社保"] * 30)
        + _rows("2025-01-08", ["社保"] * 30)
    )
    text = term_trends(rows, period_days=7).render()
    assert "未做显著性检验" in text
    assert "不是词频" in text


def test_check_terms_reports_exact_terms():
    rows = (
        _rows("2025-01-01", ["弹性退休"] * 10 + ["社保"] * 10)
        + _rows("2025-01-08", ["弹性退休"] * 2 + ["社保"] * 18)
    )
    text = check_terms(rows, ["弹性", "社保"], period_days=7)
    assert "弹性" in text and "社保" in text
    assert "样本" in text


def test_period_labels_are_short_dates():
    rows = (
        _rows("2025-01-01", ["社保"] * 25)
        + _rows("2025-01-08", ["社保"] * 25)
    )
    result = term_trends(rows, period_days=7)
    assert result.periods[0] == "01-01"
    assert all(len(label) == 5 for label in result.periods)


def test_to_dict_serializable():
    rows = (
        _rows("2025-01-01", ["社保"] * 25)
        + _rows("2025-01-08", ["社保"] * 25)
    )
    payload = term_trends(rows, period_days=7).to_dict()
    assert payload["module"] == "terms"
    assert payload["period_days"] == 7
    assert isinstance(payload["totals"], list)


def test_no_dated_rows_returns_warning():
    result = term_trends([{"cid": "c1", "created_at": None, "cleaned_text": "社保"}])
    assert result.periods == []
    assert result.warnings


def test_pytest_placeholder_math():
    # 防止有人把 pytest 从顶部 import 删掉（其他测试用近似比较时需要它）
    assert MIN_SIDE_N > 0
    assert pytest.approx(1.0) == 1.0
