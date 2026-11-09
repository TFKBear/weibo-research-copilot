"""清洗管线与清洗报告测试。

重点验证三件事：
1. 每个步骤都能被单独识别与计数（可解释）；
2. 关闭步骤后结果确实不同（可关闭）；
3. 被移除的样本可以按 cid 恢复（可回退）。
"""

from __future__ import annotations

from wrc.clean.basic import (
    Cleaner,
    is_pure_emoji,
    is_pure_link,
    is_pure_mention,
    normalize_for_dedupe,
)
from wrc.clean.report import build_report

ROWS = [
    {"cid": "1", "post_mid": "p1", "text_raw": "支持渐进式改革", "followers_count": 10},
    {"cid": "2", "post_mid": "p1", "text_raw": "支持渐进式改革", "followers_count": 12},
    {"cid": "3", "post_mid": "p1", "text_raw": "[doge]", "followers_count": 5},
    {"cid": "4", "post_mid": "p1", "text_raw": "网页链接", "followers_count": 5},
    {"cid": "5", "post_mid": "p1", "text_raw": "@张三  ", "followers_count": 5},
    {"cid": "6", "post_mid": "p1", "text_raw": "另一个完全不同的观点", "followers_count": 80},
]


# --------------------------------------------------------------------------- #
# 谓词
# --------------------------------------------------------------------------- #
def test_normalize_for_dedupe_only_collapses_whitespace_and_case():
    assert normalize_for_dedupe("你好  世界 ") == "你好 世界"
    assert normalize_for_dedupe("ABC") == normalize_for_dedupe("abc")
    # 标点必须保留差异，否则会把不同的话当成重复
    assert normalize_for_dedupe("好!") != normalize_for_dedupe("好?")


def test_pure_emoji_detection():
    assert is_pure_emoji("[doge]")
    assert is_pure_emoji("[笑cry][笑cry]")
    assert is_pure_emoji("😂😂")
    assert not is_pure_emoji("支持[doge]")
    assert not is_pure_emoji("")


def test_pure_link_detection():
    assert is_pure_link("https://m.weibo.cn/status/1")
    assert is_pure_link("网页链接")
    assert not is_pure_link("看这里 https://example.com")


def test_pure_mention_detection():
    assert is_pure_mention("@张三")
    assert is_pure_mention("@张三 @李四")
    assert not is_pure_mention("同意@张三")
    assert not is_pure_mention("没有任何提及")


# --------------------------------------------------------------------------- #
# 管线
# --------------------------------------------------------------------------- #
def test_light_tier_removes_one_of_each_category():
    result = Cleaner(tier="light").run(ROWS)
    assert result.total_input == 6
    assert result.total_kept == 2
    assert result.step_removed == {
        "dedupe_exact": 1,
        "drop_empty": 0,
        "drop_pure_emoji": 1,
        "drop_pure_link": 1,
        "drop_pure_mention": 1,
    }
    kept_cids = {row["cid"] for row in result.kept_rows}
    assert kept_cids == {"1", "6"}


def test_none_tier_keeps_everything():
    result = Cleaner(tier="none").run(ROWS)
    assert result.total_kept == result.total_input == 6
    assert result.active_steps == ()


def test_steps_can_be_disabled_individually():
    """产品宪法第 2 条：每一步都必须可以被用户关掉。"""
    result = Cleaner(tier="light", enabled_steps=("dedupe_exact",)).run(ROWS)
    assert result.total_kept == 5
    assert result.step_removed == {"dedupe_exact": 1}


def test_unknown_step_is_rejected_loudly():
    try:
        Cleaner(tier="light", enabled_steps=("不存在的步骤",)).run(ROWS)
    except ValueError as exc:
        assert "未知清洗步骤" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 ValueError")


def test_dedupe_scope_global_merges_across_posts():
    rows = [
        {"cid": "a", "post_mid": "p1", "text_raw": "同一句话"},
        {"cid": "b", "post_mid": "p2", "text_raw": "同一句话"},
    ]
    per_post = Cleaner(tier="light", dedupe_scope="post").run(rows)
    across = Cleaner(tier="light", dedupe_scope="global").run(rows)
    assert per_post.total_kept == 2
    assert across.total_kept == 1


def test_restore_undoes_a_false_positive():
    result = Cleaner(tier="light").run(ROWS)
    assert result.restore("3") is True
    assert result.total_kept == 3
    assert result.restore("3") is False  # 幂等
    assert result.restore("不存在的 cid") is False


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def test_report_counts_samples_and_distributions():
    result = Cleaner(tier="light").run(ROWS)
    report = build_report(result, ROWS, sample_size=20, seed=20240101)
    assert report.total_input == 6
    assert report.total_kept == 2
    assert report.total_removed == 4
    assert len(report.samples) == 4
    assert {sample.cid for sample in report.samples} == {"2", "3", "4", "5"}
    assert all(sample.reasons for sample in report.samples)
    assert report.length_before is not None
    assert report.length_after is not None
    assert report.followers_before is not None
    text = report.render()
    assert "清洗影响报告" in text
    assert "精确去重" in text
    assert "误杀" in text


def test_report_sampling_is_reproducible_with_same_seed():
    result = Cleaner(tier="light").run(ROWS)
    first = build_report(result, ROWS, seed=42)
    second = build_report(result, ROWS, seed=42)
    assert [s.cid for s in first.samples] == [s.cid for s in second.samples]


def test_report_serializes_to_dict():
    result = Cleaner(tier="light").run(ROWS)
    payload = build_report(result, ROWS).to_dict()
    assert payload["total_kept"] == 2
    assert isinstance(payload["steps"], list)
    assert payload["steps"][0]["key"] == "dedupe_exact"
