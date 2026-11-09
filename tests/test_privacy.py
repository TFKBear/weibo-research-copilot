"""匿名化与 k-匿名抑制测试（模块 C）。

这一层出错的后果最严重：不是功能不好用，而是研究数据不能合法使用。
所以测试要覆盖「稳定」「不可反查」「确实删干净」三个方面。
"""

from __future__ import annotations

import pytest

from wrc.privacy.anonymize import (
    IDENTIFIER_FIELDS,
    PseudoIdHasher,
    anonymize_comment,
    anonymize_post,
    assert_no_identifiers,
    extract_province,
    suppress_small_groups,
)


# --------------------------------------------------------------------------- #
# 伪匿名
# --------------------------------------------------------------------------- #
def test_pseudo_id_is_stable_across_instances(tmp_path):
    salt = tmp_path / "keys" / "salt.bin"
    first = PseudoIdHasher(salt).derive("1234567")
    second = PseudoIdHasher(salt).derive("1234567")
    assert first == second
    assert len(first) == 16


def test_pseudo_id_differs_by_input(tmp_path):
    hasher = PseudoIdHasher(tmp_path / "salt.bin")
    assert hasher.derive("a") != hasher.derive("b")
    assert hasher.derive("") == ""
    assert hasher.derive(None) == ""


def test_pseudo_id_does_not_leak_raw_value(tmp_path):
    hasher = PseudoIdHasher(tmp_path / "salt.bin")
    assert "1234567" not in hasher.derive("1234567")


def test_two_salts_produce_different_ids(tmp_path):
    one = PseudoIdHasher(tmp_path / "a.bin").derive("42")
    two = PseudoIdHasher(tmp_path / "b.bin").derive("42")
    assert one != two


# --------------------------------------------------------------------------- #
# 地区聚合
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("北京 朝阳区", "北京"),
        ("广东 深圳", "广东"),
        ("上海", "上海"),
        ("内蒙古 呼和浩特", "内蒙古"),
        ("海外 美国", "海外"),
        ("", "未知"),
        (None, "未知"),
        ("某个无法识别的地址", "未知"),
    ],
)
def test_extract_province(raw, expected):
    assert extract_province(raw) == expected


# --------------------------------------------------------------------------- #
# 字段移除
# --------------------------------------------------------------------------- #
RAW_ROW = {
    "cid": "c1",
    "post_mid": "p1",
    "parent_cid": None,
    "level": 1,
    "user_id": "998877",
    "screen_name": "张三",
    "profile_url": "https://weibo.com/u/998877",
    "location": "北京 朝阳区",
    "text_raw": "同意",
    "created_at": "2025-06-01T10:00:00+08:00",
    "like_count": 3,
    "followers_count": 120,
}


def test_anonymize_comment_drops_all_identifiers(tmp_path):
    hasher = PseudoIdHasher(tmp_path / "salt.bin")
    out = anonymize_comment(RAW_ROW, hasher)
    for field in IDENTIFIER_FIELDS:
        assert field not in out
    assert out["user_pseudo_id"] == hasher.derive("998877")
    assert out["province"] == "北京"
    assert out["text"] == "同意"


def test_anonymize_comment_whitelist_blocks_unknown_columns(tmp_path):
    """白名单机制：底层表以后加了新字段，也不会意外泄露。"""
    hasher = PseudoIdHasher(tmp_path / "salt.bin")
    out = anonymize_comment({**RAW_ROW, "ip_region": "北京", "device": "iPhone"}, hasher)
    assert "ip_region" not in out
    assert "device" not in out


def test_anonymize_post_drops_identity(tmp_path):
    hasher = PseudoIdHasher(tmp_path / "salt.bin")
    out = anonymize_post(
        {"mid": "m1", "user_id": "1", "screen_name": "甲", "text_raw": "x"}, hasher
    )
    assert "user_id" not in out and "screen_name" not in out
    assert out["mid"] == "m1"


def test_assert_no_identifiers_blocks_leak():
    with pytest.raises(AssertionError) as excinfo:
        assert_no_identifiers([{"cid": "1", "user_id": "42"}])
    assert "user_id" in str(excinfo.value)


def test_assert_no_identifiers_passes_clean_rows():
    assert_no_identifiers([{"cid": "1", "user_pseudo_id": "abc"}])


# --------------------------------------------------------------------------- #
# k-匿名抑制
# --------------------------------------------------------------------------- #
def test_suppress_small_groups_hides_small_cells_and_groups():
    result = suppress_small_groups(
        {
            "话语簇1": {"北京": 12, "上海": 2},
            "话语簇2": {"北京": 1, "上海": 1},
        },
        min_n=5,
    )
    assert result.table["话语簇1"]["北京"] == 12
    assert result.table["话语簇1"]["上海"] is None
    assert "话语簇2" not in result.table
    assert len(result.suppressed_groups) == 1
    assert result.notes


def test_suppress_small_groups_is_noop_when_all_large():
    result = suppress_small_groups({"g": {"北京": 8, "上海": 9}}, min_n=5)
    assert result.table == {"g": {"北京": 8, "上海": 9}}
    assert result.suppressed_cells == []
    assert result.notes == []
