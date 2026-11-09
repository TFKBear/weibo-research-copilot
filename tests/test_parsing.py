"""解析层测试。

这一层是最先被上游接口变化打坏的部分，所以覆盖要密：
每个容错分支都对应一次真实的踩坑。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from wrc.errors import ApiError, AuthRequired, ParseFailure
from wrc.weibo.models import CommentLevel, TimePrecision
from wrc.weibo.parsing import (
    CST,
    check_api_ok,
    extract_desktop_render_data,
    is_logged_in,
    parse_comments_page,
    parse_created_at,
    parse_desktop_search_page,
    parse_search_page,
    post_from_mblog,
    strip_html,
)

NOW = datetime(2025, 6, 1, 12, 0, tzinfo=CST)


# --------------------------------------------------------------------------- #
# 文本
# --------------------------------------------------------------------------- #
def test_strip_html_keeps_emoji_tokens():
    raw = '早啊 <a href="/n/x">#延迟退休#</a><img alt="[doge]" src="x.gif">'
    assert strip_html(raw) == "早啊 #延迟退休#[doge]"


def test_strip_html_handles_alt_after_src_and_entities():
    raw = 'A &amp; B<img src="x.gif" alt="[笑cry]">'
    assert strip_html(raw) == "A & B[笑cry]"


def test_strip_html_removes_zero_width_and_blank_lines():
    raw = "第一行\u200b\r\n\r\n\r\n第二行"
    assert strip_html(raw) == "第一行\n\n第二行"


def test_strip_html_none():
    assert strip_html(None) == ""


# --------------------------------------------------------------------------- #
# 时间
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected", "precision"),
    [
        ("刚刚", NOW, TimePrecision.MINUTE),
        ("30秒前", NOW - timedelta(seconds=30), TimePrecision.SECOND),
        ("3分钟前", NOW - timedelta(minutes=3), TimePrecision.MINUTE),
        ("今天 10:23", datetime(2025, 6, 1, 10, 23, tzinfo=CST), TimePrecision.MINUTE),
        ("昨天 21:05", datetime(2025, 5, 31, 21, 5, tzinfo=CST), TimePrecision.MINUTE),
        ("2025-05-12 08:30", datetime(2025, 5, 12, 8, 30, tzinfo=CST), TimePrecision.MINUTE),
        ("2025-05-12", datetime(2025, 5, 12, 0, 0, tzinfo=CST), TimePrecision.DAY),
        ("05-12", datetime(2025, 5, 12, 0, 0, tzinfo=CST), TimePrecision.DAY),
        ("12月31日", datetime(2024, 12, 31, 0, 0, tzinfo=CST), TimePrecision.DAY),
        (
            "Mon May 12 10:00:00 +0800 2025",
            datetime(2025, 5, 12, 10, 0, tzinfo=CST),
            TimePrecision.SECOND,
        ),
    ],
)
def test_parse_created_at(raw, expected, precision):
    value, got_precision = parse_created_at(raw, now=NOW)
    assert got_precision is precision
    assert value == expected


def test_parse_created_at_reports_day_precision_instead_of_faking_midnight_context():
    """只给到「日」的时间必须标记为 DAY，不能假装精确到秒。

    这是时间序列分析里最容易被忽略的系统性误差来源。
    """
    value, precision = parse_created_at("05-12", now=NOW)
    assert precision is TimePrecision.DAY
    assert value.hour == 0 and value.minute == 0


@pytest.mark.parametrize("raw", ["", None, "某个我认不出的格式", "昨天"])
def test_parse_created_at_unknown(raw):
    value, precision = parse_created_at(raw, now=NOW)
    assert value is None
    assert precision is TimePrecision.UNKNOWN


# --------------------------------------------------------------------------- #
# 接口外层
# --------------------------------------------------------------------------- #
def test_check_api_ok_passes():
    payload = {"ok": 1, "data": {}}
    assert check_api_ok(payload, label="t") is payload


def test_check_api_ok_login_required():
    with pytest.raises(AuthRequired):
        check_api_ok({"ok": -100, "msg": "登录"}, label="t")


def test_check_api_ok_business_error():
    with pytest.raises(ApiError) as excinfo:
        check_api_ok({"ok": 0, "msg": "参数错误"}, label="t")
    assert "参数错误" in excinfo.value.what_happened


def test_check_api_ok_bad_type():
    with pytest.raises(ParseFailure):
        check_api_ok("not a dict", label="t")


def test_is_logged_in():
    assert is_logged_in({"ok": 1, "data": {"login": True}})
    assert not is_logged_in({"ok": 1, "data": {"login": False}})
    assert not is_logged_in({"data": None})


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def _mblog(mid: str, text: str = "内容") -> dict:
    return {
        "mid": mid,
        "text": text,
        "created_at": "今天 10:23",
        "user": {"id": f"u{mid}", "screen_name": f"用户{mid}", "followers_count": 12},
        "reposts_count": "1,024",
        "comments_count": 3,
        "attitudes_count": None,
    }


def test_iter_mblogs_covers_both_card_shapes_and_dedupes():
    payload = {
        "ok": 1,
        "data": {
            "cards": [
                {"card_type": 9, "mblog": _mblog("1")},
                {"card_type": 11, "title": "用户卡片"},
                {"card_type": 9, "card_group": [{"card_type": 9, "mblog": _mblog("2")}]},
                {"card_type": 9, "mblog": _mblog("1")},  # 重复 mid
            ]
        },
    }
    page = parse_search_page(payload, page=1, now=NOW)
    assert [post.mid for post in page.posts] == ["1", "2"]


def test_post_from_mblog_parses_counts_and_precision():
    post = post_from_mblog(_mblog("42", "正文<img alt='[good]'>"), now=NOW)
    assert post.mid == "42"
    assert post.attitudes_count == 0
    assert post.reposts_count == 1024
    assert post.created_at_precision is TimePrecision.MINUTE
    assert post.engagement == 1024 + 3 + 0


def test_post_from_mblog_without_mid_raises():
    with pytest.raises(ParseFailure):
        post_from_mblog({"text": "无 id"}, now=NOW)


def test_parse_search_page_empty_cards():
    page = parse_search_page({"ok": 1, "data": {"cards": []}}, page=1)
    assert page.posts == []
    assert page.has_more is False


# --------------------------------------------------------------------------- #
# 评论
# --------------------------------------------------------------------------- #
def test_parse_comments_page_basic():
    payload = {
        "ok": 1,
        "data": {
            "data": [
                {
                    "id": "c1",
                    "text": "同意",
                    "created_at": "今天 09:00",
                    "like_counts": 5,
                    "total_number": 2,
                    "user": {"id": "u1", "screen_name": "甲", "followers_count": 88},
                }
            ],
            "max_id": 12345,
            "max_id_type": 0,
            "total_number": 99,
        }
    }
    page = parse_comments_page(payload, post_mid="p1", now=NOW)
    assert len(page.comments) == 1
    comment = page.comments[0]
    assert comment.cid == "c1"
    assert comment.like_count == 5
    assert comment.reply_count == 2
    assert comment.followers_count == 88
    assert comment.level is CommentLevel.PRIMARY
    assert page.next_max_id == 12345
    assert page.total_number == 99
    assert page.has_more is True


def test_parse_comments_page_tolerates_null_data():
    """无跟评时二级评论接口可能返回 data=None，必须返回空页而不是崩。"""
    page = parse_comments_page({"ok": 1, "data": None}, post_mid="p1")
    assert page.comments == []
    assert page.has_more is False


def test_parse_comments_page_tolerates_missing_fields():
    payload = {"ok": 1, "data": {"data": [{"id": "c9", "text": "x"}]}}
    page = parse_comments_page(payload, post_mid="p1")
    assert len(page.comments) == 1
    assert page.comments[0].like_count == 0


def test_parse_comments_page_stops_when_max_id_zero():
    payload = {"ok": 1, "data": {"data": [], "max_id": 0, "max_id_type": 0}}
    page = parse_comments_page(payload, post_mid="p1")
    assert page.has_more is False


def test_parse_comments_page_supports_web_backend_shape():
    """网页端 buildComments 把评论数组直接放在 data，游标放在顶层。

    只认 data.data 的解析器会把一个正常响应解析成 0 条 —— 这正是第 1 步
    实机测试踩到的坑，所以两种形态都必须被覆盖。
    """
    payload = {
        "ok": 1,
        "data": [
            {
                "id": "w1",
                "text": "网页端评论",
                "created_at": "2025-05-12 10:00:00",
                "likes_count": 7,
                "total_number": 1,
                "user": {"id": "u9", "followers_count": 33},
            }
        ],
        "max_id": 4321,
        "max_id_type": 0,
        "total_number": 88,
    }
    page = parse_comments_page(payload, post_mid="p1", now=NOW)
    assert len(page.comments) == 1
    assert page.comments[0].cid == "w1"
    assert page.comments[0].like_count == 7
    assert page.next_max_id == 4321
    assert page.total_number == 88


def test_parse_comments_page_handles_empty_data_list():
    assert parse_comments_page({"ok": 1, "data": []}, post_mid="p1").comments == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2733.7万", 27_337_000),
        ("1.2亿", 120_000_000),
        ("1,024", 1024),
        ("512", 512),
        ("", 0),
        (None, 0),
        (0, 0),
        (88, 88),
        ("1.5w", 15_000),
    ],
)
def test_to_int_parses_chinese_units(raw, expected):
    """粉丝数带中文单位时必须乘回去。

    实测 ``user.followers_count == "2733.7万"``；按"去非数字"处理会得到 27337，
    比真实值小 1000 倍，直接让 B2 的粉丝数分布对比失去意义。
    """
    from wrc.weibo.parsing import to_count

    assert to_count(raw) == expected


def test_parse_comments_page_reads_real_mobile_shape():
    """真实移动端 hotflow 响应形态（第 2 轮实机抓取）：

    ``{"ok":1, "data":[...], "max":..., "max_id":..., "total_number":679}``
    —— ``data`` 是数组而不是对象，游标与总数在顶层。
    """
    payload = {
        "ok": 1,
        "data": [
            {
                "id": "5308980077529786",
                "text": '好车 <span class="url-icon"><img alt="[作揖]" src="x.gif"></span>',
                "created_at": "Fri Jun 12 13:42:29 +0800 2026",
                "like_count": 1547,
                "total_number": 93,
                "floor_number": 4,
                "user": {"id": 1749127163, "screen_name": "雷军",
                         "followers_count": "2733.7万"},
            }
        ],
        "max": 200,
        "max_id": 13886361396130283,
        "total_number": 679,
    }
    page = parse_comments_page(payload, post_mid="m1")
    assert len(page.comments) == 1
    comment = page.comments[0]
    assert comment.cid == "5308980077529786"
    assert comment.like_count == 1547
    assert comment.reply_count == 93
    assert comment.followers_count == 27_337_000
    assert comment.text_raw == "好车 [作揖]"
    assert comment.created_at_precision is TimePrecision.SECOND
    assert page.next_max_id == 13886361396130283
    assert page.total_number == 679


# --------------------------------------------------------------------------- #
# 桌面页降级
# --------------------------------------------------------------------------- #
def test_extract_desktop_render_data_handles_nested_arrays():
    html = (
        'var $render_data = [{"statuses":[{"mid":"9001","text":"内容",'
        '"created_at":"05-12","user":{"id":"u1"}}],"ok":1}] ;'
    )
    render = extract_desktop_render_data(html)
    assert render is not None
    assert render[0]["statuses"][0]["mid"] == "9001"


def test_parse_desktop_search_page_builds_posts():
    html = (
        '$render_data = [{"cards":[{"card_type":9,"mblog":{"mid":"7","text":"hi",'
        '"created_at":"05-12","user":{"id":"u7"},"attitudes_count":9}}]}]'
    )
    page = parse_desktop_search_page(html, page=1, now=NOW)
    assert [post.mid for post in page.posts] == ["7"]
    assert page.posts[0].attitudes_count == 9


def test_parse_desktop_search_page_raises_when_no_render_data():
    with pytest.raises(ParseFailure):
        parse_desktop_search_page("<html>请先登录</html>")
