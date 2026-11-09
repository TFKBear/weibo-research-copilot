"""接口地址构造测试。

这些断言是「我们承诺用什么方式访问微博」的书面记录，
一旦有人改了参数就应该有测试红灯。
"""

from __future__ import annotations

import pytest

from wrc.errors import ConfigProblem
from wrc.weibo.endpoints import (
    comments_request,
    desktop_search_url,
    normalize_tag,
    search_container_id,
    search_request,
    subcomments_request,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("延迟退休", "#延迟退休#"),
        ("#延迟退休#", "#延迟退休#"),
        ("  ##延迟退休##  ", "#延迟退休#"),
    ],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "#", None])
def test_normalize_tag_rejects_empty(raw):
    with pytest.raises(ConfigProblem):
        normalize_tag(raw)


def test_search_container_id_shape():
    assert search_container_id("延迟退休") == "100103type=1&q=#延迟退休#"
    assert search_container_id("延迟退休", tab_type=2) == "100103type=2&q=#延迟退休#"


def test_search_request_params():
    request = search_request("延迟退休", page=3)
    assert request.url.endswith("/api/container/getIndex")
    assert request.params["page"] == 3
    assert request.params["page_type"] == "searchall"
    assert request.expect == "json"


def test_search_request_page_is_at_least_one():
    assert search_request("话题", page=0).params["page"] == 1


def test_comments_request_carries_cursor():
    request = comments_request("5000", max_id=123, max_id_type=2)
    assert request.params == {"id": "5000", "mid": "5000", "max_id": 123, "max_id_type": 2}
    assert request.url.endswith("/api/comments/hotflow")


def test_comments_request_referer_points_at_post_detail():
    """Referer 必须是该帖详情页。

    实测从站点首页作为 Referer 请求评论接口，会拿到 ok=1 但 data 为空；
    「像浏览器一样先打开帖子再看评论」是这个接口的隐含前提。
    """
    request = comments_request("5000")
    assert request.referer == "https://m.weibo.cn/detail/5000"


def test_comments_request_requires_mid():
    with pytest.raises(ConfigProblem):
        comments_request("")


def test_subcomments_request_requires_cid():
    with pytest.raises(ConfigProblem):
        subcomments_request("")


def test_desktop_search_url_encodes_tag():
    url = desktop_search_url("延迟退休", page=2, refer="hot")
    assert url.startswith("https://s.weibo.com/weibo?q=")
    assert "page=2" in url
    assert "Refer=hot" in url
    assert "%23" in url  # 井号必须被编码
