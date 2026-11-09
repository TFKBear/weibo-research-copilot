"""微博接口地址与参数构造。

⚠ 重要说明（Agent 执行规则第 1 条：先验证再开发）
微博的接口参数会变化，本文件里的地址是「候选形态」，
真实可用性由 ``spikes/03_probe.py`` 在用户网络环境下实测确定。
任何未经验证的排序/参数假设，都在注释里显式标注为「待验证」，
不允许在文档或 UI 里把它当成已确认事实讲给研究者。
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from ..errors import ConfigProblem

MOBILE_BASE = "https://m.weibo.cn"
DESKTOP_BASE = "https://s.weibo.com"

#: 登录态探针。返回 ``{"data": {"login": true, ...}}`` 时视为已登录。
CONFIG_PATH = "/api/config"

#: tag 检索（移动端）。containerid 的 ``type=1`` 是检索页的默认（综合）Tab。
#: 「热门 / 实时」在移动端接口中的对应取值 —— 待验证，不在 P0 依赖。
SEARCH_PATH = "/api/container/getIndex"

#: 一级评论。hotflow 是「热门评论流」，也承担主楼评论的翻页。
COMMENTS_PATH = "/api/comments/hotflow"

#: 二级评论（楼中楼）。默认关闭采集。
SUBCOMMENTS_PATH = "/api/comments/hotFlowChild"


def normalize_tag(raw: str) -> str:
    """把用户输入规范化成微博话题写法 ``#话题#``。

    >>> normalize_tag("延迟退休")
    '#延迟退休#'
    >>> normalize_tag("  ##延迟退休##  ")
    '#延迟退休#'
    """
    text = (raw or "").strip()
    text = text.strip("#").strip()
    # 用户可能连输入多个 #，逐层剥掉
    while text.startswith("#") and text.endswith("#") and len(text) > 2:
        text = text.strip("#").strip()
    if not text:
        raise ConfigProblem(
            detail=f"tag={raw!r}",
            what_happened="话题标签为空，请先输入要研究的话题，例如“延迟退休”。",
        )
    return f"#{text}#"


def bare_tag(tag: str) -> str:
    """去掉井号，用于文件名与目录名。"""
    return normalize_tag(tag).strip("#")


#: 两种检索口径。它们拿到的不是同一批帖子，必须在报告里区分：
#: - ``topic``：``q=#话题#``，走话题聚合，只有被用户打上该话题的帖子；
#: - ``keyword``：``q=关键词``，走全文检索，包含"正文里提到这个词"的帖子。
SEARCH_MODES = {
    "topic": "话题聚合（q=#话题#）",
    "keyword": "关键词全文检索（q=关键词）",
}


def search_query(raw: str, mode: str = "topic") -> str:
    """按检索口径生成 ``q`` 参数。

    这个区分不是文字游戏：话题聚合的样本是"主动参与该话题的人"，
    关键词检索的样本包含"顺口提到一次"的人。两者混为一谈，
    研究者会以为自己在研究话题参与者，实际混进了路人提及。
    """
    if mode == "keyword":
        text = (raw or "").strip().strip("#").strip()
        while text.startswith("#") and text.endswith("#") and len(text) > 2:
            text = text.strip("#").strip()
        if not text:
            raise ConfigProblem(
                detail=f"tag={raw!r}",
                what_happened="检索词为空，请先输入要研究的关键词，例如“延迟退休”。",
            )
        return text
    return normalize_tag(raw)


def search_container_id(tag: str, *, tab_type: int = 1, mode: str = "topic") -> str:
    """构造检索用 containerid。

    形态：``100103type=1&q=#话题#``（话题）或 ``100103type=1&q=关键词``（全文）。
    """
    return f"100103type={tab_type}&q={search_query(tag, mode)}"


@dataclass(frozen=True, slots=True)
class Request:
    """一次待发出的请求。

    ``expect`` 声明期望的响应形态：
    - ``json``：移动端接口，返回 JSON；
    - ``html``：桌面检索页，返回 HTML（探测脚本用它验证降级路径）。

    ``referer`` 允许按请求覆盖 Referer。评论接口对此敏感：
    从检索页直接打评论接口，与"先打开帖子详情页再拉评论"在风控看来是两种行为。
    """

    url: str
    params: dict[str, str | int]
    label: str
    expect: str = "json"
    referer: str | None = None

    def describe(self) -> str:
        return f"{self.label} → {self.url}"


def config_request() -> Request:
    return Request(url=f"{MOBILE_BASE}{CONFIG_PATH}", params={}, label="登录态检查")


def search_request(
    tag: str, *, page: int = 1, tab_type: int = 1, mode: str = "topic"
) -> Request:
    """tag 检索请求。

    Args:
        tag: 话题标签或关键词。
        page: 页码，从 1 开始。
        tab_type: 检索 Tab 编号；1 为默认综合。其他取值（热门/实时）待验证。
        mode: ``topic`` 话题聚合 / ``keyword`` 关键词全文检索。
    """
    query = search_query(tag, mode)
    return Request(
        url=f"{MOBILE_BASE}{SEARCH_PATH}",
        params={
            "containerid": search_container_id(tag, tab_type=tab_type, mode=mode),
            "page_type": "searchall",
            "page": max(1, page),
        },
        label=f"检索 {query} 第 {page} 页" + ("（全文）" if mode == "keyword" else ""),
    )


def comments_request(
    post_mid: str,
    *,
    max_id: int = 0,
    max_id_type: int = 0,
    referer: str | None = None,
) -> Request:
    """一级评论请求（移动端 hotflow）。

    翻页靠 ``max_id`` 递推；``max_id_type`` 必须与上一次响应的值一致，
    否则会拿到重复或空页 —— 这是该接口最常见的踩坑点。

    Referer 默认指向该帖的详情页，而不是站点首页：
    实测从首页 Referer 直接请求评论接口会拿到 ``ok=1`` 但 ``data`` 为空，
    与"拉取详情页评论"是两种不同的调用形态。
    """
    if not post_mid:
        raise ConfigProblem(what_happened="帖子 mid 为空，无法请求评论。")
    return Request(
        url=f"{MOBILE_BASE}{COMMENTS_PATH}",
        params={
            "id": post_mid,
            "mid": post_mid,
            "max_id": max_id,
            "max_id_type": max_id_type,
        },
        label=f"评论 mid={post_mid} max_id={max_id}",
        referer=referer or f"{MOBILE_BASE}/detail/{post_mid}",
    )


def subcomments_request(cid: str, *, max_id: int = 0, max_id_type: int = 0) -> Request:
    """二级评论请求。无跟评时该接口可能返回结构异常的数据，解析层必须容错。"""
    if not cid:
        raise ConfigProblem(what_happened="评论 cid 为空，无法请求二级评论。")
    return Request(
        url=f"{MOBILE_BASE}{SUBCOMMENTS_PATH}",
        params={"cid": cid, "max_id": max_id, "max_id_type": max_id_type},
        label=f"二级评论 cid={cid} max_id={max_id}",
    )


def desktop_search_url(tag: str, *, page: int = 1, refer: str = "g") -> str:
    """桌面检索页（HTML）。

    ⚠ 实测结论（第 1 步，未登录基线）：该站点在当前环境下会重定向到
    ``passport.weibo.com/visitor/...`` 访客关口，或直接超时。
    因此它**不再是降级采集通道**，只在 ``spikes/03_probe.py`` 里用于探测，
    主链路必须走 ``m.weibo.cn`` 接口 + 登录态。

    ``Refer=hot`` 是否仍对应「热门」排序，同样交由探测脚本判断。
    """
    q = quote(normalize_tag(tag), safe="")
    return f"{DESKTOP_BASE}/weibo?q={q}&page={max(1, page)}&Refer={refer}"


def desktop_search_request(tag: str, *, page: int = 1, refer: str = "g") -> Request:
    """桌面检索页请求（期望 HTML 响应，仅用于探测）。"""
    return Request(
        url=desktop_search_url(tag, page=page, refer=refer),
        params={},
        label=f"桌面检索 {normalize_tag(tag)} Refer={refer} 第 {page} 页",
        expect="html",
    )
