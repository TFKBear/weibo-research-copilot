"""响应解析层（纯函数）。

这一层刻意不依赖网络、浏览器或磁盘，因为：
- 微博接口会变，解析规则是最先失效的部分，必须能被独立的 fixture 测试覆盖；
- 产品宪法第 1 条要求可追溯，解析规则本身就是「生成数据的方法」的一部分，
  需要能在文档里被引用和复核。

约定：任何无法解析但结构合法的字段一律「降级」而不是抛错，
并把降级事实记录到返回对象上（如 ``TimePrecision.UNKNOWN``）。
只有整体结构不合法（拿不到 mblog 列表）才抛 ``ParseFailure``。
"""

from __future__ import annotations

import html as html_lib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from ..errors import ApiError, AuthRequired, ParseFailure
from .models import (
    CommentLevel,
    CommentPage,
    SearchPage,
    TimePrecision,
    WeiboComment,
    WeiboPost,
)

# 中国标准时间。微博时间戳均为北京时间；1991 年后中国不再使用夏令时，
# 因此固定 +08:00 与 IANA 时区库在本场景等价。
try:  # pragma: no cover - 取决于环境是否安装 tzdata
    from zoneinfo import ZoneInfo

    CST: timezone | ZoneInfo = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    CST = timezone(timedelta(hours=8), "CST")


# --------------------------------------------------------------------------- #
# 文本处理
# --------------------------------------------------------------------------- #
_IMG_ALT_RE = re.compile(r"<img\b[^>]*?\balt=\"([^\"]*)\"[^>]*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def strip_html(raw: str | None) -> str:
    """把微博正文 HTML 转成纯文本，并**保留表情符号**。

    微博的表情以 ``<img alt="[笑cry]">`` 的形式出现在正文里。
    对中文微博情感研究而言，表情是独立信号（模块 B3 的要求），
    所以这里把 alt 文本原样保留成 token，而不是当作标签剥掉。

    >>> strip_html('早啊 <a href="/n/x">#延迟退休#</a><img alt="[doge]" src="x.gif">')
    '早啊 #延迟退休#[doge]'
    """
    if not raw:
        return ""
    text = _IMG_ALT_RE.sub(lambda m: m.group(1), raw)
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html_lib.unescape(text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 时间解析
# --------------------------------------------------------------------------- #
_CTIME_RE = re.compile(
    r"^[A-Z][a-z]{2} ([A-Z][a-z]{2}) (\d{1,2}) (\d{2}:\d{2}:\d{2}) ([+-]\d{4}) (\d{4})$"
)
_SECONDS_AGO_RE = re.compile(r"^(\d+)\s*秒前$")
_MINUTES_AGO_RE = re.compile(r"^(\d+)\s*分钟前$")
_TODAY_RE = re.compile(r"^今天\s*(\d{1,2}):(\d{2})$")
_YESTERDAY_RE = re.compile(r"^昨天\s*(\d{1,2}):(\d{2})$")
_FULL_DATE_TIME_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::\d{2})?$"
)
_FULL_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_MONTH_DAY_RE = re.compile(r"^(\d{1,2})-(\d{1,2})$")
_CN_MONTH_DAY_RE = re.compile(r"^(\d{1,2})月(\d{1,2})日$")

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_created_at(
    raw: str | None, *, now: datetime | None = None
) -> tuple[datetime | None, TimePrecision]:
    """解析微博的相对/绝对时间，并如实报告精度。

    微博检索结果里同一个字段会混用「刚刚 / 3分钟前 / 今天 10:23 / 05-12 / 2024-05-12」。
    对研究者来说，把「只精确到日」的时间补成 00:00:00 是一种隐性造假，
    因此这里返回 ``TimePrecision`` 供后续分析区分。

    Returns:
        (时间对象, 精度)。无法识别时返回 ``(None, TimePrecision.UNKNOWN)``。
    """
    now = (now or datetime.now(CST)).astimezone(CST)
    text = (raw or "").strip()
    if not text:
        return None, TimePrecision.UNKNOWN

    if text in {"刚刚", "刚刚发布", "现在"}:
        return now, TimePrecision.MINUTE

    m = _SECONDS_AGO_RE.match(text)
    if m:
        return now - timedelta(seconds=int(m.group(1))), TimePrecision.SECOND

    m = _MINUTES_AGO_RE.match(text)
    if m:
        return now - timedelta(minutes=int(m.group(1))), TimePrecision.MINUTE

    m = _TODAY_RE.match(text)
    if m:
        return _at(now, now.year, now.month, now.day,
                   int(m.group(1)), int(m.group(2))), TimePrecision.MINUTE

    m = _YESTERDAY_RE.match(text)
    if m:
        y = now - timedelta(days=1)
        return _at(now, y.year, y.month, y.day,
                   int(m.group(1)), int(m.group(2))), TimePrecision.MINUTE

    m = _CTIME_RE.match(text)
    if m:
        month_name, day, clock, offset, year = m.groups()
        month = _MONTHS.get(month_name)
        if month:
            hh, mm, ss = (int(part) for part in clock.split(":"))
            sign = 1 if offset[0] == "+" else -1
            tz = timezone(sign * timedelta(hours=int(offset[1:3]),
                                           minutes=int(offset[3:5])))
            try:
                dt = datetime(int(year), month, int(day), hh, mm, ss, tzinfo=tz)
                return dt.astimezone(CST), TimePrecision.SECOND
            except ValueError:
                return None, TimePrecision.UNKNOWN

    m = _FULL_DATE_TIME_RE.match(text)
    if m:
        y, mo, d, hh, mi = (int(g) for g in m.groups())
        return _at(now, y, mo, d, hh, mi), TimePrecision.MINUTE

    m = _FULL_DATE_RE.match(text)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return _at(now, y, mo, d, 0, 0), TimePrecision.DAY

    m = _MONTH_DAY_RE.match(text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return _at(now, _infer_year(now, mo, d), mo, d, 0, 0), TimePrecision.DAY

    m = _CN_MONTH_DAY_RE.match(text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return _at(now, _infer_year(now, mo, d), mo, d, 0, 0), TimePrecision.DAY

    return None, TimePrecision.UNKNOWN


def _infer_year(now: datetime, month: int, day: int) -> int:
    """微博的「MM-DD」不含年份；若推出的日期晚于今天，则视为上一年。"""
    try:
        candidate = datetime(now.year, month, day, tzinfo=CST)
    except ValueError:
        return now.year
    return now.year - 1 if candidate.date() > now.date() else now.year


def _at(now: datetime, y: int, mo: int, d: int, hh: int, mi: int) -> datetime | None:
    try:
        return datetime(y, mo, d, hh, mi, tzinfo=CST)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# 接口包装
# --------------------------------------------------------------------------- #
def check_api_ok(payload: Any, *, label: str) -> dict[str, Any]:
    """校验微博接口的外层包装 ``{"ok": 1, "data": {...}}``。

    这是所有采集请求的必经关口：登录失效、风控、业务错误
    都在这里被转成用户可读的中文错误，而不是继续往下解析出空数据。
    """
    if not isinstance(payload, dict):
        raise ParseFailure(detail=f"{label}: 响应不是 JSON 对象，类型为 {type(payload).__name__}")

    ok_value = payload.get("ok")
    message = str(payload.get("msg") or payload.get("message") or "").strip()

    if ok_value == 1:
        return payload
    if ok_value == -100:
        raise AuthRequired(detail=f"{label}: ok=-100, msg={message!r}")
    if ok_value in (0, None) and "login" in message.lower():
        raise AuthRequired(detail=f"{label}: ok={ok_value}, msg={message!r}")
    raise ApiError(
        detail=f"{label}: ok={ok_value}, msg={message!r}",
        what_happened=(
            f"微博接口拒绝了本次请求（{label}）。"
            + (f"接口返回：{message}" if message else "")
        ),
    )


def is_logged_in(payload: Any) -> bool:
    """``/api/config`` 的登录判定。"""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return bool(isinstance(data, dict) and data.get("login"))


# --------------------------------------------------------------------------- #
# 帖子
# --------------------------------------------------------------------------- #
def iter_mblogs(payload: Any) -> Iterator[dict[str, Any]]:
    """从检索响应里按顺序取出所有 mblog 对象。

    微博检索的 cards 至少有两种形态：
    - 卡片自身带 ``mblog``（单条微博卡片）；
    - 卡片带 ``card_group``，组内每一项才是微博卡片。
    两种都要覆盖，否则会漏掉一部分结果。
    """
    seen: set[str] = set()
    for card in _cards(payload):
        candidates: list[dict[str, Any]] = []
        if isinstance(card.get("mblog"), dict):
            candidates.append(card)
        group = card.get("card_group")
        if isinstance(group, list):
            candidates.extend(item for item in group
                              if isinstance(item, dict) and isinstance(item.get("mblog"), dict))
        for item in candidates:
            mblog = item.get("mblog") or {}
            mid = _mid_of(mblog)
            if not mid or mid in seen:
                continue
            seen.add(mid)
            yield mblog


def _cards(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    cards = data.get("cards")
    if isinstance(cards, list):
        return [card for card in cards if isinstance(card, dict)]
    return []


def _mid_of(mblog: dict[str, Any]) -> str:
    for key in ("mid", "id", "idstr"):
        value = mblog.get(key)
        if value not in (None, "", 0):
            return str(value)
    return ""


#: 中文数量单位。微博把粉丝数等大数值以「2733.7万」这种字符串返回。
_COUNT_UNITS: dict[str, float] = {
    "千": 1_000,
    "万": 10_000,
    "萬": 10_000,
    "亿": 100_000_000,
    "億": 100_000_000,
    "k": 1_000,
    "K": 1_000,
    "w": 10_000,
    "W": 10_000,
}
_COUNT_RE = re.compile(r"^\s*([\d.]+)\s*([^\d.\s]{0,1})\s*$")


def _to_int(value: Any) -> int:
    """把接口里的"数字"统一成非负整数。

    ⚠ 这里有一个真实踩过的坑：微博的 ``user.followers_count`` 返回的是
    **带中文单位的字符串**，例如 ``"2733.7万"``。若按"去掉非数字字符"处理，
    会得到 27337 —— 差了 1000 倍。而 B2 的「清洗前后粉丝数分布对比」正是靠
    这个字段判断清洗有没有引入系统性偏差，错 1000 倍等于让这项分析失效。

    >>> _to_int("2733.7万")
    27337000
    >>> _to_int("1,024")
    1024
    >>> _to_int(None)
    0
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace(" ", "")
        if not text:
            return 0
        match = _COUNT_RE.match(text)
        if match:
            number_text, unit = match.groups()
            try:
                number = float(number_text)
            except ValueError:
                return 0
            multiplier = _COUNT_UNITS.get(unit, 1)
            return int(round(number * multiplier))
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else 0
    return 0


def post_from_mblog(mblog: dict[str, Any], *, now: datetime | None = None) -> WeiboPost:
    """把一条 mblog 转成 ``WeiboPost``。"""
    mid = _mid_of(mblog)
    if not mid:
        raise ParseFailure(detail=f"mblog 缺少 mid: keys={sorted(mblog)[:12]}")

    user = mblog.get("user") if isinstance(mblog.get("user"), dict) else {}
    created_raw = str(mblog.get("created_at") or "")
    created_at, precision = parse_created_at(created_raw, now=now)

    return WeiboPost(
        mid=mid,
        user_id=str(user.get("id") or user.get("idstr") or ""),
        screen_name=str(user.get("screen_name") or ""),
        text_raw=strip_html(mblog.get("text") or mblog.get("raw_text")),
        created_at=created_at,
        created_at_raw=created_raw,
        created_at_precision=precision,
        reposts_count=_to_int(mblog.get("reposts_count")),
        comments_count=_to_int(mblog.get("comments_count")),
        attitudes_count=_to_int(mblog.get("attitudes_count")),
        bid=str(mblog.get("bid") or "") or None,
        is_long_text=bool(mblog.get("isLongText")),
    )


def parse_search_page(
    payload: Any, *, page: int = 1, now: datetime | None = None
) -> SearchPage:
    """解析 tag 检索的一页。"""
    check_api_ok(payload, label=f"检索第 {page} 页")
    posts = [post_from_mblog(mblog, now=now) for mblog in iter_mblogs(payload)]
    data = payload.get("data") if isinstance(payload, dict) else {}
    has_more = True
    if isinstance(data, dict):
        if data.get("cardlistInfo"):
            info = data["cardlistInfo"]
            if isinstance(info, dict) and info.get("page") is not None:
                has_more = True
    return SearchPage(posts=posts, page=page, has_more=has_more and bool(posts))


# --------------------------------------------------------------------------- #
# 评论
# --------------------------------------------------------------------------- #
def parse_comments_page(
    payload: Any,
    *,
    post_mid: str,
    level: CommentLevel = CommentLevel.PRIMARY,
    parent_cid: str | None = None,
    now: datetime | None = None,
) -> CommentPage:
    """解析一页评论。

    容错要点（都对应真实的接口差异）：

    - 移动端 ``comments/hotflow`` 把评论放在 ``data.data``；
    - 网页端 ``ajax/statuses/buildComments`` 把评论**直接**放在 ``data``（数组）；
    - 有的形态把游标放在外层 ``payload.max_id``，有的放在 ``data.max_id``；
    - 没有跟评时，二级评论接口可能返回 ``data`` 为 ``None`` 或结构异常。

    所以这里对三种数据位置、两种游标位置都做兼容，且**任何异常结构都返回空页
    而不是抛错**——「空」与「坏」的区分交给调用方（见 ``collect_comments``），
    因为在解析层无法判断"这个帖子本来就没评论"还是"接口变了"。
    """
    check_api_ok(payload, label=f"评论 mid={post_mid}")
    body = payload if isinstance(payload, dict) else {}

    items: list[Any] = []
    container: dict[str, Any] = {}

    data = body.get("data")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        container = data
        for key in ("data", "comments", "list"):
            candidate = container.get(key)
            if isinstance(candidate, list):
                items = candidate
                break

    comments: list[WeiboComment] = []
    for item in items:
        comment = _comment_from_item(
            item, post_mid=post_mid, level=level, parent_cid=parent_cid, now=now
        )
        if comment is not None:
            comments.append(comment)

    next_max_id = _first_present(container.get("max_id"), body.get("max_id"))
    next_max_id_type = _first_present(
        container.get("max_id_type"), body.get("max_id_type")
    )
    total = _first_present(
        container.get("total_number"), body.get("total_number"), body.get("total")
    )

    return CommentPage(
        comments=comments,
        next_max_id=_to_int(next_max_id) if next_max_id not in (None, "") else None,
        next_max_id_type=(
            _to_int(next_max_id_type) if next_max_id_type not in (None, "") else None
        ),
        total_number=_to_int(total) if total not in (None, "") else None,
    )


def _first_present(*values: Any) -> Any:
    """返回第一个非 None / 非空字符串的值。"""
    for value in values:
        if value is not None and value != "":
            return value
    return None


#: 公开别名：清洗报告的分布统计也需要解析「2733.7万」这类数值，
#: 让它复用同一套规则，而不是各写一份。
to_count = _to_int


def _comment_from_item(
    item: Any,
    *,
    post_mid: str,
    level: CommentLevel,
    parent_cid: str | None,
    now: datetime | None,
) -> WeiboComment | None:
    if not isinstance(item, dict):
        return None
    cid = str(item.get("id") or item.get("cid") or item.get("idstr") or "")
    if not cid:
        return None
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    created_raw = str(item.get("created_at") or "")
    created_at, precision = parse_created_at(created_raw, now=now)

    like = item.get("like_counts")
    if like in (None, ""):
        like = item.get("like_count", item.get("likes_count"))

    reply = item.get("total_number")
    if reply in (None, ""):
        reply = item.get("reply_count", 0)

    followers = user.get("followers_count")
    if followers in (None, ""):
        followers = None
    else:
        followers = _to_int(followers)

    return WeiboComment(
        cid=cid,
        post_mid=post_mid,
        user_id=str(user.get("id") or user.get("idstr") or ""),
        screen_name=str(user.get("screen_name") or ""),
        text_raw=strip_html(item.get("text")),
        created_at=created_at,
        created_at_raw=created_raw,
        created_at_precision=precision,
        level=level,
        parent_cid=parent_cid,
        like_count=_to_int(like),
        reply_count=_to_int(reply),
        followers_count=followers,
    )


# --------------------------------------------------------------------------- #
# 桌面检索页（降级路径）
# --------------------------------------------------------------------------- #
_RENDER_DATA_MARKER = "$render_data"
_STATUS_MARKER = "$render_data"


def extract_desktop_render_data(html_text: str) -> list[Any] | None:
    """从 ``s.weibo.com`` 检索页里抠出 ``$render_data`` 的 JSON 数组。

    桌面页把首屏数据内联在 ``var $render_data = [...]`` 中。
    这里用括号配平扫描而不是正则（正则处理不了嵌套数组），
    并对字符串与转义做了基本处理。
    """
    index = html_text.find(_RENDER_DATA_MARKER)
    if index < 0:
        return None
    bracket_start = html_text.find("[", index)
    if bracket_start < 0:
        return None
    chunk = _slice_balanced(html_text, bracket_start)
    if chunk is None:
        return None
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _slice_balanced(text: str, start: int) -> str | None:
    """返回从 ``start`` 起括号配平的子串。"""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_desktop_search_page(
    html_text: str, *, page: int = 1, now: datetime | None = None
) -> SearchPage:
    """解析桌面检索页。拿不到内联数据时抛 ``ParseFailure``（可能被重定向到登录页）。"""
    render = extract_desktop_render_data(html_text)
    if not render:
        raise ParseFailure(
            detail=(
                f"桌面检索页未找到 $render_data（page={page}, len={len(html_text)}）；"
                "常见原因是被重定向到登录页或验证页。"
            )
        )
    payload = render[0] if render and isinstance(render[0], dict) else {}
    mblogs: list[dict[str, Any]] = []
    statuses = payload.get("statuses")
    if isinstance(statuses, list):
        mblogs.extend(s for s in statuses if isinstance(s, dict))
    cards = payload.get("cards")
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            if isinstance(card.get("mblog"), dict):
                mblogs.append(card["mblog"])
            group = card.get("card_group")
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict) and isinstance(item.get("mblog"), dict):
                        mblogs.append(item["mblog"])
    posts: list[WeiboPost] = []
    seen: set[str] = set()
    for mblog in mblogs:
        try:
            post = post_from_mblog(mblog, now=now)
        except ParseFailure:
            continue
        if post.mid in seen:
            continue
        seen.add(post.mid)
        posts.append(post)
    return SearchPage(posts=posts, page=page, has_more=bool(posts))
