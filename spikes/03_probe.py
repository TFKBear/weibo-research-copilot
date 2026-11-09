"""03 · 接口形态探测（回答"哪些假设是真的"）。

为什么需要这个脚本：
开发时有几件事我**无法在不接触真实环境的情况下确认**，而它们直接决定 P0 的
可行性。与其把猜测写进文档当事实，不如做成一次可复现的探测：

1. 移动端检索接口当前能否正常返回、能否翻页（第 2 页是否真的与第 1 页不重叠）；
2. 检索「排序」到底由什么参数控制 —— ``containerid`` 里的 ``type`` 是否就是 Tab，
   桌面页的 ``Refer=hot`` 是否仍对应「热门」。这决定「按热度 / 按时间」能不能实现，
   以及如果不能，我们必须在 UI 上如实说明"这是本地重排"；
3. 评论 ``max_id`` 递推是否稳定，``max_id_type`` 是否需要原样回传；
4. 二级评论接口在无跟评时返回什么形态（空值容错是否够用）；
5. 当前网络环境下是否直接吃到 432/418/验证页。

探测结果全部写入 ``data/logs/probe_<时间>.json``，可以直接作为技术验证记录。

用法：
    python spikes/03_probe.py --tag 延迟退休
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from typing import Any, Callable

import _common
from _common import add_common_arguments, console, info, note, warn
from rich.table import Table
from wrc import __version__
from wrc.errors import WrcError
from wrc.weibo.endpoints import (
    comments_request,
    desktop_search_request,
    search_request,
    subcomments_request,
)
from wrc.weibo.parsing import (
    CST,
    parse_comments_page,
    parse_desktop_search_page,
    parse_search_page,
)


class Prober:
    def __init__(self, client: Any, fetcher: Any, limiter: Any) -> None:
        self.client = client
        self.fetcher = fetcher
        self.limiter = limiter
        self.results: list[dict[str, Any]] = []

    async def probe(self, name: str, expectation: str, fn: Callable[[], Any]) -> Any:
        """跑一个探测项，任何失败都记录成结果而不是中断整轮。"""
        note(f"探测：{name}")
        try:
            value = await fn()
        except WrcError as exc:
            self.results.append(
                {
                    "name": name,
                    "expectation": expectation,
                    "ok": False,
                    "kind": exc.kind.value,
                    "detail": exc.detail[:500] or exc.what_happened,
                }
            )
            return None
        except Exception as exc:  # noqa: BLE001
            self.results.append(
                {
                    "name": name,
                    "expectation": expectation,
                    "ok": False,
                    "kind": "unexpected",
                    "detail": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
            return None
        self.results.append(
            {
                "name": name,
                "expectation": expectation,
                "ok": True,
                "finding": value,
            }
        )
        return value

    def render(self) -> None:
        table = Table(title="探测结果", title_justify="left", show_lines=True)
        table.add_column("探测项", no_wrap=True)
        table.add_column("结果", no_wrap=True)
        table.add_column("发现 / 失败原因", overflow="fold")
        for item in self.results:
            if item["ok"]:
                table.add_row(item["name"], "[green]✓ 可用[/green]",
                              _summarize(item.get("finding")))
            else:
                table.add_row(item["name"], f"[red]✗ {item['kind']}[/red]",
                              str(item.get("detail") or ""))
        console().print(table)


def _summarize(value: Any, limit: int = 240) -> str:
    if value is None:
        return "（无返回）"
    # 先用 _jsonable 归一化：finding 里可能装着 WeiboPost 这类 dataclass 对象，
    # 直接 json.dumps 会抛 TypeError 并把整份探测记录弄丢。
    text = value if isinstance(value, str) else json.dumps(
        _jsonable(value), ensure_ascii=False
    )
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text


async def run(args: argparse.Namespace) -> int:
    from wrc.session import BrowserSession
    from wrc.store import Store
    from wrc.weibo.collect import WeiboClient
    from wrc.weibo.endpoints import (
        comments_request,
        desktop_search_request,
        normalize_tag,
        search_request,
        subcomments_request,
    )
    from wrc.weibo.parsing import (
        parse_comments_page,
        parse_desktop_search_page,
        parse_search_page,
    )
    from wrc.weibo.ratelimit import RateLimiter, RequestBudget

    settings = _common.make_settings(args)
    settings.ensure_dirs()
    tag = normalize_tag(args.tag)

    _common.print_header(
        "03 · 接口形态探测",
        f"在真实环境下确认 {tag} 的检索/评论接口当前形态，不写入研究数据。",
    )
    info("本脚本只读取、不落库；探测过程中的失败会被记录，不会中断整轮。")

    store = Store(settings.db_path)
    store.init_schema()

    async with BrowserSession(settings, notify=note) as session:
        limiter = RateLimiter(
            delay_range=settings.effective_delay_range(),
            budget=RequestBudget(limit=args.max_requests or settings.max_requests_per_run),
        )
        client = WeiboClient(session.fetcher, limiter, max_retries=0, notify=warn)
        prober = Prober(client, session.fetcher, limiter)

        # ---- 1. 登录态 ---- #
        logged_in = await prober.probe(
            "登录态 /api/config",
            "data.login == true",
            client.check_login,
        )
        if not logged_in:
            # 不直接退出：未登录状态下仍有一部分探测有意义（例如公开检索是否可用），
            # 而"未登录也能拿到什么"本身就是需要记录的事实。
            warn("当前未登录。以下结果代表「未登录基线」：")
            warn("  · 检索结果可用来判断公开检索是否可用；")
            warn("  · 评论接口通常要求登录，失败属于预期，不能据此判断接口已失效。")
            warn("建议建立登录态后重跑本脚本，以获得完整探测结果。")

        # ---- 2. 移动端检索 ---- #
        page1 = await prober.probe(
            "移动检索 第1页（containerid type=1）",
            "返回若干帖子卡片",
            lambda: _search_probe(client, tag, page=1, tab_type=1),
        )
        page2 = await prober.probe(
            "移动检索 第2页（翻页是否前进）",
            "第2页与第1页 mid 不重叠",
            lambda: _search_probe(client, tag, page=2, tab_type=1),
        )
        if page1 and page2:
            overlap = len({p.mid for p in page1} & {p.mid for p in page2})
            _override_finding(
                prober, "移动检索 第2页（翻页是否前进）",
                {
                    "page1_posts": len(page1),
                    "page2_posts": len(page2),
                    "overlap_with_page1": overlap,
                    "verdict": "翻页有效" if overlap == 0 and page2 else "翻页异常或结果重复",
                },
            )

        await prober.probe(
            "移动检索 tab_type=2（排序 Tab 假设）",
            "若与 type=1 结果不同，说明 Tab 参数有效",
            lambda: _search_tab_probe(client, tag, page1),
        )

        # ---- 3. 桌面检索（降级路径 + 排序线索）---- #
        desktop_g = await prober.probe(
            "桌面检索 Refer=g（综合）",
            "能从 $render_data 解析出帖子",
            lambda: _desktop_probe(session, client, tag, refer="g"),
        )
        desktop_hot = await prober.probe(
            "桌面检索 Refer=hot（热门假设）",
            "若与 Refer=g 排序不同，说明 Refer 控制排序",
            lambda: _desktop_probe(session, client, tag, refer="hot"),
        )
        if desktop_g and desktop_hot:
            _override_finding(
                prober, "桌面检索 Refer=hot（热门假设）",
                {
                    "refer_g_first3": [p.mid for p in desktop_g[:3]],
                    "refer_hot_first3": [p.mid for p in desktop_hot[:3]],
                    "verdict": (
                        "Refer 疑似控制排序"
                        if [p.mid for p in desktop_g[:3]] != [p.mid for p in desktop_hot[:3]]
                        else "两者顺序一致，Refer 是否控制排序待更严格验证"
                    ),
                },
            )

        # ---- 4. 评论接口 ---- #
        first_post = (page1 or desktop_g or [None])[0]
        if first_post is None:
            warn("没有拿到任何帖子，跳过评论相关探测。")
            warn(
                "最常见原因：未登录 —— 检索接口对未登录请求返回 ok=-100，"
                "这说明「公开访问但不登录」这条路径不可行，必须先建立登录态。"
            )
        else:
            info(
                f"以帖子 mid={first_post.mid}（评论数 {first_post.comments_count}）"
                "作为评论探测对象"
            )
            comments_p1 = await prober.probe(
                "评论 hotflow 第1页",
                "返回 data.data 列表 + max_id",
                lambda: _comments_probe(
                    client, first_post.mid, max_id=0, max_id_type=0
                ),
            )
            if comments_p1 and comments_p1.get("comments"):
                cursor = comments_p1.get("next_max_id") or 0
                comments_p2 = await prober.probe(
                    "评论 hotflow 第2页（max_id 递推）",
                    "翻页后 cid 不重复",
                    lambda: _comments_probe(
                        client,
                        first_post.mid,
                        max_id=int(cursor),
                        max_id_type=int(comments_p1.get("next_max_id_type") or 0),
                    ),
                )
                if comments_p2:
                    overlap = len(
                        {c.cid for c in comments_p1["comments"]}
                        & {c.cid for c in comments_p2["comments"]}
                    )
                    _override_finding(
                        prober, "评论 hotflow 第2页（max_id 递推）",
                        {
                            "page1": len(comments_p1["comments"]),
                            "page2": len(comments_p2["comments"]),
                            "overlap": overlap,
                            "next_max_id_type": comments_p2.get("next_max_id_type"),
                            "verdict": (
                                "递推有效" if overlap == 0
                                else "出现重复，需改用其它翻页口径"
                            ),
                        },
                    )

                target = next(
                    (c for c in comments_p1["comments"] if c.reply_count > 0), None
                )
                if target is not None:
                    await prober.probe(
                        "二级评论 hotFlowChild",
                        "有跟评时返回子评论；无跟评时返回空结构且不抛错",
                        lambda: _subcomment_probe(client, target.cid),
                    )
                else:
                    note("第1页评论中没有带跟评的条目，跳过二级评论探测。")
            else:
                warn("评论接口没有返回数据，跳过翻页与二级评论探测。")

        prober.render()
        path = _dump(settings, prober.results, tag)
        console().print()
        console().print("怎么读这份结果", markup=False)
        console().print(
            "  · 「移动检索 第2页」重叠为 0 → P0 的翻页方案可用；", markup=False
        )
        console().print(
            "  · 「Refer=hot」与「Refer=g」顺序一致 → 站内热门排序拿不到，"
            "产品上的「按热度」只能标注为本地重排（当前实现已如此标注）；",
            markup=False,
        )
        console().print(
            "  · 「移动检索」报 auth_required (ok=-100) → 检索必须登录，"
            "先跑 python spikes/01_login.py；",
            markup=False,
        )
        console().print(f"  · 完整探测记录已写入：{path}", markup=False)
        console().print()
        if not logged_in:
            console().print(
                "  本次为未登录基线，只完成了连通性与鉴权边界的验证。"
                "建立登录态后请重跑一次以获得完整结论。",
                markup=False,
            )
        return 0 if logged_in else 2


# --------------------------------------------------------------------------- #
async def _search_probe(client: Any, tag: str, *, page: int, tab_type: int):

    payload = await client.get_json(search_request(tag, page=page, tab_type=tab_type))
    result = parse_search_page(payload, page=page)
    return result.posts


async def _search_tab_probe(client: Any, tag: str, baseline):

    payload = await client.get_json(search_request(tag, page=1, tab_type=2))
    posts = parse_search_page(payload, page=1).posts
    baseline_ids = [p.mid for p in (baseline or [])[:5]]
    return {
        "posts": len(posts),
        "first5": [p.mid for p in posts[:5]],
        "baseline_first5": baseline_ids,
        "verdict": (
            "tab_type 会影响结果集"
            if [p.mid for p in posts[:5]] != baseline_ids
            else "tab_type=2 与 type=1 结果相同，该参数可能无效"
        ),
    }


async def _desktop_probe(session: Any, client: Any, tag: str, *, refer: str):

    # 桌面页是 HTML，不走 client.get_json；但仍必须经过限速与预算
    await client._limiter.wait_turn()  # noqa: SLF001 - 这是诊断脚本，直连可接受
    text = await session.fetcher.get_text(desktop_search_request(tag, page=1, refer=refer))
    result = parse_desktop_search_page(text, page=1)
    return result.posts


async def _comments_probe(client: Any, mid: str, *, max_id: int, max_id_type: int):

    payload = await client.get_json(
        comments_request(mid, max_id=max_id, max_id_type=max_id_type)
    )
    page = parse_comments_page(payload, post_mid=mid)
    raw_keys = sorted(payload.get("data", {}).keys()) if isinstance(payload, dict) else []
    return {
        "comments": page.comments,
        "count": len(page.comments),
        "next_max_id": page.next_max_id,
        "next_max_id_type": page.next_max_id_type,
        "total_number": page.total_number,
        "data_keys": raw_keys,
    }


async def _subcomment_probe(client: Any, cid: str):

    payload = await client.get_json(subcomments_request(cid))
    page = parse_comments_page(payload, post_mid="", level=2, parent_cid=cid)  # type: ignore[arg-type]
    return {
        "children": len(page.comments),
        "next_max_id": page.next_max_id,
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }


def _override_finding(prober: Prober, name: str, finding: dict[str, Any]) -> None:
    for item in prober.results:
        if item["name"] == name:
            item["finding"] = finding
            return


def _dump(settings: Any, results: list[dict[str, Any]], tag: str):
    target = settings.log_dir / f"probe_{datetime.now(CST).strftime('%Y%m%dT%H%M%S')}.json"
    payload = {
        "ts": datetime.now(CST).isoformat(timespec="seconds"),
        "tag": tag,
        "tool_version": __version__,
        "results": _jsonable(results),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _jsonable(value: Any) -> Any:
    """把任意对象转成可 JSON 序列化的结构。

    必须显式支持 ``dataclass(slots=True)``：这类实例没有 ``__dict__``，
    靠 ``hasattr(obj, "__dict__")`` 判断会漏掉它们，最终在一次
    ``json.dumps`` 上把整份探测记录炸掉（第 1 步实际发生过）。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name, None))
            for field in dataclasses.fields(value)
        }
    if hasattr(value, "__dict__"):
        return _jsonable(
            {
                key: getattr(value, key)
                for key in ("mid", "cid", "created_at", "created_at_raw", "like_count",
                            "reply_count", "comments_count")
                if hasattr(value, key)
            }
        )
    return repr(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="探测微博接口当前形态")
    add_common_arguments(parser)
    parser.add_argument("--tag", default="延迟退休", help="用于探测的话题标签")
    sys.exit(_common.run_async(lambda: run(parser.parse_args())))
