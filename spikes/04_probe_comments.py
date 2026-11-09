"""04 · 评论接口诊断（回答「为什么一条评论都拿不到」）。

背景：第 1 步实机测试时，检索、登录、帖子预览全部正常，
但评论接口返回了 ``ok=1`` 却没有任何评论数据 —— 而平台显示该帖子有 679 条评论。
这属于"空"与"坏"没被区分开的典型故障，也是本工具必须能自己定位的问题。

本脚本把**每一个候选评论后端**都请求一次，并打印：
  · HTTP 层是否成功、接口的 ``ok`` / ``msg``；
  · ``data`` 的真实结构（是数组？是对象？键名是什么？）；
  · 拿到多少条评论；
  · 第一条评论有哪些字段 —— 这直接决定解析器能不能对上（例如
    ``like_counts`` 还是 ``likes_count``，``total_number`` 是否存在）。

完整原始响应会落盘到 ``data/logs/comments_probe_<时间>/``，
可以直接发给维护者，不需要靠猜。

用法：
    python spikes/04_probe_comments.py --mid 5308942530118155
    python spikes/04_probe_comments.py --tag 小米SU7          # 自动取评论最多的帖子
    python spikes/04_probe_comments.py --mid 5308942530118155 --fetch-mode page
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

import _common
from _common import add_common_arguments, console, info, note, warn
from rich.table import Table

from wrc import __version__
from wrc.errors import WrcError
from wrc.weibo.comment_backends import (
    MOBILE_HOTFLOW,
    MOBILE_HOTFLOW_RELOAD,
    WEB_BUILD_COMMENTS,
)
from wrc.weibo.parsing import CST, parse_comments_page


def _variants() -> list[tuple[str, Any, str]]:
    """要试的组合：(标签, 后端, 排序)。"""
    return [
        ("移动端 hotflow", MOBILE_HOTFLOW, "hot"),
        ("移动端 hotflow (is_reload)", MOBILE_HOTFLOW_RELOAD, "hot"),
        ("网页端 buildComments flow=0（热度）", WEB_BUILD_COMMENTS, "hot"),
        ("网页端 buildComments flow=1（时间）", WEB_BUILD_COMMENTS, "time"),
    ]


def _describe(payload: Any) -> dict[str, Any]:
    """把一个评论响应拆成可读的诊断信息。"""
    out: dict[str, Any] = {
        "ok": None,
        "msg": "",
        "data_type": type(payload).__name__,
        "data_keys": [],
        "items": 0,
        "item_fields": [],
        "max_id": None,
        "max_id_type": None,
        "total_number": None,
    }
    if not isinstance(payload, dict):
        return out

    out["ok"] = payload.get("ok")
    out["msg"] = str(payload.get("msg") or payload.get("message") or "")[:80]

    data = payload.get("data")
    if isinstance(data, list):
        out["data_type"] = "list"
        items = data
    elif isinstance(data, dict):
        out["data_type"] = "dict"
        out["data_keys"] = sorted(data)[:10]
        items = []
        for key in ("data", "comments", "list"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
    else:
        items = []

    out["items"] = len(items)
    if items and isinstance(items[0], dict):
        out["item_fields"] = sorted(items[0])[:16]

    for source in (payload, data if isinstance(data, dict) else {}):
        if out["max_id"] is None and source.get("max_id") is not None:
            out["max_id"] = source.get("max_id")
            out["max_id_type"] = source.get("max_id_type")
    if isinstance(data, dict):
        total = data.get("total_number") or data.get("total")
        out["total_number"] = total
    return out


async def run(args: argparse.Namespace) -> int:
    from wrc.session import BrowserSession
    from wrc.weibo.collect import WeiboClient
    from wrc.weibo.ratelimit import RateLimiter, RequestBudget

    settings = _common.make_settings(args)
    settings.ensure_dirs()

    _common.print_header(
        "04 · 评论接口诊断",
        "逐个请求候选评论接口，打印真实返回结构；不写入研究数据。",
    )

    stamp = datetime.now(CST).strftime("%Y%m%dT%H%M%S")
    out_dir = settings.log_dir / f"comments_probe_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"原始响应将保存到：{out_dir}")

    async with BrowserSession(settings, notify=note) as session:
        limiter = RateLimiter(
            delay_range=settings.effective_delay_range(),
            budget=RequestBudget(limit=args.max_requests or 40),
        )
        client = WeiboClient(session.fetcher, limiter, max_retries=0, notify=warn)

        # ---------- 确定目标帖 ---------- #
        post_mid = args.mid
        expected_total: int | None = None
        if not post_mid:
            tag = args.tag or "小米SU7"
            info(f"未指定 --mid，先在话题 {tag} 里找评论最多的帖子 …")
            posts = await client.search_posts(f"#{tag}#", limit=15, order="engagement")
            best = max(posts, key=lambda p: p.comments_count)
            post_mid = best.mid
            expected_total = best.comments_count
            info(f"选中 mid={best.mid}（评论数 {best.comments_count}）：{best.preview(40)}")
        elif args.expect is not None:
            expected_total = args.expect

        info(f"目标帖子 mid={post_mid}")

        # ---------- 逐个后端探测 ---------- #
        table = Table(title="评论接口探测结果", title_justify="left", show_lines=True)
        table.add_column("尝试", no_wrap=True)
        table.add_column("结果", no_wrap=True)
        table.add_column("响应结构 / 错误", overflow="fold")

        results: list[dict[str, Any]] = []
        for label, backend, sort in _variants():
            request = backend.request(post_mid, max_id=0, max_id_type=0, sort=sort)
            note(f"请求：{label}")
            record: dict[str, Any] = {
                "label": label,
                "backend": backend.key,
                "sort": sort,
                "url": request.url,
                "params": dict(request.params),
                "referer": request.referer,
            }
            try:
                # 诊断脚本要拿到「原始响应」而不经过 ok 校验，所以不走 client.get_json；
                # 但仍必须经过限速器，否则一次诊断就会发出一串无间隔请求。
                await client._limiter.wait_turn()  # noqa: SLF001
                text = await session.fetcher.get_text(request)
            except WrcError as exc:
                record["error"] = f"{exc.kind.value}: {exc.what_happened}"
                record["detail"] = exc.detail[:500]
                results.append(record)
                table.add_row(label, f"[red]✗ {exc.kind.value}[/red]", exc.what_happened)
                continue
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
                results.append(record)
                table.add_row(label, "[red]✗ 异常[/red]", str(exc)[:200])
                continue

            raw_path = out_dir / f"raw_{backend.key}_{sort}.json"
            raw_path.write_text(text, encoding="utf-8")
            record["raw_file"] = str(raw_path)

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                results.append({**record, "error": "非 JSON 响应"})
                table.add_row(
                    label, "[yellow]非 JSON[/yellow]", text[:200].replace("\n", " ")
                )
                continue

            shape = _describe(payload)
            page = parse_comments_page(payload, post_mid=post_mid)
            record["shape"] = shape
            record["parsed_comments"] = len(page.comments)
            record["next_max_id"] = page.next_max_id
            results.append(record)

            if page.comments:
                table.add_row(
                    label,
                    f"[green]✓ {len(page.comments)} 条[/green]",
                    f"data={shape['data_type']}，游标 max_id={page.next_max_id}，"
                    f"首条字段：{', '.join(shape['item_fields'])}",
                )
            else:
                table.add_row(
                    label,
                    "[yellow]ok 但 0 条[/yellow]",
                    f"ok={shape['ok']} msg={shape['msg'] or '（空）'} "
                    f"data={shape['data_type']} "
                    f"keys={shape['data_keys']} items={shape['items']} "
                    f"max_id={shape['max_id']}",
                )

        console().print()
        console().print(table)

        summary_path = out_dir / "probe.json"
        summary_path.write_text(
            json.dumps(
                {
                    "ts": datetime.now(CST).isoformat(timespec="seconds"),
                    "tool_version": __version__,
                    "post_mid": post_mid,
                    "expected_total": expected_total,
                    "fetch_mode": settings.fetch_mode,
                    "engine": session.engine,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ---------- 结论 ---------- #
        console().print()
        console().print("结论", markup=False)
        working = [r for r in results if r.get("parsed_comments")]
        if working:
            best = working[0]
            console().print(
                f"  ✓ 有可用后端：「{best['label']}」，解析到 {best['parsed_comments']} 条评论。",
                markup=False,
            )
            console().print(
                f"    可在采集时指定：--comment-backend {best['backend']} "
                f"--comment-sort {best['sort']}",
                markup=False,
            )
        elif expected_total and expected_total > 0:
            console().print(
                "  ✗ 所有候选后端都返回了空数据，而该帖显示有 "
                f"{expected_total} 条评论 —— 这是接口形态变化，不是「没有评论」。",
                markup=False,
            )
            console().print(
                f"    请把 {summary_path} 与同目录下的 raw_*.json 发给维护者；",
                markup=False,
            )
            console().print(
                "    原始响应里能看到接口到底改成了什么结构，这比任何猜测都可靠。",
                markup=False,
            )
        else:
            console().print(
                "  ⚠ 该帖评论数未知或为 0，无法判断是接口问题还是帖子本来没有评论。",
                markup=False,
            )
            console().print(
                "    建议用 --expect 指定该帖显示的评论数后重跑。", markup=False
            )

        console().print()
        console().print(f"  完整记录：{summary_path}", markup=False)
        console().print(f"  原始响应：{out_dir}", markup=False)
        return 0 if working else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评论接口诊断")
    add_common_arguments(parser)
    parser.add_argument("--mid", default=None, help="目标帖子的 mid（推荐直接指定）")
    parser.add_argument("--tag", default=None, help="不给 --mid 时，从该话题里挑评论最多的帖子")
    parser.add_argument("--expect", type=int, default=None, help="该帖显示的评论总数，用于判定'空'与'坏'")
    sys.exit(_common.run_async(lambda: run(parser.parse_args())))
