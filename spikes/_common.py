"""技术验证脚本的公共引导。

每个脚本都能独立运行（``python spikes/0X_xxx.py``），
这里统一处理三件事：让 ``wrc`` 可被导入、把异常转成人话、把退出码规范化。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Awaitable, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wrc.config import Settings, get_settings  # noqa: E402
from wrc.console import banner, console, info, note, report_error, warn  # noqa: E402
from wrc.errors import WrcError  # noqa: E402

EXIT_OK = 0
EXIT_USER_ERROR = 2
EXIT_INTERRUPTED = 130


def print_header(title: str, purpose: str) -> None:
    banner(title)
    console().print(f"   {purpose}", markup=False)
    console().print()


def ask_consent(confirmed_flag: bool) -> bool:
    """导出前的使用声明确认。

    设计上刻意不做「默认同意」：确认动作必须是显式的。
    """
    from wrc.privacy.consent import render_statement

    console().print(render_statement(), markup=False)
    console().print()
    if confirmed_flag:
        info("已通过 --consent 参数确认数据使用声明。")
        return True
    if not sys.stdin.isatty():
        warn("当前是非交互环境，无法询问确认；如确认请加 --consent 参数。")
        return False
    answer = input("确认以上声明并继续导出？输入 y 确认，其他任意键取消：").strip().lower()
    return answer in {"y", "yes", "是"}


def make_settings(args: argparse.Namespace, **overrides: object) -> Settings:
    payload: dict[str, object] = {}
    if getattr(args, "headless", False):
        payload["headless"] = True
    if getattr(args, "fetch_mode", None):
        payload["fetch_mode"] = args.fetch_mode
    if getattr(args, "delay_min", None):
        payload["delay_min_s"] = args.delay_min
    if getattr(args, "delay_max", None):
        payload["delay_max_s"] = args.delay_max
    if getattr(args, "max_requests", None):
        payload["max_requests_per_run"] = args.max_requests
    payload.update(overrides)
    return get_settings(**payload)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", help="无头运行（排障时才用）")
    parser.add_argument(
        "--fetch-mode",
        choices=("context", "page"),
        default=None,
        help="传输模式：context（默认，走 APIRequestContext）/ page（页面内 fetch）",
    )
    parser.add_argument("--delay-min", type=float, default=None, help="请求间最小延迟（秒）")
    parser.add_argument("--delay-max", type=float, default=None, help="请求间最大延迟（秒）")
    parser.add_argument(
        "--max-requests", type=int, default=None, help="单次采集请求硬上限"
    )


def run_async(main: Callable[[], Awaitable[int]]) -> int:
    """统一入口：异常转人话 + 退出码。"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        console().print()
        warn("已中断。已完成部分的进度保存在 data/ 目录中。")
        return EXIT_INTERRUPTED
    except WrcError as exc:
        console().print()
        report_error(exc)
        return EXIT_USER_ERROR
    except Exception as exc:  # pragma: no cover - 兜底
        console().print()
        report_error(exc)
        return EXIT_USER_ERROR


def parse_args(argv: Sequence[str] | None = None, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    return (parser or argparse.ArgumentParser()).parse_args(argv)


__all__ = [
    "EXIT_OK",
    "EXIT_USER_ERROR",
    "EXIT_INTERRUPTED",
    "PROJECT_ROOT",
    "add_common_arguments",
    "ask_consent",
    "console",
    "info",
    "make_settings",
    "note",
    "parse_args",
    "print_header",
    "run_async",
    "warn",
]
