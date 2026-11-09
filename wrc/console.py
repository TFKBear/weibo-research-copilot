"""终端输出。

硬性要求（Agent 执行规则第 4 条）：采集、清洗、分析都必须有可见输出，
用户不能面对「转圈 3 分钟然后报错」的黑箱。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.table import Table

from .errors import render_error
_console: Console | None = None


def _harden_stdio() -> None:
    """把标准输出切成 UTF-8。

    Windows 上 Python 默认用 ANSI 代码页（简中系统是 cp936，英文系统是 cp1252），
    中文提示会直接抛 ``UnicodeEncodeError`` —— 对一个全中文界面的工具来说这等于不可用。
    这里在导入时统一改成 UTF-8，并对无法编码的字符降级替换而非崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - 极少见
            pass


_harden_stdio()


def console() -> Console:
    global _console
    if _console is None:
        # 输出被重定向到文件时表格没有真实终端宽度可用，
        # rich 会退回到 80 列并把中文长句折成两行，日志会变得很难读。
        width = None if _is_tty() else 110
        _console = Console(
            highlight=False,
            soft_wrap=False,
            legacy_windows=False,
            width=width,
        )
    return _console


def _is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):  # pragma: no cover
        return False


def banner(title: str) -> None:
    console().rule(f"[bold cyan]{title}[/bold cyan]")


def step(index: int | None, message: str) -> None:
    prefix = f"[dim]({index})[/dim] " if index is not None else ""
    console().print(f"{prefix}[bold]{message}[/bold]")


def info(message: str) -> None:
    console().print(f"   {message}")


def note(message: str) -> None:
    console().print(f"   [dim]{message}[/dim]")


def warn(message: str) -> None:
    console().print(f"   [yellow]⚠ {message}[/yellow]")


def ok(message: str) -> None:
    console().print(f"   [green]✓[/green] {message}")


def fail(message: str) -> None:
    """错误行走 stderr。``Console.print`` 没有 ``file`` 参数，
    要写 stderr 得用 ``Console(stderr=True)``——早先写成 ``file=sys.stderr``
    会在**所有**命令行失败路径上抛 TypeError，把"人性化报错"变成 traceback。
    """
    stderr_console = Console(
        stderr=True, highlight=False, legacy_windows=False,
        width=None if _is_tty() else 110,
    )
    stderr_console.print(f"   [red]✗[/red] {message}")


def kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, title_justify="left", show_header=False, box=None,
                  padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value)
    console().print(table)


def block(text: str) -> None:
    """原样输出一段预格式化文本。

    两个细节都是必须的：
    - 关闭 rich 的 markup 解析 —— 微博正文里的 ``[doge]``、``[笑cry]``
      会被 rich 当成样式标签吃掉，而表情恰恰是本研究要保留的信号；
    - 关闭自动折行 —— 清洗报告是表格化文本，折行会把对齐彻底打乱。
    """
    console().print(text, markup=False, highlight=False, soft_wrap=True)


def report_error(exc: BaseException) -> None:
    """统一的用户可见错误出口。"""
    console().print(render_error(exc))


@contextmanager
def progress(description: str, total: int | None = None) -> Iterator[object]:
    """进度条。total 未知时退化为转圈。"""
    with console().status(f"[cyan]{description}[/cyan]", spinner="dots") as status:
        yield status
