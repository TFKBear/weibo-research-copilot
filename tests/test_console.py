"""终端输出的最底线：错误提示不能自己抛异常。

真实 bug：``console.fail()`` 写成 ``Console.print(..., file=sys.stderr)``，
而 rich 的 ``Console.print`` 根本没有 ``file`` 参数。结果**所有**命令行失败路径
都会抛 ``TypeError: Console.print() got an unexpected keyword argument 'file'``——
一个本该说"你填错了"的地方，变成了 Python traceback。
这对社科学生来说是最糟的失败方式：错误信息完全不可读。
"""

from __future__ import annotations

import pytest

from wrc import console as console_module


class _FakeStream:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def test_fail_writes_to_stderr_without_raising(monkeypatch):
    fake = _FakeStream()
    monkeypatch.setattr(console_module.sys, "stderr", fake)
    console_module.fail("--per-post 里的编号 2 超出范围")

    assert "✗" in fake.text
    assert "超出范围" in fake.text


def test_fail_keeps_stdout_clean(monkeypatch):
    """错误不能混进 stdout：脚本调用方是拿 stdout 当数据的。"""
    out = _FakeStream()
    err = _FakeStream()
    monkeypatch.setattr(console_module.sys, "stdout", out)
    monkeypatch.setattr(console_module.sys, "stderr", err)
    console_module.fail("出错了")

    assert "出错了" in err.text
    assert "出错了" not in out.text


def test_report_error_renders_human_message(capsys):
    from wrc.errors import ConfigProblem

    console_module.report_error(ConfigProblem(what_happened="时间窗起点晚于终点。"))
    captured = capsys.readouterr()
    assert "时间窗起点晚于终点" in captured.out


def test_all_console_helpers_run(capsys):
    """这些函数是每个命令都会走的路，任何一个抛异常都是全局故障。"""
    console_module.banner("标题")
    console_module.info("信息")
    console_module.note("备注")
    console_module.warn("警告")
    console_module.ok("完成")
    console_module.block("[doge] 正文里的表情不能被 rich 吃掉")
    console_module.kv_table("表", [("键", "值")])

    captured = capsys.readouterr()
    assert "[doge]" in captured.out
