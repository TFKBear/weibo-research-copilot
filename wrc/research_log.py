"""研究日志（可复现性的最小单位）。

从第一天就要记录，而不是等到 P2 再补：
社科研究者需要在方法部分交代「我是怎么采到这批数据的」，
这份日志就是那段文字的原始素材。

格式选择 JSONL（每行一个事件）而不是 CSV：
- 追加写不会因为中断而破坏整个文件；
- 事件字段可以随开发迭代增减，不必迁移表结构；
- 之后可以无损渲染成 ``research_log.md``。

与 ``wrc/reproducibility.py`` 的分工（P2 已拆分）：
本模块记**事件流**（什么时刻发生了什么），那是过程留痕；
正式的可复现性附录由 ``reproducibility.py`` 生成（环境、参数、校验和、复现等级），
那份是给论文附录用的成品。两者都保留，因为回答的问题不同。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import __version__
from .weibo.parsing import CST


def _now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


@dataclass(slots=True)
class ResearchLog:
    """追加式事件日志。

    Args:
        path: JSONL 文件路径。
        run_id: 一次采集会话的标识，便于把事件与数据行对应起来。
        echo: 是否同时打印到终端。
    """

    path: Path
    run_id: str
    echo: bool = False
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """记录一个事件。``fields`` 必须是可 JSON 序列化的标量/容器。"""
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "event": event,
            "tool_version": __version__,
        }
        for key, value in fields.items():
            entry[key] = _jsonable(value)
        self._events.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.echo:
            print(f"   [log] {event} {json.dumps(fields, ensure_ascii=False)}"
                  if fields else f"   [log] {event}")
        return entry

    # ------------------------------------------------------------------ #
    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def to_markdown(self, *, title: str = "研究日志") -> str:
        """把本次会话的事件渲染成人类可读的 Markdown 草稿。

        这是**过程记录**，不是论文附录；正式附录见
        ``wrc.reproducibility.build_appendix``。
        """
        lines: list[str] = [f"# {title}", ""]
        lines.append(f"- 采集会话 ID：`{self.run_id}`")
        lines.append(f"- 工具版本：`{__version__}`")
        if self._events:
            lines.append(f"- 事件数：{len(self._events)}")
        lines.append("")
        lines.append("| 时间 | 事件 | 参数 |")
        lines.append("| --- | --- | --- |")
        for entry in self._events:
            payload = {
                key: value
                for key, value in entry.items()
                if key not in {"ts", "run_id", "event", "tool_version"}
            }
            rendered = json.dumps(payload, ensure_ascii=False) if payload else ""
            rendered = rendered.replace("|", "\\|")
            lines.append(f"| {entry['ts']} | `{entry['event']}` | {rendered} |")
        lines.append("")
        return "\n".join(lines)

    def write_markdown(self, destination: Path | None = None) -> Path:
        """写 Markdown 版日志。

        ⚠ 必须先把磁盘上已有的事件读回来。真实事故：导出的
        ``research_log.md`` 只有表头、没有一行事件，因为导出发生在**另一个
        进程**里，新建的 ResearchLog 对象内存中没有任何事件。
        而这份文件正是「可复现性」最直接的证据——空表等于没有过程记录。
        """
        if not self._events and self.path.exists():
            self._events = [
                entry for entry in ResearchLog.read(self.path)
                if entry.get("run_id") == self.run_id
            ]
        target = destination or self.path.with_suffix(".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: Path) -> Iterator[dict[str, Any]]:
        """读回一份历史日志（用于复现或对比两次采集）。"""
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def new_run_id() -> str:
    """形如 ``20250101T101500-3f9a``，可读且可排序。"""
    stamp = datetime.now(CST).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


def _jsonable(value: Any) -> Any:
    """只放行可 JSON 序列化的数据。

    刻意拒绝任意对象：内部对象（Path、datetime、settings）需要显式转换，
    避免不知不觉把不可复现的状态写进日志。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
