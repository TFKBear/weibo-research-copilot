"""数据使用声明（知情同意的前置闸口，模块 C）。

工具的立场：导出不是「点一下下载」，而是一次**对研究伦理的确认动作**。
因此确认结果会被记入研究日志（含时间、声明版本、确认方式），
这一条在论文的方法/伦理部分可以直接引用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..errors import ConsentNotGiven
from ..weibo.parsing import CST

STATEMENT_VERSION = "1.0"

STATEMENT_LINES: tuple[str, ...] = (
    "本数据仅用于学术研究目的，不用于商业用途。",
    "我不会尝试重新识别数据中的任何个体，也不会将数据与外部身份信息关联。",
    "我不会公开传播原始文本到可被搜索引擎索引的场所；引用时做匿名化处理。",
    "我理解本数据来自公开可访问的微博内容，但公开可访问不等于可自由再分发。",
    "我将遵守所在机构与期刊关于人类受试者数据 / 网络数据的伦理要求。",
    "我理解导出数据已移除用户 ID、昵称、主页链接与原始地区，并保留省级聚合。",
)


def render_statement() -> str:
    lines = ["数据使用声明（导出的前置条件）", "=" * 40]
    for index, line in enumerate(STATEMENT_LINES, start=1):
        lines.append(f"{index}. {line}")
    return "\n".join(lines)


@dataclass(slots=True)
class ConsentRecord:
    """一次确认的留痕。"""

    stated_at: str
    statement_version: str
    confirmed_by: str = "user_cli"
    confirmed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "stated_at": self.stated_at,
            "statement_version": self.statement_version,
            "confirmed_by": self.confirmed_by,
        }


def confirm(confirmed: bool, *, source: str = "user_cli") -> ConsentRecord:
    """记录确认动作。``confirmed=False`` 时抛 ``ConsentNotGiven``。"""
    if not confirmed:
        raise ConsentNotGiven(
            detail=f"source={source}, statement_version={STATEMENT_VERSION}"
        )
    return ConsentRecord(
        stated_at=datetime.now(CST).isoformat(timespec="seconds"),
        statement_version=STATEMENT_VERSION,
        confirmed_by=source,
    )
