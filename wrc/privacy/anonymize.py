"""匿名化出口（模块 C）。

核心设计判断：**匿名化发生在导出环节，而不是入库环节。**

理由：
- 入库就抹掉身份，P1 的「用户社群聚类」无法把同一用户的多条评论聚合起来，
  整个用户级分析路径会被提前焊死；
- 导出环节做「稳定伪匿名」，既能做用户级分析，又保证交付出去的数据
  无法反向识别个体。伪 ID 用本地盐做 HMAC，盐不出本地、不入数据文件，
  因此同一用户在两次导出中 ID 稳定，但外部无法用微博 ID 反查。

同时实现小群体抑制（k-匿名）：任何交叉表中样本量 <5 的单元格不展示，
避免通过聚类 × 地域的交叉表反推到具体个人。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: 导出时一律移除的原始字段（模块 C 明确列举）。
IDENTIFIER_FIELDS: tuple[str, ...] = (
    "user_id",
    "screen_name",
    "profile_url",
    "location",
)

#: 伪 ID 的盐长度（字节）。
SALT_BYTES = 32
#: 伪 ID 输出长度（hex 字符）。64 bit 空间，碰撞概率对十万级样本可忽略。
PSEUDO_ID_LEN = 16
#: k-匿名的抑制阈值。
MIN_GROUP_SIZE = 5

#: 中国省级行政区（含港澳台与「海外/其他」）。
#: 只做省级聚合：需求明确保留省级、移除原始字段。
PROVINCES: tuple[str, ...] = (
    "内蒙古", "黑龙江", "香港", "澳门",
    "北京", "天津", "河北", "山西", "辽宁", "吉林",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "台湾",
    "海外", "其他",
)


class PseudoIdHasher:
    """基于本地盐的稳定伪匿名器。

    Args:
        salt_path: 盐文件路径。不存在时自动生成，
            并在同一次运行中复用，保证导出内 ID 一致。
    """

    def __init__(self, salt_path: Path) -> None:
        self.salt_path = Path(salt_path)
        self._salt = self._load_or_create_salt()

    def _load_or_create_salt(self) -> bytes:
        path = self.salt_path
        if path.exists():
            data = path.read_bytes()
            if len(data) >= SALT_BYTES:
                return data
        path.parent.mkdir(parents=True, exist_ok=True)
        salt = secrets.token_bytes(SALT_BYTES)
        path.write_bytes(salt)
        try:
            os.chmod(path, 0o600)
        except OSError:  # pragma: no cover - Windows 上语义有限
            pass
        return salt

    def derive(self, raw_value: str | int | None) -> str:
        """把原始标识映射为稳定伪 ID。空值返回空串。"""
        if raw_value in (None, ""):
            return ""
        digest = hmac.new(
            self._salt, str(raw_value).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return digest[:PSEUDO_ID_LEN]

    def __repr__(self) -> str:  # pragma: no cover - 不泄露盐
        return f"PseudoIdHasher(salt_path={self.salt_path!s})"


def extract_province(location: str | None) -> str:
    """把原始地区字符串聚合成省级。

    微博的地区字段形如 ``北京 朝阳区``、``广东 深圳``、``海外 美国``、``其他``。

    >>> extract_province("北京 朝阳区")
    '北京'
    >>> extract_province("广东 深圳")
    '广东'
    >>> extract_province("")
    '未知'
    """
    text = (location or "").strip()
    if not text:
        return "未知"
    for province in PROVINCES:
        if text.startswith(province):
            return province
    # 「海外」「其他」可能出现在尾部（如「美国 海外」），再做一次包含判断
    for province in ("海外", "其他"):
        if province in text:
            return province
    return "未知"


def anonymize_comment(
    row: dict[str, Any],
    hasher: PseudoIdHasher,
    *,
    keep_province: bool = True,
) -> dict[str, Any]:
    """对一行评论执行导出级匿名化。

    输出字段是**显式白名单**：任何未列出的字段都不会被带出去，
    这样即使以后底层表加了新字段，也不会意外泄露。
    """
    out: dict[str, Any] = {
        "cid": row.get("cid"),
        "post_mid": row.get("post_mid"),
        "parent_cid": row.get("parent_cid"),
        "level": row.get("level"),
        "user_pseudo_id": hasher.derive(row.get("user_id")),
        "text": row.get("cleaned_text") if "cleaned_text" in row else row.get("text_raw"),
        "text_raw": row.get("text_raw"),
        "created_at": row.get("created_at"),
        "created_at_raw": row.get("created_at_raw"),
        "created_at_precision": row.get("created_at_precision"),
        "like_count": row.get("like_count"),
        "reply_count": row.get("reply_count"),
        "followers_count": row.get("followers_count"),
    }
    if keep_province:
        out["province"] = extract_province(row.get("location"))
    return out


def anonymize_post(row: dict[str, Any], hasher: PseudoIdHasher) -> dict[str, Any]:
    """帖子层的匿名化。保留 mid（帖子本身是公开发布对象，属于分析单位）。"""
    return {
        "mid": row.get("mid"),
        "user_pseudo_id": hasher.derive(row.get("user_id")),
        "text_raw": row.get("text_raw"),
        "created_at": row.get("created_at"),
        "created_at_raw": row.get("created_at_raw"),
        "created_at_precision": row.get("created_at_precision"),
        "reposts_count": row.get("reposts_count"),
        "comments_count": row.get("comments_count"),
        "attitudes_count": row.get("attitudes_count"),
        "is_long_text": row.get("is_long_text"),
    }


def assert_no_identifiers(rows: Iterable[dict[str, Any]]) -> None:
    """导出前的最后一道闸：确认没有任何可识别字段漏出去。

    宁可让导出失败，也不允许带 ID 的学术数据被交出去 —— 这是模块 C 的底线。
    """
    leaked: dict[str, int] = {}
    for row in rows:
        for field_name in IDENTIFIER_FIELDS:
            if field_name in row:
                leaked[field_name] = leaked.get(field_name, 0) + 1
    if leaked:
        raise AssertionError(
            "导出数据仍含可识别字段，已阻止导出：" + ", ".join(sorted(leaked))
        )


# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SuppressionResult:
    """小群体抑制的结果。"""

    table: dict[str, dict[str, int | None]]
    suppressed_cells: list[tuple[str, str, int]] = field(default_factory=list)
    suppressed_groups: list[tuple[str, int]] = field(default_factory=list)
    min_n: int = MIN_GROUP_SIZE

    @property
    def notes(self) -> list[str]:
        notes: list[str] = []
        for group, n in self.suppressed_groups:
            notes.append(f"分组「{group}」样本量 {n} < {self.min_n}，整组不予展示。")
        if self.suppressed_cells:
            notes.append(
                f"另有 {len(self.suppressed_cells)} 个交叉单元格样本量 < {self.min_n}，"
                "已置为不展示。"
            )
        return notes


def suppress_small_groups(
    crosstab: dict[str, dict[str, int]],
    *,
    min_n: int = MIN_GROUP_SIZE,
) -> SuppressionResult:
    """对「分组 × 属性」交叉表做 k-匿名抑制。

    Args:
        crosstab: ``{分组标签: {属性标签: 计数}}``，例如
            ``{"话语簇1": {"北京": 12, "上海": 2}}``。
        min_n: 最小可展示样本量，默认 5（模块 C 的规定）。

    行为：
        - 分组总样本量 < ``min_n`` → 整组不出；
        - 单元格计数 < ``min_n`` → 该单元格置为 ``None``（不展示），
          而不是显示 0 或把 2 并入其他类目。
    """
    table: dict[str, dict[str, int | None]] = {}
    suppressed_cells: list[tuple[str, str, int]] = []
    suppressed_groups: list[tuple[str, int]] = []

    for group, cells in crosstab.items():
        total = sum(int(v) for v in cells.values())
        if total < min_n:
            suppressed_groups.append((group, total))
            continue
        rendered: dict[str, int | None] = {}
        for attribute, count in cells.items():
            count_int = int(count)
            if count_int < min_n:
                rendered[attribute] = None
                suppressed_cells.append((group, attribute, count_int))
            else:
                rendered[attribute] = count_int
        table[group] = rendered

    return SuppressionResult(
        table=table,
        suppressed_cells=suppressed_cells,
        suppressed_groups=suppressed_groups,
        min_n=min_n,
    )


PRIVACY_NOTES: tuple[str, ...] = (
    "导出数据不含 user_id / screen_name / profile_url / location 原始值。",
    "用户身份以 HMAC(本地盐, user_id) 的稳定伪 ID 表示，盐不出本地、不入数据文件。",
    "地区字段聚合为省级；原始地区字符串已移除。",
    f"交叉表中样本量 < {MIN_GROUP_SIZE} 的单元格不予展示（k-匿名抑制）。",
)
