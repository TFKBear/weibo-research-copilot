"""二级评论（回复）关系分析（P3 第 2 项）。

要回答的问题：这条帖子的讨论是**多点分散**，还是**单条评论引爆、众人围观**？
这两种形态在舆论研究里含义完全不同，而仅看评论总数是分不出来的。

数据来源是评论接口返回的 ``parent_cid`` / ``level``。有三个必须交代的现实约束：

1. **二级评论默认不采集**（产品决策 001）。所以本模块在多数会话上会得到空数据，
   这时它明确说"本次未采集二级评论"并给出怎么开启，而不是输出一张空图。
2. **清洗会打断回复链**。父评论被判定为垃圾而移除后，它的子评论就成了孤儿。
   本模块把孤儿挂到该帖的虚拟根上并**单独计数**，让研究者知道结构有多少是被清洗
   改写的——静默重挂会让人以为"讨论本来就是这样"。
3. **只有公开的结构关系，没有隐私**。节点只带层级与帖号，不带任何用户标识
   （伪 ID 也不带：一张能对应到"某人在某贴连续回复 20 条"的图就是可识别的）。

小结构同样受抑制：总节点数小于 ``min_n`` 的讨论串不在图里出现、也不列举节点，
只报"存在 2 个不足 5 条的讨论串"。这与交叉表的 k-匿名规则保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..privacy.anonymize import MIN_GROUP_SIZE

#: 一张图最多画几个讨论串，再多就看不清了。
MAX_THREADS_DRAWN = 6
#: 单个讨论串里最多画多少节点（防止一个 300 条的大串糊成一团）。
MAX_NODES_PER_THREAD = 40
#: 最大讨论串占总回复数超过这个比例时，提示"单点引爆"。
CONCENTRATION_THRESHOLD = 0.5
#: 判定"单点引爆"所需的最小回复数，低于它谈集中度没有意义。
CONCENTRATION_MIN_REPLIES = 20


@dataclass(slots=True)
class Thread:
    """一个讨论串：一条帖下的根评论及其回复。"""

    root_cid: str
    size: int
    depth: int
    #: ``[(cid, parent_cid, level, depth, text_preview)]``，供画图用
    nodes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cid": self.root_cid,
            "size": self.size,
            "depth": self.depth,
            "nodes": [
                {key: value for key, value in node.items() if key != "cid"}
                for node in self.nodes
            ],
        }


@dataclass(slots=True)
class ReplyStructureResult:
    available: bool = False
    n_comments: int = 0
    n_roots: int = 0
    n_replies: int = 0
    n_orphans: int = 0
    n_threads: int = 0
    roots_with_replies: int = 0
    max_depth: int = 0
    mean_replies_per_root: float = 0.0
    top_threads: list[Thread] = field(default_factory=list)
    suppressed_threads: int = 0
    concentration: float = 0.0
    min_n: int = MIN_GROUP_SIZE
    warnings: list[str] = field(default_factory=list)
    method: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def shape_label(self) -> str:
        if not self.available:
            return "无二级评论数据"
        if self.n_replies == 0:
            return "仅有根评论（无人回复）"
        if self.concentration >= CONCENTRATION_THRESHOLD:
            return "单点引爆型"
        if self.max_depth >= 3:
            return "多层对话型"
        return "多点分散型"

    def render(self) -> str:
        lines = ["二级评论回复结构", "=" * 62]
        if not self.available:
            lines.append("本次数据中没有二级评论，无法分析回复结构。")
            if self.n_comments:
                lines.append(
                    f"（本次会话有 {self.n_comments} 条评论，但没有任何 level≥2 的记录。"
                    "这有两种可能：①采集时未开启二级评论；②开启了但该批帖子确实无人回复。"
                    "评论接口不返回「是否采集了二级评论」的标记，所以工具无法区分这两种情况——"
                    "请回想采集设置，不要默认「没人回复」。）"
                )
            for warning in self.warnings:
                lines.append(f"⚠ {warning}")
            lines.append("")
            lines.append(
                "说明：二级评论默认不采集（产品决策 001，避免在研究者未确认的情况下"
                "扩大采集范围）。如需分析回复关系，请在采集时勾选「采集二级评论」。"
            )
            return "\n".join(lines)

        lines.append(f"- 评论总数（清洗后保留）：{self.n_comments}")
        lines.append(f"- 根评论：{self.n_roots} 条；二级及以下回复：{self.n_replies} 条")
        lines.append(f"- 产生过回复的根评论：{self.roots_with_replies} 条")
        lines.append(
            f"- 有回复的讨论串：{self.n_threads} 个；"
            f"最大回复层级：{self.max_depth} 层（根评论以外）"
        )
        if self.n_roots:
            lines.append(f"- 平均每条根评论获得回复：{self.mean_replies_per_root:.2f} 条")
        if self.n_orphans:
            lines.append(
                f"- 孤儿回复（父评论已在清洗中移除）：{self.n_orphans} 条，"
                "已挂到该帖虚拟根下"
            )
        lines.append("")
        lines.append(f"形态判断：**{self.shape_label()}**")
        lines.append("")

        if self.top_threads:
            lines.append(f"规模最大的讨论串（仅列节点数 ≥{self.min_n} 的串）")
            lines.append(f"{'讨论串':<10}{'节点数':>7}{'深度':>6}  代表文本")
            lines.append("-" * 62)
            for index, thread in enumerate(self.top_threads, start=1):
                preview = first_text(thread)
                lines.append(
                    f"{'#' + str(index):<10}{thread.size:>7}{thread.depth:>6}  {preview}"
                )
            lines.append("")
        if self.suppressed_threads:
            lines.append(
                f"注：另有 {self.suppressed_threads} 个讨论串节点数 <{self.min_n}，"
                "不予展示（小结构可能对应到具体的个别对话）。"
            )
        if self.n_replies >= CONCENTRATION_MIN_REPLIES:
            lines.append(
                f"注：最大讨论串占全部回复的 {self.concentration:.0%}。"
                + (
                    "集中度较高，讨论更像「一条评论引爆、众人跟帖」，"
                    "而不是多点分散的意见交流。"
                    if self.concentration >= CONCENTRATION_THRESHOLD
                    else "集中度不高，讨论相对分散。"
                )
            )
        lines.append("")
        for warning in self.warnings:
            lines.append(f"⚠ {warning}")
        lines.append(
            "说明：结构只用到层级与父子引用，不含任何用户标识；"
            "同一条串内的多次回复不再区分是谁发的。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "replies",
            "available": self.available,
            "n_comments": self.n_comments,
            "n_roots": self.n_roots,
            "n_replies": self.n_replies,
            "n_orphans": self.n_orphans,
            "n_threads": self.n_threads,
            "roots_with_replies": self.roots_with_replies,
            "max_depth": self.max_depth,
            "mean_replies_per_root": round(self.mean_replies_per_root, 4),
            "shape": self.shape_label(),
            "top_threads": [thread.to_dict() for thread in self.top_threads],
            "suppressed_threads": self.suppressed_threads,
            "concentration": round(self.concentration, 4),
            "min_n": self.min_n,
            "warnings": self.warnings,
            "method": self.method,
        }


def first_text(thread: Thread) -> str:
    """讨论串里第一个节点的文本预览（通常是根评论）。"""
    for node in thread.nodes:
        if node.get("text_preview"):
            return str(node["text_preview"])
    return "（无文本）"


# --------------------------------------------------------------------------- #
def build_reply_structure(
    rows: Iterable[dict[str, Any]],
    *,
    min_n: int = MIN_GROUP_SIZE,
    max_threads: int = MAX_THREADS_DRAWN,
    max_nodes: int = MAX_NODES_PER_THREAD,
) -> ReplyStructureResult:
    """从评论行重建回复结构。

    Args:
        rows: 含 ``cid`` / ``parent_cid`` / ``level``（可选 ``post_mid``、文本）的行。
    """
    rows = list(rows)
    by_cid: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("cid") or "")
        if cid:
            by_cid[cid] = row

    children: dict[str, list[str]] = {}
    roots: list[str] = []
    orphans = 0
    for cid, row in by_cid.items():
        parent = row.get("parent_cid")
        level = int(row.get("level") or 1)
        if level <= 1 or not parent:
            roots.append(cid)
            continue
        parent = str(parent)
        if parent in by_cid:
            children.setdefault(parent, []).append(cid)
        else:
            # 父评论被清洗掉了：当作该帖下的孤儿，不静默接到别的根上
            orphans += 1
            roots.append(cid)

    threads: list[Thread] = []
    max_depth_overall = 0
    for root_cid in roots:
        nodes, depth = _walk(root_cid, by_cid, children)
        max_depth_overall = max(max_depth_overall, depth)
        if len(nodes) <= 1 and not children.get(root_cid):
            continue  # 没有任何回复，不构成讨论串
        threads.append(Thread(
            root_cid=root_cid,
            size=len(nodes),
            depth=depth,
            nodes=_trim(nodes, max_nodes),
        ))

    threads.sort(key=lambda thread: -thread.size)
    kept = [thread for thread in threads if thread.size >= min_n]
    suppressed = len(threads) - len(kept)

    replies = sum(thread.size - 1 for thread in threads)
    concentration = (kept[0].size - 1) / replies if (kept and replies) else 0.0
    has_replies = any(int(row.get("level") or 1) >= 2 for row in by_cid.values())

    warnings = _warnings(
        available=bool(by_cid),
        n_roots=len(roots),
        replies=replies,
        orphans=orphans,
        suppressed=suppressed,
        min_n=min_n,
    )

    return ReplyStructureResult(
        available=has_replies,
        n_comments=len(by_cid),
        n_roots=len(roots) - orphans,
        n_replies=replies,
        n_orphans=orphans,
        n_threads=len(threads),
        roots_with_replies=len(threads),
        max_depth=max_depth_overall,
        mean_replies_per_root=(replies / len(roots)) if roots else 0.0,
        top_threads=kept[:max_threads],
        suppressed_threads=suppressed,
        concentration=concentration,
        min_n=min_n,
        warnings=warnings,
        method={
            "module": "replies",
            "method": "按 parent_cid 重建父子树，深度优先展开",
            "min_n": min_n,
            "max_depth": max_depth_overall,
            "concentration_threshold": CONCENTRATION_THRESHOLD,
            "orphan_policy": "父评论缺失的回复挂到该帖虚拟根，并单独计数",
            "identifier_policy": "只使用层级与父子引用，不含用户标识",
        },
    )


# --------------------------------------------------------------------------- #
def _walk(
    root_cid: str, by_cid: dict[str, dict[str, Any]], children: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], int]:
    """广度优先展开一棵树，返回节点列表与最大深度。"""
    nodes: list[dict[str, Any]] = []
    queue: list[tuple[str, str | None, int]] = [(root_cid, None, 0)]
    seen: set[str] = set()
    depth = 0
    while queue:
        cid, parent, level = queue.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        row = by_cid.get(cid, {})
        depth = max(depth, level)
        nodes.append({
            "cid": cid,
            "parent_cid": parent,
            "level": level,
            "text_preview": _preview(row),
        })
        for child in children.get(cid, []):
            queue.append((child, cid, level + 1))
    return nodes, depth


def _trim(nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """超大讨论串只保留先展开的若干节点，避免图上糊成一团。

    截断后仍保留 ``parent_cid``，所以画图时父子线不会指错，
    只是有些中间节点不显示——这一点在图上以"…"注明。
    """
    if len(nodes) <= limit:
        return nodes
    return nodes[:limit]


def _preview(row: dict[str, Any], width: int = 24) -> str:
    text = str(row.get("cleaned_text") or row.get("text_raw") or "").strip()
    text = " ".join(text.split())
    return text[:width] + "…" if len(text) > width else text


def _warnings(
    *,
    available: bool,
    n_roots: int,
    replies: int,
    orphans: int,
    suppressed: int,
    min_n: int,
) -> list[str]:
    warnings: list[str] = []
    if not available:
        warnings.append(
            "评论行里没有任何 level≥2 的记录：既可能是未采集二级评论，"
            "也可能是该批帖子确实无人回复，数据本身无法区分。"
        )
        return warnings
    if n_roots == 0:
        warnings.append("没有识别出根评论，数据可能异常。")
    if replies and replies < 10:
        warnings.append(
            f"回复只有 {replies} 条，结构判断极不稳定，建议扩大采集后再解读。"
        )
    if orphans:
        warnings.append(
            f"{orphans} 条回复的父评论在清洗中被移除。清洗替这些回复"
            "「重新挑了父节点」，所以本图的层数比原始数据浅，"
            "论文中应说明这一处理。"
        )
    if suppressed:
        warnings.append(
            f"{suppressed} 个讨论串节点数 <{min_n}，已按小结构抑制规则从列表与图中移除。"
        )
    return warnings
