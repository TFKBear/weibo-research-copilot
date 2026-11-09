"""AI 判定层（纯函数 + 一个编排函数）。

这里的所有判定逻辑都不依赖模型：输入向量、输出结论。
这样即使没有模型，我们也能用构造数据把判定规则测透；
而换模型、换阈值时，判定行为是可预期的。

两类判定：

1. **营销话术**：与可读的种子模板比对，输出"最接近哪条模板、相似度多少"。
2. **协同行为**：在数据集内部找互相高度相似的簇。这一条不需要任何种子，
   因为它找的是"同一段文案被反复投放"这个现象本身。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..errors import AIUnavailable
from .seeds import MARKETING_SEEDS

#: 默认阈值。
#:
#: ⚠ 这不是凭经验估的，是标定出来的（tools/calibrate_ai.py）：
#: 12 条营销正样本与最近模板的相似度落在 0.644–0.844，
#: 80 条真实评论落在 0–0.606，中间有明显空隙，取 0.62。
#:
#: 为什么不能凭直觉设成 0.8：bge 系列在中文短句上的余弦相似度是"压缩"的，
#: 两句都在推销、措辞完全不同的话也只有 0.6 上下。设 0.8 会一条都抓不到，
#: 而界面上只会显示"AI 识别 0 条"——功能看起来正常，实际完全失效。
DEFAULT_MARKETING_THRESHOLD = 0.62
DEFAULT_COORDINATED_THRESHOLD = 0.85
#: 至少多少条互相似才算一个"协同行为簇"。
#: 2 条相同太容易是巧合；3 条起才有"投放"的意味。
DEFAULT_MIN_CLUSTER_SIZE = 3


@dataclass(slots=True)
class MarketingHit:
    score: float
    seed: str

    @property
    def reason(self) -> str:
        return f"ai_marketing:{self.score:.2f}"


@dataclass(slots=True)
class AIAnalysis:
    """一次 AI 判定的完整结果，可整体写进研究日志。"""

    enabled: bool = False
    model_name: str = ""
    embedding_dim: int = 0
    analysed: int = 0
    marketing: dict[str, MarketingHit] = field(default_factory=dict)
    coordinated: dict[str, int] = field(default_factory=dict)
    cluster_count: int = 0
    marketing_threshold: float = DEFAULT_MARKETING_THRESHOLD
    coordinated_threshold: float = DEFAULT_COORDINATED_THRESHOLD
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def flags_for(self, cid: str) -> list[str]:
        """把某条评论的 AI 判定转成清洗步骤能用的理由字符串。"""
        reasons: list[str] = []
        hit = self.marketing.get(cid)
        if hit is not None:
            reasons.append(hit.reason)
        size = self.coordinated.get(cid)
        if size:
            reasons.append(f"ai_coordinated:簇内{size}条")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "analysed": self.analysed,
            "marketing_flagged": len(self.marketing),
            "coordinated_flagged": len(self.coordinated),
            "cluster_count": self.cluster_count,
            "marketing_threshold": self.marketing_threshold,
            "coordinated_threshold": self.coordinated_threshold,
            "min_cluster_size": self.min_cluster_size,
            "elapsed_s": self.elapsed_s,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
def best_seed_match(
    vectors: Any, seed_vectors: Any, *, threshold: float
) -> list[MarketingHit | None]:
    """为每条向量找出最接近的种子模板。

    Args:
        vectors: (n, dim) 已归一化。
        seed_vectors: (k, dim) 已归一化。
        threshold: 相似度阈值。

    Returns:
        长度 n 的列表，未超过阈值为 None。
    """
    np = _numpy()
    if len(vectors) == 0 or len(seed_vectors) == 0:
        return [None] * len(vectors)

    scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(
        seed_vectors, dtype=np.float32
    ).T
    out: list[MarketingHit | None] = []
    for row in range(scores.shape[0]):
        column = int(np.argmax(scores[row]))
        score = float(scores[row, column])
        if score >= threshold:
            out.append(MarketingHit(score=round(score, 3), seed=MARKETING_SEEDS[column]))
        else:
            out.append(None)
    return out


def cluster_similar_texts(
    vectors: Any,
    *,
    threshold: float,
    min_cluster_size: int,
    block: int = 256,
) -> tuple[dict[int, int], int]:
    """在数据集内部找高相似文本簇。

    用并查集把"相似度 ≥ 阈值"的样本连起来，得到连通分量。
    分块计算相似度，避免一次分配 n×n 的大矩阵。

    Returns:
        (每一行所属簇的大小, 达到最小规模的簇数量)。
        单点簇大小为 1（调用方据此判断是否算协同行为）。
    """
    np = _numpy()
    n = len(vectors)
    if n == 0:
        return {}, 0

    matrix = np.asarray(vectors, dtype=np.float32)
    parent = list(range(n))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = matrix[start:stop] @ matrix.T
        rows, cols = np.nonzero(sims >= threshold)
        for offset, other in zip(rows.tolist(), cols.tolist()):
            left = start + offset
            if left == other:
                continue
            union(left, other)

    sizes: dict[int, int] = {}
    for index in range(n):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1

    per_row = {index: sizes[find(index)] for index in range(n)}
    cluster_count = sum(1 for size in sizes.values() if size >= min_cluster_size)
    return per_row, cluster_count


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def analyse(
    rows: Sequence[dict[str, Any]],
    embedder: Any,
    *,
    marketing_threshold: float = DEFAULT_MARKETING_THRESHOLD,
    coordinated_threshold: float = DEFAULT_COORDINATED_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> AIAnalysis:
    """对一批评论行做 AI 判定。

    Args:
        rows: 每条需含 ``cid`` 与 ``text_raw``。
        embedder: 具备 ``encode(texts) -> EmbeddingResult`` 的对象。
    """
    started = time.monotonic()
    result = AIAnalysis(
        enabled=True,
        marketing_threshold=marketing_threshold,
        coordinated_threshold=coordinated_threshold,
        min_cluster_size=min_cluster_size,
    )
    if not rows:
        result.notes.append("没有可分析的评论。")
        return result

    cids = [str(row.get("cid")) for row in rows]
    texts = [str(row.get("text_raw") or "") for row in rows]

    encoded = embedder.encode(texts, on_progress=on_progress)
    result.model_name = getattr(encoded, "model_name", "")
    result.embedding_dim = int(getattr(encoded, "dim", 0))
    result.analysed = len(texts)
    vectors = encoded.vectors

    seeds = embedder.encode(list(MARKETING_SEEDS))
    hits = best_seed_match(
        vectors, seeds.vectors, threshold=marketing_threshold
    )
    for cid, hit in zip(cids, hits):
        if hit is not None:
            result.marketing[cid] = hit

    sizes, cluster_count = cluster_similar_texts(
        vectors,
        threshold=coordinated_threshold,
        min_cluster_size=min_cluster_size,
    )
    result.cluster_count = cluster_count
    for index, cid in enumerate(cids):
        size = sizes.get(index, 1)
        if size >= min_cluster_size:
            result.coordinated[cid] = size

    result.elapsed_s = round(time.monotonic() - started, 2)
    result.notes.append(
        f"句向量模型：{result.model_name}（{result.embedding_dim} 维）；"
        f"营销阈值 {marketing_threshold}，协同行为阈值 {coordinated_threshold}、"
        f"最小簇规模 {min_cluster_size}。"
    )
    if result.marketing:
        result.notes.append(
            f"命中营销模板 {len(result.marketing)} 条；"
            "每条都记录了相似度与最接近的模板，可在导出数据中复核。"
        )
    if result.coordinated:
        result.notes.append(
            f"发现 {cluster_count} 个高相似簇，涉及 {len(result.coordinated)} 条评论；"
            "相似度簇不等于水军，需研究者人工复核。"
        )
    return result


def _numpy() -> Any:
    try:
        import numpy as np

        return np
    except Exception as exc:  # noqa: BLE001
        raise AIUnavailable(detail=f"numpy 不可用：{exc}") from exc
