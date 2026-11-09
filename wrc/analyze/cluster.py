"""评论级话语聚类（模块 D2 之一）。

目的不是"把文本分堆"，而是回答社科问题：**围绕这个话题，主要存在哪几套说法？**

三个设计选择：

1. **K 不预判**。先用轮廓系数在 2..k_max 之间扫描给出建议区间，再由研究者确认。
   强制用户先定 K，等于把自己的先验偷偷塞进研究设计。
2. **关键词要能反映"这一簇区别于其他簇的用语"**，而不是"这一簇最常用的词"——
   后者在每个簇里都会出现"小米""汽车"。所以用类内频率 / 全局频率的比值。
3. **代表性评论取距质心最近的几条**，因为它最能代表这个簇的"平均话语"，
   与随机抽样的意义不同，这一点写进方法卡片。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

KEYWORD_TOP_N = 6
REPRESENTATIVE_TOP_N = 3
#: 轮廓系数低于此值就提示"簇边界不清"。0.25 是聚类文献里通行的"弱结构"门槛。
WEAK_STRUCTURE = 0.25

#: 中文停用词（精简版）。只去掉真正没有区分力的词——
#: 停用词表越长，关键词越容易被"抽干"，研究者反而看不出这一簇在说什么。
STOPWORDS: frozenset[str] = frozenset(
    """
的 了 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己
这 那 这个 那个 这些 那些 他 她 它 我们 你们 他们 什么 怎么 为什么 因为 所以 但是
而且 如果 虽然 可以 应该 还是 就是 这样 那样 已经 现在 时候 问题 觉得 知道 认为
真的 确实 其实 可能 或者 以及 不过 然后 还有 一下 一直 比较 非常 特别 感觉
哈哈 哈哈哈 啊 吧 呢 吗 哦 嗯 呀 哈 呗 啦 嘛
""".split()
)

_EMOJI_TOKEN = re.compile(r"\[[^\[\]\s]{1,12}\]")
_KEEP = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]+")


@dataclass(slots=True)
class ClusterResult:
    k: int = 0
    labels: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    cids: list[str] = field(default_factory=list)
    sizes: dict[int, int] = field(default_factory=dict)
    keywords: dict[int, list[str]] = field(default_factory=dict)
    representatives: dict[int, list[str]] = field(default_factory=dict)
    silhouette_curve: list[tuple[int, float]] = field(default_factory=list)
    silhouette: float | None = None
    suggested_k: int | None = None
    method: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = ["话语聚类结果", "=" * 62]
        lines.append(f"样本 {len(self.texts)} 条，K={self.k}，轮廓系数 {self.silhouette}")
        if self.suggested_k is not None:
            lines.append(f"数据建议 K={self.suggested_k}（你在运行时确认的 K={self.k}）")
        if self.silhouette is not None and self.silhouette < WEAK_STRUCTURE:
            lines.append("")
            lines.append(
                f"⚠ 轮廓系数 {self.silhouette} 低于 {WEAK_STRUCTURE}，"
                "说明各簇的边界很不清晰。"
            )
            lines.append(
                "  常见原因：样本量太小、或这个话题本身没有明显的话语分野。"
                "这种情况下簇的标签不宜直接写进论文，"
                "建议先扩大样本（300 条以上）或换一个争议更集中的话题。"
            )
        lines.append("")
        for cluster_id in sorted(self.sizes, key=lambda x: -self.sizes[x]):
            words = "、".join(self.keywords.get(cluster_id, [])) or "（无显著关键词）"
            lines.append(f"簇 {cluster_id}｜{self.sizes[cluster_id]} 条｜关键词：{words}")
            for index, text in enumerate(self.representatives.get(cluster_id, []), start=1):
                lines.append(f"    {index}. {text[:60]}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "sizes": self.sizes,
            "keywords": {str(k): v for k, v in self.keywords.items()},
            "representatives": {str(k): v for k, v in self.representatives.items()},
            "silhouette": self.silhouette,
            "silhouette_curve": self.silhouette_curve,
            "suggested_k": self.suggested_k,
            "method": self.method,
            "assignments": [
                {"cid": cid, "cluster": label}
                for cid, label in zip(self.cids, self.labels)
            ],
        }


# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """分词并过滤停用词、单字与纯数字。"""
    import jieba

    cleaned = _EMOJI_TOKEN.sub(" ", text or "")
    tokens: list[str] = []
    for raw in jieba.lcut(cleaned):
        token = raw.strip()
        if not token or token in STOPWORDS:
            continue
        if not _KEEP.fullmatch(token):
            continue
        if len(token) < 2 and not token.isascii():
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def suggest_k(vectors: Any, *, k_min: int = 2, k_max: int | None = None,
              seed: int = 20240101) -> list[tuple[int, float]]:
    """用轮廓系数扫描 K，返回 [(k, score)]。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(vectors)
    upper = min(k_max or 10, n - 1)
    curve: list[tuple[int, float]] = []
    for k in range(max(2, k_min), upper + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(vectors, labels, metric="cosine"))
        except Exception:  # noqa: BLE001
            continue
        curve.append((k, round(score, 4)))
    return curve


def cluster_discourse(
    rows: Sequence[dict[str, Any]],
    *,
    embedder: Any,
    k: int | None = None,
    k_max: int = 10,
    seed: int = 20240101,
    emit: Callable[[str, str], None] | None = None,
) -> ClusterResult:
    """对清洗后的评论做话语聚类。

    Args:
        k: 用户确认的聚类数；为 None 时用轮廓系数建议值。
    """
    notify = emit or (lambda _level, _message: None)
    from sklearn.cluster import KMeans

    items = [
        (str(row.get("cid")), str(row.get("cleaned_text") or row.get("text_raw") or "").strip())
        for row in rows
    ]
    items = [(cid, text) for cid, text in items if text]
    if len(items) < 3:
        raise ValueError("样本少于 3 条，无法聚类")

    cids = [cid for cid, _ in items]
    texts = [text for _, text in items]

    notify("info", f"编码 {len(texts)} 条文本为句向量 …")
    encoded = embedder.encode(texts)
    vectors = encoded.vectors

    notify("info", "扫描最优 K（轮廓系数）…")
    curve = suggest_k(vectors, k_max=k_max, seed=seed)
    suggested = max(curve, key=lambda pair: pair[1])[0] if curve else 2
    chosen = k or suggested
    chosen = max(2, min(chosen, len(texts) - 1))
    notify("info", f"  建议 K={suggested}，使用 K={chosen}")

    model = KMeans(n_clusters=chosen, random_state=seed, n_init=10)
    labels = model.fit_predict(vectors)
    silhouette = None
    if len(set(labels)) > 1:
        from sklearn.metrics import silhouette_score

        try:
            silhouette = round(float(silhouette_score(vectors, labels, metric="cosine")), 4)
        except Exception:  # noqa: BLE001
            silhouette = None

    # ---- 关键词：类内频率 / 全局频率 ----
    cluster_tokens: dict[int, Counter[str]] = defaultdict(Counter)
    global_tokens: Counter[str] = Counter()
    for label, text in zip(labels, texts):
        tokens = tokenize(text)
        cluster_tokens[int(label)].update(tokens)
        global_tokens.update(tokens)

    total = sum(global_tokens.values()) or 1
    sizes = Counter(int(x) for x in labels)
    keywords: dict[int, list[str]] = {}
    for cluster_id, counter in cluster_tokens.items():
        cluster_total = sum(counter.values()) or 1
        share = sizes[cluster_id] / len(texts)
        scored: list[tuple[float, str]] = []
        for token, count in counter.items():
            if count < 2:
                continue
            tf_cluster = count / cluster_total
            tf_global = global_tokens[token] / total
            # 除以 share 是为了纠正"大簇天然词多"的偏差
            lift = (tf_cluster / (tf_global * share + 1e-9))
            scored.append((math.log1p(lift) * math.log1p(count), token))
        scored.sort(reverse=True)
        keywords[cluster_id] = [token for _, token in scored[:KEYWORD_TOP_N]]

    # ---- 代表性评论：距质心最近 ----
    representatives: dict[int, list[str]] = {}
    centroids = model.cluster_centers_
    for cluster_id in sizes:
        members = [index for index, label in enumerate(labels) if int(label) == cluster_id]
        if not members:
            continue
        centroid = centroids[cluster_id]
        distances = sorted(
            members,
            key=lambda index: float(
                ((vectors[index] - centroid) ** 2).sum()
            ),
        )
        representatives[cluster_id] = [texts[i][:120] for i in distances[:REPRESENTATIVE_TOP_N]]

    method = {
        "module": "cluster_discourse",
        "embedding_model": getattr(encoded, "model_name", "unknown"),
        "embedding_dim": int(getattr(encoded, "dim", 0)),
        "n_clusters": chosen,
        "suggested_k": suggested,
        "silhouette": silhouette,
        "silhouette_curve": curve,
        "k_selection": "轮廓系数扫描（余弦距离），再由研究者确认",
        "keyword_method": "类内词频/全局词频比值（c-TF-IDF 思路），词频≥2",
        "representative_method": "距簇质心最近的 3 条（代表该簇平均话语，非随机抽样）",
        "random_state": seed,
        "n_samples": len(texts),
    }

    notify("ok", f"聚类完成：K={chosen}，轮廓系数 {silhouette}")
    return ClusterResult(
        k=chosen,
        labels=[int(x) for x in labels],
        texts=texts,
        cids=cids,
        sizes={int(k_): int(v) for k_, v in sizes.items()},
        keywords=keywords,
        representatives=representatives,
        silhouette_curve=curve,
        silhouette=silhouette,
        suggested_k=suggested,
        method=method,
    )
