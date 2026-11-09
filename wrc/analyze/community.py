"""用户级社群聚类（模块 D2 之二）。

与话语聚类的区别：这里聚的是**人**，不是话。

特征取自需求文档：评论频率、文本长度、表情使用率、互动水平、粉丝规模、
跨帖活跃度。缺失的那几项（是否原创/转发、与热门评论的互动模式）在评论接口
里拿不到，这一点必须写进方法卡片，而不是假装做到了。

角色标签的原则：**描述性命名，不做因果判断**。
"高频参与型"是"这个簇的评论频率高于其他簇"，不是"这些人是水军"。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "评论数",
    "平均字数",
    "表情率",
    "平均获赞",
    "粉丝数(log)",
    "参与帖子数",
)

#: 角色命名的触发词：特征 z 值高于此阈值才计入角色名。
_ROLE_THRESHOLD = 0.6
_ROLE_WORDS = {
    "评论数": "高频参与",
    "平均字数": "长文本论述",
    "表情率": "情绪化表达",
    "平均获赞": "受认可",
    "粉丝数(log)": "高影响力",
    "参与帖子数": "跨帖活跃",
}

_EMOJI_TOKEN = re.compile(r"\[[^\[\]\s]{1,12}\]")
_EMOJI_UNICODE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


@dataclass(slots=True)
class CommunityResult:
    k: int = 0
    user_labels: dict[str, int] = field(default_factory=dict)
    user_sizes: dict[int, int] = field(default_factory=dict)
    roles: dict[int, str] = field(default_factory=dict)
    centroids: dict[int, dict[str, float]] = field(default_factory=dict)
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    silhouette_curve: list[tuple[int, float]] = field(default_factory=list)
    silhouette: float | None = None
    suggested_k: int | None = None
    explained_variance: float | None = None
    method: dict[str, Any] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_labels)

    def render(self) -> str:
        lines = ["用户社群聚类结果", "=" * 62]
        lines.append(f"用户数：{self.n_users}，K={self.k}，轮廓系数 {self.silhouette}")
        if self.explained_variance is not None:
            lines.append(f"前两个主成分解释方差：{self.explained_variance:.1%}")
        if self.silhouette is not None and self.silhouette < 0.25:
            lines.append("")
            lines.append(
                f"⚠ 轮廓系数 {self.silhouette} 偏低，社群边界不清晰。"
                "小样本下这是常态，角色标签仅供参考，不宜作为论文结论。"
            )
        singles = [cid for cid, size in self.user_sizes.items() if size < 3]
        if singles:
            lines.append("")
            lines.append(
                f"⚠ 有 {len(singles)} 个簇少于 3 人："
                + "、".join(f"簇{cid}({self.user_sizes[cid]}人)" for cid in singles)
            )
            lines.append(
                "  这通常说明 K-Means 把个别极端账号（粉丝极多、发评极频繁）单独切了出来，"
                "而不是真的找到了「社群」。这类簇只能当个案看，不能当群体结论。"
            )
        lines.append("")
        lines.append("特征说明：" + "、".join(FEATURE_NAMES))
        lines.append("")
        for cluster_id in sorted(self.user_sizes, key=lambda x: -self.user_sizes[x]):
            role = self.roles.get(cluster_id, f"簇{cluster_id}")
            lines.append(
                f"簇 {cluster_id}｜{role}｜{self.user_sizes[cluster_id]} 位用户"
            )
            profile = self.centroids.get(cluster_id, {})
            detail = "  ".join(
                f"{name}{value:+.2f}σ"
                for name, value in profile.items()
            )
            lines.append(f"    画像（相对全体）：{detail}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_users": self.n_users,
            "user_sizes": {str(k): v for k, v in self.user_sizes.items()},
            "roles": {str(k): v for k, v in self.roles.items()},
            "centroids": {str(k): v for k, v in self.centroids.items()},
            # coords 也存下来：报告重生成时要靠它重画 PCA 散点图
            "coords": {user: list(point) for user, point in self.coords.items()},
            "silhouette": self.silhouette,
            "silhouette_curve": self.silhouette_curve,
            "suggested_k": self.suggested_k,
            "explained_variance": self.explained_variance,
            "feature_names": list(FEATURE_NAMES),
            "method": self.method,
            "assignments": [
                {"user_pseudo_id": user, "cluster": label}
                for user, label in self.user_labels.items()
            ],
        }


# --------------------------------------------------------------------------- #
def _emoji_count(text: str) -> int:
    return len(_EMOJI_TOKEN.findall(text)) + len(_EMOJI_UNICODE.findall(text))


def build_feature_matrix(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[float]]]:
    """把评论行按用户聚合成特征矩阵。"""
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        user = str(row.get("user_pseudo_id") or row.get("user_id") or "")
        if not user:
            continue
        bucket[user].append(row)

    users: list[str] = []
    matrix: list[list[float]] = []
    for user, items in bucket.items():
        texts = [
            str(item.get("cleaned_text") or item.get("text_raw") or "") for item in items
        ]
        lengths = [len(text) for text in texts] or [0]
        emoji_ratio = sum(
            _emoji_count(text) / max(1, len(text)) for text in texts
        ) / len(texts)
        likes = [float(item.get("like_count") or 0) for item in items]
        followers = [
            float(item.get("followers_count") or 0) for item in items if item.get("followers_count")
        ]
        import math

        follower_value = math.log10(1.0 + (sum(followers) / len(followers) if followers else 0.0))
        posts = len({str(item.get("post_mid")) for item in items})

        users.append(user)
        matrix.append(
            [
                float(len(items)),
                sum(lengths) / len(lengths),
                emoji_ratio,
                sum(likes) / len(likes),
                follower_value,
                float(posts),
            ]
        )
    return users, matrix


def _role_for(z_scores: dict[str, float]) -> str:
    """按特征 z 值给簇起一个描述性名字。"""
    positive = [
        (word, z) for name, z in z_scores.items() if (word := _ROLE_WORDS.get(name))
        and z > _ROLE_THRESHOLD
    ]
    positive.sort(key=lambda pair: -pair[1])
    if not positive:
        return "普通参与型"
    return "·".join(word for word, _ in positive[:2]) + "型"


def cluster_communities(
    rows: Sequence[dict[str, Any]],
    *,
    k: int | None = None,
    k_max: int = 8,
    seed: int = 20240101,
) -> CommunityResult:
    """按用户粒度聚类并生成可解释的角色标签。"""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    users, raw = build_feature_matrix(rows)
    if len(users) < 3:
        raise ValueError("去重后用户少于 3 人，无法做社群聚类")

    matrix = np.asarray(raw, dtype=float)

    # 计数类特征（评论数 / 平均获赞 / 参与帖子数）是重尾分布：
    # 一个明星账号的获赞可以是其他人的上千倍。直接标准化会让 K-Means
    # 把它单独切成一个簇，得到"43 人一簇 + 两个单点簇"这种伪结构。
    # 先取 log1p 再标准化，能显著缓解。
    design = matrix.copy()
    for column in (0, 3, 5):
        design[:, column] = np.log1p(np.maximum(design[:, column], 0.0))

    scaled = StandardScaler().fit_transform(design)

    upper = min(k_max, len(users) - 1)
    curve: list[tuple[int, float]] = []
    for candidate in range(2, upper + 1):
        model = KMeans(n_clusters=candidate, random_state=seed, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        try:
            curve.append(
                (candidate, round(float(silhouette_score(scaled, labels)), 4))
            )
        except Exception:  # noqa: BLE001
            continue

    suggested = max(curve, key=lambda pair: pair[1])[0] if curve else 2
    chosen = max(2, min(k or suggested, len(users) - 1))
    model = KMeans(n_clusters=chosen, random_state=seed, n_init=10)
    labels = model.fit_predict(scaled)

    silhouette = None
    if len(set(labels)) > 1:
        try:
            silhouette = round(float(silhouette_score(scaled, labels)), 4)
        except Exception:  # noqa: BLE001
            silhouette = None

    # ---- 每个簇的画像（相对全体的 z 值） ----
    overall_mean = design.mean(axis=0)
    overall_std = design.std(axis=0)
    overall_std[overall_std == 0] = 1.0

    sizes: dict[int, int] = {}
    centroids: dict[int, dict[str, float]] = {}
    roles: dict[int, str] = {}
    for cluster_id in range(chosen):
        members = design[labels == cluster_id]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        z = (centroid - overall_mean) / overall_std
        profile = {name: round(float(value), 2) for name, value in zip(FEATURE_NAMES, z)}
        centroids[cluster_id] = profile
        roles[cluster_id] = _role_for(profile)
        sizes[cluster_id] = int(len(members))

    # ---- PCA 二维坐标（可视化用） ----
    explained = None
    coords: dict[str, tuple[float, float]] = {}
    if len(users) >= 3:
        pca = PCA(n_components=2, random_state=seed)
        projected = pca.fit_transform(scaled)
        explained = float(pca.explained_variance_ratio_.sum())
        coords = {
            user: (round(float(point[0]), 4), round(float(point[1]), 4))
            for user, point in zip(users, projected)
        }

    method = {
        "module": "cluster_community",
        "features": "、".join(FEATURE_NAMES),
        "n_clusters": chosen,
        "suggested_k": suggested,
        "silhouette": silhouette,
        "silhouette_curve": curve,
        "pca_explained_variance": round(explained, 4) if explained is not None else None,
        "standardized": True,
        "transform": "评论数/平均获赞/参与帖子数 先取 log1p，再标准化（抑制重尾影响）",
        "random_state": seed,
        "n_users": len(users),
        "n_comments": len(rows),
        "role_method": "按簇质心在各特征上的 z 值命名（描述性，非因果判断）",
        "missing_features": (
            "评论接口不提供「是否原创/转发」与「与热门评论的互动模式」，"
            "这两项未纳入特征；角色标签仅基于行为与文本特征。"
        ),
    }

    return CommunityResult(
        k=chosen,
        user_labels={user: int(label) for user, label in zip(users, labels)},
        user_sizes=sizes,
        roles=roles,
        centroids=centroids,
        coords=coords,
        silhouette_curve=curve,
        silhouette=silhouette,
        suggested_k=suggested,
        explained_variance=explained,
        method=method,
    )
