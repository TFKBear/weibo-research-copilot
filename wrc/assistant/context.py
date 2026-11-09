"""会话上下文快照。

助手要能回答"我的样本量够不够聚类"这类问题，前提是它**知道现在是什么状况**。
没有上下文的通用回答（"一般来说 100 条以上比较合适"）对具体研究没有价值。

所以这里把当前会话的关键事实整理成一段结构化文本，随问题一起送给模型。
这段文本同时也是可审计的：研究者能看到"助手当时依据的是什么数据"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionContext:
    """当前研究状态。字段尽量用"人能读懂"的方式存。"""

    run_id: str | None = None
    tag: str = ""
    n_posts: int = 0
    n_comments: int = 0
    n_kept: int = 0
    n_users: int = 0
    clean_tier: str = "未清洗"
    pipeline_version: str = ""
    has_sentiment: bool = False
    has_discourse: bool = False
    has_community: bool = False
    sentiment_distribution: dict[str, int] = field(default_factory=dict)
    agreement_rate: float | None = None
    discourse_k: int | None = None
    discourse_silhouette: float | None = None
    community_k: int | None = None
    community_silhouette: float | None = None
    llm_model: str = ""
    ai_models_ready: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def retention_rate(self) -> float:
        return (self.n_kept / self.n_comments) if self.n_comments else 0.0

    def to_prompt(self) -> str:
        """整理成给模型看的中文摘要。"""
        if not self.run_id:
            return "当前还没有任何采集数据。用户尚未开始采集。"

        lines = [
            f"话题：{self.tag or '未记录'}",
            f"候选帖子：{self.n_posts} 篇",
            f"采集评论：{self.n_comments} 条",
        ]
        if self.clean_tier and self.clean_tier != "未清洗":
            lines.append(
                f"清洗档位：{self.clean_tier}（管线 {self.pipeline_version}），"
                f"保留 {self.n_kept} 条（{self.retention_rate:.1%}）"
            )
        else:
            lines.append("清洗：尚未执行")

        if self.has_sentiment:
            total = sum(self.sentiment_distribution.values()) or 1
            parts = "、".join(
                f"{label} {count} 条（{count / total:.0%}）"
                for label, count in sorted(
                    self.sentiment_distribution.items(), key=lambda kv: -kv[1]
                )
            )
            line = f"情感分析：已完成，分布为 {parts}"
            if self.agreement_rate is not None:
                line += f"；双通道不一致率 {self.agreement_rate:.1%}"
            lines.append(line)
        else:
            lines.append("情感分析：未执行")

        if self.has_discourse:
            line = f"话语聚类：K={self.discourse_k}"
            if self.discourse_silhouette is not None:
                line += f"，轮廓系数 {self.discourse_silhouette}"
            lines.append(line)
        else:
            lines.append("话语聚类：未执行")

        if self.has_community:
            line = f"用户社群聚类：K={self.community_k}，参与用户 {self.n_users} 位"
            if self.community_silhouette is not None:
                line += f"，轮廓系数 {self.community_silhouette}"
            lines.append(line)
        else:
            lines.append("用户社群聚类：未执行")

        if self.llm_model:
            lines.append(f"分析用模型：{self.llm_model}")
        lines.append(f"本地 AI 模型就绪：{'是' if self.ai_models_ready else '否'}")
        return "\n".join(f"- {line}" for line in lines)


def build_context(store: Any, settings: Any | None = None) -> SessionContext:
    """从数据库读取当前会话状态。"""
    from ..ai.status import detect_status
    from ..llm.client import LlmClient

    run_id = store.latest_run_id()
    if run_id is None:
        return SessionContext()

    run = store.get_run(run_id) or {}
    posts = store.load_posts(run_id)
    rows = store.load_analysis_rows(run_id, only_kept=False)
    kept = [row for row in rows if row.get("kept")]

    context = SessionContext(
        run_id=run_id,
        tag=str(run.get("tag") or ""),
        n_posts=len(posts),
        n_comments=len(rows),
        n_kept=len(kept),
        n_users=len({str(row.get("user_id") or "") for row in kept} - {""}),
        clean_tier=str(kept[0].get("clean_tier")) if kept else "未清洗",
        pipeline_version=str(kept[0].get("pipeline_version")) if kept else "",
    )

    summaries = store.latest_analysis_summaries(run_id)
    sentiment = summaries.get("sentiment")
    if sentiment:
        context.has_sentiment = True
        context.sentiment_distribution = dict(sentiment.get("distribution") or {})
        agreement = sentiment.get("agreement") or {}
        if isinstance(agreement, dict) and "disagreement_rate" in agreement:
            context.agreement_rate = float(agreement["disagreement_rate"])

    discourse = summaries.get("cluster_discourse")
    if discourse:
        context.has_discourse = True
        context.discourse_k = int(discourse.get("k") or 0) or None
        silhouette = discourse.get("silhouette")
        context.discourse_silhouette = float(silhouette) if silhouette is not None else None

    community = summaries.get("cluster_community")
    if community:
        context.has_community = True
        context.community_k = int(community.get("k") or 0) or None
        context.n_users = int(community.get("n_users") or 0)
        silhouette = community.get("silhouette")
        context.community_silhouette = float(silhouette) if silhouette is not None else None

    try:
        context.llm_model = LlmClient().config.model
    except Exception:  # noqa: BLE001
        context.llm_model = ""

    if settings is not None:
        try:
            embed_ok = detect_status(settings.ai_embedding_model_dir).ready
            sent_ok = detect_status(settings.ai_sentiment_model_dir).ready
            context.ai_models_ready = bool(embed_ok and sent_ok)
        except Exception:  # noqa: BLE001
            context.ai_models_ready = False

    return context
