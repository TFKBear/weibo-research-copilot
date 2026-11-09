"""分析服务层：把「跑分析」和「把结果存下来」绑成一件事。

为什么需要这一层：
之前 `tools/run_sentiment.py` 跑完只写一个 JSON 文件，结果进不了数据库，
也就进不了导出层——那等于分析白做。凡是有产出就必须落库，
否则研究者隔天想重新导出，就得重跑一次 LLM（花钱、且结果可能变）。

约定：每个分析运行有自己的 `analysis_id`，与采集会话 `run_id` 多对一。
同一会话可以跑多次分析（换模型、换 K），历史全部保留，导出时按时间合并。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable

from .cluster import ClusterResult, cluster_discourse
from .community import CommunityResult, cluster_communities
from .compare import ComparisonResult, RunProfile, build_comparison, profile_from_rows
from .replies import ReplyStructureResult, build_reply_structure
from .sentiment import SentimentResult, analyse
from .stance import StanceResult, analyse_stance
from .timeseries import TimeSeriesResult, build_timeseries

#: 历史版本里同一个采集参数改过名，对比前必须归一，
#: 否则会把"改名"读成"参数不同"，得出"两次采集不可比"的假警报。
_PARAM_ALIASES = {
    "comments_target_per_post": "comments_per_post",
    "posts_preview": "post_limit",
}


def _summary_of(result: Any) -> dict:
    """把结果对象转成入库用的摘要。

    额外存一份 `_rendered`：报告是可以**随时从数据库重新生成**的，
    否则"先跑情感、再用 --no-sentiment 重跑聚类"就会把情感那一节弄丢，
    研究者还得重跑一次 LLM 才能把报告补全。
    """
    payload = dict(result.to_dict())
    payload["_rendered"] = result.render()
    return payload


def new_analysis_id(module: str) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{module[:6]}-{os.urandom(2).hex()}"


def run_sentiment(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    llm_client: Any,
    local_classifier: Any | None = None,
    limit: int | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[SentimentResult, str]:
    """跑情感分析并入库。返回 (结果, analysis_id)。"""
    notify = emit or (lambda _level, _message: None)
    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")
    if limit:
        rows = rows[:limit]

    result = analyse(
        rows,
        llm_client=llm_client,
        local_classifier=local_classifier,
        emit=notify,
    )
    analysis_id = new_analysis_id("sentiment")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="sentiment",
        params={"limit": limit},
        method=result.method,
        summary=_summary_of(result),
    )
    store.save_sentiment_items(
        analysis_id,
        [item.to_dict() for item in result.items],
    )
    notify("ok", f"情感结果已入库（analysis_id={analysis_id}）")
    return result, analysis_id


def run_discourse(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    k: int | None = None,
    k_max: int = 10,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[ClusterResult, str]:
    """跑话语聚类并入库。"""
    from ..ai.embedder import build_embedder
    from ..ai.status import detect_status

    status = detect_status(settings.ai_embedding_model_dir)
    if not status.ready:
        raise ValueError(f"句向量模型未就绪：{status.summary()}")

    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")

    embedder = build_embedder(status.onnx_path, status.tokenizer_path)
    result = cluster_discourse(rows, embedder=embedder, k=k, k_max=k_max, emit=emit)

    analysis_id = new_analysis_id("discourse")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="cluster_discourse",
        params={"k": k, "k_max": k_max},
        method=result.method,
        summary=_summary_of(result),
    )
    store.save_discourse_clusters(analysis_id, result.to_dict()["assignments"])
    if emit:
        emit("ok", f"聚类结果已入库（analysis_id={analysis_id}）")
    return result, analysis_id


def run_community(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    k: int | None = None,
    k_max: int = 8,
) -> tuple[CommunityResult, str]:
    """跑用户社群聚类并入库。"""
    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")

    result = cluster_communities(rows, k=k, k_max=k_max)
    analysis_id = new_analysis_id("community")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="cluster_community",
        params={"k": k, "k_max": k_max},
        method=result.method,
        summary=_summary_of(result),
    )
    store.save_community_clusters(
        analysis_id,
        result.to_dict()["assignments"],
        {str(key): value for key, value in result.roles.items()},
    )
    return result, analysis_id


def run_stance(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    llm_client: Any,
    topic: str | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[StanceResult, str]:
    """跑立场分析并入库。

    立场必须锚定到**具体政策/话题**：同一条评论对不同议题的立场可能相反。
    所以默认取本次采集的 tag 作为立场对象，也允许调用方显式指定。
    """
    notify = emit or (lambda _level, _message: None)
    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")

    run = store.get_run(run_id) or {}
    resolved_topic = topic or str(run.get("tag") or "") or "本次话题"

    result = analyse_stance(
        rows, topic=resolved_topic, llm_client=llm_client, emit=notify
    )
    analysis_id = new_analysis_id("stance")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="stance",
        params={"topic": resolved_topic},
        method=result.method,
        summary=_summary_of(result),
    )
    store.save_stance_items(analysis_id, [item.to_dict() for item in result.items])
    notify("ok", f"立场结果已入库（analysis_id={analysis_id}）")
    return result, analysis_id


def run_timeseries(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    granularity: str | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[TimeSeriesResult, str]:
    """跑时间演变分析并入库。

    时间演变**不重新调模型**：它只是把已有的情感/簇结果按时间重新聚合。
    所以对手上有分析结果但没做时间维度的旧会话，这一步可以随时补跑，
    不产生额外费用，也不会改动已有的情感标签。
    """
    notify = emit or (lambda _level, _message: None)
    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")

    merged = _merge_analysis(rows, store.load_comment_analysis(run_id))
    result = build_timeseries(merged, granularity=granularity)

    analysis_id = new_analysis_id("timeseries")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="timeseries",
        params={"granularity": granularity},
        method=result.method,
        summary=_summary_of(result),
    )
    notify(
        "ok",
        f"时间演变已入库（analysis_id={analysis_id}，{len(result.buckets)} 个时段）",
    )
    return result, analysis_id


def run_replies(
    run_id: str,
    *,
    store: Any,
    settings: Any,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[ReplyStructureResult, str]:
    """重建二级评论结构并入库。

    数据从**清洗后保留**的评论里取：被判定为垃圾的回复不该参与结构判断，
    否则水军刷屏会直接被读成"讨论热烈"。
    """
    notify = emit or (lambda _level, _message: None)
    rows = store.load_analysis_rows(run_id)
    if not rows:
        raise ValueError(f"会话 {run_id} 没有清洗后的数据")

    result = build_reply_structure(rows)
    analysis_id = new_analysis_id("replies")
    store.save_analysis_run(
        analysis_id=analysis_id,
        run_id=run_id,
        module="replies",
        params={"min_n": result.min_n},
        method=result.method,
        summary=_summary_of(result),
    )
    if not result.available:
        notify("info", "本次采集不含二级评论，回复结构报告为空（不是错误）。")
    else:
        notify(
            "ok",
            f"回复结构已入库（{result.n_threads} 个讨论串，形态：{result.shape_label()}）",
        )
    return result, analysis_id


def build_stance_crosstab(store: Any, run_id: str) -> Any | None:
    """构造「立场 × 情感」交叉表；两个维度缺一个就返回 None。

    抽成函数是因为报告、图表、导出三处都要它，各写一遍必然各错一遍。
    """
    from .stance_crosstab import sentiment_by_stance

    merged = _merge_analysis(
        store.load_analysis_rows(run_id), store.load_comment_analysis(run_id)
    )
    if not any(row.get("stance") for row in merged):
        return None
    if not any(row.get("sentiment") for row in merged):
        return None
    return sentiment_by_stance(merged)


def build_discourse_stance_crosstab(store: Any, run_id: str) -> Any | None:
    """构造「话语簇 × 立场」交叉表；缺聚类或立场任一就返回 None。"""
    from .stance_crosstab import stance_by_discourse

    merged = _merge_analysis(
        store.load_analysis_rows(run_id), store.load_comment_analysis(run_id)
    )
    if not any(row.get("stance") for row in merged):
        return None
    assignments = {
        str(row["cid"]): int(row["cluster_discourse"])
        for row in merged
        if row.get("cluster_discourse") is not None
    }
    if not assignments:
        return None
    return stance_by_discourse(merged, assignments)


def resolve_events(
    store: Any, run_id: str, events: list[Any] | None = None
) -> list[Any]:
    """取得本次会话的事件列表：显式传入的会写回采集参数，否则从参数里读。

    为什么要把 None 和空列表分开：
    - ``None`` = "没指定，用该会话已保存的"；
    - ``[]``  = "明确不要事件"（对应 ``--no-events``）。
    早期两者都当假值处理，于是 --no-events 根本清不掉已保存的事件。
    """
    from .events import events_from_params

    if events is not None:
        if events:
            store.merge_run_params(run_id, {
                "events": [event.to_dict() for event in events],
            })
        else:
            store.merge_run_params(run_id, {"events": []})
        return events

    run = store.get_run(run_id) or {}
    try:
        params = json.loads(run.get("params_json") or "{}")
    except json.JSONDecodeError:
        params = {}
    return events_from_params(params)


def run_event_study(
    store: Any, run_id: str, events: list[Any] | None = None
) -> tuple[list[Any], list[Any]]:
    """跑事件前后对比。返回 (结果列表, 使用的事件列表)。"""
    from .events import analyse_events

    specs = resolve_events(store, run_id, events)
    if not specs:
        return [], []

    merged = _merge_analysis(
        store.load_analysis_rows(run_id), store.load_comment_analysis(run_id)
    )
    has_stance = any(row.get("stance") for row in merged)
    metrics = ("stance", "sentiment") if has_stance else ("sentiment",)
    return analyse_events(merged, events=specs, metrics=metrics), specs


def _merge_analysis(
    rows: list[dict[str, Any]], comment_analysis: dict[str, dict]
) -> list[dict[str, Any]]:
    """把分析结果按 cid 贴回数据行。

    时间演变、交叉表、回复关系都要「原始字段 + 分析标签」同时在场，
    这段合并逻辑原来散在各工具里，抽出来避免三处各写一遍（也各错一遍）。
    """
    merged: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("cid"))
        extra = comment_analysis.get(cid, {})
        merged.append({
            **row,
            "sentiment": extra.get("sentiment"),
            "intensity": extra.get("intensity"),
            # 本地对照通道的字段必须一起带出来：人工抽查表要并排显示
            # 「LLM 判定」与「本地判定」，缺了这两列，抽查表就只剩一列可看。
            "local_sentiment": extra.get("local_sentiment"),
            "local_score": extra.get("local_score"),
            "in_sample": bool(extra.get("in_sample")),
            "stance": extra.get("stance"),
            "stance_confidence": extra.get("stance_confidence"),
            "cluster_discourse": extra.get("cluster_discourse"),
        })
    return merged


# --------------------------------------------------------------------------- #
# 跨话题对比
# --------------------------------------------------------------------------- #
def _normalise_params(params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    for old, new in _PARAM_ALIASES.items():
        if old in merged and new not in merged:
            merged[new] = merged[old]
    merged.setdefault("mode", "live")
    return merged


def select_comparable_runs(
    store: Any, *, include_mock: bool = False, limit: int = 6
) -> list[str]:
    """挑选用于对比的采集会话。

    排序规则：**不同话题各取最近一次**排前面，同话题的重复采集补在后面。
    跨话题对比当然优先比不同话题；但只有一次话题时，同话题的重复采集仍有意义
    ——它回答的是「采集稳不稳定」，对比模块会据此给出对应告警。
    """
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT r.run_id, r.params_json, r.tag FROM collection_runs r "
            "WHERE EXISTS (SELECT 1 FROM clean_comments c WHERE c.run_id = r.run_id) "
            "ORDER BY r.created_at DESC"
        ).fetchall()

    distinct: list[str] = []
    repeated: list[str] = []
    seen_tags: set[str] = set()
    for row in rows:
        try:
            params = json.loads(row["params_json"] or "{}")
        except json.JSONDecodeError:
            params = {}
        if not include_mock and params.get("mode") == "mock":
            continue
        tag = str(row["tag"] or "")
        if tag and tag in seen_tags:
            repeated.append(str(row["run_id"]))
            continue
        if tag:
            seen_tags.add(tag)
        distinct.append(str(row["run_id"]))
    # 有 2 个以上不同话题时，不掺同话题的重复采集——
    # 把它们混进来只会让"跨话题对比"变成"同一话题的采集稳定性检查"。
    if len(distinct) >= 2:
        return distinct[:limit]
    return (distinct + repeated)[:limit]


def build_profiles(
    store: Any, run_ids: list[str]
) -> tuple[list[RunProfile], list[str]]:
    """把若干会话整理成对比画像。返回 (画像列表, 被跳过的原因)。"""
    profiles: list[RunProfile] = []
    skipped: list[str] = []
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            skipped.append(f"{run_id}：数据库里没有这条采集记录")
            continue
        try:
            params = json.loads(run.get("params_json") or "{}")
        except json.JSONDecodeError:
            params = {}
        params = _normalise_params(params)
        params["created_at"] = str(run.get("created_at") or "")

        kept = store.load_analysis_rows(run_id)
        if not kept:
            skipped.append(f"{run_id}：还没有清洗后的数据")
            continue
        # 档位是可从数据反查的事实（clean_comments 里就存着），
        # 老会话参数里没记全时用它补上——这不是猜测，是读回自己的记录。
        params.setdefault("clean_tier", kept[0].get("clean_tier") or "未记录")
        params.setdefault("pipeline_version", kept[0].get("pipeline_version") or "未记录")
        total = len(store.load_analysis_rows(run_id, only_kept=False))
        merged = _merge_analysis(kept, store.load_comment_analysis(run_id))
        with store.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM raw_posts WHERE run_id=?", (run_id,)
            ).fetchone()
        profiles.append(profile_from_rows(
            run_id=run_id,
            tag=str(run.get("tag") or ""),
            params=params,
            rows=merged,
            n_posts=int(row["n"]) if row else 0,
            all_rows_count=total,
        ))
    return profiles, skipped


def run_comparison(
    store: Any,
    *,
    run_ids: list[str] | None = None,
    include_mock: bool = False,
    limit: int = 6,
) -> tuple[ComparisonResult, list[str]]:
    """跨话题对比：不调模型、不写库，纯读数据算差异。"""
    ids = run_ids or select_comparable_runs(
        store, include_mock=include_mock, limit=limit
    )
    profiles, skipped = build_profiles(store, ids)
    return build_comparison(profiles), skipped
