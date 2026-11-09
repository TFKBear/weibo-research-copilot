"""跨话题对比（P3 第 3 项）。

跨话题对比是本工具里**最容易产出假结论**的功能，所以这个模块的第一件事
不是算差异，而是判断"这两次采集能不能比"：

- 一次 mock 一次 live → 不能比。mock 是内置样例数据，拿它和真实采集比
  等于拿教科书案例和田野材料比。
- 每帖评论上限 30 vs 100 → 评论量不能直接比，因为上限就不同。
- 清洗档位不同 → 情感占比不能直接比，因为被剔除的营销/协同内容比例不同。
- 采集时间相隔很久 → 平台侧的话题热度本身已变。

所以输出里分成三件事：
1. **可比性判断**（能不能比、哪些维度能比）；
2. **差异清单**（到底是哪些参数不一样，附具体取值）；
3. 能比的维度上的对比表。

写得保守一点是刻意的：一个研究者如果把"话题 A 评论更多"写进论文，
而两次采集的每帖上限不同，那这句话就是错的——工具必须在源头拦住它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .cluster import tokenize

#: 参与可比性检查的采集参数，按"影响什么"排列。
PARAM_LABELS: tuple[tuple[str, str, str], ...] = (
    ("mode", "采集模式", "mock 与 live 的数据不可互相比较"),
    ("search_mode", "检索口径", "话题聚合与关键词全文拿到的不是同一批帖子"),
    ("comment_window_label", "评论时间窗", "窗口不同则两批样本覆盖的时段不同"),
    ("comment_backend", "评论后端", "不同后端返回的评论集合不同"),
    ("comment_sort", "评论排序", "热度排序与时间排序得到的样本不同"),
    ("post_limit", "选帖数", "选帖数不同则绝对量不可比"),
    ("comments_per_post", "每帖评论上限", "上限不同则绝对量不可比"),
    ("clean_tier", "清洗档位", "档位不同则情感/主题占比不可直接比"),
    ("include_subcomments", "二级评论", "是否含二级评论影响评论集合"),
)

#: 只有这些参数一致，绝对量（评论数）才可比。
VOLUME_KEYS = ("mode", "post_limit", "comments_per_post", "comment_window_label")
#: 只有这些参数一致，占比（情感/主题）才可比。
SHARE_KEYS = ("mode", "clean_tier", "comment_sort", "comment_backend",
              "include_subcomments", "search_mode", "comment_window_label")

SENTIMENT_ORDER = ("positive", "neutral", "negative")
SENTIMENT_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}


@dataclass(slots=True)
class RunProfile:
    """一次采集的可比性画像。"""

    run_id: str
    tag: str
    mode: str = "live"
    created_at: str = ""
    n_posts: int = 0
    n_comments: int = 0
    n_kept: int = 0
    retention: float = 0.0
    sentiment: dict[str, float] = field(default_factory=dict)
    sentiment_n: int = 0
    top_terms: list[tuple[str, int]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def share(self, label: str) -> float | None:
        return self.sentiment.get(label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tag": self.tag,
            "mode": self.mode,
            "created_at": self.created_at,
            "n_posts": self.n_posts,
            "n_comments": self.n_comments,
            "n_kept": self.n_kept,
            "retention": round(self.retention, 4),
            "sentiment": {key: round(value, 4) for key, value in self.sentiment.items()},
            "sentiment_n": self.sentiment_n,
            "top_terms": [list(item) for item in self.top_terms],
            "params": {
                key: self.params.get(key) for key, _label, _why in PARAM_LABELS
            },
        }


@dataclass(slots=True)
class ComparisonResult:
    profiles: list[RunProfile] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    volume_comparable: bool = True
    share_comparable: bool = True
    warnings: list[str] = field(default_factory=list)
    method: dict[str, Any] = field(default_factory=dict)

    @property
    def comparable(self) -> bool:
        return self.volume_comparable and self.share_comparable

    def verdict(self) -> str:
        if not self.profiles:
            return "没有可对比的采集"
        if self.comparable:
            return "可直接对比"
        if self.share_comparable:
            return "仅占比可对比（绝对量不可比）"
        if self.volume_comparable:
            return "仅绝对量可对比（占比不可比）"
        return "不可对比"

    # ------------------------------------------------------------------ #
    def render(self) -> str:
        lines = ["跨话题对比", "=" * 62]
        if not self.profiles:
            lines.append("没有可对比的采集会话。先采集两个及以上话题再回来。")
            return "\n".join(lines)

        lines.append(f"参与对比的采集：{len(self.profiles)} 次")
        for index, profile in enumerate(self.profiles, start=1):
            tag = profile.tag or "（无话题）"
            lines.append(
                f"  {index}. {tag}｜{profile.run_id}｜{profile.mode}"
                f"｜{profile.created_at or '时间未记录'}"
            )
        lines.append("")
        lines.append(f"可比性判断：**{self.verdict()}**")
        lines.append("")

        if self.differences:
            lines.append("参数差异（这些差异决定了哪些结论能说）")
            lines.append("-" * 62)
            for item in self.differences:
                lines.append(f"  · {item}")
            lines.append("")

        lines.append(
            f"{'话题':<14}{'评论数':>7}{'保留率':>8}"
            + "".join(f"{SENTIMENT_ZH[label]:>8}" for label in SENTIMENT_ORDER)
        )
        lines.append("-" * 62)
        for profile in self.profiles:
            tag = (profile.tag or "—")[:12]
            row = f"{tag:<14}{profile.n_comments:>7}{profile.retention:>8.0%}"
            for label in SENTIMENT_ORDER:
                share = profile.share(label)
                if share is None or not self.share_comparable:
                    row += f"{'—':>8}"
                else:
                    row += f"{share:>7.0%} "
            lines.append(row)
        if not self.share_comparable:
            lines.append("")
            lines.append("注：情感占比以「—」显示——两次采集的清洗档位或评论来源不同，")
            lines.append("    直接比占比会把「清洗差异」读成「舆论差异」。")

        lines.append("")
        lines.append("各话题高频词（仅供描述，未做词频统计检验）")
        lines.append("-" * 62)
        for profile in self.profiles:
            terms = "、".join(f"{term}({count})" for term, count in profile.top_terms[:8])
            lines.append(f"  {profile.tag or profile.run_id}：{terms or '（无）'}")

        lines.append("")
        for warning in self.warnings:
            lines.append(f"⚠ {warning}")
        lines.append("")
        lines.append(
            "说明：本表不包含显著性检验，也不做「谁更热」的因果判断。"
            "样本来自各话题的热门评论流，代表的是高互动评论，不是话题全量。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "compare",
            "profiles": [profile.to_dict() for profile in self.profiles],
            "differences": self.differences,
            "volume_comparable": self.volume_comparable,
            "share_comparable": self.share_comparable,
            "verdict": self.verdict(),
            "warnings": self.warnings,
            "method": self.method,
        }


# --------------------------------------------------------------------------- #
def build_comparison(profiles: Iterable[RunProfile]) -> ComparisonResult:
    """比较若干采集画像，并判断哪些维度可比。"""
    profiles = list(profiles)
    if not profiles:
        return ComparisonResult(
            method=_method(0, True, True),
        )

    differences = _differences(profiles)
    volume_comparable = _same(profiles, VOLUME_KEYS)
    share_comparable = _same(profiles, SHARE_KEYS)

    warnings = _warnings(profiles, volume_comparable, share_comparable)
    return ComparisonResult(
        profiles=profiles,
        differences=differences,
        volume_comparable=volume_comparable,
        share_comparable=share_comparable,
        warnings=warnings,
        method=_method(len(profiles), volume_comparable, share_comparable),
    )


def profile_from_rows(
    *,
    run_id: str,
    tag: str,
    params: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    n_posts: int = 0,
    all_rows_count: int | None = None,
    term_limit: int = 8,
) -> RunProfile:
    """把一次采集的数据整理成对比用的画像。

    Args:
        rows: 清洗后保留的评论行（可带 ``sentiment`` 标签）。
        all_rows_count: 清洗前的总数；省略时退化为 ``len(rows)``。
    """
    sentiment_counts = {label: 0 for label in SENTIMENT_ORDER}
    labelled = 0
    for row in rows:
        label = row.get("sentiment")
        if label in sentiment_counts:
            sentiment_counts[label] += 1
            labelled += 1
    sentiment = (
        {label: count / labelled for label, count in sentiment_counts.items()}
        if labelled else {}
    )

    counter: dict[str, int] = {}
    for row in rows:
        text = str(row.get("cleaned_text") or row.get("text_raw") or "")
        for token in tokenize(text):
            counter[token] = counter.get(token, 0) + 1
    top_terms = sorted(counter.items(), key=lambda item: -item[1])[:term_limit]

    total = all_rows_count if all_rows_count is not None else len(rows)
    return RunProfile(
        run_id=run_id,
        tag=tag,
        mode=str(params.get("mode") or "live"),
        created_at=str(params.get("created_at") or ""),
        n_posts=n_posts,
        n_comments=total,
        n_kept=len(rows),
        retention=(len(rows) / total) if total else 0.0,
        sentiment=sentiment,
        sentiment_n=labelled,
        top_terms=[(term, count) for term, count in top_terms],
        params=_flatten_params(params),
    )


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """把嵌套参数压成可比对的标量。

    时间窗在参数里是一个 dict（since/until/active/label），而可比性检查按
    "同一 key 的取值集合大小"判断差异；dict 不可哈希，直接塞进去会抛
    ``TypeError: unhashable type: 'dict'``。这里统一压成一句 label，
    既可比，也正好是报告里能直接读的说法。
    """
    flat = dict(params or {})
    window = flat.get("comment_window")
    if isinstance(window, dict):
        flat["comment_window_label"] = (
            window.get("label")
            or ("不限时间" if not window.get("active") else
                f"{window.get('since') or '不限'} ～ {window.get('until') or '不限'}")
        )
    else:
        flat.setdefault("comment_window_label", "不限时间")
    flat.setdefault("search_mode", "topic")
    return flat


# --------------------------------------------------------------------------- #
def _same(profiles: Sequence[RunProfile], keys: Sequence[str]) -> bool:
    for key in keys:
        values = {profile.params.get(key) for profile in profiles}
        if len(values) > 1:
            return False
    return True


def _differences(profiles: Sequence[RunProfile]) -> list[str]:
    items: list[str] = []
    for key, label, why in PARAM_LABELS:
        values: dict[str, int] = {}
        missing = 0
        for profile in profiles:
            value = profile.params.get(key)
            if value is None:
                missing += 1
                continue
            values[str(value)] = values.get(str(value), 0) + 1
        if len(values) < 2 and not (values and missing):
            continue
        if missing and values:
            rendered = "；".join(f"{value}（{count} 次）"
                                 for value, count in values.items())
            items.append(
                f"{label}不同：{rendered}；未记录（{missing} 次） —— "
                f"{why}。未记录不等于相同，无法假定一致。"
            )
        else:
            rendered = "；".join(f"{value}（{count} 次）"
                                 for value, count in values.items())
            items.append(f"{label}不同：{rendered} —— {why}")
    return items


def _warnings(
    profiles: Sequence[RunProfile],
    volume_comparable: bool,
    share_comparable: bool,
) -> list[str]:
    warnings: list[str] = []
    modes = {profile.mode for profile in profiles}
    if "mock" in modes and "live" in modes:
        warnings.append(
            "对比中包含 mock（内置样例）会话。mock 数据是为了跑通流程而生成的，"
            "与真实采集放在一起比较没有意义，请只用 live 会话做对比。"
        )
    tags = [profile.tag for profile in profiles if profile.tag]
    if len(tags) != len(set(tags)):
        warnings.append(
            "存在同名话题的多次采集。同一话题的重复采集更适合比较"
            "「采集稳定性」，而不是当作不同话题对比。"
        )
    if not volume_comparable:
        warnings.append(
            "评论量不能直接横向比较：各次采集的选帖数或每帖上限不同，"
            "数量差异首先反映的是采集参数差异。"
        )
    if not share_comparable:
        warnings.append(
            "情感占比不能直接横向比较：清洗档位或评论来源不同，"
            "占比差异里混着清洗差异。"
        )
    small = [profile for profile in profiles if profile.n_kept < 30]
    if small:
        names = "、".join((profile.tag or profile.run_id) for profile in small)
        warnings.append(
            f"{len(small)} 次采集的保留评论不足 30 条（{names}），"
            "占比波动会很大，结论不宜写进论文。"
        )
    if any(profile.sentiment_n == 0 for profile in profiles):
        warnings.append(
            "有采集会话还没有情感分析结果，表中占比为「—」。"
            "请先对各会话跑一次情感分析再对比。"
        )
    return warnings


def _method(n_runs: int, volume_comparable: bool, share_comparable: bool) -> dict[str, Any]:
    return {
        "module": "compare",
        "compared_runs": n_runs,
        "volume_comparable": volume_comparable,
        "share_comparable": share_comparable,
        "checked_params": [label for _key, label, _why in PARAM_LABELS],
        "note": "只比较结构与占比，不做显著性检验，不做热度排序",
    }
