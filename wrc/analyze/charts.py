"""分析图表（模块 D1/D2 的"输出图表"要求）。

为什么自己写而不是让用户自己画：
需求文档明确要求「整体情感分布图 + 分帖子情感分布 + 时间序列情感趋势」，
这些是论文里要用的图，散落在 CSV 里等于没做。

三个工程细节：

1. **强制 Agg 后端**：matplotlib 默认后端在无显示环境（或服务端）会报错，
   而本工具没有交互式绘图需求——只出 PNG。
2. **中文字体**：matplotlib 默认字体没有中文字形，会画出一堆方框。
   这里按 Windows/Linux 常见字体依次尝试，并在全部失败时**明确降级为英文标签**
   而不是静默出方框图。
3. **图注里带样本量**：一张"83% 正面"的图如果不写 n=6，读者会误判可信度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..console import note

#: 中文字体候选，按优先级。
_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Arial Unicode MS",
)

_LABELS_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}
_COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}

_font_ready: bool | None = None


def _setup(size: int = 10) -> Any:
    """初始化绘图环境，返回 pyplot。"""
    global _font_ready
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 每次绘图都重新取得候选字体：不能只缓存布尔值，却在后续调用使用
    # 首次调用的局部变量。Seaborn 每次 set_theme 都会覆盖字体配置。
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in _CJK_FONTS if name in available]
    if not chosen and _font_ready is not False:
        note("未找到中文字体，图表将使用英文标签（不影响数据）")
    _font_ready = bool(chosen)
    # 统一视觉风格：白底 + 淡网格 + 加粗标题。论文插图要的是"一眼能读"，
    # 而不是 matplotlib 的默认灰底小字。seaborn 没装时退回默认样式，不报错。
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook", palette="deep")
        plt.rcParams.update({
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })
    except Exception:  # noqa: BLE001 - seaborn 不可用就保持 matplotlib 默认
        pass
    # 放在样式处理之后，且不放在可选依赖的异常吞并范围内。
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chosen + ["DejaVu Sans"],
        "font.size": size,
        "axes.unicode_minus": False,
        "figure.dpi": 200,
    })
    return plt


def _label(label: str) -> str:
    return _LABELS_ZH.get(label, label) if _font_ready else label


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
def cleaning_comparison(
    raw_rows: Sequence[dict[str, Any]],
    kept_rows: Sequence[dict[str, Any]],
    path: Path,
) -> Path | None:
    """清洗前后的分布对比（需求 B2 的"图"）。

    为什么必须画图而不是只给分位数表：
    研究者要判断的是「清洗是否引入了**系统性偏差**」——那是一个分布形状问题。
    如果被删掉的主要是长文本或高粉丝用户，均值/中位数可能只动一点点，
    但形状（尾部、双峰）会明显变形。表看不出来，图一眼能看出来。
    """
    if not raw_rows or not kept_rows:
        return None

    def lengths(rows: Sequence[dict[str, Any]], key: str) -> list[int]:
        values: list[int] = []
        for row in rows:
            text = str(row.get(key) or "")
            if text:
                values.append(len(text))
        return values

    def followers(rows: Sequence[dict[str, Any]]) -> list[int]:
        values = []
        for row in rows:
            count = row.get("followers_count")
            if count is None:
                continue
            try:
                number = int(count)
            except (TypeError, ValueError):
                continue
            if number > 0:  # log 轴不能有 0
                values.append(number)
        return values

    before_len = lengths(raw_rows, "text_raw")
    after_len = lengths(kept_rows, "cleaned_text")
    before_fans = followers(raw_rows)
    after_fans = followers(kept_rows)
    if not before_len or not after_len:
        return None

    plt = _setup()
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # 左：评论长度。用 density 而不是计数——两次样本量不同，计数不可比。
    bins = 40
    axes[0].hist(before_len, bins=bins, density=True, alpha=0.45,
                 color="#6b7280", label="清洗前" if _font_ready else "before")
    axes[0].hist(after_len, bins=bins, density=True, alpha=0.75,
                 color="#1d4ed8", label="清洗后" if _font_ready else "after")
    axes[0].set_xlabel("字符数" if _font_ready else "characters")
    axes[0].set_ylabel("密度" if _font_ready else "density")
    axes[0].set_title(
        f"评论长度分布（前 {len(before_len)} / 后 {len(after_len)}）"
        if _font_ready
        else f"Comment length (n={len(before_len)}/{len(after_len)})"
    )
    axes[0].legend()

    # 右：粉丝数（log 轴）。清洗最可能在这条线上引入偏差，必须单独看。
    if before_fans and after_fans:
        import numpy as np

        before_log = np.log10(np.asarray(before_fans, dtype=float))
        after_log = np.log10(np.asarray(after_fans, dtype=float))
        axes[1].hist(before_log, bins=30, density=True, alpha=0.45,
                     color="#6b7280", label="清洗前" if _font_ready else "before")
        axes[1].hist(after_log, bins=30, density=True, alpha=0.75,
                     color="#1d4ed8", label="清洗后" if _font_ready else "after")
        axes[1].set_xlabel("粉丝数 log10" if _font_ready else "followers log10")
        axes[1].set_title(
            "评论者粉丝数分布" if _font_ready else "Followers distribution"
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5, 0.5,
            "接口未返回粉丝数" if _font_ready else "no follower data",
            ha="center", va="center", transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    for axe in axes:
        axe.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "清洗影响：分布形状是否被改变" if _font_ready else "Cleaning impact",
        fontsize=11,
    )
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
def sentiment_distribution(result: Any, path: Path) -> Path | None:
    """整体情感分布（水平条形：中文标签不挤，比例一眼可比）。"""
    counts = result.distribution or {}
    total = sum(counts.values())
    if total == 0:
        return None
    plt = _setup()
    order = ["positive", "neutral", "negative"]
    values = [counts.get(label, 0) for label in order]
    figure, axes = plt.subplots(figsize=(6.4, 3.2))
    bars = axes.barh(
        [_label(label) for label in order], values,
        color=[_COLORS[label] for label in order], height=0.6,
    )
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value}（{value / total:.0%}）",
            va="center", fontsize=9,
        )
    axes.set_xlabel("评论数" if _font_ready else "comments")
    axes.set_title(
        f"整体情感分布（n={total}）" if _font_ready else f"Sentiment distribution (n={total})"
    )
    axes.invert_yaxis()
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, max(values) * 1.25)
    return _save(figure, path)


def sentiment_by_post(result: Any, path: Path, *, max_posts: int = 12) -> Path | None:
    """分帖子情感分布（堆叠条形）。"""
    per_post = result.per_post or {}
    if not per_post:
        return None
    posts = list(per_post.items())[:max_posts]
    order = ["positive", "neutral", "negative"]
    plt = _setup()
    figure, axes = plt.subplots(figsize=(7.2, 0.55 * len(posts) + 1.8))
    left = [0] * len(posts)
    for label in order:
        values = [per_post[post].get(label, 0) for post, _ in posts]
        axes.barh(
            [post[-6:] for post, _ in posts], values, left=left,
            color=_COLORS[label], label=_label(label), height=0.6,
        )
        left = [a + b for a, b in zip(left, values)]
    for index, (post, counts) in enumerate(posts):
        total = sum(counts.values())
        axes.text(left[index] + 0.2, index, f"n={total}", va="center", fontsize=8)
    axes.set_xlabel("评论数" if _font_ready else "comments")
    axes.set_title("分帖子情感分布" if _font_ready else "Sentiment by post")
    axes.legend(frameon=False, fontsize=8, loc="lower right")
    axes.invert_yaxis()
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, (max(left) if left else 1) * 1.2)
    return _save(figure, path)


def sentiment_trend(result: Any, path: Path) -> Path | None:
    """时间序列情感趋势（各极性占比按日）。"""
    daily = result.daily or {}
    if len(daily) < 2:
        # 只有一天时画"趋势"没有意义，直接不出图，避免误导
        return None
    order = ["positive", "neutral", "negative"]
    days = sorted(daily)
    plt = _setup()
    figure, axes = plt.subplots(figsize=(7.2, 3.4))
    for label in order:
        ratios = []
        for day in days:
            counts = daily[day]
            total = sum(counts.values()) or 1
            ratios.append(counts.get(label, 0) / total)
        axes.plot(days, ratios, marker="o", markersize=4,
                  color=_COLORS[label], label=_label(label), linewidth=1.6)
    axes.set_ylim(0, 1)
    axes.set_ylabel("占比" if _font_ready else "share")
    axes.set_title("情感倾向的时间变化" if _font_ready else "Sentiment over time")
    axes.legend(frameon=False, fontsize=8)
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    for tick in axes.get_xticklabels():
        tick.set_rotation(30)
        tick.set_fontsize(8)
    return _save(figure, path)


def stance_distribution(result: Any, path: Path) -> Path | None:
    """立场分布（水平条形）。

    顺序固定为 支持 / 反对 / 中立 / 无关，且「无关」用浅灰——
    它不该和三种表态并列争夺注意力，但必须出现，
    否则读者会以为全部评论都在谈这个议题。
    """
    counts = getattr(result, "distribution", None) or {}
    total = sum(counts.values())
    if not total:
        return None
    order = ["support", "oppose", "neutral", "irrelevant"]
    zh = {"support": "支持", "oppose": "反对", "neutral": "中立", "irrelevant": "无关"}
    colors = {"support": "#2e7d32", "oppose": "#c62828",
              "neutral": "#9e9e9e", "irrelevant": "#cfcfcf"}
    values = [counts.get(label, 0) for label in order]
    plt = _setup()
    figure, axes = plt.subplots(figsize=(6.4, 3.2))
    bars = axes.barh(
        [zh[label] if _font_ready else label for label in order], values,
        color=[colors[label] for label in order], height=0.6,
    )
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value}（{value / total:.0%}）", va="center", fontsize=9,
        )
    topic = getattr(result, "topic", "")
    axes.set_title(
        (f"立场分布（对象：{topic}，n={total}）" if _font_ready
         else f"Stance distribution (n={total})"),
        fontsize=10,
    )
    axes.set_xlabel("评论数" if _font_ready else "comments")
    axes.invert_yaxis()
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, max(values) * 1.25)
    return _save(figure, path)


def cluster_sizes(result: Any, path: Path) -> Path | None:
    """话语聚类的簇大小。"""
    sizes = getattr(result, "sizes", None) or {}
    if not sizes:
        return None
    plt = _setup()
    order = sorted(sizes, key=lambda key: -sizes[key])
    figure, axes = plt.subplots(figsize=(6.4, 0.5 * len(order) + 1.6))
    values = [sizes[key] for key in order]
    bars = axes.barh([f"簇 {key}" if _font_ready else f"c{key}" for key in order],
                     values, color="#1565c0", height=0.6)
    for bar, value in zip(bars, values):
        axes.text(bar.get_width() + max(values) * 0.02,
                  bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    silhouette = getattr(result, "silhouette", None)
    title = f"话语聚类规模（K={result.k}"
    title += f"，轮廓系数 {silhouette}）" if silhouette is not None else "）"
    axes.set_title(title if _font_ready else f"Cluster sizes (K={result.k})")
    axes.set_xlabel("评论数" if _font_ready else "comments")
    axes.invert_yaxis()
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, max(values) * 1.2)
    return _save(figure, path)


def community_scatter(result: Any, path: Path) -> Path | None:
    """用户社群的 PCA 二维散点。"""
    coords = getattr(result, "coords", None) or {}
    labels = getattr(result, "user_labels", None) or {}
    roles = getattr(result, "roles", None) or {}
    if not coords:
        return None
    plt = _setup()
    figure, axes = plt.subplots(figsize=(6.4, 5.0))
    palette = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00",
               "#00838f", "#4e342e", "#37474f"]
    for cluster_id in sorted(set(labels.values())):
        points = [coords[user] for user, label in labels.items()
                  if label == cluster_id and user in coords]
        if not points:
            continue
        axes.scatter(
            [point[0] for point in points], [point[1] for point in points],
            s=38, alpha=0.85, color=palette[cluster_id % len(palette)],
            label=f"{roles.get(cluster_id, cluster_id)}（{len(points)}人）"
            if _font_ready else f"c{cluster_id} (n={len(points)})",
        )
    explained = getattr(result, "explained_variance", None)
    subtitle = f"，PC1+PC2 解释方差 {explained:.0%}" if explained is not None else ""
    axes.set_title(
        (f"用户社群分布（n={result.n_users}{subtitle}）" if _font_ready
         else f"User communities (n={result.n_users})")
    )
    axes.set_xlabel("主成分 1" if _font_ready else "PC1")
    axes.set_ylabel("主成分 2" if _font_ready else "PC2")
    axes.legend(frameon=False, fontsize=8)
    axes.grid(alpha=0.2)
    axes.spines[["top", "right"]].set_visible(False)
    return _save(figure, path)


def sentiment_by_stance(table: Any, path: Path) -> Path | None:
    """各立场内部的情感构成（堆叠条形，按行内占比）。

    为什么用占比而不是绝对数：立场分组大小差十倍时，"反对组有 200 条负面"
    这种绝对数字会让人误以为只有反对组在生气。占比才看得到结构。
    被抑制的单元格不计入分母——否则一个被藏起来的格子会悄悄改掉比例。
    """
    from .stance_crosstab import row_shares

    groups = list(getattr(table, "groups", None) or [])
    cells_table = getattr(table, "table", None) or {}
    if not groups or not cells_table:
        return None
    shares = row_shares(table)
    attributes = list(getattr(table, "attributes", None) or [])
    colors = {"正面": "#2e7d32", "中性": "#9e9e9e", "负面": "#c62828"}

    plt = _setup()
    figure, axes = plt.subplots(figsize=(6.6, 0.62 * len(groups) + 2.0))
    positions = list(range(len(groups)))
    left = [0.0] * len(groups)
    for attribute in attributes:
        values = [(shares.get(group, {}).get(attribute) or 0.0) for group in groups]
        axes.barh(positions, values, left=left, height=0.58,
                  color=colors.get(attribute, "#90a4ae"),
                  label=attribute if _font_ready else attribute)
        for index, value in enumerate(values):
            if value >= 0.08:
                axes.text(
                    left[index] + value / 2, positions[index], f"{value:.0%}",
                    ha="center", va="center", fontsize=8, color="white",
                )
        left = [a + b for a, b in zip(left, values)]

    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{group}（n={getattr(table, 'group_totals', {}).get(group, 0)}）"
         for group in groups] if _font_ready else groups,
        fontsize=9,
    )
    axes.set_xlim(0, 1)
    axes.set_xlabel("行内占比" if _font_ready else "share within group")
    axes.set_title(
        "各立场内部的情感构成" if _font_ready else "Sentiment within each stance",
        fontsize=10,
    )
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    axes.invert_yaxis()
    axes.spines[["top", "right"]].set_visible(False)
    return _save(figure, path)


def _event_positions(buckets: Sequence[str], events: Sequence[Any]) -> list[tuple[float, str]]:
    """把事件日期映射到横轴位置。

    事件日期不一定落在某个桶上（比如事件当天没有评论），
    所以取"第一个不早于事件日的桶"——宁可把线画在事件之后一格，
    也不要画在之前：把事件标早了会让人误以为变化发生在事件前。
    """
    positions: list[tuple[float, str]] = []
    for event in events:
        date_text = str(getattr(event, "date", event) or "")
        if not date_text:
            continue
        for index, bucket in enumerate(buckets):
            if str(bucket)[:10] >= date_text:
                positions.append((float(index), date_text))
                break
    return positions


def _draw_events(axes: Any, positions: Sequence[tuple[float, str]]) -> None:
    for position, date_text in positions:
        axes.axvline(position, color="#6a1b9a", linestyle="-", linewidth=1.0,
                     alpha=0.7, zorder=2)
        axes.annotate(
            date_text[5:], xy=(position, 0.97), xycoords=("data", "axes fraction"),
            ha="left", va="top", fontsize=7, color="#6a1b9a",
            rotation=90, annotation_clip=False,
        )


def stance_trend(result: Any, path: Path, *, events: Sequence[Any] = ()) -> Path | None:
    """立场占比随时间的变化（支持 / 反对 为主，中立与无关为灰线）。

    为什么把「无关」也画出来：如果某段时间「无关」飙升，
    支持率与反对率的百分比会被机械地稀释——那是抽样变化，不是态度变化。
    不画它，读者就会把稀释读成"态度转向"。
    """
    buckets = list(getattr(result, "buckets", None) or [])
    shares = getattr(result, "stance_share", None) or {}
    if len(buckets) < 2 or not shares:
        return None
    low_n = set(getattr(result, "low_n_buckets", None) or [])
    styles = {
        "support": ("#2e7d32", "支持", "-", 1.7),
        "oppose": ("#c62828", "反对", "-", 1.7),
        "neutral": ("#78909c", "中立", ":", 1.1),
        "irrelevant": ("#bdbdbd", "无关", "--", 1.1),
    }
    plt = _setup()
    figure, axes = plt.subplots(figsize=(7.6, 3.6))
    positions = list(range(len(buckets)))
    for label, (color, zh, style, width) in styles.items():
        values = shares.get(label)
        if not values:
            continue
        axes.plot(positions, values, linestyle=style, color=color, linewidth=width,
                  label=zh if _font_ready else label, zorder=3)
        solid = [(x, v) for x, v in zip(positions, values) if buckets[x] not in low_n]
        hollow = [(x, v) for x, v in zip(positions, values) if buckets[x] in low_n]
        if solid:
            axes.plot([p[0] for p in solid], [p[1] for p in solid], "o",
                      markersize=4, color=color, zorder=4)
        if hollow:
            axes.plot([p[0] for p in hollow], [p[1] for p in hollow], "o",
                      markersize=4, markerfacecolor="white",
                      markeredgecolor=color, markeredgewidth=1.0, zorder=4)
    axes.set_ylim(0, 1)
    axes.set_xlim(-0.6, len(buckets) - 0.4)
    axes.set_xticks(positions)
    axes.set_xticklabels(buckets, rotation=45, fontsize=7)
    axes.set_ylabel("立场占比" if _font_ready else "share")
    axes.set_title(
        ("立场占比随时间（空心点＝该时段样本量过小）" if _font_ready
         else "Stance over time (hollow = low n)"),
        fontsize=10,
    )
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    _draw_events(axes, _event_positions(buckets, events))
    return _save(figure, path)


def timeseries_evolution(result: Any, path: Path, *, events: Sequence[Any] = ()) -> Path | None:
    """时间演变：左轴情感占比折线 + 右轴评论量柱。

    为什么把量的柱和占比的线画在一起：只画占比时，一个只有 1 条评论的时段
    会画出"100% 负面"的尖峰，看着像舆情突变。柱子让人一眼看出那是噪音。
    低样量时段的点用空心标记，进一步提醒别读它。
    """
    buckets = list(getattr(result, "buckets", None) or [])
    if len(buckets) < 2:
        return None
    volume = list(getattr(result, "volume", None) or [])
    shares = getattr(result, "sentiment_share", None) or {}
    low_n = set(getattr(result, "low_n_buckets", None) or [])
    if not shares and not volume:
        return None

    plt = _setup()
    figure, axes = plt.subplots(figsize=(7.6, 3.8))
    positions = list(range(len(buckets)))

    # 右轴：评论量（浅灰柱，作为"这段数据有多可信"的底噪参考）
    axes_volume = axes.twinx()
    axes_volume.bar(positions, volume, color="#cfcfcf", width=0.62, zorder=0)
    axes_volume.set_ylabel(
        "评论数" if _font_ready else "comments", fontsize=9, color="#777777"
    )
    axes_volume.tick_params(axis="y", labelsize=8, colors="#777777")
    axes_volume.set_zorder(0)
    axes.set_zorder(1)
    axes.patch.set_visible(False)
    axes_volume.set_ylim(0, max(volume) * 2.2 if volume else 1)

    for label in ("positive", "neutral", "negative"):
        values = shares.get(label)
        if not values:
            continue
        axes.plot(
            positions, values, color=_COLORS[label], linewidth=1.6,
            label=_label(label), zorder=3,
        )
        solid = [(x, v) for x, v in zip(positions, values) if buckets[x] not in low_n]
        hollow = [(x, v) for x, v in zip(positions, values) if buckets[x] in low_n]
        if solid:
            axes.plot([p[0] for p in solid], [p[1] for p in solid],
                      "o", markersize=4.5, color=_COLORS[label], zorder=4)
        if hollow:
            axes.plot([p[0] for p in hollow], [p[1] for p in hollow],
                      "o", markersize=4.5, markerfacecolor="white",
                      markeredgecolor=_COLORS[label], markeredgewidth=1.2, zorder=4)

    axes.set_ylim(0, 1)
    axes.set_xlim(-0.6, len(buckets) - 0.4)
    axes.set_xticks(positions)
    axes.set_xticklabels(buckets, rotation=30, fontsize=8)
    axes.set_ylabel("占比" if _font_ready else "share", fontsize=9)
    axes.set_title(
        (
            f"情感占比的时间变化（n={getattr(result, 'n_total', 0)}，"
            "空心点＝样本量过小）"
        ) if _font_ready else "Sentiment over time (hollow = low n)",
        fontsize=10,
    )
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    _draw_events(axes, _event_positions(buckets, events))
    return _save(figure, path)


def reply_forest(result: Any, path: Path) -> Path | None:
    """回复结构图：把每个讨论串画成一棵横躺的树。

    为什么横躺（x = 层级，y = 节点顺序）：中文标题与文字在横向空间里好排，
    而且"深不深"一眼可见——向右伸得越远，说明对话层级越多。

    只画已经通过小结构抑制的串。图下方直接写明"节点数 <N 的串未画出"，
    避免读者把"图上只有 2 个串"误读成"只有 2 个串存在"。
    """
    threads = list(getattr(result, "top_threads", None) or [])
    if not threads:
        return None
    plt = _setup()
    figure_height = min(3.0 + 0.55 * sum(min(t.size, 6) for t in threads), 11.0)
    figure, axes = plt.subplots(figsize=(7.4, figure_height))

    offset = 0.0
    for index, thread in enumerate(threads, start=1):
        nodes = list(getattr(thread, "nodes", None) or [])
        if not nodes:
            continue
        positions = _tree_positions(nodes)
        if not positions:
            continue
        lowest = min(y for _x, y in positions.values())
        shifted = {
            cid: (x, y - lowest + offset) for cid, (x, y) in positions.items()
        }
        height = max(y for _x, y in shifted.values()) - offset

        for cid, (x, y) in shifted.items():
            parent = next(
                (node.get("parent_cid") for node in nodes if node.get("cid") == cid),
                None,
            )
            if parent and parent in shifted:
                px, py = shifted[parent]
                axes.plot([px, x], [py, y], color="#9e9e9e", linewidth=0.9, zorder=1)
        for cid, (x, y) in shifted.items():
            is_root = x == 0
            axes.scatter(
                [x], [y], s=64 if is_root else 34,
                color="#1565c0" if is_root else "#90a4ae",
                zorder=3, edgecolors="white", linewidths=0.8,
            )
        axes.annotate(
            f"#{index}（{thread.size} 条，深 {thread.depth}）"
            if _font_ready else f"#{index} (n={thread.size}, d={thread.depth})",
            xy=(-0.35, offset + height / 2),
            ha="right", va="center", fontsize=8, color="#37474f",
            annotation_clip=False,
        )
        offset += height + 1.4

    max_level = max(
        (max(x for x, _y in _tree_positions(list(t.nodes)).values()) if t.nodes else 0)
        for t in threads
    )
    axes.set_xlim(-0.4, max_level + 0.6)
    axes.set_xticks(range(max_level + 1))
    axes.set_xticklabels(
        ["根评论"] + [f"第{i}层回复" for i in range(1, max_level + 1)]
        if _font_ready else ["root"] + [f"L{i}" for i in range(1, max_level + 1)],
        fontsize=8,
    )
    axes.set_yticks([])
    axes.set_title(
        ("回复结构（仅画出节点数 ≥%d 的讨论串）" % getattr(result, "min_n", 5))
        if _font_ready else "Reply structure (threads with n >= %d)" % getattr(result, "min_n", 5),
        fontsize=10,
    )
    axes.spines[["top", "right", "left"]].set_visible(False)
    suppressed = getattr(result, "suppressed_threads", 0)
    if suppressed:
        axes.text(
            0.0, -0.06,
            (f"另有 {suppressed} 个讨论串节点数不足，未画出" if _font_ready
             else f"{suppressed} more thread(s) not drawn"),
            transform=axes.transAxes, fontsize=8, color="#c62828",
        )
    return _save(figure, path)


def _tree_positions(nodes: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """计算树形布局：x = 层级，y = 叶节点顺序（父节点取子节点均值）。

    截断过的串里可能有节点的父节点不在列表内，这类节点按新的根处理，
    否则会整棵树消失——宁可多画一个孤立点，也不要静默丢节点。
    """
    index = {str(node.get("cid")): node for node in nodes if node.get("cid")}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for cid in index:
        parent = index[cid].get("parent_cid")
        if parent and str(parent) in index:
            children.setdefault(str(parent), []).append(cid)
        else:
            roots.append(cid)

    levels: dict[str, int] = {}
    queue = [(cid, 0) for cid in roots]
    while queue:
        cid, level = queue.pop(0)
        if cid in levels:
            continue
        levels[cid] = level
        for child in children.get(cid, []):
            queue.append((child, level + 1))

    positions: dict[str, tuple[float, float]] = {}
    counter = [0.0]

    def place(cid: str) -> float:
        kids = children.get(cid, [])
        if not kids:
            y = counter[0]
            counter[0] += 1.0
        else:
            ys = [place(child) for child in kids]
            y = sum(ys) / len(ys)
        positions[cid] = (float(levels.get(cid, 0)), y)
        return y

    for root in roots:
        place(root)
    return positions


def compare_sentiment(result: Any, path: Path) -> Path | None:
    """跨话题情感占比对比（分组水平条形）。

    只在占比可比时才出图；不可比时**直接不出图**，让报告里的「—」和文字说明
    成为唯一信息来源——一张画出来的图比一张表格更容易被当作结论。
    """
    profiles = list(getattr(result, "profiles", None) or [])
    if len(profiles) < 2 or not getattr(result, "share_comparable", False):
        return None
    labelled = [profile for profile in profiles if profile.sentiment_n]
    if len(labelled) < 2:
        return None

    plt = _setup()
    import numpy as np

    topics = [(profile.tag or profile.run_id)[:12] for profile in labelled]
    positions = np.arange(len(topics))
    width = 0.26
    figure, axes = plt.subplots(figsize=(7.4, 0.9 * len(topics) + 2.2))
    for offset, label in enumerate(("positive", "neutral", "negative")):
        values = [profile.sentiment.get(label, 0.0) for profile in labelled]
        axes.barh(
            positions + (offset - 1) * width, values, height=width,
            color=_COLORS[label], label=_label(label),
        )
        for index, value in enumerate(values):
            axes.text(
                value + 0.01, positions[index] + (offset - 1) * width,
                f"{value:.0%}", va="center", fontsize=7,
            )
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{name}（n={profile.sentiment_n}）" for name, profile in zip(topics, labelled)]
        if _font_ready else [f"{name} (n={profile.sentiment_n})"
                             for name, profile in zip(topics, labelled)],
        fontsize=8,
    )
    axes.set_xlim(0, 1.18)
    axes.set_xlabel("占比" if _font_ready else "share")
    axes.set_title(
        "各话题情感占比对比（仅占比可比时出图）" if _font_ready
        else "Sentiment share by topic",
        fontsize=10,
    )
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    axes.invert_yaxis()
    axes.grid(axis="x", alpha=0.2)
    axes.spines[["top", "right"]].set_visible(False)
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# 可直接选择的图表清单
# --------------------------------------------------------------------------- #
#: 用户可选的图表：(key, 文件名, 中文名, 依赖)。依赖用于判断"这次分析有没有这个结果"，
#: 缺依赖就安静跳过，而不是画一张空图。
CHART_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("sentiment_distribution", "sentiment_distribution.png", "整体情感分布", "sentiment"),
    ("sentiment_by_post", "sentiment_by_post.png", "分帖子情感分布", "sentiment"),
    ("sentiment_trend", "sentiment_trend.png", "情感时间趋势", "sentiment"),
    ("stance_distribution", "stance_distribution.png", "立场分布", "stance"),
    ("sentiment_by_stance", "sentiment_by_stance.png", "各立场的情感构成", "stance_crosstab"),
    ("discourse_cluster_sizes", "discourse_cluster_sizes.png", "话语簇规模", "discourse"),
    ("community_pca", "community_pca.png", "用户社群（PCA）", "community"),
    ("timeseries_evolution", "timeseries_evolution.png", "评论量与结构的时间演变", "timeseries"),
    ("stance_trend", "stance_trend.png", "立场占比的时间变化", "timeseries"),
    ("activity_heatmap", "activity_heatmap.png", "发布时段热力图（星期×小时）", "rows"),
    ("term_frequency", "term_frequency.png", "高频词", "rows"),
    ("reply_forest", "reply_forest.png", "回复结构", "replies"),
    ("compare_sentiment", "compare_sentiment.png", "跨话题对比", "compare"),
    ("cleaning_comparison", "cleaning_comparison.png", "清洗前后分布对比", "cleaning"),
)

CHART_LABELS: dict[str, str] = {key: label for key, _file, label, _needs in CHART_SPECS}


def chart_keys() -> list[str]:
    return [key for key, _file, _label, _needs in CHART_SPECS]


def activity_heatmap(rows: Sequence[dict[str, Any]], path: Path) -> Path | None:
    """评论发布时段热力图：星期（行）× 小时（列）。

    社科写作里"什么时候有人在说"经常和"说了什么"一样重要——
    它决定了采集时段能否代表日常讨论，也能看出工作时段/深夜的舆论节奏差异。
    """
    from datetime import datetime as _dt

    import numpy as np

    grid = np.zeros((7, 24), dtype=float)
    for row in rows:
        raw = row.get("created_at")
        if not raw:
            continue
        try:
            when = _dt.fromisoformat(str(raw))
        except ValueError:
            continue
        grid[when.weekday()][when.hour] += 1
    if grid.sum() == 0:
        return None

    plt = _setup()
    figure, axes = plt.subplots(figsize=(9.0, 3.2))
    image = axes.imshow(grid, aspect="auto", cmap="YlOrRd")
    axes.set_yticks(range(7))
    axes.set_yticklabels(
        ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        if _font_ready else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    axes.set_xticks(range(0, 24, 3))
    axes.set_xticklabels([f"{hour:02d}" for hour in range(0, 24, 3)])
    axes.set_xlabel("小时" if _font_ready else "hour")
    axes.set_title(
        f"发布时段分布（n={int(grid.sum())}）" if _font_ready else "Activity by weekday/hour"
    )
    figure.colorbar(image, ax=axes, shrink=0.85, label="评论数" if _font_ready else "count")
    figure.tight_layout()
    return _save(figure, path)


def term_frequency(
    rows: Sequence[dict[str, Any]], path: Path, *, top: int = 15
) -> Path | None:
    """高频词条形图（用与聚类同一套分词，避免"图里的词"和"簇里的词"不一致）。"""
    from .cluster import tokenize

    counter: dict[str, int] = {}
    for row in rows:
        text = str(row.get("cleaned_text") or row.get("text_raw") or "")
        for token in tokenize(text):
            if len(token) < 2:
                continue
            counter[token] = counter.get(token, 0) + 1
    if not counter:
        return None
    ordered = sorted(counter.items(), key=lambda item: -item[1])[:top]
    words = [word for word, _count in ordered][::-1]
    values = [count for _word, count in ordered][::-1]

    plt = _setup()
    figure, axes = plt.subplots(figsize=(6.4, max(2.6, 0.32 * len(words))))
    axes.barh(words, values, color="#1d4ed8", height=0.65)
    for index, value in enumerate(values):
        axes.text(value, index, f" {value}", va="center", fontsize=8)
    axes.set_xlabel("出现次数" if _font_ready else "count")
    axes.set_title(
        f"高频词（前 {len(words)}）" if _font_ready else f"Top {len(words)} terms"
    )
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_xlim(0, max(values) * 1.15)
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
def render_all(
    output_dir: Path,
    *,
    sentiment: Any | None = None,
    discourse: Any | None = None,
    community: Any | None = None,
    timeseries: Any | None = None,
    replies: Any | None = None,
    stance: Any | None = None,
    stance_crosstab: Any | None = None,
    compare: Any | None = None,
    events: Sequence[Any] = (),
    rows: Sequence[dict[str, Any]] | None = None,
    cleaning_rows: tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]]] | None = None,
    include: Sequence[str] | None = None,
) -> list[Path]:
    """按需画图，返回生成的文件列表。

    Args:
        rows: 评论行（用于时段热力图与高频词）。缺它就跳过这两张，不报错。
        cleaning_rows: ``(raw_rows, kept_rows)``，用于清洗影响图。
        include: 想画的图表 key（见 ``CHART_SPECS``）；``None`` 表示全画。
            文件名也接受，方便命令行直接写 ``--charts sentiment_trend``。

    任何一张图失败都不应中断整体——图表是"锦上添花"，
    不能因为它让分析结果丢掉。
    """
    output_dir = Path(output_dir)
    # 注意区分 None 与空列表：None = 全画（没指定）；[] = 一张都不画（用户明确全取消）
    wanted = {item.strip() for item in include} if include is not None else None

    def keep(key: str, filename: str) -> bool:
        if wanted is None:
            return True
        return key in wanted or filename in wanted

    produced: list[Path] = []
    plan: list[tuple[str, Any]] = []
    if sentiment is not None:
        plan += [
            ("sentiment_distribution.png",
             lambda path: sentiment_distribution(sentiment, path)),
            ("sentiment_by_post.png",
             lambda path: sentiment_by_post(sentiment, path)),
            ("sentiment_trend.png",
             lambda path: sentiment_trend(sentiment, path)),
        ]
    if stance is not None:
        plan.append(("stance_distribution.png",
                     lambda path: stance_distribution(stance, path)))
    if stance_crosstab is not None:
        plan.append(("sentiment_by_stance.png",
                     lambda path: sentiment_by_stance(stance_crosstab, path)))
    if discourse is not None:
        plan.append(("discourse_cluster_sizes.png",
                     lambda path: cluster_sizes(discourse, path)))
    if community is not None:
        plan.append(("community_pca.png",
                     lambda path: community_scatter(community, path)))
    if timeseries is not None:
        plan.append(("timeseries_evolution.png",
                     lambda path: timeseries_evolution(timeseries, path, events=events)))
        plan.append(("stance_trend.png",
                     lambda path: stance_trend(timeseries, path, events=events)))
    if rows:
        plan.append(("activity_heatmap.png", lambda path: activity_heatmap(rows, path)))
        plan.append(("term_frequency.png", lambda path: term_frequency(rows, path)))
    if replies is not None:
        plan.append(("reply_forest.png", lambda path: reply_forest(replies, path)))
    if compare is not None:
        plan.append(("compare_sentiment.png",
                     lambda path: compare_sentiment(compare, path)))
    if cleaning_rows is not None:
        raw_rows, kept_rows = cleaning_rows
        plan.append(("cleaning_comparison.png",
                     lambda path: cleaning_comparison(raw_rows, kept_rows, path)))

    for filename, draw in plan:
        key = filename.removesuffix(".png")
        if not keep(key, filename):
            continue
        target = output_dir / filename
        try:
            saved = draw(target)
        except Exception as exc:  # noqa: BLE001
            note(f"图表 {filename} 生成失败：{type(exc).__name__}: {exc}")
            continue
        if saved is not None:
            produced.append(saved)
    return produced
