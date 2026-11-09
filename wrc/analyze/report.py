"""报告层（模块 F1 的"报告层"雏形）。

论文要的不是一堆 CSV，而是一份能看懂、能改、能引用的文档：
方法卡片 + 核心图表 + 清洗影响报告 + 已知局限。

这里产出的是 **Markdown**，不是 Word/PDF。理由：
- Markdown 在 Word / Typora / 网页里都能直接打开或粘贴；
- 研究者要做的是"二次编辑"，而 .docx 生成后再改格式很别扭；
- 转 PDF 可以在浏览器里一步完成，工具不必背上排版引擎。

已知局限：PDF 需要排版引擎，工具不生成 PDF；请用 Word 的「另存为 PDF」。
Word（.docx）导出见 ``wrc.analyze.docx_export``。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .. import __version__
from .method_card import build_method_card

_TITLE = {
    "sentiment": "情感分析",
    "cluster_discourse": "话语聚类",
    "cluster_community": "用户社群聚类",
}


def build_analysis_report(
    *,
    run: dict[str, Any],
    cleaning: dict[str, Any] | None = None,
    sentiment: Any | None = None,
    discourse: Any | None = None,
    community: Any | None = None,
    crosstab: Any | None = None,
    timeseries: Any | None = None,
    replies: Any | None = None,
    stance: Any | None = None,
    stance_crosstab: Any | None = None,
    discourse_stance: Any | None = None,
    event_results: Sequence[Any] = (),
    changepoint_text: str = "",
    term_text: str = "",
    collection_text: str = "",
    charts: Sequence[Path] = (),
    output_dir: Path | None = None,
) -> str:
    """拼装完整分析报告（Markdown）。"""
    lines: list[str] = []
    lines.append(f"# 微博舆论分析报告 · {run.get('tag', '')}".rstrip(" ·"))
    lines.append("")
    lines.append(f"- 采集会话：`{run.get('run_id', '—')}`")
    if run.get("created_at"):
        lines.append(f"- 采集时间：{run['created_at']}（北京时间）")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 工具版本：`{__version__}`")
    lines.append("")
    lines.append(
        "> 本报告由博研助手生成，用于论文写作的初稿。**方法与参数是实际运行值**，"
        "但研究设计、抽样理由与伦理审批须由研究者自行撰写；"
        "所有自动标注结果建议先人工抽查再采用。"
    )
    lines.append("")
    lines.append(
        "> **本报告的定位**：回答「网友对这项政策是支持还是反对」，"
        "并描述讨论结构在政策**实施期**如何演变。"
        "它**不做事件研究**——第九节的「话语转折点扫描」只用于定位候选窗口，"
        "不能当作事件效应写入论文（理由见 `docs/DECISIONS/013`）。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---------- 采集口径（不编号：它是方法前提，不是分析步骤） ----------
    if collection_text:
        lines.append("## 采集口径与样本边界")
        lines.append("")
        lines.append("```")
        lines.append(collection_text)
        lines.append("```")
        lines.append("")

    # ---------- 一、清洗 ----------
    if cleaning:
        lines.append("## 一、数据清洗")
        lines.append("")
        total = cleaning.get("total_input")
        kept = cleaning.get("total_kept")
        rate = cleaning.get("retention_rate")
        lines.append(f"- 清洗档位：{cleaning.get('tier', '—')}")
        lines.append(f"- 管线版本：`{cleaning.get('pipeline_version', '—')}`")
        if total and kept is not None:
            rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "—"
            lines.append(f"- 输入 {total} 条，保留 {kept} 条（{rate_text}）")
        steps = cleaning.get("steps") or []
        if steps:
            lines.append("")
            lines.append("| 步骤 | 移除 | 占输入 |")
            lines.append("| --- | ---: | ---: |")
            for step in steps:
                share = step.get("share")
                share_text = f"{share:.1%}" if isinstance(share, (int, float)) else "—"
                lines.append(
                    f"| {step.get('label', '')} | {step.get('removed', 0)} | {share_text} |"
                )
        chart = _find(charts, "cleaning_comparison.png", output_dir)
        if chart:
            lines.append("")
            lines.append(f"![清洗前后分布对比]({chart})")
            lines.append("")
            lines.append(
                "> 看这张图只问一件事：前后两条分布是不是**同一个形状**。"
                "如果被删掉的集中在短文本或低粉丝账号，形状会变形——那说明清洗引入了"
                "系统性偏差，需要调档位或在论文里交代。只看条数减少多少看不出这个。"
            )
        lines.append("")

    # ---------- 二、情感分析 ----------
    if sentiment is not None:
        lines.append("## 二、情感分析")
        lines.append("")
        lines.append("```")
        lines.append(sentiment.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "sentiment_distribution.png", output_dir)
        if chart:
            lines.append(f"![整体情感分布]({chart})")
            lines.append("")
        chart = _find(charts, "sentiment_by_post.png", output_dir)
        if chart:
            lines.append(f"![分帖子情感分布]({chart})")
            lines.append("")
        chart = _find(charts, "sentiment_trend.png", output_dir)
        if chart:
            lines.append(f"![情感时间趋势]({chart})")
            lines.append("")

    # ---------- 三、立场 ----------
    if stance is not None:
        lines.append("## 三、立场分布（支持 / 反对）")
        lines.append("")
        lines.append("```")
        lines.append(stance.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "stance_distribution.png", output_dir)
        if chart:
            lines.append(f"![立场分布]({chart})")
            lines.append("")
        # 情感 × 立场 的交叉才是真正的分析点：
        # "情绪负面但立场支持"这类格子，正是情感标签单独看不出来的东西。
        lines.append(
            "> 情绪与立场是两个独立维度：同一条评论可以「情绪负面、立场支持」"
            "（如担心政策执行伤及一线工人）。判读结论时请同时看第二节与本节，"
            "不要用情感分布代替立场分布。"
        )
        lines.append("")

    # ---------- 四、话语聚类 ----------
    if discourse is not None:
        lines.append("## 四、话语聚类")
        lines.append("")
        lines.append("```")
        lines.append(discourse.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "term_frequency.png", output_dir)
        if chart:
            lines.append(f"![高频词]({chart})")
            lines.append("")
            lines.append(
                "> 高频词用的是与聚类同一套分词，所以图里的词和簇的代表词对得上；"
                "它描述的是**这批样本**说了什么，不是平台全量。"
            )
            lines.append("")
        chart = _find(charts, "discourse_cluster_sizes.png", output_dir)
        if chart:
            lines.append(f"![话语聚类规模]({chart})")
            lines.append("")

    # ---------- 四、用户社群 ----------
    if community is not None:
        lines.append("## 五、用户社群聚类")
        lines.append("")
        lines.append("```")
        lines.append(community.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "community_pca.png", output_dir)
        if chart:
            lines.append(f"![用户社群分布]({chart})")
            lines.append("")

    # ---------- 五、交叉表 ----------
    has_any_crosstab = (
        crosstab is not None
        or stance_crosstab is not None
        or discourse_stance is not None
    )
    if has_any_crosstab:
        lines.append("## 六、交叉表（含小群体抑制）")
        lines.append("")
        lines.append(
            "> 抑制规则：任何分组样本量小于最小可展示样本时，该行或该单元格不予展示，"
            "且**不计入行内占比的分母**。这不是排版省略，而是伦理约束——"
            "小群体交叉表可能反推个体。"
        )
        lines.append("")
    if crosstab is not None:
        lines.append("### 一、话语簇 × 情感")
        lines.append("")
        lines.append("```")
        lines.append(crosstab.render())
        lines.append("```")
        lines.append("")
    if stance_crosstab is not None:
        from .stance_crosstab import render as render_stance_crosstab

        lines.append("### 二、情感 × 立场")
        lines.append("")
        lines.append("```")
        lines.append(render_stance_crosstab(
            stance_crosstab,
            title=f"情感 × 立场 交叉表（最小可展示样本 n≥{stance_crosstab.min_n}）",
            group_label="立场",
        ))
        lines.append("```")
        lines.append("")
        chart = _find(charts, "sentiment_by_stance.png", output_dir)
        if chart:
            lines.append(f"![各立场的情感构成]({chart})")
            lines.append("")
    if discourse_stance is not None:
        from .stance_crosstab import render as render_stance_crosstab

        lines.append("### 三、话语簇 × 立场")
        lines.append("")
        lines.append("```")
        lines.append(render_stance_crosstab(
            discourse_stance,
            title=f"话语簇 × 立场 交叉表（最小可展示样本 n≥{discourse_stance.min_n}）",
            group_label="话语簇",
        ))
        lines.append("```")
        lines.append("")
        lines.append(
            "> 读法：同一列（如「反对」）在不同簇里的占比差异，"
            "才是「不同话语阵营态度不同」的证据；只比计数会被簇大小带偏。"
        )
        lines.append("")

    # ---------- 六、时间演变 ----------
    if timeseries is not None:
        lines.append("## 七、时间演变")
        lines.append("")
        lines.append("```")
        lines.append(timeseries.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "activity_heatmap.png", output_dir)
        if chart:
            lines.append(f"![发布时段热力图]({chart})")
            lines.append("")
            lines.append(
                "> 热力图看的是「什么时候有人在说」，不是「说了什么」。"
                "它能说明这批样本集中在工作日还是深夜，从而决定能不能声称"
                "覆盖了日常讨论节奏。"
            )
            lines.append("")
        chart = _find(charts, "timeseries_evolution.png", output_dir)
        if chart:
            lines.append(f"![情感占比的时间变化]({chart})")
            lines.append("")
        chart = _find(charts, "stance_trend.png", output_dir)
        if chart:
            lines.append(f"![立场占比的时间变化]({chart})")
            lines.append("")
            lines.append(
                "> 「无关」那条灰线要看一眼：它涨的时候，支持率与反对率的百分比"
                "会被机械稀释，那是抽样结构变化，不是态度变化。"
            )
            lines.append("")
        if term_text:
            lines.append("### 术语随时间的变化")
            lines.append("")
            lines.append("```")
            lines.append(term_text)
            lines.append("```")
            lines.append("")
            lines.append(
                "> 这一段回答的是「**关注点在变什么**」，不是「态度在变什么」。"
                "两者常被混为一谈：立场占比稳定而术语迁移，说明讨论在具体化，"
                "而不是态度翻转。"
            )
            lines.append("")

    # ---------- 七、回复结构 ----------
    if replies is not None:
        lines.append("## 八、二级评论回复结构")
        lines.append("")
        lines.append("```")
        lines.append(replies.render())
        lines.append("```")
        lines.append("")
        chart = _find(charts, "reply_forest.png", output_dir)
        if chart:
            lines.append(f"![回复结构]({chart})")
            lines.append("")

    # ---------- 九、事件前后对比 ----------
    # 这一节**总是出现**：报告结构固定，读者才能按"一到九"逐节核对。
    # 没做事件研究就明说没做、以及怎么做，而不是让第九节凭空消失。
    if not event_results and not changepoint_text:
        lines.append("## 九、事件前后对比")
        lines.append("")
        lines.append(
            "本次没有做事件前后对比：没有指定事件日期，也没有做话语转折点扫描。"
        )
        lines.append("")
        lines.append(
            "> 事件日期必须由研究者指定（例如政策公布日）。工具不会拿"
            "「评论量最大的那天」冒充事件——那是用结果定义原因。"
            "要补做：在「4 · 分析与图表」填事件日期（格式 2026-09-29），"
            "或命令行 `--event 2026-09-29:政策公布:7:7`。"
        )
        lines.append("")
    if event_results or changepoint_text:
        from .events import render_all as render_events

        lines.append("## 九、事件前后对比")
        lines.append("")
        if event_results:
            lines.append("```")
            lines.append(render_events(event_results))
            lines.append("```")
            lines.append("")
            chart = _find(charts, "stance_trend.png", output_dir)
            if chart:
                lines.append(f"![立场随时间（竖线为事件节点）]({chart})")
                lines.append("")
            lines.append(
                "> 事件日期由研究者指定。**本节的 p 值只是参考**："
                "用不用统计检验、显著性水平取多少，属于研究设计与期刊口径问题。"
            )
            lines.append("")
        if changepoint_text:
            lines.append("### 话语转折点扫描（不是事件研究）")
            lines.append("")
            lines.append("```")
            lines.append(changepoint_text)
            lines.append("```")
            lines.append("")
            lines.append(
                "> 这一小节只在研究者要求时生成。它的作用是**定位候选窗口**，"
                "不是认定事件——把变动最大的日期直接当成事件日，就是用结果定义原因。"
            )
            lines.append("")

    # ---------- 方法卡片 ----------
    lines.append("---")
    lines.append("")
    lines.append("# 方法卡片")
    lines.append("")
    if sentiment is not None:
        lines.append(build_method_card(
            module="sentiment", method=sentiment.method, cleaning=cleaning, run=run,
            limitations=[
                "情感标注由模型完成，未做人工双编码校验；"
                "信度抽样只反映通道间一致性，不等于编码员间信度。",
                "本地对照通道为多语种通用模型，在中文微博口语上置信度偏低，仅作粗筛参照。",
            ],
        ))
        lines.append("")
    if stance is not None:
        lines.append(build_method_card(
            module="stance", method=stance.method, cleaning=cleaning, run=run,
            limitations=[
                "立场标注由大语言模型完成，未经双人编码校验；"
                "研究者人工复核只覆盖抽取的样本。",
                "「无关」类别不计入支持率与反对率；论文中必须写明分母口径"
                "（有效表态数 或 支持+反对）。",
                "同一条评论可能同时含支持与反对的表述，本分析只取主导立场。",
            ],
        ))
        lines.append("")
    if discourse is not None:
        lines.append(build_method_card(
            module="cluster_discourse", method=discourse.method,
            cleaning=cleaning, run=run,
            limitations=[
                "聚类结果依赖句向量模型与随机初始化；已固定 random_state 以便复现。",
                "代表性评论为距质心最近者，代表该簇平均话语，不等于随机抽样。",
            ],
        ))
        lines.append("")
    if community is not None:
        lines.append(build_method_card(
            module="cluster_community", method=community.method,
            cleaning=cleaning, run=run,
            limitations=[
                "角色标签是描述性命名，不是对用户性质的判断。",
                "小样本下聚类不稳定，建议 300 条以上再做社群解读。",
            ],
        ))
        lines.append("")
    if timeseries is not None:
        lines.append(build_method_card(
            module="timeseries", method=timeseries.method,
            cleaning=cleaning, run=run,
            limitations=[
                "曲线描述的是已采集样本的时间分布，不是平台全量舆论的演化；"
                "评论取自热门评论流，早期评论可能整体缺席。",
                "样本量过小的时段占比不可解读，报告已标注并排除出趋势判断。",
                "趋势判断只是前后两段占比的描述性对比，未做统计检验。",
                "立场占比随时间的分母是「该时段有立场标签的评论数」；"
                "「无关」占比上升会稀释支持率与反对率。",
            ],
        ))
        lines.append("")
    if replies is not None:
        lines.append(build_method_card(
            module="replies", method=replies.method, cleaning=cleaning, run=run,
            limitations=[
                "二级评论默认不采集；若本研究未开启该选项，回复结构分析不成立。",
                "父评论被清洗移除后，其回复会被挂到该帖虚拟根，层级可能比原始数据浅。",
                "节点数小于最小可展示样本的讨论串不予展示，以免对应到个别对话。",
            ],
        ))
        lines.append("")
    if event_results:
        first = event_results[0]
        lines.append(build_method_card(
            module="event",
            method={
                "module": "event",
                "event": first.event.to_dict(),
                "metrics": sorted({item.metric for item in event_results}),
                "min_window_n": 30,
                "test": "两比例 z 检验（正态近似，期望频数 ≥5 才报告 p 值）",
                "window_rule": "前段 [事件日−N, 事件日)；后段 [事件日, 事件日+N]",
            },
            cleaning=cleaning, run=run,
            limitations=[
                "这是描述性对比，不是因果推断；不能据此声称事件「导致」了变化。",
                "事件日期由研究者依据外部信息确定，日期若有偏差结论随之偏移。",
                "同期若发生其它事件，前后差异无法归因到单一事件。",
                "窗口长度会直接影响结果；论文中应说明窗口为何这样取。",
            ],
        ))
        lines.append("")

    # ---------- 六、局限 ----------
    lines.append("---")
    lines.append("")
    lines.append("## 已知局限（务必在论文中披露）")
    lines.append("")
    for item in (
        "评论来自平台「热门评论流」接口，偏向高互动评论，不代表该帖全部评论的分布。",
        "检索结果的排序与平台站内排序不完全一致；本地重排不等于平台排序。",
        "检索结果的部分时间字段仅精确到「日」，已在数据中以 created_at_precision 标注。",
        "所有自动标注（情感、聚类）均未经人工双编码校验，正式研究前应抽样复核。",
        "事件前后对比为描述性结果；事件日期与窗口由研究者设定，"
        "不能据此作因果推断。",
    ):
        lines.append(f"- {item}")
    lines.append("")

    if charts:
        lines.append("## 图表清单")
        lines.append("")
        for chart in charts:
            lines.append(f"- `{Path(chart).name}`")
        lines.append("")

    return "\n".join(lines)


def _find(charts: Sequence[Path], name: str, output_dir: Path | None) -> str | None:
    """在图表列表里找某一张，返回相对路径（便于 markdown 在导出目录内渲染）。"""
    for chart in charts:
        if Path(chart).name == name:
            if output_dir is not None:
                try:
                    return str(Path(chart).relative_to(output_dir)).replace("\\", "/")
                except ValueError:
                    pass
            return Path(chart).name
    return None


class _Rendered:
    """把数据库里存的摘要包装成"看起来像分析结果"的对象。

    这样 `build_analysis_report` 既能吃实时的分析结果，也能吃历史摘要——
    于是报告可以随时从数据库重新生成，不必重跑 LLM。
    """

    def __init__(self, summary: dict[str, Any]) -> None:
        self._summary = summary or {}

    def render(self) -> str:
        return str(self._summary.get("_rendered") or "（该次分析未保存渲染文本）")

    @property
    def method(self) -> dict[str, Any]:
        return dict(self._summary.get("method") or {})


def build_report_from_summaries(
    *,
    run: dict[str, Any],
    cleaning: dict[str, Any] | None,
    summaries: dict[str, dict[str, Any]],
    crosstab: Any | None = None,
    timeseries: Any | None = None,
    replies: Any | None = None,
    stance: Any | None = None,
    stance_crosstab: Any | None = None,
    discourse_stance: Any | None = None,
    event_results: Sequence[Any] = (),
    changepoint_text: str = "",
    term_text: str = "",
    collection_text: str = "",
    charts: Sequence[Path] = (),
    output_dir: Path | None = None,
) -> str:
    """只用数据库里存的分析摘要重建完整报告。

    Args:
        summaries: ``{module: summary_dict}``，module 取值
            ``sentiment`` / ``cluster_discourse`` / ``cluster_community``。
        crosstab: 可选的 ``CrosstabResult``；交叉表要现算（依赖簇分配），
            所以从摘要重建时由调用方算好后传进来。
        collection_text: 「采集口径」文本块（见 ``analyze/collection.py``）。
    """
    return build_analysis_report(
        run=run,
        cleaning=cleaning,
        sentiment=_Rendered(summaries["sentiment"]) if "sentiment" in summaries else None,
        discourse=(
            _Rendered(summaries["cluster_discourse"])
            if "cluster_discourse" in summaries else None
        ),
        community=(
            _Rendered(summaries["cluster_community"])
            if "cluster_community" in summaries else None
        ),
        crosstab=crosstab,
        # 显式传入的结果优先：现算的比从摘要里读回来的更新
        timeseries=(
            timeseries
            if timeseries is not None
            else (_Rendered(summaries["timeseries"]) if "timeseries" in summaries else None)
        ),
        replies=(
            replies
            if replies is not None
            else (_Rendered(summaries["replies"]) if "replies" in summaries else None)
        ),
        stance=(
            stance
            if stance is not None
            else (_Rendered(summaries["stance"]) if "stance" in summaries else None)
        ),
        stance_crosstab=stance_crosstab,
        discourse_stance=discourse_stance,
        event_results=event_results,
        changepoint_text=changepoint_text,
        term_text=term_text,
        collection_text=collection_text,
        charts=charts,
        output_dir=output_dir,
    )


def write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
