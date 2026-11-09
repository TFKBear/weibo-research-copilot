"""方法卡片（模块 D3）。

需求：每份分析输出顶部附一张方法卡片，内容至少包括模型名称/版本、
提示词摘要、关键超参数、抽样率、清洗档位、运行时间戳，
并且用户可以**直接粘贴进论文的方法部分草稿**。

所以这里不只输出参数表，还生成一段可以直接抄进论文的中文方法描述——
那是研究者真正需要的东西。参数表用于自查，成段文字用于投稿。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__

MODULE_NAMES = {
    "sentiment": "情感分析",
    "cluster_discourse": "话语聚类",
    "cluster_community": "用户社群聚类",
    "timeseries": "时间演变",
    "replies": "回复关系",
    "compare": "跨话题对比",
    "stance": "立场",
    "event": "事件前后对比",
}


def build_method_card(
    *,
    module: str,
    method: dict[str, Any],
    cleaning: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> str:
    """生成方法卡片（Markdown）。"""
    title = MODULE_NAMES.get(module, module)
    run = run or {}
    lines: list[str] = []
    lines.append(f"## 方法卡片 · {title}")
    lines.append("")
    lines.append("### 一、数据")
    lines.append("")
    lines.append(f"- 采集会话：`{run.get('run_id', '—')}`")
    if run.get("tag"):
        lines.append(f"- 话题：{run.get('tag')}")
    mode = run.get("search_mode")
    if mode:
        from .collection import search_mode_text

        lines.append(f"- 检索口径：{search_mode_text(mode)}")
    window = run.get("comment_window") or {}
    if window.get("active") or window.get("label"):
        from .collection import window_text

        lines.append(f"- 评论时间窗：{window_text(window)}")
    if run.get("comment_sort"):
        sort_zh = "热度优先（平台热门流）" if run["comment_sort"] == "hot" else "按时间倒序"
        if run.get("sort_applied") is False and run["comment_sort"] == "time":
            sort_zh += "（⚠ 后端忽略排序参数，实际未生效）"
        lines.append(f"- 评论排序：{sort_zh}")
    if run.get("covered_from") and run.get("covered_to"):
        lines.append(f"- 实际覆盖：{run['covered_from']} ～ {run['covered_to']}")
    if run.get("created_at"):
        lines.append(f"- 采集时间：{run.get('created_at')}（北京时间）")
    if cleaning:
        kept = cleaning.get("total_kept")
        total = cleaning.get("total_input")
        rate = cleaning.get("retention_rate")
        lines.append(f"- 清洗档位：{cleaning.get('tier', '—')}")
        lines.append(f"- 清洗管线版本：`{cleaning.get('pipeline_version', '—')}`")
        if total and kept is not None:
            rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "—"
            lines.append(f"- 清洗后样本：{total} 条中保留 {kept} 条（{rate_text}）")
    lines.append("")

    lines.append("### 二、方法参数")
    lines.append("")
    for key, value in method.items():
        if value is None or value == "" or isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}：{value}")
    lines.append("")
    for key, value in method.items():
        if isinstance(value, dict) and value:
            lines.append(f"- {key}：")
            for sub_key, sub_value in value.items():
                lines.append(f"  - {sub_key}：{sub_value}")
    lines.append("")

    if limitations:
        lines.append("### 三、已知局限（建议在论文中披露）")
        lines.append("")
        for item in limitations:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("### 四、可直接引用的一段话")
    lines.append("")
    lines.append(_paper_paragraph(module, method, cleaning, run))
    lines.append("")
    lines.append("---")
    lines.append(
        f"本卡片由博研助手 v{__version__} 于 "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 自动生成。"
    )
    lines.append(
        "**请核对后使用**：参数为实际运行值，但研究设计、抽样理由与伦理审批"
        "需由研究者自行补写。"
    )
    return "\n".join(lines)


def _paper_paragraph(
    module: str,
    method: dict[str, Any],
    cleaning: dict[str, Any] | None,
    run: dict[str, Any],
) -> str:
    if module == "sentiment":
        model = method.get("llm_model", "未知模型")
        prompt_version = method.get("prompt_version", "—")
        ratio = method.get("sample_ratio") or 0.05
        cap = method.get("sample_cap") or 200
        floor = method.get("sample_floor") or 30
        threshold = method.get("disagreement_threshold") or 0.15
        local = method.get("local_model") or "未启用本地对照模型"
        tier = (cleaning or {}).get("tier") or "未记录"
        pipeline_version = (cleaning or {}).get("pipeline_version") or "未记录"
        tag = run.get("tag") or "研究话题"
        total = (cleaning or {}).get("total_input")
        kept = (cleaning or {}).get("total_kept")
        rate = (cleaning or {}).get("retention_rate")
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "未记录"
        corpus = (
            f"{total} 条，经清洗后保留 {kept} 条（保留率 {rate_text}）"
            if total and kept is not None else "（样本量未记录）"
        )
        # 文风说明：按实证取向的社媒情感分析论文的通行写法组织——
        # 语料（来源/规模/清洗）→ 标注过程（工具与参数）→ 信度 → 效度 → 局限，
        # 五段各自独立成句，不把参数塞进一个长句。
        return (
            f"本研究以微博为数据来源，围绕{tag}话题采集用户评论共 {corpus}，"
            f"以单条评论为基本分析单元。清洗采用「{tier}」档位（管线版本 "
            f"`{pipeline_version}`），清洗决策逐条可追溯。\n\n"
            f"情感标注由两条独立通道完成。主通道调用大语言模型（`{model}`，"
            f"提示词版本 `{prompt_version}`），以零样本/少样本方式将每条评论编码为"
            f"正面、中性、负面三类，并输出 0—1 的情感强度。提示词明确区分情感极性"
            f"与立场表达，并约定反问句、反语、表情符号与网络用语的判定规则。\n\n"
            f"为估计标注信度，另以本地中文情感分类模型（`{local}`）对 "
            f"min({cap}, max({floor}, 总量×{ratio:.0%})) 条随机子样本并行编码，"
            f"两通道不一致率超过 {threshold:.0%} 时提示人工复核；"
            f"不一致率本身作为「该批数据是否值得采信」的信号报告，"
            f"而非判定哪一通道正确。\n\n"
            f"标注效度以研究者的人工编码为参照：从清洗后样本中随机抽取 20 条，"
            f"由研究者独立编码后与模型标注比对，一致率见下文「人工复核」栏。"
            f"正式发表建议将复核样本扩大至总量的 10%—20%，并报告编码员间信度系数"
            f"（如 Krippendorff's α）；本研究为单人编码，故仅报告一致率。\n\n"
            f"需要说明的是，自动标注不能替代人工编码。模型在反问句与褒扬式站队"
            f"表达上存在系统性偏误，提示词已针对这两类模式补充判定规则与示例，"
            f"但泛化能力仍受限于微博口语的语境依赖。"
        )

    if module == "cluster_discourse":
        k = method.get("n_clusters", "—")
        model = method.get("embedding_model", "句向量模型")
        return (
            f"本研究对评论文本进行话语聚类。首先使用 {model} 将评论文本编码为"
            f"句向量，随后采用 K-Means 算法聚类（K={k}）。聚类数由轮廓系数与"
            f"肘部法则共同建议后人工确认，而非预先假定。各簇通过"
            f"类内词频与全体词频的比值提取代表性关键词，并给出距质心最近的"
            f"代表性评论供人工判读。"
        )

    if module == "cluster_community":
        k = method.get("n_clusters", "—")
        features = method.get("features", "行为与文本特征")
        return (
            f"本研究在用户粒度构造行为特征（{features}），采用 K-Means 聚类"
            f"（K={k}）划分用户社群，并以主成分分析降至二维作可视化。"
            f"各聚类的角色标签依据其质心在各项特征上的相对水平生成，"
            f"属于描述性命名而非因果判断。"
        )

    if module == "stance":
        topic = method.get("topic") or run.get("tag") or "本次话题"
        prompt_version = method.get("prompt_version", "—")
        model = method.get("llm_model") or "大语言模型"
        return (
            f"本研究对评论文本进行**立场编码**，编码对象为「{topic}」。"
            f"与情感极性不同，立场指受访者对具体政策议题的支持或反对倾向，"
            f"二者是独立维度：对政策持支持态度的用户也可能表达负面情绪"
            f"（如担忧执行方式伤及特定群体）。\n\n"
            f"编码由大语言模型（`{model}`，提示词版本 `{prompt_version}`）完成，"
            f"采用零样本/少样本方式，将每条评论编码为支持、反对、中立、无关四类，"
            f"并给出 0—1 的置信度。提示词中给出判断顺序（先判断是否在谈该议题，"
            f"再判断希望其发生与否），并约定反讽按实际立场编码。\n\n"
            f"「无关」指评论未涉及该议题（签到、闲聊、纯表情等），"
            f"不计入支持率与反对率；报告支持率时须写明分母口径"
            f"（有效表态数，或 支持+反对）。\n\n"
            f"**信度口径**：立场维度采用「单通道（大语言模型）+ 人工复核」，"
            f"不设双通道交叉验证。这一选择是在情感通道已做双通道比对的前提下做出的："
            f"立场没有可用的本地对照模型，强行拼凑第二通道只会制造虚假的信度证据。"
            f"因此立场结论的可信度依赖于提示词质量与人工复核覆盖率，"
            f"不宜与情感标签的信度等量齐观。"
        )

    if module == "event":
        event = (method or {}).get("event") or {}
        date_text = event.get("date") or "—"
        label = event.get("label") or "事件"
        pre = event.get("pre_days", "—")
        post = event.get("post_days", "—")
        return (
            f"本研究采用事件前后对比（event study）考察「{label}」（{date_text}）"
            f"前后舆论分布的变化。以事件日为界，取事件前 {pre} 天"
            f"（不含事件当日）与事件后 {post} 天（含事件当日）两个窗口，"
            f"分别计算立场与情感各类别的占比，并报告前后差值。\n\n"
            f"统计上采用**两比例 z 检验**（正态近似）比较同类别在两个窗口的占比，"
            f"仅在四个期望频数均不小于 5 时报告 p 值；不满足该假设时明确标注"
            f"「正态近似不可靠」。任一侧样本量不足 30 条时只报告频数、不作推断。\n\n"
            f"需要说明的是，这是**描述性对比，不是因果推断**。窗口长度的选择、"
            f"事件日期的准确性，以及同期发生的其它事件，都可能影响结果；"
            f"事件日期由研究者依据外部信息确定，工具不代为选择。"
        )

    if module == "timeseries":
        unit = method.get("granularity", "按日")
        min_n = method.get("min_bucket_n", 5)
        span = method.get("time_span_days", "—")
        return (
            f"本研究将评论文本{unit}聚合，计算各时段的情感极性占比与话语簇占比，"
            f"并以评论文量作为可比性的参照。样本量小于 {min_n} 条的时段不解读占比，"
            f"趋势判断以时间前后两段的占比差为据，仅作描述性比较，未做统计检验。"
            f"数据时间跨度为 {span} 天。需要说明的是，评论取自平台热门评论流，"
            f"平台按热度而非时间返回结果，故本曲线描述已采集样本的时间分布，"
            f"不能等同于话题舆论的完整演化轨迹。"
        )

    if module == "replies":
        depth = method.get("max_depth", "—")
        return (
            f"本研究以评论间的父子引用关系（parent_cid）重建回复结构，"
            f"统计各帖回复树的规模、深度（最大 {depth} 层）与集中度，"
            f"用以区分「多点分散讨论」与「单点引爆式争论」两种互动形态。"
            f"该分析只使用结构关系，不使用任何用户身份信息；"
            f"节点数小于最小可展示样本的子树整体不予展示。"
        )

    if module == "compare":
        runs = method.get("compared_runs", "—")
        return (
            f"本研究对比了 {runs} 个话题的采集结果，比较其评论量、清洗保留率、"
            f"情感分布与高频关键词。由于各话题的采集参数（选帖数、每帖评论上限）"
            f"并不相同，绝对量的差异不能直接解释为话题热度差异，"
            f"故对比以**占比与结构**为主，并在方法卡片中列明各次采集的参数。"
        )

    return "（该模块暂无自动生成的方法段落）"


def write_method_card(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
