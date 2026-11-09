"""方法顾问（模块 E1）。

需求对助手的边界写得很清楚，这里逐条对应：

- **不替代用户做分析决策**，角色是"方法顾问 + 操作向导"；
- 回答方法论问题时，要结合**当前上下文**（采了多少条、什么档位、做没做分析），
  而不是给通用套话；
- 用户停在某一步不知所措时，给出 **2–3 个带按钮的下一步建议**。

所以这里有两个彼此独立的入口：
1. `ask()` —— 自然语言问答，走 LLM，注入上下文；
2. `suggest_next_steps()` —— **规则驱动**（不走 LLM）。

为什么建议用规则而不是 LLM：下一步该干什么由数据状态唯一决定，
用规则算出来的建议是确定性的、可复现的，而且不会因为模型抽风给出
"建议你换个话题"这种废话。LLM 负责解释，规则负责指路。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import WrcError
from .context import SessionContext

SOFTWARE_GUIDE = """## 本软件（博研助手）操作手册——回答"我该怎么做"时必须照着说

软件的界面从左到右、从上到下分五个区，用户按顺序操作：

1. **采集**（第 1 区）
   - 「话题标签」填要研究的话题（如 延迟退休），可不带 #；
   - 「候选帖子数」决定检索多少篇帖子（20–500，默认 100）；
   - 「检索方式」：话题聚合（只有主动打该话题的帖子）/ 关键词全文（含仅提及的帖子）；
   - 「每帖评论数」是**每篇帖子采多少条**（10–2000，默认 100）；
   - 「评论时间窗」可只采某段时间的评论；采满即停，采不满会报告覆盖度；
   - 「评论排序」热度优先（平台默认流）或时间优先（仅网页端后端生效）；
   - 「本次请求上限」是硬闸门：界面会实时估算"期望样本 vs 需要多少次请求"，
     两者矛盾时会提示。社科研究要样本就同时提高帖子数与请求上限；
   - 填好后点「检索帖子（第一步）」，在下面勾选要采的帖子。
2. **选择帖子**（第 2 区）：点「选」列的方框勾选；可用「设置勾选帖子的目标数…」
   逐帖指定采多少条（大帖多采、小帖少采，避免抽样偏差）。
3. **数据清洗**（第 3 区）：轻度/中度/重度；不确定就用中度（推荐）。
   重度需要 AI 模型。
4. **分析与图表**（第 4 区）：勾选情感分析、聚类、立场分析；再勾选要哪些图（14 种可选），
   或点「只留核心四张」。有「采集完成后自动分析」开关，默认开。
5. **导出与声明**（第 5 区）：勾选《数据使用声明》后才能导出；数据默认匿名化。

主按钮「一键完成：采集 → 清洗 → 分析 → 报告」会把 1–5 全部跑完，
结果在导出目录：raw/（原始）、analysis/（逐条结果 CSV）、report/（报告 + 图表 + Word）、
appendix.md（可复现性附录）、method_note.md（可直接抄进论文的方法段）、
research_log.md（过程日志）。

常见坑（要在回答里主动提醒）：
- 平台会以「暂无数据」拒绝部分帖子（真实运行里 20 篇有 9 篇被拒），
  想要足够样本就多勾几篇；
- 热门流的时间不单调，设了时间窗也不保证窗口内采全；要可辩护的覆盖用"时间优先"；
- 聚类轮廓系数低于 0.25 时簇边界不清晰，不宜当作分类结论；
- 情感标注是模型结果，正式写论文前应抽 20 条人工核对；
- 「无关」立场不计入支持率分母，分母口径必须在论文里写明。"""

SYSTEM_PROMPT = """你是社会科学研究的方法顾问，服务对象是社科类本科生与研究生。
你同时是这个软件（博研助手）的使用向导——**你要懂这个软件怎么操作**，
用户问"我想研究某个话题，在这个软件里该怎么走"时，必须给出具体到界面按钮的步骤。

## 你的角色
你**不是**替用户做分析决策的人。你要做的是：
1. 解释方法学概念（清洗档位、信度、聚类、情感极性与立场的区别等）；
2. 根据用户当前的数据状况，指出下一步该做什么、为什么；
3. 告诉用户在软件里**点哪里**能完成这件事；
4. 当用户的问题超出工具能力时，明确说出来，不要编造功能。

## 必须遵守的三条
- **不要给出研究结论**。你可以说"这类样本通常不足以支撑社群划分"，
  但不能说"所以小米的口碑是负面的"。结论由研究者下。
- **不要虚构数据和功能**。如果上下文里没有某项数据，就说"当前没有这项数据"。
- **涉及数字时必须基于上下文**。不要用"一般来说 100 条够了"这种套话，
  要看上下文里实际有多少条。"""+ SOFTWARE_GUIDE + """

## 回答风格
- 中文，简洁，先给结论再解释理由；
- 300 字以内，除非用户明确要求展开；
- 涉及操作时按「第几区 → 点哪个按钮」的顺序说，不要只说"去设置里调"；
- 如果适合继续操作，最后用一句话指出下一步。"""


@dataclass(slots=True)
class Suggestion:
    """一个带动作的下一步建议。"""

    title: str
    reason: str
    action: str
    """动作标识，界面据此决定按钮行为（如 ``run_clean`` / ``run_analysis``）。"""
    command: str = ""
    """对应的命令行写法，方便没有界面的场景。"""


def suggest_next_steps(context: SessionContext, *, limit: int = 3) -> list[Suggestion]:
    """按当前数据状态给出确定性建议（规则驱动，不走 LLM）。"""
    suggestions: list[Suggestion] = []

    if not context.run_id:
        return [
            Suggestion(
                title="先采集一个话题",
                reason="还没有任何数据。用你关心的话题标签开始第一次采集。",
                action="collect",
                command='python -m wrc.cli run --tag "你的话题" --select 1,2 --comments 50 --consent',
            )
        ]

    if context.clean_tier in ("", "未清洗", None):
        suggestions.append(
            Suggestion(
                title="先清洗数据",
                reason="还没跑清洗。未清洗的数据里含纯表情、纯链接与广告，会污染后续分析。",
                action="clean",
                command="在界面里点「开始采集并分析」，或命令行重跑 run --tier moderate",
            )
        )
        return suggestions[:limit]

    if context.n_kept < 100:
        suggestions.append(
            Suggestion(
                title="扩大样本量",
                reason=(
                    f"当前只有 {context.n_kept} 条评论。"
                    "样本量不足时，情感分布的置信区间很宽、聚类也容易切出伪结构"
                    "（本项目实测 46 条时轮廓系数只有 0.12）。"
                ),
                action="collect_more",
                command='python -m wrc.cli run --tag "<话题>" --select 1,2,3,4 --comments 100 --consent',
            )
        )

    if not context.has_sentiment:
        suggestions.append(
            Suggestion(
                title="跑情感分析",
                reason="还没有情感标注。这是舆论倾向研究最基础的指标。",
                action="run_sentiment",
                command="python tools/run_analysis.py --limit 200",
            )
        )

    if context.has_sentiment and context.agreement_rate is not None:
        if context.agreement_rate > 0.15:
            suggestions.append(
                Suggestion(
                    title="人工复核情感标注",
                    reason=(
                        f"双通道不一致率 {context.agreement_rate:.1%}，超过 15% 阈值。"
                        "建议抽 20 条人工核对，再决定是否采用自动标注。"
                    ),
                    action="review_sentiment",
                    command="从 analysis_comments.csv 抽 20 条核对 sentiment 列",
                )
            )

    if not context.has_discourse:
        suggestions.append(
            Suggestion(
                title="做话语聚类",
                reason="聚类能帮你识别这个话题下存在哪几套主要说法，而不只是看总量。",
                action="run_cluster",
                command="python tools/run_cluster.py",
            )
        )
    elif context.discourse_silhouette is not None and context.discourse_silhouette < 0.25:
        suggestions.append(
            Suggestion(
                title="暂缓解读聚类结果",
                reason=(
                    f"轮廓系数只有 {context.discourse_silhouette}，簇边界不清晰。"
                    "建议先扩大样本或换一个争议更集中的话题。"
                ),
                action="expand_sample",
                command="换话题或增加每帖评论数后重采",
            )
        )

    if context.has_sentiment and (context.has_discourse or context.has_community):
        suggestions.append(
            Suggestion(
                title="生成报告与方法卡片",
                reason="分析已就绪。生成报告可以拿到可直接引用进论文的方法段落。",
                action="build_report",
                command="python tools/run_report.py",
            )
        )

    if not suggestions:
        suggestions.append(
            Suggestion(
                title="结果已就绪，建议人工抽查",
                reason="采集、清洗、分析都已完成。下一步是人工确认标注质量，然后写论文。",
                action="review",
                command="见 docs/NEEDS-REVIEW.md",
            )
        )
    return suggestions[:limit]


def render_suggestions(suggestions: list[Suggestion]) -> str:
    lines = ["下一步建议："]
    for index, item in enumerate(suggestions, start=1):
        lines.append(f"  {index}. {item.title}")
        lines.append(f"     {item.reason}")
    return "\n".join(lines)


def ask(
    question: str,
    context: SessionContext,
    *,
    client: Any,
) -> str:
    """就方法/操作问题咨询助手。

    Args:
        question: 用户的自然语言问题。
        context: 当前会话状态（会注入提示词）。
        client: ``wrc.llm.LlmClient``。
    """
    question = (question or "").strip()
    if not question:
        raise WrcError(
            kind=_kind(),
            what_happened="问题为空。",
            what_you_can_do=("输入一个具体的方法或操作问题，例如：我的样本量够做聚类吗？",),
        )

    user_prompt = (
        "## 当前研究状态\n"
        f"{context.to_prompt()}\n\n"
        "## 用户的问题\n"
        f"{question}"
    )
    return client.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=1200,
    )


def _kind() -> Any:
    from ..errors import ErrorKind

    return ErrorKind.CONFIG
