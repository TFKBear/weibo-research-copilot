"""情感分析（模块 D1）：双通道 + 信度抽样。

设计要点，每一条都对应需求文档里的具体要求：

1. **不采用单一模型黑箱输出**。
   通道一：LLM 零样本/少样本，输出 `sentiment` + `intensity`；
   通道二：本地中文情感模型，作为交叉参照。
2. **提示词里明确区分「情感极性」与「立场」**——
   负面情绪不等于反对立场。这是社科情感标注最常被混淆的一点。
3. **信度抽样**：随机抽 min(200, 总数×5%) 条跑双通道，
   不一致率超过 15% 时显著提示并展示不一致样本。
4. **方法卡片**：每次分析都产出可粘贴进论文的模型/提示词/超参数记录。

⚠ 实测结论（必须写进方法卡片，不能藏）：
本地通道用的多语种模型在中文微博口语上置信度偏低
（25 条真实评论中 18 条 <0.6，且把「@人民日报 @央视新闻」判为 positive）。
因此它**只作粗筛参照**：不一致率本身是"该不该相信自动标注"的信号，
而不是"哪个通道更对"的判决。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

PROMPT_VERSION = "sentiment-1.2.0"
"""提示词版本。

1.0.0 → 1.1.0 的唯一原因：研究者人工复核发现两类系统性偏误——
**反问句被当成立场表达**（"你们的钱还不够花吗？"判 negative）、
**颂扬/站队式表达被判中性**（"谢谢雷总""这是阳谋"判 neutral）。
两类错例已作为 few-shot 示例写进提示词，见 ``FEW_SHOT_USER`` 第 6–9 条。

⚠ 换了提示词就必须重做信度抽样：旧的人工标注结论对新提示词不成立。
"""
LABELS: tuple[str, ...] = ("positive", "neutral", "negative")
LABELS_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}

#: 需求文档规定的不一致率告警阈值。
DISAGREEMENT_THRESHOLD = 0.15
#: 抽样比例与上限。
SAMPLE_RATIO = 0.05
SAMPLE_CAP = 200
#: 抽样下限。
#:
#: 需求文档写的是 min(200, 总数×5%)。但小样本下 5% 会退化成无意义的个位数：
#: 25 条评论按 5% 只抽到 1.25 条 → 2 条，算出的不一致率是 50% 或 0%，
#: 纯属噪声，却会被当成"信度不达标"的告警。任何抽到 2 条的比率都不该进论文。
#: 所以加一个 30 条的下限：小样本就全量比对，把偏离需求的做法写进方法卡片。
SAMPLE_FLOOR = 30
#: 每次 LLM 调用标注多少条。太小则调用次数多，太大则单次输出易截断。
BATCH_SIZE = 20

SYSTEM_PROMPT = """你是社会科学研究中的文本编码助手。你的任务是标注微博评论的**情感极性**。

## 三个类别
- positive：表达积极情绪（满意、赞赏、期待、鼓励）
- negative：表达消极情绪（不满、担忧、愤怒、失望、批评）
- neutral：陈述事实、提出问题、转述信息，或没有明显情绪倾向

## 必须区分的两件事
**情感极性 ≠ 立场**。请只标注情感极性，不要标注"支持/反对某政策"。
- "我很担心这个政策会伤害一线工人" → negative（担忧情绪），即使说话人可能支持政策目标
- "这个方案我仔细读过，条款设计确实有考虑弹性" → neutral（客观陈述）
- "太好了，终于有人管这事了" → positive
反讽按实际语义标注：事故语境下的"质量真好" → negative。

## 难点一：反问句怎么判（这条最容易错）
**反问句不等于负面，也不等于正话反说。**判定顺序：
1. 先问"这句话是在**索取信息**，还是在**表达态度**"。
2. 单纯提问、求解释、把问题抛回给对方 → **neutral**，intensity 记 0.2–0.3。
3. 只有带明确反讽标记时（"[doge]"、上下文语境明显反着说、夸张到不可能成立），
   才按字面反义判 negative。
- "你们的钱还不够花吗？" → neutral（在提问，不是讽刺）
- "今年烧了多少辆车了？质量真好" → negative（"质量真好"是明显反语）
- "这是在质疑防撞梁设计好不好看？" → neutral（把问题抛回去，没有表态）

## 难点二：尊称、颂扬、站队怎么判（这类句面没有情绪词，但确实是正面）
判据不是"有没有情绪形容词"，而是**说话人有没有在表达认可或维护**。
- 使用尊称并给予正面评价（"谢谢雷总""相信雷总"）→ positive
- 为某人/某公司/某产品**辩护、夸耀、站队**（"这是阳谋，就看你敢不敢跟"）→ positive
- 单纯转述、复述事实，不带评价 → neutral
- 讽刺性的"夸"仍然按反语处理（见上一条）

## 难点三：没有情绪词的抱怨怎么判（第二大错源）
"陈述一个让人不满的处境"就是负面，**不需要感叹号、脏话或情绪形容词**。
判据是"这句话在指出谁的损失 / 缺失 / 不公"，而不是"有没有情绪词"。
- "主要社保钱花没了" → negative（0.5，指出自己的钱被花掉）
- "很多老年人没退休金的" → negative（0.5，指出制度覆盖不到的人）
- "缴够15年是不是就可以停掉社保了" → negative（0.4，隐含"不想再缴"）
- "我今年30岁" → neutral（纯事实，不涉及损失或不公）
- "现在不是58了吗" → neutral（在确认信息）

## 难点四：自嘲、辛酸叙述不一定负面
说话人可能一边诉苦、一边表示理解或认同。看他**对政策/事件的最终态度**：
- "我是80后的尾巴，我不是卖惨……凡事总不能紧着我，还是感谢国家" → positive（0.5）
  （诉苦只是铺垫，落点是理解与感谢）
- "作为90年的牛马，退休年龄延迟后已经将自我预期调整到65周岁退休[允悲]" → negative（0.55）
  （落点是被迫接受，没有认同）
区分办法：找句子最后落点有没有"理解/感谢/认同/支持"这类表态；没有就按诉苦本身判负面。

## 表情与网络用语
"😂""[笑cry]""[doge]""呵呵""笑死"在微博语境中常表示嘲讽或无奈 → 通常 negative。
"[赞]""[good]""[打call]" 表示认可 → positive。
只有表情、没有文字内容的（如"[doge]"）→ neutral，intensity 记 0.2。

## 强度
intensity 取 0 到 1：0.1=极弱（几乎中性），0.5=中等，0.9=强烈（如连续的愤怒表达）。

## 输出格式
只输出 JSON，不要解释：
{"items":[{"i":1,"s":"negative","intensity":0.7}, {"i":2,"s":"neutral","intensity":0.2}]}
其中 i 是输入条目的编号，s 只能是 positive / neutral / negative。"""

#: few-shot 示例。第 6–9 条来自**第一轮**人工复核的不一致样本；
#: 第 10–14 条来自**第三轮**（真实会话 20261026T195346-6d31，20 条复核里错 4 条）：
#: 错因高度集中——"没有情绪词的抱怨"被判中性（3 例）、"诉苦但认同"被判负面（1 例）。
#: 把真实错例放进示例，比继续加抽象规则更有效。
FEW_SHOT_USER = """1. 支持渐进式改革，但配套的就业保障必须跟上
2. 今年烧了多少辆车了？质量真好
3. 我算了一下，按这个方案我这代要到63岁
4. 谢谢雷总保我平安，对方全责
5. 这个政策讨论了好几年了
6. 你们的钱还不够花吗？
7. 这是在质疑防撞梁设计好不好看？
8. 说车企做的是阳谋，看友商敢不敢跟
9. 谢谢雷总，相信小米的质量
10. 主要社保钱花没了
11. 很多老年人没退休金的
12. 缴够15年是不是就可以停掉社保了
13. 我是80后的尾巴，不是卖惨，但凡事总不能紧着我，还是感谢国家
14. 作为90年的牛马，退休年龄延迟后，已经将自我预期调整到65周岁退休[允悲]"""

FEW_SHOT_ASSISTANT = """{"items":[
{"i":1,"s":"neutral","intensity":0.4},
{"i":2,"s":"negative","intensity":0.85},
{"i":3,"s":"neutral","intensity":0.3},
{"i":4,"s":"positive","intensity":0.8},
{"i":5,"s":"neutral","intensity":0.2},
{"i":6,"s":"neutral","intensity":0.3},
{"i":7,"s":"neutral","intensity":0.3},
{"i":8,"s":"positive","intensity":0.6},
{"i":9,"s":"positive","intensity":0.7},
{"i":10,"s":"negative","intensity":0.5},
{"i":11,"s":"negative","intensity":0.5},
{"i":12,"s":"negative","intensity":0.4},
{"i":13,"s":"positive","intensity":0.5},
{"i":14,"s":"negative","intensity":0.55}]}"""


@dataclass(slots=True)
class SentimentItem:
    """一条评论的情感标注结果。"""

    cid: str
    post_mid: str
    text: str
    created_at: str | None = None
    llm_label: str | None = None
    llm_intensity: float | None = None
    local_label: str | None = None
    local_score: float | None = None
    local_probs: dict[str, float] = field(default_factory=dict)
    in_sample: bool = False

    @property
    def label(self) -> str | None:
        """主通道判定（LLM 优先，缺失时退回本地通道）。"""
        return self.llm_label or self.local_label

    @property
    def agrees(self) -> bool | None:
        if self.llm_label is None or self.local_label is None:
            return None
        return self.llm_label == self.local_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "post_mid": self.post_mid,
            "sentiment": self.llm_label,
            "intensity": self.llm_intensity,
            "local_sentiment": self.local_label,
            "local_score": self.local_score,
            "in_sample": self.in_sample,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class AgreementReport:
    """两通道一致性（编码信度）报告。"""

    sample_size: int = 0
    compared: int = 0
    disagree: int = 0
    low_confidence: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.disagree / self.compared if self.compared else 0.0

    @property
    def flagged(self) -> bool:
        return self.compared > 0 and self.rate > DISAGREEMENT_THRESHOLD

    @property
    def low_confidence_rate(self) -> float:
        return self.low_confidence / self.compared if self.compared else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "compared": self.compared,
            "disagree": self.disagree,
            "disagreement_rate": round(self.rate, 4),
            "threshold": DISAGREEMENT_THRESHOLD,
            "flagged": self.flagged,
            "low_confidence": self.low_confidence,
            "low_confidence_rate": round(self.low_confidence_rate, 4),
        }


@dataclass(slots=True)
class SentimentResult:
    items: list[SentimentItem] = field(default_factory=list)
    agreement: AgreementReport = field(default_factory=AgreementReport)
    distribution: dict[str, int] = field(default_factory=dict)
    per_post: dict[str, dict[str, int]] = field(default_factory=dict)
    daily: dict[str, dict[str, int]] = field(default_factory=dict)
    method: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.items)

    def render(self) -> str:
        lines: list[str] = []
        lines.append("情感分析结果")
        lines.append("=" * 62)
        lines.append(f"样本量：{self.total} 条")
        total = max(1, sum(self.distribution.values()))
        lines.append("")
        lines.append("整体情感分布：")
        for label in LABELS:
            count = self.distribution.get(label, 0)
            lines.append(
                f"  {LABELS_ZH[label]:<4}{count:>6} 条   {count / total:>6.1%}"
            )
        lines.append("")
        if self.per_post:
            lines.append("分帖子情感分布：")
            for post_mid, counts in list(self.per_post.items())[:10]:
                n = max(1, sum(counts.values()))
                parts = "  ".join(
                    f"{LABELS_ZH[label]} {counts.get(label, 0)}"
                    f"({counts.get(label, 0) / n:.0%})"
                    for label in LABELS
                )
                lines.append(f"  {post_mid}  n={n}  {parts}")
            lines.append("")
        if self.daily:
            lines.append("按日情感趋势（正面占比）：")
            for day, counts in sorted(self.daily.items()):
                n = max(1, sum(counts.values()))
                pos = counts.get("positive", 0) / n
                bar = "█" * int(pos * 20)
                lines.append(f"  {day}  n={n:<4}{pos:>5.0%} {bar}")
            lines.append("")
        agreement = self.agreement
        lines.append("编码信度（两通道一致性抽样）：")
        lines.append(
            f"  抽样 {agreement.sample_size} 条，可比对 {agreement.compared} 条，"
            f"不一致 {agreement.disagree} 条 → 不一致率 {agreement.rate:.1%}"
            f"（阈值 {DISAGREEMENT_THRESHOLD:.0%}）"
        )
        if agreement.flagged:
            lines.append(
                "  ⚠ 模型间分歧较大，建议人工复核下面这些不一致样本，"
                "不要在论文中直接采用自动标注结果。"
            )
        else:
            lines.append("  ✓ 两通道一致率在可接受范围。")
        lines.append(
            f"  本地通道低置信占比 {agreement.low_confidence_rate:.0%}"
            "（该通道在中文微博口语上置信度偏低，仅作粗筛参照）"
        )
        if agreement.examples:
            lines.append("  不一致样本（最多 10 条）：")
            for item in agreement.examples[:10]:
                lines.append(
                    f"    · LLM={item['llm']:<8} 本地={item['local']:<8} "
                    f"{item['text'][:40]}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "distribution": self.distribution,
            "per_post": self.per_post,
            "daily": self.daily,
            "agreement": self.agreement.to_dict(),
            "method": self.method,
        }


# --------------------------------------------------------------------------- #
def _extract_items(rows: Sequence[dict[str, Any]]) -> list[SentimentItem]:
    items: list[SentimentItem] = []
    for row in rows:
        text = str(row.get("cleaned_text") or row.get("text_raw") or "").strip()
        if not text:
            continue
        created = row.get("created_at")
        items.append(
            SentimentItem(
                cid=str(row.get("cid")),
                post_mid=str(row.get("post_mid") or ""),
                text=text,
                created_at=str(created)[:10] if created else None,
            )
        )
    return items


def _build_user_prompt(texts: Sequence[str]) -> str:
    body = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
    return f"请标注以下 {len(texts)} 条评论：\n{body}"


#: 本地模型的原始标签 → 三分类。不同模型的类别体系不一样：
#: - 多语言 5 类模型：Very Negative … Very Positive
#: - cardiffnlp 系列：LABEL_0/1/2（负/中/正）
#: - 我们最早的 3 类模型：negative/neutral/positive
#: 统一在这里折叠，**不改动模型原始输出**，以便附录里仍能记录真实类别体系。
LOCAL_LABEL_MAP: dict[str, str] = {
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "负面": "negative",
    "中性": "neutral",
    "正面": "positive",
    "消极": "negative",
    "积极": "positive",
}


def normalise_local(prediction: Any) -> tuple[str, float, dict[str, float]]:
    """把本地模型的原始判定折叠成三分类，返回 (标签, 置信度, 三分类概率)。

    为什么不直接取 argmax 的原始标签：5 类模型的 "Very Negative" 在我们
    的三分类框架里不是一个类别，直接写进 `local_sentiment` 列会让
    "两通道是否一致"永远判为不一致——那是把一个标签体系问题错当成模型分歧。
    """
    probs3 = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    mapped = False
    for raw, value in (getattr(prediction, "probs", None) or {}).items():
        key = LOCAL_LABEL_MAP.get(str(raw).strip().lower())
        if key is None:
            continue
        probs3[key] += float(value)
        mapped = True
    if not mapped:
        # 认不出的标签体系：原样返回，宁可让下游看到陌生标签，也不硬猜
        return (
            str(getattr(prediction, "label", "")),
            float(getattr(prediction, "score", 0.0)),
            dict(getattr(prediction, "probs", None) or {}),
        )
    label = max(probs3, key=lambda key: probs3[key])
    return label, round(probs3[label], 4), {k: round(v, 4) for k, v in probs3.items()}


def _parse_labels(payload: Any, expected: int) -> dict[int, tuple[str, float]]:
    """解析 LLM 返回，返回 {序号: (标签, 强度)}。序号从 1 开始。"""
    out: dict[int, tuple[str, float]] = {}
    raw_items: Any = payload
    if isinstance(payload, dict):
        raw_items = payload.get("items", payload.get("results", payload.get("data", [])))
    if not isinstance(raw_items, list):
        return out
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("i", entry.get("index", 0)))
        except (TypeError, ValueError):
            continue
        if index < 1 or index > expected:
            continue
        label = str(entry.get("s", entry.get("sentiment", ""))).strip().lower()
        if label not in LABELS:
            continue
        try:
            intensity = float(entry.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        out[index] = (label, max(0.0, min(1.0, intensity)))
    return out


def analyse(
    rows: Sequence[dict[str, Any]],
    *,
    llm_client: Any,
    local_classifier: Any | None = None,
    sample_size: int | None = None,
    seed: int = 20240101,
    emit: Callable[[str, str], None] | None = None,
) -> SentimentResult:
    """执行情感分析。

    Args:
        rows: 清洗后的评论行（需含 cid / text / created_at）。
        llm_client: `wrc.llm.LlmClient`，主通道。
        local_classifier: `OnnxSequenceClassifier`，第二通道；为 None 时不做信度抽样。
        sample_size: 抽样条数；缺省按 min(200, 总数×5%)。
        seed: 抽样随机种子（可复现）。
        emit: 进度回调。
    """
    notify = emit or (lambda _level, _message: None)
    items = _extract_items(rows)
    if not items:
        return SentimentResult(method={"error": "没有可分析的文本"})

    # ---------- 通道一：LLM ----------
    notify("info", f"通道一（LLM）：标注 {len(items)} 条 …")
    done = 0
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        texts = [item.text for item in batch]
        try:
            payload = llm_client.complete_json(
                system=SYSTEM_PROMPT,
                user=FEW_SHOT_USER + "\n\n" + FEW_SHOT_ASSISTANT
                + "\n\n" + _build_user_prompt(texts),
                max_tokens=2500,
            )
            parsed = _parse_labels(payload, len(batch))
        except Exception as exc:  # noqa: BLE001
            notify("warn", f"  第 {start // BATCH_SIZE + 1} 批标注失败：{exc}")
            parsed = {}
        for index, item in enumerate(batch, start=1):
            hit = parsed.get(index)
            if hit:
                item.llm_label, item.llm_intensity = hit
        done += len(batch)
        notify("info", f"  进度 {done}/{len(items)}")

    missing = sum(1 for item in items if item.llm_label is None)
    if missing:
        notify("warn", f"  有 {missing} 条未能从 LLM 获得标签（将回退到本地通道）")

    # ---------- 通道二：本地模型（仅抽样） ----------
    agreement = AgreementReport()
    if local_classifier is not None:
        planned = round(len(items) * SAMPLE_RATIO)
        n = sample_size if sample_size is not None else min(
            SAMPLE_CAP, max(SAMPLE_FLOOR, planned)
        )
        n = min(n, len(items))
        rng = random.Random(seed)
        sample = rng.sample(items, n)
        notify("info", f"通道二（本地模型）：抽样 {n} 条做交叉参照 …")
        predictions = local_classifier.predict([item.text for item in sample])
        for item, prediction in zip(sample, predictions):
            item.in_sample = True
            label, score, probs = normalise_local(prediction)
            item.local_label = label
            item.local_score = score
            item.local_probs = probs

        agreement.sample_size = n
        for item in sample:
            if item.agrees is None:
                continue
            agreement.compared += 1
            if not item.agrees:
                agreement.disagree += 1
                if len(agreement.examples) < 10:
                    agreement.examples.append(
                        {
                            "cid": item.cid,
                            "llm": item.llm_label,
                            "local": item.local_label,
                            "text": item.text,
                        }
                    )
            if item.local_score is not None and item.local_score < 0.6:
                agreement.low_confidence += 1
        notify(
            "ok" if not agreement.flagged else "warn",
            f"  不一致率 {agreement.rate:.1%}（{agreement.disagree}/{agreement.compared}）",
        )

    # ---------- 聚合 ----------
    distribution: Counter[str] = Counter()
    per_post: dict[str, Counter[str]] = defaultdict(Counter)
    daily: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        label = item.label
        if label is None:
            continue
        distribution[label] += 1
        if item.post_mid:
            per_post[item.post_mid][label] += 1
        if item.created_at:
            daily[item.created_at][label] += 1

    method = {
        "module": "sentiment",
        "prompt_version": PROMPT_VERSION,
        "llm_model": getattr(getattr(llm_client, "config", None), "model", "unknown"),
        "llm_base_url": getattr(getattr(llm_client, "config", None), "base_url", ""),
        "llm_stats": dict(getattr(llm_client, "stats", {}) or {}),
        "local_model": getattr(local_classifier, "model_name", None),
        "local_labels": list(getattr(local_classifier, "labels", []) or []),
        "sample_ratio": SAMPLE_RATIO,
        "sample_cap": SAMPLE_CAP,
        "sample_floor": SAMPLE_FLOOR,
        "sample_seed": seed,
        "batch_size": BATCH_SIZE,
        "disagreement_threshold": DISAGREEMENT_THRESHOLD,
        "labelled": sum(1 for item in items if item.label is not None),
        "unlabelled": sum(1 for item in items if item.label is None),
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if local_classifier is not None:
        method["local_note"] = (
            "本地通道为多语种通用模型，实测在中文微博口语上置信度偏低"
            "（低置信占比见 agreement.low_confidence_rate），仅作粗筛参照。"
        )

    return SentimentResult(
        items=items,
        agreement=agreement,
        distribution=dict(distribution),
        per_post={k: dict(v) for k, v in per_post.items()},
        daily={k: dict(v) for k, v in sorted(daily.items())},
        method=method,
    )
