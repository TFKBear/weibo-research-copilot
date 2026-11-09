"""立场分析（研究者选定的第二维度：B 方案）。

为什么单独做一个维度，而不是把情感"改"成立场：

研究问题是「网友支持还是反对某项政策」，而情感极性回答不了它。
两条真实评论就能说明差别：

- 「我很担心这个政策会伤害一线工人」→ 情感 negative，立场可能**支持**政策目标
- 「多存钱，存资产，黄金」→ 情感 neutral，立场其实**不信任**政策

人是这样，模型也是。上一轮人工复核里，研究者有两条判错理由直接写的是
"属于是反对态度""实质上站队公司"——**他要的从来就是立场，不是情绪**。

设计上的三个决定：

1. **立场必须锚定到具体政策**。同一条评论对不同议题的立场可能相反，
   所以提示词里带上本次采集的话题（`topic`），并要求"无关"单独成类。
2. **四分类：支持 / 反对 / 中立 / 无关**。"无关"是必须的：
   微博热评区大量是"签到""路过""抢前排"，把这类硬塞进"中立"会污染分布。
3. **不改情感维度的任何东西**。两个维度并列存储、并列报告，
   研究者可以交叉看（"情绪负面但立场支持"正是最有价值的格子）。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

PROMPT_VERSION = "stance-1.0.0"
STANCE_LABELS: tuple[str, ...] = ("support", "oppose", "neutral", "irrelevant")
STANCE_ZH = {
    "support": "支持",
    "oppose": "反对",
    "neutral": "中立",
    "irrelevant": "无关",
}
DEFAULT_BATCH_SIZE = 10
"""每批标注多少条。

一开始用 20（与情感模块一致），实测 25 批里挂了 9 批——
大模型偶尔会在 JSON 前后加解释或 markdown 围栏，一次毁掉 20 条。
降到 10 并配合"失败就二分重试"，可以把损失控制在个位数。
"""

#: 单批失败时的最小拆分单位。到 1 条还解析不出来才算真失败。
MIN_SPLIT_SIZE = 1

SYSTEM_PROMPT = """你是社会科学研究中的文本编码助手。你的任务是判断微博评论对**某一项具体政策**的**立场**。

## 四个类别
- support：支持、赞同、为该政策辩护、希望它落地
- oppose：反对、质疑其合理性、要求修改或撤销
- neutral：对政策有明确表态的意愿，但态度居中（如"看具体执行""有待观察"）
- irrelevant：**没有在谈这项政策**（签到、抢前排、纯表情、聊别的话题）

## 判断顺序（按这个顺序想，不要跳步）
1. 这条评论**是否在谈这项政策**？不是 → `irrelevant`。
2. 是的话，说话人**希望这项政策发生还是不希望**？
   希望 / 为其辩护 → `support`；不希望 / 质疑其合理 → `oppose`。
3. 明确在谈、但既不支持也不反对 → `neutral`。

## 关键：立场不等于情绪
- 「我很担心这个政策伤害一线工人」→ **support**（担心执行，但认同政策目标本身）
- 「政策初衷是好的，可执行得太急」→ **support**（如果他认为该继续推进）
- 「年轻人交了一辈子都拿不回来」→ **oppose**
- 「这个政策讨论了好几年了」→ **neutral**（陈述事实，没有表态）
- 「打卡」「路过」「[doge]」→ **irrelevant**

## 反讽与网络用语
反讽按**实际立场**判：
- 「真是为人民着想啊[doge]」→ oppose
- 「这样的好政策多来点」→ 若无反讽语境，判 support

## 置信度
confidence 取 0—1：0.9=立场鲜明直说，0.5=需要推断，0.2=几乎看不出态度。

## 输出格式
**只输出 JSON，不要任何解释、不要 markdown 代码块、不要前后缀。**
{"items":[{"i":1,"stance":"oppose","confidence":0.8},{"i":2,"stance":"irrelevant","confidence":0.9}]}
i 是输入条目编号，stance 只能是 support / oppose / neutral / irrelevant。"""


@dataclass(slots=True)
class StanceItem:
    cid: str
    post_mid: str
    text: str
    stance: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "stance": self.stance,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class StanceResult:
    topic: str = ""
    distribution: dict[str, int] = field(default_factory=dict)
    per_post: dict[str, dict[str, int]] = field(default_factory=dict)
    n_total: int = 0
    n_labelled: int = 0
    n_failed: int = 0
    items: list[StanceItem] = field(default_factory=list)
    method: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def labelled_share(self) -> float:
        return (self.n_labelled / self.n_total) if self.n_total else 0.0

    def render(self) -> str:
        lines = [
            f"立场分布（对象：{self.topic or '本次话题'}，n={self.n_labelled}）",
            "=" * 62,
        ]
        if not self.n_labelled:
            lines.append("没有得到任何立场标签，请检查模型是否可用。")
            return "\n".join(lines)
        total = self.n_labelled
        for label in STANCE_LABELS:
            count = self.distribution.get(label, 0)
            lines.append(
                f"{STANCE_ZH[label]:<6}{count:>6} 条  {count / total:>6.1%}"
            )
        lines.append("")
        if self.n_failed:
            lines.append(
                f"⚠ 有 {self.n_failed} 条没能得到标签（模型返回格式异常），已从分母中剔除。"
            )
        lines.append(
            "说明：立场是**针对本次话题/政策**的表述，与情感极性是两个独立维度。"
            "「无关」表示该评论没有在谈这项政策（签到、闲聊、纯表情等），"
            "它不计入支持率也不计入反对率。"
        )
        lines.append("")
        lines.append(
            "计算支持率时请只用「支持 ÷（支持 + 反对）」或「支持 ÷ 有效表态数」，"
            "两种口径都要在论文里写明；把「无关」放进分母会系统性稀释比例。"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": "stance",
            "topic": self.topic,
            "distribution": self.distribution,
            "per_post": self.per_post,
            "n_total": self.n_total,
            "n_labelled": self.n_labelled,
            "n_failed": self.n_failed,
            "method": self.method,
        }


# --------------------------------------------------------------------------- #
def analyse_stance(
    rows: Sequence[dict[str, Any]],
    *,
    topic: str,
    llm_client: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    emit: Callable[[str, str], None] | None = None,
) -> StanceResult:
    """对每条评论判定其对本话题的立场。"""
    notify = emit or (lambda _level, _message: None)
    items = [
        StanceItem(
            cid=str(row.get("cid")),
            post_mid=str(row.get("post_mid") or ""),
            text=str(row.get("cleaned_text") or row.get("text_raw") or "").strip(),
        )
        for row in rows
        if str(row.get("cleaned_text") or row.get("text_raw") or "").strip()
    ]
    if not items:
        return StanceResult(topic=topic, method={"error": "没有可分析的文本"})

    notify("info", f"立场分析（对象：{topic or '本次话题'}）：标注 {len(items)} 条 …")
    failed = 0
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        hits, batch_failed = _annotate(
            batch, topic=topic, llm_client=llm_client, notify=notify,
            depth=start // batch_size + 1,
        )
        failed += batch_failed
        for offset, (label, confidence) in hits.items():
            batch[offset].stance = label
            batch[offset].confidence = confidence
        notify("info", f"  进度 {min(start + batch_size, len(items))}/{len(items)}")

    distribution: Counter[str] = Counter()
    per_post: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        if item.stance is None:
            continue
        distribution[item.stance] += 1
        per_post[item.post_mid][item.stance] += 1

    labelled = sum(distribution.values())
    notify(
        "ok" if labelled else "warn",
        f"  立场标签 {labelled} 条，失败 {failed} 条",
    )
    return StanceResult(
        topic=topic,
        distribution=dict(distribution),
        per_post={key: dict(value) for key, value in per_post.items()},
        n_total=len(items),
        n_labelled=labelled,
        n_failed=failed,
        items=items,
        method={
            "module": "stance",
            "topic": topic,
            "prompt_version": PROMPT_VERSION,
            "labels": list(STANCE_LABELS),
            "batch_size": batch_size,
            "llm_model": getattr(getattr(llm_client, "config", None), "model", None),
            # 注意：LlmClient.stats 是 property（返回 dict），不是方法。
            # 早期写成 stats() 会 TypeError，且假客户端用方法实现时测试发现不了——
            # 所以测试里的假客户端也改成了 property。
            "llm_stats": dict(getattr(llm_client, "stats", {}) or {}),
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "definition": "立场相对于本次话题/政策；与情感极性独立存储",
            "note": (
                "「无关」不计入支持率与反对率；"
                "论文中请写明分母口径（有效表态数 或 支持+反对）"
            ),
            "reliability": "单通道（LLM）+ 人工复核；不设双通道交叉验证",
        },
    )


def _annotate(
    batch: Sequence[StanceItem],
    *,
    topic: str,
    llm_client: Any,
    notify: Callable[[str, str], None],
    depth: int = 1,
) -> tuple[dict[int, tuple[str, float]], int]:
    """标注一批；解析失败就二分重试，返回 ({批内偏移: (标签, 置信度)}, 失败数)。

    为什么要二分而不是"整批丢弃"：实测 25 批里挂了 9 批，
    整批丢弃等于丢掉 240 条数据。二分后通常只有真正回不出 JSON 的那一两条
    会失败，其余都能救回来——数据是花钱调接口换来的，能不丢就不丢。
    """
    try:
        payload = llm_client.complete_json(
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(topic, [item.text for item in batch]),
        )
        parsed = _parse_items(payload, expected=len(batch))
    except Exception as exc:  # noqa: BLE001
        if len(batch) > MIN_SPLIT_SIZE:
            notify(
                "warn",
                f"  第 {depth} 批（{len(batch)} 条）解析失败，二分重试："
                f"{type(exc).__name__}",
            )
            middle = len(batch) // 2
            left, left_failed = _annotate(
                batch[:middle], topic=topic, llm_client=llm_client,
                notify=notify, depth=depth,
            )
            right, right_failed = _annotate(
                batch[middle:], topic=topic, llm_client=llm_client,
                notify=notify, depth=depth,
            )
            merged = dict(left)
            merged.update({index + middle: value for index, value in right.items()})
            return merged, left_failed + right_failed
        notify("warn", f"  单条解析失败，跳过：{type(exc).__name__}: {exc}")
        return {}, len(batch)

    hits: dict[int, tuple[str, float]] = {}
    for offset in range(len(batch)):
        hit = parsed.get(offset + 1)
        if hit is not None:
            hits[offset] = hit
    missing = len(batch) - len(hits)
    if missing and len(batch) > MIN_SPLIT_SIZE:
        # 部分缺失：只对缺的那几条单独再试一次
        for offset in range(len(batch)):
            if offset in hits:
                continue
            sub, _sub_failed = _annotate(
                [batch[offset]], topic=topic, llm_client=llm_client,
                notify=notify, depth=depth,
            )
            if sub:
                hits[offset] = sub[0]
                missing -= 1
    return hits, missing


def _build_user_prompt(topic: str, texts: Sequence[str]) -> str:
    topic_text = topic or "本次讨论的话题"
    body = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
    return (
        f"本次要判断立场的对象是：{topic_text}\n"
        f"请判断以下 {len(texts)} 条评论对「{topic_text}」的立场：\n{body}"
    )


def _parse_items(payload: Any, *, expected: int) -> dict[int, tuple[str, float]]:
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
        label = str(entry.get("stance", entry.get("label", ""))).strip().lower()
        # 容忍模型输出中文标签
        for key, zh in STANCE_ZH.items():
            if label == zh:
                label = key
        if label not in STANCE_LABELS:
            continue
        try:
            confidence = float(entry.get("confidence", entry.get("conf", 0.5)))
        except (TypeError, ValueError):
            confidence = 0.5
        out[index] = (label, max(0.0, min(1.0, confidence)))
    return out


def stance_of(row: dict[str, Any]) -> str | None:
    """从分析行里取立场标签（供交叉表、时间演变复用）。"""
    value = row.get("stance")
    return str(value) if value else None
