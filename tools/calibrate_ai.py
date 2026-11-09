"""用真实数据标定 AI 阈值。

为什么必须做这一步：
阈值 0.82 是我在没有模型时凭经验写的。模型真的跑起来后才发现，
bge 系列在中文短句上的余弦相似度分布是"压缩"的——两句都在推销的完全不同文案，
相似度也只有 0.6 左右。用 0.82 当门槛，结果是**一条都抓不到**：
功能看起来"启用了"，实际什么也没做。

这比功能不可用更危险：清洗报告会显示"AI 营销识别：0 条"，
研究者会以为"这批数据没有营销内容"，而实际上只是门槛高到失效。

本脚本的做法：
- 正样本：12 条真实口吻的营销/导流文案（与内置模板措辞不同）；
- 负样本：从你导出的真实评论里抽样；
- 分别计算"与最近模板的相似度"，找出能把两者分开的位置。

用法：
    python tools/calibrate_ai.py
    python tools/calibrate_ai.py --sample 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrc.ai.seeds import MARKETING_SEEDS  # noqa: E402
from wrc.ai.status import detect_status  # noqa: E402

#: 正样本：口吻自然、不套模板的营销/导流文案。
#: 刻意与 seeds.py 里的表述不同，避免"自己考自己"。
POSITIVES: list[str] = [
    "有想法的可以加我聊一下，随时在线",
    "需要的直接私我联系方式，价格好说",
    "这款我用过，真心推荐，链接放在评论区了",
    "姐妹们冲，我已经回购两次了，不踩雷",
    "内部渠道价，比官网便宜不少，私信详聊",
    "招几个兼职帮忙做数据，时间自由，日结",
    "关注我不迷路，每天分享好物",
    "物流很快，客服态度也好，值得买",
    "想入手的可以问我，我有渠道能拿到优惠",
    "老板人特别好，第二次买了，一如既往的好",
    "现在下单还有赠品，活动就到这周末",
    "需要同款的姐妹扣1，我发链接给你",
]

#: 负样本兜底（当找不到真实导出数据时使用）。
FALLBACK_NEGATIVES: list[str] = [
    "这款车续航表现一般，高速掉电有点快",
    "支持小米，希望国产车越做越好",
    "我算了一下，按渐进式方案我大概要到63岁",
    "政策设计需要考虑不同群体的差异",
    "不同意，延迟退休会挤压年轻人就业",
    "看了官方原文，其实是小步调整、弹性实施",
    "这个价格还是有点贵，再等等看",
    "外观确实好看，但内饰用料一般",
    "有底盘吗？不会是还坐在电池上吧",
    "今年烧了多少辆车了？质量真好",
]


def load_real_comments(limit: int) -> list[str]:
    exports = ROOT / "data" / "exports"
    if not exports.exists():
        return []
    files = sorted(exports.glob("*/raw/raw_comments.csv"))
    texts: list[str] = []
    for path in files[-3:]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    text = (row.get("text_raw") or "").strip()
                    if 4 <= len(text) <= 120:
                        texts.append(text)
        except OSError:
            continue
    # 简单去重，避免同一条被重复计入
    seen: set[str] = set()
    unique = [t for t in texts if not (t in seen or seen.add(t))]
    return unique[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=80)
    args = parser.parse_args()

    status = detect_status(ROOT / "models")
    if not status.ready:
        print("AI 未就绪：", status.summary())
        for step in status.remediation():
            print("  -", step)
        return 2

    from wrc.ai.embedder import build_embedder

    embedder = build_embedder(status.onnx_path, status.tokenizer_path)
    seeds = list(MARKETING_SEEDS)
    seed_vectors = embedder.encode(seeds).vectors

    negatives = load_real_comments(args.sample)
    source = "真实导出数据"
    if not negatives:
        negatives = FALLBACK_NEGATIVES
        source = "内置样例（未找到导出数据）"

    positives = POSITIVES
    pos_scores: list[tuple[float, str]] = []
    neg_scores: list[tuple[float, str]] = []

    vectors = embedder.encode(positives + negatives).vectors
    import numpy as np

    scores = np.asarray(vectors) @ np.asarray(seed_vectors).T
    for index, text in enumerate(positives):
        pos_scores.append((float(scores[index].max()), text))
    for offset, text in enumerate(negatives):
        neg_scores.append((float(scores[len(positives) + offset].max()), text))

    pos_scores.sort()
    neg_scores.sort(reverse=True)

    print(f"正样本 {len(positives)} 条（营销文案）")
    print(f"负样本 {len(negatives)} 条（来源：{source}）")
    print()
    print("正样本 与最近模板 的相似度（升序）：")
    for score, text in pos_scores:
        print(f"  {score:.3f}  {text[:26]}")
    print()
    print("负样本 与最近模板 的相似度（降序前 15）：")
    for score, text in neg_scores[:15]:
        print(f"  {score:.3f}  {text[:26]}")
    print()

    import statistics

    pos_min = pos_scores[0][0]
    neg_max = neg_scores[0][0]
    pos_median = statistics.median(s for s, _ in pos_scores)
    neg_median = statistics.median(s for s, _ in neg_scores)

    print(f"正样本：最低 {pos_min:.3f}  中位数 {pos_median:.3f}")
    print(f"负样本：最高 {neg_max:.3f}  中位数 {neg_median:.3f}")

    # 选一个兼顾召回与误杀的门槛：优先落在两者之间且靠近正样本低分端
    if pos_min > neg_max:
        suggested = round((pos_min + neg_max) / 2, 2)
        verdict = "完全可分"
    elif pos_median > neg_median:
        suggested = round((pos_median + neg_max) / 2, 2)
        suggested = min(max(suggested, neg_max), pos_min) if pos_min > neg_max else round(pos_median - 0.02, 2)
        verdict = "部分重叠，取折中"
    else:
        suggested = round(pos_median - 0.02, 2)
        verdict = "重叠严重，需人工复核"

    print(f"\n判定：{verdict}")
    print(f"建议阈值：{suggested}")
    print(
        "  说明：阈值取在负样本最高分之上、正样本最低分之下；\n"
        "        若两者重叠，则优先保证不误杀（宁可少抓，也不删真实声音）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
