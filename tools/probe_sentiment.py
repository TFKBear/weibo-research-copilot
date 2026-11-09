"""情感模型实测：用真实评论检查中文效果，再决定要不要接进产品。

为什么这一步不能省：
多语种通用情感模型在中文微博口语上经常失效——反讽、网络用语、表情符号
（如"质量真好"配合"今年烧了多少辆车"的语境）会让它给出与人类相反的判断。
如果不先测就接进情感分析，结果会写成论文里的"情感分布"。

本脚本只读导出数据，不修改任何东西。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "distilbert-sentiment-zh"


def load_comments(limit: int) -> list[str]:
    exports = ROOT / "data" / "exports"
    texts: list[str] = []
    for path in sorted(exports.glob("*/raw/raw_comments.csv"))[-3:]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    text = (row.get("text_raw") or "").strip()
                    if 4 <= len(text) <= 100:
                        texts.append(text)
        except OSError:
            continue
    seen: set[str] = set()
    return [t for t in texts if not (t in seen or seen.add(t))][:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--text", default=None, help="直接测一句话")
    args = parser.parse_args()

    onnx_files = list(MODEL_DIR.rglob("*.onnx")) if MODEL_DIR.exists() else []
    tokenizers = list(MODEL_DIR.rglob("tokenizer.json")) if MODEL_DIR.exists() else []
    if not onnx_files or not tokenizers:
        print(f"模型未就绪：{MODEL_DIR}")
        print("期望至少有 model_quantized.onnx + tokenizer.json + config.json")
        return 2

    from wrc.analyze.classifier import OnnxSequenceClassifier

    clf = OnnxSequenceClassifier(onnx_files[0], tokenizers[0])
    print(f"模型：{onnx_files[0].name}")
    print(f"类别：{clf.labels}")

    if args.text:
        texts = [args.text]
    else:
        texts = load_comments(args.sample)
        if not texts:
            print("没有可用的真实评论，改为内置样例。")
            texts = [
                "支持小米，希望国产车越做越好",
                "今年烧了多少辆车了？质量真好",
                "有底盘吗？不会是还坐在电池上吧",
                "这款车外观确实好看，内饰用料一般",
                "加V信领取优惠，数量有限先到先得",
                "我不是针对谁，我是说在座的都一般",
                "笑死，这也能吹",
                "客观说，性价比还可以",
            ]

    print(f"\n共 {len(texts)} 条，开始推理 …")
    results = clf.predict(texts)
    print()
    print(f"{'标签':<10}{'置信':>6}   分布(neg/neu/pos)          文本")
    for text, result in zip(texts, results):
        neg = result.prob("negative")
        neu = result.prob("neutral")
        pos = result.prob("positive")
        print(
            f"{result.label:<10}{result.score:>6.3f}   "
            f"{neg:.2f}/{neu:.2f}/{pos:.2f}          {text[:30]}"
        )

    from collections import Counter

    counts = Counter(r.label for r in results)
    print(f"\n分布：{dict(counts)}")

    low = sum(1 for r in results if r.score < 0.6)
    print(f"置信度 <0.6 的比例：{low}/{len(results)}")
    if low / max(1, len(results)) > 0.4:
        print("⚠ 低置信占比偏高，说明该模型对这类中文文本不确定。")
        print("  这不一定是 bug——但它意味着「第二通道」只能作为对照，不能作为结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
