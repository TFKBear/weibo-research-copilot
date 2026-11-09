"""在真实采集结果上跑一次情感分析，验证双通道与信度抽样是否工作。

只读数据库，不改任何数据；输出报告 + 方法卡片到控制台。

用法：
    python tools/run_sentiment.py                 # 用最近一次采集
    python tools/run_sentiment.py --run 20261025T010257 --limit 40
    python tools/run_sentiment.py --no-local      # 只跑 LLM 通道
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SENTIMENT_DIR = ROOT / "models" / "distilbert-sentiment-zh"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--limit", type=int, default=30, help="最多分析多少条（控制成本）")
    parser.add_argument("--no-local", action="store_true", help="跳过本地对照通道")
    args = parser.parse_args()

    from wrc.analyze.method_card import build_method_card
    from wrc.analyze.service import run_sentiment
    from wrc.config import get_settings
    from wrc.llm.client import LlmClient
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()  # 幂等；老库要靠它补出 analysis_* 表
    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2

    rows = store.load_analysis_rows(run_id)
    if not rows:
        print(f"会话 {run_id} 没有清洗后的数据。")
        return 2
    rows = rows[: args.limit]
    print(f"会话 {run_id}：取 {len(rows)} 条清洗后评论做情感分析\n")

    client = LlmClient()
    ok, reason = client.available()
    print(f"LLM：{client.config.model} @ {client.config.base_url}（{reason}）")
    if not ok:
        print("LLM 不可用，无法做主通道。先在 .env 里配置 WRC_LLM_API_KEY_ENV。")
        return 2

    classifier = None
    if not args.no_local:
        onnx_files = sorted(SENTIMENT_DIR.rglob("*.onnx")) if SENTIMENT_DIR.exists() else []
        tokenizers = sorted(SENTIMENT_DIR.rglob("tokenizer.json")) if SENTIMENT_DIR.exists() else []
        if onnx_files and tokenizers:
            from wrc.analyze.classifier import OnnxSequenceClassifier

            classifier = OnnxSequenceClassifier(onnx_files[0], tokenizers[0])
            print(f"本地对照：{onnx_files[0].name}")
        else:
            print("本地对照模型缺失，跳过信度抽样")

    result, analysis_id = run_sentiment(
        run_id,
        store=store,
        settings=settings,
        llm_client=client,
        local_classifier=classifier,
        limit=args.limit,
        emit=lambda level, message: print(f"  [{level}] {message}"),
    )
    print()
    print(f"analysis_id = {analysis_id}")
    print(result.render())
    print()
    print(build_method_card(
        module="sentiment",
        method=result.method,
        limitations=[
            "情感标注由模型完成，未做人工双编码校验；信度抽样只反映通道间一致性，不等于编码员间信度。",
            "本地对照通道为多语种通用模型，在中文微博口语上置信度偏低，仅作粗筛参照。",
            "采集样本来自平台热门评论流，不代表该帖全部评论的分布。",
        ],
    ))

    import json

    out = ROOT / "data" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sentiment_{run_id}.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存：{out / f'sentiment_{run_id}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
