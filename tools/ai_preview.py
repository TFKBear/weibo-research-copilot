"""AI 清洗预览：拿你已有的真实采集结果，看看重度档会删掉什么。

只读——不会修改数据库，也不会重新导出。用途是"先看看 AI 会怎么判，
再决定要不要真的用重度档跑一次"。

用法：
    python tools/ai_preview.py                # 用最近一次采集
    python tools/ai_preview.py --run 20261025T010257
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrc.ai.analysis import analyse  # noqa: E402
from wrc.ai.status import detect_status  # noqa: E402
from wrc.config import get_settings  # noqa: E402
from wrc.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="采集会话 ID；默认取最近一次")
    args = parser.parse_args()

    settings = get_settings()
    status = detect_status(settings.ai_embedding_model_dir)
    if not status.ready:
        print("AI 未就绪：", status.summary())
        return 2

    store = Store(settings.db_path)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT run_id FROM collection_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        print("还没有任何采集记录。请先在界面里跑一次采集。")
        return 2
    run_id = args.run or row["run_id"]

    rows = store.load_comments(run_id)
    if not rows:
        print(f"会话 {run_id} 没有评论数据。")
        return 2

    print(f"会话：{run_id}    评论：{len(rows)} 条")
    print(f"模型：{status.onnx_path.name}\n")

    from wrc.ai.embedder import build_embedder

    embedder = build_embedder(status.onnx_path, status.tokenizer_path)

    def progress(done: int, total: int) -> None:
        if done == total or done % 256 == 0:
            print(f"  编码中 {done}/{total} …", end="\r")

    analysis = analyse(
        rows,
        embedder,
        marketing_threshold=settings.ai_marketing_threshold,
        coordinated_threshold=settings.ai_coordinated_threshold,
        min_cluster_size=settings.ai_min_cluster_size,
        on_progress=progress,
    )
    print(" " * 40, end="\r")

    print(f"阈值：营销 {analysis.marketing_threshold} / 协同 {analysis.coordinated_threshold}"
          f"（最小簇 {analysis.min_cluster_size}）")
    print(f"用时：{analysis.elapsed_s}s\n")

    print(f"【营销话术】命中 {len(analysis.marketing)} 条")
    by_cid = {str(r.get("cid")): r for r in rows}
    for cid, hit in sorted(
        analysis.marketing.items(), key=lambda kv: -kv[1].score
    )[:10]:
        text = str(by_cid.get(cid, {}).get("text_raw") or "")[:34]
        print(f"  {hit.score:.3f}  {text}")

    print(f"\n【协同行为】命中 {len(analysis.coordinated)} 条，"
          f"共 {analysis.cluster_count} 个簇")
    for cid, size in sorted(analysis.coordinated.items(), key=lambda kv: -kv[1])[:8]:
        text = str(by_cid.get(cid, {}).get("text_raw") or "")[:34]
        print(f"  簇内 {size} 条  {text}")

    if not analysis.marketing and not analysis.coordinated:
        print("\n没有命中。这在真实数据上是正常结果（大多数人不是水军），")
        print("但也可能是阈值偏严——用 python tools/calibrate_ai.py 复核。")
    else:
        print("\n提醒：相似度高不等于水军。亲友团、粉丝控评、同一个梗刷屏都会成簇，")
        print("      工具只标出「可能有协同」，结论必须你自己下。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
