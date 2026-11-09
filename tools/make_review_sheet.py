"""生成情感标注人工抽查表（把"抽 20 条核对"变成"打开表格逐行打分"）。

为什么要有这个工具：
审核清单写"抽 20 条人工核对"，但研究者的第一反应是"从哪抽？抽完记哪儿？
怎么算错了几条？"——每个疑问都是一个放弃的理由。所以这里直接把表做出来：
20 条随机样本、模型判定、本地对照、以及**留空待填的三列**。

两种抽样模式，用途完全不同，不要混用：

- ``stratified``（默认）：按三类情感的实际比例分层抽样，每类至少 3 条。
  这份表算出来的错误率**可以**用来估计整体标注质量。
- ``disagreement``：优先抽两个通道判断不一致的样本。
  这份表是"找 bug 用的"，**不能**用来算错误率——
  因为它故意过采样了最可能错的那批，算出来必然偏高。

随机种子固定，所以同一会话重复生成得到同一批样本（可复现）。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20240101
LABELS_ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}
ORDER = ("positive", "neutral", "negative")

#: ★ 开头的列是留给人工填的，空着交付。
COLUMNS = [
    "序号",
    "cid",
    "评论原文",
    "LLM判定",
    "强度",
    "本地模型判定",
    "本地分数",
    "两通道一致",
    "★你的判定",
    "★判得对吗",
    "★错在哪里",
    "备注",
]


def _rows(store, run_id: str) -> list[dict]:
    from wrc.analyze.service import _merge_analysis

    kept = store.load_analysis_rows(run_id)
    merged = _merge_analysis(kept, store.load_comment_analysis(run_id))
    return [row for row in merged if row.get("sentiment")]


def _sample(rows: list[dict], *, n: int, mode: str, rng: random.Random) -> list[dict]:
    if mode == "disagreement":
        disagree = [
            row for row in rows
            if row.get("local_sentiment")
            and row["local_sentiment"] != row.get("sentiment")
        ]
        rng.shuffle(disagree)
        picked = disagree[:n]
        if len(picked) < n:
            rest = [row for row in rows if row not in picked]
            rng.shuffle(rest)
            picked += rest[: n - len(picked)]
        return _sort_by_sentiment(picked)

    # 分层：按比例分配，但每类至少 3 条（否则可能一条正面都抽不到，
    # 就没法判断模型是不是"一律判负面"）
    by_label: dict[str, list[dict]] = {label: [] for label in ORDER}
    for row in rows:
        by_label.setdefault(str(row["sentiment"]), []).append(row)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    for label in ORDER:
        bucket = by_label.get(label) or []
        picked += bucket[: min(3, len(bucket))]
    remaining = n - len(picked)
    if remaining > 0:
        pool = [row for row in rows if row not in picked]
        rng.shuffle(pool)
        picked += pool[:remaining]
    return _sort_by_sentiment(picked[:n])


def _sort_by_sentiment(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (
        ORDER.index(str(row.get("sentiment"))) if str(row.get("sentiment")) in ORDER else 9
    ))


def _payload(rows: list[dict]) -> list[dict]:
    payload = []
    for index, row in enumerate(rows, start=1):
        llm = row.get("sentiment") or ""
        local = row.get("local_sentiment") or ""
        payload.append({
            "序号": index,
            # cid 是 16–19 位数字，Excel 会把它转成 5.34E+15 并永久丢失精度；
            # 前导单引号让 Excel 按文本处理。研究者若用别的软件，去掉引号即可。
            "cid": f"'{row.get('cid', '')}",
            "评论原文": row.get("cleaned_text") or row.get("text_raw") or "",
            "LLM判定": LABELS_ZH.get(llm, llm),
            "强度": row.get("intensity") if row.get("intensity") is not None else "",
            "本地模型判定": LABELS_ZH.get(local, local) if local else "（未抽样）",
            "本地分数": row.get("local_score") if row.get("local_score") is not None else "",
            "两通道一致": ("一致" if llm == local else "不一致") if local else "—",
            "★你的判定": "",
            "★判得对吗": "",
            "★错在哪里": "",
            "备注": "",
        })
    return payload


def _write(path: Path, payload: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：Excel 直接双击打开不乱码。这是社科学生最常用的方式。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    parser.add_argument("--n", type=int, default=20, help="抽多少条（默认 20）")
    parser.add_argument(
        "--mode", choices=("stratified", "disagreement"), default="stratified",
        help="stratified=算错误率用；disagreement=找提示词问题用",
    )
    parser.add_argument("--list-runs", action="store_true", help="列出可选会话")
    args = parser.parse_args()

    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init_schema()

    if args.list_runs:
        with store.connect() as conn:
            for row in conn.execute(
                "SELECT run_id, tag, created_at FROM collection_runs "
                "ORDER BY created_at DESC"
            ):
                print(f"{row['run_id']}  {row['tag']}  {row['created_at']}")
        return 0

    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2

    rows = _rows(store, run_id)
    if not rows:
        print(f"会话 {run_id} 还没有情感标注结果。先跑：python tools/run_analysis.py --run {run_id}")
        return 2

    rng = random.Random(SEED)
    picked = _sample(rows, n=args.n, mode=args.mode, rng=rng)
    payload = _payload(picked)

    out_dir = settings.export_dir / run_id
    suffix = "" if args.mode == "stratified" else "-找bug用"
    path = _write(out_dir / f"抽查表-情感标注{suffix}.csv", payload)

    print(f"抽查表：{path}")
    print(f"会话：{run_id}；样本量 {len(picked)}/{len(rows)} 条；模式：{args.mode}")
    print()
    print("怎么用：")
    print("  1. 双击打开这个 CSV（Excel 会正确显示中文）")
    print("  2. 只看前三列读完原文，然后在「★你的判定」里填 正面/中性/负面/无法判断")
    print("  3. 「★判得对吗」填 对 或 错；填「错」的顺便写一句错在哪里")
    print("  4. 数一下错了多少条：错 ≤2 条 → 通过；错 ≥3 条 → 提示词需要改")
    if args.mode == "disagreement":
        print()
        print("⚠ 这是「找 bug」用的表：故意多抽了两通道不一致的样本，")
        print("   算出来的错误率必然偏高，不要拿它当整体标注质量。")
    print()
    print("本地模型那一列只是参照——它本身也不可靠（置信度低的居多），")
    print("不要因为「跟本地模型不一致」就判 LLM 错。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
