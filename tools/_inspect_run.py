"""一次性核查脚本：检查最近一次真实运行的日志与产物（只读）。"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "20261026T195346-6d31"
EXPORT = ROOT / "data" / "exports" / RUN
LOG = ROOT / "data" / "logs" / "research_log.jsonl"


def load_events() -> list[dict]:
    if not LOG.exists():
        return []
    events = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("run_id") == RUN:
            events.append(entry)
    return events


def main() -> None:
    events = load_events()
    print(f"=== 研究日志：{len(events)} 条事件 ===")
    print("事件类型：", dict(Counter(e["event"] for e in events).most_common()))

    done = [e for e in events if e["event"] == "comments_done"]
    print(f"\n=== 评论采集：成功 {len(done)} 篇 ===")
    total = 0
    for e in done:
        w = e.get("window") or {}
        total += int(e.get("collected") or 0)
        print(
            f"  {e.get('post_mid')}: 采到 {e.get('collected')} 条 | 停止={e.get('stopped_reason')}"
            f" | 排序生效={e.get('sort_applied')} | 窗口内={w.get('in_window')}"
            f" 窗口外={w.get('out_of_window')} 未知时间={w.get('unknown_time')}"
            f" 覆盖完整={w.get('complete')}"
        )
    print(f"  合计入库：{total} 条")

    skip = [e for e in events if e["event"] == "post_skipped"]
    print(f"\n=== 跳过的帖子：{len(skip)} 篇 ===")
    for e in skip:
        print(f"  {e.get('post_mid')} → {e.get('what_happened')}")

    plans = [e for e in events if e["event"] == "collection_plan"]
    if plans:
        print("\n=== 采集方案（第一篇）===")
        first = plans[0]
        print(f"  目标={first.get('target')} 排序={first.get('sort')}"
              f" 排序生效={first.get('sort_applied')} 可提前停={first.get('can_stop_early')}"
              f" 页数上限={first.get('page_cap')}")
        print(f"  时间窗={first.get('window')}")

    print("\n=== 数据库 ===")
    con = sqlite3.connect(ROOT / "data" / "wrc.sqlite3")
    con.row_factory = sqlite3.Row
    run = con.execute("SELECT * FROM collection_runs WHERE run_id=?", (RUN,)).fetchone()
    params = json.loads(run["params_json"] or "{}")
    print(f"  状态={run['status']} 备注={run['note']} 话题={run['tag']}")
    print(f"  时间窗={params.get('comment_window')}")
    print(f"  检索口径={params.get('search_mode')} 评论排序={params.get('comment_sort')}"
          f" 后端={params.get('comment_backend')} 上限={params.get('max_requests')}")
    for label, sql in (
        ("raw_comments", "SELECT COUNT(*) FROM raw_comments WHERE run_id=?"),
        ("clean 保留", "SELECT COUNT(*) FROM clean_comments WHERE run_id=? AND kept=1"),
        ("clean 全部", "SELECT COUNT(*) FROM clean_comments WHERE run_id=?"),
    ):
        print(f"  {label}: {con.execute(sql, (RUN,)).fetchone()[0]}")
    modules = [r["module"] for r in con.execute(
        "SELECT module FROM analysis_runs WHERE run_id=?", (RUN,))]
    print(f"  分析模块：{modules or '（一个也没跑）'}")

    print("\n=== 导出目录 ===")
    for path in sorted(EXPORT.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(EXPORT)}  {path.stat().st_size:,} 字节")
    if not (EXPORT / "report").exists():
        print("  ⚠ 没有 report/ 目录 —— 分析（情感/立场/聚类）与报告都没生成")

    print("\n=== 附录「二、采集」===")
    appendix = EXPORT / "appendix.md"
    if appendix.exists():
        text = appendix.read_text(encoding="utf-8")
        start = text.find("## 二、采集")
        print(text[start:start + 800] if start >= 0 else "  找不到该章节")

    print("\n=== 方法说明「一、采集」===")
    note = EXPORT / "method_note.md"
    if note.exists():
        text = note.read_text(encoding="utf-8")
        start = text.find("## 一、采集")
        end = text.find("## 二、清洗")
        print(text[start:end] if start >= 0 else "  找不到该章节")


if __name__ == "__main__":
    main()
