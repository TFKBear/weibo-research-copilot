"""对比人工复核结论与新提示词的标注结果（只读，一次性核查用）。"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "20261026T195346-6d31"
SHEET = ROOT / "data" / "exports" / RUN / "抽查表-情感标注-找bug用.csv"
ZH = {"positive": "正面", "neutral": "中性", "negative": "负面"}


def latest_sentiment(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT analysis_id FROM analysis_runs "
        "WHERE run_id=? AND module='sentiment' ORDER BY created_at DESC LIMIT 1",
        (RUN,),
    ).fetchone()
    return row[0]


def main() -> None:
    con = sqlite3.connect(ROOT / "data" / "wrc.sqlite3")
    con.row_factory = sqlite3.Row
    analysis_id = latest_sentiment(con)
    # 用评论原文对齐：表格被 Excel 存过之后 cid 变成 5.34E+15 这种科学计数，
    # 精度已丢失，拿 cid 匹配会全部对不上（这本身是 make_review_sheet 的问题，
    # 已改为写带前导单引号的文本形式）。
    labels_by_text = {
        str(row["text"]).strip(): row["sentiment"]
        for row in con.execute(
            "SELECT c.cid, c.sentiment, r.text_raw AS text "
            "FROM analysis_comments c JOIN raw_comments r "
            "ON r.cid = c.cid AND r.run_id = ? "
            "WHERE c.analysis_id=?",
            (RUN, analysis_id),
        )
    }
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8-sig")))
    wrong = []
    for row in rows:
        text = (row["评论原文"] or "").strip()
        model = ZH.get(str(labels_by_text.get(text)), "(缺)")
        human = (row["★你的判定"] or "").strip()
        if model != human:
            wrong.append((row["序号"], text[:34], model, human))
    print(f"提示词版本对应 analysis_id：{analysis_id}")
    print(f"人工标注 {len(rows)} 条；模型错 {len(wrong)} 条 → 错误率 {len(wrong)/len(rows):.0%}")
    for index, text, model, human in wrong:
        print(f"  #{index} 模型={model} 人工={human} ｜ {text}")


if __name__ == "__main__":
    main()
