"""分层存储（原始层 / 清洗层）。

设计要点（对应产品宪法第 1 条与模块 F1）：

- **原始层（raw_*）保留全部字段**，包括 user_id。理由：用户级聚类必须能
  稳定识别「同一个人」，如果入库就抹掉身份，P1 的用户社群分析无从做起。
  匿名化的正确位置在**导出**，不在入库 —— 入库即脱敏会让分析失真。
- **清洗层（clean_*) 独立成表**，与原始层同 key，随时可以从原始层重跑清洗。
  这是「清洗可关闭、可回退」的技术基础。
- 用 SQLite 而不是先上数据库：研究级数据量（万级到十万级行）SQLite 完全够，
  且零部署成本，学生换个电脑拷一个文件就能带走自己的研究数据。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from . import __version__
from .weibo.models import WeiboComment, WeiboPost
from .weibo.parsing import CST

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    tag          TEXT NOT NULL,
    order_mode   TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',
    note         TEXT
);

CREATE TABLE IF NOT EXISTS raw_posts (
    run_id               TEXT NOT NULL,
    mid                  TEXT NOT NULL,
    bid                  TEXT,
    user_id              TEXT,
    screen_name          TEXT,
    text_raw             TEXT,
    created_at           TEXT,
    created_at_raw       TEXT,
    created_at_precision TEXT,
    reposts_count        INTEGER,
    comments_count       INTEGER,
    attitudes_count      INTEGER,
    is_long_text         INTEGER,
    source               TEXT,
    fetched_at           TEXT NOT NULL,
    PRIMARY KEY (run_id, mid)
);

CREATE TABLE IF NOT EXISTS raw_comments (
    run_id               TEXT NOT NULL,
    cid                  TEXT NOT NULL,
    post_mid             TEXT NOT NULL,
    parent_cid           TEXT,
    level                INTEGER NOT NULL,
    user_id              TEXT,
    screen_name          TEXT,
    text_raw             TEXT,
    created_at           TEXT,
    created_at_raw       TEXT,
    created_at_precision TEXT,
    like_count           INTEGER,
    reply_count          INTEGER,
    followers_count      INTEGER,
    source               TEXT,
    fetched_at           TEXT NOT NULL,
    PRIMARY KEY (run_id, cid)
);

CREATE INDEX IF NOT EXISTS idx_raw_comments_post
    ON raw_comments (run_id, post_mid);

CREATE TABLE IF NOT EXISTS clean_comments (
    run_id          TEXT NOT NULL,
    cid             TEXT NOT NULL,
    post_mid        TEXT NOT NULL,
    cleaned_text    TEXT,
    kept            INTEGER NOT NULL,
    drop_reasons    TEXT,
    clean_tier      TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    PRIMARY KEY (run_id, cid)
);

CREATE TABLE IF NOT EXISTS research_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    event        TEXT NOT NULL,
    payload_json TEXT
);

-- 分析结果层。刻意与分析运行元信息分开：
-- 一次分析可能只产出一部分（只跑情感、或只跑聚类），
-- 把参数与方法卡片记在 analysis_runs，把逐条结果记在下面的明细表。
CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    module       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    params_json  TEXT,
    method_json  TEXT,
    summary_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_run
    ON analysis_runs (run_id, module, created_at);

CREATE TABLE IF NOT EXISTS analysis_comments (
    analysis_id     TEXT NOT NULL,
    cid             TEXT NOT NULL,
    sentiment       TEXT,
    intensity       REAL,
    local_sentiment TEXT,
    local_score     REAL,
    in_sample       INTEGER,
    cluster_discourse INTEGER,
    PRIMARY KEY (analysis_id, cid)
);

CREATE TABLE IF NOT EXISTS analysis_users (
    analysis_id      TEXT NOT NULL,
    user_pseudo_id   TEXT NOT NULL,
    cluster_community INTEGER,
    role             TEXT,
    PRIMARY KEY (analysis_id, user_pseudo_id)
);

-- 立场维度。刻意与 analysis_comments 分开：
-- 情感和立场是两个独立维度，各有各的 analysis_id，
-- 分表后"只跑情感不跑立场"或"重跑立场不影响情感"都不会互相踩。
CREATE TABLE IF NOT EXISTS analysis_stance (
    analysis_id TEXT NOT NULL,
    cid         TEXT NOT NULL,
    stance      TEXT,
    confidence  REAL,
    PRIMARY KEY (analysis_id, cid)
);
"""


def _now() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def _dt(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


class Store:
    """SQLite 存储门面。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ #
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('tool_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (__version__,),
            )

    # ------------------------------------------------------------------ #
    def create_run(
        self,
        *,
        run_id: str,
        tag: str,
        order_mode: str,
        params: dict,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collection_runs "
                "(run_id, created_at, tag, order_mode, params_json, tool_version, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'running')",
                (run_id, _now(), tag, order_mode,
                 json.dumps(params, ensure_ascii=False, indent=2), __version__),
            )

    def merge_run_params(self, run_id: str, extra: dict) -> None:
        """把后到的参数并进 ``params_json``。

        为什么要单独一个方法：采集和清洗是两步，而"清洗档位"要到清洗时才确定。
        如果只在采集时写一次参数，跨话题对比时就永远拿不到清洗档位，
        只能得出"参数未记录"的结论——那等于让可比性判断形同虚设。
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT params_json FROM collection_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return
            try:
                params = json.loads(row["params_json"] or "{}")
            except json.JSONDecodeError:
                params = {}
            params.update(extra)
            conn.execute(
                "UPDATE collection_runs SET params_json=? WHERE run_id=?",
                (json.dumps(params, ensure_ascii=False, indent=2), run_id),
            )

    def finish_run(self, run_id: str, *, status: str, note: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE collection_runs SET status=?, note=? WHERE run_id=?",
                (status, note, run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM collection_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ #
    def save_posts(self, run_id: str, posts: Iterable[WeiboPost]) -> int:
        rows = [
            (
                run_id, post.mid, post.bid, post.user_id, post.screen_name,
                post.text_raw, _dt(post.created_at), post.created_at_raw,
                post.created_at_precision.value, post.reposts_count,
                post.comments_count, post.attitudes_count, int(post.is_long_text),
                post.source, _now(),
            )
            for post in posts
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO raw_posts (run_id, mid, bid, user_id, screen_name, "
                "text_raw, created_at, created_at_raw, created_at_precision, "
                "reposts_count, comments_count, attitudes_count, is_long_text, "
                "source, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, mid) DO NOTHING",
                rows,
            )
        return len(rows)

    def save_comments(self, run_id: str, comments: Iterable[WeiboComment]) -> int:
        rows = [
            (
                run_id, comment.cid, comment.post_mid, comment.parent_cid,
                int(comment.level.value), comment.user_id, comment.screen_name,
                comment.text_raw, _dt(comment.created_at), comment.created_at_raw,
                comment.created_at_precision.value, comment.like_count,
                comment.reply_count, comment.followers_count, comment.source, _now(),
            )
            for comment in comments
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO raw_comments (run_id, cid, post_mid, parent_cid, level, "
                "user_id, screen_name, text_raw, created_at, created_at_raw, "
                "created_at_precision, like_count, reply_count, followers_count, "
                "source, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, cid) DO NOTHING",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------ #
    def load_comments(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM raw_comments WHERE run_id=? ORDER BY level, cid",
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def load_posts(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM raw_posts WHERE run_id=? ORDER BY attitudes_count DESC",
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def load_analysis_rows(self, run_id: str, *, only_kept: bool = True) -> list[dict]:
        """分析用的数据行：原始字段 + 清洗后的文本与决策。

        分析必须基于「清洗后保留」的数据，否则被判定为噪声的条目会重新进入
        情感分布——那等于白清洗。同时把清洗决策一并带出，便于回溯。
        """
        raw = {row["cid"]: row for row in self.load_comments(run_id)}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM clean_comments WHERE run_id=?", (run_id,)
            ).fetchall()
        merged: list[dict] = []
        for clean_row in rows:
            base = raw.get(clean_row["cid"])
            if base is None:
                continue
            kept = bool(clean_row["kept"])
            if only_kept and not kept:
                continue
            merged.append(
                {
                    **base,
                    "cleaned_text": clean_row["cleaned_text"],
                    "kept": kept,
                    "drop_reasons": clean_row["drop_reasons"],
                    "clean_tier": clean_row["clean_tier"],
                    "pipeline_version": clean_row["pipeline_version"],
                }
            )
        return merged

    def latest_run_id(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM collection_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return str(row["run_id"]) if row else None

    # ------------------------------------------------------------------ #
    # 分析结果层
    # ------------------------------------------------------------------ #
    def save_analysis_run(
        self,
        *,
        analysis_id: str,
        run_id: str,
        module: str,
        params: dict,
        method: dict,
        summary: dict,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_runs "
                "(analysis_id, run_id, module, created_at, params_json, method_json, summary_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    analysis_id, run_id, module, _now(),
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(method, ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False),
                ),
            )

    def save_sentiment_items(self, analysis_id: str, items: list[dict]) -> int:
        rows = [
            (
                analysis_id,
                str(item.get("cid")),
                item.get("sentiment"),
                item.get("intensity"),
                item.get("local_sentiment"),
                item.get("local_score"),
                1 if item.get("in_sample") else 0,
                None,
            )
            for item in items
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO analysis_comments "
                "(analysis_id, cid, sentiment, intensity, local_sentiment, local_score, "
                "in_sample, cluster_discourse) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(analysis_id, cid) DO UPDATE SET "
                "sentiment=excluded.sentiment, intensity=excluded.intensity, "
                "local_sentiment=excluded.local_sentiment, "
                "local_score=excluded.local_score, in_sample=excluded.in_sample",
                rows,
            )
        return len(rows)

    def save_stance_items(self, analysis_id: str, items: list[dict]) -> int:
        """写入立场标签（模块 stance）。"""
        rows = [
            (
                analysis_id,
                str(item.get("cid")),
                item.get("stance"),
                item.get("confidence"),
            )
            for item in items
            if item.get("stance")
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO analysis_stance (analysis_id, cid, stance, confidence) "
                "VALUES (?,?,?,?) "
                "ON CONFLICT(analysis_id, cid) DO UPDATE SET "
                "stance=excluded.stance, confidence=excluded.confidence",
                rows,
            )
        return len(rows)

    def save_discourse_clusters(self, analysis_id: str, assignments: list[dict]) -> int:
        rows = [
            (analysis_id, str(item.get("cid")), int(item.get("cluster", -1)))
            for item in assignments
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO analysis_comments (analysis_id, cid, cluster_discourse) "
                "VALUES (?,?,?) "
                "ON CONFLICT(analysis_id, cid) DO UPDATE SET "
                "cluster_discourse=excluded.cluster_discourse",
                rows,
            )
        return len(rows)

    def save_community_clusters(self, analysis_id: str, assignments: list[dict],
                                roles: dict[str, str]) -> int:
        rows = [
            (
                analysis_id,
                str(item.get("user_pseudo_id")),
                int(item.get("cluster", -1)),
                roles.get(str(item.get("cluster"))),
            )
            for item in assignments
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO analysis_users "
                "(analysis_id, user_pseudo_id, cluster_community, role) VALUES (?,?,?,?) "
                "ON CONFLICT(analysis_id, user_pseudo_id) DO UPDATE SET "
                "cluster_community=excluded.cluster_community, role=excluded.role",
                rows,
            )
        return len(rows)

    def latest_analysis_id(self, run_id: str) -> str | None:
        """取该会话最新一次分析（不区分模块），用于导出时补齐分析列。"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT analysis_id FROM analysis_runs WHERE run_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            return str(row["analysis_id"]) if row else None

    def analysis_runs(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_runs WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_analysis_summaries(self, run_id: str) -> dict[str, dict]:
        """每个模块最近一次的完整摘要（含渲染文本），供报告重建。"""
        out: dict[str, dict] = {}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT module, analysis_id, created_at, summary_json, method_json "
                "FROM analysis_runs WHERE run_id=? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        for row in rows:
            try:
                summary = json.loads(row["summary_json"] or "{}")
            except json.JSONDecodeError:
                summary = {}
            if not summary.get("method"):
                try:
                    summary["method"] = json.loads(row["method_json"] or "{}")
                except json.JSONDecodeError:
                    summary["method"] = {}
            summary["analysis_id"] = row["analysis_id"]
            summary["created_at"] = row["created_at"]
            out[str(row["module"])] = summary  # 后写的覆盖先写的 = 取最新
        return out

    def _latest_analysis_ids(self, run_id: str) -> list[str]:
        """每个模块只取最近一次分析。

        为什么要按模块取最新：同一会话可能先用模型 A 跑一次情感、再用模型 B 重跑。
        如果把两次结果按 cid 合并，导出的 CSV 会混着两套模型标签，
        而方法卡片只能记录其中一套——那是不可复现的数据。

        用 rowid 而不是 created_at 判断先后：同一秒内跑完两次分析时
        时间戳会相同，按时间排序会变成任意取一个。
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT module, analysis_id FROM analysis_runs "
                "WHERE run_id=? AND rowid IN ("
                "  SELECT MAX(rowid) FROM analysis_runs WHERE run_id=? GROUP BY module"
                ")",
                (run_id, run_id),
            ).fetchall()
            return [str(row["analysis_id"]) for row in rows if row["analysis_id"]]

    def load_comment_analysis(self, run_id: str) -> dict[str, dict]:
        """把该会话**每个模块最新一次**的分析结果按 cid 合并成一张表。"""
        analysis_ids = self._latest_analysis_ids(run_id)
        if not analysis_ids:
            return {}
        placeholders = ",".join("?" for _ in analysis_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM analysis_comments WHERE analysis_id IN ({placeholders})",
                tuple(analysis_ids),
            ).fetchall()
        merged: dict[str, dict] = {}
        for row in rows:
            cid = str(row["cid"])
            bucket = merged.setdefault(cid, {})
            for key in ("sentiment", "intensity", "local_sentiment", "local_score",
                        "cluster_discourse"):
                value = row[key]
                if value is not None:
                    bucket[key] = value
            if row["in_sample"]:
                bucket["in_sample"] = True
        # 立场在另一张表里，单独并入（模块名 stance）
        with self.connect() as conn:
            stance_rows = conn.execute(
                f"SELECT cid, stance, confidence FROM analysis_stance "
                f"WHERE analysis_id IN ({placeholders})",
                tuple(analysis_ids),
            ).fetchall()
        for row in stance_rows:
            if row["stance"] is None:
                continue
            merged.setdefault(str(row["cid"]), {}).update({
                "stance": row["stance"],
                "stance_confidence": row["confidence"],
            })
        return merged

    def load_user_analysis(self, run_id: str) -> dict[str, dict]:
        """用户级分析结果（社群聚类），同样只取最近一次。"""
        analysis_ids = self._latest_analysis_ids(run_id)
        if not analysis_ids:
            return {}
        placeholders = ",".join("?" for _ in analysis_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM analysis_users WHERE analysis_id IN ({placeholders})",
                tuple(analysis_ids),
            ).fetchall()
        merged: dict[str, dict] = {}
        for row in rows:
            merged[str(row["user_pseudo_id"])] = {
                "cluster_community": row["cluster_community"],
                "role": row["role"],
            }
        return merged

    def counts(self, run_id: str) -> dict[str, int]:
        with self.connect() as conn:
            out: dict[str, int] = {}
            for table in ("raw_posts", "raw_comments", "clean_comments"):
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE run_id=?", (run_id,)
                ).fetchone()
                out[table] = int(row["n"]) if row else 0
            return out

    # ------------------------------------------------------------------ #
    def save_clean(
        self,
        run_id: str,
        rows: Iterable[dict],
        *,
        clean_tier: str,
        pipeline_version: str,
    ) -> int:
        """写入清洗层。``rows`` 需含 ``cid`` / ``post_mid`` / ``cleaned_text``
        / ``kept`` / ``drop_reasons``。"""
        payload = [
            (
                run_id, row["cid"], row["post_mid"], row.get("cleaned_text", ""),
                int(bool(row.get("kept", True))),
                json.dumps(row.get("drop_reasons", []), ensure_ascii=False),
                clean_tier, pipeline_version,
            )
            for row in rows
        ]
        if not payload:
            return 0
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO clean_comments (run_id, cid, post_mid, cleaned_text, "
                "kept, drop_reasons, clean_tier, pipeline_version) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, cid) DO UPDATE SET "
                "cleaned_text=excluded.cleaned_text, kept=excluded.kept, "
                "drop_reasons=excluded.drop_reasons, clean_tier=excluded.clean_tier, "
                "pipeline_version=excluded.pipeline_version",
                payload,
            )
        return len(payload)

    def record_event(self, run_id: str, event: str, payload: dict | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO research_events (run_id, ts, event, payload_json) "
                "VALUES (?,?,?,?)",
                (run_id, _now(), event,
                 json.dumps(payload or {}, ensure_ascii=False)),
            )

    def events(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_events WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]


def has_table(path: Path, table: str) -> bool:
    if not path.exists():
        return False
    with closing(sqlite3.connect(str(path))) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
