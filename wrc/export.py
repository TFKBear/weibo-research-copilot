"""分层导出（模块 F1，第 1 步先落地原始层与分析层的最小可用版本）。

三条硬规则：

1. **任何层都不带可识别字段**。导出前会跑一次 ``assert_no_identifiers``，
   宁可让导出失败，也不输出带 ID 的学术数据。
2. **必须先有同意记录**。没有 ``ConsentRecord`` 就直接拒绝导出。
3. **每个文件带 SHA-256**。研究者可以把校验和写进附录，
   证明"论文里的数据就是这一份"，这是可复现性审查里最实用的一条。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import __version__
from .errors import ConfigProblem
from .privacy.anonymize import (
    PRIVACY_NOTES,
    PseudoIdHasher,
    anonymize_comment,
    anonymize_post,
    assert_no_identifiers,
)
from .privacy.consent import ConsentRecord
from .weibo.parsing import CST

RAW_COMMENT_COLUMNS: tuple[str, ...] = (
    "cid", "post_mid", "parent_cid", "level", "user_pseudo_id",
    "text_raw", "created_at", "created_at_raw", "created_at_precision",
    "like_count", "reply_count", "followers_count", "province",
)

ANALYSIS_COMMENT_COLUMNS: tuple[str, ...] = (
    "cid", "post_mid", "parent_cid", "level", "user_pseudo_id",
    "text", "text_raw", "created_at", "created_at_raw", "created_at_precision",
    "like_count", "reply_count", "followers_count", "province",
    "kept", "drop_reasons", "clean_tier", "pipeline_version",
    # 分析列：有对应分析结果时才填，否则留空。
    # 这些列是「分析结果能不能用进论文」的关键——没有它们，
    # 情感分析的结果只能停在 tools/ 的输出里。
    "sentiment", "intensity",
    "local_sentiment", "local_score", "in_sample",
    # 立场维度：与情感并列，不是替代关系。
    # 要回答"支持还是反对"，这一列才是主变量，情感是协变量。
    "stance", "stance_confidence",
    "cluster_discourse", "cluster_community", "user_role",
)

RAW_POST_COLUMNS: tuple[str, ...] = (
    "mid", "user_pseudo_id", "text_raw", "created_at", "created_at_raw",
    "created_at_precision", "reposts_count", "comments_count",
    "attitudes_count", "is_long_text",
)


@dataclass(slots=True)
class ExportedFile:
    path: Path
    rows: int
    sha256: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "rows": self.rows,
            "sha256": self.sha256,
            "description": self.description,
        }


@dataclass(slots=True)
class ExportManifest:
    run_id: str
    created_at: str
    out_dir: Path
    files: list[ExportedFile] = field(default_factory=list)
    consent: ConsentRecord | None = None
    method: dict[str, Any] = field(default_factory=dict)

    def write(self) -> Path:
        path = self.out_dir / "manifest.json"
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "tool_version": __version__,
            "consent": self.consent.to_dict() if self.consent else None,
            "method": self.method,
            "privacy": list(PRIVACY_NOTES),
            "files": [item.to_dict() for item in self.files],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


class Exporter:
    """按层导出。"""

    def __init__(self, store: Any, *, hasher: PseudoIdHasher, export_root: Path) -> None:
        self._store = store
        self._hasher = hasher
        self._root = Path(export_root)

    # ------------------------------------------------------------------ #
    def export_raw(
        self,
        run_id: str,
        *,
        consent: ConsentRecord,
        out_dir: Path | None = None,
        method: dict[str, Any] | None = None,
    ) -> ExportManifest:
        """原始层：清洗前数据（仍匿名化）。"""
        target = out_dir or self._run_dir(run_id, "raw")
        manifest = self._new_manifest(run_id, target, consent, method)

        comments = [
            anonymize_comment(row, self._hasher) for row in self._store.load_comments(run_id)
        ]
        assert_no_identifiers(comments)
        manifest.files.append(
            _write_csv(
                target / "raw_comments.csv",
                comments,
                RAW_COMMENT_COLUMNS,
                "清洗前评论（已匿名化），保留完整时间戳与回复关系字段",
            )
        )

        posts = [
            anonymize_post(row, self._hasher) for row in self._store.load_posts(run_id)
        ]
        assert_no_identifiers(posts)
        manifest.files.append(
            _write_csv(
                target / "raw_posts.csv",
                posts,
                RAW_POST_COLUMNS,
                "本次采集涉及的帖子元信息（已匿名化）",
            )
        )
        return manifest

    def export_analysis(
        self,
        run_id: str,
        *,
        consent: ConsentRecord,
        out_dir: Path | None = None,
        method: dict[str, Any] | None = None,
    ) -> ExportManifest:
        """分析层：清洗后数据 + 清洗决策 + 分析结果（若已运行过分析）。"""
        target = out_dir or self._run_dir(run_id, "analysis")
        manifest = self._new_manifest(run_id, target, consent, method)

        raw = {row["cid"]: row for row in self._store.load_comments(run_id)}
        clean = self._load_clean(run_id)
        if not clean:
            raise ConfigProblem(
                what_happened="分析层导出失败：本次采集还没有执行清洗。",
                detail=f"run_id={run_id} 的 clean_comments 为空",
            )

        # 分析结果（可能没有）。缺失时相关列留空，不报错——
        # 导出「只有清洗、还没做分析」的数据是完全正常的中间状态。
        comment_analysis: dict[str, dict] = {}
        user_analysis: dict[str, dict] = {}
        try:
            comment_analysis = self._store.load_comment_analysis(run_id)
            user_analysis = self._store.load_user_analysis(run_id)
        except Exception:  # noqa: BLE001 - 老库没有分析表时不应阻断导出
            comment_analysis, user_analysis = {}, {}
        if comment_analysis or user_analysis:
            manifest.method.setdefault("分析结果", "已并入（见 sentiment / cluster_* 列）")
        else:
            manifest.method.setdefault("分析结果", "未运行（相关列留空）")

        raw = {row["cid"]: row for row in self._store.load_comments(run_id)}
        clean = self._load_clean(run_id)
        if not clean:
            raise ConfigProblem(
                what_happened="分析层导出失败：本次采集还没有执行清洗。",
                detail=f"run_id={run_id} 的 clean_comments 为空",
            )

        rows: list[dict[str, Any]] = []
        for clean_row in clean:
            raw_row = raw.get(clean_row["cid"])
            if raw_row is None:
                continue
            merged = {
                **raw_row,
                "cleaned_text": clean_row["cleaned_text"],
                "clean_tier": clean_row["clean_tier"],
                "pipeline_version": clean_row["pipeline_version"],
            }
            anonymized = anonymize_comment(merged, self._hasher)
            anonymized["kept"] = bool(clean_row["kept"])
            anonymized["drop_reasons"] = _join_reasons(clean_row["drop_reasons"])
            anonymized["clean_tier"] = clean_row["clean_tier"]
            anonymized["pipeline_version"] = clean_row["pipeline_version"]

            # 分析列：按 cid 与用户伪 ID 合并；缺失即留空。
            analysis = comment_analysis.get(str(clean_row["cid"]), {})
            for key in ("sentiment", "intensity", "local_sentiment",
                        "local_score", "stance", "stance_confidence",
                        "cluster_discourse"):
                if key in analysis:
                    anonymized[key] = analysis[key]
            if analysis.get("in_sample"):
                anonymized["in_sample"] = True
            user = user_analysis.get(anonymized["user_pseudo_id"], {})
            if user:
                anonymized["cluster_community"] = user.get("cluster_community")
                anonymized["user_role"] = user.get("role")
            rows.append(anonymized)

        assert_no_identifiers(rows)
        manifest.files.append(
            _write_csv(
                target / "analysis_comments.csv",
                rows,
                ANALYSIS_COMMENT_COLUMNS,
                "清洗后评论 + 清洗决策（kept=False 的行也保留，便于复核误杀）",
            )
        )

        # 交叉表也导出，且**抑制在数据层完成**。
        # 若只把抑制做在界面/报告里，研究者把 CSV 拉到 Excel 自己交叉
        # 就绕过去了——那 k-匿名等于没做。
        crosstab_file = _write_crosstab(
            target, self._store, run_id, comment_analysis
        )
        if crosstab_file is not None:
            manifest.files.append(crosstab_file)
        stance_file = _write_stance_crosstab(target, comment_analysis)
        if stance_file is not None:
            manifest.files.append(stance_file)
        cluster_stance_file = _write_cluster_stance_crosstab(target, comment_analysis)
        if cluster_stance_file is not None:
            manifest.files.append(cluster_stance_file)
        return manifest

    # ------------------------------------------------------------------ #
    def _load_clean(self, run_id: str) -> list[dict]:
        with self._store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM clean_comments WHERE run_id=? ORDER BY cid", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def _run_dir(self, run_id: str, layer: str) -> Path:
        target = self._root / run_id / layer
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _new_manifest(
        self,
        run_id: str,
        out_dir: Path,
        consent: ConsentRecord,
        method: dict[str, Any] | None,
    ) -> ExportManifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        return ExportManifest(
            run_id=run_id,
            created_at=datetime.now(CST).isoformat(timespec="seconds"),
            out_dir=out_dir,
            consent=consent,
            method=method or {},
        )


# --------------------------------------------------------------------------- #
def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Iterable[str],
    description: str,
) -> ExportedFile:
    """写 CSV。

    编码用 ``utf-8-sig``：社科学生大多用 Excel 打开，
    不带 BOM 的中文 CSV 在 Excel 里会变成乱码 —— 这是最常被忽略的可用性细节。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in fieldnames})

    return ExportedFile(
        path=path,
        rows=len(rows),
        sha256=sha256_of(path),
        description=description,
    )


def _write_crosstab(
    target: Path,
    store: Any,
    run_id: str,
    comment_analysis: dict[str, dict],
) -> "ExportedFile | None":
    """导出「话语簇 × 情感倾向」交叉表（含小群体抑制）。

    只在这两个模块都有结果时才写；否则返回 None，让 manifest 里干净地少一个文件，
    而不是生成一张空表让人以为"交叉出来什么都没有"。
    """
    if not comment_analysis:
        return None
    has_cluster = any(
        item.get("cluster_discourse") is not None for item in comment_analysis.values()
    )
    has_sentiment = any(item.get("sentiment") for item in comment_analysis.values())
    if not (has_cluster and has_sentiment):
        return None

    from .analyze.crosstab import discourse_by_sentiment

    assignments = {
        cid: int(item["cluster_discourse"])
        for cid, item in comment_analysis.items()
        if item.get("cluster_discourse") is not None
    }
    rows = [
        {"cid": cid, "sentiment": item.get("sentiment")}
        for cid, item in comment_analysis.items()
    ]
    try:
        result = discourse_by_sentiment(rows, assignments)
    except Exception:  # noqa: BLE001 - 交叉表失败不应让导出整体失败
        return None
    if not result.groups:
        return None

    columns, payload = result.to_csv_rows()
    return _write_csv(
        target / "crosstab_discourse_sentiment.csv",
        payload,
        columns,
        (
            f"话语簇 × 情感倾向交叉表；单元格「—」表示样本量 <{result.min_n}，"
            "已按 k-匿名规则抑制（整行为 — 表示该簇整体样本量过小）"
        ),
    )


def _write_stance_crosstab(
    target: Path,
    comment_analysis: dict[str, dict],
) -> "ExportedFile | None":
    """导出「立场 × 情感」交叉表（含小群体抑制）。

    这张表是研究结论的落点：它能一眼看出"立场支持但情绪负面"那批人。
    抑制规则与别处一致：单元格 <5 条写「—」，且不计入行内占比。
    """
    has_stance = any(item.get("stance") for item in comment_analysis.values())
    has_sentiment = any(item.get("sentiment") for item in comment_analysis.values())
    if not (has_stance and has_sentiment):
        return None

    from .analyze.stance_crosstab import sentiment_by_stance, to_csv_rows

    rows = [
        {"cid": cid, "stance": item.get("stance"), "sentiment": item.get("sentiment")}
        for cid, item in comment_analysis.items()
    ]
    try:
        table = sentiment_by_stance(rows)
    except Exception:  # noqa: BLE001 - 交叉表失败不应让导出整体失败
        return None
    if not table.groups:
        return None

    columns, payload = to_csv_rows(table)
    return _write_csv(
        target / "crosstab_stance_sentiment.csv",
        payload,
        columns,
        (
            f"立场 × 情感 交叉表；「—」表示样本量 <{table.min_n} 已抑制，"
            "且不计入行内占比；「支持但负面」列是需要重点解读的那一格"
        ),
    )


def _write_cluster_stance_crosstab(
    target: Path,
    comment_analysis: dict[str, dict],
) -> "ExportedFile | None":
    """导出「话语簇 × 立场」交叉表。回答"不同话语阵营的态度结构是否不同"。"""
    has_stance = any(item.get("stance") for item in comment_analysis.values())
    assignments = {
        cid: int(item["cluster_discourse"])
        for cid, item in comment_analysis.items()
        if item.get("cluster_discourse") is not None
    }
    if not (has_stance and assignments):
        return None

    from .analyze.stance_crosstab import stance_by_discourse, to_csv_rows

    rows = [
        {"cid": cid, "stance": item.get("stance")}
        for cid, item in comment_analysis.items()
    ]
    try:
        table = stance_by_discourse(rows, assignments)
    except Exception:  # noqa: BLE001
        return None
    if not table.groups:
        return None

    columns, payload = to_csv_rows(
        table, group_label="话语簇", keep_support_negative_column=False
    )
    return _write_csv(
        target / "crosstab_cluster_stance.csv",
        payload,
        columns,
        (
            f"话语簇 × 立场 交叉表；「—」表示样本量 <{table.min_n} 已抑制，"
            "不计入行内占比；对比各簇的「反对」占比"
        ),
    )


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return value


def _join_reasons(value: Any) -> str:
    """库里存 JSON 数组（结构化），导出成 ``a;b``（Excel 里可读）。

    两者都要：数据库要保持可查询的结构，研究者打开 CSV 时不该看到 ``[\"...\"]``。
    """
    if not value:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return str(value)
    if isinstance(parsed, list):
        return ";".join(str(item) for item in parsed)
    return str(parsed)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_method_note(
    path: Path,
    *,
    run: dict[str, Any],
    params: dict[str, Any],
    cleaning: dict[str, Any] | None,
    manifest_rows: dict[str, int],
) -> Path:
    """写一份可直接粘进论文方法草稿的说明。

    这是 P1「方法卡片」的采集/清洗部分先行落地 ——
    第 1 步的验收要求里包含"去重导出"，如果导出的数据说不清是怎么来的，
    这次导出的学术价值就是零。
    """
    lines: list[str] = []
    lines.append("# 数据采集与清洗方法说明（草稿）")
    lines.append("")
    lines.append("> 本文档由博研助手自动生成，用于论文方法部分的初稿。")
    lines.append("> 请根据你的研究设计自行补充：研究问题、抽样理由、伦理审批信息。")
    lines.append("")
    lines.append("## 一、采集")
    lines.append("")
    lines.append(f"- 采集会话 ID：`{run.get('run_id')}`")
    lines.append("- 数据平台：微博（m.weibo.cn 公开接口）")
    lines.append(f"- 检索话题：{run.get('tag')}")
    lines.append(f"- 采集时间：{run.get('created_at')}（北京时间）")
    lines.append(f"- 工具版本：`{__version__}`")
    for key, value in params.items():
        lines.append(f"- {key}：{value}")
    lines.append("")
    lines.append("## 二、清洗")
    lines.append("")
    if cleaning:
        rate = cleaning.get("retention_rate")
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "—"
        lines.append(f"- 清洗档位：{cleaning.get('tier')}")
        lines.append(f"- 管线版本：`{cleaning.get('pipeline_version')}`")
        lines.append(
            f"- 输入 {cleaning.get('total_input')} 条，保留 {cleaning.get('total_kept')} 条，"
            f"保留率 {rate_text}"
        )
        for step in cleaning.get("steps", []):
            share = step.get("share")
            share_text = f"{share:.1%}" if isinstance(share, (int, float)) else "—"
            description = str(step.get("description") or "").strip()
            lines.append(
                f"  - {step.get('label', '')}：移除 {step.get('removed', 0)} 条"
                f"（{share_text}）" + (f"—— {description}" if description else "")
            )
    else:
        lines.append("- 本次未执行清洗。")
    lines.append("")
    lines.append("## 三、伦理与匿名化")
    lines.append("")
    for note in PRIVACY_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 四、数据文件")
    lines.append("")
    for name, count in manifest_rows.items():
        lines.append(f"- `{name}`：{count} 行")
    lines.append("")
    lines.append("## 五、已知局限（请务必在论文中披露）")
    lines.append("")
    sort_note = (
        "评论按时间倒序取自网页端接口；若设了时间窗，窗口早期的评论可能采不足，"
        "实际覆盖区间见附录「二、采集」，不要直接照抄设定的窗口。"
        if str(params.get("评论排序") or "").startswith("time")
        else "评论来源为平台「热门评论流」接口，偏向高互动评论，不代表该帖全部评论的分布；"
             "热门流的时间不单调，设了时间窗也不保证窗口内采全。"
    )
    for limitation in (
        sort_note,
        "检索结果排序与平台站内排序不完全一致；本工具的本地重排不等于平台排序。",
        "检索结果的时间字段部分仅精确到「日」，已通过 created_at_precision 字段标注。",
        "清洗档位与各步骤移除量已记录；如使用轻度档，样本中可能仍含营销与协同行为内容。",
        "检索口径若是「关键词全文」，样本含仅提及该词的帖子，不等于话题参与者。",
    ):
        lines.append(f"- {limitation}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_readme(path: Path, *, manifest: ExportManifest) -> Path:
    lines = [
        "本目录由博研助手导出",
        "=" * 40,
        f"采集会话：{manifest.run_id}",
        f"导出时间：{manifest.created_at}",
        "",
        "文件清单：",
    ]
    for item in manifest.files:
        lines.append(f"  - {item.path.name}（{item.rows} 行）{item.description}")
        lines.append(f"    sha256: {item.sha256}")
    lines.append("")
    lines.append("匿名化处理：")
    for note in PRIVACY_NOTES:
        lines.append(f"  - {note}")
    lines.append("")
    lines.append("CSV 使用 UTF-8 BOM 编码，可直接用 Excel 打开。")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
