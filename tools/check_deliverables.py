"""交付物自检：一次采集导出的东西齐不齐、对不对得上。

为什么要有这个工具：

导出目录是研究者**真正拿去写论文**的东西。缺一个文件、CSV 里漏了立场列、
报告与附录的时间窗写得不一样——这些都不会让程序报错，但会让人在几天后
提交论文时才发现。等到那时，前面的采集与清洗都白做了。

所以把它做成可执行的自检，而不是一份"请自行核对"的清单。

用法：
    python tools/check_deliverables.py                 # 检查最近一次采集
    python tools/check_deliverables.py --run <run_id>
    python tools/check_deliverables.py --docs          # 只检查文档引用是否齐全
    python tools/check_deliverables.py --strict        # 有 WARN 也返回非零

退出码：0 全部通过；1 有 ERROR（或有 WARN 且 --strict）；2 找不到采集。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 一次完整导出必须存在的文件（相对导出目录）。
REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("raw/raw_comments.csv", "原始评论（匿名化后）"),
    ("raw/manifest.json", "原始层清单与校验和"),
    ("raw/README.txt", "原始层说明"),
    ("analysis/analysis_comments.csv", "逐条分析结果"),
    ("analysis/manifest.json", "分析层清单与校验和"),
    ("analysis/README.txt", "分析层说明"),
    ("report/report.md", "分析报告"),
    ("report/report.docx", "Word 版报告"),
    ("appendix.md", "可复现性附录"),
    ("method_note.md", "方法说明草稿"),
    ("research_log.md", "研究日志"),
)

#: 分析数据里绝对不能出现的可识别字段（需求 C）。
IDENTIFIER_COLUMNS = ("user_id", "screen_name", "profile_url", "location", "raw_cid_owner")

#: 报告必须有的章节。
REPORT_HEADINGS = (
    "## 采集口径与样本边界",
    "## 一、数据清洗",
    "## 二、情感分析",
    "## 三、立场分布",
    "## 七、时间演变",
    "## 八、二级评论回复结构",
    "## 九、事件前后对比",
    "## 方法卡片 · 情感分析",
    "## 已知局限",
)

#: 附录必须有的章节。
APPENDIX_HEADINGS = (
    "## 一、运行环境",
    "## 二、采集",
    "## 三、清洗",
    "## 四、分析模块",
    "## 六、可复现性分级",
    "## 八、产出文件与校验和",
    "## 十、已知局限",
)


@dataclass(slots=True)
class Issue:
    level: str  # "ERROR" / "WARN"
    message: str

    def render(self) -> str:
        return f"  [{self.level}] {self.message}"


@dataclass(slots=True)
class Report:
    run_id: str
    export_dir: Path
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [item for item in self.issues if item.level == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"交付物自检 · {self.run_id}", "=" * 62, f"目录：{self.export_dir}"]
        if not self.issues:
            lines.append("  ✓ 全部通过")
            return "\n".join(lines)
        for issue in self.issues:
            lines.append(issue.render())
        lines.append("")
        lines.append(f"结论：{len(self.errors)} 个问题、{len(self.warnings)} 个提醒")
        return "\n".join(lines)


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _section(text: str, heading: str) -> str:
    """取出某个二级标题下的内容（到下一个二级标题为止）。"""
    index = text.find(heading)
    if index < 0:
        return ""
    rest = text[index + len(heading):]
    following = re.search(r"\n## ", rest)
    return rest[: following.start()] if following else rest


def _value_of(text: str, label: str) -> str | None:
    match = re.search(rf"{label}[：:]\s*(.+)", text)
    return match.group(1).strip() if match else None


def check_run(export_dir: Path, run_id: str) -> Report:
    """检查一个导出目录。纯读，不改任何东西。"""
    export_dir = Path(export_dir)
    report = Report(run_id=run_id, export_dir=export_dir)

    for relative, purpose in REQUIRED_FILES:
        path = export_dir / relative
        if not path.exists():
            report.issues.append(Issue("ERROR", f"缺文件：{relative}（{purpose}）"))
        elif path.stat().st_size == 0:
            report.issues.append(Issue("ERROR", f"空文件：{relative}（{purpose}）"))

    # ---- 数据层：匿名化与分析列 ----
    analysis_csv = export_dir / "analysis/analysis_comments.csv"
    if analysis_csv.exists():
        header = _read_csv_header(analysis_csv)
        leaked = [column for column in header if column in IDENTIFIER_COLUMNS]
        if leaked:
            report.issues.append(
                Issue("ERROR", f"分析数据含可识别字段：{', '.join(leaked)}")
            )
        for column in ("sentiment", "stance"):
            if column not in header:
                report.issues.append(
                    Issue("WARN", f"分析数据缺 {column} 列（该模块可能没跑）")
                )

    raw_csv = export_dir / "raw/raw_comments.csv"
    if raw_csv.exists():
        header = _read_csv_header(raw_csv)
        leaked = [column for column in header if column in IDENTIFIER_COLUMNS]
        if leaked:
            report.issues.append(
                Issue("ERROR", f"原始数据含可识别字段：{', '.join(leaked)}")
            )
        if "user_pseudo_id" not in header:
            report.issues.append(
                Issue("ERROR", "原始数据缺 user_pseudo_id：无法追溯又不能识别个体")
            )

    # ---- 清单一致性 ----
    manifest_path = export_dir / "analysis/manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            report.issues.append(Issue("ERROR", f"分析层清单损坏：{exc}"))
            manifest = {}
        files = manifest.get("files") or []
        missing = [
            item.get("path") for item in files
            if item.get("path") and not (export_dir / item["path"]).exists()
        ]
        if missing:
            report.issues.append(
                Issue("ERROR", f"清单登记但实际缺失：{', '.join(str(m) for m in missing[:5])}")
            )
        if not manifest.get("consent"):
            report.issues.append(Issue("WARN", "清单里没有记录数据使用声明"))

    # ---- 报告结构 ----
    report_md = export_dir / "report/report.md"
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8")
        for heading in REPORT_HEADINGS:
            if heading not in text:
                report.issues.append(Issue("ERROR", f"报告缺章节：{heading}"))
        if not re.search(r"!\[[^\]]*\]\([^)]+\.png\)", text):
            report.issues.append(Issue("ERROR", "报告里没有任何图片引用"))
        for name in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", text):
            if not (export_dir / "report" / name).exists():
                report.issues.append(Issue("ERROR", f"报告引用的图不存在：{name}"))
        charts = sorted((export_dir / "report").glob("*.png"))
        # 图表数量是**用户选择**的结果（界面可只留核心四张），不该当成缺陷；
        # 只有一张都没有才是问题——那说明画图这步根本没跑。
        if not charts:
            report.issues.append(Issue("ERROR", "report/ 里一张图都没有"))
        elif len(charts) < 3:
            report.issues.append(
                Issue("WARN", f"图表只有 {len(charts)} 张（可能是你只勾选了少数几张）")
            )

    # ---- 附录结构 ----
    appendix = export_dir / "appendix.md"
    if appendix.exists():
        text = appendix.read_text(encoding="utf-8")
        for heading in APPENDIX_HEADINGS:
            if heading not in text:
                report.issues.append(Issue("ERROR", f"附录缺章节：{heading}"))

    # ---- 口径一致性（真实产物里最容易分叉的地方）----
    if report_md.exists() and appendix.exists():
        report_text = report_md.read_text(encoding="utf-8")
        appendix_text = appendix.read_text(encoding="utf-8")
        for label in ("检索口径", "评论时间窗"):
            left = _value_of(_section(report_text, "## 采集口径与样本边界"), label)
            right = _value_of(_section(appendix_text, "## 二、采集"), label)
            if left and right and left != right:
                report.issues.append(
                    Issue("ERROR", f"{label}不一致：报告「{left}」/ 附录「{right}」")
                )
        for label in ("实际覆盖",):
            left = _value_of(_section(report_text, "## 采集口径与样本边界"), label)
            right = _value_of(_section(appendix_text, "## 二、采集"), label)
            if left and right and not right.startswith(left):
                report.issues.append(
                    Issue("ERROR", f"{label}不一致：报告「{left}」/ 附录「{right}」")
                )

    # ---- 方法说明 ----
    method_note = export_dir / "method_note.md"
    if method_note.exists():
        text = method_note.read_text(encoding="utf-8")
        for heading in ("## 一、采集", "## 五、已知局限"):
            if heading not in text:
                report.issues.append(Issue("ERROR", f"方法说明缺章节：{heading}"))
        if "已知局限" in text and "论文" not in text:
            report.issues.append(Issue("WARN", "方法说明的局限段没提论文披露"))

    # ---- 研究日志 ----
    # 真实事故：导出的 research_log.md 只有表头、一行事件都没有
    # （导出在另一个进程里，内存中没有事件）。而它正是过程记录本身。
    research_log = export_dir / "research_log.md"
    if research_log.exists():
        text = research_log.read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("| 2")]
        if not rows:
            report.issues.append(
                Issue("ERROR", "研究日志没有任何事件行（可能没从磁盘读回事件）")
            )
        elif len(rows) < 3:
            report.issues.append(
                Issue("WARN", f"研究日志只有 {len(rows)} 条事件，过程记录过少")
            )

    return report


def check_docs(root: Path = ROOT) -> list[Issue]:
    """文档引用自检：决策记录与清单里提到的文件是否真的存在。"""
    issues: list[Issue] = []
    docs = Path(root) / "docs"
    decisions = sorted((docs / "DECISIONS").glob("*.md"))
    if not decisions:
        issues.append(Issue("ERROR", "docs/DECISIONS 下没有任何决策记录"))
    for path in (docs / "NEEDS-REVIEW.md", docs / "SPEC-COVERAGE.md"):
        if not path.exists():
            issues.append(Issue("ERROR", f"缺文档：{path.name}"))
            continue
        text = path.read_text(encoding="utf-8")
        for reference in re.findall(r"`(docs/[^`]+\.(?:md|pdf))`", text):
            if not (Path(root) / reference).exists():
                issues.append(Issue("ERROR", f"{path.name} 引用了不存在的文件：{reference}"))
    # 决策记录编号必须连续，缺号说明有记录没写完或被误删
    numbers = []
    for path in decisions:
        match = re.match(r"(\d{3})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        expected = list(range(1, max(numbers) + 1))
        missing = sorted(set(expected) - set(numbers))
        if missing:
            issues.append(
                Issue("WARN", f"决策记录缺号：{', '.join(f'{n:03d}' for n in missing)}")
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="交付物自检")
    parser.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    parser.add_argument("--dir", default=None, help="直接指定导出目录（覆盖 --run）")
    parser.add_argument("--docs", action="store_true", help="只检查文档引用")
    parser.add_argument("--strict", action="store_true", help="有 WARN 也返回非零")
    args = parser.parse_args(argv)

    if args.docs:
        issues = check_docs()
        for issue in issues:
            print(issue.render())
        if not issues:
            print("文档引用自检：全部通过")
        return 0 if not [i for i in issues if i.level == "ERROR"] else 1

    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()

    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2

    export_dir = Path(args.dir) if args.dir else Path(settings.export_dir) / run_id
    if not export_dir.exists():
        print(f"导出目录不存在：{export_dir}")
        print("先跑一次导出：python -m wrc.cli run … 或 wrc reclean --run <id>")
        return 2

    result = check_run(export_dir, run_id)
    print(result.render())

    doc_issues = [issue for issue in check_docs() if issue.level == "ERROR"]
    for issue in doc_issues:
        print(issue.render())

    failed = bool(result.errors or doc_issues)
    if args.strict and result.warnings:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
