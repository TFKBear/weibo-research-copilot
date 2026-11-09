"""交付物自检工具本身要可信。

一个自检工具最糟的失败方式是"什么都不报"——研究者会以为东西齐了。
所以这里既测"能通过"，也逐个测"该报的必须报"：
缺文件、泄露可识别字段、口径不一致、报告缺章节，每一种都要触发 ERROR。
"""

from __future__ import annotations

import json

import pytest

from tools.check_deliverables import check_docs, check_run


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _csv(path, header: list[str]) -> None:
    _write(path, ",".join(header) + "\n1,2\n")


def _build_export(root, *, report_extra="", appendix_window="不限时间（采该帖可见的最新评论）"):
    """造一个"合格"的导出目录，各测试按需破坏其中一处。"""
    _csv(root / "raw/raw_comments.csv", ["cid", "text_raw", "user_pseudo_id", "province"])
    _write(root / "raw/README.txt", "原始层")
    _write(root / "raw/manifest.json", json.dumps({"files": [], "consent": {}}))
    _csv(root / "analysis/analysis_comments.csv",
         ["cid", "sentiment", "stance", "cluster_discourse", "user_pseudo_id"])
    _write(root / "analysis/README.txt", "分析层")
    _write(
        root / "analysis/manifest.json",
        json.dumps({"files": [], "consent": {"confirmed": True}}, ensure_ascii=False),
    )
    _write(root / "report/sentiment_distribution.png", "fake")
    _write(
        root / "report/report.md",
        "\n".join(
            [
                "# 报告",
                "## 采集口径与样本边界",
                "检索口径：话题聚合（q=#话题#）",
                "评论时间窗：不限时间（采该帖可见的最新评论）",
                "实际覆盖：2026-09-02T09:14:00 ～ 2026-09-20T22:40:00",
                "## 一、数据清洗",
                "## 二、情感分析",
                "![整体情感分布](sentiment_distribution.png)",
                "## 三、立场分布（支持 / 反对）",
                "## 七、时间演变",
                "## 八、二级评论回复结构",
                "## 九、事件前后对比",
                "## 方法卡片 · 情感分析",
                "## 已知局限（务必在论文中披露）",
                report_extra,
            ]
        ),
    )
    _write(root / "report/report.docx", "fake")
    _write(
        root / "appendix.md",
        "\n".join(
            [
                "# 附录",
                "## 一、运行环境",
                "## 二、采集",
                "检索口径：话题聚合（q=#话题#）",
                f"评论时间窗：{appendix_window}",
                "实际覆盖：2026-09-02T09:14:00 ～ 2026-09-20T22:40:00（由已采数据反算）",
                "## 三、清洗",
                "## 四、分析模块",
                "## 六、可复现性分级",
                "## 八、产出文件与校验和",
                "## 十、已知局限",
            ]
        ),
    )
    _write(
        root / "method_note.md",
        "# 方法说明\n\n## 一、采集\n\n## 五、已知局限（请务必在论文中披露）\n",
    )
    _write(
        root / "research_log.md",
        "# 研究日志\n\n| 时间 | 事件 | 参数 |\n| --- | --- | --- |\n"
        "| 2026-10-26T19:53:46+08:00 | `run_start` | {\"tag\": \"#延迟退休#\"} |\n"
        "| 2026-10-26T19:54:04+08:00 | `comments_done` | {\"collected\": 50} |\n"
        "| 2026-10-26T19:57:11+08:00 | `collect_done` | {\"comments\": 404} |\n",
    )


def test_complete_export_passes(tmp_path):
    _build_export(tmp_path)
    result = check_run(tmp_path, "r1")
    assert result.ok, [issue.message for issue in result.issues]


def test_missing_file_is_an_error(tmp_path):
    _build_export(tmp_path)
    (tmp_path / "method_note.md").unlink()
    result = check_run(tmp_path, "r1")
    assert not result.ok
    assert any("method_note.md" in issue.message for issue in result.errors)


def test_identifier_leak_in_analysis_csv_is_fatal(tmp_path):
    _build_export(tmp_path)
    _csv(tmp_path / "analysis/analysis_comments.csv",
         ["cid", "user_id", "sentiment", "stance"])
    result = check_run(tmp_path, "r1")
    assert any("可识别字段" in issue.message for issue in result.errors)


def test_raw_without_pseudo_id_is_an_error(tmp_path):
    """去掉真实 ID 之后必须留下稳定伪 ID，否则既不能追溯也做不了用户级分析。"""
    _build_export(tmp_path)
    _csv(tmp_path / "raw/raw_comments.csv", ["cid", "text_raw"])
    result = check_run(tmp_path, "r1")
    assert any("user_pseudo_id" in issue.message for issue in result.errors)


def test_report_missing_heading_is_an_error(tmp_path):
    _build_export(tmp_path)
    text = (tmp_path / "report/report.md").read_text(encoding="utf-8")
    _write(tmp_path / "report/report.md", text.replace("## 三、立场分布（支持 / 反对）", "## 三、"))
    result = check_run(tmp_path, "r1")
    assert any("立场分布" in issue.message for issue in result.errors)


def test_referenced_chart_must_exist(tmp_path):
    _build_export(tmp_path)
    (tmp_path / "report/sentiment_distribution.png").unlink()
    result = check_run(tmp_path, "r1")
    assert any("引用的图不存在" in issue.message for issue in result.errors)


def test_window_wording_mismatch_is_caught(tmp_path):
    """报告说"不限时间"，附录说别的——这就是分叉，必须报错。"""
    _build_export(tmp_path, appendix_window="不限时间")
    result = check_run(tmp_path, "r1")
    assert any("评论时间窗不一致" in issue.message for issue in result.errors)


def test_coverage_mismatch_is_caught(tmp_path):
    _build_export(tmp_path)
    text = (tmp_path / "appendix.md").read_text(encoding="utf-8")
    _write(tmp_path / "appendix.md", text.replace("2026-09-02T09:14:00", "2026-09-01T00:00:00"))
    result = check_run(tmp_path, "r1")
    assert any("实际覆盖不一致" in issue.message for issue in result.errors)


def test_missing_manifest_file_entry_is_caught(tmp_path):
    _build_export(tmp_path)
    _write(
        tmp_path / "analysis/manifest.json",
        json.dumps({"files": [{"path": "analysis/not-there.csv"}], "consent": {}}),
    )
    result = check_run(tmp_path, "r1")
    assert any("清单登记但实际缺失" in issue.message for issue in result.errors)


def test_missing_consent_is_a_warning_not_an_error(tmp_path):
    _build_export(tmp_path)
    _write(tmp_path / "analysis/manifest.json", json.dumps({"files": [], "consent": {}}))
    result = check_run(tmp_path, "r1")
    assert result.ok
    assert any("数据使用声明" in issue.message for issue in result.warnings)


def test_docs_references_resolve():
    """仓库内文档引用必须都能落地，别指着不存在的文件说"见 XXX"。"""
    issues = [issue for issue in check_docs() if issue.level == "ERROR"]
    assert issues == [], [issue.message for issue in issues]


@pytest.mark.parametrize("relative", ["raw/raw_comments.csv", "report/report.md"])
def test_zero_byte_file_is_an_error(tmp_path, relative):
    _build_export(tmp_path)
    (tmp_path / relative).write_text("", encoding="utf-8")
    result = check_run(tmp_path, "r1")
    assert any("空文件" in issue.message for issue in result.errors)


def test_empty_research_log_is_an_error(tmp_path):
    """真实事故：导出在另一个进程里发生，日志没从磁盘读回事件，只剩表头。"""
    _build_export(tmp_path)
    _write(tmp_path / "research_log.md", "# 研究日志\n\n| 时间 | 事件 | 参数 |\n| --- | --- | --- |\n")
    result = check_run(tmp_path, "r1")
    assert any("没有任何事件行" in issue.message for issue in result.errors)
