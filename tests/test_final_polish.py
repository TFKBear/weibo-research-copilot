"""终版打磨：默认参数、报告结构稳定、助手懂软件。

这些点来自真实使用反馈：
- 打开软件时默认话题是「小米SU7」，用户以为已经选好了 → 必须留空；
- 候选帖子数写死 20、每帖 50 条 → 社科研究要的是大样本，必须可调；
- 报告缺了「九、事件前后对比」整节 → 结构不稳定，读者没法按节核对；
- 问助手"我想研究某个话题该怎么做"，它得知道界面怎么点。
"""

from __future__ import annotations

from wrc.analyze.report import build_analysis_report
from wrc.assistant.advisor import SOFTWARE_GUIDE, SYSTEM_PROMPT
from wrc.pipeline import MAX_COMMENTS_PER_POST


def test_report_always_has_section_nine():
    """没做事件研究也要有第九节，说明为什么没做、怎么补做。"""
    text = build_analysis_report(
        run={"run_id": "r1", "tag": "#t#"},
        cleaning={"tier": "moderate", "pipeline_version": "v", "total_input": 10,
                  "total_kept": 10, "retention_rate": 1.0},
    )
    assert "## 九、事件前后对比" in text
    assert "没有指定事件日期" in text
    # 不能让读者以为"第九节丢了"
    assert "用结果定义原因" in text


def test_comments_per_post_cap_allows_social_science_samples():
    """社科研究常要几百条/帖；上限卡在 500 会让大样本研究做不了。"""
    assert MAX_COMMENTS_PER_POST >= 1000


def test_assistant_prompt_knows_the_software():
    """助手必须知道五个区怎么操作，否则"指引"就是空话。"""
    for phrase in ("检索帖子（第一步）", "一键完成", "候选帖子数", "每帖评论数",
                   "图表", "导出"):
        assert phrase in SYSTEM_PROMPT, phrase
    assert "常见坑" in SOFTWARE_GUIDE
