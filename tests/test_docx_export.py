"""Word 导出的测试。

重点不是"能不能生成一个文件"（那太容易通过），而是：
- 表格真的变成 Word 表格（不是一堆带竖线的段落）；
- 图片真的嵌进去了（不是留下一个 ``![](x.png)``）；
- 代码块里的等宽内容不丢字（方法卡片就住在里面）；
- 图表缺失时不崩，而是留一行可读的说明。
"""

from __future__ import annotations

import pytest

docx = pytest.importorskip("docx")  # 没装 python-docx 就整体跳过，不算失败


SAMPLE = """\
# 微博舆论分析报告 · 测试话题

- 采集会话：`20250101T000000-abcd`
- 生成时间：2025-01-01 10:00:00

> 本报告由博研助手生成，用于论文写作的初稿。

---

## 一、数据清洗

- 清洗档位：moderate
- 输入 50 条，保留 46 条（92.0%）

| 步骤 | 移除 | 占输入 |
| --- | ---: | ---: |
| 去重 | 4 | 8.0% |
| 广告 | 0 | 0.0% |

## 二、情感分析

```
情感分布（LLM 通道）
====================
正面 12 条（26.1%）
负面 20 条（43.5%）
```

![整体情感分布](sentiment_distribution.png)
"""


def test_docx_contains_headings_and_paragraphs(tmp_path):
    from wrc.analyze.docx_export import markdown_to_docx

    out = markdown_to_docx(SAMPLE, tmp_path / "report.docx", base_dir=tmp_path)
    assert out.exists() and out.stat().st_size > 0

    document = docx.Document(str(out))
    texts = [paragraph.text for paragraph in document.paragraphs]
    joined = "\n".join(texts)
    assert "微博舆论分析报告 · 测试话题" in joined
    assert "清洗档位：moderate" in joined
    # 标题用的 Heading 样式，不是加粗正文
    styles = {paragraph.style.name for paragraph in document.paragraphs}
    assert "Heading 1" in styles
    assert "Heading 2" in styles


def test_table_becomes_real_word_table(tmp_path):
    from wrc.analyze.docx_export import markdown_to_docx

    out = markdown_to_docx(SAMPLE, tmp_path / "report.docx", base_dir=tmp_path)
    document = docx.Document(str(out))
    assert len(document.tables) == 1
    table = document.tables[0]
    assert len(table.rows) == 3  # 表头 + 两行数据，分隔行已被丢弃
    assert table.cell(0, 0).text == "步骤"
    assert table.cell(2, 0).text == "广告"
    # 表头加粗是报告可读性的最低要求
    assert table.cell(0, 0).paragraphs[0].runs[0].bold is True


def test_code_block_text_is_preserved(tmp_path):
    from wrc.analyze.docx_export import markdown_to_docx

    out = markdown_to_docx(SAMPLE, tmp_path / "report.docx", base_dir=tmp_path)
    document = docx.Document(str(out))
    joined = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "正面 12 条（26.1%）" in joined
    assert "负面 20 条（43.5%）" in joined


def test_missing_image_leaves_a_readable_note(tmp_path):
    from wrc.analyze.docx_export import markdown_to_docx

    out = markdown_to_docx(SAMPLE, tmp_path / "report.docx", base_dir=tmp_path)
    document = docx.Document(str(out))
    joined = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "图片未找到" in joined


def test_existing_image_is_embedded(tmp_path):
    """用 matplotlib 现画一张小图，确认真的嵌入而不是只写了个占位说明。"""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3, 2))
    axis.plot([1, 2, 3], [3, 2, 1])
    image_path = tmp_path / "sentiment_distribution.png"
    figure.savefig(image_path, dpi=80)
    plt.close(figure)

    from wrc.analyze.docx_export import markdown_to_docx

    markdown = "# 报告\n\n![整体情感分布](sentiment_distribution.png)\n"
    out = markdown_to_docx(markdown, tmp_path / "report.docx", base_dir=tmp_path)
    document = docx.Document(str(out))
    # 图片是 inline shape，不算段落文字；有 shape 就说明嵌进去了
    assert len(document.inline_shapes) == 1
    assert document.inline_shapes[0].width > 0


def test_inline_formatting_is_applied(tmp_path):
    from wrc.analyze.docx_export import markdown_to_docx

    markdown = "# 报告\n\n这里有 **加粗** 和 `code` 还有 *斜体*。\n"
    out = markdown_to_docx(markdown, tmp_path / "report.docx", base_dir=tmp_path)
    document = docx.Document(str(out))
    paragraph = next(p for p in document.paragraphs if "加粗" in p.text)
    flags = {(run.text, run.bold, run.italic) for run in paragraph.runs}
    assert ("加粗", True, None) in flags
    assert any(text == "code" for text, _bold, _italic in flags)
    assert any(italic for _text, _bold, italic in flags)


def test_export_report_docx_uses_stem(tmp_path):
    from wrc.analyze.docx_export import export_report_docx

    path = export_report_docx("# 报告\n\n正文。\n", tmp_path)
    assert path.name == "report.docx"
    assert path.exists()
