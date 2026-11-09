"""Markdown 报告 → Word(.docx) 转换（P2「报告导出」的落地）。

为什么要 Word 而不是只给 Markdown：
社科论文的正文、批注、导师修改全在 Word 里。让研究者把 Markdown 粘贴过去，
表格和图片会散架，方法卡片里的等宽块也会变形。所以需求是「一键拿到 .docx」，
而不是「拿到一份你还得自己排版的半成品」。

设计取舍：

- **不引入 Markdown 解析库**。本工具只生成固定形态的报告（标题/表格/图/代码块/
  列表/引用），为这几种结构装一个通用 Markdown 引擎不划算，而且通用引擎对
  「代码块里其实是方法卡片」这种语境的排版往往还不如手写。
- **不生成 PDF**。PDF 需要排版引擎（reportlab/weasyprint），装得上但会显著
  拉高安装体积与失败面。Word 另存 PDF 是一步操作，且研究者本来就要改 Word。
  这条取舍写进 ``docs/NEEDS-REVIEW.md``。
- **图片按目录相对解析**。报告里的 ``![](xxx.png)`` 是相对导出目录的，
  转换时用 ``base_dir`` 解析；找不到图不报错（报告仍可用），只在该处留一行说明。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["markdown_to_docx", "DOCX_SUFFIX"]

DOCX_SUFFIX = ".docx"

#: 图片最大宽度（cm）。A4 正文区约 16cm，留一点余量给左右边距。
MAX_IMAGE_WIDTH_CM = 15.5

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_FENCE = re.compile(r"^\s*```(\w*)\s*$")

#: 行内格式。``**粗体**`` 必须排在 ``*斜体*`` 之前，否则会被前半段先吃掉。
_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<!\*)\*[^*\n]+?\*(?!\*))")


def markdown_to_docx(
    markdown: str,
    out_path: Path | str,
    *,
    base_dir: Path | str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """把报告 Markdown 转成 .docx。

    Args:
        markdown: 报告正文（``build_analysis_report`` 的返回值）。
        out_path: 输出路径；目录会自动创建。
        base_dir: 解析相对图片路径的基准目录，通常是报告所在目录。
        title: 文档属性里的标题；省略时取一级标题。
        subtitle: 可选的封面副标题（写进文档属性，不占正文）。

    Returns:
        实际写入的文件路径。
    """
    # 延迟导入：没装 python-docx 时，其他功能（采集/清洗/分析）不该被拖累。
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(base_dir) if base_dir is not None else path.parent

    doc = Document()
    _setup_styles(doc, qn=qn, Pt=Pt)

    blocks = _tokenize(markdown)
    first_heading = next(
        (text for kind, text, _ in blocks if kind == "heading" and text), ""
    )
    doc.core_properties.title = title or _plain(first_heading) or "微博舆论分析报告"
    if subtitle:
        doc.core_properties.subject = subtitle

    for kind, text, extra in blocks:
        if kind == "heading":
            level = int(extra or 1)
            doc.add_heading(_plain(text), level=level)
        elif kind == "paragraph":
            _add_inline(doc.add_paragraph(), text, qn=qn, Pt=Pt)
        elif kind == "code":
            _add_code_block(doc, extra, qn=qn, Pt=Pt)
        elif kind == "bullet":
            level = int(extra or 0)
            style = "List Bullet" if level <= 0 else f"List Bullet {min(level + 1, 3)}"
            _add_inline(doc.add_paragraph(style=style), text, qn=qn, Pt=Pt)
        elif kind == "numbered":
            level = int(extra or 0)
            style = "List Number" if level <= 0 else f"List Number {min(level + 1, 3)}"
            _add_inline(doc.add_paragraph(style=style), text, qn=qn, Pt=Pt)
        elif kind == "quote":
            paragraph = doc.add_paragraph(style="Intense Quote")
            _add_inline(paragraph, text, qn=qn, Pt=Pt)
        elif kind == "rule":
            _add_horizontal_rule(doc, qn=qn)
        elif kind == "table":
            _add_table(doc, extra, qn=qn)
        elif kind == "image":
            _add_image(
                doc, alt=text, src=str(extra), root=root, Pt=Pt, align=WD_ALIGN_PARAGRAPH,
            )

    doc.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# 词法分析：把 Markdown 变成 (类型, 文本, 附加信息) 序列
# --------------------------------------------------------------------------- #
def _tokenize(markdown: str) -> list[tuple[str, str, Any]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[tuple[str, str, Any]] = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        # --- 代码围栏：整块收走，内部不再解释任何 Markdown ---
        fence = _FENCE.match(line)
        if fence:
            index += 1
            body: list[str] = []
            while index < total and not _FENCE.match(lines[index]):
                body.append(lines[index])
                index += 1
            index += 1  # 跳过结束围栏
            blocks.append(("code", "", "\n".join(body)))
            continue

        if not line.strip():
            index += 1
            continue

        # --- 表格：连续的 | 开头的行 ---
        if line.lstrip().startswith("|"):
            table_lines: list[str] = []
            while index < total and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = _parse_table(table_lines)
            if rows:
                blocks.append(("table", "", rows))
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(("heading", heading.group(2).strip(), len(heading.group(1))))
            index += 1
            continue

        if _RULE.match(line):
            blocks.append(("rule", "", None))
            index += 1
            continue

        image = _IMAGE.match(line.strip())
        if image:
            alt, src = image.group(1), image.group(2).strip()
            blocks.append(("image", alt, src))
            index += 1
            continue

        bullet = _BULLET.match(line)
        if bullet:
            level = len(bullet.group(1)) // 2
            blocks.append(("bullet", bullet.group(2).strip(), level))
            index += 1
            continue

        numbered = _NUMBERED.match(line)
        if numbered:
            level = len(numbered.group(1)) // 2
            blocks.append(("numbered", numbered.group(2).strip(), level))
            index += 1
            continue

        if line.lstrip().startswith(">"):
            blocks.append(("quote", line.lstrip().lstrip(">").strip(), None))
            index += 1
            continue

        # --- 普通段落：把连续的非空行并成一段 ---
        paragraph: list[str] = [line.strip()]
        index += 1
        while index < total:
            probe = lines[index]
            if not probe.strip():
                break
            if (
                _HEADING.match(probe)
                or _FENCE.match(probe)
                or _RULE.match(probe)
                or _BULLET.match(probe)
                or _NUMBERED.match(probe)
                or probe.lstrip().startswith("|")
                or probe.lstrip().startswith(">")
                or _IMAGE.match(probe.strip())
            ):
                break
            paragraph.append(probe.strip())
            index += 1
        blocks.append(("paragraph", " ".join(paragraph), None))

    # 图片在 tokenize 阶段先解析成 image，真正判断文件是否存在放到渲染时
    return blocks


def _parse_table(table_lines: Sequence[str]) -> list[list[str]]:
    """解析管道表格；返回去掉分隔行的二维列表。"""
    rows: list[list[str]] = []
    for raw in table_lines:
        stripped = raw.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        if _TABLE_DIVIDER.match(raw):
            continue
        cells = [_plain(cell.strip()) for cell in stripped.split("|")]
        if any(cells):
            rows.append(cells)
    return rows


# --------------------------------------------------------------------------- #
# 渲染辅助
# --------------------------------------------------------------------------- #
def _plain(text: str) -> str:
    """剥掉行内标记，用于标题/表格这类不该带格式的地方。"""
    text = text.replace("\\|", "|")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _setup_styles(doc: Any, *, qn: Any, Pt: Any) -> None:
    """设定中文字体与页边距。

    python-docx 默认模板的标题字体是英文字体，中文会回退成宋体且字号不协调；
    这里显式指定 eastAsia 字体，Word 打开后才是"像一篇论文"的样子。
    """
    from docx.shared import Cm as _Cm

    section = doc.sections[0]
    section.left_margin = _Cm(2.6)
    section.right_margin = _Cm(2.6)
    section.top_margin = _Cm(2.5)
    section.bottom_margin = _Cm(2.5)

    for style_name in ("Normal", "Heading 1", "Heading 2", "Heading 3",
                       "List Bullet", "List Number", "Intense Quote"):
        try:
            style = doc.styles[style_name]
        except KeyError:
            continue
        font = style.font
        rpr = style.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(fonts)
        fonts.set(qn("w:eastAsia"), "宋体" if style_name == "Normal" else "黑体")
        fonts.set(qn("w:ascii"), "Times New Roman" if style_name == "Normal" else "Arial")
        fonts.set(qn("w:hAnsi"), "Times New Roman" if style_name == "Normal" else "Arial")

    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)


def _add_inline(paragraph: Any, text: str, *, qn: Any, Pt: Any) -> None:
    """写入一段带行内格式的文本。"""
    if not text:
        return
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            paragraph.add_run(text[position:match.start()])
        chunk = match.group(0)
        if chunk.startswith("**") and chunk.endswith("**"):
            paragraph.add_run(chunk[2:-2]).bold = True
        elif chunk.startswith("`") and chunk.endswith("`"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        else:
            paragraph.add_run(chunk[1:-1]).italic = True
        position = match.end()
    if position < len(text):
        paragraph.add_run(text[position:])


def _add_code_block(doc: Any, body: str, *, qn: Any, Pt: Any) -> None:
    """代码块 → 等宽、小字号、带底纹的段落。

    报告里的代码块装的其实是方法卡片的纯文本表格，
    用段落保留换行比塞进表格更接近原文的阅读感。
    """
    for line in body.split("\n"):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.left_indent = Pt(12)
        run = paragraph.add_run(line if line.strip() else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        ppr = paragraph._p.get_or_add_pPr()
        shading = ppr.makeelement(qn("w:shd"), {
            qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): "F5F5F5",
        })
        ppr.append(shading)


def _add_horizontal_rule(doc: Any, *, qn: Any) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = 0
    paragraph.paragraph_format.space_after = 0
    ppr = paragraph._p.get_or_add_pPr()
    borders = ppr.makeelement(qn("w:pBdr"), {})
    bottom = borders.makeelement(qn("w:bottom"), {
        qn("w:val"): "single", qn("w:sz"): "6",
        qn("w:space"): "1", qn("w:color"): "BFBFBF",
    })
    borders.append(bottom)
    ppr.append(borders)


def _add_table(doc: Any, rows: Iterable[Sequence[str]], *, qn: Any) -> None:
    rows = [list(row) for row in rows]
    if not rows:
        return
    width = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for col_index in range(width):
            cell = table.cell(row_index, col_index)
            # 不要先赋 cell.text：那会往格子里塞一个空 run，
            # 后面 add_run 的格式就落到 runs[1] 上了。
            paragraph = cell.paragraphs[0]
            value = row[col_index] if col_index < len(row) else ""
            run = paragraph.add_run(value)
            if row_index == 0:
                run.bold = True
    doc.add_paragraph()


def _add_image(
    doc: Any, *, alt: str, src: str, root: Path, Pt: Any, align: Any
) -> None:
    from docx.shared import Cm as _Cm

    candidate = Path(src)
    if not candidate.is_absolute():
        candidate = (root / src).resolve()
    if not candidate.exists():
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(f"[图片未找到：{src}]")
        run.italic = True
        paragraph.alignment = align.CENTER
        return

    paragraph = doc.add_paragraph()
    paragraph.alignment = align.CENTER
    paragraph.add_run().add_picture(str(candidate), width=_Cm(MAX_IMAGE_WIDTH_CM))
    if alt:
        caption = doc.add_paragraph()
        caption.alignment = align.CENTER
        run = caption.add_run(alt)
        run.italic = True
        run.font.size = Pt(9)


# --------------------------------------------------------------------------- #
def export_report_docx(
    markdown: str,
    out_dir: Path | str,
    *,
    stem: str = "report",
) -> Path:
    """便捷入口：在 ``out_dir`` 里生成 ``<stem>.docx``。"""
    target = Path(out_dir)
    return markdown_to_docx(markdown, target / f"{stem}{DOCX_SUFFIX}", base_dir=target)
