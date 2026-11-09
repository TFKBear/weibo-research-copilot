"""连续出图必须保持中文字形，覆盖主题重置字体的真实回归。"""
import warnings

import pytest

from wrc.analyze import charts


def test_repeated_setup_renders_chinese_without_missing_glyphs(monkeypatch):
    from matplotlib import font_manager
    import matplotlib as mpl

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in charts._CJK_FONTS if name in available]
    if not chosen:
        pytest.skip("本机没有可用于中文渲染验收的字体")
    monkeypatch.setattr(charts, "_font_ready", None)
    with mpl.rc_context(), warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        for _ in range(3):
            plt = charts._setup()
            assert plt.rcParams["font.sans-serif"][0] == chosen[0]
            fig, ax = plt.subplots()
            ax.set_title("中文图表：情感分布与高频词")
            ax.set_xlabel("评论数量")
            ax.bar(["小米", "汽车"], [2, 3])
            fig.canvas.draw()
            plt.close(fig)
