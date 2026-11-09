"""界面冒烟：验证所有控件能正确构建，不弹窗打扰人。

做法：设置 ``WRC_GUI_SMOKE=1`` 让窗口以隐藏方式创建，建完就销毁。
这样既能发现"某个控件属性写错导致启动即崩"，又不会真的弹出窗口。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["WRC_GUI_SMOKE"] = "1"


def _fake_posts(count: int):
    from wrc.weibo.models import TimePrecision, WeiboPost

    return [
        WeiboPost(
            mid=f"m{index}", user_id="u", screen_name="某人", text_raw=f"帖子 {index}",
            created_at=None, created_at_raw="", created_at_precision=TimePrecision.UNKNOWN,
            comments_count=10 * index, attitudes_count=index,
        )
        for index in range(1, count + 1)
    ]


def _check_checkboxes(app) -> None:
    """复选框：勾选状态按 mid 存，全选/反选/清空都要对得上。"""
    app.posts = _fake_posts(3)
    app.tree.delete(*app.tree.get_children())
    for index in range(1, 4):
        app.tree.insert("", "end", iid=str(index), values=("",))
        app._render_row(index)

    app._toggle_checked(1)
    assert app.checked == {"m1"}, app.checked
    assert app._checked_posts()[0].mid == "m1"

    app._select_all()
    assert len(app._checked_posts()) == 3
    app._invert_selection()
    assert app.checked == set(), app.checked
    app._select_none()
    app._toggle_checked(2)
    app._toggle_checked(2)  # 再点一次取消
    assert app.checked == set(), app.checked

    # 勾选的帖子才进入采集参数
    app._toggle_checked(3)
    app.per_post_targets["m3"] = 123
    opts = app._collect_options()
    assert opts.per_post_targets == {"m3": 123}, opts.per_post_targets
    print(f"复选框：勾选 {sorted(app.checked)}，采集目标 {opts.targets_for(app._checked_posts())}")


def _check_empty_state(app) -> None:
    """空状态：第一次打开时列表是空的，必须有一句话告诉人下一步做什么。"""
    app.posts = []
    app._update_post_hint()
    assert app.post_hint.winfo_manager() == "pack", "空列表时提示应当可见"
    assert "推荐话题" in app.post_hint.cget("text")

    app.posts = _fake_posts(1)
    app._update_post_hint()
    assert app.post_hint.winfo_manager() == "", "有候选帖子后提示应当隐藏"

    app.posts = []
    app._update_post_hint()
    print("空状态：无候选帖子时提示可见，有数据时自动隐藏")


def _check_window_parsing(app) -> None:
    """时间窗：只填日期时终点取当天 23:59；空值 = 不限。"""
    app.since_var.set("2026-09-01")
    app.until_var.set("2026-09-30")
    since, until = app._window_from_ui()
    assert since.day == 1 and since.hour == 0, since
    assert until.day == 30 and until.hour == 23, until

    app._clear_window()
    assert app._window_from_ui() == (None, None)

    app._quick_window(7)
    since, until = app._window_from_ui()
    assert since is not None and until is not None and since <= until
    assert (until - since).days == 6, (since, until)

    app.since_var.set("2026-10-01")
    app.until_var.set("2026-09-01")  # 填反了
    try:
        app._window_from_ui()
    except Exception as exc:  # noqa: BLE001
        assert "起点晚于终点" in str(exc), exc
    else:
        raise AssertionError("时间窗填反时应当报错，而不是静默继续")
    app._clear_window()
    print("时间窗：解析、快捷窗口、起止颠倒拦截均通过")


def _check_topic_panel(app) -> None:
    """话题推荐面板：候选表与选中逻辑，不联网、不调模型。"""
    from wrc.assistant.topic import ProbeResult, TagCandidate

    assert app.tag_tree.heading("tag")["text"] == "话题"
    probe = ProbeResult(
        tag="延迟退休", ok=True, posts=12, pages=2, median_engagement=8,
        latest_post_at="2026-10-20", samples=["样例一", "样例二"],
    )
    candidate = TagCandidate(tag="延迟退休", reason="政策主标签", probe=probe)
    app.tag_candidates = [candidate]
    app.tag_tree.insert(
        "", "end", iid="1",
        values=(candidate.tag, probe.describe(), candidate.reason),
    )
    app.tag_tree.selection_set("1")
    assert app._selected_candidate().tag == "延迟退休"
    # 量级描述必须是实测口径，不能出现模型猜测的数字
    assert "首 2 页共 12 条" in probe.describe()
    print("话题推荐：候选表、实测口径与选中逻辑均通过")


def _check_merged_flow(app) -> None:
    """合并后的主按钮：一次点击应走完 采集→清洗→分析→报告。"""
    app.start_button.cget("text")
    assert "采集" in app.start_button.cget("text")
    assert "报告" in app.start_button.cget("text")
    # 默认自动分析：这是上一轮真实运行暴露的问题（采完就停，没有报告）
    assert app.auto_analyze_var.get() is True
    assert app.do_sentiment_var.get() is True
    print("合并流程：主按钮一键到底，默认自动分析 = 开")


def _check_chart_picker(app) -> None:
    """图表选择：默认全选，取消后不应再画。"""
    from wrc.analyze.charts import CHART_SPECS

    keys = [key for key, _f, _l, _n in CHART_SPECS]
    assert set(app.chart_vars) == set(keys)
    assert app._chart_selection() == keys, "默认应当全选"

    app._set_charts(False)
    assert app._chart_selection() == []
    app._set_core_charts()
    core = set(app._chart_selection())
    assert core == {"sentiment_distribution", "stance_distribution",
                    "timeseries_evolution", "cleaning_comparison"}, core
    app._set_charts(True)
    print(f"图表选择：{len(keys)} 种可选，全选/全不选/核心四张均生效")


def main() -> int:
    try:
        from wrc.gui.app import App
    except Exception as exc:  # noqa: BLE001
        print(f"界面模块导入失败：{type(exc).__name__}: {exc}")
        return 2

    try:
        app = App()
    except Exception as exc:  # noqa: BLE001
        print(f"界面构建失败：{type(exc).__name__}: {exc}")
        return 2

    try:
        for _ in range(6):
            app.update()
        widgets = len(app.winfo_children())
        print(f"界面构建成功：顶层区域 {widgets} 个，标签 {app.tag_var.get()!r}")
        # 真实反馈：默认话题会让人以为已经选好；候选帖子数不能写死 20
        assert app.tag_var.get() == "", app.tag_var.get()
        assert app.post_limit_var.get() >= 20, app.post_limit_var.get()
        assert app.comments_var.get() >= 50, app.comments_var.get()
        assert app.estimate_label.cget("text"), "请求量估算提示不能为空"
        print(f"默认参数：候选帖 {app.post_limit_var.get()} 篇 × 每帖 {app.comments_var.get()} 条；"
              f"请求上限 {app.max_requests_var.get()}")
        print(f"帖子列表列：{[app.tree.heading(c)['text'] for c in app.tree['columns']]}")
        # 逐帖目标数：默认全部走全局值，界面不该伪造"用户已逐帖决策"
        opts = app._collect_options()
        assert opts.per_post_targets == {}, opts.per_post_targets
        # 列表为空时刷新目标列不能崩（改全局值会触发这条路径）
        app.per_post_targets["dummy"] = 1
        app._refresh_target_column()
        app.per_post_targets.clear()
        print(f"逐帖目标：默认 {app._global_target()} 条，未选帖子时覆盖表为空")

        _check_checkboxes(app)
        _check_empty_state(app)
        _check_window_parsing(app)
        _check_topic_panel(app)
        _check_merged_flow(app)
        _check_chart_picker(app)
    finally:
        app.destroy()
    print("界面冒烟通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
