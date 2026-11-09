"""博研助手 · 图形界面。

为什么用 Tkinter：
- 标准库自带，用户不需要额外装任何东西；
- 这个界面本质是「表单 + 日志窗口」，不需要复杂图表；
- 后续如果要打包成 exe，Tkinter 的体积和依赖问题最小。

两条工程约定：
1. **所有耗时操作都在后台线程**，主线程只负责画界面。否则窗口会假死。
2. **业务逻辑全在 `wrc.pipeline`**，这里只做参数收集与状态显示。
   界面和命令行不能各写一套流程，否则迟早出现"界面能跑、命令行报错"的分叉。
"""

from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .. import __version__
from ..clean.basic import PIPELINE_VERSION, TIER_META
from ..config import get_settings
from ..errors import ApiError, ConfigProblem, WrcError, render_error
from ..pipeline import (
    MAX_COMMENTS_PER_POST,
    CleanOptions,
    CollectOptions,
    ResearchPipeline,
    estimate_comment_requests,
)
from ..privacy.consent import confirm, render_statement
from ..weibo.models import WeiboPost
from ..weibo.window import parse_window_time, recent_days

ORDER_LABELS = {
    "按互动总量（本地重排）": "engagement",
    "按发帖时间（本地重排）": "time",
    "按接口返回顺序（平台）": "platform",
}

COMMENT_SORT_LABELS = {
    "热度优先（平台热门流）": "hot",
    "时间优先（仅网页端生效）": "time",
}

#: 复选框列的两种符号。用字符而不是真控件：Treeview 没有原生复选框，
#: 而为了勾选把列表换成一行行 Checkbutton 会在几百条候选时明显卡顿。
CHECKED_BOX = "☑"
UNCHECKED_BOX = "☐"

LOG_COLORS = {
    "info": "#1f2933",
    "ok": "#0b6b3a",
    "warn": "#8a5a00",
    "error": "#a11212",
    "muted": "#6b7280",
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"博研助手 · 微博舆论研究工具  v{__version__}")
        # 窗口必须一打开就能看全：真实反馈是"按钮和滚动条被折叠，要手动拉大窗口"。
        # 先按屏幕可用尺寸给一个足够大的初始窗口，再尽量最大化。
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(1560, int(screen_w * 0.94))
        height = min(1000, int(screen_h * 0.90))
        self.geometry(f"{width}x{height}+{max(0, (screen_w - width) // 2)}+{max(0, (screen_h - height) // 2)}")
        self.minsize(1100, 720)
        if os.name == "nt" and not os.environ.get("WRC_GUI_SMOKE"):
            try:
                self.state("zoomed")  # Windows 下最大化，保证所有分区一眼可见
            except tk.TclError:  # pragma: no cover - 少数窗口管理器不支持
                pass
        if os.environ.get("WRC_GUI_SMOKE"):
            # 自检模式：只验证控件能建起来，不真的弹窗口打扰人
            self.withdraw()

        self.settings = get_settings()
        self.settings.ensure_dirs()
        self.pipeline = ResearchPipeline(self.settings, emit=self._emit)

        self.posts: list[WeiboPost] = []
        # 勾选状态按 mid 存，不按行号：重新检索或刷新表格时不会串行
        self.checked: set[str] = set()
        # 逐帖评论目标覆盖：{post_mid: n}。没有覆盖的帖子用「每帖评论数」。
        self.per_post_targets: dict[str, int] = {}
        self.ai_status = None
        self.queue: queue.Queue = queue.Queue()
        self.job_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self._pending_done = None
        self._last_out_dir: Path | None = None
        # 允许直接对上一次采集做分析：重启界面后不必重新采一遍才能用分析功能
        try:
            self._last_run_id: str | None = self.pipeline.store.latest_run_id()
        except Exception:  # noqa: BLE001
            self._last_run_id = None

        self._setup_fonts()
        self._build_ui()
        self.after(120, self._poll_queue)
        self._refresh_ai_status_async()
        # 启动横幅必须带版本与时间。上一次实际问题就是"界面开着旧代码"：
        # 进程在 23:43 启动，代码在 00:51 改过，日志里看不出任何痕迹，
        # 于是把旧行为当成了新 bug。以后一眼能对出来。
        self._log(
            "info",
            f"版本 {__version__}｜清洗管线 {PIPELINE_VERSION}｜"
            f"启动于 {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
        self._log(
            "info",
            "若刚更新过代码，请关闭本窗口重新启动——程序不会热更新。",
        )
        self._log("info", "就绪。先点「检查登录态」，确认登录后再检索帖子。")

    # ------------------------------------------------------------------ #
    # 界面搭建
    # ------------------------------------------------------------------ #
    def _setup_fonts(self) -> None:
        try:
            from tkinter import font as tkfont

            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
                tkfont.nametofont(name).configure(family="Microsoft YaHei UI", size=10)
        except Exception:  # pragma: no cover - 字体缺失时用默认
            pass

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(14, 10, 14, 6))
        header.pack(fill="x")
        ttk.Label(
            header, text="博研助手 · 微博舆论研究工具",
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")
        self.login_label = ttk.Label(header, text="登录状态：未知", foreground="#6b7280")
        self.login_label.pack(side="right", padx=(10, 0))
        ttk.Button(header, text="检查登录态", command=self._on_check_login).pack(
            side="right", padx=6
        )
        ttk.Button(header, text="扫码登录", command=self._on_login).pack(side="right")

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self._build_form(left)
        self._build_log(right)
        self._build_footer()

    def _build_form(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0, width=620)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        # 滚轮：不绑的话只能拖右侧细滚动条，用户会以为"下面没有了"
        def _on_wheel(event):
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta * 3, "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)

        self._build_collect_section(body)
        self._build_post_section(body)
        self._build_clean_section(body)
        self._build_analysis_section(body)
        self._build_export_section(body)

    def _build_analysis_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent, text=" 4 · 分析与图表（会自动纳入下面的「一键完成」） ", padding=10
        )
        frame.pack(fill="x", padx=4, pady=(0, 8))

        self.do_sentiment_var = tk.BooleanVar(value=True)
        self.do_cluster_var = tk.BooleanVar(value=True)
        self.do_stance_var = tk.BooleanVar(value=True)

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Checkbutton(
            row, text="情感分析（双通道 + 信度抽样）",
            variable=self.do_sentiment_var,
        ).pack(side="left")
        ttk.Label(row, text="最多分析").pack(side="left", padx=(12, 4))
        # 默认 0 = 全量。实测一条评论的标注成本极低（200 条 ≈ 10 次调用），
        # 而抽样分析会让导出的 CSV 出现大量空 sentiment 列，
        # 后续聚类/交叉表只能基于被标注的那部分——为省几分钱牺牲数据完整性不划算。
        self.sentiment_limit_var = tk.IntVar(value=0)
        ttk.Spinbox(
            row, from_=0, to=5000, increment=50, width=6,
            textvariable=self.sentiment_limit_var,
        ).pack(side="left")
        ttk.Label(row, text="条（0 = 全量）", foreground="#6b7280").pack(side="left")

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            row2, text="聚类分析（话语聚类 + 用户社群）",
            variable=self.do_cluster_var,
        ).pack(side="left")
        ttk.Checkbutton(
            row2, text="立场分析（支持 / 反对 / 中立 / 无关）",
            variable=self.do_stance_var,
        ).pack(side="left", padx=(16, 0))

        row3 = ttk.Frame(frame)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Label(row3, text="事件日期").pack(side="left")
        # 留空 = 不做事件分析。事件日期必须由研究者给：
        # 工具不会用"评论量最大的那天"替代，那是数据特征不是事件。
        self.event_date_var = tk.StringVar(value="")
        ttk.Entry(row3, textvariable=self.event_date_var, width=12).pack(
            side="left", padx=(6, 4)
        )
        ttk.Label(row3, text="格式 2026-09-29，留空则不做").pack(side="left")
        ttk.Label(row3, text="前后窗口").pack(side="left", padx=(12, 4))
        self.event_window_var = tk.IntVar(value=7)
        ttk.Spinbox(
            row3, from_=1, to=60, increment=1, width=4,
            textvariable=self.event_window_var,
        ).pack(side="left")
        ttk.Label(row3, text="天").pack(side="left")

        # ---- 图表选择 ----
        chart_box = ttk.LabelFrame(frame, text=" 图表（可多选，决定报告里出现哪些图） ", padding=6)
        chart_box.pack(fill="x", pady=(8, 0))
        from ..analyze.charts import CHART_SPECS

        self.chart_vars: dict[str, tk.BooleanVar] = {}
        columns = 3
        grid = ttk.Frame(chart_box)
        grid.pack(fill="x")
        for index, (key, _filename, label, _needs) in enumerate(CHART_SPECS):
            var = tk.BooleanVar(value=True)
            self.chart_vars[key] = var
            ttk.Checkbutton(grid, text=label, variable=var).grid(
                row=index // columns, column=index % columns, sticky="w", padx=2, pady=1
            )
        chart_bar = ttk.Frame(chart_box)
        chart_bar.pack(fill="x", pady=(4, 0))
        ttk.Button(chart_bar, text="全选", command=lambda: self._set_charts(True)).pack(side="left")
        ttk.Button(chart_bar, text="全不选", command=lambda: self._set_charts(False)).pack(
            side="left", padx=6
        )
        ttk.Button(chart_bar, text="只留核心四张", command=self._set_core_charts).pack(side="left")
        ttk.Label(
            chart_bar,
            text="核心四张 = 情感分布 / 立场分布 / 时间演变 / 清洗影响",
            foreground="#6b7280",
        ).pack(side="left", padx=8)

        self.analyze_button = ttk.Button(
            frame, text="开始分析并生成图表与报告", command=self._on_analyze
        )
        self.analyze_button.pack(anchor="w", pady=(6, 0))
        # 合并：采集完成后自动接着分析。理由很实际——真实运行里研究者采完就停了，
        # 拿到的是"有数据没报告"的半成品，还得回头再点一次才知道要再点一次。
        self.auto_analyze_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="采集完成后自动跑上面勾选的分析并生成报告（一次完成）",
            variable=self.auto_analyze_var,
        ).pack(anchor="w", pady=(6, 0))
        self.analyze_hint = ttk.Label(
            frame,
            text="也可以先只采数据：取消上面的自动分析，之后用这个按钮单独分析。"
                 "报告与图表生成到导出目录的 report/ 子目录。",
            foreground="#6b7280",
        )
        self.analyze_hint.pack(fill="x", pady=(4, 0))

        compare_row = ttk.Frame(frame)
        compare_row.pack(fill="x", pady=(8, 0))
        self.compare_button = ttk.Button(
            compare_row, text="跨话题对比（最近几次采集）", command=self._on_compare
        )
        self.compare_button.pack(side="left")
        ttk.Label(
            compare_row,
            text="先判断这些采集能不能比，再出对比表",
            foreground="#6b7280",
        ).pack(side="left", padx=(8, 0))

        # ---- 方法助手 ----
        divider = ttk.Separator(frame, orient="horizontal")
        divider.pack(fill="x", pady=8)

        ask_row = ttk.Frame(frame)
        ask_row.pack(fill="x")
        ttk.Label(ask_row, text="问方法助手").pack(side="left")
        self.ask_var = tk.StringVar()
        entry = ttk.Entry(ask_row, textvariable=self.ask_var)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        entry.bind("<Return>", lambda _event: self._on_ask())
        ttk.Button(ask_row, text="提问", command=self._on_ask).pack(side="left")
        ttk.Button(
            ask_row, text="下一步建议", command=self._on_suggest
        ).pack(side="left", padx=(6, 0))

        ttk.Label(
            frame,
            text="助手结合当前数据状态回答方法问题，不替你做研究结论；"
                 "「下一步建议」不走大模型，是确定性的规则判断。",
            foreground="#6b7280",
        ).pack(fill="x", pady=(4, 0))

    def _build_collect_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text=" 1 · 采集 ", padding=10)
        frame.pack(fill="x", padx=4, pady=(4, 8))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="话题标签").pack(side="left")
        # 留空：默认填一个话题会让人误以为已经选好了
        self.tag_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.tag_var, width=26).pack(
            side="left", padx=(8, 12)
        )
        ttk.Label(row, text="候选帖子数").pack(side="left")
        self.post_limit_var = tk.IntVar(value=100)
        ttk.Spinbox(
            row, from_=20, to=500, increment=20, width=6,
            textvariable=self.post_limit_var,
            command=self._update_estimate,
        ).pack(side="left", padx=(6, 12))
        ttk.Label(row, text="排序").pack(side="left")
        self.order_var = tk.StringVar(value="按互动总量（本地重排）")
        ttk.Combobox(
            row, textvariable=self.order_var, values=list(ORDER_LABELS),
            state="readonly", width=22,
        ).pack(side="left", padx=(8, 0))

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="检索方式").pack(side="left")
        self.search_mode_var = tk.StringVar(value="topic")
        ttk.Radiobutton(
            row2, text="话题聚合", value="topic", variable=self.search_mode_var,
        ).pack(side="left", padx=(8, 2))
        ttk.Radiobutton(
            row2, text="关键词全文（会混入仅提及的帖子）", value="keyword",
            variable=self.search_mode_var,
        ).pack(side="left")

        row3 = ttk.Frame(frame)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Label(row3, text="每帖评论数").pack(side="left")
        self.comments_var = tk.IntVar(value=100)
        ttk.Spinbox(
            row3, from_=10, to=2000, increment=50, width=7,
            textvariable=self.comments_var,
            command=self._update_estimate,
        ).pack(side="left", padx=(8, 16))
        # 改全局值时，帖子列表里"本次目标"那一列要跟着变
        self.comments_var.trace_add("write", lambda *_: self._refresh_target_column())
        self.comments_var.trace_add("write", lambda *_: self._update_estimate())

        ttk.Label(row3, text="评论排序").pack(side="left")
        self.comment_sort_var = tk.StringVar(value="热度优先（平台热门流）")
        ttk.Combobox(
            row3, textvariable=self.comment_sort_var,
            values=list(COMMENT_SORT_LABELS), state="readonly", width=24,
        ).pack(side="left", padx=(8, 16))

        self.subcomments_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row3, text="同时采集二级评论（楼中楼，请求量会翻几倍）",
            variable=self.subcomments_var,
        ).pack(side="left")
        ttk.Label(row3, text="每条最多").pack(side="left", padx=(8, 2))
        self.subcomment_target_var = tk.IntVar(value=10)
        ttk.Spinbox(
            row3, from_=1, to=100, increment=5, width=5,
            textvariable=self.subcomment_target_var,
        ).pack(side="left")
        ttk.Label(row3, text="条", foreground="#6b7280").pack(side="left", padx=(2, 0))

        # ---- 时间窗 ----
        row_win = ttk.Frame(frame)
        row_win.pack(fill="x", pady=(8, 0))
        ttk.Label(row_win, text="评论时间窗").pack(side="left")
        self.since_var = tk.StringVar(value="")
        self.until_var = tk.StringVar(value="")
        ttk.Entry(row_win, textvariable=self.since_var, width=17).pack(
            side="left", padx=(8, 2)
        )
        ttk.Label(row_win, text="至").pack(side="left")
        ttk.Entry(row_win, textvariable=self.until_var, width=17).pack(
            side="left", padx=(2, 8)
        )
        ttk.Button(
            row_win, text="近 7 天", width=8,
            command=lambda: self._quick_window(7),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            row_win, text="近 30 天", width=8,
            command=lambda: self._quick_window(30),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            row_win, text="不限", width=6, command=self._clear_window
        ).pack(side="left")

        ttk.Label(
            frame,
            text=(
                "时间窗说明：填了窗口后，「每帖评论数」指的是**窗口内**要凑够多少条——"
                "采满即停。格式 2026-09-01 或 2026-09-01 14:30；"
                "只填日期时，起点算当天 00:00、终点算当天 23:59。"
            ),
            foreground="#6b7280",
            wraplength=980,
            justify="left",
        ).pack(fill="x", pady=(6, 0))
        ttk.Label(
            frame,
            text=(
                "排序说明：微博移动端评论接口不认排序参数，选「时间优先」会被平台忽略，"
                "日志与报告会如实标注「排序未生效」；能真正按时间取到数据的只有网页端后端。"
            ),
            foreground="#6b7280",
            wraplength=980,
            justify="left",
        ).pack(fill="x", pady=(2, 0))

        row3 = ttk.Frame(frame)
        row3.pack(fill="x", pady=(10, 0))
        ttk.Label(row3, text="本次请求上限").pack(side="left")
        self.max_requests_var = tk.IntVar(value=400)
        scale = ttk.Scale(
            row3, from_=20, to=3000, orient="horizontal", length=320,
            command=lambda value: self.max_requests_var.set(int(float(value))),
        )
        scale.set(400)
        scale.pack(side="left", padx=(8, 8))
        self.max_requests_label = ttk.Label(row3, text="400 次")
        self.max_requests_label.pack(side="left")
        self.max_requests_var.trace_add(
            "write",
            lambda *_: self.max_requests_label.configure(
                text=f"{self.max_requests_var.get()} 次"
            ),
        )
        ttk.Label(
            frame,
            text="提示：请求之间会随机等待 1.5–3 秒。上限越高，连续请求时间越长，被风控的概率越大。",
            foreground="#6b7280",
        ).pack(fill="x", pady=(6, 0))
        # 实时估算：样本期望值（帖子数 × 每帖条数）和请求上限必须一起看，
        # 否则用户会设一个"大样本"，然后被上限悄悄截断。
        self.estimate_label = ttk.Label(
            frame, text="", foreground="#8a5a00", wraplength=980, justify="left"
        )
        self.estimate_label.pack(fill="x", pady=(2, 0))
        self._update_estimate()

        ttk.Button(frame, text="检索帖子（第一步）", command=self._on_search).pack(
            anchor="w", pady=(10, 0)
        )

        self._build_topic_section(frame)

    # ------------------------------------------------------------------ #
    # 话题推荐（需求 E2：量级必须实测，禁止模型猜测）
    # ------------------------------------------------------------------ #
    def _build_topic_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text=" 不知道用哪个话题？让助手提几个（量级为实测值） ",
                             padding=8)
        box.pack(fill="x", pady=(10, 0))

        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="描述事件/人物/议题").pack(side="left")
        self.topic_desc_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.topic_desc_var, width=40).pack(
            side="left", padx=(8, 8)
        )
        self.topic_button = ttk.Button(
            row, text="推荐话题", command=self._on_recommend_tags
        )
        self.topic_button.pack(side="left")

        self.tag_tree = ttk.Treeview(
            box, columns=("tag", "level", "reason"), show="headings", height=4
        )
        for key, label, width in (
            ("tag", "话题", 130),
            ("level", "实测量级", 150),
            ("reason", "推荐理由", 420),
        ):
            self.tag_tree.heading(key, text=label)
            self.tag_tree.column(key, width=width, anchor="w")
        self.tag_tree.pack(fill="x", pady=(6, 0))
        self.tag_candidates: list = []

        bar = ttk.Frame(box)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(
            bar, text="用选中话题检索", command=self._on_use_tag
        ).pack(side="left")
        ttk.Button(
            bar, text="查看样例帖子", command=self._on_sample_posts
        ).pack(side="left", padx=6)
        ttk.Label(
            bar,
            text="量级是工具真搜出来的，不是模型猜的；先看样例再决定，别直接拿一个词就开跑。",
            foreground="#6b7280",
        ).pack(side="left", padx=8)

    def _build_post_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text=" 2 · 选择要采集评论的帖子 ", padding=10)
        frame.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        columns = ("pick", "idx", "time", "engage", "comments", "target", "text")
        self.tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=8, selectmode="browse"
        )
        headings = {
            "pick": ("选", 38),
            "idx": ("#", 42),
            "time": ("发帖时间", 130),
            "engage": ("互动", 70),
            "comments": ("评论数", 70),
            "target": ("本次目标", 80),
            "text": ("内容摘要", 250),
        }
        for key, (label, width) in headings.items():
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="w" if key == "text" else "center")
        self.tree.pack(fill="both", expand=True)
        # 空状态提示：第一次打开时列表是空的，什么都不说会让人以为坏了
        self.post_hint = ttk.Label(
            frame,
            text=(
                "还没有候选帖子：① 在上方填话题 → ② 点「检索帖子（第一步）」"
                " → ③ 在这里勾选要采评论的帖子"
            ),
            foreground="#8a5a00",
            wraplength=980,
            justify="left",
        )
        self.post_hint.pack(fill="x", pady=(4, 0))
        # 点「选」这一列 = 切换勾选；点其他列只移动高亮行
        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<space>", self._toggle_checked_current)

        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="全选", command=self._select_all).pack(side="left")
        ttk.Button(bar, text="全不选", command=self._select_none).pack(
            side="left", padx=6
        )
        ttk.Button(bar, text="反选", command=self._invert_selection).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(
            bar, text="设置勾选帖子的目标数…", command=self._on_set_target
        ).pack(side="left", padx=(12, 6))
        ttk.Button(
            bar, text="恢复为全局值", command=self._on_reset_target
        ).pack(side="left")
        self.pick_label = ttk.Label(bar, text="已勾选 0 篇", foreground="#6b7280")
        self.pick_label.pack(side="left", padx=10)

        ttk.Label(
            frame,
            text=(
                "点「选」列的方框或按空格勾选；只有勾选的帖子会被采集。"
                "评论多的帖子适合做话语分析，评论少的适合做个案。"
            ),
            foreground="#6b7280",
            wraplength=980,
            justify="left",
        ).pack(fill="x", pady=(4, 0))
        ttk.Label(
            frame,
            text=(
                "为什么允许逐帖改目标数：全局定值会让小帖子过度采集、大帖子严重不足，"
                "那是抽样偏差。这里改的只是「采多少」，抽样理由仍要你自己在论文里写。"
            ),
            foreground="#6b7280",
            wraplength=980,
            justify="left",
        ).pack(fill="x", pady=(2, 0))

    def _build_clean_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text=" 3 · 数据清洗 ", padding=10)
        frame.pack(fill="x", padx=4, pady=(0, 8))

        self.tier_var = tk.StringVar(value="moderate")
        for key, meta in TIER_META.items():
            row = ttk.Frame(frame)
            row.pack(fill="x", anchor="w")
            ttk.Radiobutton(
                row, text=str(meta["label"]), value=key, variable=self.tier_var,
                command=self._update_tier_hint,
            ).pack(side="left")
            ttk.Label(row, text=str(meta["summary"]), foreground="#6b7280").pack(
                side="left", padx=(6, 0)
            )

        self.dedupe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="启用严格去重（完全相同的内容跨帖只保留一条；研究水军模板时请取消勾选）",
            variable=self.dedupe_var,
        ).pack(fill="x", pady=(8, 0))

        self.bots_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="过滤疑似僵尸号（粉丝数 < 5；只能拿到粉丝数，可能误伤真实的低互动用户）",
            variable=self.bots_var,
        ).pack(fill="x")

        ai_row = ttk.Frame(frame)
        ai_row.pack(fill="x", pady=(8, 0))
        self.ai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ai_row, text="启用 AI 模型（重度档需要；不需要 torch，纯 ONNX）",
            variable=self.ai_var,
        ).pack(side="left")
        self.ai_label = ttk.Label(ai_row, text="AI 状态：检测中 …", foreground="#6b7280")
        self.ai_label.pack(side="left", padx=(10, 0))
        ttk.Button(ai_row, text="重新检测", command=self._refresh_ai_status_async).pack(
            side="left", padx=6
        )
        ttk.Button(ai_row, text="打开模型目录", command=self._open_model_dir).pack(
            side="left"
        )

        ttk.Label(
            frame,
            text="AI 判定会记录「与哪条模板相似、相似度多少」；不启用时用规则表，判定同样可逐条复核。",
            foreground="#6b7280",
        ).pack(fill="x", pady=(6, 0))

    def _build_export_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text=" 5 · 导出与声明 ", padding=10)
        frame.pack(fill="x", padx=4, pady=(0, 8))

        self.consent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="我已阅读并同意《数据使用声明》（导出前必须确认）",
            variable=self.consent_var,
        ).pack(fill="x")
        ttk.Button(frame, text="查看声明全文", command=self._show_consent).pack(
            anchor="w", pady=(6, 0)
        )
        ttk.Label(
            frame,
            text=f"导出目录：{self.settings.export_dir}",
            foreground="#6b7280",
        ).pack(fill="x", pady=(6, 0))

    def _build_log(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text=" 运行日志 ", padding=8)
        frame.pack(fill="both", expand=True, padx=(6, 0))
        self.log = tk.Text(frame, wrap="word", height=30, relief="flat", borderwidth=0)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for level, color in LOG_COLORS.items():
            self.log.tag_configure(level, foreground=color)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self, padding=(14, 6, 14, 12))
        footer.pack(fill="x")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=220)
        self.progress.pack(side="left")
        self.status_label = ttk.Label(footer, text="就绪", foreground="#6b7280")
        self.status_label.pack(side="left", padx=10)
        self.start_button = ttk.Button(
            footer, text="一键完成：采集 → 清洗 → 分析 → 报告", command=self._on_start
        )
        self.start_button.pack(side="right")
        self.stop_button = ttk.Button(
            footer, text="停止采集", command=self._on_stop, state="disabled"
        )
        self.stop_button.pack(side="right", padx=6)

    # ------------------------------------------------------------------ #
    # 日志与线程
    # ------------------------------------------------------------------ #
    def _emit(self, level: str, message: str) -> None:
        """pipeline 的进度回调（可能来自后台线程）。"""
        self.queue.put(("log", level, message))

    def _log(self, level: str, message: str) -> None:
        """写一行日志。每行带时间戳——出问题时能直接和文件修改时间对账。"""
        tag = level if level in LOG_COLORS else "info"
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        if message:
            self.log.insert("end", f"{stamp}  {message}\n", tag)
        else:
            self.log.insert("end", "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[1], item[2])
                elif kind == "done":
                    self._finish_job()
                    if self._pending_done is not None:
                        callback, self._pending_done = self._pending_done, None
                        callback(item[1])
                elif kind == "error":
                    self._finish_job()
                    self._pending_done = None
                    self._log("error", "")
                    self._log("error", item[1])
                    messagebox.showerror("出错了", item[1].splitlines()[0])
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _start_job(self, func, on_done) -> bool:
        if self.job_thread is not None:
            self._log("warn", "上一个操作还没结束，请等它完成。")
            return False
        self.stop_event.clear()
        self._pending_done = on_done
        self.progress.start(12)
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="运行中 …")

        def runner() -> None:
            try:
                result = func()
                self.queue.put(("done", result))
            except WrcError as exc:
                self.queue.put(("error", render_error(exc)))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("error", render_error(exc)))

        self.job_thread = threading.Thread(target=runner, daemon=True)
        self.job_thread.start()
        return True

    def _finish_job(self) -> None:
        self.job_thread = None
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="就绪")

    # ------------------------------------------------------------------ #
    # 各项操作
    # ------------------------------------------------------------------ #
    def _collect_options(self) -> CollectOptions:
        try:
            comments = int(self.comments_var.get())
        except (tk.TclError, ValueError):
            comments = 100
        try:
            limit = int(self.max_requests_var.get())
        except (tk.TclError, ValueError):
            limit = 400
        try:
            posts = int(self.post_limit_var.get())
        except (tk.TclError, ValueError, AttributeError):
            posts = 100
        try:
            since, until = self._window_from_ui()
        except ConfigProblem:
            # 检索阶段不需要时间窗；解析失败留给"开始采集"时再报，避免检索被挡住
            since, until = None, None
        return CollectOptions(
            tag=self.tag_var.get().strip(),
            order=ORDER_LABELS[self.order_var.get()],
            post_limit=max(20, min(500, posts)),
            search_mode=self.search_mode_var.get(),
            comments_per_post=max(1, min(MAX_COMMENTS_PER_POST, comments)),
            per_post_targets=self._targets_for_selected(),
            include_subcomments=bool(self.subcomments_var.get()),
            subcomment_target=max(1, min(100, int(self.subcomment_target_var.get() or 10))),
            comment_sort=COMMENT_SORT_LABELS[self.comment_sort_var.get()],
            max_requests=max(20, min(3000, limit)),
            since=since,
            until=until,
        )

    def _clean_options(self) -> CleanOptions:
        return CleanOptions(
            tier=self.tier_var.get(),
            dedupe_scope="global" if self.dedupe_var.get() else "off",
            use_ai=bool(self.ai_var.get()),
            filter_bots=bool(self.bots_var.get()),
        )

    def _on_check_login(self) -> None:
        def job():
            from ..session import BrowserSession

            async def flow():
                async with BrowserSession(self.settings, notify=lambda m: None) as session:
                    state = await session.describe_state()
                    return await session.check_login(), state["engine"], state["cookie_count"]

            return asyncio.run(flow())

        def done(result):
            logged, engine, cookies = result
            if logged:
                self.login_label.configure(text="登录状态：已登录", foreground="#0b6b3a")
                self._log("ok", f"登录态有效（引擎 {engine}，Cookie {cookies} 个）")
            else:
                self.login_label.configure(text="登录状态：未登录", foreground="#a11212")
                self._log("warn", "尚未登录，请点「扫码登录」。")

        self._log("info", "正在检查登录态 …")
        self._start_job(job, done)

    def _on_login(self) -> None:
        def job():
            from ..session import BrowserSession

            async def flow():
                async with BrowserSession(
                    self.settings, notify=lambda m: self.queue.put(("log", "info", m))
                ) as session:
                    return await session.login_interactive()

            return asyncio.run(flow())

        def done(ok):
            if ok:
                self.login_label.configure(text="登录状态：已登录", foreground="#0b6b3a")
                self._log("ok", "登录成功，登录态已保存。")
            else:
                self._log("warn", "登录没有完成。可以重试，或用命令行 --cookie 方式。")

        self._log("info", "即将打开浏览器窗口，请在窗口内扫码登录（最长 5 分钟）。")
        self._start_job(job, done)

    def _update_post_hint(self) -> None:
        label = getattr(self, "post_hint", None)
        if label is None:
            return
        if self.posts:
            label.pack_forget()
        else:
            label.configure(
                text=(
                    "没有候选帖子：换个说法再检索，或在上方用「推荐话题」让助手给几个候选。"
                    "登录态失效也会导致检索为空——可先点「检查登录态」。"
                )
            )
            label.pack(fill="x", pady=(4, 0))

    def _update_estimate(self) -> None:
        """按当前设置估算请求量与可用样本上限，并提醒矛盾。"""
        label = getattr(self, "estimate_label", None)
        if label is None:
            return
        try:
            posts = max(1, int(self.post_limit_var.get()))
        except (tk.TclError, ValueError):
            posts = 100
        try:
            per_post = max(1, int(self.comments_var.get()))
        except (tk.TclError, ValueError):
            per_post = 100
        try:
            cap = max(10, int(self.max_requests_var.get()))
        except (tk.TclError, ValueError):
            cap = 120
        estimated = estimate_comment_requests([per_post] * posts)
        want = posts * per_post
        text = (
            f"当前设置：候选 {posts} 篇 × 每帖 {per_post} 条 ≈ 期望样本 {want:,} 条；"
            f"约需 {estimated:,} 次请求，上限 {cap} 次"
        )
        if estimated > cap:
            achievable = int(cap * want / estimated)
            text += (
                f" —— ⚠ 请求数不够，实际大约只能采到 {achievable:,} 条；"
                "请提高上限，或减少候选帖子数/每帖条数。"
            )
        label.configure(text=text)

    def _on_search(self) -> None:
        opts = self._collect_options()
        if not opts.tag.strip("# "):
            messagebox.showwarning("缺少参数", "请先填写话题标签。")
            return
        def job():
            return asyncio.run(self.pipeline.preview_posts(opts))

        def done(posts):
            self.posts = list(posts)
            # 换了一批帖子，旧的勾选与逐帖目标已经没有对应对象
            self.checked.clear()
            self.per_post_targets.clear()
            self.tree.delete(*self.tree.get_children())
            for index in range(1, len(self.posts) + 1):
                self.tree.insert("", "end", iid=str(index), values=("",))
                self._render_row(index)
            self._update_post_hint()
            self._log(
                "ok",
                f"检索完成，共 {len(self.posts)} 篇帖子。勾选要采集评论的帖子（点「选」列）。",
            )

        self._log("info", f"正在检索 {opts.normalised_tag()} …")
        self._start_job(job, done)

    # ---- 话题推荐 ----
    def _selected_candidate(self):
        selection = self.tag_tree.selection()
        if not selection:
            return None
        try:
            index = int(selection[0])
        except ValueError:
            return None
        if 1 <= index <= len(self.tag_candidates):
            return self.tag_candidates[index - 1]
        return None

    def _on_recommend_tags(self) -> None:
        description = self.topic_desc_var.get().strip()
        if len(description) < 2:
            messagebox.showwarning(
                "先描述一下", "写一句你想研究什么（例如「2026 年延迟退休实施后的舆论」）。"
            )
            return

        def job():
            from ..assistant.topic import recommend_tags
            from ..llm.client import LlmClient

            client = LlmClient()
            ok, reason = client.available()
            if not ok:
                raise ConfigProblem(
                    what_happened="话题推荐需要大模型，但现在不可用。",
                    detail=f"{reason}\n可以先手动在「话题标签」里填一个词继续做研究。",
                )
            return asyncio.run(
                recommend_tags(
                    description, client=client, settings=self.settings,
                    emit=lambda level, message: self._log(level, message),
                )
            )

        def done(candidates):
            self.tag_candidates = list(candidates)
            self.tag_tree.delete(*self.tag_tree.get_children())
            for index, candidate in enumerate(self.tag_candidates, start=1):
                probe = candidate.probe
                level = probe.describe() if probe is not None else "未探测"
                self.tag_tree.insert(
                    "", "end", iid=str(index),
                    values=(candidate.tag, level, candidate.reason),
                )
            if not candidates:
                self._log("warn", "模型没给出可用候选，可以换个说法再试，或手填话题。")
                return
            self._log(
                "ok",
                f"候选 {len(candidates)} 个。先「查看样例帖子」确认内容再「用选中话题检索」。"
            )

        self._log("info", "正在让助手提候选话题（每个候选都会真搜一次量级）…")
        self._start_job(job, done)

    def _on_use_tag(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            messagebox.showwarning("没有选中", "请先在候选列表里选中一个话题。")
            return
        self.tag_var.set(candidate.tag)
        self._log("info", f"话题已设为「{candidate.tag}」，正在检索帖子 …")
        self._on_search()

    def _on_sample_posts(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            messagebox.showwarning("没有选中", "请先在候选列表里选中一个话题。")
            return
        probe = candidate.probe
        if probe is not None and probe.samples:
            body = "\n".join(f"· {line}" for line in probe.samples)
            messagebox.showinfo(
                f"「{candidate.tag}」的样例帖子",
                f"{probe.describe()}\n\n{body}\n\n"
                "这些是探测时真实检索到的帖子摘要。要采评论请点「用选中话题检索」。",
            )
            return

        # 探测阶段没留样例（或探测失败）：现场再查一次，而不是拿模型的话搪塞
        def job():
            return asyncio.run(
                self.pipeline.preview_posts(
                    CollectOptions(tag=candidate.tag, post_limit=5)
                )
            )

        def done(posts):
            if not posts:
                messagebox.showwarning(
                    "没有帖子", f"「{candidate.tag}」下没有检索到帖子。"
                )
                return
            body = "\n".join(
                f"· {post.preview(46)}（互动 {post.engagement:,}）" for post in posts[:5]
            )
            messagebox.showinfo(
                f"「{candidate.tag}」的样例帖子", f"实时检索到 {len(posts)} 篇：\n\n{body}"
            )

        self._log("info", f"正在为「{candidate.tag}」拉取样例帖子 …")
        self._start_job(job, done)

    def _on_start(self) -> None:
        selected = self._checked_posts()
        if not selected:
            messagebox.showwarning(
                "没有勾选帖子",
                "请先在列表里勾选至少一篇帖子（点「选」列的方框）。",
            )
            return
        if not self.consent_var.get():
            messagebox.showwarning("需要确认声明", "请先勾选《数据使用声明》。")
            return
        # 时间窗格式错误必须在这里明确报出来，不能悄悄当成"不限时间"采一轮
        try:
            self._window_from_ui()
        except ConfigProblem as exc:
            messagebox.showwarning("时间窗有问题", render_error(exc))
            return

        clean_opts = self._clean_options()
        if clean_opts.use_ai:
            if self.ai_status is None or not self.ai_status.ready:
                messagebox.showwarning(
                    "AI 尚未就绪",
                    "勾选了 AI 但模型还没准备好。\n\n"
                    "可以点「重新检测」查看缺什么，或取消勾选 AI 用规则表清洗。",
                )
                return
        elif clean_opts.tier == "heavy":
            messagebox.showwarning(
                "重度档需要 AI",
                "重度档包含 AI 判定。请勾选「启用 AI 模型」，或改用中度档。",
            )
            return

        opts = self._collect_options()
        self._log("info", "=" * 40)
        self._log("info", f"开始采集 {len(selected)} 篇帖子的评论 …")

        def job():
            run_id, comments, used = asyncio.run(
                self.pipeline.collect(opts, selected, should_stop=self.stop_event.is_set)
            )
            self.queue.put(("log", "ok", f"采集完成：{comments} 条评论，{used} 次请求。"))
            if comments == 0:
                raise ApiError(
                    what_happened="没有采到任何评论，流程已停止。",
                    detail="所有评论后端都返回空；请先在命令行运行 python -m wrc.cli doctor 诊断。",
                )
            report, ai_summary = self.pipeline.clean(run_id, clean_opts)
            self.queue.put(("log", "info", ""))
            # 整份报告作为一条消息发出：逐行加时间戳会把表格搅乱
            self.queue.put(("log", "info", report.render()))
            consent = confirm(True, source="gui")
            manifest = self.pipeline.export(
                run_id,
                report=report,
                clean_opts=clean_opts,
                collect_opts=opts,
                consent=consent,
                ai_summary=ai_summary,
                requests_used=used,
            )
            out_dir = manifest.out_dir.parent
            self.pipeline.write_research_log(run_id, out_dir)
            appendix = None
            try:
                appendix = self.pipeline.write_appendix(run_id, out_dir)
            except Exception as exc:  # noqa: BLE001
                self.queue.put(
                    ("log", "warn", f"可复现附录生成失败：{type(exc).__name__}: {exc}")
                )
            return {
                "run_id": run_id,
                "comments": comments,
                "kept": report.total_kept,
                "rate": report.retention_rate,
                "out_dir": out_dir,
                "appendix": appendix,
            }

        def done(result):
            self._last_out_dir = result["out_dir"]
            self._last_run_id = result["run_id"]
            self._log("ok", "=" * 40)
            self._log("ok", f"全部完成：会话 {result['run_id']}")
            self._log(
                "ok",
                f"原始 {result['comments']} 条，清洗后保留 {result['kept']} 条"
                f"（{result['rate']:.1%}）",
            )
            self._log("ok", f"导出目录：{result['out_dir']}")
            if result.get("appendix"):
                self._log("info", f"可复现附录：{result['appendix']}")
            wants_analysis = bool(self.auto_analyze_var.get()) and any((
                self.do_sentiment_var.get(),
                self.do_cluster_var.get(),
                self.do_stance_var.get(),
            ))
            if wants_analysis:
                self._log("info", "=" * 40)
                self._log("info", "采集与导出完成，接着自动分析并生成报告 …")
                # 采集已经落盘，分析失败不会毁掉数据；这里只把控制权交给分析流程
                self._on_analyze()
                return
            if messagebox.askyesno(
                "完成", f"数据已导出到：\n{result['out_dir']}\n\n现在打开这个目录吗？"
            ):
                self._open_path(result["out_dir"])

        self._start_job(job, done)

    # ---- 图表选择 ----
    def _set_charts(self, value: bool) -> None:
        for var in self.chart_vars.values():
            var.set(value)

    def _set_core_charts(self) -> None:
        core = {"sentiment_distribution", "stance_distribution",
                "timeseries_evolution", "cleaning_comparison"}
        for key, var in self.chart_vars.items():
            var.set(key in core)

    def _chart_selection(self) -> list[str]:
        return [key for key, var in self.chart_vars.items() if var.get()]

    # ------------------------------------------------------------------ #
    # 分析
    # ------------------------------------------------------------------ #
    def _on_analyze(self) -> None:
        run_id = self._last_run_id
        if not run_id:
            messagebox.showwarning(
                "还没有数据",
                "请先完成一次采集（点「开始采集并分析」）。\n"
                "分析基于该次采集里已清洗的数据。",
            )
            return
        if not (
            self.do_sentiment_var.get()
            or self.do_cluster_var.get()
            or self.do_stance_var.get()
        ):
            messagebox.showwarning("没有选择", "请至少勾选一项分析。")
            return

        do_sentiment = bool(self.do_sentiment_var.get())
        do_cluster = bool(self.do_cluster_var.get())
        do_stance = bool(self.do_stance_var.get())

        event_specs = None
        raw_event = (self.event_date_var.get() or "").strip()
        if raw_event:
            try:
                from ..analyze.events import EventSpec

                window = int(self.event_window_var.get())
                event_specs = [EventSpec(
                    date=raw_event, label="", pre_days=window, post_days=window,
                )]
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning(
                    "事件日期格式不对",
                    f"请按 2026-09-29 的格式填：{exc}",
                )
                return
        try:
            limit = int(self.sentiment_limit_var.get())
        except (tk.TclError, ValueError):
            limit = 0
        # 0 或负数 = 全量（None 表示不截断）
        limit = limit if limit > 0 else None
        settings = self.settings
        include_charts = self._chart_selection()
        if not include_charts:
            self._log("warn", "你一张图都没选：分析照跑，但报告里不会有图。")

        self._log("info", "=" * 40)
        self._log("info", f"开始分析：会话 {run_id}（图表 {len(include_charts)} 种）")

        def emit(level: str, message: str) -> None:
            self.queue.put(("log", level, message))

        def job():
            from ..analyze import charts as chart_module
            from ..analyze.report import build_analysis_report, write_report
            from ..analyze.service import (
                run_community, run_discourse, run_sentiment,
            )
            from ..llm.client import LlmClient
            from ..store import Store

            store = Store(settings.db_path)
            store.init_schema()
            rows = store.load_analysis_rows(run_id)
            if not rows:
                raise ApiError(
                    what_happened="这次采集还没有清洗后的数据，无法分析。",
                )
            run_row = store.get_run(run_id) or {}
            run_info = {
                "run_id": run_id,
                "tag": run_row.get("tag", ""),
                "created_at": run_row.get("created_at", ""),
            }
            all_rows = store.load_analysis_rows(run_id, only_kept=False)
            cleaning = {
                "tier": rows[0].get("clean_tier", "未记录"),
                "pipeline_version": rows[0].get("pipeline_version", "未记录"),
                "total_input": len(all_rows),
                "total_kept": len(rows),
                "retention_rate": len(rows) / len(all_rows) if all_rows else 0.0,
                "steps": [],
            }

            sentiment_result = None
            stance_result = None
            discourse_result = None
            community_result = None

            if do_sentiment:
                client = LlmClient()
                ok, reason = client.available()
                if not ok:
                    emit("warn", f"跳过情感分析：LLM 不可用（{reason}）")
                else:
                    from ..ai.status import detect_sentiment_status
                    from ..analyze.classifier import OnnxSequenceClassifier

                    classifier = None
                    status = detect_sentiment_status(settings)
                    if status.ready:
                        classifier = OnnxSequenceClassifier(
                            status.onnx_path, status.tokenizer_path,
                            model_name=status.model_dir.name,
                        )
                        emit("info", f"本地对照模型：{status.model_dir.name}")
                    else:
                        emit("info", f"本地对照模型未就绪（{status.summary()}），只跑 LLM 通道")
                    sentiment_result, _ = run_sentiment(
                        run_id, store=store, settings=settings,
                        llm_client=client, local_classifier=classifier,
                        limit=limit, emit=emit,
                    )

            if do_stance:
                stance_client = LlmClient()
                ok, reason = stance_client.available()
                if not ok:
                    emit("warn", f"跳过立场分析：LLM 不可用（{reason}）")
                else:
                    try:
                        from ..analyze.service import run_stance

                        stance_result, _ = run_stance(
                            run_id, store=store, settings=settings,
                            llm_client=stance_client,
                            topic=run_info.get("tag") or None,
                            emit=emit,
                        )
                    except Exception as exc:  # noqa: BLE001
                        emit("warn", f"立场分析跳过：{type(exc).__name__}: {exc}")

            if do_cluster:
                try:
                    discourse_result, _ = run_discourse(
                        run_id, store=store, settings=settings, emit=emit
                    )
                except Exception as exc:  # noqa: BLE001
                    emit("warn", f"话语聚类跳过：{type(exc).__name__}: {exc}")
                try:
                    community_result, _ = run_community(
                        run_id, store=store, settings=settings
                    )
                except Exception as exc:  # noqa: BLE001
                    emit("warn", f"用户社群聚类跳过：{type(exc).__name__}: {exc}")

            # 回复结构：二级评论没采时这一步会返回空结果（并说明原因），不算失败。
            replies_result = None
            try:
                from ..analyze.service import run_replies

                replies_result, _ = run_replies(
                    run_id, store=store, settings=settings, emit=emit
                )
            except Exception as exc:  # noqa: BLE001
                emit("warn", f"回复结构跳过：{type(exc).__name__}: {exc}")

            # 时间演变：不额外调模型，只把已有标签按时间重排，所以总是跑。
            timeseries_result = None
            try:
                from ..analyze.service import run_timeseries

                timeseries_result, _ = run_timeseries(
                    run_id, store=store, settings=settings, emit=emit
                )
            except Exception as exc:  # noqa: BLE001
                emit("warn", f"时间演变跳过：{type(exc).__name__}: {exc}")

            # 交叉表：话语簇 × 情感倾向。两者都有才做，缺一个就整节省略，
            # 不拿空表糊弄读者。小群体抑制在这一层完成，导出的表格天然合规。
            crosstab_result = None
            if sentiment_result is not None and discourse_result is not None:
                from ..analyze.crosstab import discourse_by_sentiment

                assignments = {
                    str(item["cid"]): int(item["cluster"])
                    for item in discourse_result.to_dict()["assignments"]
                    if item.get("cluster") is not None
                }
                if assignments:
                    crosstab_result = discourse_by_sentiment(rows, assignments)

            out_dir = settings.export_dir / run_id / "report"

            # 立场 × 情感 / 簇 × 立场：两个维度都有结果才有意义，缺一个就整块省略
            try:
                from ..analyze.service import (
                    build_discourse_stance_crosstab, build_stance_crosstab,
                )

                stance_table = build_stance_crosstab(store, run_id)
                cluster_stance_table = build_discourse_stance_crosstab(store, run_id)
            except Exception as exc:  # noqa: BLE001
                stance_table = None
                cluster_stance_table = None
                emit("warn", f"交叉表跳过：{type(exc).__name__}: {exc}")

            emit("info", "生成图表 …")
            # 事件前后对比（有事件日期才做）
            try:
                from ..analyze.service import run_event_study

                event_results, _events = run_event_study(
                    store, run_id, event_specs
                )
            except Exception as exc:  # noqa: BLE001
                event_results = []
                emit("warn", f"事件前后对比跳过：{type(exc).__name__}: {exc}")

            generated = chart_module.render_all(
                out_dir,
                sentiment=sentiment_result,
                stance=stance_result,
                stance_crosstab=stance_table,
                discourse=discourse_result,
                community=community_result,
                timeseries=timeseries_result,
                replies=replies_result,
                events=[result.event for result in event_results],
                rows=rows,
                cleaning_rows=(store.load_comments(run_id), rows),
                include=include_charts,
            )
            # 采集口径：与命令行报告同一份口径，避免两条路径给出不同说法
            collection_text = ""
            try:
                from ..analyze.collection import (
                    load_params, render_collection, summarise_collection,
                )
                from ..research_log import ResearchLog

                log_events: list[dict] = []
                log_path = settings.research_log_path
                if log_path.exists():
                    log_events = [
                        event for event in ResearchLog.read(log_path)
                        if event.get("run_id") == run_id
                    ]
                collection_text = render_collection(
                    summarise_collection(
                        params=load_params(store.get_run(run_id) or {}),
                        rows=store.load_analysis_rows(run_id),
                        log_events=log_events,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                emit("warn", f"采集口径小节跳过：{type(exc).__name__}: {exc}")

            content = build_analysis_report(
                run=run_info,
                cleaning=cleaning,
                sentiment=sentiment_result,
                stance=stance_result,
                stance_crosstab=stance_table,
                discourse_stance=cluster_stance_table,
                event_results=event_results,
                discourse=discourse_result,
                community=community_result,
                crosstab=crosstab_result,
                timeseries=timeseries_result,
                replies=replies_result,
                collection_text=collection_text,
                charts=generated,
                output_dir=out_dir,
            )
            report_path = write_report(out_dir / "report.md", content)

            docx_path = None
            try:
                from ..analyze.docx_export import export_report_docx

                docx_path = export_report_docx(content, out_dir)
            except Exception as exc:  # noqa: BLE001
                emit("warn", f"Word 导出失败（Markdown 仍可用）：{type(exc).__name__}: {exc}")

            return {
                "out_dir": out_dir,
                "report_path": report_path,
                "docx_path": docx_path,
                "charts": [path.name for path in generated],
            }

        def done(result):
            self._log("ok", "=" * 40)
            self._log("ok", f"分析完成，图表 {len(result['charts'])} 张")
            for name in result["charts"]:
                self._log("info", f"  · {name}")
            self._log("ok", f"报告：{result['report_path']}")
            if result.get("docx_path"):
                self._log("ok", f"Word：{result['docx_path']}")
            if messagebox.askyesno(
                "分析完成",
                f"图表与报告已生成到：\n{result['out_dir']}\n\n现在打开这个目录吗？",
            ):
                self._open_path(result["out_dir"])

        self._start_job(job, done)

    def _on_stop(self) -> None:
        self.stop_event.set()
        self._log("warn", "已请求停止：当前这篇帖子采集完就会停下，已采数据会保留。")

    # ------------------------------------------------------------------ #
    # 跨话题对比
    # ------------------------------------------------------------------ #
    def _on_compare(self) -> None:
        """对比最近几次采集。

        刻意先出「可比性判断」再出表：如果两次采集的参数不同，
        直接并排列数字会让人得出错误的"话题 A 更热"。
        """
        from ..store import Store

        self._log("info", "=" * 40)
        self._log("info", "跨话题对比 …")

        def emit(level: str, message: str) -> None:
            self.queue.put(("log", level, message))

        def job():
            from ..analyze import charts
            from ..analyze.service import run_comparison

            store = Store(self.settings.db_path)
            store.init_schema()
            result, skipped = run_comparison(store)
            for reason in skipped:
                emit("warn", f"跳过 {reason}")
            if len(result.profiles) < 2:
                raise ConfigProblem(
                    what_happened="可对比的采集不足 2 次。",
                    detail="先采集第二个话题，或确认已有采集完成了清洗。",
                )

            out_dir = self.settings.export_dir / "_comparison"
            out_dir.mkdir(parents=True, exist_ok=True)
            text = result.render()
            path = out_dir / "compare.md"
            path.write_text("# 跨话题对比\n\n```\n" + text + "\n```\n", encoding="utf-8")
            generated = charts.render_all(out_dir, compare=result)

            docx_path = None
            try:
                from ..analyze.docx_export import markdown_to_docx

                docx_path = markdown_to_docx(
                    path.read_text(encoding="utf-8"),
                    out_dir / "compare.docx", base_dir=out_dir,
                )
            except Exception as exc:  # noqa: BLE001
                emit("warn", f"Word 导出失败：{type(exc).__name__}: {exc}")

            return {
                "out_dir": out_dir,
                "path": path,
                "docx_path": docx_path,
                "verdict": result.verdict(),
                "text": text,
                "charts": [item.name for item in generated],
            }

        def done(result):
            self._log("ok", "=" * 40)
            self._log("info", result["text"])
            self._log("ok", f"可比性判断：{result['verdict']}")
            self._log("ok", f"报告：{result['path']}")
            if result.get("docx_path"):
                self._log("ok", f"Word：{result['docx_path']}")
            if messagebox.askyesno(
                "对比完成",
                f"对比结果已生成到：\n{result['out_dir']}\n\n现在打开这个目录吗？",
            ):
                self._open_path(result["out_dir"])

        self._start_job(job, done)

    # ------------------------------------------------------------------ #
    # 方法助手
    # ------------------------------------------------------------------ #
    def _on_suggest(self) -> None:
        """规则驱动的下一步建议（不走 LLM，结果确定可复现）。"""
        from ..assistant import build_context, render_suggestions, suggest_next_steps
        from ..store import Store

        try:
            store = Store(self.settings.db_path)
            store.init_schema()
            context = build_context(store, self.settings)
        except Exception as exc:  # noqa: BLE001
            self._log("error", f"读取当前状态失败：{exc}")
            return
        self._log("info", "=" * 40)
        self._log("info", "当前状态：")
        for line in context.to_prompt().splitlines():
            self._log("muted", line)
        self._log("info", render_suggestions(suggest_next_steps(context)))

    def _on_ask(self) -> None:
        question = (self.ask_var.get() or "").strip()
        if not question:
            self._log("warn", "请先输入问题，例如：我的样本量够做聚类吗？")
            return

        def job():
            from ..assistant import ask, build_context
            from ..llm.client import LlmClient
            from ..store import Store

            store = Store(self.settings.db_path)
            store.init_schema()
            context = build_context(store, self.settings)
            client = LlmClient()
            ok, reason = client.available()
            if not ok:
                raise ApiError(
                    what_happened=f"大模型不可用（{reason}），暂时无法回答。",
                    detail="可以在命令行用 python -m wrc.cli ask --suggest 看规则建议。",
                )
            return ask(question, context, client=client)

        def done(answer):
            self._log("info", f"问：{question}")
            for line in str(answer).splitlines():
                self._log("ok", f"  {line}")
            self.ask_var.set("")

        self._log("info", f"正在咨询助手：{question}")
        self._start_job(job, done)

    # ------------------------------------------------------------------ #
    def _selected_posts(self) -> list[WeiboPost]:
        """高亮行的帖子（用于「设置目标数」这类针对当前行的操作）。"""
        return [self.posts[int(iid) - 1] for iid in self.tree.selection()]

    def _checked_posts(self) -> list[WeiboPost]:
        """勾选的帖子——只有它们会被采集。"""
        return [post for post in self.posts if str(post.mid) in self.checked]

    def _row_index_of(self, event) -> int | None:
        row = self.tree.identify_row(event.y)
        if not row:
            return None
        try:
            return int(row)
        except ValueError:
            return None

    def _on_tree_click(self, event) -> None:
        """只在点「选」列时切换勾选，点别处保持普通的行选择行为。"""
        if self.tree.identify("region", event.x, event.y) != "cell":
            return
        column = self.tree.identify_column(event.x)
        if column != "#1":  # 第 1 列 = pick
            return
        index = self._row_index_of(event)
        if index is None:
            return
        self._toggle_checked(index)

    def _toggle_checked_current(self, _event=None):
        index = self._highlighted_index()
        if index is not None:
            self._toggle_checked(index)
        return "break"

    def _highlighted_index(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            return None
        try:
            return int(selection[0])
        except ValueError:
            return None

    def _toggle_checked(self, index: int) -> None:
        if not 1 <= index <= len(self.posts):
            return
        post = self.posts[index - 1]
        mid = str(post.mid)
        if mid in self.checked:
            self.checked.discard(mid)
        else:
            self.checked.add(mid)
        self._render_row(index)

    def _render_row(self, index: int) -> None:
        """把一行的勾选符号与目标数重画。所有改状态的入口都走这里，避免两处不一致。"""
        if not 1 <= index <= len(self.posts):
            return
        post = self.posts[index - 1]
        when = post.created_at.strftime("%Y-%m-%d %H:%M") if post.created_at else "未知"
        box = CHECKED_BOX if str(post.mid) in self.checked else UNCHECKED_BOX
        self.tree.item(
            str(index),
            values=(
                box, index, when, f"{post.engagement:,}",
                f"{post.comments_count:,}", str(self._target_of(post)),
                post.preview(48),
            ),
        )
        self._update_pick_label()

    def _update_pick_label(self) -> None:
        label = getattr(self, "pick_label", None)
        if label is None:
            return
        checked = len(self._checked_posts())
        label.configure(
            text=f"已勾选 {checked}/{len(self.posts)} 篇",
            foreground="#0b6b3a" if checked else "#6b7280",
        )

    def _select_all(self) -> None:
        self.checked = {str(post.mid) for post in self.posts}
        self._refresh_target_column()

    def _select_none(self) -> None:
        self.checked.clear()
        self._refresh_target_column()

    def _invert_selection(self) -> None:
        self.checked = {
            str(post.mid) for post in self.posts if str(post.mid) not in self.checked
        }
        self._refresh_target_column()

    # ---- 时间窗 ----
    def _quick_window(self, days: int) -> None:
        since, until = recent_days(days)
        self.since_var.set(since.strftime("%Y-%m-%d %H:%M"))
        self.until_var.set(until.strftime("%Y-%m-%d %H:%M"))
        self._log("info", f"时间窗已设为最近 {days} 天：{since:%Y-%m-%d} ～ {until:%Y-%m-%d}")

    def _clear_window(self) -> None:
        self.since_var.set("")
        self.until_var.set("")
        self._log("info", "时间窗已清空：采集该帖可见的最新评论（不限时间）。")

    def _window_from_ui(self) -> tuple[object, object]:
        """解析界面上的时间窗；格式错误时弹窗说明而不是静默当成不限。"""
        raw_since = self.since_var.get().strip()
        raw_until = self.until_var.get().strip()
        try:
            since = parse_window_time(raw_since)
        except ValueError as exc:
            raise ConfigProblem(
                what_happened="时间窗的起点看不懂。",
                detail=f"{exc}（起点那一栏填的是「{raw_since}」）",
            ) from exc
        try:
            until = parse_window_time(raw_until, end_of_day=True)
        except ValueError as exc:
            raise ConfigProblem(
                what_happened="时间窗的终点看不懂。",
                detail=f"{exc}（终点那一栏填的是「{raw_until}」）",
            ) from exc
        if since and until and since > until:
            raise ConfigProblem(
                what_happened="时间窗起点晚于终点。",
                detail=f"起点 {since:%Y-%m-%d %H:%M}，终点 {until:%Y-%m-%d %H:%M}，请检查是否填反。",
            )
        return since, until

    # ---- 逐帖目标数 ----
    def _global_target(self) -> int:
        try:
            return max(1, min(MAX_COMMENTS_PER_POST, int(self.comments_var.get())))
        except (tk.TclError, ValueError):
            return 50

    def _target_of(self, post: WeiboPost) -> int:
        return int(self.per_post_targets.get(str(post.mid), self._global_target()))

    def _targets_for_selected(self) -> dict[str, int]:
        """只回传"被改过"的帖子，避免把 20 个默认值当成用户决策写进采集参数。"""
        return {
            str(post.mid): int(self.per_post_targets[str(post.mid)])
            for post in self._checked_posts()
            if str(post.mid) in self.per_post_targets
        }

    def _refresh_target_column(self) -> None:
        tree = getattr(self, "tree", None)
        if tree is None:  # 采集区先于帖子列表构建，此时还没有表格
            return
        for iid in tree.get_children():
            self._render_row(int(iid))
        self._update_pick_label()

    def _target_scope(self) -> list[WeiboPost]:
        """「设置目标数」的作用范围：优先勾选的帖子，否则用高亮行。"""
        checked = self._checked_posts()
        return checked or self._selected_posts()

    def _on_set_target(self) -> None:
        scope = self._target_scope()
        if not scope:
            messagebox.showwarning(
                "没有勾选帖子", "请先勾选要设置目标数的帖子（点「选」列）。"
            )
            return
        current = self._target_of(scope[0])
        value = simpledialog.askinteger(
            "设置目标评论数",
            f"为勾选/选中的 {len(scope)} 篇帖子设置目标评论数（1–500）：\n"
            "留空或取消则不修改。",
            initialvalue=current, minvalue=1, maxvalue=MAX_COMMENTS_PER_POST, parent=self,
        )
        if value is None:
            return
        for post in scope:
            self.per_post_targets[str(post.mid)] = int(value)
        self._refresh_target_column()
        planned = estimate_comment_requests(
            [self._target_of(post) for post in scope]
        )
        self._log(
            "info",
            f"已把 {len(scope)} 篇帖子的目标数设为 {value} 条；"
            f"按最小页估算约需 {planned} 次请求（上限 {self.max_requests_var.get()} 次）。",
        )

    def _on_reset_target(self) -> None:
        scope = self._target_scope()
        if not scope:
            return
        for post in scope:
            self.per_post_targets.pop(str(post.mid), None)
        self._refresh_target_column()
        self._log("info", f"已把 {len(scope)} 篇帖子恢复为全局值 {self._global_target()} 条。")

    def _update_tier_hint(self) -> None:
        tier = self.tier_var.get()
        meta = TIER_META.get(tier, {})
        if meta.get("needs_ai") and not self.ai_var.get():
            self._log("warn", "重度档包含 AI 判定，请同时勾选「启用 AI 模型」。")

    def _show_consent(self) -> None:
        messagebox.showinfo("数据使用声明", render_statement())

    def _refresh_ai_status_async(self) -> None:
        def job():
            return self.pipeline.ai_status()

        def done(status):
            self.ai_status = status
            color = "#0b6b3a" if status.ready else "#8a5a00"
            self.ai_label.configure(text=f"AI 状态：{status.summary()}", foreground=color)
            self._log("info", f"AI {status.summary()}")
            if not status.ready:
                for step in status.remediation():
                    self._log("muted", f"    · {step}")

        self._start_job(job, done)

    def _open_model_dir(self) -> None:
        target = self.settings.ai_model_dir
        target.mkdir(parents=True, exist_ok=True)
        self._open_path(target)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:  # noqa: BLE001
            self._log("warn", f"无法打开目录：{exc}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
