"""逐帖评论目标数（需求 A2-3）。

需求原文：*每篇帖子的评论采集数量独立配置，而不是全局固定 1000/2000/5000。
全局固定值在社科场景下会导致"小帖子过度采集、大帖子严重不足"的抽样偏差。*

这条要求容易被读成界面细节，其实是**抽样效度**问题：
同一话题下，爆款帖有 20 万条评论、个案帖只有 30 条，用同一个目标数去采，
最终样本量由帖子热度决定，而不是由研究设计决定。所以：
- 覆盖能力放在采集模型（`CollectOptions.target_for`）里，CLI 与界面共用；
- 界面只回传"被改过"的帖子，不把 20 个默认值伪装成研究者的决策；
- 采集前估算请求数，总和超上限时先提醒，而不是中途被掐断。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from wrc.pipeline import (
    MAX_COMMENTS_PER_POST,
    CollectOptions,
    ResearchPipeline,
    estimate_comment_requests,
)
from wrc.store import Store
from wrc.weibo.models import TimePrecision, WeiboComment, WeiboPost


# --------------------------------------------------------------------------- #
# 覆盖逻辑
# --------------------------------------------------------------------------- #
def test_target_for_uses_global_value_when_no_override():
    opts = CollectOptions(tag="#延迟退休#", comments_per_post=80)
    assert opts.target_for("m1") == 80


def test_target_for_prefers_override():
    opts = CollectOptions(
        tag="#延迟退休#", comments_per_post=80, per_post_targets={"m1": 200}
    )
    assert opts.target_for("m1") == 200
    assert opts.target_for("m2") == 80


def test_target_for_clamps_and_falls_back():
    opts = CollectOptions(
        tag="#t#",
        comments_per_post=50,
        per_post_targets={"zero": 0, "huge": 99999, "junk": "很多"},  # type: ignore[dict-item]
    )
    assert opts.target_for("zero") == 1
    assert opts.target_for("huge") == MAX_COMMENTS_PER_POST
    # 非数字不能悄悄变成 1 条，要退回全局值——否则用户以为采了 50 条，实际只有 1 条
    assert opts.target_for("junk") == 50


def test_targets_for_returns_resolved_map():
    posts = [_post("m1"), _post("m2")]
    opts = CollectOptions(tag="#t#", comments_per_post=40, per_post_targets={"m2": 90})
    assert opts.targets_for(posts) == {"m1": 40, "m2": 90}


# --------------------------------------------------------------------------- #
# 请求数估算
# --------------------------------------------------------------------------- #
def test_estimate_grows_with_target():
    assert estimate_comment_requests([10]) == 2
    assert estimate_comment_requests([50]) == 6
    assert estimate_comment_requests([50, 50]) == 12
    assert estimate_comment_requests([500]) == 51


def test_estimate_ignores_empty_selection():
    assert estimate_comment_requests([]) == 0


# --------------------------------------------------------------------------- #
# 命令行写法 --per-post 1=200,3=50
# --------------------------------------------------------------------------- #
def test_index_targets_parses_pairs():
    from wrc.cli import _index_targets

    assert _index_targets("1=200") == [(1, 200)]
    assert _index_targets("1=200, 3=50") == [(1, 200), (3, 50)]


def test_index_targets_rejects_bad_syntax_with_readable_message():
    from wrc.cli import _index_targets

    for bad, hint in (
        ("1-200", "="),
        ("1=0", "1–500"),
        ("1=999", "1–500"),
        ("0=5", "从 1 开始"),
        ("x=5", "非数字"),
    ):
        with pytest.raises(ValueError) as excinfo:
            _index_targets(bad)
        assert hint in str(excinfo.value), (bad, str(excinfo.value))


def test_parse_per_post_maps_numbers_to_mids():
    from wrc.cli import _parse_per_post

    posts = [_post("m1"), _post("m2"), _post("m3")]
    assert _parse_per_post("1=200,3=50", posts) == {"m1": 200, "m3": 50}


def test_parse_per_post_rejects_out_of_range():
    from wrc.cli import _parse_per_post

    posts = [_post("m1")]
    assert _parse_per_post("2=50", posts) is None


def test_collect_options_reads_subcomment_target():
    from wrc.cli import _collect_options, build_parser

    args = build_parser().parse_args(
        ["run", "--tag", "#t#", "--select", "1", "--subcomment-target", "25"]
    )
    assert _collect_options(args).subcomment_target == 25


# --------------------------------------------------------------------------- #
# 采集流程真的按逐帖目标去取
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    return store


def _post(mid: str) -> WeiboPost:
    return WeiboPost(
        mid=mid, bid=mid, user_id="u1", screen_name="某人", text_raw="帖子",
        created_at=None, created_at_raw="05-12", created_at_precision=TimePrecision.DAY,
        reposts_count=1, comments_count=50, attitudes_count=1,
    )


class _FakeBrowser:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _RecordingClient:
    """记下每帖实际请求的目标数。"""

    def __init__(self) -> None:
        self.targets: dict[str, int] = {}
        self.subcomment_targets: list[int] = []

    async def check_login(self) -> bool:
        return True

    async def collect_comments(self, post_mid: str, *, target: int, **kwargs):
        self.targets[post_mid] = target
        if "subcomment_target" in kwargs:
            self.subcomment_targets.append(kwargs["subcomment_target"])
        comments = [_comment(f"{post_mid}-{i}") for i in range(min(3, target))]
        return SimpleNamespace(
            comments=comments, count=len(comments),
            backend="mobile_hotflow", stopped_reason="reached_target",
            window={}, plan={},
        )


def _comment(cid: str) -> WeiboComment:
    return WeiboComment(
        cid=cid, post_mid="m1", user_id="u1", screen_name="某人", text_raw="评论",
        created_at=None, created_at_raw="今天 10:00",
        created_at_precision=TimePrecision.MINUTE,
    )


def _settings(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path,
        db_path=tmp_path / "wrc.sqlite3",
        log_dir=tmp_path,
        export_dir=tmp_path / "exports",
        research_log_path=tmp_path / "research_log.jsonl",
        ensure_dirs=lambda: None,
        effective_delay_range=lambda: (0.0, 0.0),
    )


def _collect(store, tmp_path, opts: CollectOptions, posts):
    pipeline = ResearchPipeline(_settings(tmp_path))
    client = _RecordingClient()
    import wrc.session as session_module

    saved = session_module.BrowserSession
    original_client = ResearchPipeline._client
    original_limiter = ResearchPipeline._limiter
    session_module.BrowserSession = _FakeBrowser  # type: ignore[assignment]
    ResearchPipeline._client = lambda self, *_a, **_k: client  # type: ignore[assignment]
    ResearchPipeline._limiter = lambda self, _opts: SimpleNamespace(  # type: ignore[assignment]
        budget=SimpleNamespace(used=5)
    )
    try:
        run_id, _total, _used = asyncio.run(pipeline.collect(opts, posts))
    finally:
        session_module.BrowserSession = saved  # type: ignore[assignment]
        ResearchPipeline._client = original_client  # type: ignore[assignment]
        ResearchPipeline._limiter = original_limiter  # type: ignore[assignment]
    return run_id, client


def test_collect_requests_each_post_with_its_own_target(store, tmp_path):
    posts = [_post("m1"), _post("m2"), _post("m3")]
    opts = CollectOptions(
        tag="#延迟退休#",
        comments_per_post=50,
        per_post_targets={"m1": 300, "m3": 5},
    )
    _run_id, client = _collect(store, tmp_path, opts, posts)

    # 大帖子多采、小帖子少采，中间那帖保持全局值
    assert client.targets == {"m1": 300, "m2": 50, "m3": 5}


def test_subcomment_target_is_passed_down(store, tmp_path):
    """二级评论目标数必须真的传到采集层，不能只落在参数里。"""
    posts = [_post("m1")]
    opts = CollectOptions(
        tag="#延迟退休#", comments_per_post=5,
        include_subcomments=True, subcomment_target=25,
    )
    _run_id, client = _collect(store, tmp_path, opts, posts)
    assert client.subcomment_targets == [25]


def test_run_params_record_resolved_per_post_targets(store, tmp_path):
    posts = [_post("m1"), _post("m2")]
    opts = CollectOptions(
        tag="#延迟退休#", comments_per_post=50, per_post_targets={"m1": 300}
    )
    run_id, _client = _collect(store, tmp_path, opts, posts)

    import json

    params = json.loads(store.get_run(run_id)["params_json"])
    # 必须记录"每帖最终采了多少目标"，否则附录无法复现抽样过程
    assert params["per_post_targets"] == {"m1": 300, "m2": 50}
    assert params["comments_per_post"] == 50


def test_over_budget_plan_warns_before_starting(store, tmp_path):
    seen: list[tuple[str, str]] = []
    posts = [_post(f"m{index}") for index in range(1, 5)]
    opts = CollectOptions(
        tag="#延迟退休#",
        comments_per_post=200,
        max_requests=10,  # 4 帖 × (20 页 + 1) 远超上限
    )

    pipeline = ResearchPipeline(_settings(tmp_path), emit=lambda lv, msg: seen.append((lv, msg)))
    client = _RecordingClient()
    import wrc.session as session_module

    saved = session_module.BrowserSession
    original_client = ResearchPipeline._client
    original_limiter = ResearchPipeline._limiter
    session_module.BrowserSession = _FakeBrowser  # type: ignore[assignment]
    ResearchPipeline._client = lambda self, *_a, **_k: client  # type: ignore[assignment]
    ResearchPipeline._limiter = lambda self, _opts: SimpleNamespace(  # type: ignore[assignment]
        budget=SimpleNamespace(used=1)
    )
    try:
        asyncio.run(pipeline.collect(opts, posts))
    finally:
        session_module.BrowserSession = saved  # type: ignore[assignment]
        ResearchPipeline._client = original_client  # type: ignore[assignment]
        ResearchPipeline._limiter = original_limiter  # type: ignore[assignment]

    warnings = [msg for level, msg in seen if level == "warn"]
    assert any("超过本次上限" in msg for msg in warnings), warnings
