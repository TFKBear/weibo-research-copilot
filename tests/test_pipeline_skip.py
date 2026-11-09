"""采集容错：单帖失败不能毁掉整轮采集。

这个测试来自一次真实事故：用户按「4 篇帖子 × 100 条」采集，
第 1 篇拿到 100 条，第 2 篇接口返回 `ok=0 暂无数据`（帖子被删/评论不可见），
旧代码直接抛出 → 整轮中止 → 那 100 条连清洗都没跑过，等于白采。
用户重试了一次，又是同样结果，浪费两次请求。

现在的规则：
- 单篇失败 → 跳过、记进研究日志、继续下一篇；
- 运行状态标为 ``partial``，并在 note 里写明跳过了几篇；
- 只有"一篇都没采到"才算失败（抛 EmptyResult）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from wrc.errors import CommentSourceEmpty, EmptyResult
from wrc.pipeline import CollectOptions, ResearchPipeline
from wrc.store import Store
from wrc.weibo.models import TimePrecision, WeiboComment, WeiboPost


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


def _comment(cid: str) -> WeiboComment:
    return WeiboComment(
        cid=cid, post_mid="m1", user_id="u1", screen_name="某人", text_raw="评论",
        created_at=None, created_at_raw="今天 10:00",
        created_at_precision=TimePrecision.MINUTE,
    )


class _FakeBrowser:
    """替代 Playwright：不启动浏览器，只提供 async context manager。"""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeClient:
    """按 post_mid 决定"采到"还是"报错"。"""

    def __init__(self, behaviour: dict[str, Exception | int]) -> None:
        self._behaviour = behaviour
        self.calls: list[str] = []

    async def check_login(self) -> bool:
        return True

    async def collect_comments(self, post_mid: str, *, target: int, **_kwargs):
        self.calls.append(post_mid)
        outcome = self._behaviour.get(post_mid)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, SimpleNamespace):
            return outcome
        count = outcome if isinstance(outcome, int) else 2
        comments = [_comment(f"{post_mid}-{index}") for index in range(count)]
        return SimpleNamespace(
            comments=comments, count=len(comments),
            backend="mobile_hotflow", stopped_reason="reached_target",
        )


def _run(store: Store, tmp_path, behaviour, posts):
    pipeline = ResearchPipeline(_settings(tmp_path))
    client = _FakeClient(behaviour)

    import wrc.pipeline as pipeline_module

    original_browser = getattr(pipeline_module, "BrowserSession", None)
    original_client = ResearchPipeline._client
    original_limiter = ResearchPipeline._limiter
    import wrc.session as session_module

    saved = session_module.BrowserSession
    session_module.BrowserSession = _FakeBrowser  # type: ignore[assignment]
    ResearchPipeline._client = lambda self, *_a, **_k: client  # type: ignore[assignment]
    ResearchPipeline._limiter = lambda self, _opts: SimpleNamespace(  # type: ignore[assignment]
        budget=SimpleNamespace(used=7)
    )
    try:
        return asyncio.run(pipeline.collect(
            CollectOptions(tag="#测试#", comments_per_post=10, post_limit=len(posts)),
            posts,
        )), client
    finally:
        session_module.BrowserSession = saved  # type: ignore[assignment]
        ResearchPipeline._client = original_client  # type: ignore[assignment]
        ResearchPipeline._limiter = original_limiter  # type: ignore[assignment]
        if original_browser is not None:  # pragma: no cover - 仅保持引用
            pass


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


def test_single_failed_post_does_not_abort_the_run(store, tmp_path):
    """真实事故的回归测试：第 2 篇报"暂无数据"，第 1、3 篇应照常入库。"""
    posts = [_post("m1"), _post("m2"), _post("m3")]
    behaviour = {
        "m1": 2,
        "m2": CommentSourceEmpty(post_mid="m2", expected_total=50, attempts=()),
        "m3": 3,
    }
    (run_id, total, _used), client = _run(store, tmp_path, behaviour, posts)

    assert client.calls == ["m1", "m2", "m3"]  # 没有在第 2 篇停下
    assert total == 5  # 2 + 3，第 2 篇跳过
    run = store.get_run(run_id)
    assert run["status"] == "partial"
    assert "跳过" in (run["note"] or "")


def test_all_posts_failing_is_a_real_error(store, tmp_path):
    posts = [_post("m1"), _post("m2")]
    behaviour = {
        "m1": CommentSourceEmpty(post_mid="m1", expected_total=50, attempts=()),
        "m2": CommentSourceEmpty(post_mid="m2", expected_total=50, attempts=()),
    }
    with pytest.raises(EmptyResult) as excinfo:
        _run(store, tmp_path, behaviour, posts)
    assert "没有拿到任何评论" in excinfo.value.what_happened
    # detail 要逐篇说明原因，而不是含糊地说"平台没内容"
    assert "共尝试 2 篇帖子" in excinfo.value.detail
    assert "m1" in excinfo.value.detail and "m2" in excinfo.value.detail


def test_skips_are_recorded_in_research_log(store, tmp_path):
    posts = [_post("m1"), _post("m2")]
    behaviour = {
        "m1": 1,
        "m2": CommentSourceEmpty(post_mid="m2", expected_total=50, attempts=()),
    }
    (run_id, _total, _used), _client = _run(store, tmp_path, behaviour, posts)

    from wrc.research_log import ResearchLog

    events = list(ResearchLog.read(tmp_path / "research_log.jsonl"))
    kinds = [event["event"] for event in events]
    assert "post_skipped" in kinds
    skipped = next(event for event in events if event["event"] == "post_skipped")
    assert skipped["post_mid"] == "m2"
    assert skipped["run_id"] == run_id


def test_clean_run_keeps_status_done(store, tmp_path):
    posts = [_post("m1")]
    (run_id, total, _used), _client = _run(store, tmp_path, {"m1": 4}, posts)
    assert total == 4
    assert store.get_run(run_id)["status"] == "done"


@pytest.mark.parametrize("error_kind, expected_calls", [
    ("api", ["m1", "m2"]), ("auth_required", ["m1"]),
    ("risk_blocked", ["m1"]), ("budget", ["m1"]),
])
def test_partial_result_is_persisted_and_global_errors_stop(
    store, tmp_path, error_kind, expected_calls,
):
    partial = SimpleNamespace(
        comments=[_comment("saved")], count=1, backend="mobile_hotflow",
        stopped_reason="pagination_error", error_kind=error_kind,
        error_message="后续页失败",
    )
    (run_id, total, _), client = _run(
        store, tmp_path, {"m1": partial, "m2": 2}, [_post("m1"), _post("m2")])
    assert client.calls == expected_calls
    assert total == (3 if error_kind == "api" else 1)
    assert store.get_run(run_id)["status"] == "partial"
    assert any(c["cid"] == "saved" for c in store.load_comments(run_id))
