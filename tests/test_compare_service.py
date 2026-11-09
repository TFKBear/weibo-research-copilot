"""跨话题对比的编排层测试（service 层，需要真实 SQLite）。

`test_compare.py` 测的是纯计算逻辑；这里测的是"从数据库里取出来的参数
能不能支撑可比性判断"。这块曾经有个真 bug：清洗档位要到清洗时才确定，
而参数只在采集时写一次，导致所有老会话都显示"清洗档位未记录"，
可比性判断永远退化为"不可比"。
"""

from __future__ import annotations

import pytest

from wrc.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    return store


def _seed(store: Store, run_id: str, tag: str, *, tier="moderate",
          comments=6, per_post=50, mode="live", params_extra=None):
    from wrc.weibo.models import TimePrecision, WeiboComment

    params = {
        "mode": mode,
        "post_limit": 2,
        "comments_per_post": per_post,
        "include_subcomments": False,
        "comment_backend": "auto",
        "comment_sort": "hot",
    }
    params.update(params_extra or {})
    store.create_run(run_id=run_id, tag=tag, order_mode="engagement", params=params)
    store.save_comments(run_id, [
        WeiboComment(
            cid=f"{run_id}-{index}", post_mid="p1", user_id=f"u{index}",
            screen_name="某人", text_raw=f"评论{index} 内容",
            created_at=None, created_at_raw="今天 10:00",
            created_at_precision=TimePrecision.MINUTE,
        )
        for index in range(comments)
    ])
    store.save_clean(
        run_id,
        [
            {"cid": f"{run_id}-{index}", "post_mid": "p1",
             "cleaned_text": f"评论{index} 内容", "kept": True, "drop_reasons": []}
            for index in range(comments)
        ],
        clean_tier=tier,
        pipeline_version="basic-0.2.0",
    )


def test_merge_run_params_adds_late_known_values(store):
    _seed(store, "r1", "#甲#")
    store.merge_run_params("r1", {"clean_tier": "heavy", "use_ai": True})
    import json

    params = json.loads(store.get_run("r1")["params_json"])
    assert params["clean_tier"] == "heavy"
    assert params["use_ai"] is True
    # 原有参数不能被覆盖掉
    assert params["comments_per_post"] == 50


def test_merge_run_params_is_safe_for_unknown_run(store):
    store.merge_run_params("不存在的会话", {"clean_tier": "light"})  # 不应抛异常


def test_select_prefers_distinct_tags(store):
    from wrc.analyze.service import select_comparable_runs

    _seed(store, "a1", "#话题甲#")
    _seed(store, "b1", "#话题甲#")
    _seed(store, "c1", "#话题乙#")
    selected = select_comparable_runs(store, limit=6)
    tags = [store.get_run(run_id)["tag"] for run_id in selected]
    # 有 2 个不同话题时，不掺同话题的重复采集
    assert len(selected) == 2
    assert set(tags) == {"#话题甲#", "#话题乙#"}


def test_select_pads_with_repeats_when_only_one_topic(store):
    from wrc.analyze.service import select_comparable_runs

    _seed(store, "a1", "#唯一话题#")
    _seed(store, "b1", "#唯一话题#")
    selected = select_comparable_runs(store, limit=6)
    assert len(selected) == 2  # 只有一次话题时，重复采集仍有"稳定性"意义


def test_select_excludes_mock_by_default(store):
    from wrc.analyze.service import select_comparable_runs

    _seed(store, "live1", "#甲#")
    _seed(store, "mock1", "#乙#", mode="mock")
    assert select_comparable_runs(store) == ["live1"]
    assert set(select_comparable_runs(store, include_mock=True)) == {"live1", "mock1"}


def test_clean_tier_is_backfilled_from_data(store):
    """老会话参数里没记档位时，从 clean_comments 读回来，而不是标"未记录"。"""
    from wrc.analyze.service import build_profiles

    _seed(store, "r1", "#甲#", tier="light")
    import json

    params = json.loads(store.get_run("r1")["params_json"])
    params.pop("clean_tier", None)
    store.merge_run_params("r1", params)  # 把档位抹掉，模拟旧数据

    profiles, skipped = build_profiles(store, ["r1"])
    assert skipped == []
    assert profiles[0].params["clean_tier"] == "light"


def test_build_profiles_skips_runs_without_clean_data(store):
    from wrc.analyze.service import build_profiles

    store.create_run(run_id="empty", tag="#空#", order_mode="engagement", params={})
    profiles, skipped = build_profiles(store, ["empty"])
    assert profiles == []
    assert skipped and "清洗" in skipped[0]


def test_run_comparison_end_to_end(store):
    from wrc.analyze.service import run_comparison

    _seed(store, "r1", "#甲#", per_post=30)
    _seed(store, "r2", "#乙#", per_post=50)
    result, skipped = run_comparison(store)
    assert skipped == []
    assert len(result.profiles) == 2
    # 每帖上限不同 → 绝对量不可比，但占比可比
    assert result.volume_comparable is False
    assert result.share_comparable is True
    assert "仅占比可对比" in result.verdict()
    assert any("每帖评论上限" in item for item in result.differences)
