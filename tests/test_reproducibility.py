"""可复现性附录（研究日志正式版）的测试。

这一层的价值全在"说的是真的"：
- 校验和不能算错、不能自引用；
- 复现等级必须逐环节给，不能笼统说"可复现"；
- 「清洗可逐位复现」这句话必须**被重放校验验证过**，而不是写在纸上。

所以测试重点：附录结构完整性 + 重放校验能真的发现不一致 + 推断有依据时会标注。
"""

from __future__ import annotations

import pytest

from wrc.reproducibility import (
    AppendixFacts,
    EXACT,
    NONE,
    build_appendix,
    cleaning_facts,
    collect_files,
    environment_snapshot,
    reproducibility_items,
    verify_clean_replay,
    write_appendix,
)
from wrc.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    return store


def _seed(store: Store, run_id: str = "r1", *, tier="moderate",
          texts=None, filters_bots=False, dated: bool = False):
    from datetime import datetime, timedelta, timezone

    from wrc.weibo.models import TimePrecision, WeiboComment

    texts = texts or [
        "小米的做工确实不错",
        "小米的做工确实不错",          # 精确重复
        "[doge]",                      # 纯表情
        "https://example.com/x",       # 纯链接
        "@某人",                        # 纯 @
        "这车的保值率堪忧",
    ]
    cst = timezone(timedelta(hours=8))
    comments = [
        WeiboComment(
            cid=f"c{index}", post_mid="p1", user_id=f"u{index}",
            screen_name="某人", text_raw=text,
            created_at=(datetime(2026, 9, 20, 10, 0, tzinfo=cst) if dated else None),
            created_at_raw="今天 10:00", created_at_precision=TimePrecision.MINUTE,
            followers_count=0 if filters_bots and index == 0 else 400,
        )
        for index, text in enumerate(texts)
    ]
    store.create_run(
        run_id=run_id, tag="#测试#", order_mode="engagement",
        params={"mode": "live", "comments_per_post": 50, "post_limit": 2},
    )
    store.save_comments(run_id, comments)

    from wrc.clean.basic import Cleaner
    from wrc.pipeline import CleanOptions

    options = CleanOptions(tier=tier, dedupe_scope="global")
    options.filter_bots = filters_bots
    result = Cleaner(
        tier=tier, dedupe_scope="global", enabled_steps=options.resolve_steps()
    ).run(store.load_comments(run_id))
    store.save_clean(
        run_id, result.rows, clean_tier=tier, pipeline_version=result.pipeline_version
    )
    return result


# --------------------------------------------------------------------------- #
# 附录渲染
# --------------------------------------------------------------------------- #
def test_appendix_has_all_sections_and_states_missing_values():
    facts = AppendixFacts(
        run={"run_id": "r1", "tag": "#测试#", "created_at": "2025-01-01T10:00:00+08:00"},
        params={"mode": "live", "comments_per_post": 50},
        cleaning={"tier": "moderate", "pipeline_version": "basic-0.2.0",
                  "total_input": 10, "total_kept": 8, "retention_rate": 0.8,
                  "steps": [{"label": "精确去重", "removed": 2, "share": 0.2}]},
        summaries={},
        files=[{"name": "analysis/a.csv", "size_kb": 1.0, "sha256": "a" * 64}],
        env=environment_snapshot(),
        generated_at="2025-01-01 12:00:00",
    )
    text = build_appendix(facts)
    for heading in ("一、运行环境", "二、采集", "三、清洗", "四、分析模块",
                    "六、可复现性分级", "八、产出文件与校验和",
                    "九、复核步骤", "十、已知局限"):
        assert heading in text
    assert "采集会话：`r1`" in text
    assert "精确去重" in text
    # 没采到的东西写"未记录"，不许留空
    assert "未记录" in text


def test_mock_run_is_stamped_as_unusable():
    facts = AppendixFacts(
        run={"run_id": "m1", "tag": "#样例#"},
        params={"mode": "mock"},
        env=environment_snapshot(),
    )
    text = build_appendix(facts)
    assert "非真实数据" in text
    assert "不得" in text and "mock" in text.lower()


def test_llm_section_states_uncontrolled_sampling():
    facts = AppendixFacts(
        run={"run_id": "r1", "tag": "#t#"},
        params={"mode": "live"},
        llm={"base_url": "https://x/v1", "model": "m", "calls": 3,
             "prompt_tokens": 100, "completion_tokens": 50},
        env=environment_snapshot(),
    )
    text = build_appendix(facts)
    assert "调用次数：3" in text
    assert "不受本工具控制" in text or "不由本工具控制" in text


# --------------------------------------------------------------------------- #
# 复现等级
# --------------------------------------------------------------------------- #
def test_reproducibility_grades_are_per_stage_and_honest():
    facts = AppendixFacts(
        summaries={"sentiment": {"method": {}}, "cluster_discourse": {"method": {}}},
        cleaning={"pipeline_version": "basic-0.2.0"},
    )
    items = {item.stage: item for item in reproducibility_items(facts)}
    assert items["清洗"].level == EXACT
    assert items["情感标注（LLM 通道）"].level == NONE
    assert items["话语聚类"].level == EXACT
    # 聚类"可复现"必须附带模型文件一致这个前提，否则就是过度承诺
    assert "SHA-256" in items["话语聚类"].requirement
    # 采集永远不可能逐位复现
    assert "近似" in items["数据采集"].level


def test_no_sentiment_means_no_llm_row():
    items = reproducibility_items(AppendixFacts())
    assert not any("LLM" in item.stage for item in items)


# --------------------------------------------------------------------------- #
# 文件指纹
# --------------------------------------------------------------------------- #
def test_collect_files_skips_self_reference_and_derived_files(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "appendix.md").write_text("self", encoding="utf-8")
    (tmp_path / "appendix.docx").write_bytes(b"PK")
    (tmp_path / "chart.png").write_bytes(b"\x89PNG")
    (tmp_path / "report.md").write_text("r", encoding="utf-8")

    names = {item["name"] for item in collect_files(tmp_path)}
    assert names == {"analysis/a.csv", "report.md"}
    # 附录不能把自己的校验和写进正文（写入后必然对不上）
    assert not any("appendix" in name for name in names)


def test_collect_files_missing_dir_is_empty(tmp_path):
    assert collect_files(tmp_path / "不存在") == []


# --------------------------------------------------------------------------- #
# 清洗重放
# --------------------------------------------------------------------------- #
def test_cleaning_facts_derived_from_data(store):
    _seed(store)
    facts = cleaning_facts(store, "r1")
    assert facts["tier"] == "moderate"
    assert facts["total_input"] == 6
    assert facts["total_kept"] < 6
    labels = {step["label"] for step in facts["steps"]}
    assert "精确去重" in labels
    assert "纯表情" in labels


def test_cleaning_facts_none_without_data(store):
    store.create_run(run_id="empty", tag="#空#", order_mode="engagement", params={})
    assert cleaning_facts(store, "empty") is None


def test_replay_is_consistent_on_untouched_data(store):
    _seed(store)
    result = verify_clean_replay(store, "r1")
    assert result.replayable is True
    assert result.checked == 6
    assert result.consistent is True
    assert "完全一致" in result.render()


def test_replay_detects_tampering(store):
    _seed(store)
    # 人为改一条清洗结论：把一条本该被删的改成保留
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT cid FROM clean_comments WHERE run_id='r1' AND kept=0 LIMIT 1"
        ).fetchall()
        cid = str(rows[0]["cid"])
        conn.execute(
            "UPDATE clean_comments SET kept=1, drop_reasons='[]' "
            "WHERE run_id='r1' AND cid=?", (cid,),
        )
    result = verify_clean_replay(store, "r1")
    assert result.consistent is False
    assert result.mismatched
    assert cid in result.render()
    assert "不要" in result.render() or "请勿" in result.render()


def test_replay_refuses_when_ai_flags_were_used(store):
    _seed(store)
    with store.connect() as conn:
        rows = conn.execute("SELECT cid FROM clean_comments WHERE run_id='r1' LIMIT 1").fetchall()
        conn.execute(
            "UPDATE clean_comments SET kept=0, drop_reasons='[\"ai_marketing\"]' "
            "WHERE run_id='r1' AND cid=?", (str(rows[0]["cid"]),),
        )
    result = verify_clean_replay(store, "r1")
    assert result.replayable is False
    assert "AI" in result.reason


def test_replay_infers_bot_filter_from_stored_reasons(store):
    _seed(store, filters_bots=True)
    result = verify_clean_replay(store, "r1")
    assert result.filter_bots is True
    assert result.filter_bots_inferred is True
    assert result.consistent is True
    assert "推定" in result.render()


def test_replay_uses_recorded_filter_flag_without_inference(store):
    _seed(store, filters_bots=True)
    store.merge_run_params("r1", {"filter_bots": True})
    result = verify_clean_replay(store, "r1")
    assert result.filter_bots is True
    assert result.filter_bots_inferred is False


def test_replay_without_data_is_not_an_error(store):
    store.create_run(run_id="empty", tag="#空#", order_mode="engagement", params={})
    result = verify_clean_replay(store, "empty")
    assert result.replayable is False
    assert "没有清洗记录" in result.render()


def test_replay_scope_inference_is_disclosed(store):
    _seed(store)
    result = verify_clean_replay(store, "r1")
    assert result.scope_assumed is True  # 参数里没有 dedupe_scope
    assert "推定" in result.render()


# --------------------------------------------------------------------------- #
def test_write_appendix_creates_file(tmp_path):
    path = write_appendix(tmp_path / "out" / "appendix.md", "# 附录\n\n内容。\n")
    assert path.exists()
    assert "附录" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 采集口径进附录（时间窗 / 检索口径 / 实际覆盖）
# --------------------------------------------------------------------------- #
def test_value_renders_dicts_readably():
    """时间窗是 dict；附录不能用 Python repr 糊一行上去。"""
    from wrc.reproducibility import _value

    text = _value({"active": True, "label": "2026-09-01 00:00 ～ 2026-09-30 23:59"})
    assert "active=是" in text
    assert "{" not in text


def test_collect_facts_records_collection_provenance(store, tmp_path):
    """附录必须写清检索口径、时间窗与实际覆盖——这三项决定样本代表性。"""
    from types import SimpleNamespace

    from wrc.reproducibility import collect_facts

    _seed(store, "r1", dated=True)
    settings = SimpleNamespace(
        export_dir=tmp_path / "exports",
        data_dir=tmp_path,
        db_path=tmp_path / "wrc.sqlite3",
        ai_embedding_model_dir=tmp_path / "no-model",
        ai_sentiment_model_dir=tmp_path / "no-model",
    )
    facts = collect_facts(store, settings, "r1", with_hashes=False)
    text = build_appendix(facts)

    assert "检索口径" in text
    assert "评论时间窗" in text
    assert "实际覆盖" in text
    # 派生值不该在参数表里重复出现，否则会被误读成"采集时填的参数"
    table = text.split("| 参数 | 取值 |")[1] if "| 参数 | 取值 |" in text else ""
    assert "comment_window_label" not in table
    assert "covered_from" not in table
