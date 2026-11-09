"""回复结构（二级评论）分析的测试。

关键不是"能不能算出层数"，而是：
- 没采二级评论时说"没有"，而不是画一张空图；
- 父评论被清洗掉后不静默接到别的根上（孤儿要单独计数）；
- 小讨论串不展示（可识别性）；
- 导出的结构里不能出现用户标识。
"""

from __future__ import annotations

from wrc.analyze.replies import (
    CONCENTRATION_MIN_REPLIES,
    CONCENTRATION_THRESHOLD,
    build_reply_structure,
)


def _root(cid, text="根评论"):
    return {"cid": cid, "parent_cid": None, "level": 1, "cleaned_text": text}


def _reply(cid, parent, level=2, text="回复"):
    return {"cid": cid, "parent_cid": parent, "level": level, "cleaned_text": text}


def test_no_reply_data_is_reported_not_drawn():
    rows = [_root("r1"), _root("r2")]
    result = build_reply_structure(rows)
    assert result.available is False
    assert result.n_replies == 0
    text = result.render()
    assert "没有二级评论" in text
    assert "采集二级评论" in text


def test_empty_input_degrades():
    result = build_reply_structure([])
    assert result.available is False
    assert "无法分析回复结构" in result.render()


def test_simple_thread_metrics():
    rows = [_root("r1"), _root("r2")]
    rows += [_reply(f"a{i}", "r1") for i in range(8)]
    result = build_reply_structure(rows, min_n=2)
    assert result.available is True
    assert result.n_comments == 10
    assert result.n_roots == 2
    assert result.n_replies == 8
    assert result.n_threads == 1
    assert result.max_depth == 1  # 根评论以外的层数
    assert result.roots_with_replies == 1


def test_depth_counts_nested_replies():
    rows = [_root("r1"), _reply("a", "r1"), _reply("b", "a", level=3),
            _reply("c", "b", level=4)]
    result = build_reply_structure(rows, min_n=1)
    assert result.max_depth == 3


def test_orphans_from_cleaning_are_counted_and_flagged():
    rows = [
        _root("r1"),
        _reply("a1", "r1"),
        _reply("a2", "r1"),
        # 父评论被清洗掉了
        _reply("o1", "deleted-parent"),
    ]
    result = build_reply_structure(rows, min_n=1)
    assert result.n_orphans == 1
    assert any("父评论在清洗中被移除" in warning for warning in result.warnings)


def test_small_threads_are_suppressed_from_listing():
    rows = [_root("big")] + [_reply(f"b{i}", "big") for i in range(6)]
    rows += [_root("tiny"), _reply("t1", "tiny")]
    result = build_reply_structure(rows, min_n=5)
    assert result.suppressed_threads == 1
    assert all(thread.size >= 5 for thread in result.top_threads)
    assert "不予展示" in result.render()


def test_single_point_detonation_shape():
    rows = [_root("hot")] + [_reply(f"h{i}", "hot") for i in range(30)]
    result = build_reply_structure(rows, min_n=2)
    assert result.concentration >= CONCENTRATION_THRESHOLD
    assert result.shape_label() == "单点引爆型"
    assert result.n_replies >= CONCENTRATION_MIN_REPLIES


def test_dispersed_shape_when_many_small_threads():
    rows = []
    for index in range(10):
        rows.append(_root(f"r{index}"))
        rows.append(_reply(f"c{index}", f"r{index}"))
    result = build_reply_structure(rows, min_n=1)
    assert result.shape_label() == "多点分散型"
    assert result.concentration < CONCENTRATION_THRESHOLD


def test_roots_without_replies_are_not_threads():
    """没有 level≥2 的记录时，不能断言"没人回复"——数据无法区分未采集与真无人回复。"""
    rows = [_root(f"r{i}") for i in range(5)]
    result = build_reply_structure(rows)
    assert result.n_threads == 0
    assert result.available is False
    text = result.render()
    assert "无法区分" in text
    assert "无人回复" in text


def test_empty_threads_are_still_counted_when_replies_exist():
    rows = [_root("r1"), _root("r2"), _reply("a1", "r1")]
    result = build_reply_structure(rows, min_n=1)
    assert result.available is True
    assert result.n_roots == 2
    assert result.n_threads == 1


def test_no_user_identifiers_in_export():
    rows = [_root("r1")] + [_reply(f"a{i}", "r1") for i in range(3)]
    for index, row in enumerate(rows):
        row["user_id"] = f"u{index}"
        row["screen_name"] = f"昵称{index}"
    payload = build_reply_structure(rows, min_n=1).to_dict()
    dumped = str(payload)
    assert "user_id" not in dumped
    assert "screen_name" not in dumped
    assert "昵称" not in dumped
    # 节点里也不该出现 cid（能对应到具体评论就接近可识别了）
    for thread in payload["top_threads"]:
        for node in thread["nodes"]:
            assert "cid" not in node


def test_large_thread_is_trimmed_but_keeps_parents():
    rows = [_root("r1")] + [_reply(f"a{i}", "r1") for i in range(100)]
    result = build_reply_structure(rows, min_n=1, max_nodes=20)
    assert result.top_threads[0].size == 101  # 规模如实报告
    assert len(result.top_threads[0].nodes) == 20  # 图上只画 20 个


def test_cycles_do_not_hang():
    """脏数据里可能出现互相引用的父子关系，不能让程序卡死。"""
    rows = [
        {"cid": "a", "parent_cid": "b", "level": 2, "cleaned_text": "a"},
        {"cid": "b", "parent_cid": "a", "level": 2, "cleaned_text": "b"},
    ]
    result = build_reply_structure(rows, min_n=1)
    assert result.n_comments == 2  # 能返回就说明没有死循环


def test_render_states_privacy_policy():
    rows = [_root("r1")] + [_reply(f"a{i}", "r1") for i in range(5)]
    text = build_reply_structure(rows, min_n=2).render()
    assert "不含任何用户标识" in text


def test_few_replies_trigger_instability_warning():
    rows = [_root("r1"), _reply("a1", "r1"), _reply("a2", "r1")]
    result = build_reply_structure(rows, min_n=1)
    assert any("不稳定" in warning for warning in result.warnings)
