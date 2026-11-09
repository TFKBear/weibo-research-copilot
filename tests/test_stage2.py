"""第 2 步新增能力的测试：清洗档位、规则表、AI 判定、去重策略。

这一批测试的重点不是"功能能不能跑"，而是**判据是否可解释**：
每条被删的评论都必须能说出"因为哪条规则/哪个相似度"。
"""

from __future__ import annotations

import numpy as np
import pytest

from wrc.ai.analysis import (
    AIAnalysis,
    MarketingHit,
    best_seed_match,
    cluster_similar_texts,
)
from wrc.clean.basic import TIER_STEPS, Cleaner, TIER_META
from wrc.clean.rules import is_suspected_bot, match_ad
from wrc.pipeline import CleanOptions


# --------------------------------------------------------------------------- #
# 规则表
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "加V信领取优惠",
        "想买的私信我",
        "专业代购，长期合作",
        "点击下方链接直接下单",
        "电话 13800138000 详聊",
    ],
)
def test_strong_ad_rules_hit(text):
    hit = match_ad(text)
    assert hit is not None
    assert hit.strength == "strong"
    assert hit.labels


@pytest.mark.parametrize(
    "text",
    [
        "这个价格有优惠吗，看起来还行",
        "我还是觉得等降价再买比较好",
        "客观说，这款车续航表现一般",
        "支持一下，希望国产车越做越好",
    ],
)
def test_weak_rules_do_not_fire_alone(text):
    """弱规则只命中一条时不能判定为广告。

    这是刻意的保守设计：'优惠''下单'在正常消费者表达里非常常见，
    单条命中就删，等于系统性删掉真实的批评与讨论。
    """
    assert match_ad(text) is None


def test_weak_rules_fire_when_two_hit():
    hit = match_ad("优惠力度很大，直接下单链接在这里")
    assert hit is not None
    assert hit.strength == "weak"
    assert len(hit.labels) >= 2


@pytest.mark.parametrize(
    ("followers", "expected"),
    [(0, True), (3, True), (4, True), (5, False), (1000, False), (None, False)],
)
def test_suspected_bot(followers, expected):
    assert is_suspected_bot(followers) is expected


# --------------------------------------------------------------------------- #
# 清洗档位
# --------------------------------------------------------------------------- #
ROWS = [
    {"cid": "a", "post_mid": "p1", "text_raw": "这个观点我认同", "followers_count": 100},
    {"cid": "b", "post_mid": "p1", "text_raw": "加V信领取优惠", "followers_count": 100},
    {"cid": "c", "post_mid": "p1", "text_raw": "说得好", "followers_count": 2},
    {"cid": "d", "post_mid": "p2", "text_raw": "这个观点我认同", "followers_count": 100},
]


def test_light_tier_keeps_ads_and_bots():
    """轻度档不该碰广告与僵尸号——那属于中度档的职责。"""
    result = Cleaner(tier="light", dedupe_scope="post").run(ROWS)
    kept = {row["cid"] for row in result.kept_rows}
    assert kept == {"a", "b", "c", "d"}


def test_moderate_tier_drops_ads_by_default():
    result = Cleaner(tier="moderate", dedupe_scope="post").run(ROWS)
    reasons = {row["cid"]: row["drop_reasons"] for row in result.rows}
    assert reasons["b"] and reasons["b"][0].startswith("drop_ad_keywords:")


def test_moderate_tier_does_not_drop_low_follower_users_by_default():
    """默认不按粉丝数删人。

    接口不返回注册时间，只凭"粉丝 <5"就删，会系统性删掉真实的低互动用户
    （新用户、退休用户、只看不发的沉默用户）。这是「可解释清洗优先于极限去噪」
    的底线，所以它必须由用户显式勾选。
    """
    result = Cleaner(tier="moderate", dedupe_scope="post").run(ROWS)
    assert "c" in {row["cid"] for row in result.kept_rows}


def test_bot_filter_works_when_explicitly_enabled():
    result = Cleaner(
        tier="moderate", dedupe_scope="post",
        enabled_steps=("drop_suspected_bots",),
    ).run(ROWS)
    kept = {row["cid"] for row in result.kept_rows}
    assert "c" not in kept
    reasons = {row["cid"]: row["drop_reasons"] for row in result.rows}
    assert reasons["c"][0].startswith("drop_suspected_bots:")


def test_heavy_tier_uses_external_ai_flags():
    flags = {"a": ["ai_marketing:0.88"], "d": ["ai_coordinated:簇内4条"]}
    result = Cleaner(tier="heavy", dedupe_scope="post").run(ROWS, external_flags=flags)
    # b 被广告规则删，a/d 被 AI 删；c 是低粉丝用户，默认保留（不按粉丝数删人）
    assert {row["cid"] for row in result.kept_rows} == {"c"}
    reasons = {row["cid"]: row["drop_reasons"] for row in result.rows}
    assert reasons["a"] == ["ai_marketing:0.88"]
    assert reasons["d"] == ["ai_coordinated:簇内4条"]


def test_heavy_tier_without_flags_does_not_drop_silently():
    """勾了重度档但没跑 AI 时，AI 步骤不能凭空删东西。"""
    result = Cleaner(tier="heavy", dedupe_scope="post").run(ROWS)
    assert {row["cid"] for row in result.kept_rows} == {"a", "c", "d"}


def test_dedupe_off_keeps_cross_post_duplicates():
    """用户要研究水军模板时，跨帖重复本身就是研究对象，不能删。"""
    result = Cleaner(tier="light", dedupe_scope="global", enabled_steps=()).run(ROWS)
    assert result.total_kept == len(ROWS)

    strict = Cleaner(tier="light", dedupe_scope="global").run(ROWS)
    assert strict.total_kept == len(ROWS) - 1  # a 与 d 完全相同，跨帖只留一条


def test_every_tier_has_meta_for_gui():
    for key in TIER_STEPS:
        assert key in TIER_META
        assert TIER_META[key]["label"]
        assert TIER_META[key]["summary"]


# --------------------------------------------------------------------------- #
# AI 判定（纯函数，不需要模型）
# --------------------------------------------------------------------------- #
def test_best_seed_match_respects_threshold():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    seeds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    hits = best_seed_match(vectors, seeds, threshold=0.9)
    assert hits[0] is not None and hits[0].score == 1.0
    assert hits[1] is not None and hits[1].score == 1.0
    assert hits[2] is None  # 0.707 < 0.9


def test_cluster_detects_template_blocks_and_leaves_singletons():
    vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    sizes, clusters = cluster_similar_texts(
        vectors, threshold=0.95, min_cluster_size=3
    )
    assert sizes[0] == 3 and sizes[1] == 3 and sizes[2] == 3
    assert sizes[3] == 1
    assert clusters == 1


def test_cluster_ignores_pairs_when_min_size_is_three():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sizes, clusters = cluster_similar_texts(
        vectors, threshold=0.95, min_cluster_size=3
    )
    assert sizes[0] == 2
    assert clusters == 0


def test_analysis_flags_carry_explainable_numbers():
    analysis = AIAnalysis(enabled=True)
    analysis.marketing["x"] = MarketingHit(score=0.876, seed="加V信领取优惠")
    analysis.coordinated["x"] = 4
    assert analysis.flags_for("x") == ["ai_marketing:0.88", "ai_coordinated:簇内4条"]
    assert analysis.flags_for("y") == []


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
def test_clean_options_dedupe_switch():
    on = CleanOptions(tier="moderate", dedupe_scope="global")
    off = CleanOptions(tier="moderate", dedupe_scope="off")
    assert "dedupe_exact" in on.resolve_steps()
    assert "dedupe_exact" not in off.resolve_steps()


def test_clean_options_heavy_includes_ai_steps():
    steps = CleanOptions(tier="heavy").resolve_steps()
    assert "ai_marketing" in steps and "ai_coordinated" in steps
    assert "drop_ad_keywords" in steps


def test_bot_filter_is_opt_in_only():
    assert "drop_suspected_bots" not in CleanOptions(tier="moderate").resolve_steps()
    assert "drop_suspected_bots" in CleanOptions(
        tier="moderate", filter_bots=True
    ).resolve_steps()


# --------------------------------------------------------------------------- #
# AI 自检
# --------------------------------------------------------------------------- #
def test_detect_status_reports_actionable_steps(tmp_path):
    from wrc.ai.status import detect_status

    status = detect_status(tmp_path / "不存在")
    assert not status.ready
    assert status.remediation()
    assert any("onnx" in step.lower() or "模型" in step for step in status.remediation())


def test_detect_status_finds_model_files_in_subfolder(tmp_path):
    """用户把整个仓库解压进来时，文件在 onnx/ 子目录里，也要能认出来。"""
    from wrc.ai.status import detect_status

    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_quantized.onnx").write_bytes(b"fake")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    status = detect_status(tmp_path)
    assert status.model_ok
    assert status.onnx_path is not None
    assert status.onnx_path.name == "model_quantized.onnx"
    assert status.tokenizer_path is not None


def test_detect_status_prefers_full_precision_model(tmp_path):
    from wrc.ai.status import detect_status

    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"full")
    (tmp_path / "onnx" / "model_quantized.onnx").write_bytes(b"quant")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    status = detect_status(tmp_path)
    assert status.onnx_path is not None
    assert status.onnx_path.name == "model.onnx"
