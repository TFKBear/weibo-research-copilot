"""方法助手测试。

重点测「规则驱动的下一步建议」——它必须是确定性的，
因为同一个数据状态应该永远给出同一个建议；用 LLM 做这件事反而不可复现。
"""

from __future__ import annotations

from wrc.assistant import SessionContext, suggest_next_steps
from wrc.assistant.context import SessionContext as Ctx


def test_empty_state_suggests_collecting():
    steps = suggest_next_steps(SessionContext())
    assert len(steps) == 1
    assert steps[0].action == "collect"


def test_uncleaned_state_suggests_cleaning():
    ctx = Ctx(run_id="r1", tag="#t#", n_comments=50, n_kept=50, clean_tier="")
    steps = suggest_next_steps(ctx)
    assert steps[0].action == "clean"


def test_small_sample_is_flagged():
    ctx = Ctx(run_id="r1", n_comments=46, n_kept=46, clean_tier="moderate")
    steps = suggest_next_steps(ctx)
    actions = [step.action for step in steps]
    assert "collect_more" in actions
    reason = next(step.reason for step in steps if step.action == "collect_more")
    assert "46" in reason          # 建议必须基于实际数字，不能是套话


def test_large_sample_skips_expand_suggestion():
    ctx = Ctx(run_id="r1", n_comments=400, n_kept=380, clean_tier="moderate")
    actions = [step.action for step in suggest_next_steps(ctx)]
    assert "collect_more" not in actions


def test_high_disagreement_triggers_manual_review():
    ctx = Ctx(
        run_id="r1", n_comments=400, n_kept=380, clean_tier="moderate",
        has_sentiment=True, agreement_rate=0.42,
    )
    steps = suggest_next_steps(ctx)
    assert any(step.action == "review_sentiment" for step in steps)


def test_weak_cluster_silhouette_triggers_caution():
    ctx = Ctx(
        run_id="r1", n_comments=400, n_kept=380, clean_tier="moderate",
        has_sentiment=True, has_discourse=True, discourse_k=4,
        discourse_silhouette=0.12,
    )
    steps = suggest_next_steps(ctx)
    assert any(step.action == "expand_sample" for step in steps)


def test_finished_workflow_suggests_report_and_review():
    ctx = Ctx(
        run_id="r1", n_comments=400, n_kept=380, clean_tier="moderate",
        has_sentiment=True, agreement_rate=0.05,
        has_discourse=True, discourse_k=4, discourse_silhouette=0.42,
    )
    steps = suggest_next_steps(ctx)
    assert steps
    assert any(step.action in {"build_report", "review"} for step in steps)


def test_suggestions_are_capped():
    ctx = Ctx(run_id="r1", n_comments=46, n_kept=46, clean_tier="moderate")
    assert len(suggest_next_steps(ctx, limit=2)) <= 2


def test_context_prompt_contains_actual_numbers():
    ctx = Ctx(
        run_id="r1", tag="#小米SU7#", n_posts=1, n_comments=50, n_kept=46,
        clean_tier="moderate", pipeline_version="basic-0.2.0",
        has_sentiment=True, sentiment_distribution={"negative": 14, "positive": 6},
        agreement_rate=0.2,
    )
    text = ctx.to_prompt()
    assert "#小米SU7#" in text
    assert "46" in text
    assert "1%" not in text.replace("92.0%", "X")  # 保留率算对
    assert "moderate" in text
    assert "negative" in text
    assert "20.0%" in text


def test_context_prompt_without_data():
    assert "还没有任何采集数据" in SessionContext().to_prompt()
