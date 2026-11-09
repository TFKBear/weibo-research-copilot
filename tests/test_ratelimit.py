"""频率纪律测试（Agent 执行规则第 2 条）。"""

from __future__ import annotations

import asyncio
import random

import pytest

from wrc.errors import BudgetExceeded
from wrc.weibo.ratelimit import RateLimiter, RequestBudget


def test_budget_consumes_and_raises_at_limit():
    budget = RequestBudget(limit=2)
    assert budget.consume() == 1
    assert budget.consume() == 2
    with pytest.raises(BudgetExceeded) as excinfo:
        budget.consume()
    assert "2/2" in excinfo.value.what_happened
    assert budget.remaining == 0


def test_budget_warns_only_once_at_threshold():
    budget = RequestBudget(limit=10, warn_ratio=0.8)
    warnings = 0
    for _ in range(10):
        budget.consume()
        warnings += int(budget.should_warn())
    assert warnings == 1


def test_limiter_sleeps_between_requests_but_not_before_first():
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    limiter = RateLimiter(
        delay_range=(1.5, 3.0),
        budget=RequestBudget(limit=5),
        rng=random.Random(1),
        sleep=fake_sleep,
    )

    async def scenario() -> None:
        await limiter.wait_turn()
        await limiter.wait_turn()
        await limiter.wait_turn()

    asyncio.run(scenario())
    assert len(slept) == 2  # 第一次请求前不需要等待
    assert all(1.5 <= value <= 3.0 for value in slept)


def test_limiter_propagates_budget_exhaustion():
    async def fake_sleep(_seconds: float) -> None:
        return None

    limiter = RateLimiter(
        delay_range=(0.0, 0.0), budget=RequestBudget(limit=1), sleep=fake_sleep
    )

    async def scenario() -> None:
        await limiter.wait_turn()
        await limiter.wait_turn()

    with pytest.raises(BudgetExceeded):
        asyncio.run(scenario())


def test_backoff_grows_and_is_capped():
    limiter = RateLimiter(
        delay_range=(0.0, 0.0),
        budget=RequestBudget(limit=100),
        backoff_base_s=60.0,
        backoff_max_s=600.0,
        rng=random.Random(3),
    )
    first = limiter.backoff_seconds(0)
    third = limiter.backoff_seconds(2)
    huge = limiter.backoff_seconds(10)
    assert 51 <= first <= 69          # 60s ± 15%
    assert 204 <= third <= 276        # 240s ± 15%
    assert huge == pytest.approx(600.0, abs=0.1)  # 封顶


def test_delay_range_normalizes_reversed_bounds():
    limiter = RateLimiter(delay_range=(3.0, 1.5), budget=RequestBudget(limit=5))
    assert limiter.delay_range == (1.5, 3.0)
