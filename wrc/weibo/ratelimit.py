"""采集频率纪律（Agent 执行规则第 2 条）。

三个硬约束，缺一不可：
1. **随机延迟**：请求之间 1.5–3 秒均匀抖动，避免固定间隔形成机器节律。
2. **指数退避**：命中限流后按 60s → 120s → 240s 退避，带抖动，封顶 10 分钟。
3. **硬性请求预算**：单次采集（单账号单 tag）的出站请求总数封顶，
   接近上限时主动提示用户，而不是等微博来封。

预算计的是**全部出站请求**（含重试），因为风控看到的是真实流量，
不是我们逻辑上的「一次采集任务」。
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from ..errors import BudgetExceeded


@dataclass(slots=True)
class RequestBudget:
    """出站请求计数器。"""

    limit: int
    used: int = 0
    warn_ratio: float = 0.8
    _warned: set[int] = field(default_factory=set)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def ratio(self) -> float:
        return self.used / self.limit if self.limit else 1.0

    def consume(self) -> int:
        """记一次请求；超限时抛 ``BudgetExceeded``。"""
        if self.used >= self.limit:
            raise BudgetExceeded(used=self.used, limit=self.limit)
        self.used += 1
        return self.used

    def should_warn(self) -> bool:
        """是否第一次跨过告警线。"""
        if self.limit <= 0:
            return False
        threshold = int(self.limit * self.warn_ratio)
        if self.used >= threshold and threshold not in self._warned:
            self._warned.add(threshold)
            return True
        return False


class RateLimiter:
    """带抖动的全局节流器。

    串行化是刻意的：并发会让风控判定「异常流量」的概率显著上升，
    而研究场景对吞吐要求很低（几百到几千条评论）。
    """

    def __init__(
        self,
        *,
        delay_range: tuple[float, float] = (1.5, 3.0),
        budget: RequestBudget | None = None,
        backoff_base_s: float = 60.0,
        backoff_max_s: float = 600.0,
        rng: random.Random | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        lo, hi = delay_range
        self.delay_range = (min(lo, hi), max(lo, hi))
        self.budget = budget or RequestBudget(limit=10_000)
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_request_at: float | None = None
        self.total_slept_s: float = 0.0

    # ---------------------------------------------------------------- #
    def next_delay(self) -> float:
        lo, hi = self.delay_range
        return self._rng.uniform(lo, hi)

    def backoff_seconds(self, attempt: int) -> float:
        """第 ``attempt`` 次重试前的等待秒数（attempt 从 0 开始）。"""
        raw = self.backoff_base_s * (2 ** max(0, attempt))
        jittered = raw * self._rng.uniform(0.85, 1.15)
        return min(self.backoff_max_s, round(jittered, 1))

    async def wait_turn(self) -> int:
        """等待轮到自己发下一个请求，并消耗一次预算。

        Returns:
            已使用的请求序号（从 1 开始）。

        Raises:
            BudgetExceeded: 预算用尽。调用方必须停下来问用户，而不是重试。
        """
        async with self._lock:
            used = self.budget.consume()
            if self._last_request_at is not None:
                delay = self.next_delay()
                self.total_slept_s += delay
                await self._sleep(delay)
            self._last_request_at = time.monotonic()
            return used

    async def sleep_backoff(self, attempt: int) -> float:
        seconds = self.backoff_seconds(attempt)
        self.total_slept_s += seconds
        await self._sleep(seconds)
        return seconds

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RateLimiter(delay={self.delay_range}, used={self.budget.used}/"
            f"{self.budget.limit})"
        )
