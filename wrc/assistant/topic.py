"""话题推荐（模块 E2 / P2）。

需求里对这块的约束是全篇最硬的一条：

> 严禁 LLM 在未做实际探测的情况下虚构 tag 的讨论量。

所以这里的流程是**先让 LLM 想 tag，再逐个真搜一遍**，把探测结果附在候选上。
LLM 只负责"想到可能相关的标签"，**量级信息一律由真实请求提供**。

关于"量级"的口径也必须诚实：
微博检索接口不返回话题的总帖数（``cardlistInfo.total`` 在移动端检索里通常缺失），
所以这里报告的是 **探测到的量级信号**：
首两页能拿到多少条不同帖子、互动量的中位数、最新一条的时间。
它回答的是"这个标签是活的还是死的"，而不是"这个话题一共有多少条"。
把探测值说成总量，就是另一种形式的虚构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

PROMPT_VERSION = "topic-1.0.0"

#: 一次最多探测几个标签（每个标签约 2 次请求，受全局请求预算约束）。
MAX_PROBES = 5

SYSTEM_PROMPT = """你是社会科学研究的话题标签顾问，服务对象是研究微博舆论的学生。

任务：根据用户描述的事件/人物/议题，提出 5 个可用于**检索微博话题**的候选标签。

## 硬性要求
- 只用你认为**真实存在**的微博话题写法（形如「延迟退休」「小米SU7」这样的词），
  不要编造带年份后缀的假标签，也不要输出 `#` 符号。
- **绝对不要估计讨论量或帖子数**。数量由工具实测，你猜的数字会误导研究者。
- 每个标签给一句推荐理由，说明它为什么可能聚合了相关讨论
  （例如"事件初期的主要聚合标签""官方媒体的统一用语"）。

## 输出格式
只输出 JSON：
{"tags":[{"tag":"延迟退休","reason":"政策讨论的长期聚合标签"},
         {"tag":"渐进式延迟退休","reason":"政策正式表述，官方报道多用此说法"}]}"""


@dataclass(slots=True)
class ProbeResult:
    """一次真实检索探测的结果。"""

    tag: str
    ok: bool = False
    posts: int = 0
    pages: int = 0
    median_engagement: int = 0
    latest_post_at: str | None = None
    samples: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def level(self) -> str:
        """把探测结果翻译成量级描述（基于实测，不是猜测）。"""
        if not self.ok:
            return "无法探测"
        if self.posts == 0:
            return "未检索到帖子"
        if self.posts <= 5:
            return "讨论很少"
        if self.posts <= 15:
            return "有一定讨论"
        return "讨论活跃"

    def describe(self) -> str:
        if not self.ok:
            return f"探测失败：{self.error}"
        parts = [f"{self.level}（首 {self.pages} 页共 {self.posts} 条）"]
        if self.median_engagement:
            parts.append(f"互动中位数 {self.median_engagement}")
        if self.latest_post_at:
            parts.append(f"最新一条 {self.latest_post_at}")
        return "，".join(parts)


@dataclass(slots=True)
class TagCandidate:
    tag: str
    reason: str
    probe: ProbeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "reason": self.reason,
            "probe": (
                {
                    "ok": self.probe.ok,
                    "level": self.probe.level,
                    "posts": self.probe.posts,
                    "pages": self.probe.pages,
                    "median_engagement": self.probe.median_engagement,
                    "latest_post_at": self.probe.latest_post_at,
                    "samples": self.probe.samples,
                    "error": self.probe.error,
                }
                if self.probe
                else None
            ),
        }

    def render(self, index: int) -> str:
        lines = [f"  {index}. {self.tag}"]
        lines.append(f"     理由：{self.reason}")
        if self.probe is not None:
            lines.append(f"     实测：{self.probe.describe()}")
            for sample in self.probe.samples[:2]:
                lines.append(f"       · {sample}")
        else:
            lines.append("     实测：未探测")
        return "\n".join(lines)


def parse_tags(payload: Any) -> list[tuple[str, str]]:
    """解析 LLM 返回的标签列表。"""
    if isinstance(payload, dict):
        items = payload.get("tags") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    out: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str):
            tag, reason = item, ""
        elif isinstance(item, dict):
            tag = str(item.get("tag") or item.get("name") or "").strip()
            reason = str(item.get("reason") or "").strip()
        else:
            continue
        tag = tag.strip().strip("#").strip()
        if tag and tag not in {existing for existing, _ in out}:
            out.append((tag, reason))
    return out


def propose_tags(description: str, *, client: Any, limit: int = MAX_PROBES) -> list[tuple[str, str]]:
    """让模型提出候选标签（只提名字和理由，不提数量）。"""
    payload = client.complete_json(
        system=SYSTEM_PROMPT,
        user=f"用户描述：{description}\n\n请提出 {limit} 个候选话题标签。",
        max_tokens=900,
    )
    return parse_tags(payload)[:limit]


async def probe_tag(
    tag: str,
    *,
    settings: Any,
    pages: int = 2,
    samples: int = 2,
) -> ProbeResult:
    """真实检索一次，拿到该标签的量级信号。"""
    from ..session import BrowserSession
    from ..weibo.collect import WeiboClient
    from ..weibo.ratelimit import RateLimiter, RequestBudget

    result = ProbeResult(tag=tag)
    limiter = RateLimiter(
        delay_range=settings.effective_delay_range(),
        budget=RequestBudget(limit=max(4, pages * 2)),
    )
    try:
        async with BrowserSession(settings, notify=lambda _m: None) as session:
            client = WeiboClient(session.fetcher, limiter, max_retries=1)
            if not await client.check_login():
                result.error = "登录态无效"
                return result
            posts = await client.search_posts(
                f"#{tag}#", limit=10 * pages, max_pages=pages, order="engagement"
            )
            result.ok = True
            result.posts = len(posts)
            result.pages = pages
            if posts:
                engagements = sorted(post.engagement for post in posts)
                result.median_engagement = engagements[len(engagements) // 2]
                dated = [post for post in posts if post.created_at is not None]
                if dated:
                    newest = max(dated, key=lambda post: post.created_at)
                    result.latest_post_at = newest.created_at.strftime("%Y-%m-%d")
                result.samples = [post.preview(30) for post in posts[:samples]]
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
    return result


async def recommend_tags(
    description: str,
    *,
    client: Any,
    settings: Any,
    limit: int = MAX_PROBES,
    emit: Callable[[str, str], None] | None = None,
) -> list[TagCandidate]:
    """完整流程：LLM 提候选 → 逐个真搜 → 附实测结果。"""
    notify = emit or (lambda _level, _message: None)
    proposed = propose_tags(description, client=client, limit=limit)
    if not proposed:
        return []

    candidates: list[TagCandidate] = []
    for index, (tag, reason) in enumerate(proposed, start=1):
        notify("info", f"({index}/{len(proposed)}) 探测标签 {tag} …")
        probe = await probe_tag(tag, settings=settings)
        notify("info", f"  {probe.describe()}")
        candidates.append(TagCandidate(tag=tag, reason=reason, probe=probe))

    # 按实测活跃度排序：把"检索不到"的排最后，避免推荐一堆死标签
    candidates.sort(key=lambda item: -(item.probe.posts if item.probe else 0))
    return candidates


def render_candidates(candidates: Sequence[TagCandidate]) -> str:
    lines = ["候选话题（量级为实测探测值，不是模型估计）", "=" * 62]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(candidate.render(index))
    lines.append("")
    lines.append(
        "说明：微博检索接口不返回话题总帖数，上面的「实测」是首若干页能拿到的"
        "帖子数与互动水平，用来判断标签是否活跃，不等于话题总量。"
    )
    return "\n".join(lines)
