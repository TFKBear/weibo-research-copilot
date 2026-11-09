"""离线演示数据（仅供 ``--mock`` 与测试使用，不参与真实采集）。

存在的理由：技术验证脚本必须在**没有登录态、没有网络**的情况下也能被完整跑通，
否则一旦微博接口波动，我们连"自己的管线是否还正常"都判断不了。
这本身也是可复现性的一部分：任何人都能用同一份合成数据复现同一条管线。
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .errors import ApiError
from .weibo.endpoints import Request

_MAX_BASE = 1_000_000
_PAGE_SIZE = 20

_POST_TEMPLATES = [
    "【讨论】延迟退休对不同群体的影响差异很大，一线体力劳动者和办公室职员的处境完全不同，"
    "政策设计需要考虑这种差异 <a href='/n/x'>#延迟退休#</a>",
    "我算了一下，如果按渐进式方案，我这代人大概要到 63 岁。<img alt=\"[思考]\" src=\"x.gif\">",
    "支持渐进式改革，但配套的就业保障和职业培训必须跟上，否则只是把压力转移给个人。",
    "转发了最新政策解读，几个关键点值得注意：自愿弹性、小步调整、统筹兼顾。<img alt=\"[good]\" src=\"y.gif\">",
    "身边已经有同事开始重新规划职业路径了，政策的预期管理比政策本身更重要。",
    "这个话题下情绪化的表达很多，建议先把官方原文读一遍再讨论。",
]

_COMMENT_BANK = [
    "确实，体力劳动者和脑力劳动者的实际退休年龄感受差别太大了。",
    "支持渐进式，但希望弹性选择能真正落地，而不是纸面权利。",
    "我父母那一代已经够辛苦了，真的不希望再往后延。",
    "配套的养老和医疗保障跟上了吗？只谈退休年龄不谈保障是不完整的。",
    "个人觉得分行业、分岗位制定更合理。",
    "楼上说得有道理，但具体怎么分行业操作起来很难。",
    "已经在重新规划我的职业路径了，四十年太长了。",
    "这个政策讨论了好几年，希望能听到更多一线工人的声音。",
    "不同意。延迟退休会挤压年轻人的就业机会。",
    "不完全同意上面的说法，老年劳动力与青年岗位并不完全替代。",
    "建议先把养老金并轨的问题解决了再说。",
    "看了官方原文，其实是「小步调整、弹性实施」，和网传的不一样。",
    "转发扩散，希望更多人理性讨论。",
]

#: 会被"轻度"档清洗掉的典型样本，用来让清洗报告有真实可看的内容。
_NOISE_COMMENTS = [
    "[doge]",
    "[笑cry][笑cry][笑cry]",
    "😂😂😂",
    "网页链接",
    "https://m.weibo.cn/status/5000000000000001",
    "@人民日报 @央视新闻",
    "@张三 ",
]

_DUPLICATE_COMMENTS = [
    "确实，体力劳动者的实际情况完全不同。",
    "希望政策多听听一线工人的声音。",
]

#: 与评论主句组合的后半句，用来让合成数据具备足够的话语多样性，
#: 避免演示时去重比例虚高、掩盖真实管线行为。
_COMMENT_TAILS = [
    "",
    "，尤其是制造业和建筑业的从业者。",
    "，我身边就有这样的例子。",
    "，不过也要看到政策的渐进性。",
    "，希望后续有更细的配套细则。",
    "，这一点在各地执行上差异很大。",
    "，评论区的讨论也印证了这一点。",
    "，但我不确定这个判断的依据。",
]


def build_scenario(
    *,
    tag: str = "延迟退休",
    posts: int = 6,
    comments_per_post: int = 24,
    seed: int = 7,
) -> Callable[[Request], Any]:
    """构造一个确定性的接口模拟器。"""

    rng = random.Random(seed)
    post_ids = [f"5{idx:015d}" for idx in range(1, posts + 1)]
    post_meta = {
        mid: {
            "comments_target": comments_per_post + idx * 3,
            "created_at": f"2025-05-{10 + idx:02d}",
            "text": _POST_TEMPLATES[idx % len(_POST_TEMPLATES)],
            "attitudes": rng.randint(200, 90_000),
            "comments_count": rng.randint(50, 4_000),
            "reposts_count": rng.randint(10, 3_000),
            "followers": rng.randint(30, 500_000),
        }
        for idx, mid in enumerate(post_ids)
    }
    comment_cache: dict[str, list[dict[str, Any]]] = {}

    def _comments_for(mid: str) -> list[dict[str, Any]]:
        if mid in comment_cache:
            return comment_cache[mid]
        meta = post_meta[mid]
        total = meta["comments_target"]
        items: list[dict[str, Any]] = []
        for index in range(total):
            roll = rng.random()
            if roll < 0.10:
                text = _NOISE_COMMENTS[index % len(_NOISE_COMMENTS)]
            elif roll < 0.15:
                text = _DUPLICATE_COMMENTS[index % len(_DUPLICATE_COMMENTS)]
            else:
                base = _COMMENT_BANK[index % len(_COMMENT_BANK)].rstrip("。")
                tail = _COMMENT_TAILS[(index * 5 + 1) % len(_COMMENT_TAILS)]
                text = f"{base}{tail}"
            items.append(
                {
                    "id": f"{mid}{index:04d}",
                    "text": text,
                    "created_at": "今天 10:23" if index % 2 else "昨天 21:05",
                    "like_counts": rng.randint(0, 480),
                    "total_number": rng.randint(0, 3),
                    "user": {
                        "id": f"u{rng.randint(1000, 1400)}",
                        "screen_name": f"用户{index:03d}",
                        "followers_count": rng.choice([3, 12, 88, 512, 4_800, 120_000]),
                    },
                }
            )
        comment_cache[mid] = items
        return items

    def router(request: Request) -> Any:
        url = request.url
        if url.endswith("/api/config"):
            return {"ok": 1, "data": {"login": True, "uid": "mock-user", "st": True}}

        if "container/getIndex" in url:
            page = int(request.params.get("page", 1))
            if page > 1:
                return {"ok": 1, "data": {"cards": []}}
            cards: list[Any] = []
            for index, mid in enumerate(post_ids):
                meta = post_meta[mid]
                mblog = {
                    "id": mid,
                    "mid": mid,
                    "bid": f"Mock{index}",
                    "created_at": meta["created_at"],
                    "text": meta["text"],
                    "user": {
                        "id": f"p{index}",
                        "screen_name": f"发布者{index}",
                        "followers_count": meta["followers"],
                    },
                    "reposts_count": meta["reposts_count"],
                    "comments_count": meta["comments_count"],
                    "attitudes_count": meta["attitudes"],
                    "isLongText": False,
                }
                if index % 3 == 0:
                    # 模拟 card_group 形态，确保两条解析路径都被走一遍
                    cards.append(
                        {
                            "card_type": 9,
                            "card_group": [{"card_type": 9, "mblog": mblog}],
                        }
                    )
                else:
                    cards.append({"card_type": 9, "mblog": mblog})
            return {"ok": 1, "data": {"cards": cards}}

        if "comments/hotflow" in url:
            mid = str(request.params.get("mid") or request.params.get("id") or "")
            if mid not in comment_cache and mid not in post_meta:
                raise ApiError(detail=f"mock: 未知 mid={mid}")
            items = _comments_for(mid)
            max_id = int(request.params.get("max_id") or 0)
            offset = 0 if max_id == 0 else _MAX_BASE - max_id
            window = items[offset : offset + _PAGE_SIZE]
            next_offset = offset + len(window)
            next_max_id = _MAX_BASE - next_offset if next_offset < len(items) else 0
            return {
                "ok": 1,
                "data": {
                    "data": window,
                    "max_id": next_max_id,
                    "max_id_type": 0,
                    "total_number": len(items),
                },
            }

        if "hotFlowChild" in url:
            return {"ok": 1, "data": {"data": [], "max_id": 0, "max_id_type": 0}}

        raise ApiError(detail=f"mock: 未覆盖的请求 {url}")

    return router
