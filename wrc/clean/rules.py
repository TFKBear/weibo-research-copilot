"""中度清洗档的规则表。

规则清洗与 AI 清洗的分工（这一分工是刻意的）：

- **规则表负责"可枚举的广告"**：加了微信、留了手机号、写着"刷单日结"。
  这些有明确字面特征，规则命中即可解释到具体那一条规则，写进论文附录可复核。
- **AI 负责"没有字面特征的话术"**：换了个说法、没有关键词、但语气和结构与
  营销模板高度相似。这种用规则必然误杀，只能靠语义相似度，并且必须给出
  "与哪条模板相似、相似度多少"。

规则表里的每一条都带一个中文标签，命中后会写进清洗报告的 `drop_reasons`，
例如 ``drop_ad_keywords:加V私信``。研究者能一眼看出"因为什么被删"，
而不是看到一个笼统的"判定为广告"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rule:
    """一条清洗规则。"""

    key: str
    label: str
    pattern: re.Pattern[str]

    def match(self, text: str) -> str | None:
        found = self.pattern.search(text)
        return found.group(0) if found else None


def _rule(key: str, label: str, pattern: str) -> Rule:
    return Rule(key=key, label=label, pattern=re.compile(pattern, re.IGNORECASE))


#: 强规则：单独命中即可判定为营销/引流。
#:
#: ⚠ 正则写法上的两个教训（都被测试锁住了）：
#: 1. 不能对 ``\b`` 加量词（``\b?`` 是非法正则，会直接编译失败）；
#: 2. 「加V」这类模式必须给裸字母 ``v`` 加排除断言，否则
#:    「参加VIP活动」会被误判成营销——中文里 ``加`` 太常见了。
STRONG_RULES: tuple[Rule, ...] = (
    _rule(
        "wechat",
        "加微信/加V",
        r"(加|＋|\+)\s*(我|一下|个)?\s*"
        r"(微信|weixin|wechat|v信|薇信|威信|vx|wx|v(?![a-zA-Z]))",
    ),
    _rule("wechat_contact", "微信联系", r"(微信|wechat|weixin)\s*(联系|咨询|同号)"),
    _rule("qq", "留QQ", r"(加|扣扣|企鹅)\s*qq|qq\s*[:：]?\s*\d{5,}"),
    _rule("dm", "私信/私聊联系", r"(私信|私聊|私我|滴滴我)"),
    _rule(
        "commerce",
        "代购/刷单/招代理",
        r"(代购|刷单|刷销量|刷信誉|招代理|招加盟|兼职日结|日结工资|招收学员)",
    ),
    _rule("shop", "引导下单", r"(点击|戳)\s*(链接|下方|下方链接|蓝字).{0,8}(购买|下单|领|抢)"),
    _rule("phone", "留手机号", r"1[3-9]\d{9}"),
    _rule("promo_code", "优惠码/内部价", r"(优惠码|折扣码|内部价|渠道价)"),
)

#: 弱规则：单独出现不算，连续命中两条以上才判定为营销。
#: 理由："优惠""包邮""下单"在正常的消费者吐槽里非常常见，
#: 单条命中就删会把真实的批评声音一起删掉——那正是本产品最不能犯的错。
WEAK_RULES: tuple[Rule, ...] = (
    _rule("discount", "优惠/低价", r"(优惠|低价|特价|折扣|促销|清仓|秒杀|包邮|现货|领券)"),
    _rule("buy", "下单/购买", r"(下单|购买|入手|拍下|链接)"),
    _rule("bonus", "红包/返现", r"(红包|返现|返利|抽奖|免费送)"),
)

#: 判定为疑似僵尸号 / 营销号的粉丝数上限。
#:
#: ⚠ 已知局限：模块 B1 原本要求"粉丝数 <5 且注册时间 <30 天"，
#: 但评论接口**不返回注册时间**（实测字段里没有 created_at 的用户维度信息）。
#: 所以这里只能退化为"粉丝数 < 5"，并在清洗报告与方法说明里如实标注
#: "仅依据粉丝数，未能校验账号年龄"。宁可承认局限，也不假装做到了。
ZOMBIE_FOLLOWER_THRESHOLD = 5


@dataclass(frozen=True, slots=True)
class AdMatch:
    """一次广告判定的结果，携带可解释信息。"""

    strength: str  # "strong" | "weak"
    labels: tuple[str, ...]
    matched: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "drop_ad_keywords:" + "+".join(self.labels)


def match_ad(text: str) -> AdMatch | None:
    """判断一条文本是否命中营销规则。

    Returns:
        命中时返回 `AdMatch`（含规则标签与命中片段）；未命中返回 None。

    >>> match_ad("加V信领取优惠") is not None
    True
    """
    strong_labels: list[str] = []
    matched: list[str] = []
    for rule in STRONG_RULES:
        found = rule.match(text)
        if found:
            strong_labels.append(rule.label)
            matched.append(found)

    if strong_labels:
        return AdMatch("strong", tuple(strong_labels), tuple(matched))

    weak_labels: list[str] = []
    for rule in WEAK_RULES:
        found = rule.match(text)
        if found:
            weak_labels.append(rule.label)
            matched.append(found)

    if len(weak_labels) >= 2:
        return AdMatch("weak", tuple(weak_labels), tuple(matched))
    return None


def is_suspected_bot(followers: int | None) -> bool:
    """疑似僵尸号判定（仅依据粉丝数，见上方局限说明）。"""
    return followers is not None and followers < ZOMBIE_FOLLOWER_THRESHOLD


def describe_rules() -> list[tuple[str, str, str]]:
    """给方法卡片/清洗报告用的规则清单：(类型, 标签, 说明)。"""
    rows: list[tuple[str, str, str]] = []
    for rule in STRONG_RULES:
        rows.append(("强规则", rule.label, f"正则 {rule.pattern.pattern}"))
    for rule in WEAK_RULES:
        rows.append(
            ("弱规则", rule.label, "需同时命中两条及以上才判定为营销")
        )
    return rows
