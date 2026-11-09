"""可解释清洗管线的「轻度」档实现（第 1 步交付范围）。

产品宪法第 2 条：清洗的每一步必须对用户可见、可调、可关闭。
因此这里的每个步骤都是**独立、可命名、可计数、可关闭**的，
而不是一个笼统的 ``clean()`` 函数。

本文件当前只实现模块 B1 的「轻度」档所需步骤：

1. ``dedupe_exact``  精确重复（同帖内文本完全一致，仅忽略空白与大小写差异）
2. ``drop_empty``    清洗后为空
3. ``drop_pure_emoji`` 纯表情（表情 token 与标点之外无实义内容）
4. ``drop_pure_link``  纯链接 / 「网页链接」
5. ``drop_pure_mention`` 纯 @ 提及

HTML 标签剥离发生在解析层（``wrc.weibo.parsing.strip_html``），
因为它同时承担「把表情 <img alt="[doge]"> 还原成 token」的语义工作，
若放到清洗层会被误当成标签一同剥掉。清洗报告里会如实说明这一点。

「中度 / 重度」档（广告规则表、僵尸号、BERT 分类、协同行为过滤）属于
模块 B 的完整交付，按交付节奏在第 2 步实现，不在本文件里提前开工。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import rules

PIPELINE_VERSION = "basic-0.2.0"

# --------------------------------------------------------------------------- #
# 正则
# --------------------------------------------------------------------------- #
_EMOJI_TOKEN_RE = re.compile(r"\[[^\[\]\s]{1,12}\]")
_UNICODE_EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\u2b00-\u2bff"
    "\ufe0f\u20e3"
    "]"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_LINK_PLACEHOLDER = "网页链接"
_MENTION_RE = re.compile(r"@[A-Za-z0-9_\-\u4e00-\u9fa5]{1,30}")
#: 判定「只剩标点/空白」时被视为无实义的字符
_PUNCT_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class StepSpec:
    """一个清洗步骤的元信息，用于报告与方法卡片。"""

    key: str
    label: str
    description: str


STEPS: tuple[StepSpec, ...] = (
    StepSpec(
        "dedupe_exact",
        "精确去重",
        "同一帖子下，去除空白差异后文本完全相同的重复条目；保留接口最先返回的那一条。",
    ),
    StepSpec(
        "drop_empty",
        "去除空文本",
        "清洗后不含任何字符的条目（例如只有零宽字符）。",
    ),
    StepSpec(
        "drop_pure_emoji",
        "去除纯表情",
        "除表情 token / emoji 与标点外没有任何实义内容的条目。",
    ),
    StepSpec(
        "drop_pure_link",
        "去除纯链接",
        "整条只包含链接或「网页链接」占位符的条目。",
    ),
    StepSpec(
        "drop_pure_mention",
        "去除纯 @",
        "整条只包含 @ 提及的条目。",
    ),
    StepSpec(
        "drop_ad_keywords",
        "广告关键词过滤",
        "命中强规则（留联系方式、引导下单等）或同时命中两条以上弱规则"
        "（优惠/下单/返现）的条目，每条都记录命中了哪条规则。",
    ),
    StepSpec(
        "drop_suspected_bots",
        "疑似僵尸号过滤",
        "粉丝数 < 5 的评论者。⚠ 仅依据粉丝数：评论接口不返回注册时间，"
        "无法校验账号年龄，该局限会写进清洗报告。",
    ),
    StepSpec(
        "ai_marketing",
        "AI 营销话术识别",
        "与内置营销模板的语义相似度超过阈值的条目，记录相似度与最接近的模板。",
    ),
    StepSpec(
        "ai_coordinated",
        "AI 协同行为识别",
        "与数据集内其他评论高度相似、构成簇的条目（疑似模板化投放），记录簇内条数。",
    ),
)

#: 各档位默认启用的步骤。
#:
#: 轻度：只做「明显不是内容」的清理。
#: 中度：加上可枚举的广告与疑似僵尸号（全部基于规则，可解释到具体规则）。
#: 重度：再加上 AI 两类判定（需要模型；没有模型时该档不可用，不会静默降级）。
TIER_STEPS: dict[str, tuple[str, ...]] = {
    "none": (),
    "light": (
        "dedupe_exact",
        "drop_empty",
        "drop_pure_emoji",
        "drop_pure_link",
        "drop_pure_mention",
    ),
    "moderate": (
        "dedupe_exact",
        "drop_empty",
        "drop_pure_emoji",
        "drop_pure_link",
        "drop_pure_mention",
        "drop_ad_keywords",
    ),
    "heavy": (
        "dedupe_exact",
        "drop_empty",
        "drop_pure_emoji",
        "drop_pure_link",
        "drop_pure_mention",
        "drop_ad_keywords",
        "ai_marketing",
        "ai_coordinated",
    ),
}

#: 不随档位自动启用、必须用户显式勾选的步骤。
#:
#: ``drop_suspected_bots`` 放在这里是一个刻意的产品判断：
#: 需求文档要求的是「粉丝数 <5 **且** 注册 <30 天」，但评论接口根本不返回
#: 注册时间。只拿粉丝数当判据，会把大量真实但低互动的新用户、退休用户、
#: 沉默用户一起删掉——那恰恰是「可解释清洗优先于极限去噪」要防的事。
#: 所以它必须是可选项，并且默认关闭，界面上直接写明可能误伤。
OPT_IN_STEPS: dict[str, str] = {
    "drop_suspected_bots": "过滤疑似僵尸号（粉丝数 < 5；仅依据粉丝数，可能误伤真实低互动用户）",
}

#: 给界面用的档位说明。GUI 直接展示这些文字，避免用户面对 "light/moderate/heavy"。
TIER_META: dict[str, dict[str, object]] = {
    "none": {
        "label": "不清洗",
        "summary": "保留全部原始评论，只做最基础的空白处理。",
        "needs_ai": False,
    },
    "light": {
        "label": "轻度",
        "summary": "去重 + 去除纯表情/纯链接/纯@。适合话语分析、表情符号研究。",
        "needs_ai": False,
    },
    "moderate": {
        "label": "中度（推荐）",
        "summary": "轻度 + 广告关键词过滤。每条都记录命中了哪条规则，可逐条复核。",
        "needs_ai": False,
    },
    "heavy": {
        "label": "重度（需 AI）",
        "summary": "中度 + AI 识别营销话术与协同行为。需要下载模型，判定结果附带相似度。",
        "needs_ai": True,
    },
}


@dataclass(slots=True)
class CleaningResult:
    """清洗结果。``rows`` 与输入行一一对应，顺序保持一致。"""

    tier: str
    pipeline_version: str
    rows: list[dict]
    step_removed: dict[str, int] = field(default_factory=dict)
    step_samples: dict[str, list[int]] = field(default_factory=dict)
    active_steps: tuple[str, ...] = ()

    @property
    def total_input(self) -> int:
        return len(self.rows)

    @property
    def total_kept(self) -> int:
        return sum(1 for row in self.rows if row["kept"])

    @property
    def total_removed(self) -> int:
        return self.total_input - self.total_kept

    @property
    def kept_rows(self) -> list[dict]:
        return [row for row in self.rows if row["kept"]]

    def restore(self, cid: str) -> bool:
        """人工判定「误杀」后恢复某条。

        返回是否确实发生了变化。这是模块 B2 要求的一键恢复能力的最小实现。
        """
        for row in self.rows:
            if row["cid"] == cid:
                if row["kept"]:
                    return False
                row["kept"] = True
                row["drop_reasons"] = []
                row["restored_by_user"] = True
                return True
        return False


# --------------------------------------------------------------------------- #
# 单步骤实现
# --------------------------------------------------------------------------- #
def normalize_for_dedupe(text: str) -> str:
    """去重用的规范化：只压空白与统一大小写，不改动标点与表情。

    刻意保守：改动越少，越容易在论文里解释「我到底去重掉了什么」。
    """
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def is_pure_emoji(text: str) -> bool:
    stripped = _EMOJI_TOKEN_RE.sub("", text or "")
    stripped = _UNICODE_EMOJI_RE.sub("", stripped)
    stripped = _PUNCT_STRIP_RE.sub("", stripped)
    if stripped:
        return False
    return bool(_EMOJI_TOKEN_RE.search(text or "") or _UNICODE_EMOJI_RE.search(text or ""))


def is_pure_link(text: str) -> bool:
    stripped = _URL_RE.sub("", text or "").replace(_LINK_PLACEHOLDER, "")
    return not _PUNCT_STRIP_RE.sub("", stripped)


def is_pure_mention(text: str) -> bool:
    stripped = _MENTION_RE.sub("", text or "")
    if stripped == (text or ""):  # 压根没有 @
        return False
    return not _PUNCT_STRIP_RE.sub("", stripped)


def _clean_text(text: str) -> str:
    """轻量规范化：压缩空白，去掉首尾空白。不改写用词。"""
    return re.sub(r"[ \t\u00a0]+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Cleaner:
    """按档位组合步骤，并保证每次移除都被记账。"""

    tier: str = "light"
    dedupe_scope: str = "post"
    """``post``：仅在帖子内部去重（默认，风险最低）；``global``：跨帖去重。"""
    enabled_steps: Sequence[str] | None = None
    """为 ``None`` 时按档位启用；显式传入可关闭任意步骤（产品宪法第 2 条）。"""

    _flags: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    """外部（AI）传入的判定结果，run() 时填充。"""

    def active_steps(self) -> tuple[StepSpec, ...]:
        default_for_tier = TIER_STEPS
        keys = (
            self.enabled_steps
            if self.enabled_steps is not None
            else default_for_tier.get(self.tier, ())
        )
        known = {spec.key: spec for spec in STEPS}
        unknown = [key for key in keys if key not in known]
        if unknown:
            raise ValueError(f"未知清洗步骤: {unknown}")
        return tuple(known[key] for key in keys)

    # ------------------------------------------------------------------ #
    def run(
        self,
        rows: Iterable[dict],
        *,
        external_flags: dict[str, Sequence[str]] | None = None,
    ) -> CleaningResult:
        """执行清洗。

        Args:
            rows: 每条至少包含 ``cid`` / ``post_mid`` / ``text_raw``；
                可选 ``followers_count``。
            external_flags: 由外部（AI 模块）算好的判定，形如
                ``{cid: ["ai_marketing:0.88", "ai_coordinated:簇内4条"]}``。
                把 AI 结果作为"已经算好的罪名表"传进来，而不是让 Cleaner 去调模型，
                这样清洗逻辑保持纯函数、可离线测试。
        """
        source = list(rows)
        self._flags = {
            str(cid): list(reasons) for cid, reasons in (external_flags or {}).items()
        }
        steps = self.active_steps()
        result = CleaningResult(
            tier=self.tier,
            pipeline_version=PIPELINE_VERSION,
            rows=[
                {
                    "cid": str(row["cid"]),
                    "post_mid": str(row.get("post_mid") or ""),
                    "cleaned_text": _clean_text(str(row.get("text_raw") or "")),
                    "kept": True,
                    "drop_reasons": [],
                }
                for row in source
            ],
            active_steps=tuple(spec.key for spec in steps),
        )
        for spec in steps:
            removed = self._apply(spec, result.rows, source)
            result.step_removed[spec.key] = len(removed)
            result.step_samples[spec.key] = removed
        return result

    def _apply(
        self, spec: StepSpec, rows: list[dict], source: list[dict]
    ) -> list[int]:
        handler = getattr(self, f"_step_{spec.key}")
        return handler(rows, source)

    # ------------------------------------------------------------------ #
    def _drop(
        self,
        rows: list[dict],
        index: int,
        spec: str,
        *,
        reasons: Sequence[str] | None = None,
    ) -> None:
        row = rows[index]
        if not row["kept"]:
            return
        row["kept"] = False
        row["drop_reasons"] = list(reasons) if reasons else [spec]
        row["dropped_by"] = spec

    def _step_dedupe_exact(self, rows: list[dict], _source: list[dict]) -> list[int]:
        seen: dict[tuple[str, str], int] = {}
        removed: list[int] = []
        for index, row in enumerate(rows):
            if not row["kept"]:
                continue
            scope_key = row["post_mid"] if self.dedupe_scope == "post" else ""
            key = (scope_key, normalize_for_dedupe(row["cleaned_text"]))
            if key in seen:
                self._drop(rows, index, "dedupe_exact")
                removed.append(index)
            else:
                seen[key] = index
        return removed

    def _step_drop_empty(self, rows: list[dict], _source: list[dict]) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if row["kept"] and not row["cleaned_text"]:
                self._drop(rows, index, "drop_empty")
                removed.append(index)
        return removed

    def _step_drop_pure_emoji(self, rows: list[dict], _source: list[dict]) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if row["kept"] and is_pure_emoji(row["cleaned_text"]):
                self._drop(rows, index, "drop_pure_emoji")
                removed.append(index)
        return removed

    def _step_drop_pure_link(self, rows: list[dict], _source: list[dict]) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if row["kept"] and is_pure_link(row["cleaned_text"]):
                self._drop(rows, index, "drop_pure_link")
                removed.append(index)
        return removed

    def _step_drop_pure_mention(self, rows: list[dict], _source: list[dict]) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if row["kept"] and is_pure_mention(row["cleaned_text"]):
                self._drop(rows, index, "drop_pure_mention")
                removed.append(index)
        return removed

    # ---------------- 中度档 ---------------- #
    def _step_drop_ad_keywords(self, rows: list[dict], _source: list[dict]) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if not row["kept"]:
                continue
            hit = rules.match_ad(row["cleaned_text"])
            if hit is not None:
                self._drop(rows, index, "drop_ad_keywords",
                           reasons=[f"{hit.reason}｜{hit.strength}"])
                removed.append(index)
        return removed

    def _step_drop_suspected_bots(
        self, rows: list[dict], source: list[dict]
    ) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if not row["kept"]:
                continue
            followers = source[index].get("followers_count")
            if isinstance(followers, str):
                from ..weibo.parsing import to_count

                followers = to_count(followers)
            if rules.is_suspected_bot(followers):
                self._drop(
                    rows, index, "drop_suspected_bots",
                    reasons=[f"drop_suspected_bots:粉丝{followers}"],
                )
                removed.append(index)
        return removed

    # ---------------- 重度档（AI） ---------------- #
    def _step_ai_marketing(self, rows: list[dict], _source: list[dict]) -> list[int]:
        return self._drop_by_flag_prefix(rows, "ai_marketing")

    def _step_ai_coordinated(self, rows: list[dict], _source: list[dict]) -> list[int]:
        return self._drop_by_flag_prefix(rows, "ai_coordinated")

    def _drop_by_flag_prefix(self, rows: list[dict], prefix: str) -> list[int]:
        removed = []
        for index, row in enumerate(rows):
            if not row["kept"]:
                continue
            reasons = [r for r in self._flags.get(row["cid"], []) if r.startswith(prefix)]
            if reasons:
                self._drop(rows, index, prefix, reasons=reasons)
                removed.append(index)
        return removed


def describe_steps(steps: Sequence[StepSpec]) -> list[tuple[str, str]]:
    """给方法卡片与清洗报告用的步骤说明。"""
    return [(spec.label, spec.description) for spec in steps]
