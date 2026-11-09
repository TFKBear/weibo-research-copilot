"""立场维度测试（研究者选定的 B 方案）。

要守住的三件事：
1. 立场必须锚定到具体政策——提示词里要带话题名，否则模型无从判断"支持什么"；
2. 「无关」必须单独成类，且**不计入支持率/反对率**的分母；
3. 立场与情感并列存储、互不覆盖（同一行既有 sentiment 又有 stance）。
"""

from __future__ import annotations

import json

import pytest

from wrc.analyze.stance import (
    STANCE_LABELS,
    analyse_stance,
    stance_of,
)
from wrc.store import Store


class _FakeClient:
    """按预设返回 JSON；可指定某一批抛异常。

    ``retry_labels`` 用于模拟"失败后二分重试"的返回——真实的模型
    在更小的批次上往往能正常给出 JSON，这也是二分重试有效的原因。
    """

    def __init__(self, labels: list[str], *, fail_batch: int | None = None,
                 confidences: list[float] | None = None,
                 retry_labels: list[str] | None = None,
                 retry_fails: bool = False) -> None:
        self.labels = labels
        self.fail_batch = fail_batch
        self.confidences = confidences or [0.8] * len(labels)
        self.retry_labels = retry_labels
        self.retry_fails = retry_fails
        self.calls: list[str] = []
        self.config = type("C", (), {"model": "fake-model"})()

    @property
    def stats(self) -> dict:
        """与真实 LlmClient 一致：这是 property，不是方法。

        早期假客户端把它写成方法，于是"`stats()` 被调用"这个 bug 在测试里
        完全看不见，直到跑真实数据才炸。假客户端必须贴着真实接口长。
        """
        return {"calls": len(self.calls)}

    def complete_json(self, *, system: str, user: str, max_tokens=None):
        self.calls.append(user)
        first_call = len(self.calls) == 1
        if self.fail_batch is not None and len(self.calls) - 1 == self.fail_batch:
            raise RuntimeError("模拟接口失败")
        if not first_call and self.retry_fails:
            raise RuntimeError("模拟重试仍然失败")
        labels = (
            self.labels if first_call or self.retry_labels is None
            else self.retry_labels
        )
        confidences = (
            self.confidences if first_call or self.retry_labels is None
            else [0.8] * len(labels)
        )
        items = []
        for index, label in enumerate(labels, start=1):
            items.append({
                "i": index,
                "stance": label,
                "confidence": confidences[index - 1] if index - 1 < len(confidences) else 0.8,
            })
        return {"items": items}


def _rows(n: int) -> list[dict]:
    return [
        {"cid": f"c{index}", "post_mid": "p1", "cleaned_text": f"评论内容 {index}"}
        for index in range(n)
    ]


# --------------------------------------------------------------------------- #
def test_prompt_is_anchored_to_the_policy():
    client = _FakeClient(["support"])
    analyse_stance(_rows(1), topic="#延迟退休#", llm_client=client)
    assert "#延迟退休#" in client.calls[0]
    assert "立场" in client.calls[0]


def test_distribution_and_per_post():
    labels = ["support", "oppose", "oppose", "irrelevant"]
    result = analyse_stance(_rows(4), topic="#t#", llm_client=_FakeClient(labels))
    assert result.distribution == {"support": 1, "oppose": 2, "irrelevant": 1}
    assert result.n_labelled == 4
    assert result.n_total == 4
    assert result.per_post["p1"]["oppose"] == 2


def test_irrelevant_is_reported_separately():
    """「无关」不能混进中立，否则支持率会被稀释。"""
    result = analyse_stance(
        _rows(3), topic="#t#", llm_client=_FakeClient(["support", "neutral", "irrelevant"])
    )
    text = result.render()
    assert "无关" in text
    assert "不计入" in text
    # 渲染里要提醒分母口径
    assert "分母" in text


def test_unknown_labels_are_dropped_not_coerced():
    """非法标签不能被硬塞进某一类；重试也拿不到就计为失败。"""
    client = _FakeClient(
        ["support", "maybe", "中立"],
        retry_labels=[],
        retry_fails=True,
    )
    result = analyse_stance(_rows(3), topic="#t#", llm_client=client)
    assert result.distribution.get("support") == 1
    assert result.distribution.get("neutral") == 1
    assert result.n_failed == 1


def test_batch_failure_is_recovered_by_bisection():
    """整批解析失败时二分重试，把数据救回来——而不是丢掉整批。

    这是真实事故的回归测试：25 批里有 9 批解析失败，
    整批丢弃等于丢 240 条（约一半样本）。
    """
    rows = _rows(25)
    client = _FakeClient(["oppose"] * 20, fail_batch=0, retry_labels=["oppose"])
    result = analyse_stance(rows, topic="#t#", llm_client=client)
    assert result.n_failed == 0  # 二分后全部救回
    assert result.n_labelled == 25
    assert result.distribution["oppose"] == 25
    assert len(client.calls) > 2  # 确实发生了重试


def test_single_item_that_always_fails_is_counted():
    rows = _rows(2)
    client = _FakeClient(["support"], fail_batch=0, retry_labels=[], retry_fails=True)
    result = analyse_stance(rows, topic="#t#", llm_client=client)
    assert result.n_failed == 2
    assert result.n_labelled == 0


def test_method_records_topic_and_prompt_version():
    result = analyse_stance(_rows(2), topic="#延迟退休#", llm_client=_FakeClient(["support", "oppose"]))
    method = result.method
    assert method["topic"] == "#延迟退休#"
    assert method["prompt_version"].startswith("stance-")
    assert method["labels"] == list(STANCE_LABELS)
    assert "分母" in method["note"]


def test_empty_rows_degrades():
    result = analyse_stance([], topic="#t#", llm_client=_FakeClient([]))
    assert result.n_total == 0
    assert "没有得到任何立场标签" in result.render()


def test_low_confidence_is_kept_but_visible():
    client = _FakeClient(["support"], confidences=[0.12])
    result = analyse_stance(_rows(1), topic="#t#", llm_client=client)
    assert result.items[0].confidence == 0.12


# --------------------------------------------------------------------------- #
def test_stance_roundtrip_in_store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    store.create_run(run_id="r1", tag="#t#", order_mode="engagement", params={})
    store.save_analysis_run(
        analysis_id="a-stance", run_id="r1", module="stance",
        params={"topic": "#t#"}, method={}, summary={"_rendered": "x"},
    )
    store.save_stance_items("a-stance", [
        {"cid": "c1", "stance": "support", "confidence": 0.9},
        {"cid": "c2", "stance": "oppose", "confidence": 0.7},
        {"cid": "c3", "stance": None, "confidence": None},  # 无标签不写入
    ])
    merged = store.load_comment_analysis("r1")
    assert merged["c1"]["stance"] == "support"
    assert merged["c1"]["stance_confidence"] == 0.9
    assert "c3" not in merged


def test_stance_and_sentiment_coexist(tmp_path):
    """两个维度并存：同一行既有情感又有立场，互不覆盖。"""
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    store.create_run(run_id="r1", tag="#t#", order_mode="engagement", params={})
    store.save_analysis_run(analysis_id="a-s", run_id="r1", module="sentiment",
                            params={}, method={}, summary={"_rendered": "x"})
    store.save_sentiment_items("a-s", [{"cid": "c1", "sentiment": "negative"}])
    store.save_analysis_run(analysis_id="a-t", run_id="r1", module="stance",
                            params={}, method={}, summary={"_rendered": "x"})
    store.save_stance_items("a-t", [{"cid": "c1", "stance": "support"}])

    merged = store.load_comment_analysis("r1")
    assert merged["c1"]["sentiment"] == "negative"
    assert merged["c1"]["stance"] == "support"


def test_merge_analysis_carries_stance():
    from wrc.analyze.service import _merge_analysis

    rows = [{"cid": "c1", "post_mid": "p1"}]
    merged = _merge_analysis(rows, {"c1": {"sentiment": "negative", "stance": "support"}})
    assert merged[0]["stance"] == "support"
    assert merged[0]["sentiment"] == "negative"


def test_stance_of_helper():
    assert stance_of({"stance": "oppose"}) == "oppose"
    assert stance_of({"stance": None}) is None
    assert stance_of({}) is None


# --------------------------------------------------------------------------- #
def test_local_label_normalisation_handles_five_class_model():
    """5 类模型（Very Negative…）必须折叠成三分类，否则"一致率"永远为 0。"""
    from wrc.analyze.sentiment import normalise_local

    class _Pred:
        label = "Very Negative"
        score = 0.4
        probs = {"Very Negative": 0.4, "Negative": 0.3, "Neutral": 0.2,
                 "Positive": 0.05, "Very Positive": 0.05}

    label, score, probs = normalise_local(_Pred())
    assert label == "negative"
    assert score == pytest.approx(0.7)
    assert set(probs) == {"positive", "neutral", "negative"}


def test_local_label_normalisation_passes_through_unknown_scheme():
    from wrc.analyze.sentiment import normalise_local

    class _Pred:
        label = "LABEL_7"
        score = 0.5
        probs = {"LABEL_7": 0.5, "LABEL_8": 0.5}

    label, _score, probs = normalise_local(_Pred())
    assert label == "LABEL_7"  # 不认识的体系不硬猜
    assert "LABEL_7" in probs


def test_report_has_stance_section():
    from wrc.analyze.report import build_analysis_report

    class _Shim:
        method = {"module": "stance", "topic": "#延迟退休#", "prompt_version": "stance-1.0.0"}

        def render(self) -> str:
            return "立场分布（对象：#延迟退休#，n=10）"

    text = build_analysis_report(
        run={"run_id": "r1", "tag": "#延迟退休#"},
        stance=_Shim(),
    )
    assert "## 三、立场分布" in text
    assert "支持 / 反对" in text
    # 必须提醒"情绪≠立场"，避免研究者把情感分布当立场分布
    assert "不要用情感分布代替立场分布" in text
    assert "方法卡片 · 立场" in text


def test_report_states_implementation_phase_framing():
    """研究主线定为「实施期」后，报告开头必须自己说清楚，而不是靠研究者记着。"""
    from wrc.analyze.report import build_analysis_report

    text = build_analysis_report(run={"run_id": "r1", "tag": "#延迟退休#"})
    assert "本报告的定位" in text
    assert "实施期" in text
    # 转折点扫描不能被读成事件研究
    assert "不做事件研究" in text
