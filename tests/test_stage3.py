"""P1 分析层测试（情感 + 聚类 + 方法卡片）。

重点测**判定逻辑与可解释性**，不测模型本身的效果——
模型效果无法用单元测试保证，只能靠 tools/ 下的实机探针。
"""

from __future__ import annotations

import numpy as np
import pytest

from wrc.analyze.cluster import cluster_discourse, suggest_k, tokenize
from wrc.analyze.community import _role_for, build_feature_matrix, cluster_communities
from wrc.analyze.method_card import build_method_card
from wrc.analyze.sentiment import (
    AgreementReport,
    SentimentItem,
    _parse_labels,
)
from wrc.llm.client import parse_json_loose


# --------------------------------------------------------------------------- #
# LLM 输出解析
# --------------------------------------------------------------------------- #
def test_parse_json_loose_plain():
    assert parse_json_loose('{"a": 1}') == {"a": 1}


def test_parse_json_loose_with_code_fence():
    text = '```json\n{"items": [{"i": 1, "s": "negative"}]}\n```'
    assert parse_json_loose(text)["items"][0]["s"] == "negative"


def test_parse_json_loose_with_preamble():
    text = '好的，结果如下：\n{"items": []}\n以上。'
    assert parse_json_loose(text) == {"items": []}


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #
def test_parse_labels_maps_indices():
    payload = {"items": [{"i": 1, "s": "negative", "intensity": 0.8},
                         {"i": 2, "s": "neutral", "intensity": 0.3}]}
    parsed = _parse_labels(payload, expected=2)
    assert parsed[1] == ("negative", 0.8)
    assert parsed[2] == ("neutral", 0.3)


def test_parse_labels_rejects_unknown_label_and_bad_index():
    payload = {"items": [{"i": 1, "s": "angry"}, {"i": 99, "s": "positive"},
                         {"i": "x", "s": "positive"}]}
    assert _parse_labels(payload, expected=2) == {}


def test_parse_labels_clamps_intensity():
    payload = {"items": [{"i": 1, "s": "positive", "intensity": 5}]}
    assert _parse_labels(payload, expected=1)[1][1] == 1.0


# --------------------------------------------------------------------------- #
# 信度报告
# --------------------------------------------------------------------------- #
def test_agreement_flags_above_threshold():
    report = AgreementReport(sample_size=100, compared=20, disagree=4)
    assert report.rate == pytest.approx(0.2)
    assert report.flagged is True


def test_agreement_not_flagged_below_threshold():
    report = AgreementReport(sample_size=100, compared=20, disagree=2)
    assert report.rate == pytest.approx(0.1)
    assert report.flagged is False


def test_agreement_empty_is_not_flagged():
    assert AgreementReport().flagged is False


def test_sentiment_item_prefers_llm_label_and_tracks_agreement():
    item = SentimentItem(cid="1", post_mid="p", text="t",
                         llm_label="negative", local_label="positive")
    assert item.label == "negative"
    assert item.agrees is False
    missing = SentimentItem(cid="2", post_mid="p", text="t", llm_label="neutral")
    assert missing.agrees is None
    assert missing.label == "neutral"


# --------------------------------------------------------------------------- #
# 分词与关键词
# --------------------------------------------------------------------------- #
def test_tokenize_drops_stopwords_and_single_chars():
    tokens = tokenize("这个车的底盘用料确实不错")
    assert "这个" not in tokens          # 停用词
    assert "的" not in tokens
    assert "底盘" in tokens
    assert all(len(t) >= 2 or t.isascii() for t in tokens)


def test_tokenize_keeps_emoji_out_of_keywords():
    tokens = tokenize("支持小米[good][doge]")
    assert all("[" not in t for t in tokens)
    assert "小米" in tokens


# --------------------------------------------------------------------------- #
# 用户特征
# --------------------------------------------------------------------------- #
def test_build_feature_matrix_aggregates_per_user():
    rows = [
        {"user_pseudo_id": "u1", "cleaned_text": "支持小米[good]", "like_count": 10,
         "followers_count": 100, "post_mid": "p1"},
        {"user_pseudo_id": "u1", "cleaned_text": "再来一条", "like_count": 20,
         "followers_count": 100, "post_mid": "p2"},
        {"user_pseudo_id": "u2", "cleaned_text": "反对", "like_count": 0,
         "followers_count": 5, "post_mid": "p1"},
    ]
    users, matrix = build_feature_matrix(rows)
    assert set(users) == {"u1", "u2"}
    u1 = matrix[users.index("u1")]
    assert u1[0] == 2          # 评论数
    assert u1[5] == 2          # 参与帖子数
    import math

    assert u1[4] == pytest.approx(math.log10(101), abs=1e-6)  # log10(1+100)


def test_role_naming_is_descriptive():
    role = _role_for({"评论数": 1.8, "平均字数": 1.2, "表情率": -0.5,
                      "平均获赞": 0.1, "粉丝数(log)": -1.0, "参与帖子数": 0.3})
    assert "高频参与" in role
    assert "长文本论述" in role


def test_role_falls_back_to_ordinary():
    assert _role_for({name: 0.0 for name in
                      ("评论数", "平均字数", "表情率", "平均获赞",
                       "粉丝数(log)", "参与帖子数")}) == "普通参与型"


# --------------------------------------------------------------------------- #
# 聚类
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """把文本映射到固定向量，用于离线测聚类流程。"""

    model_name = "fake"
    labels: list[str] = []

    def __init__(self, vectors):
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def encode(self, texts, on_progress=None):
        from wrc.ai.embedder import EmbeddingResult

        subset = self._vectors[: len(texts)]
        norms = np.linalg.norm(subset, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return EmbeddingResult(
            vectors=subset / norms, dim=subset.shape[1], elapsed_s=0.0,
            model_name="fake",
        )


def _blobs() -> tuple[list[dict], _FakeEmbedder]:
    rows, vectors = [], []
    for index in range(12):
        group = index // 4
        base = np.zeros(6, dtype=np.float32)
        base[group] = 1.0
        rows.append({"cid": str(index), "cleaned_text": f"第{group}类话语{index}"})
        vectors.append(base)
    return rows, _FakeEmbedder(vectors)


def test_suggest_k_finds_separated_blobs():
    _, embedder = _blobs()
    encoded = embedder.encode([f"x{i}" for i in range(12)])
    curve = suggest_k(encoded.vectors, k_max=6)
    assert curve
    best = max(curve, key=lambda pair: pair[1])[0]
    assert best == 3


def test_cluster_discourse_produces_keywords_and_representatives():
    rows, embedder = _blobs()
    # k_max=4：假数据只有 3 个不同向量，扫更多 K 只会触发 KMeans 收敛告警
    result = cluster_discourse(rows, embedder=embedder, k=3, k_max=4)
    assert result.k == 3
    assert len(result.sizes) == 3
    assert sum(result.sizes.values()) == 12
    assert all(len(result.representatives.get(c, [])) > 0 for c in result.sizes)


def test_cluster_discourse_rejects_tiny_samples():
    with pytest.raises(ValueError):
        cluster_discourse([{"cid": "1", "cleaned_text": "x"}], embedder=_FakeEmbedder([[1.0]]))


def test_cluster_communities_produces_roles():
    rows = []
    for index in range(15):
        rows.append(
            {
                "user_pseudo_id": f"u{index}",
                "cleaned_text": "评论" * (index % 5 + 1),
                "like_count": index,
                "followers_count": index * 10,
                "post_mid": f"p{index % 3}",
            }
        )
    result = cluster_communities(rows, k=2)
    assert result.k == 2
    assert len(result.user_labels) == 15
    assert set(result.roles) == set(result.user_sizes)
    assert result.coords


# --------------------------------------------------------------------------- #
# 方法卡片
# --------------------------------------------------------------------------- #
def test_method_card_has_paste_ready_paragraph():
    card = build_method_card(
        module="sentiment",
        method={
            "llm_model": "deepseek-v4.1-flash",
            "prompt_version": "sentiment-1.0.0",
            "sample_ratio": 0.05,
            "sample_cap": 200,
            "sample_floor": 30,
            "disagreement_threshold": 0.15,
            "local_model": "distilbert-sentiment-zh",
        },
        cleaning={"tier": "moderate", "pipeline_version": "basic-0.2.0",
                  "total_input": 50, "total_kept": 46, "retention_rate": 0.92},
        run={"run_id": "run-1", "tag": "#小米SU7#"},
        limitations=["样本来自热门评论流"],
    )
    assert "方法卡片 · 情感分析" in card
    assert "deepseek-v4.1-flash" in card
    assert "可直接引用的一段话" in card
    assert "#小米SU7#" in card
    assert "热门评论流" in card
