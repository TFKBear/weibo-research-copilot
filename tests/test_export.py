"""导出层测试。

最关键的两条：导出的 CSV 里**绝不能**出现可识别字段；分析层必须带上清洗决策。
"""

from __future__ import annotations

import csv

import pytest

from wrc.errors import ConfigProblem
from wrc.export import ANALYSIS_COMMENT_COLUMNS, Exporter, sha256_of
from wrc.privacy.anonymize import PseudoIdHasher
from wrc.privacy.consent import confirm
from wrc.store import Store
from wrc.weibo.models import TimePrecision, WeiboComment


def _comment(cid: str, text: str, user_id: str = "998877") -> WeiboComment:
    return WeiboComment(
        cid=cid,
        post_mid="p1",
        user_id=user_id,
        screen_name="张三",
        text_raw=text,
        created_at=None,
        created_at_raw="05-12",
        created_at_precision=TimePrecision.DAY,
        like_count=3,
        reply_count=0,
        followers_count=120,
    )


@pytest.fixture()
def populated_store(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    store.create_run(run_id="run-1", tag="#延迟退休#", order_mode="engagement", params={"a": 1})
    store.save_comments("run-1", [_comment("c1", "支持渐进式"), _comment("c2", "反对")])
    store.save_clean(
        "run-1",
        [
            {"cid": "c1", "post_mid": "p1", "cleaned_text": "支持渐进式", "kept": True,
             "drop_reasons": []},
            {"cid": "c2", "post_mid": "p1", "cleaned_text": "反对", "kept": False,
             "drop_reasons": ["drop_pure_emoji"]},
        ],
        clean_tier="light",
        pipeline_version="basic-0.1.0",
    )
    return store


def test_export_raw_writes_anonymized_csv(populated_store, tmp_path):
    exporter = Exporter(
        populated_store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_raw("run-1", consent=confirm(True))

    assert len(manifest.files) == 2  # 评论 + 帖子
    comments_file = next(f for f in manifest.files if f.path.name == "raw_comments.csv")
    assert comments_file.rows == 2

    raw_bytes = comments_file.path.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf"), "Excel 需要 BOM 才能正确显示中文"

    with comments_file.path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = set(rows[0].keys())
    assert "user_id" not in header
    assert "screen_name" not in header
    assert "profile_url" not in header
    assert "location" not in header
    assert "user_pseudo_id" in header
    assert rows[0]["user_pseudo_id"]
    assert "998877" not in raw_bytes.decode("utf-8-sig")


def test_export_analysis_keeps_cleaning_decisions(populated_store, tmp_path):
    exporter = Exporter(
        populated_store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_analysis("run-1", consent=confirm(True))
    file = manifest.files[0]
    assert set(ANALYSIS_COMMENT_COLUMNS) <= set(file.path.read_text(encoding="utf-8-sig").splitlines()[0].split(","))

    with file.path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["cid"]: row for row in csv.DictReader(handle)}
    assert rows["c1"]["kept"] == "true"
    assert rows["c2"]["kept"] == "false"
    assert rows["c2"]["drop_reasons"] == "drop_pure_emoji"
    assert rows["c1"]["clean_tier"] == "light"


def test_export_analysis_without_cleaning_is_refused(tmp_path):
    store = Store(tmp_path / "wrc.sqlite3")
    store.init_schema()
    store.create_run(run_id="r2", tag="#t#", order_mode="engagement", params={})
    store.save_comments("r2", [_comment("c9", "x")])
    exporter = Exporter(
        store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    with pytest.raises(ConfigProblem):
        exporter.export_analysis("r2", consent=confirm(True))


def test_manifest_records_checksums_and_consent(populated_store, tmp_path):
    exporter = Exporter(
        populated_store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_raw("run-1", consent=confirm(True), method={"检索排序": "互动"})
    path = manifest.write()
    payload = path.read_text(encoding="utf-8")
    assert "sha256" in payload
    assert manifest.consent is not None
    assert manifest.consent.statement_version

    for item in manifest.files:
        assert item.sha256 == sha256_of(item.path)
        assert item.rows >= 0


def test_sha256_is_stable(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("内容", encoding="utf-8")
    assert sha256_of(target) == sha256_of(target)


# --------------------------------------------------------------------------- #
# 交叉表导出：k-匿名抑制必须发生在数据层
# --------------------------------------------------------------------------- #
def _seed_analyzed(store: Store) -> None:
    """种子数据：两个簇，其中簇1 太小（整组抑制），簇0 内有一个小单元格。"""
    from wrc.weibo.models import TimePrecision, WeiboComment

    comments = [
        WeiboComment(
            cid=f"c{index}", post_mid="p1", user_id=f"u{index}",
            screen_name="某人", text_raw=f"评论{index}", created_at=None,
            created_at_raw="05-12", created_at_precision=TimePrecision.DAY,
        )
        for index in range(12)
    ]
    store.save_comments("run-1", comments)
    store.save_clean(
        "run-1",
        [
            {"cid": f"c{index}", "post_mid": "p1", "cleaned_text": f"评论{index}",
             "kept": True, "drop_reasons": []}
            for index in range(12)
        ],
        clean_tier="light",
        pipeline_version="basic-0.2.0",
    )
    store.save_analysis_run(
        analysis_id="a-sent", run_id="run-1", module="sentiment",
        params={}, method={}, summary={"_rendered": "x"},
    )
    # 前 10 条：簇 0（正面 8 / 负面 2）；后 2 条：簇 1
    items = [
        {"cid": f"c{index}", "sentiment": "positive" if index < 8 else "negative"}
        for index in range(12)
    ]
    store.save_sentiment_items("a-sent", items)

    store.save_analysis_run(
        analysis_id="a-clu", run_id="run-1", module="cluster_discourse",
        params={}, method={}, summary={"_rendered": "x"},
    )
    store.save_discourse_clusters("a-clu", [
        {"cid": f"c{index}", "cluster": 0 if index < 10 else 1}
        for index in range(12)
    ])


def test_export_analysis_writes_crosstab_with_suppression(populated_store, tmp_path):
    _seed_analyzed(populated_store)
    exporter = Exporter(
        populated_store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_analysis("run-1", consent=confirm(True))
    item = next(
        (f for f in manifest.files
         if f.path.name == "crosstab_discourse_sentiment.csv"),
        None,
    )
    assert item is not None, "分析层导出应包含交叉表"
    assert "抑制" in item.description

    import csv

    with item.path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_group = {row["分组"]: row for row in rows}
    # 簇0：正面 8 保留、负面 2 抑制
    assert by_group["簇0"]["positive"] == "8"
    assert by_group["簇0"]["negative"] == "—"
    # 簇1：整组 2 条 < 5 → 全部抑制，但总数照实写
    assert by_group["簇1"]["该组总样本"] == "2"
    assert by_group["簇1"]["positive"] == "—"


def test_crosstab_export_skipped_without_both_modules(populated_store, tmp_path):
    """只跑了情感、没跑聚类时，不该生成一张空交叉表。"""
    from wrc.weibo.models import TimePrecision, WeiboComment

    populated_store.save_comments("run-1", [
        WeiboComment(cid="c9", post_mid="p1", user_id="u9", screen_name="某人",
                     text_raw="只有文本", created_at=None, created_at_raw="05-12",
                     created_at_precision=TimePrecision.DAY)
    ])
    populated_store.save_clean(
        "run-1",
        [{"cid": "c9", "post_mid": "p1", "cleaned_text": "只有文本",
          "kept": True, "drop_reasons": []}],
        clean_tier="light", pipeline_version="basic-0.2.0",
    )
    populated_store.save_analysis_run(
        analysis_id="a2", run_id="run-1", module="sentiment",
        params={}, method={}, summary={"_rendered": "x"},
    )
    populated_store.save_sentiment_items("a2", [{"cid": "c9", "sentiment": "positive"}])

    exporter = Exporter(
        populated_store,
        hasher=PseudoIdHasher(tmp_path / "salt.bin"),
        export_root=tmp_path / "exports",
    )
    manifest = exporter.export_analysis("run-1", consent=confirm(True))
    names = {f.path.name for f in manifest.files}
    assert "crosstab_discourse_sentiment.csv" not in names


# --------------------------------------------------------------------------- #
# 方法说明的局限段要跟着采集口径走（不能写死"热门评论流"）
# --------------------------------------------------------------------------- #
def _method_note(tmp_path, params):
    from wrc.export import write_method_note

    path = tmp_path / "method_note.md"
    write_method_note(
        path,
        run={"run_id": "r1", "tag": "#延迟退休#", "created_at": "2026-10-25T23:23:35"},
        params=params,
        cleaning=None,
        manifest_rows={},
    )
    return path.read_text(encoding="utf-8")


def test_method_note_limitations_follow_hot_sort(tmp_path):
    text = _method_note(tmp_path, {"评论排序": "hot"})
    assert "热门评论流" in text
    # 热门流的时间不单调——这点必须说，否则研究者会以为设了窗口就采全了
    assert "时间不单调" in text


def test_method_note_limitations_follow_time_sort(tmp_path):
    text = _method_note(tmp_path, {"评论排序": "time"})
    assert "热门评论流" not in text
    assert "窗口早期的评论可能采不足" in text
    assert "不要直接照抄设定的窗口" in text


def test_method_note_warns_about_keyword_mode(tmp_path):
    text = _method_note(tmp_path, {"检索口径": "关键词全文检索（q=关键词）"})
    assert "关键词全文" in text
    assert "不等于话题参与者" in text
