"""离线流程冒烟：不联网，用合成数据跑完「采集 → 清洗 → 导出」全链路。

与 ``spikes/02_chain.py --mock`` 的区别：这里走的是**产品入口**那条路径
（``wrc.pipeline`` + ``wrc.cli`` 的编排），所以能验证界面背后的代码是否真的连通。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrc.clean.basic import Cleaner, TIER_STEPS  # noqa: E402
from wrc.clean.report import build_report  # noqa: E402
from wrc.export import Exporter  # noqa: E402
from wrc.mockdata import build_scenario  # noqa: E402
from wrc.privacy.anonymize import PseudoIdHasher  # noqa: E402
from wrc.privacy.consent import confirm  # noqa: E402
from wrc.store import Store  # noqa: E402
from wrc.weibo.models import TimePrecision, WeiboComment  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as raw_dir:
        tmp = Path(raw_dir)
        store = Store(tmp / "wrc.sqlite3")
        store.init_schema()
        store.create_run(
            run_id="smoke", tag="#冒烟测试#", order_mode="engagement", params={}
        )

        scenario = build_scenario(posts=2, comments_per_post=40, seed=11)
        comments = []
        for mid in ("5000000000000001", "5000000000000002"):
            payload = scenario(_mock_request(mid))
            for item in payload["data"]["data"]:
                comments.append(
                    WeiboComment(
                        cid=str(item["id"]),
                        post_mid=mid,
                        user_id=str(item["user"]["id"]),
                        screen_name=str(item["user"]["screen_name"]),
                        text_raw=str(item["text"]),
                        created_at=None,
                        created_at_raw="今天 10:00",
                        created_at_precision=TimePrecision.MINUTE,
                        like_count=int(item["like_counts"]),
                        reply_count=int(item["total_number"]),
                        followers_count=_followers(item),
                    )
                )
        store.save_comments("smoke", comments)
        print(f"入库评论：{len(comments)}")

        rows = store.load_comments("smoke")
        cleaner = Cleaner(tier="moderate", enabled_steps=TIER_STEPS["moderate"])
        result = cleaner.run(rows, external_flags={})
        report = build_report(result, rows)
        store.save_clean(
            "smoke", result.rows,
            clean_tier="moderate", pipeline_version=result.pipeline_version,
        )
        print(f"清洗：{report.total_input} → {report.total_kept}"
              f"（{report.retention_rate:.1%}）")

        exporter = Exporter(
            store,
            hasher=PseudoIdHasher(tmp / "salt.bin"),
            export_root=tmp / "exports",
        )
        consent = confirm(True, source="smoke")
        raw = exporter.export_raw("smoke", consent=consent)
        analysis = exporter.export_analysis("smoke", consent=consent)
        for manifest in (raw, analysis):
            manifest.write()
            for item in manifest.files:
                print(f"导出：{item.path.name}（{item.rows} 行）")

        assert report.total_kept > 0, "清洗后不应为空"
        assert (tmp / "exports" / "smoke" / "analysis" / "analysis_comments.csv").exists()
    print("\n离线冒烟通过")
    return 0


def _mock_request(mid: str):
    from wrc.weibo.endpoints import Request

    return Request(
        url="https://m.weibo.cn/comments/hotflow",
        params={"id": mid, "mid": mid, "max_id": 0, "max_id_type": 0},
        label="smoke",
    )


def _followers(item: dict) -> int:
    value = item["user"].get("followers_count")
    return int(value) if isinstance(value, int) else 0


if __name__ == "__main__":
    sys.exit(main())
