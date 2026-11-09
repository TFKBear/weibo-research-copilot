"""在真实采集结果上跑两种聚类，验证 K 选择、关键词与角色标签。

只读数据库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--k", type=int, default=None, help="强制指定 K（默认按轮廓系数建议）")
    parser.add_argument("--which", choices=("both", "discourse", "community"), default="both")
    args = parser.parse_args()

    from wrc.analyze.method_card import build_method_card
    from wrc.analyze.service import run_community, run_discourse
    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()  # 幂等；老库要靠它补出 analysis_* 表
    run_id = args.run or store.latest_run_id()
    if run_id is None:
        print("还没有采集记录。")
        return 2
    rows = store.load_analysis_rows(run_id)
    if not rows:
        print(f"会话 {run_id} 没有清洗后数据。")
        return 2
    print(f"会话 {run_id}：{len(rows)} 条清洗后评论\n")

    out_dir = ROOT / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("both", "discourse"):
        print("=== 话语聚类 ===")
        try:
            result, analysis_id = run_discourse(
                run_id, store=store, settings=settings, k=args.k,
                emit=lambda level, message: print(f"  [{level}] {message}"),
            )
        except ValueError as exc:
            print(f"  跳过：{exc}")
            return 2
        print(f"\nanalysis_id = {analysis_id}")
        print(result.render())
        print()
        print(build_method_card(
            module="cluster_discourse",
            method=result.method,
            limitations=[
                "聚类结果依赖句向量模型与随机初始化的 K-Means；已固定 random_state 以便复现。",
                "关键词基于词频比值，已去除停用词；网络新词可能被切分错误。",
                "代表性评论是距质心最近者，代表该簇的平均话语，不等于随机抽样。",
            ],
        ))
        (out_dir / f"cluster_discourse_{run_id}.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print()

    if args.which in ("both", "community"):
        print("=== 用户社群聚类 ===")
        try:
            community, analysis_id = run_community(
                run_id, store=store, settings=settings, k=args.k
            )
        except ValueError as exc:
            print(f"  跳过：{exc}")
            return 0
        print(f"\nanalysis_id = {analysis_id}")
        print(community.render())
        print(community.render())
        print()
        print(build_method_card(
            module="cluster_community",
            method=community.method,
            limitations=[
                "角色标签是按簇质心 z 值生成的描述性命名，不是对用户性质的判断。",
                "样本用户数与评论数有限，小样本下聚类结果不稳定，建议 300 条以上再解读。",
                "评论接口不提供账号年龄与原创/转发区分，特征集因此不完整。",
            ],
        ))
        (out_dir / f"cluster_community_{run_id}.json").write_text(
            json.dumps(community.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n结果已保存到：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
