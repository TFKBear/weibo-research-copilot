"""跨话题对比（P3）。

用法：

```bash
python tools/run_compare.py                 # 自动挑最近若干次 live 采集
python tools/run_compare.py --runs A,B      # 指定会话
python tools/run_compare.py --all           # 连 mock 会话一起看（会有强烈告警）
```

为什么默认排除 mock：mock 是内置样例数据，用来跑通流程的。
把它和真实采集画在同一张图上，读者分不出来，结论就废了。

为什么对比结果不写库：它是**派生视图**——随时可由现有采集重算，
且不产生新的分析标签。写进去只会让"有多少次分析"这个问题变模糊。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None, help="逗号分隔的采集会话 ID")
    parser.add_argument("--all", action="store_true", help="包含 mock 会话")
    parser.add_argument("--limit", type=int, default=6, help="最多比较几次采集")
    args = parser.parse_args()

    from wrc.analyze import charts
    from wrc.analyze.service import run_comparison
    from wrc.config import get_settings
    from wrc.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    store.init_schema()

    run_ids = (
        [item.strip() for item in args.runs.split(",") if item.strip()]
        if args.runs else None
    )
    result, skipped = run_comparison(
        store, run_ids=run_ids, include_mock=args.all, limit=args.limit
    )
    for reason in skipped:
        print(f"跳过 {reason}")

    if len(result.profiles) < 2:
        print("可对比的采集不足 2 次。")
        print("提示：先采集第二个话题，例如")
        print('  python -m wrc.cli run --tag "#另一个话题#" --auto-select 2 --comments 50 --consent')
        return 2

    print(result.render())

    out_dir = settings.export_dir / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compare.md"
    path.write_text("# 跨话题对比\n\n```\n" + result.render() + "\n```\n", encoding="utf-8")
    print()
    print(f"报告：{path}")

    for chart in charts.render_all(out_dir, compare=result):
        print(f"图表：{chart}")

    try:
        from wrc.analyze.docx_export import markdown_to_docx

        docx_path = markdown_to_docx(
            path.read_text(encoding="utf-8"), out_dir / "compare.docx", base_dir=out_dir
        )
        print(f"Word：{docx_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Word 导出失败：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
