"""生成可复现性附录（P2「研究日志正式版」）。

用法：

```bash
python tools/run_appendix.py                     # 最近一次采集
python tools/run_appendix.py --run <会话ID>
python tools/run_appendix.py --all              # 为所有会话各生成一份
python tools/run_appendix.py --no-hash          # 跳过模型/文件指纹（快，但附录不完整）
```

产物是 `data/exports/<会话ID>/appendix.md`，同时转成 `appendix.docx`
（Word 版方便直接贴进论文附录）。

**为什么可以随时重跑**：附录里的 SHA-256 会随文件内容变化。
数据重新导出过、报告重新生成过，就应该重跑一次附录，否则校验和对不上。
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
    parser.add_argument("--run", default=None, help="采集会话 ID；默认最近一次")
    parser.add_argument("--all", action="store_true", help="为所有会话各生成一份")
    parser.add_argument(
        "--no-hash", action="store_true",
        help="跳过模型与文件指纹（大模型文件哈希较慢）",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="在附录里写入本机路径（默认不写，避免流出到公开材料）",
    )
    args = parser.parse_args()

    from wrc.config import get_settings
    from wrc.reproducibility import write_for_run
    from wrc.store import Store

    settings = get_settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    store.init_schema()

    if args.all:
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM collection_runs ORDER BY created_at DESC"
            ).fetchall()
        run_ids = [str(row["run_id"]) for row in rows]
    else:
        run_id = args.run or store.latest_run_id()
        if run_id is None:
            print("还没有采集记录。")
            return 2
        run_ids = [run_id]

    if not run_ids:
        print("没有可生成附录的会话。")
        return 2

    failures = 0
    for run_id in run_ids:
        try:
            path = write_for_run(
                store, settings, run_id,
                with_hashes=not args.no_hash,
                show_paths=args.show_paths,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"✗ {run_id}：{type(exc).__name__}: {exc}")
            continue
        print(f"✓ {path}（{path.stat().st_size / 1024:.1f} KB）")

        try:
            from wrc.analyze.docx_export import markdown_to_docx

            docx_path = markdown_to_docx(
                path.read_text(encoding="utf-8"),
                path.with_suffix(".docx"),
                base_dir=path.parent,
            )
            print(f"  Word：{docx_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Word 导出失败（Markdown 已生成）：{type(exc).__name__}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
