"""统一启动入口：双击 / ``python main.py`` 都会打开图形界面。

命令行用法见 ``python -m wrc.cli --help``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        from wrc.gui.app import main as gui_main
    except Exception as exc:  # noqa: BLE001 - 出错时也要让人看得懂
        print("无法启动图形界面：", exc)
        print("可能原因：缺少 tkinter，或依赖未安装。")
        print("可以先用命令行：python -m wrc.cli doctor")
        input("按回车退出 …")
        return 2
    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
