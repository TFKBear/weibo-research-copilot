"""一键自检：把编译、测试、离线下游流程合成一条命令。

存在的原因很实际：开发过程中如果每检查一项就开一个终端窗口，
会不断打断使用者。所以约定「一次只跑一条命令、一次跑完所有检查」。

用法：
    python tools/check.py            # 编译 + 全部测试 + 离线流程冒烟
    python tools/check.py --network  # 额外做一次真实网络自检（需要已登录）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, command: list[str]) -> tuple[bool, str]:
    print(f"\n=== {label} ===")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = (result.stdout or "") + (result.stderr or "")
    print(output.strip()[-4000:] if output.strip() else "(无输出)")
    return result.returncode == 0, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true", help="额外跑真实网络自检")
    parser.add_argument(
        "--skip-deliverables", action="store_true",
        help="跳过交付物自检（只想验证代码时用；交付物自检反映的是本机数据状态）",
    )
    args = parser.parse_args()

    failures: list[str] = []

    ok, _ = _run("语法编译", [sys.executable, "-m", "compileall", "-q", "wrc", "tests"])
    if not ok:
        failures.append("语法编译")

    ok, _ = _run("单元测试", [
        sys.executable, "-m", "pytest", "-q",
        # 固定 basetemp 并关掉 cache：Windows 上 %TEMP%\pytest-of-<user> 的软链接
        # 偶尔会因权限残留导致整轮测试报错，看起来像代码坏了，其实是环境问题。
        "--basetemp=.pytest_tmp",
        "-p", "no:cacheprovider",
        "-p", "no:warnings",
    ])
    if not ok:
        failures.append("单元测试")

    ok, _ = _run("离线流程冒烟", [sys.executable, "tools/smoke_offline.py"])
    if not ok:
        failures.append("离线流程冒烟")

    ok, _ = _run("界面冒烟", [sys.executable, "tools/smoke_gui.py"])
    if not ok:
        failures.append("界面冒烟")

    # 交付物自检：只有本机已经导出过真实采集时才跑。
    # 没有数据的环境（比如刚 clone 下来）不该因此判失败，
    # 所以「找不到采集」算跳过，不算失败。
    # 注意：它检查的是**本机数据状态**（比如最近一次采集有没有跑分析），
    # 与代码质量无关；只想验证代码时用 --skip-deliverables。
    if args.skip_deliverables:
        print("\n=== 交付物自检 ===\n（按 --skip-deliverables 跳过）")
    else:
        ok, output = _run("交付物自检", [sys.executable, "tools/check_deliverables.py"])
        if not ok:
            if "还没有采集记录" in output or "导出目录不存在" in output:
                print("（跳过：本机还没有导出过采集结果）")
            else:
                failures.append("交付物自检（本机最近一次采集的产物不完整）")

    if args.network:
        ok, _ = _run("环境自检（网络）", [sys.executable, "-m", "wrc.cli", "doctor"])
        if not ok:
            failures.append("环境自检（网络）")

    print("\n" + "=" * 50)
    if failures:
        print("失败：" + "、".join(failures))
        return 2
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
