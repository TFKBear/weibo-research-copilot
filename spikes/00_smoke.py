"""00 · 环境自检。

目的：在碰微博数据之前，先确认「这台机器能不能把请求发出去」。
它不需要登录，也不需要采集任何内容，只回答三个问题：

1. 依赖是否齐全（Playwright / rich / pydantic-settings …）；
2. Chromium 能否启动；
3. 应用内的传输链路能否真的拿到 m.weibo.cn 的 JSON 响应。

用法：
    python spikes/00_smoke.py
    python spikes/00_smoke.py --headless
"""

from __future__ import annotations

import argparse
import platform
import sys

import _common
from _common import EXIT_OK, EXIT_USER_ERROR, add_common_arguments, console, info, note


def _check_imports() -> list[tuple[str, str, bool]]:
    checks: list[tuple[str, str, bool]] = []
    for module_name, label in (
        ("playwright", "Playwright"),
        ("rich", "rich"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("httpx", "httpx"),
        ("jieba", "jieba（B3 分词，暂未启用）"),
    ):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "?")
            checks.append((label, str(version), True))
        except Exception as exc:  # noqa: BLE001
            checks.append((label, f"导入失败: {exc}", False))
    return checks


async def main(args: argparse.Namespace) -> int:
    from wrc.session import BrowserSession
    from wrc.weibo.endpoints import MOBILE_BASE, config_request

    settings = _common.make_settings(args)
    settings.ensure_dirs()

    console().print("环境信息", markup=False)
    console().print(f"  Python      {sys.version.split()[0]}", markup=False)
    console().print(f"  Platform    {platform.platform()}", markup=False)
    console().print(f"  data_dir    {settings.data_dir}", markup=False)
    console().print()

    console().print("依赖检查", markup=False)
    all_ok = True
    for label, version, ok in _check_imports():
        mark = "✓" if ok else "✗"
        console().print(f"  {mark} {label:<34}{version}", markup=False)
        all_ok = all_ok and ok
    if not all_ok:
        info("有依赖缺失，请先执行：pip install -e .")
        return EXIT_USER_ERROR
    console().print()

    console().print("浏览器与传输链路", markup=False)
    async with BrowserSession(settings, notify=lambda message: note(message)) as session:
        info("Chromium 已启动。")
        info(f"登录态文件：{settings.storage_state_path}"
             f"（{'已存在' if session.has_saved_state else '尚未创建'}）")
        try:
            payload = await session.fetcher.get_json(config_request())
        except Exception as exc:  # noqa: BLE001
            from wrc.console import report_error

            console().print("  ✗ 无法访问 m.weibo.cn 接口", markup=False)
            report_error(exc)
            info("常见原因：本机网络/代理无法访问微博，或需要先在浏览器里打开一次。")
            return EXIT_USER_ERROR
        logged_in = bool(payload.get("data", {}).get("login")) if isinstance(payload, dict) else False
        info(f"接口可达：{MOBILE_BASE} 返回合法 JSON。")
        info(f"登录状态：{'已登录' if logged_in else '未登录（这是预期状态，说明连通性正常）'}")
        state = await session.describe_state()
        info(f"浏览器引擎：{state['engine']}")
        info(f"Cookie 数量：{state['cookie_count']}，字段：{', '.join(state['cookie_names'][:12]) or '（无）'}")
    console().print()

    console().print("下一步", markup=False)
    console().print("  1) 建立登录态：python spikes/01_login.py", markup=False)
    console().print("  2) 跑通全链路：python spikes/02_chain.py --mock   （离线自检）", markup=False)
    console().print("  3) 真实采集：  python spikes/02_chain.py --tag 延迟退休 --comments 50", markup=False)
    console().print("  4) 接口探测：  python spikes/03_probe.py --tag 延迟退休", markup=False)
    return EXIT_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="环境自检（不需要登录）")
    add_common_arguments(parser)
    parser.set_defaults(headless=None)
    sys.exit(_common.run_async(lambda: main(parser.parse_args())))
