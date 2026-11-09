"""01 · 建立并验证登录态（模块 A1）。

两种路径，按需求文档的推荐顺序实现：

路径一（推荐）
    弹出可见浏览器 → 用户手动扫码/登录 → 保存 storage_state 供复用。
    我们不实现账密登录：那等于把用户的密码接进工具，风险与伦理都不划算。

路径二（降级）
    ``--cookie``：从浏览器开发者工具复制 Cookie 字符串粘贴进来。
    仅在路径一失败时启用。

用法：
    python spikes/01_login.py                      # 扫码登录
    python spikes/01_login.py --check              # 只检查现有登录态
    python spikes/01_login.py --cookie "SUB=..."   # Cookie 降级路径
    python spikes/01_login.py --cookie -           # 从标准输入读 Cookie
"""

from __future__ import annotations

import argparse
import sys

import _common
from _common import EXIT_OK, EXIT_USER_ERROR, add_common_arguments, console, info, note, warn


async def run(args: argparse.Namespace) -> int:
    from wrc.session import BrowserSession, parse_cookie_string

    settings = _common.make_settings(
        args, **({"login_timeout_s": args.timeout} if args.timeout else {})
    )
    settings.ensure_dirs()

    _common.print_header(
        "01 · 登录态",
        "验证「复用浏览器登录态」这条技术路径是否在你的网络环境下可行。",
    )

    cookie_string = args.cookie
    if cookie_string == "-":
        cookie_string = sys.stdin.read()
        info(f"已从标准输入读取 {len(cookie_string)} 个字符的 Cookie。")
    if cookie_string:
        parsed = parse_cookie_string(cookie_string)
        if "SUB" not in parsed:
            warn("粘贴的 Cookie 中没有 SUB 字段 —— 这是微博登录态的核心字段，通常意味着复制不完整。")

    async with BrowserSession(settings, notify=lambda message: note(message)) as session:
        state = await session.describe_state()
        info(f"已有登录态文件：{state['storage_state_exists']}"
             f"（{settings.storage_state_path}）")

        if args.check:
            ok = False
            try:
                ok = await session.check_login()
            except Exception as exc:  # noqa: BLE001
                from wrc.console import report_error

                report_error(exc)
            console().print()
            if ok:
                info("✓ 登录态有效，可以直接开始采集。")
                return EXIT_OK
            warn("✗ 登录态无效或不存在，请运行本脚本的登录流程（不带 --check）。")
            return EXIT_USER_ERROR

        if cookie_string:
            ok = await session.import_cookie_string(cookie_string)
        else:
            ok = await session.login_interactive()

        final = await session.describe_state()
        console().print()
        console().print("登录态摘要（不含任何 Cookie 值）", markup=False)
        console().print(f"  是否已登录      {ok}", markup=False)
        console().print(f"  存储文件        {final['storage_state_path']}", markup=False)
        console().print(f"  Cookie 数量     {final['cookie_count']}", markup=False)
        console().print(f"  Cookie 字段     {', '.join(final['cookie_names'][:15]) or '（无）'}",
                        markup=False)
        console().print(f"  传输模式        {final['fetch_mode']}", markup=False)
        console().print()

        if ok:
            info("下一步：python spikes/02_chain.py --tag 延迟退休 --select 1,2 --comments 50")
            return EXIT_OK
        warn(
            "登录未成功。可以尝试："
            "1) 重新运行并完成扫码；"
            "2) 用 --cookie 走降级路径；"
            "3) 加 --fetch-mode page 换一种传输模式再试。"
        )
        return EXIT_USER_ERROR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建立并验证微博登录态")
    add_common_arguments(parser)
    parser.add_argument("--check", action="store_true", help="只检查现有登录态，不做登录")
    parser.add_argument(
        "--cookie",
        default=None,
        help="降级路径：直接粘贴 Cookie 字符串；传 '-' 从标准输入读取",
    )
    parser.add_argument("--timeout", type=int, default=None, help="等待登录的秒数")
    sys.exit(_common.run_async(lambda: run(parser.parse_args())))
