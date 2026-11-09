"""浏览器会话与登录态（模块 A1）。

两条路径，按需求文档的判定实现：

- **路径一（推荐）**：弹出可见浏览器，用户手动扫码/登录，
  我们把整个浏览器上下文存成 ``storage_state`` 并复用。
  不自行实现账密登录 —— 那是把用户密码接进工具，风险与伦理都不划算。
- **路径二（降级）**：用户从开发者工具复制 Cookie 字符串，粘贴进来。

传输模式（``fetch_mode``）：
- ``context``：用 Playwright 的 APIRequestContext 直连接口，天然带上上下文的
  Cookie 与 UA，不受 CORS/页面来源限制，是默认值；
- ``page``：在 ``m.weibo.cn`` 页面的 JS 上下文里发 fetch，用真实浏览器指纹，
  作为风控加强时的排障手段。

两者共享同一套 Cookie，切换不需要重新登录。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .config import Settings
from .errors import (
    AuthRequired,
    BrowserUnavailable,
    ConfigProblem,
    NetworkProblem,
    WrcError,
)
from .fetching import FetchResult, build_url, interpret_response, parse_json
from .weibo.endpoints import MOBILE_BASE, Request, config_request
from .weibo.parsing import is_logged_in

#: 自动化特征弱化。Chromium 默认会把 navigator.webdriver 置为 true，
#: 登录页与验证页的风控会据此加分。
_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
]


def parse_cookie_string(raw: str) -> dict[str, str]:
    """解析浏览器开发者工具里复制的 Cookie 字符串。

    >>> parse_cookie_string("SUB=_2A25; SUBP=x; _T_WM=abc")
    {'SUB': '_2A25', 'SUBP': 'x', '_T_WM': 'abc'}
    """
    cookies: dict[str, str] = {}
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, value = chunk.partition("=")
        name = name.strip()
        if name:
            cookies[name] = value.strip()
    return cookies


class BrowserFetcher:
    """基于浏览器上下文的传输实现。"""

    _PAGE_FETCH_JS = """
    async (payload) => {
      const { url, headers } = payload;
      try {
        const res = await fetch(url, {
          method: 'GET',
          credentials: 'include',
          headers: headers,
          cache: 'no-store'
        });
        const text = await res.text();
        return { status: res.status, text: text, finalUrl: res.url, error: null };
      } catch (e) {
        return { status: 0, text: '', finalUrl: '', error: String(e) };
      }
    }
    """

    def __init__(self, session: "BrowserSession") -> None:
        self._session = session

    async def _raw(self, request: Request) -> FetchResult:
        session = self._session
        settings = session.settings
        url = build_url(request)
        headers = {
            "accept": "application/json, text/plain, */*",
            "x-requested-with": "XMLHttpRequest",
            "mweibo-pwa": "1",
            "referer": request.referer or settings.api_referer,
        }

        if settings.fetch_mode == "page":
            return await self._raw_via_page(request, url, headers)
        return await self._raw_via_context(request, url, headers)

    async def _raw_via_context(
        self, request: Request, url: str, headers: dict[str, str]
    ) -> FetchResult:
        context = self._session.context
        try:
            response = await context.request.get(
                url,
                headers=headers,
                timeout=self._session.settings.request_timeout_ms,
                fail_on_status_code=False,
            )
            text = await response.text()
            return interpret_response(
                request,
                status=response.status,
                text=text,
                final_url=response.url,
                transport="context",
            )
        except PlaywrightTimeoutError as exc:
            # Playwright 的异常文本包含完整请求日志，用户读不懂也不需要读，
            # 只取首行进错误详情，完整栈留给日志。
            raise NetworkProblem(
                detail=f"{request.label}: {str(exc).splitlines()[0]}"
            ) from exc
        except PlaywrightError as exc:
            raise NetworkProblem(
                detail=f"{request.label}: {str(exc).splitlines()[0]}"
            ) from exc

    async def _raw_via_page(
        self, request: Request, url: str, headers: dict[str, str]
    ) -> FetchResult:
        page = await self._session.ensure_api_page()
        try:
            payload = await page.evaluate(
                self._PAGE_FETCH_JS,
                {"url": url, "headers": {k: v for k, v in headers.items()
                                         if k.lower() not in {"referer"}}},
            )
        except PlaywrightError as exc:
            raise NetworkProblem(detail=f"{request.label}: 页面内 fetch 失败 ({exc})") from exc

        if not isinstance(payload, dict):
            raise NetworkProblem(detail=f"{request.label}: 页面内 fetch 返回了非预期结构")
        if payload.get("error"):
            raise NetworkProblem(
                detail=f"{request.label}: {payload['error']}（可尝试 WRC_FETCH_MODE=context）"
            )
        return interpret_response(
            request,
            status=int(payload.get("status") or 0),
            text=str(payload.get("text") or ""),
            final_url=str(payload.get("finalUrl") or ""),
            transport="page",
        )

    async def get_text(self, request: Request) -> str:
        return (await self._raw(request)).text

    async def get_json(self, request: Request) -> Any:
        return parse_json(request, await self.get_text(request))


class BrowserSession:
    """管理 Playwright 生命周期、登录态与传输。"""

    def __init__(
        self,
        settings: Settings,
        *,
        notify: Callable[[str], None] | None = None,
        log: Any | None = None,
    ) -> None:
        self.settings = settings
        self._notify = notify or (lambda _message: None)
        self.log = log
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._api_page: Page | None = None
        self._fetcher: BrowserFetcher | None = None
        self.engine: str = "未启动"
        """实际使用的浏览器引擎（chromium / chrome / msedge）。

        必须可回查：换了渲染引擎理论上可能影响页面结构与接口可见性，
        它属于研究方法的一部分。
        """

    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "BrowserSession":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self) -> None:
        settings = self.settings
        settings.ensure_dirs()

        self._pw = await async_playwright().start()
        try:
            self._browser, self.engine = await self._launch_browser()

            context_kwargs: dict[str, Any] = {
                "user_agent": settings.user_agent,
                "viewport": {"width": 1280, "height": 900},
                "locale": "zh-CN",
                "timezone_id": "Asia/Shanghai",
            }
            if self.has_saved_state:
                context_kwargs["storage_state"] = str(settings.storage_state_path)

            self._context = await self._browser.new_context(**context_kwargs)
            self._context.set_default_timeout(settings.request_timeout_ms)
            self._context.set_default_navigation_timeout(settings.nav_timeout_ms)
            # 抹掉最明显的自动化指纹
            await self._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
        except WrcError:
            await self.stop()
            raise
        except PlaywrightError as exc:
            # 启动失败必须清理 driver 子进程，否则 Windows 上会留下一堆
            # "unclosed transport" 告警，把真正的错误信息淹掉。
            await self.stop()
            raise ConfigProblem(
                detail=str(exc).splitlines()[0],
                what_happened="浏览器启动失败，无法开始采集。",
            ) from exc

        self._fetcher = BrowserFetcher(self)

        self._record(
            "browser_start",
            headless=settings.headless,
            fetch_mode=settings.fetch_mode,
            restored_state=self.has_saved_state,
            engine=self.engine,
        )

    async def _launch_browser(self) -> tuple[Browser, str]:
        """启动浏览器，必要时回退到本机已安装的 Chrome / Edge。

        为什么需要回退：Playwright 内置 Chromium 的下载源在国内经常极慢甚至卡死，
        而 Windows 必带 Edge、多数用户也装了 Chrome。用 ``channel`` 直接驱动本地浏览器
        既省掉下载，行为也仍然是标准浏览器，不影响采集正确性。
        「实际用了哪个引擎」会写进研究日志 —— 它属于方法的一部分。
        """
        settings = self.settings
        errors: list[tuple[str, str]] = []

        if settings.browser_channel:
            candidates: list[str | None] = [settings.browser_channel]
        else:
            candidates = [None]
            if settings.system_browser_fallback:
                candidates.extend(
                    part.strip()
                    for part in settings.fallback_channels.split(",")
                    if part.strip()
                )

        for channel in candidates:
            label = channel or "chromium"
            kwargs: dict[str, Any] = {
                "headless": settings.headless,
                "args": list(_LAUNCH_ARGS),
            }
            if channel:
                kwargs["channel"] = channel
            try:
                browser = await self._pw.chromium.launch(**kwargs)
            except PlaywrightError as exc:
                errors.append((label, str(exc).splitlines()[0]))
                continue
            if channel:
                self._record("browser_fallback", engine=channel,
                             reason=errors[-1][1] if errors else "配置指定")
            return browser, label

        detail = "; ".join(f"{name}: {message}" for name, message in errors)
        raise BrowserUnavailable(detail=detail)

    async def stop(self) -> None:
        for closer in (self._browser, self._pw):
            if closer is None:
                continue
            try:
                if isinstance(closer, Browser):
                    await closer.close()
                else:
                    await closer.stop()
            except Exception:  # pragma: no cover - 退出阶段尽力而为
                pass
        self._browser = None
        self._pw = None
        self._context = None
        self._api_page = None

    # ------------------------------------------------------------------ #
    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise ConfigProblem(
                what_happened="浏览器会话尚未启动。",
                detail="BrowserSession.start() 未被调用",
            )
        return self._context

    @property
    def fetcher(self) -> BrowserFetcher:
        if self._fetcher is None:
            raise ConfigProblem(
                what_happened="浏览器会话尚未启动。",
                detail="BrowserSession.start() 未被调用",
            )
        return self._fetcher

    @property
    def has_saved_state(self) -> bool:
        path = self.settings.storage_state_path
        return path.exists() and path.stat().st_size > 2

    # ------------------------------------------------------------------ #
    async def ensure_api_page(self) -> Page:
        """``fetch_mode=page`` 需要的、常驻在 m.weibo.cn 的页面。"""
        if self._api_page is not None and not self._api_page.is_closed():
            return self._api_page
        page = await self.context.new_page()
        try:
            await page.goto(f"{MOBILE_BASE}/", wait_until="domcontentloaded")
        except PlaywrightTimeoutError as exc:
            raise NetworkProblem(detail=f"打开 m.weibo.cn 超时: {exc}") from exc
        if "m.weibo.cn" not in (page.url or ""):
            await page.close()
            raise ConfigProblem(
                detail=f"实际落地页={page.url}",
                what_happened=(
                    "打开 m.weibo.cn 后被重定向到其他站点，页面内 fetch 模式不可用。"
                ),
            )
        self._api_page = page
        return page

    # ------------------------------------------------------------------ #
    async def check_login(self) -> bool:
        """用 ``/api/config`` 判定登录态。失败即抛人类可读错误。"""
        payload = await self.fetcher.get_json(config_request())
        return is_logged_in(payload)

    async def login_interactive(self, *, timeout_s: int | None = None) -> bool:
        """路径一：弹窗让用户手动登录，成功后保存 storage_state。"""
        settings = self.settings
        timeout = timeout_s or settings.login_timeout_s
        page = await self.context.new_page()
        self._notify(f"正在打开登录页：{settings.login_url}")
        try:
            await page.goto(settings.login_url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError as exc:
            raise NetworkProblem(detail=f"打开登录页超时: {exc}") from exc

        self._notify(
            "请在弹出的浏览器窗口中完成登录（扫码或账号登录）。"
            f"工具会自动检测，最长等待 {timeout} 秒。"
        )
        self._record("login_attempt", strategy="interactive", url=settings.login_url)

        deadline = time.monotonic() + timeout
        interval = 3.0
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            try:
                if await self.check_login():
                    await self.save_state()
                    self._record("login_success", strategy="interactive")
                    self._notify("登录成功，登录态已保存。")
                    return True
            except WrcError as exc:
                # 登录尚未完成时接口报错是正常现象，不应打断用户
                self._record("login_probe_error", kind=exc.kind.value,
                             detail=exc.detail[:300])
            except Exception as exc:  # pragma: no cover
                self._record("login_probe_error", kind="unexpected", detail=str(exc)[:300])

        self._record("login_timeout", strategy="interactive", timeout_s=timeout)
        self._notify("等待登录超时。建议：重新运行登录命令，或改用 Cookie 方式。")
        return False

    async def import_cookie_string(self, raw: str) -> bool:
        """路径二（降级）：导入用户粘贴的 Cookie 字符串。"""
        cookies = parse_cookie_string(raw)
        if not cookies:
            raise ConfigProblem(
                what_happened="粘贴的 Cookie 内容为空或格式不正确。",
                detail=f"length={len(raw or '')}",
            )
        self._notify(f"已解析到 {len(cookies)} 个 Cookie 字段，正在注入浏览器上下文。")
        payload: list[dict[str, Any]] = []
        for domain in (".weibo.cn", ".weibo.com"):
            for name, value in cookies.items():
                payload.append(
                    {"name": name, "value": value, "domain": domain, "path": "/"}
                )
        await self.context.add_cookies(payload)

        if await self.check_login():
            await self.save_state()
            self._record("login_success", strategy="cookie_string",
                         cookie_names=sorted(cookies))
            self._notify("Cookie 有效，登录态已保存。")
            return True

        self._record("login_failed", strategy="cookie_string",
                     cookie_names=sorted(cookies))
        self._notify(
            "Cookie 已注入但没有通过登录校验。最常见的原因是缺少 SUB 字段，"
            "或 Cookie 已过期 —— 请在已登录的浏览器里重新复制完整的 Cookie。"
        )
        return False

    # ------------------------------------------------------------------ #
    async def save_state(self) -> Path:
        """持久化浏览器上下文。文件权限收紧：它等价于一份凭证。"""
        path = self.settings.storage_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        await self.context.storage_state(path=str(path))
        _tighten_permissions(path)
        self._record("storage_state_saved", path=path, size_bytes=path.stat().st_size)
        return path

    async def describe_state(self) -> dict[str, Any]:
        """给「登录态诊断」用的可读信息，不含任何 Cookie 值。"""
        cookies = await self.context.cookies()
        names = sorted({cookie["name"] for cookie in cookies})
        return {
            "storage_state_exists": self.has_saved_state,
            "storage_state_path": str(self.settings.storage_state_path),
            "cookie_count": len(cookies),
            "cookie_names": names,
            "fetch_mode": self.settings.fetch_mode,
            "engine": self.engine,
        }

    # ------------------------------------------------------------------ #
    def _record(self, event: str, **fields: Any) -> None:
        if self.log is not None:
            try:
                self.log.record(event, **fields)
            except Exception:  # pragma: no cover - 日志失败不应中断采集
                pass


def _tighten_permissions(path: Path) -> None:
    """尽力把凭证文件设为仅本人可读写。

    Windows 上 POSIX 权限语义有限，os.chmod 只影响只读位；
    真正起作用的是用户目录的 ACL。这里做的是「不把权限放得更宽」。
    """
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover
        pass


async def login_once(settings: Settings, *,
                     notify: Callable[[str], None] | None = None,
                     cookie_string: str | None = None) -> dict[str, Any]:
    """一次性登录流程，供 ``spikes/01_login.py`` 使用。"""
    notify = notify or (lambda _m: None)
    async with BrowserSession(settings, notify=notify) as session:
        ok = False
        if cookie_string:
            ok = await session.import_cookie_string(cookie_string)
        else:
            ok = await session.login_interactive()
        state = await session.describe_state()
        return {"logged_in": ok, **state}
