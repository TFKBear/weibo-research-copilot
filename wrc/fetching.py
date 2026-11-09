"""传输层抽象。

把「怎么把请求发出去」与「怎么理解微博的响应」分开：

- 具体实现（``BrowserFetcher``）只负责拿到 ``(status, text)``；
- ``interpret_response()`` 统一把 HTTP 语义翻译成产品级错误。

这样以后要加一个纯 HTTP 的后备传输，或者把整条链路搬到服务端，
都不用改采集逻辑；同时也让错误分级可以被 fixture 测试覆盖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import urlencode

from .errors import (
    ApiError,
    AuthRequired,
    NetworkProblem,
    ParseFailure,
    RateLimited,
    RiskBlocked,
)
from .weibo.endpoints import Request

#: 一旦命中这些状态码，就说明我们在被限流，必须退避。
_RATE_LIMIT_STATUS = {418, 429, 432, 503}
#: 风控/人机验证。
_RISK_STATUS = {403, 412}


def build_url(request: Request) -> str:
    if not request.params:
        return request.url
    sep = "&" if "?" in request.url else "?"
    return f"{request.url}{sep}{urlencode(request.params)}"


@dataclass(slots=True)
class FetchResult:
    """一次已通过状态码检查的响应。"""

    text: str
    status: int = 200
    final_url: str = ""
    transport: str = "unknown"


@runtime_checkable
class Fetcher(Protocol):
    """采集层依赖的最小传输接口。"""

    async def get_text(self, request: Request) -> str: ...

    async def get_json(self, request: Request) -> Any: ...


# --------------------------------------------------------------------------- #
def interpret_response(
    request: Request,
    *,
    status: int,
    text: str,
    final_url: str = "",
    transport: str = "unknown",
) -> FetchResult:
    """把 HTTP 结果翻译成「要么可用，要么是可读错误」。"""
    label = request.label

    if status == 0:
        raise NetworkProblem(detail=f"{label}: 浏览器未拿到响应（可能是网络中断）")

    if status in _RATE_LIMIT_STATUS:
        raise RateLimited(status=status, detail=f"{label}: status={status}")

    if status in _RISK_STATUS:
        raise RiskBlocked(status=status, detail=f"{label}: status={status}")

    if status in (401, 407):
        raise AuthRequired(detail=f"{label}: status={status}")

    if "passport.weibo" in (final_url or ""):
        raise AuthRequired(detail=f"{label}: 被重定向到登录页 {final_url}")

    if 300 <= status < 400:
        raise AuthRequired(
            detail=f"{label}: 未跟随的重定向 status={status} → {final_url}",
            what_happened="请求被重定向回登录流程，当前登录态可能已失效。",
        )

    if status >= 500:
        raise ApiError(
            detail=f"{label}: status={status}",
            what_happened=f"微博服务器暂时不可用（HTTP {status}），本次请求没有拿到数据。",
            status=status,
        )

    if status != 200:
        raise ApiError(detail=f"{label}: status={status}", status=status)

    head = text.lstrip()[:1]
    if head == "<":
        # 返回了 HTML：要么是登录页，要么是验证/风控页。
        if "passport" in text or "登录" in text[:4000]:
            raise AuthRequired(detail=f"{label}: 响应是 HTML 登录页")
        if request.expect == "html":
            # 探测降级路径时期望的就是 HTML，交给调用方解析
            return FetchResult(text=text, status=status, final_url=final_url,
                               transport=transport)
        raise RiskBlocked(
            status=status,
            detail=f"{label}: 响应是 HTML 页面而非 JSON，长度={len(text)}",
        )

    if request.expect == "html" and head != "<":
        raise ParseFailure(
            detail=f"{label}: 期望桌面检索页（HTML），实际拿到非 HTML 内容"
        )

    return FetchResult(text=text, status=status, final_url=final_url, transport=transport)


def parse_json(request: Request, text: str) -> Any:
    """解析 JSON 响应。失败时给出「接口可能已调整」而不是裸 JSONDecodeError。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:200].replace("\n", " ")
        raise ParseFailure(
            detail=f"{request.label}: JSON 解析失败 ({exc}); 片段={snippet!r}"
        ) from exc


# --------------------------------------------------------------------------- #
class MockFetcher:
    """离线测试与 ``--mock`` 模式用的传输实现。

    路由是一个可调用对象，接收 ``Request`` 并返回 dict（JSON）或 str（原文）。
    之所以用可调用而不是固定字典：blog 接口的参数组合很多
    （§tag 的第 N 页、某帖子的第 k 页评论），用回调表达最直白。
    """

    def __init__(self, router: Callable[[Request], Any]) -> None:
        self._router = router
        self.calls: list[Request] = []

    async def get_text(self, request: Request) -> str:
        self.calls.append(request)
        payload = self._router(request)
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)

    async def get_json(self, request: Request) -> Any:
        return parse_json(request, await self.get_text(request))
