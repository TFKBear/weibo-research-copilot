"""LLM 调用抽象（模块 A 的硬约束：不得硬编码任何厂商 SDK）。

需求文档要求情感分析、话题推荐、助手问答三处共用同一个 `LLMClient` 接口，
支持 OpenAI / DeepSeek / 本地模型切换。这里用 **OpenAI 兼容协议**实现，
因为上面三种后端都支持它——换后端只改 base_url 和 key 变量名，不改代码。

配置（.env）：
    WRC_LLM_BASE_URL=https://opencode.ai/zen/go/v1
    WRC_LLM_API_KEY_ENV=OPENCODE_API_KEY
    WRC_LLM_MODEL=deepseek-v4.1-flash
    WRC_LLM_HEADERS={"x-opencode-session":"..."}   # 可选，JSON
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigProblem, ErrorKind, WrcError


class LlmUnavailable(WrcError):
    """LLM 未配置或不可用。"""

    def __init__(self, detail: str = "", *, base_url: str = "", model: str = "") -> None:
        super().__init__(
            ErrorKind.CONFIG,
            "调用大模型失败：没有可用的 LLM 配置或连接不通。",
            detail=detail,
            what_you_can_do=(
                "在 .env 里设置 WRC_LLM_API_KEY_ENV 指向一个已存在的环境变量名，"
                "例如 WRC_LLM_API_KEY_ENV=OPENCODE_API_KEY",
                "设置 WRC_LLM_BASE_URL（默认 https://opencode.ai/zen/go/v1）"
                "与 WRC_LLM_MODEL（默认 deepseek-v4.1-flash）",
                f"当前 base_url={base_url or '未设置'}，model={model or '未设置'}",
            ),
            retryable=True,
        )


DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_API_KEY_ENV = "OPENCODE_API_KEY"
DEFAULT_MODEL = "deepseek-v4.1-flash"


@dataclass(slots=True)
class LlmConfig:
    """一次 LLM 调用的配置。

    ⚠ 默认值必须写成模块级常量，不能写 `cls.base_url`：
    `@dataclass(slots=True)` 会把类属性替换成 slot 描述符，
    读取 `cls.base_url` 拿到的是 `member_descriptor` 而不是默认值字符串。
    """

    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    model: str = DEFAULT_MODEL
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    max_tokens: int = 4000
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LlmConfig":
        raw_headers = os.environ.get("WRC_LLM_HEADERS", "").strip()
        headers: dict[str, str] = {}
        if raw_headers:
            try:
                parsed = json.loads(raw_headers)
                if isinstance(parsed, dict):
                    headers = {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass
        obj = cls(
            base_url=os.environ.get("WRC_LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key_env=os.environ.get("WRC_LLM_API_KEY_ENV", DEFAULT_API_KEY_ENV),
            model=os.environ.get("WRC_LLM_MODEL", DEFAULT_MODEL),
            headers=headers,
        )
        # opencode 的网关要求 x-opencode-session 头，不然直接 400。
        # 这是实测出来的，不是猜的；缺省时自动补一个固定值，避免用户踩坑。
        if "opencode.ai" in obj.base_url and "x-opencode-session" not in {
            k.lower() for k in obj.headers
        }:
            obj.headers["x-opencode-session"] = "dsh-weibo-research-copilot"
        return obj

    def api_key(self) -> str:
        """读取 API Key。

        先看进程环境变量；Windows 上再退一步读注册表的用户级环境变量。
        退这一步是有原因的：用户常常在程序启动**之后**才设环境变量，
        而 Windows 进程的环境是启动时的快照——不补这个回退，
        用户会看到"我明明设了却说没设"，且不知道要重启。
        """
        value = os.environ.get(self.api_key_env, "").strip()
        if value:
            return value
        return _windows_user_env(self.api_key_env)


def _windows_user_env(name: str) -> str:
    """读取 HKCU\\Environment 里的用户级环境变量（仅 Windows）。"""
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value).strip() if value else ""
    except Exception:  # noqa: BLE001 - 读不到就是没有，不影响主流程
        return ""


class LlmClient:
    """OpenAI 兼容的对话补全客户端。"""

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig.from_env()
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    # ------------------------------------------------------------------ #
    def available(self) -> tuple[bool, str]:
        if not self.config.api_key():
            return False, f"环境变量 {self.config.api_key_env} 为空"
        try:
            import httpx  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"httpx 不可用：{exc}"
        return True, "ok"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "calls": self._calls,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    # ------------------------------------------------------------------ #
    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        """发一次对话补全，返回助手文本。"""
        ok, reason = self.available()
        if not ok:
            raise LlmUnavailable(
                detail=reason,
                base_url=self.config.base_url,
                model=self.config.model,
            )

        import httpx

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key()}",
            "Content-Type": "application/json",
            **self.config.headers,
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        last_error = ""
        for attempt in range(2):
            try:
                with httpx.Client(timeout=self.config.timeout_s) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                        time.sleep(2.0)
                        continue
                    raise LlmUnavailable(
                        detail=last_error,
                        base_url=self.config.base_url,
                        model=self.config.model,
                    )
                data = response.json()
            except LlmUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise LlmUnavailable(
                    detail=last_error,
                    base_url=self.config.base_url,
                    model=self.config.model,
                ) from exc

            self._calls += 1
            usage = data.get("usage") or {}
            self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self._completion_tokens += int(usage.get("completion_tokens") or 0)
            choices = data.get("choices") or []
            if not choices:
                raise LlmUnavailable(detail="响应里没有 choices", model=self.config.model)
            message = choices[0].get("message") or {}
            return str(message.get("content") or "")

        raise LlmUnavailable(detail=last_error, model=self.config.model)

    # ------------------------------------------------------------------ #
    def complete_json(self, *, system: str, user: str, max_tokens: int | None = None) -> Any:
        """要求模型输出 JSON 并解析。容忍代码块包裹与前后废话。"""
        text = self.complete(system=system, user=user, max_tokens=max_tokens)
        return parse_json_loose(text)


def parse_json_loose(text: str) -> Any:
    """从可能带 ```json 围栏或前后说明文字的回复里抠出 JSON。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            body = lines[1:]
            if body and body[-1].strip().startswith("```"):
                body = body[:-1]
            cleaned = "\n".join(body).strip()

    start = cleaned.find("{")
    bracket = cleaned.find("[")
    candidates = [index for index in (start, bracket) if index >= 0]
    if candidates:
        cleaned = cleaned[min(candidates) :]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 退一步：括号配平截取
    for opener, closer in (("{", "}"), ("[", "]")):
        begin = cleaned.find(opener)
        if begin < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(begin, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[begin : index + 1])
                    except json.JSONDecodeError:
                        break
    raise ConfigProblem(
        what_happened="大模型返回的内容不是合法 JSON，本次调用作废。",
        detail=text[:400],
    )
