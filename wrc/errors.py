"""错误分级与人性化转译。

产品宪法要求：面向社科学生展示的永远是
「发生了什么 + 你可以怎么做」，而不是 Python traceback。

因此所有面向用户可见的失败必须是一个 `WrcError` 及其子类，
由 `render()` 统一转译为中文可执行提示。原始异常保存在 `.__cause__`
和 `.detail` 中，只写入日志，不直接抛给用户。
"""

from __future__ import annotations

import textwrap
from enum import Enum


class ErrorKind(str, Enum):
    """错误的语义分类，决定文案、是否可重试、是否计入安全阈值。"""

    AUTH_REQUIRED = "auth_required"  # 登录态缺失/过期
    RATE_LIMITED = "rate_limited"  # 限流（418/429/432）
    RISK_BLOCKED = "risk_blocked"  # 风控/验证码（403/验证页）
    NETWORK = "network"  # 网络层失败
    API = "api"  # 接口返回了业务错误
    PARSE = "parse"  # 响应结构变化，解析失败
    BUDGET = "budget"  # 触发本次采集的硬性请求上限
    CONFIG = "config"  # 用户配置错误
    CONSENT = "consent"  # 伦理声明未确认
    EMPTY = "empty"  # 查询成功但结果为空


_FALLBACK = "工具遇到了一个未预期的错误，已记录到日志。"


class WrcError(Exception):
    """面向用户的可读错误基类。

    Args:
        kind: 语义分类。
        detail: 供日志/开发者使用的技术细节（可含 traceback 文本）。
        what_happened: 面向用户的一句话解释。
        what_you_can_do: 面向用户的下一步建议，按推荐顺序排列。
        retryable: 是否值得自动/手动重试。
        status: HTTP 状态码（若适用）。
    """

    retryable_default: bool = False

    def __init__(
        self,
        kind: ErrorKind,
        what_happened: str,
        *,
        detail: str = "",
        what_you_can_do: tuple[str, ...] = (),
        retryable: bool | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(what_happened)
        self.kind = kind
        self.what_happened = what_happened
        self.detail = detail
        self.what_you_can_do = what_you_can_do
        self.retryable = self.retryable_default if retryable is None else retryable
        self.status = status

    # ------------------------------------------------------------------ #
    def render(self, *, width: int = 78) -> str:
        """转译为面向终端用户的多行中文提示。"""
        lines = [f"⚠  {self.what_happened}"]
        if self.detail:
            hint = _one_line(self.detail)
            if hint:
                lines.append(textwrap.fill(f"  技术细节：{hint}", width=width,
                                           subsequent_indent=" " * 12))
        if self.what_you_can_do:
            lines.append("   你可以这样做：")
            for i, action in enumerate(self.what_you_can_do, start=1):
                lines.append(textwrap.fill(f"     {i}. {action}", width=width,
                                           subsequent_indent=" " * 8))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        return f"[{self.kind.value}] {self.what_happened}"


class AuthRequired(WrcError):
    def __init__(self, detail: str = "", *, what_happened: str | None = None) -> None:
        super().__init__(
            ErrorKind.AUTH_REQUIRED,
            what_happened or "微博登录态缺失或已过期，无法继续采集。",
            detail=detail,
            what_you_can_do=(
                "重新运行登录命令（spikes/01_login.py）扫码登录；"
                "登录后浏览器上下文会自动保存为 storage_state。",
                "如果扫码不可用，退回到 Cookie 粘贴方式（--cookie 参数）。",
            ),
            retryable=False,
        )


class RateLimited(WrcError):
    def __init__(self, *, status: int | None = None, wait_s: float | None = None,
                 detail: str = "") -> None:
        wait_text = f"工具已自动暂停 {int(wait_s)} 秒" if wait_s else "工具将自动暂停后重试"
        actions = [
            "等待后重试，本次采集进度不会丢失。",
            "减少本次采集数量或缩小 tag 范围，降低请求总量。",
            "更换时段（微博的风控强度有明显的时间分布差异），或更换登录账号。",
        ]
        super().__init__(
            ErrorKind.RATE_LIMITED,
            f"微博暂时限制了本次请求（{status or '未知状态码'}）。{wait_text}。",
            detail=detail,
            what_you_can_do=tuple(actions),
            retryable=True,
            status=status,
        )


class RiskBlocked(WrcError):
    def __init__(self, *, status: int | None = None, detail: str = "") -> None:
        super().__init__(
            ErrorKind.RISK_BLOCKED,
            "微博对当前账号弹出了人机验证或风控拦截，采集已停止。",
            detail=detail,
            what_you_can_do=(
                "在打开的浏览器窗口中手动完成验证，然后重新开始采集。",
                "如果反复触发，请降低采集频率（调大 WRC_DELAY_MIN_S / WRC_DELAY_MAX_S）。",
                "确认登录账号为正常的个人账号，新注册或长期未使用的账号风险更高。",
            ),
            retryable=False,
            status=status,
        )


class NetworkProblem(WrcError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            ErrorKind.NETWORK,
            "网络请求没有成功完成（连接超时或被中断）。",
            detail=detail,
            what_you_can_do=(
                "检查本机网络与代理设置后重试。",
                "如果使用了校园网 / 代理，请确认可以正常打开 m.weibo.cn。",
            ),
            retryable=True,
        )


class ApiError(WrcError):
    def __init__(self, detail: str = "", *, what_happened: str | None = None,
                 status: int | None = None) -> None:
        super().__init__(
            ErrorKind.API,
            what_happened or "微博接口返回了业务错误，本次请求没有拿到数据。",
            detail=detail,
            what_you_can_do=(
                "稍后将本次请求重试一次。",
                "如果持续出现，请把日志中的 detail 一并反馈（接口结构可能已调整）。",
            ),
            retryable=True,
            status=status,
        )


class ParseFailure(WrcError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            ErrorKind.PARSE,
            "拿到了响应，但内容结构与工具预期不符，可能是微博接口已调整。",
            detail=detail,
            what_you_can_do=(
                "运行 spikes/03_probe.py 探测当前可用的接口形态。",
                "把 data/logs/ 下对应日志与原始响应片段反馈给维护者。",
            ),
            retryable=False,
        )


class BudgetExceeded(WrcError):
    def __init__(self, detail: str = "", *, used: int = 0, limit: int = 0) -> None:
        super().__init__(
            ErrorKind.BUDGET,
            f"本次采集已达到安全请求上限（{used}/{limit} 次），已主动停止。",
            detail=detail,
            what_you_can_do=(
                "缩小本次采集范围（更少的帖子、更少的评论），重新开始一次采集。",
                "更换时段或更换账号后再继续，不要在同一时段强行提高上限。",
            ),
            retryable=False,
        )


class ConsentNotGiven(WrcError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            ErrorKind.CONSENT,
            "导出数据前必须确认数据使用声明，本次导出已取消。",
            detail=detail,
            what_you_can_do=(
                "阅读声明后重新运行，并显式加上确认参数或输入 y 确认。",
            ),
            retryable=False,
        )


class EmptyResult(WrcError):
    def __init__(self, detail: str = "", *, what_happened: str | None = None) -> None:
        super().__init__(
            ErrorKind.EMPTY,
            what_happened or "查询本身成功，但没有拿到任何结果。",
            detail=detail,
            what_you_can_do=(
                "确认 tag 写法：微博话题通常需要写成 #延迟退休# 的形式，工具会自动补全井号。",
                "换一个更大或更通用的话题标签再试一次。",
            ),
            retryable=False,
        )


class ConfigProblem(WrcError):
    def __init__(self, detail: str = "", *, what_happened: str | None = None) -> None:
        super().__init__(
            ErrorKind.CONFIG,
            what_happened or "配置有误，工具无法开始采集。",
            detail=detail,
            what_you_can_do=(
                "检查 .env 文件或命令行参数，修正后重试。",
            ),
            retryable=False,
        )


class CommentSourceEmpty(WrcError):
    """评论接口「成功返回但没有任何评论」。

    这是本工具最危险的一种失败，必须单独成一个错误类型：
    接口返回 ``ok=1``，但拿不到评论数据。此时若按「平台没有更多评论」处理，
    研究就会在一个**空样本**上继续做清洗与情感分析，而且日志里看不出异常。
    一个帖子明明有 679 条评论却采到 0 条，这不是结论，是故障。
    """

    def __init__(
        self,
        *,
        post_mid: str = "",
        expected_total: int | None = None,
        attempts: tuple[tuple[str, str], ...] = (),
        detail: str = "",
    ) -> None:
        tried = "；".join(f"{name}→{outcome}" for name, outcome in attempts) or "无"
        super().__init__(
            ErrorKind.API,
            (
                f"评论接口返回了空数据：帖子 mid={post_mid} "
                f"（平台显示评论数 {expected_total if expected_total is not None else '未知'}），"
                "但所有候选接口都没有返回任何评论。"
            ),
            detail=detail or f"尝试过的接口：{tried}",
            what_you_can_do=(
                f"运行 python spikes/04_probe_comments.py --mid {post_mid} "
                "查看各候选接口的真实返回，确定当前可用的接口形态。",
                "换一篇评论数较少（几十条）的帖子再试，排除长文/视频类帖子的特殊形态。",
                "把探测脚本输出的 raw_*.json 反馈给维护者，这类问题无法靠猜测修复。",
            ),
            retryable=False,
        )


class AIUnavailable(WrcError):
    """AI 能力不可用（运行库缺失 / 模型文件缺失或损坏）。

    设计上必须"可降级"：AI 是可选增强，缺了它工具仍应能用规则表完成清洗。
    所以这个错误只在用户**明确勾选了 AI** 时才抛出，并且必须告诉用户
    到底缺什么、去哪里补。
    """

    def __init__(self, detail: str = "", *, model_dir: str = "") -> None:
        actions = [
            "下载模型文件（约 23MB 的量化版）："
            "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model_quantized.onnx",
            "下载分词器："
            "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json",
            f"把两个文件放进这个目录：{model_dir or 'models/'}（可以带子文件夹）",
            "运行库缺失时执行：pip install onnxruntime tokenizers",
            "或者取消勾选「启用 AI 模型」，用规则表清洗（不影响其他功能）。",
        ]
        super().__init__(
            ErrorKind.CONFIG,
            "启用了 AI 模型，但 AI 环境还没准备好。",
            detail=detail,
            what_you_can_do=tuple(actions),
            retryable=False,
        )


class BrowserUnavailable(WrcError):
    """浏览器引擎缺失或版本不匹配。

    这是新用户第一眼最可能撞上的问题，也是最容易被原始 traceback 淹没的问题：
    Playwright 的报错里全是路径与版本号，学生看不出"跑一行命令就能解决"。
    """

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            ErrorKind.CONFIG,
            "浏览器引擎未安装或版本与当前 Playwright 不匹配，无法启动采集浏览器。",
            detail=detail,
            what_you_can_do=(
                "在项目目录执行一次：python -m playwright install chromium",
                "若下载很慢或卡住（国内常见），先设置镜像再执行："
                "PowerShell 里 $env:PLAYWRIGHT_DOWNLOAD_HOST='https://cdn.npmmirror.com/binaries/playwright'，"
                "然后重跑上面的命令。",
                "如果本机已装 Chrome/Edge，也可以跳过下载，改用系统浏览器："
                "设置环境变量 WRC_BROWSER_CHANNEL=chrome（或 msedge）。",
            ),
            retryable=False,
        )


# ---------------------------------------------------------------------- #
def _one_line(text: str, limit: int = 240) -> str:
    """把可能很长的技术细节压成一行，避免污染用户可读区。"""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat


def render_error(exc: BaseException) -> str:
    """统一入口：任何异常都能得到一段面向用户的文本。"""
    if isinstance(exc, WrcError):
        return exc.render()
    return (
        f"⚠  {_FALLBACK}\n"
        f"   技术细节：{_one_line(f'{type(exc).__name__}: {exc}')}"
    )
