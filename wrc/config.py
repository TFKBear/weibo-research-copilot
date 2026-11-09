"""集中配置。

所有可调参数都走环境变量（前缀 ``WRC_``）或 ``.env``，
理由：社科研究者需要能复现「当时用的是多大延迟、多少条上限」，
所以这些值必须可被记录进研究日志，而不是散落在代码常量里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 桌面 Chrome UA。
#: 选它而不是移动端 UA 的原因：扫码登录页（weibo.com/login.php）在桌面 UA 下
#: 才呈现二维码形态，这是路径一（推荐）能否顺利用起来的前提；
#: 而 m.weibo.cn 的 JSON 接口本身不区分 UA 形态，通过与否取决于登录 Cookie。
#: 若实测发现接口对移动端 UA 更友好，用 WRC_USER_AGENT 覆盖即可，无需改代码。
DEFAULT_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)

#: 备选的移动端 UA，供探测脚本做对照实验。
DEFAULT_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
    "Mobile/15E148 Safari/604.1"
)

#: 默认登录入口。桌面 UA 下这个页面提供二维码。
DEFAULT_LOGIN_URL = "https://weibo.com/login.php"


class Settings(BaseSettings):
    """运行时配置。

    环境变量命名规则：``WRC_`` + 字段名大写，例如 ``WRC_DELAY_MIN_S=2.0``。
    """

    model_config = SettingsConfigDict(
        env_prefix="WRC_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------- 路径 ---------------- #
    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    # ---------------- 浏览器 / 登录 ---------------- #
    headless: bool = Field(
        default=False,
        description="登录与人工验证必须可见，故默认有头运行。",
    )
    browser_channel: str | None = Field(
        default=None,
        description="指定 Playwright 浏览器通道（如 'chrome'、'msedge'）；留空用内置 Chromium。",
    )
    system_browser_fallback: bool = Field(
        default=True,
        description=(
            "内置 Chromium 不可用时，是否自动尝试本机已安装的 Chrome / Edge。"
            "国内网络下载 Playwright 浏览器经常很慢，这个回退能直接绕开下载。"
        ),
    )
    fallback_channels: str = Field(
        default="chrome,msedge",
        description="自动回退时依次尝试的浏览器通道，逗号分隔。",
    )
    fetch_mode: Literal["context", "page"] = Field(
        default="context",
        description=(
            "'context'：用 Playwright 的 APIRequestContext 直接请求接口"
            "（不关心页面来源与 CORS，最稳，默认）；"
            "'page'：在已登录 m.weibo.cn 页面的 JS 上下文里 fetch"
            "（真实浏览器指纹，作为排障与对照手段）。"
        ),
    )
    user_agent: str = Field(default=DEFAULT_DESKTOP_UA)
    login_url: str = Field(default=DEFAULT_LOGIN_URL)
    api_referer: str = Field(
        default="https://m.weibo.cn/",
        description="调用 m 站接口时伪造的 Referer，避免被判定为跨站调用。",
    )
    login_timeout_s: int = Field(default=300, ge=30, le=1800)
    nav_timeout_ms: int = Field(default=30_000, ge=5_000)

    # ---------------- 采集频率纪律 ---------------- #
    delay_min_s: float = Field(default=1.5, ge=0.5, description="请求间最小随机延迟")
    delay_max_s: float = Field(default=3.0, ge=0.5, description="请求间最大随机延迟")
    max_requests_per_run: int = Field(
        default=120,
        ge=1,
        description="单次采集（单账号单 tag）的请求硬上限；统计全部出站请求，含重试。",
    )
    budget_warn_ratio: float = Field(default=0.8, gt=0, le=1.0)
    backoff_base_s: float = Field(default=60.0, ge=1.0)
    backoff_max_s: float = Field(default=600.0, ge=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    request_timeout_ms: int = Field(default=20_000, ge=1_000)

    # ---------------- 采集默认值 ---------------- #
    default_post_preview_count: int = Field(
        default=20, ge=1, le=50, description="tag 检索后向前端预览的帖子数"
    )
    default_comments_per_post: int = Field(
        default=50, ge=1, le=5_000, description="每篇帖子的默认评论目标量"
    )
    collect_subcomments: bool = Field(
        default=False, description="二级评论默认关闭：数据量会呈数量级增长。"
    )

    # ---------------- AI（可选增强） ---------------- #
    ai_model_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="模型根目录（仅用于展示与兼容）。",
    )
    ai_embedding_model_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "bge-small-zh-v1.5",
        description=(
            "句向量模型目录（聚类、营销模板相似度用）。必须与情感分类模型分开："
            "两者都被导出成 *_quantized.onnx，目录指错就会静默互换，"
            "把分类器的 3 维 logits 当成句向量。"
        ),
    )
    ai_sentiment_model_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "sentiment-twitter-xlmr",
        description=(
            "本地情感分类模型目录（情感分析第二通道用）。"
            "默认是 twitter-xlm-roberta（1.98 亿条推文训练的多语言模型），"
            "在真实微博评论上实测优于更新但域不匹配的 5 类模型；"
            "旧目录 `distilbert-sentiment-zh` 会被自动回退识别（见 ai/status.py）。"
        ),
    )
    ai_marketing_threshold: float = Field(
        default=0.62,
        ge=0.3,
        le=1.0,
        description=(
            "与营销模板的余弦相似度阈值，越低越激进。"
            "0.62 是用 12 条营销正样本 + 80 条真实评论标定出来的："
            "正样本 0.644–0.844、真实评论 0–0.606，取在两者之间。"
            "⚠ 换模型必须重新标定（python tools/calibrate_ai.py），"
            "bge 系列的相似度分布是压缩的，凭直觉设 0.8 会导致一条都抓不到。"
        ),
    )
    ai_coordinated_threshold: float = Field(
        default=0.85, ge=0.5, le=1.0,
        description="数据集内判为同一高相似簇的阈值。",
    )
    ai_min_cluster_size: int = Field(
        default=3, ge=2, le=20,
        description="至少多少条互相高相似才算一个协同行为簇。",
    )
    ai_batch_size: int = Field(default=16, ge=1, le=128)

    # ---------------- 校验 ---------------- #
    @field_validator("data_dir", mode="after")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    # ---------------- 派生路径 ---------------- #
    @property
    def auth_dir(self) -> Path:
        return self.data_dir / "auth"

    @property
    def storage_state_path(self) -> Path:
        """扫码登录后保存的浏览器上下文（Cookie + localStorage）。"""
        return self.auth_dir / "storage_state.json"

    @property
    def keys_dir(self) -> Path:
        return self.data_dir / "keys"

    @property
    def pseudo_salt_path(self) -> Path:
        """匿名化用的盐。只保存在本地，绝不导出、绝不进入数据文件。"""
        return self.keys_dir / "pseudo_salt.bin"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "wrc.sqlite3"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def research_log_path(self) -> Path:
        return self.log_dir / "research_log.jsonl"

    # ---------------- 行为 ---------------- #
    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.auth_dir,
            self.keys_dir,
            self.export_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def effective_delay_range(self) -> tuple[float, float]:
        lo, hi = self.delay_min_s, self.delay_max_s
        return (lo, hi) if lo <= hi else (hi, lo)


def get_settings(**overrides: object) -> Settings:
    """读取配置；overrides 用于测试或 CLI 覆盖。"""
    return Settings(**overrides)  # type: ignore[arg-type]
