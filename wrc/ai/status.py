"""AI 能力自检：运行库在不在、模型文件在不在。

这一层的存在意义：用户会自己下载模型放进项目文件夹，所以"为什么 AI 用不了"
必须有精确答案，不能只给一句"AI 不可用"。自检要能说出缺哪一个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 官方模型仓库（体积大但直连通常可用）。
MODEL_URL_OFFICIAL = "https://huggingface.co/Xenova/bge-small-zh-v1.5"
#: 国内镜像（同一个仓库，目录结构一致）。
MODEL_URL_MIRROR = "https://hf-mirror.com/Xenova/bge-small-zh-v1.5"
#: 单文件下载地址：量化版约 23MB，全精度版约 91MB。
MODEL_FILE_QUANTIZED = (
    "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model_quantized.onnx"
)
MODEL_FILE_FULL = (
    "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx"
)
TOKENIZER_FILE_URL = (
    "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json"
)


@dataclass(slots=True)
class AIStatus:
    """AI 子系统的可用性快照。"""

    model_dir: Path
    runtime_ok: bool = False
    runtime_detail: str = ""
    onnx_path: Path | None = None
    tokenizer_path: Path | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def model_ok(self) -> bool:
        return self.onnx_path is not None and self.tokenizer_path is not None

    @property
    def ready(self) -> bool:
        return self.runtime_ok and self.model_ok

    def summary(self) -> str:
        if self.ready:
            size = self.onnx_path.stat().st_size / (1024 * 1024) if self.onnx_path else 0
            return f"AI 就绪（{self.onnx_path.name}，{size:.0f}MB）" if self.onnx_path else "AI 就绪"
        if not self.runtime_ok:
            return "AI 不可用：缺少运行库"
        return "AI 不可用：缺少模型文件"

    def remediation(self) -> list[str]:
        """给用户的修复步骤（按顺序）。"""
        steps: list[str] = []
        if not self.runtime_ok:
            steps.append("运行库缺失：pip install onnxruntime tokenizers")
        if self.onnx_path is None:
            steps.append(f"下载模型文件（约 23MB）：{MODEL_FILE_QUANTIZED}")
        if self.tokenizer_path is None:
            steps.append(f"下载分词器：{TOKENIZER_FILE_URL}")
        if steps:
            steps.append(f"把下载到的东西放进这个目录（可以带子文件夹）：{self.model_dir}")
        return steps


def _check_runtime() -> tuple[bool, str]:
    missing: list[str] = []
    for module in ("onnxruntime", "tokenizers", "numpy"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module}（{type(exc).__name__}）")
    if missing:
        return False, "缺少：" + ", ".join(missing)
    return True, "onnxruntime / tokenizers / numpy 均可导入"


def _pick_onnx(model_dir: Path) -> Path | None:
    """在模型目录里找 ONNX 权重。

    允许用户直接把整个仓库解压进来（``onnx/model.onnx`` 这种层级），
    也允许只扔一个文件进来。优先全精度，其次量化版。
    """
    if not model_dir.exists():
        return None
    candidates = sorted(model_dir.rglob("*.onnx"))
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        if name == "model.onnx":
            return (0, -path.stat().st_size)
        if "quantized" in name:
            return (2, -path.stat().st_size)
        return (1, -path.stat().st_size)

    return sorted(candidates, key=rank)[0]


def _pick_tokenizer(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    # 优先 tokenizer.json（HuggingFace 的 fast tokenizer 格式）
    preferred = sorted(model_dir.rglob("tokenizer.json"))
    if preferred:
        return preferred[0]
    return None


def detect_status(model_dir: Path) -> AIStatus:
    """检测 AI 能力。不会抛异常，任何问题都变成可读的 problems 条目。"""
    status = AIStatus(model_dir=Path(model_dir))
    status.runtime_ok, status.runtime_detail = _check_runtime()

    if not status.model_dir.exists():
        status.problems.append(f"模型目录不存在：{status.model_dir}")
        return status

    status.onnx_path = _pick_onnx(status.model_dir)
    status.tokenizer_path = _pick_tokenizer(status.model_dir)
    if status.onnx_path is None:
        status.problems.append("模型目录里没有找到任何 .onnx 权重文件")
    if status.tokenizer_path is None:
        status.problems.append("模型目录里没有找到 tokenizer.json")
    return status


# --------------------------------------------------------------------------- #
# 情感分类模型的兼容解析
# --------------------------------------------------------------------------- #
#: 旧版默认目录名。升级到多语言 5 类模型后仍要能找到它——
#: 否则"用旧模型跑过的数据"会变成跑不动的历史包袱。
LEGACY_SENTIMENT_DIR = "distilbert-sentiment-zh"


def resolve_sentiment_model_dir(settings: Any) -> Path:
    """定位情感模型目录：优先配置值，其次旧的默认目录。"""
    configured = Path(getattr(settings, "ai_sentiment_model_dir"))
    if detect_status(configured).ready:
        return configured
    root = Path(getattr(settings, "ai_model_dir", configured.parent))
    legacy = root / LEGACY_SENTIMENT_DIR
    if legacy != configured and legacy.exists() and detect_status(legacy).ready:
        return legacy
    return configured


def detect_sentiment_status(settings: Any) -> AIStatus:
    """情感模型状态（含旧目录回退）。"""
    return detect_status(resolve_sentiment_model_dir(settings))
