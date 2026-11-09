"""ONNX 序列分类器（情感分析第二通道的底座）。

为什么不直接用 transformers：那要装 torch（200MB+），而本产品走 ONNX 路线
（onnxruntime 约 20MB）。这个类只做一件事：把文本变成「各类别概率」。

⚠ 与 embedder 的关键差别：句向量模型取 [CLS] 位置的**隐状态**，
   而序列分类模型还要再过一层分类头。ONNX 导出里通常已包含分类头，
   所以这里直接取 logits（形状 `(batch, num_labels)`），再自己 softmax。
   如果导出的是 `(batch, seq, labels)` 这种带位置的张量，说明导出方式不同，
   本类会明确报错而不是静默给出错误结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..errors import AIUnavailable

DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16


@dataclass(slots=True)
class Classification:
    """一条文本的分类结果。"""

    label: str
    score: float
    """最高类别概率，作为置信度使用。"""
    probs: dict[str, float]

    def prob(self, label: str) -> float:
        return float(self.probs.get(label, 0.0))


class OnnxSequenceClassifier:
    """通用的文本分类器（情感、情绪等都走它）。"""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        labels: Sequence[str] | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        model_name: str = "",
    ) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(detail=f"导入分类器运行库失败：{exc}") from exc

        self._np = np
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_length = max_length
        self.model_name = model_name or self.model_path.stem

        self.labels = list(labels) if labels else _labels_from_config(self.model_path)

        try:
            self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            self._tokenizer.enable_truncation(max_length=max_length)
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(detail=f"加载分词器失败（{self.tokenizer_path}）：{exc}") from exc

        try:
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(detail=f"加载分类模型失败（{self.model_path}）：{exc}") from exc

        self._input_names = {item.name for item in self._session.get_inputs()}

    # ------------------------------------------------------------------ #
    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def predict(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Classification]:
        np = self._np
        total = len(texts)
        out: list[Classification] = []
        for start in range(0, total, batch_size):
            window = [t or "" for t in texts[start : start + batch_size]]
            logits = self._logits(window)
            probs = _softmax(np, logits)
            for row in probs:
                index = int(np.argmax(row))
                label = self.labels[index] if index < len(self.labels) else f"label_{index}"
                out.append(
                    Classification(
                        label=label,
                        score=round(float(row[index]), 4),
                        probs={
                            (self.labels[i] if i < len(self.labels) else f"label_{i}"):
                            round(float(value), 4)
                            for i, value in enumerate(row)
                        },
                    )
                )
            if on_progress:
                on_progress(min(start + batch_size, total), total)
        return out

    # ------------------------------------------------------------------ #
    def _logits(self, texts: list[str]) -> object:
        np = self._np
        encodings = self._tokenizer.encode_batch(texts)
        width = max((len(enc.ids) for enc in encodings), default=1)
        width = max(1, width)

        input_ids = np.zeros((len(encodings), width), dtype=np.int64)
        attention = np.zeros((len(encodings), width), dtype=np.int64)
        for row, enc in enumerate(encodings):
            ids = enc.ids[:width]
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1

        feeds: dict[str, object] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        hidden = self._session.run(None, feeds)[0]
        ndim = getattr(hidden, "ndim", 0)
        if ndim == 2:
            return hidden
        if ndim == 3:
            raise AIUnavailable(
                detail=(
                    f"分类模型输出是 3 维张量 {tuple(hidden.shape)}，"
                    "说明该 ONNX 导出不含分类头（只有编码器）。"
                    "请换一个导出了 `logits` 的模型。"
                )
            )
        raise AIUnavailable(detail=f"分类模型输出维度异常：{ndim}")


# --------------------------------------------------------------------------- #
def _softmax(np: object, logits: object) -> object:
    """数值稳定的 softmax。"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _labels_from_config(model_path: Path) -> list[str]:
    """从同目录（或其上级）的 config.json 读取 id2label。"""
    for candidate in (model_path.parent / "config.json", model_path.parent.parent / "config.json"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        id2label = payload.get("id2label")
        if isinstance(id2label, dict) and id2label:
            return [str(id2label[str(i)]) for i in range(len(id2label))]
    raise AIUnavailable(
        detail=(
            f"找不到类别标签：{model_path.parent} 下没有可用的 config.json(id2label)。"
            "请在配置里显式给出 labels。"
        )
    )
