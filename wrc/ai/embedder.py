"""ONNX 句向量编码器。

为什么用 ONNX 而不是 PyTorch / sentence-transformers：
- 体积：onnxruntime 约 20MB，torch CPU 约 200MB+，打包分发差别巨大；
- 启动：onnxruntime 加载 23–91MB 的模型是秒级，torch 要几秒；
- 依赖：只要 numpy + tokenizers，不需要编译工具链。

模型：BAAI/bge-small-zh-v1.5（Xenova 提供的 ONNX 导出），
中文短文本检索效果稳定，CLS 池化 + L2 归一化。

⚠ 这一层只负责"把文本变成向量"。判定逻辑全部在 analysis.py 的纯函数里，
   因为判定才是研究结论的一部分，必须能脱离模型单独测试。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..errors import AIUnavailable

#: 单条文本的最大 token 数。评论区几乎没有长文本，128 足够且快很多。
DEFAULT_MAX_LENGTH = 128
#: 一次推理多少条。CPU 上 16 条一批是比较稳的折中。
DEFAULT_BATCH_SIZE = 16
#: 句向量的最小合理维度。低于它基本可以断定指错了模型。
MIN_PLAUSIBLE_DIM = 32


@dataclass(slots=True)
class EmbeddingResult:
    vectors: object  # numpy.ndarray, 形状 (n, dim)
    dim: int
    elapsed_s: float
    model_name: str


class OnnxEmbedder:
    """把文本编码成归一化的句向量。"""

    def __init__(
        self,
        onnx_path: Path,
        tokenizer_path: Path,
        *,
        model_name: str = "",
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.model_name = model_name or self.onnx_path.stem
        self.max_length = max_length

        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(
                detail=f"导入 AI 运行库失败：{type(exc).__name__}: {exc}"
            ) from exc

        self._np = np
        try:
            self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            self._tokenizer.enable_truncation(max_length=max_length)
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(
                detail=f"加载分词器失败（{self.tokenizer_path}）：{exc}"
            ) from exc

        try:
            self._session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(
                detail=f"加载 ONNX 模型失败（{self.onnx_path}）：{exc}"
            ) from exc

        self._input_names = {item.name for item in self._session.get_inputs()}
        self._output_names = [item.name for item in self._session.get_outputs()]

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return 1  # 占位：真实维度在第一次编码后才知道

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """批量编码。返回 L2 归一化后的向量矩阵。"""
        np = self._np
        started = time.monotonic()
        total = len(texts)
        if total == 0:
            return EmbeddingResult(
                vectors=np.zeros((0, 0), dtype=np.float32),
                dim=0,
                elapsed_s=0.0,
                model_name=self.model_name,
            )

        chunks: list[object] = []
        for start in range(0, total, batch_size):
            window = [t or "" for t in texts[start : start + batch_size]]
            chunks.append(self._encode_batch(window))
            if on_progress:
                on_progress(min(start + batch_size, total), total)

        matrix = np.vstack(chunks).astype(np.float32)
        # L2 归一化后，余弦相似度退化为点积，后面算相似度快很多
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        # 维度自检：句向量至少几十维。若只有 2–10 维，几乎可以肯定是把
        # **分类模型**的 logits 当成了句向量（两种模型都叫 *_quantized.onnx，
        # 目录指错就会静默互换）。这种错误不会抛异常，只会产出毫无意义的结果，
        # 所以在出口处主动拦下来。
        if matrix.shape[1] < MIN_PLAUSIBLE_DIM:
            raise AIUnavailable(
                detail=(
                    f"句向量维度只有 {matrix.shape[1]} 维，不像句向量模型"
                    f"（{self.onnx_path}）。很可能 ai_embedding_model_dir 指到了"
                    "情感/分类模型目录。"
                )
            )

        return EmbeddingResult(
            vectors=matrix,
            dim=int(matrix.shape[1]),
            elapsed_s=round(time.monotonic() - started, 2),
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------ #
    def _encode_batch(self, texts: list[str]) -> object:
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

        outputs = self._session.run(None, feeds)
        hidden = outputs[0]
        if getattr(hidden, "ndim", 0) == 3:
            # (batch, seq, hidden) → 取 [CLS] 位置的向量（BGE 系列用 CLS 池化）
            return hidden[:, 0, :]
        return hidden


def build_embedder(
    onnx_path: Path,
    tokenizer_path: Path,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> OnnxEmbedder:
    return OnnxEmbedder(
        onnx_path, tokenizer_path, model_name=onnx_path.stem, max_length=max_length
    )
