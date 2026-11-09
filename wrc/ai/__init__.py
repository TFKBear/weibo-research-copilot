"""可选的 AI 增强层。

设计立场（对应产品宪法第 1、2 条）：

- **AI 是可选档位，不是默认路径。** 不启用时工具完全可用，纯规则表清洗。
- **AI 的每个判断都必须能说出理由。** 因此这一层不做"黑箱二分类"，而是：
  1. 把文本编码成句向量；
  2. 与**可读的营销模板**比对，输出"与哪条模板相似、相似度多少"；
  3. 在数据集内部找高度相似的评论簇（协同行为），输出"这一簇有 N 条互相相似"。
- **运行时不引入重量级框架。** 用 ONNX Runtime 而不是 PyTorch：
  模型 91MB（或量化版 23MB）、运行库约 20MB，打包进 exe 也带得动。
"""

from __future__ import annotations

from .analysis import AIAnalysis, analyse
from .status import AIStatus, detect_status

__all__ = ["AIAnalysis", "AIStatus", "analyse", "detect_status"]
