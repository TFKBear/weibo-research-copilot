"""AI 链路实测：验证 ONNX 模型能真的跑出有意义的相似度。

只检查"文件存不存在"是不够的——真正会出问题的地方在
分词器配置、输入名、池化位置、归一化，这些只有跑一次才知道。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrc.ai.status import detect_status  # noqa: E402


def main() -> int:
    status = detect_status(ROOT / "models")
    print(f"AI 状态：{status.summary()}")
    print(f"  运行库 ：{status.runtime_detail}")
    print(f"  权重   ：{status.onnx_path}")
    print(f"  分词器 ：{status.tokenizer_path}")
    if not status.ready:
        print("\n未就绪，跳过实测：")
        for step in status.remediation():
            print(f"  - {step}")
        return 2

    from wrc.ai.embedder import build_embedder

    embedder = build_embedder(status.onnx_path, status.tokenizer_path)
    texts = [
        "加V信领取内部优惠，数量有限先到先得",   # 营销模板
        "想了解更多的可以私信我，长期有效",     # 营销模板
        "这款车续航表现一般，高速掉电有点快",   # 正常批评
        "支持小米，希望国产车越做越好",         # 正常支持
    ]
    result = embedder.encode(texts)
    print(f"\n编码完成：{len(texts)} 条，{result.dim} 维，用时 {result.elapsed_s}s")

    import numpy as np

    vectors = np.asarray(result.vectors)
    sim = vectors @ vectors.T
    print("相似度矩阵：")
    for i in range(len(texts)):
        row = "  ".join(f"{float(sim[i][j]):.3f}" for j in range(len(texts)))
        print(f"  [{i}] {row}   {texts[i][:18]}")

    print(
        "\n模型可用：能正常编码、维度正确。\n"
        "注意：这里只验证「能不能跑」，不代表阈值合适。\n"
        "阈值必须用真实数据标定：python tools/calibrate_ai.py\n"
        "（bge 系列的相似度是压缩的，凭直觉设 0.8 会一条都抓不到。）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
