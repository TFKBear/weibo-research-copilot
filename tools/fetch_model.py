"""下载本地情感分析模型（可选组件，走 ONNX，不需要 torch）。

## 选型过程（2026 年在 HuggingFace 上实测，不是拍脑袋）

按下载量与口碑筛出三个候选，再**用真实微博评论对比**：

| 模型 | 下载量 | 收藏 | 发布 | 类别 | 微博口语表现 |
| --- | --- | --- | --- | --- | --- |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | ~75 万 | 279 | 2022 | 3 类 | ✅ 采用 |
| `tabularisai/multilingual-sentiment-analysis` | ~12.7 万 | 398 | 2024 | 5 类 | ❌ 实测不可用 |
| `lxyuan/distilbert-...-sentiments-student` | ~58 万 | 316 | 2023 | 3 类 | 原默认，可用但偏弱 |

**为什么最终选 2022 年的那个而不是 2024 年的新模型**：
新模型类别更多、收藏更多，但在真实微博评论上判定混乱——"打卡"被判 Very Negative
（0.66）、"年轻人交一辈子都拿不回来"被判 Very Positive。置信度普遍在 0.3–0.5，
说明它在中文口语上基本没有学到东西。

而 twitter-xlm-roberta 是在 **1.98 亿条推文**上训练的（八种语言，含中文），
社交媒体的口语、反讽、话题标签与微博最接近。**"新"不等于"合适"**，
这一条要写进方法卡片：模型选择依据是实测表现，不是发布时间。

## 用法

```bash
python tools/fetch_model.py                  # 下载推荐模型（约 280MB）
python tools/fetch_model.py --model old      # 旧的 3 类 distilbert（对照用）
python tools/fetch_model.py --check          # 只检查现状
python tools/fetch_model.py --force          # 重新下载
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 可选模型注册表。新模型要经过"真实数据对比"才能进这张表，
#: 否则就变成"按下载量选模型"——那是把选型外包给排行榜。
MODELS: dict[str, dict[str, object]] = {
    "twitter-xlmr": {
        "repo": "onnx-community/twitter-xlm-roberta-base-sentiment-ONNX",
        "target": "models/sentiment-twitter-xlmr",
        "note": "多语言（含中文）推文情感三分类；社交媒体域匹配最好，推荐使用",
        "files": (
            ("onnx/model_quantized.onnx", "model_quantized.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("special_tokens_map.json", "special_tokens_map.json"),
        ),
    },
    "distilbert-5c": {
        "repo": "onnx-community/multilingual-sentiment-analysis-ONNX",
        "target": "models/sentiment-v2-multilingual",
        "note": "2024 年 5 类多语言模型；实测在中文微博上判定混乱，**不推荐**（保留供对照）",
        "files": (
            ("onnx/model_quantized.onnx", "model_quantized.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
            ("special_tokens_map.json", "special_tokens_map.json"),
        ),
    },
    "old": {
        "repo": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "target": "models/distilbert-sentiment-zh",
        "note": "原默认 3 类模型；可用但置信度偏低",
        "files": (
            ("onnx/model.onnx", "model_quantized.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
        ),
    },
}

DEFAULT_MODEL = "twitter-xlmr"


def _download(client, base: str, remote: str, local: Path, *, force: bool) -> None:
    if local.exists() and not force and local.stat().st_size > 0:
        print(f"  跳过（已存在）：{local.name}（{local.stat().st_size/1e6:.1f}MB）")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    with client.stream("GET", f"{base}/{remote}", follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 16):
                handle.write(chunk)
                done += len(chunk)
                if total and done % (64 << 20) < (1 << 16):
                    print(f"    {local.name}: {done/1e6:.0f}/{total/1e6:.0f}MB", flush=True)
    tmp.replace(local)
    print(f"  完成：{local.name}（{local.stat().st_size/1e6:.1f}MB）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--check", action="store_true", help="只检查现状")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    spec = MODELS[args.model]
    target = ROOT / str(spec["target"])
    files = spec["files"]

    print(f"模型：{args.model}")
    print(f"  来源：{spec['repo']}")
    print(f"  说明：{spec['note']}")

    if args.check:
        print(f"  目录：{target}")
        for _remote, local_name in files:  # type: ignore[union-attr]
            path = target / local_name
            state = f"{path.stat().st_size/1e6:.1f}MB" if path.exists() else "缺失"
            print(f"    {local_name:<28}{state}")
        return 0

    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        print(f"缺少 httpx：{exc}（pip install httpx）")
        return 2

    base = f"https://huggingface.co/{spec['repo']}/resolve/main"
    print(f"  下载到：{target}")
    with httpx.Client() as client:
        for remote, local_name in files:  # type: ignore[union-attr]
            try:
                _download(client, base, remote, target / local_name, force=args.force)
            except Exception as exc:  # noqa: BLE001
                print(f"  失败：{local_name} —— {type(exc).__name__}: {exc}")
                return 1

    config = target / "config.json"
    if config.exists():
        labels = json.loads(config.read_text(encoding="utf-8")).get("id2label")
        print(f"  类别：{list(labels.values()) if labels else '未知'}")

    print()
    print("下一步：")
    print("  python -m wrc.cli ai          # 确认模型就绪")
    print("  python tools/run_analysis.py  # 重跑分析（本地通道会用新模型）")
    print()
    print("提示：换模型后请重跑一次标定，旧阈值不一定适用：")
    print("  python tools/calibrate_ai.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
