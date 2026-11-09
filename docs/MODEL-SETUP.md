# AI 模型安装说明（重度清洗档 / AI 辅助）

> **本机状态：已安装完成。**
> `models/bge-small-zh-v1.5/model_quantized.onnx`（22.90 MB）+
> `models/bge-small-zh-v1.5/tokenizer.json`（0.42 MB），
> 运行库 `onnxruntime` / `tokenizers` 已就绪，不需要 torch。
> 校验命令：`python -m wrc.cli ai`

AI 是**可选**的。不装模型时，轻度/中度档完全可用，所有功能不受影响。
下面是换机器或重装时的完整步骤。

---

## 一、要下载什么

只需要**两个文件**，加起来约 23MB（量化版）或 91MB（全精度版）：

| 文件 | 直链 | 大小 |
| --- | --- | --- |
| 量化版权重（**推荐，体积小**） | https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model_quantized.onnx | 22.9 MB |
| 全精度版权重（效果略好） | https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx | 90.5 MB |
| 分词器（**必须**） | https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json | 约 0.3 MB |

模型仓库主页（想自己挑文件用这个）：
https://huggingface.co/Xenova/bge-small-zh-v1.5

> 两个链接在本机实测可直连（HTTP 200）。如果打不开，用镜像
> `https://hf-mirror.com/Xenova/bge-small-zh-v1.5`，目录结构完全一样。
>
> 为什么是 `Xenova` 这个仓库而不是 `BAAI` 官方仓库：官方只提供 PyTorch 权重
> （需要装 torch，200MB+），而 Xenova 提供了等价的 ONNX 导出。
> 本工具用 ONNX 运行，不需要 torch。

---

## 二、放到哪里

在项目根目录下建这个结构（`models/` 目录工具会自动创建）：

```
weibo-research-copilot/
└── models/
    └── bge-small-zh-v1.5/
        ├── model_quantized.onnx
        └── tokenizer.json
```

**这两个文件放在 `models/` 下面的任意子目录都可以**，工具会自动递归查找。
所以你也可以直接把整个仓库解压进来（那样会是 `models/onnx/model.onnx` + `models/tokenizer.json`），同样能认出来。

如果两个权重都在，工具优先用全精度的 `model.onnx`。

---

## 三、确认是否安装成功

三种方式任选：

```powershell
# 方式 1：命令行
python -m wrc.cli ai

# 方式 2：环境自检
python -m wrc.cli doctor

# 方式 3：界面里点「重新检测」，看「AI 状态」那一行
```

成功时显示类似：

```
AI 就绪（model_quantized.onnx，23MB）
```

失败时它会明确告诉**缺哪一个文件**，不会只说"AI 不可用"。

---

## 四、运行库（一般不用管）

本机已安装 `onnxruntime` 与 `tokenizers`。若换一台机器提示缺运行库：

```powershell
pip install onnxruntime tokenizers
```

---

## 五、AI 到底做了什么（以及它不做什么）

启用后，重度档会在规则清洗之外再做两类判断，**两个都不给黑箱结论**：

1. **营销话术识别**
   把每条评论与 12 条可读的营销模板做语义相似度比较（默认阈值 **0.62**）。
   判定结果会写成 `ai_marketing:0.71`，意思是"与最接近的模板相似度 0.71"。
   这 12 条模板可以直接写进论文附录。

2. **协同行为识别**
   在**你这批数据内部**找互相高度相似（默认 ≥0.85）且至少 3 条的簇，
   判定结果写成 `ai_coordinated:簇内4条`。
   这一项不需要任何模板，因为它找的是"同一段文案被反复投放"这个现象本身。

**它不做的事**：不做情感分析、不做主题分类、不判断立场。
那些属于 P1 分析模块，会单独出方法卡片。

**重要提醒**：高相似度簇不等于水军。亲友团、粉丝控评、同一个梗刷屏，
都会形成高相似簇。工具只负责把"可能存在协同"的样本标出来，
**结论必须由研究者自己下**——清洗报告里会明确写这一点。

---

## 五之二、阈值是量出来的，不是拍出来的

这一条比安装步骤更重要。

bge 系列在中文短句上的余弦相似度是**压缩**的：两句都在推销、措辞完全不同的话，
相似度也只有 0.6 上下。如果凭直觉把阈值设成 0.8，结果会是
**一条都抓不到**，而界面上只会显示"AI 识别 0 条"——功能看起来正常，实际完全失效。

### 标定方法

```bash
python tools/calibrate_ai.py
```

它用 12 条自然口吻的营销文案当正样本、你的真实评论当负样本，
算出两组「与最近模板的相似度」分布，给出建议阈值。

**本机实测结果**：

| | 与最近模板的相似度 |
| --- | --- |
| 营销文案（12 条） | 0.644 – 0.844，中位 0.734 |
| 真实评论（80 条） | 0 – 0.606，中位 0.515 |

两者完全可分 → 取 **0.62**。

> **换模型、换语料、换领域，都必须重新跑一次标定。**
> 阈值不随模型迁移，只随语料成立。

### 预览 AI 会删掉什么

正式跑重度档之前，可以先看：

```bash
python tools/ai_preview.py            # 用最近一次采集结果
python tools/ai_preview.py --run 20261025T010257
```

只读，不修改任何数据。

---

## 六、成本与限制

| 项目 | 实测值 |
| --- | --- |
| 首次加载模型 | 约 1 秒 |
| 编码速度（本机 CPU） | **约 230 条/秒**（50 条 0.22 秒） |
| 1000 条评论 | 约 5 秒 |
| 内存占用 | 约 300–400MB |
| 是否需要 GPU | 不需要 |

阈值可调（环境变量或 `.env`）：

```bash
WRC_AI_MARKETING_THRESHOLD=0.62    # 调低=更激进，误杀更多
WRC_AI_COORDINATED_THRESHOLD=0.85
WRC_AI_MIN_CLUSTER_SIZE=3          # 至少几条才算一簇
```
