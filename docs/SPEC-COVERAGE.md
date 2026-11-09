# 原始需求覆盖度自查（SPEC-COVERAGE）

> 对照对象：项目首条指令里的《微博舆论研究助手 · AI Agent 开发提示词》。
> 逐条核对，不按印象打分。每条都给**落地位置**与**验证方式**，
> 因为"我说做完了"本身不是证据。
>
> 更新：2026-10-26（第 3 次自查，本轮补齐 A2-3 逐帖目标数）

图例：✅ 完成 ｜ 🟡 部分完成（能用，但和需求原文有差距）｜ ❌ 未做 ｜ ⛔ 你已明确不做

---

## 一、模块 A：登录与检索层（P0）

| 需求 | 状态 | 落地位置 / 证据 |
| --- | --- | --- |
| A1 路径一：弹窗扫码登录 + `storage_state` 复用 | ✅ | `wrc/session.py` `login_interactive()` / `save_state()`；`wrc/cli.py login` |
| A1 路径二：粘贴 Cookie 字符串降级 | ✅ | `wrc/session.py` `import_cookie_string()` + `parse_cookie_string()` |
| A1 验收：带登录态加载 `#话题#` 搜索页不触发 432 | ✅ | 真实采集跑通（会话 `20261025T232335-db20`，5 帖 / 500 条） |
| A2-1 先展示帖子列表预览供勾选，不自动全抓 | ✅ | `pipeline.preview_posts()`（前 20 条）+ 界面「2 · 选择要采集评论的帖子」**复选框列**（☑/☐，空格切换，全选/全不选/反选） |
| A2-2 默认按热度，支持按时间 | ✅ | 帖子：`ORDER_LABELS` 三档（互动总量 / 发帖时间 / 平台顺序），**本地重排**并如实标注；**评论**：`--comment-sort hot|time` + 界面「评论排序」，`sort_applied` 如实记录 |
| A2-3 **每帖评论数独立配置** | ✅ | `CollectOptions.per_post_targets` / `target_for()`；`pipeline.collect()` 逐帖取值；界面「本次目标」列 + 「设置勾选帖子的目标数…」 |
| A2-4 时间窗采集评论（本轮新增） | ✅ | `wrc/weibo/window.py`（`TimeWindow` / `plan_comment_collection` / `WindowCoverage`）；界面时间窗输入 + 近 7/30 天快捷键；CLI `--since/--until` |
| A2-5 按平台特性定制停止条件（本轮新增） | ✅ | 时间单调流「越过窗口起点即停」并宣称覆盖完成；热门流不这么做，报告"窗口未完整覆盖"及原因 |
| **A2-6 检索口径：话题聚合 vs 关键词全文**（本轮新增） | ✅ | `search_query(raw, mode)`；界面单选 + CLI `--search-mode`；口径进参数、报告、`method_note` |
| A2 技术提示：`max_id` 递推翻页 + 空值容错 | ✅ | `weibo/collect.py` `collect_comments()`（cursor 停滞 / 空游标 / 总量不足都有分支） |
| A2 二级评论默认关闭 | ✅ | `include_subcomments=False`；界面默认不勾 |

**A 模块结论：完成。** 勾选已按你的要求改成**复选框**；
原「多选表格」与原文的差异就此消除。

---

## 二、模块 B：数据清洗管线（P0）

| 需求 | 状态 | 落地位置 / 证据 |
| --- | --- | --- |
| B1 轻度：精确去重 / 去纯表情 / 纯链接 / 纯 @ / HTML 剥离 | ✅ | `clean/basic.py` `TIER_STEPS["light"]` + `rules.py` |
| B1 中度：+ 广告关键词表 + 僵尸号 | 🟡 | 广告表 ✅；僵尸号做成**显式可选项且默认关闭**——需求要"粉丝数 <5 **且** 注册 <30 天"，评论接口不返回注册时间，只按粉丝数删会系统性误伤真实低互动用户。理由写在 `OPT_IN_STEPS` 注释与清洗报告里 |
| B1 重度：+ BERT 二分类 + SBERT 余弦 >0.85 协同过滤 | ✅ | `heavy` 档 = `ai_marketing`（与模板语义相似度）+ `ai_coordinated`（簇内相似度）；无模型时该档**不可用而不是静默降级** |
| B1 自定义 | ✅ | 界面逐步勾选框（步骤级开关），不只有三档 |
| B2 各步骤移除数量与占比 | ✅ | `clean/report.py` `CleaningReport.render()` |
| B2 被移除样本随机 20 条 | ✅ | `DEFAULT_SAMPLE_SIZE = 20`，固定种子 `DEFAULT_SEED` 保证"哪 20 条"可复现 |
| B2 一键恢复某条（误杀判定） | ✅ | `CleanResult.restore(cid)`（命令行 `wrc reclean`） |
| B2 清洗前后长度 / 粉丝数分布对比 | ✅ **本轮补齐图** | `charts.cleaning_comparison()` → `cleaning_comparison.png`（长度 density + 粉丝数 log10），报告第一节引用 |
| B3 jieba 分词 + 停用词 + 表情编码为 token | ✅ | `analyze/terms.py` / 聚类前处理；表情在 `community._emoji_count()` 有独立特征 |

**B 模块结论：完成 2 处偏差**（僵尸号降级为可选、分布对比未出图）。两处都在报告里主动披露，不是隐瞒。

---

## 三、模块 C：伦理与匿名化（P0）

| 需求 | 状态 | 证据 |
| --- | --- | --- |
| 导出移除 `user_id` / `screen_name` / `profile_url` / `location` | ✅ | `privacy/anonymize.py` `_DROP_FIELDS` + `assert_no_identifiers()` 兜底断言 |
| location 聚合到省级，原始值移除 | ✅ | `extract_province()`（省级行政区表，含港澳台/海外） |
| 导出前使用声明确认框 | ✅ | `privacy/consent.py` `render_statement()` + 界面「5 · 导出」必须勾选 |
| 聚类人口属性交叉 <5 抑制 | ✅ | `min_n=5` 抑制，显示 `—` **且不计入行内占比分母**；报告第六节 |

**C 模块结论：完成。**

---

## 四、模块 D：分析（P1）

| 需求 | 状态 | 证据 |
| --- | --- | --- |
| D1 双通道（LLM + 本地 BERT） | ✅ | `sentiment.py` 通道一 LLM（含 intensity）、通道二 ONNX 本地模型 |
| D1 信度抽样 `min(200, 5%)`，不一致 >15% 告警并展示样本 | ✅ | `AgreementReport`、`DISAGREEMENT_THRESHOLD=0.15`；报告第二节 |
| D1 区分情感极性与立场 | ✅ | 情感（正面/中性/负面）与立场（支持/反对/中立/无关）**两个独立维度、两套 analysis_id**；报告明确"不要用情感分布代替立场分布" |
| D2 话语聚类（评论级）+ 代表关键词 + 示例 | ✅ | `analyze/cluster.py`（bge-small-zh 嵌入 + KMeans + 轮廓系数） |
| D2 用户社群聚类 + PCA 可视化 | ✅ | `analyze/community.py` + `community_pca.png` |
| D2 **可解释角色标签**（不是 Cluster 0/1/2） | ✅ | `_role_for()` 按特征 z 值命名（如"高互动传播型"），并注明"描述性命名，不做因果判断" |
| D2 K 值不强制预判，先给肘部/轮廓建议区间 | ✅ | `silhouette_curve` + 建议区间再由用户确认 |
| D3 方法卡片（模型/提示词/超参/抽样率/档位/时间戳） | ✅ | `analyze/method_card.py`，报告含 **7 张**卡片，可按模块单独引用 |

**D 模块结论：完成。** 立场的双通道按你的决定不做（单通道 LLM + 人工复核），已在方法卡片写明。

---

## 五、模块 E：AI 助手与交互（P1 / P2）

| 需求 | 状态 | 证据 |
| --- | --- | --- |
| E1 方法顾问（清洗档位、样本量够不够、情感 vs 立场） | ✅ | `assistant/advisor.py` + 界面「问方法助手」+ `wrc ask` |
| E1 主动给 2–3 个带按钮的下一步建议 | ✅ | `suggest_next_steps(context, limit=3)`；界面「下一步建议」按钮（规则驱动、确定性、可复现） |
| E1 结合当前上下文作答（采了多少条、哪个 tag、哪档清洗） | ✅ | `assistant/context.py` `build_context(store)` |
| E2 tag 量级必须真实探测，禁止 LLM 编造 | ✅ | `assistant/topic.py` `probe_tag()` 真发请求；报告"探测到的量级信号"而非总量，并明说口径 |
| E2 附推荐理由 + 查看样例帖子 | ✅ **本轮界面化** | 界面「话题推荐」面板：候选表（话题 / 实测量级 / 理由）+ 「用选中话题检索」+「查看样例帖子」；无样例时现场真查一次 |

**E 模块结论：完成。** 话题推荐已从命令行搬进界面，"查看样例帖子"不再是缺口。

---

## 六、模块 F：导出与研究闭环（P2）

| 需求 | 状态 | 证据 |
| --- | --- | --- |
| F1 原始数据层（匿名化后） | ✅ | `raw/raw_comments.csv` + `raw_posts.csv` + `manifest.json` |
| F1 分析数据层（+ 情感 / 立场 / 聚类 / 用户特征） | ✅ | `analysis/analysis_comments.csv` + 3 张交叉表 CSV + manifest |
| F1 报告层（方法卡片 + 图表 + 清洗报告）Word | ✅ | `report/report.md` + `report.docx`（9 节、7 卡片、9 图） |
| F1 报告层 PDF | ⛔ | 你说不做 |
| F2 研究日志 `research_log.md` | ✅ | `wrc/research_log.py`（逐事件 JSONL → Markdown），导出目录内 |
| F2 可复现性（附加，超出原文） | ✅ | `appendix.md/.docx`：分阶段复现等级、模型 SHA-256、`verify_clean_replay()`；**并记录检索口径 / 时间窗 / 实际覆盖**（`DECISIONS/017`） |
| **对比可比性**（附加） | ✅ | 9 条判据含「检索口径」与「评论时间窗」：窗口不同则绝对量与占比都判为不可比 |
| **口径唯一来源**（附加） | ✅ | `search_mode_text()` / `window_text()` 一处定义，报告/附录/方法卡片/method_note 逐字一致，`test_wording_comes_from_one_place` 盯住 |

**F 模块结论：完成（PDF 除外，已确认不做）。**

---

## 七、Agent 执行规则（硬约束）

| 规则 | 状态 | 证据 |
| --- | --- | --- |
| 1 先技术验证再 UI | ✅ | `spikes/` 探测脚本 → 才写 `session/pipeline` |
| 2 频率纪律（1.5–3s 随机延迟、硬上限、接近上限提示） | ✅ | `weibo/ratelimit.py` `RateLimiter` + `RequestBudget`；本轮又加了"逐帖目标总和 vs 上限"的**采集前**预警 |
| 3 LLM 可替换（不硬编码厂商 SDK） | ✅ | `llm/client.py` OpenAI 兼容协议；`LlmClient` 抽象，换后端只改 base_url/model |
| 4 每一步可见输出 | ✅ | `emit(level, msg)` 统一进度通道（GUI 日志窗 + CLI 终端） |
| 5 错误信息人性化 | ✅ | `errors.py` `render_error()`：发生了什么 / 可以怎么做，不出 traceback |
| 6 不擅自扩范围 | ✅ | P2/PDF/二级评论/平台 AI 清洗均按你的决定未启动 |

---

## 八、交付节奏

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 1 技术验证 + 架构文档 | ✅ | `docs/ARCHITECTURE.md` |
| 2 P0 管线（CLI 可用） | ✅ | `wrc run` / `preview` / `reclean` / `clean` |
| 3 P1 分析 + 方法卡片 | ✅ | `wrc analyze` / `report` |
| 4 P1 界面 + 助手 | 🟡 | 界面完成、`smoke_gui` 通过；主按钮已合并为「一键完成：采集→清洗→分析→报告」；但"非技术用户无说明书跑通全流程"**必须由真人验证** |
| 5 P2 推荐 + 导出 + 日志 | 🟡 | 导出与日志 ✅；话题推荐只在命令行 |

---

## 九、还差什么（按宪法优先级排序）

| 序 | 缺口 | 优先级 | 成本 | 卡在哪 |
| --- | --- | --- | --- | --- |
| 1 | 真人可用性验证（第 4 步验收） | P1 验收 | — | **只能你或同学来做**（唯一阻断项） |
| 2 | 二级评论可视化关系 | P3 | 中 | 你已决定暂不采集二级评论 |
| 3 | PDF 导出 | P2-F1 | 中 | ⛔ 你已决定不做 |

其余模块均已收口：A（检索/复选框/逐帖目标/时间窗/排序/检索口径）、
B（三档+自定义+清洗报告+恢复+分布图）、C（匿名化+声明+小群体抑制）、
D（双通道情感+信度抽样+两类聚类+角色标签+7 张方法卡片）、
E（方法顾问+按钮建议+上下文+话题推荐界面）、F（分层导出+Word+研究日志+附录）。

工程保障（超出原文）：口径唯一来源与四处逐字一致、交付物自检
（`tools/check_deliverables.py`，11 个必需文件 + 隐私 + 结构 + 口径四层）、
一键自检 `tools/check.py` 串起编译/测试/两次冒烟/交付物检查。

**除此之外没有已知的功能性缺口。** 需要说清的三个"数据层"限制（不是功能缺失，是结论可靠性）：
聚类轮廓系数 0.1327（簇结构弱）、立场为单通道 LLM（无第二编码员）、
2024 年公布日与数据相差两年（已按实施期定稿，不再补采）。

另有一个**平台层**限制（不是实现问题）：热门流不保证时间窗覆盖。
要可辩护的窗口覆盖，必须 `--comment-sort time` + 网页端后端；
这在界面上直接标注，报告里也会写"窗口未完整覆盖"。
