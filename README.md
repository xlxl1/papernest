# PaperNest · 文巢

个人科研文献 Agent：把「搜文献 → 读文献 → 攒文献 → 引文献 → **写文献**」装进一个有长期记忆的库。
核心成本结构：**每篇论文只让 LLM 精读一次，之后全部走缓存与 RAG**。

> 立项方案文档目前保留在本地项目上级目录，未纳入代码仓库；如需公开展示，可另行放入
> `docs/` 后再补充链接。

## 仓库结构

```text
papernest/
├── papernest/          # 核心 Python 包：检索、RAG、Agent、写作与导入
├── web/index.html      # 单页 Web 界面
├── tests/              # 离线确定性测试（unittest）
├── cli.py              # 命令行入口
├── requirements.txt    # Python 依赖
├── docker-compose.yml  # 应用启动配置
└── .env.example        # 配置模板（不含密钥）
```

运行时数据库、PDF、向量索引、上传文件和评测报告保存在 `data/`，已由 `.gitignore`
统一排除；真实密钥只放在本地 `.env`，不要提交到远程仓库。

如果要发布到 GitHub，建议先执行：

```bash
git add -A
git status --short                 # 确认没有 .env、数据库或 PDF
git diff --cached --check          # 检查空白字符和冲突标记
git commit -m "chore: prepare repository for GitHub"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 当前状态：四周计划主体完成 · 二期写作流水线已落地 · 三期本地入库与引文网络已落地

- [x] **W1 Agent 核心**：Router/Planner（确定性意图路由）· Tool registry（5 工具）· 有界状态机执行 · `/api/agent/run` · 执行轨迹落库可回放
- [x] **W2 证据与评测**：页级 chunk（PyMuPDF 按页入库）· evidence quote（支撑证据句 + 机械回取校验）· claim verification（L2 卡片逐条 + 综述逐句）· **自建评测集**（当前 82 条：检索 32 关键词串 + 32 自然语言形态 · QA 10 · 引用 8） · **三指标评测脚本** · **QASPER 公开基准接入**（paper_hit@k + 证据句 Recall 两层指标）
- [x] **W3 工程化**：异步任务（SQLite 任务表）· SSE 实时进度（含心跳保活）· token / latency / cost 逐次落账 · 步骤级超时（协作式取消）+ 瞬态故障重试（4 次抖动退避）· 离线 Demo（`cli.py demo`）· 流式对话（`/api/chat/stream`）· **RRF 混合检索 + 口径消融** · **规模实验命令**（`cli.py scale`，吞吐/去重/p50/p95）· **换提供方向量零成本迁移**
- [x] **W4 包装与展示**：生成综述入口 · 执行时间线 UI（Agent 页 + 历史回放）· **写作台（润色 + 评审）** · **文献汇报 PPT** · Docker 一键启动 · 架构图（下）· 数字一览（文末）
- [x] **二期·多 Agent 写作流水线**：选题 / 大纲 / 撰写 / 文献 / 润色五 Agent 确定性流水线（见下节）
- [x] **三期·把「个人文献库」补完整**（见下节）：**本地 PDF 上传入库** · **BibTeX / RIS / Zotero CSV 导入** ·
      **引文网络（阅读缺口 + 共被引推荐）** · **服务端多轮对话记忆** · 迭代检索（PRF 补位，
      当前库重测 Recall@15 0.8125→0.8281，**+0.0156 在噪声内，方向为正但不显著**）·
      课题工作区网页端
- [x] **四期·发现与阅读**：**L2 全文进全库检索**（QASPER hit@1 0.410→0.467）· **修好 QASPER 基准链路**
      （下载源全部 404 + 数据格式已变，此前是坏的）· **原文按页渲染，证据链在界面闭环** ·
      arXiv 新论文订阅（批内 IDF + 广度折扣）· 结构化对比矩阵（页码逐条校验）·
      PDF 版面结构解析（章节切分 + 参考文献抽取 → 反哺引文图）· **491 条自动化测试**
- [x] **五期·收尾与生态**：**迭代问答闭环接线**（`ask --deep` / `/api/answer/deep`，此前实现了但没有任何入口）·
      综述与写作选题切到迭代检索口径 · 对话历史 UI（回放/继续/删除）· **Obsidian 笔记导出**（citekey 双链）·
      Ollama/llama.cpp 本地模型配置样例
- [x] **六期·RCS 增强问答**：LLM 重排 + 逐篇定向摘要 + **依据句逐页机械回取校验**（比 PaperQA2 多的一层）·
      A/B 实测（gold 引用覆盖 0.818→1.0；无证据率无可靠改善，如实记录）· **533 条自动化测试**
      （RCS 的 22 条用例大半在测失败模式而非 happy path）
- [x] **七期·文档处理深度**：**按章节切分检索单元**（等上下文预算下证据召回是按页切的
      2.2~3.2 倍，n=69 配对检验 p=0.0013）· 表格感知切分（整行不截断、表头跨块重复）·
      **多格式入库**（Word/PPT/HTML/Markdown/纯文本，三态计数）· 章节树两级检索 ·
      检索精排层（四后端可插拔，**实测 BM25 精排是净负收益**）· **710 条测试**
- [x] **八期·向量层收口**：Milvus 后端抽象与同步实现（`PAPERNEST_VECTOR_BACKEND` 默认 `milvus`）——
      ⚠️ **但主检索热路径仍走 SQLite BLOB + NumPy 精确 kNN，从不 import 这一层**，
      `get_store()` 的调用方只有 `cli.py vec` 子命令。**不能说「线上检索由 Milvus 承载」**（接线是待办）·
      **embedding 与 chat 分开配置**（`EMBED_API_BASE`/`EMBED_API_KEY`——此前向量路因端点无 `/embeddings` 路由而**静默退化成纯 FTS**，
      修复后自建集 Recall@5 0.677→0.943；⚠️ 该数字在 20.7% 向量覆盖率下测得，见「数字一览」的前提说明）·
      784 条章节块向量真正接进 `search_hybrid` · Cross-Encoder 精排真接上验证（召回饱和时 0/32 题变化，默认 off）
- [x] **九期·审计与收口（2026-09-03）**：全仓生产化审计（14 维度 · 每条发现两个独立视角对抗核验）后修掉三条：
      **① `index_paper` 的删除少了 `kind` 过滤**，一次 `cli.py embed` 会静默抹掉 715/784 条 chunk 向量（真库副本验证零丢失）·
      **② 降级信号只算不传**——`search_hybrid` 的降级文案从未被返回，调用方判 `mode == "fts"` 而 mode 实为 `"fts+chunks"`，
      向量整路挂掉时一句提示都没有；新增 `papernest/degrade.py`（稳定 code 词汇表）+ 健康探针向量覆盖率闸门 ·
      **③ 查询预处理**：标点归一、CJK 切三元窗口（实测 FTS5 trigram 按**字符**计，二元词面命中恒为 0）、
      LIKE 兜底不再按入库顺序排——自然语言提问 Recall@5 0.5312→0.7552（p=0.0071）·
      新增 `papernest/stats.py`（符号翻转配对检验）与 32 条自然语言形态评测题
      · **④ 部署安全收口**：默认零鉴权 + 前端不带口令导致「锁不可启用」、`.bib` 可写入的 `oa_pdf_url` 构成 SSRF 链路，均已修（见「工程加固」）
      · **⑤ 并发与阻塞点**：`assign_indices` 的读-改-写竞态（实测 24 线程并发只拿到 5 个不同编号、映射表丢 19 篇）· `upload_pdfs`/`import_bibliography_file` 是 `async def` 却跑同步 IO 占死事件循环 · agent 的「硬截止」只截断调用方（改成协作式取消，见 `papernest/deadline.py`）
      · **⑥ 上下文组装**：把块级命中接进 `rag.prepare`，并用新增的 `cli.py qasper ctxab` 三臂配对实测——「换成按章节块组装」是**负收益**，已否决并保持默认关闭（见「上下文组装：一条负结果」）
      · **⑦ 对自己这批修复做独立对抗审计**（7 维度 · 每条再核验）：确认 23 条、修掉其中 8 条 must-fix——包括非 ASCII 口令让 /api/* 恒 500、前端三处资源加载绕过口令、`rcs`/`agent` 两条链路的降级仍在被丢弃（正是修复 ② 要消灭的形状）、SSRF 防线把库里 5 篇 arXiv 的 http:// 链接永久挡死、`stats.describe` 对负结果说「方向为正」· **1155 条测试**
- [ ] Demo 视频录制（材料就绪，待录）
- [x] **补齐全库向量**（2026-09-03，443 篇 / 约 3.5k 条嵌入）：覆盖 20.7% → **100%**，
      线上口径全部重测（见「评测」一节的配对对比）；`chunk` 向量 784 条**零丢失**，
      是修复 ① 的生产验证
- [ ] 补难题进评测集：现有题目分辨不出「向量覆盖 20.7% vs 100%」（各指标只变 1 题）

## 路线图（对标 2024-2026 开源生态调研）

按「价值/实现成本」排序，来自对 PaperQA2 / STORM / papersgpt / gpt-researcher /
react-pdf-highlighter / ragas 等项目的对比调研：

1. ~~**RCS 检索增强**~~ —— **已实现**（见上文「RCS 增强问答」，含 A/B 与风险清单）；
   剩余待办：摘要串行 → 并行化、summary 字段本身的 claim 级校验、按问题类型分流
   （聚焦型问题受益、列举型问题可能受损）；
2. **ragas 式 faithfulness 指标**：把回答拆成 claim 逐条对照上下文用 LLM 判真——与现有的
   「无证据率」（机械判 [n] 有无）互补，一个查「有没有引」、一个查「引得对不对」；
3. **PDF 高亮标注 + 选中即问**（react-pdf-highlighter，标注落库不改原 PDF）——个人文献工具的体验分水岭；
4. **Zotero 单向导入 + note 写回**（pyzotero，只写 child note、412 冲突重试——papersgpt 已验证这条路）；
5. **MCP server 输出**：把检索/问答/引文网络暴露为 MCP 工具，接入 Claude/IDE 生态；
6. **Undermind 式检索 Agent**：LLM 相关性评审 + 覆盖率停止条件，叠在已有 PRF 之上；
7. **STORM 视角引导的综述规划**（高成本高差异化，二期）。

## 三期 · 把「个人文献库」补完整

一二期能搜、能读、能写，但有三个洞让它称不上「个人」文献库：**手上的 PDF 进不来**、
**Zotero 里攒的几百篇搬不进来**、**只能管已有的、发现不了该读而没读的**。三期补的就是这三件事。

**本地 PDF 上传入库**（`papernest/pdfimport.py`）：字节流 → sha256 与 `norm_key` 双重去重 →
版面启发式抽元数据 → 按页抽全文 → 可选生成卡片。元数据抽取用**最大字号文本块**定标题、
正则扫 DOI/arXiv、`Abstract` 小标题定位摘要；**抽不准的字段如实留空并写进 warnings，绝不编造**
（抽不出作者就是空，不会拿页眉凑数），可在网页或 `cli.py fix-meta` 逐条修正——改标题/DOI 会重算
`norm_key`，与已有论文撞车时**报冲突而不是静默合并**。落盘文件名恒为 sha256，用户传来的
文件名不参与任何路径拼接（路径穿越用例已覆盖）。

**BibTeX / RIS / Zotero CSV 导入**（`papernest/bibimport.py`）：手写解析器（不引第三方依赖），
处理嵌套花括号 `{The {BERT} Model}`、`@string` 宏、LaTeX 重音转义、两种 author 写法、
`%` 注释行、折行的 DOI/URL；**一条 entry 解析失败不影响其余条目**，如实计入 errors。
入库按 `norm_key` 去重，已存在的走「**补空不覆盖**」——缺摘要就补摘要，已有的标题不会被覆盖。
与 `cite.py` 的导出做了**往返一致性测试**（导出 → 解析回来 → 逐字段相等）。

**引文网络**（`papernest/graph.py`）：从 Semantic Scholar 顺带把 references / citations 拉成边，
**边用 `norm_key` 存而不是 `paper_id`**——库外论文日后被正常检索入库时自动接上。有了边就能算
两件检索算不出来的事：
- **阅读缺口**：被库内多篇引用、自己却不在库里的论文。实测 23 篇论文拉出 2093 条边后，
  排第一的是被库内引用 **10 次**的 Marzetta 2010（Massive MIMO 奠基作）——库里一直没有它。
- **相关推荐**：共被引 + 文献耦合，不是词面相似，也不调 LLM。

**服务端多轮对话记忆**（`papernest/chat.py`）：原来的「多轮」是假的——前端确实发了最近 12 条历史，
但 `rag.prepare` 只取最后一条 user 消息，其余全丢。现在会话是一等对象：历史与**每轮实际进上下文的文献**
一起落库（可回放、可审计）；**会话内 `[n]` 编号固定指向同一篇论文**；追问（「它的局限是什么」
「第 3 篇怎么做的」）会把上文问题拼进检索查询（0 token），被点名的来源即使本轮没召回也会拉回上下文。

```bash
python cli.py import-pdf ~/papers/            # 整个目录入库，按内容去重
python cli.py fix-meta 512 --title "正确的标题" --year 2023
python cli.py import-bib zotero-export.bib    # 从 Zotero / EndNote 搬家
python cli.py graph fetch --limit 30          # 拉引文边
python cli.py graph gaps                      # 该读而没读的文献
python cli.py graph related --ids 146,355     # 共被引 + 文献耦合推荐
```

## 二期 · 多 Agent 写作流水线（写作系统）

一期缺的最后一环「写」：**选题 → 大纲 → 逐节撰写（⇄ 文献）→ 润色 → 机械终检**，
五个 Agent 是职责分离（独立 prompt / 产物落库 / 缓存键 / 模型档位），由确定性流水线串联，
两个人工检查点（定题、改纲）——不是五个模型自由对话，也没有引入 LangGraph（固定 DAG
用一期任务表 + 状态机就够，接缝留在 `pipeline.py`，需要时可单文件替换编排器）。

**与「论文生成器」的分界线——引用白名单制**：撰写 Agent 可用的每条 `[n]` 必须来自
文献 Agent 放行的白名单（证据句通过机械回取校验的库内文献）；白名单外的编号在生成期
被剥离并如实计数，终检报告给出**草稿引用核验率**；库内外都找不到支撑的论断如实标
「（待补证据）」，**不生成实验数据，不编造文献**。导出稿（Markdown / Word）自带
「AI 参与说明」页脚与 GB/T 7714 参考文献。

**节级内容寻址缓存**：`hash(prompt 版本, 节标题, 论点, 字数, 术语表)` 为键（不含大纲
版本——改别处不影响本节），重跑未变节 **0 token**；大纲人工修改版本 +1 可回滚，
失败重跑不覆盖旧版。

**质量闭环**：每节初稿 → 结构化评审（复用一期 review：总评 / 1-10 分 / major-minor
分级问题）→ 评分 <7 或有 major 带意见重写，上限可配（默认 2 次），仍不达标如实交付；
全文装配后跑**不加 LLM 的机械终检**（悬空 / 未核验编号、待补证据数、字数、引用核验率）。

网页「写作流水线」页：五阶段进度条 + 选题候选（证据句可点开原文）+ 大纲编辑器 +
章节草稿（[n] 角标悬浮看证据句、点击进文献详情）+ 终检报告 + 一键导出。

```bash
# CLI 全流程（每个阶段都是异步任务，检查点人工确认；离线不配 key 也能走通）
python cli.py write "大模型 Agent 的评测方法研究"   # 创建 + 选题 Agent（候选带 [n] 证据）
python cli.py write-pick <run_id> 1                 # 检查点①：按候选序号定题 → 大纲
python cli.py write-outline <run_id>                # 查看大纲（机械校验备注）
python cli.py write-go <run_id>                     # 检查点②：逐节撰写⇄文献白名单⇄评审重写
python cli.py write-status <run_id>                 # 各节状态 / 评审分 / 引用数
python cli.py write-finish <run_id>                 # 润色 + 机械终检 + 导出 md/docx
# 网页：POST /api/write/runs → 检查点 /topic → /outline → /polish，GET /export 下载
```

学术诚信红线（写进代码）：不生成实验数据 / 结果 / 图表数据；引用白名单从机制上杜绝
编造参考文献；导出稿声明 AI 参与范围，最终文本责任在作者。

## 检索与对话

**RRF 混合检索（默认）**：线上检索 = 向量排序 + FTS 词面排序的 RRF 融合，两路都命中的文献排最前；
向量未配置或调用失败时自动退化纯 FTS。评测脚本支持口径消融：

```bash
python cli.py eval --retrieval auto   # 线上同款（向量可用 → hybrid）
python cli.py eval --retrieval fts    # 强制纯 FTS
# 两者数字差即混合检索的贡献（见下方评测表）
```

**迭代检索（伪相关反馈补位）**：先按原查询召回，再从头部文献的关键词/标题派生补充查询，
**原始排序原封不动占前排、派生结果只填尾巴**。跨语言场景（中文问、英文论文）尤其受用。
全程 0 次 LLM 调用，因此可以直接在评测集上做 A/B——这个功能的取舍完全由数字决定：

（2026-09-03 在 503 篇库上重测，纯 FTS 底座、关键词串形态 32 条）

| 检索口径 | Recall@5 | @10 | @15 | @20 |
|---|---|---|---|---|
| 单轮 | 0.7552 | 0.7969 | 0.8125 | 0.8125 |
| 派生查询 + RRF 融合 | **0.4740** | 0.6250 | 0.7969 | 0.8438 |
| 派生查询「只补位」 | 0.7552 | **0.8281** | **0.8281** | **0.8438** |

第一版做的是「派生查询用 RRF 融进主排序」，**实测小 k 上大幅变差**：派生词太宽
（"large language model" 这种），典型的 query drift——PRF 独有候选里 gold 只占 0.9%，
把它们提上来就是把 gold 挤下去。所以线上口径改成「只补位」：小 k 与单轮**逐条相同**
（结构上不可能变差）。

**收益要如实说小**：k≥15 时 0.8125 → 0.8281，**+0.0156 就是 32 题里的半道题**，
按本项目自己的显著性门槛远不足以判定。此前版本报的是「0.766 → 0.828（+0.0625）」，
那是在 455 篇库、且查询预处理修复之前测的——单轮基线随之抬高后，PRF 的增量被吃掉了大半。
诚实的说法是「**方向为正、量级在噪声内，作为宽上下文档位的补位策略保留**」，
而不是「PRF 把 Recall 提了 6 个点」。

那个负结果保留为 `deep-rrf` 口径，可随时复现（0 token、0 外网）：

```bash
$env:EMBED_MODEL=''
python cli.py eval --k 15 --retrieval fts       # 单轮基线
python cli.py eval --k 15 --retrieval deep-rrf  # 负结果，留作对照
python cli.py eval --k 15 --retrieval deep      # 只补位
```

（踩过的坑：`expand_queries` 里原本直接遍历 `set`，字符串哈希随机化让同一份评测集
连跑两次得出 0.8281 和 0.7969 两个数——排序键必须是全序，否则「可复现」是句空话。）

**查询预处理：用户真会打的输入不能把检索打废**（2026-09-03）。原来的切词只剥双引号，
其余标点原样进词面；而 `search_fts` 又把每个词面包成 FTS5 短语，在 trigram 分词下
等价于**字面子串匹配**。三条真实失败形态（真库上复现）：

| 查询 | 修复前 top1 | 修复后 top1 |
|---|---|---|
| `What is XL-MIMO?` | Zero-shot Reading Comprehension（靠 `What` 命中） | Revisiting Near-Far Field Boundary in XL-MIMO |
| `(channel estimation)` | **返回空**（FTS 0 行，LIKE 也 0 行） | Distributed Massive MIMO Channel Estimation |
| `有哪些关于大模型智能体评测的论文？` | 检索增强生成的评测方法综述（自建笔记） | MLLM-Tool / CartoAgent / LLM Agent Frameworks |

三处修法：① 先做标点归一再切词（保留 `+ # . - /`，C++/C#/GPT-4/XL-MIMO 靠它们），
词面首尾的 `.-/` 剥掉；② **CJK 片段切三元窗口，不是二元**；③ LIKE 兜底按命中词面数排序
（标题 2 分 / 摘要 1 分），不再 `ORDER BY id DESC`——原来它把「最新入库」当「最相关」，
再以权重 1.0 灌进 RRF 的最高一路。

**② 那条值得单独说：一个被字节/字符搞错的假设。** 直觉上「2 个汉字 = 6 字节，够构成 trigram」，
所以放行中文二元词面就行。**实测推翻**——FTS5 的 trigram 分词器按**字符**计：

| 词面 | 字符数 | 本库 papers_fts 命中 |
|---|---|---|
| `智能` / `模型` / `信道` / `ai` | 2 | **全部 0** |
| `智能体` / `大模型` / `信道估` / `LLM` | 3 | 7 / 1 / 11 / 17 |

二元词面**永远**进不了 trigram 索引，连 ASCII 也一样。所以正确修法是切三元
（`db.MIN_FTS_CHARS = 3`，常量旁写了实测依据，`tests/test_query_preprocessing.py`
有一条用例专门钉住它——哪天这个前提不成立会立刻变红）。

效果（两个评测集配对检验，0 token、0 外网，见 `papernest/stats.py`）：

| 口径 | 修复前 | 修复后 | 变化的题 | p |
|---|---|---|---|---|
| 自然语言提问 Recall@5 | 0.5312 | **0.7552** | 10/32 | **0.0071 显著** |
| 自然语言提问 Recall@10 | 0.6302 | 0.8125 | 8/32 | 0.0244 显著 |
| 关键词串 Recall@5 | 0.6771 | 0.7552 | 3/32 | 0.256 **不显著** |
| QASPER paper_hit@5 | 0.6364 | 0.7045 | 10/88 | 0.106 **不显著** |

**如实说：只有「自然语言提问」这一档是显著的。** 另两个方向为正但样本量不足以判定——
关键词串那一档之所以测不出来，正是因为评测集原有的 32 条查询里**一个问号、一个疑问词都没有**。
为此新增了 32 条 `form: "natural"` 的题：**同一批 gold、只改写 query 形态**
（带 `derived_from` 可追溯，不自造 gold——那会掉进「用系统自己的召回当标准答案」的自证陷阱）。
`cli.py eval` 现在两类**分开报**，并给出 `form_gap`。

**精排：一条推断，和一个修完 bug 才浮出来的结论**。`rerank.py` 四个可插拔后端
（api / llm / bm25 / off）。BM25 精排在 QASPER（122 题）上把 hit@1 从 0.4672 打到
**0.3197**、且池越大越差——由此推断「精排只在打分器比召回器信息量更大时才成立」
（BM25 只有词面，而 RRF 里本来就含词面排序，用它重排等于净丢向量那一路）。

接上真 Cross-Encoder（DashScope `qwen3.7-text-rerank`）验证时，测了两次，
中间修掉了「chunk 向量没接进检索」那个 bug：

| 口径 | 修 bug 前 Recall@5 | 修 bug 后 | 有变化的题 | p |
|---|---|---|---|---|
| 不精排 | 0.9427 | **0.9740** | — | — |
| Cross-Encoder 精排 | 0.9740 | 0.9740 | **0/32** | 1.0000 |
| BM25 精排 | 0.8750 | 0.8750 | 4/32 | 0.1285 |

**修 bug 前 Cross-Encoder 看着 +0.031，修完之后一道题都改不动。**
原来那份收益不是「精排的价值」，而是它替检索补上了本该召回、却因为
chunk 向量没接线而漏掉的两篇。根因修掉，收益就没有了。

所以那条推断要再加一个前提：**而且召回本身得留有可改进的余地**。
召回接近饱和（Recall@5=0.974）时，再强的打分器也只是重排已经对的结果。
Cross-Encoder 每次查询多一次调用（3398→6336ms，+87%）——**买不到召回、只买延迟**，
所以默认 off。

（DashScope 的 rerank **不在 OpenAI 兼容路径下**，走原生地址；换 Cohere / Jina /
SiliconFlow 改 `PAPERNEST_RERANK_URL` 即可，两种返回格式都认。）

**「建了但没接上」——本项目第三次踩同一类坑**。784 个章节块花 221 秒全部向量化之后，
消融显示贡献**恰好 +0.0000**（自建集与 QASPER 三个 k 全部逐位相同）。
精确到这个程度就不正常——一查发现 `search_hybrid` 的向量路只读 `kind='paper'`，
chunks 参与检索的唯一途径是 `chunks_fts`（**关键词索引，不是向量**）。
**784 条向量一条都没被读过：不是没效果，是没接上。**

前两次是同一个形状：`deep_answer` 实现了但没有任何入口、`section_chunks` 写好了
但主入库路径仍按页切。教训是**「跑通了」和「接上了」是两件事**——
交付前要问的不是「这个功能能不能跑」，而是「线上那条路径真的会走到它吗」。
现在 `_paper_matrix` 装 `kind IN ('paper','chunk')`，`search_papers` 按 paper_id
取最高分去重（否则 top-5 会被章节多的一两篇论文占满，「检索到 5 篇」其实只有 1 篇）。

**chunk 向量的实际收益**（库副本上做消融、真库只读；符号翻转检验 20000 次）：

| 指标 | 无 → 有 | 差值 | 有变化的题 | p | 结论 |
|---|---|---|---|---|---|
| 自建@5 | 0.9427 → 0.9740 | +0.0312 | 1/32 | 1.0000 | 噪声内 |
| 自建@10 | 0.9844 → 0.9844 | +0.0000 | 0/32 | 1.0000 | 噪声内 |
| **QASPER@1** | 0.4333 → 0.5667 | **+0.1333** | 8/60 | **0.0073** | **显著** |
| QASPER@5 | 0.6167 → 0.7167 | +0.1000 | 8/60 | 0.0711 | 边缘 |

**收益全部落在 QASPER 上，而且只有 QASPER@1 过了显著性。** 这是可解释的而非偶然：
QASPER 的题问的是**论文正文**，chunk 向量正是为此而建；自建集的 gold 是**论文级**，
标题摘要向量已经够用，所以那 +0.0312 只是一道题、p=1.0。

这也是**「同一个改动在两个评测集上表现不同」的第三次**——但这次不是矛盾而是分工：
一个测「找得到哪篇论文」，一个测「找得到论文里的哪段」。用错评测集，
就会把一个 +0.13 的改进误判成「基本没用」。

（插一段自我纠错：第一次跑消融时自建@10 显示 −0.0312，我据此写过「chunk 向量会挤占 @10」。
重测是 +0.0000、0/32 题变化——**那个下降没有复现，是我拿单次结果下了结论**。
这已经是本轮第三次因为「只跑一次就下判断」而要回头改口径。）

**章节块向量化**：库里 784 个章节全文块此前**一条向量都没有**——语义检索只覆盖标题与摘要，
正文只能靠关键词字面命中（中文问、英文正文时词面对不上，而向量正是解决这个的）。
`cli.py vec embed-chunks` 补齐，`--dry-run` 先试算成本。

```bash
python cli.py vec status                  # 后端 / 真相条数 / 索引条数 / 块覆盖率
python cli.py vec embed-chunks --dry-run  # 只试算 token，0 调用
python cli.py vec embed-chunks            # 真嵌入
python cli.py vec rebuild                 # 从 SQLite 重建后端索引
```

**向量后端可切换（milvus / numpy；chroma 代码保留但已停用）**：`PAPERNEST_VECTOR_BACKEND=milvus|numpy`。
SQLite 的 `vectors` 表**始终是真相来源**，Chroma 只是派生索引——两份数据不一致时以 SQLite 为准，
`cli.py vec rebuild` 随时能从 SQLite 全量重建。这样「两份数据要同步」这个向量库方案里
最容易出 bug 的地方，被降级成「派生索引可重建」。

1024 维实测 p50，两个后端同规模对比：

| 向量条数 | Chroma（HNSW） | numpy 缓存矩阵 |
|---|---|---|
| 1,000 | 1.99 ms | **0.14 ms** |
| 5,000 | 2.66 ms | **0.91 ms** |
| 20,000 | **2.91 ms** | 3.21 ms |

Chroma 用 HNSW，查询时间几乎**不随规模增长**；numpy 是暴力精确扫描，随规模线性增长。
**拐点约两万条**——库内目前约 1200 条向量，所以没有外部向量服务时 numpy 即可；默认后端仍是 Milvus，理由是部署形态而非速度（见下文）。
HNSW 是近似索引，用库内 451 条**真实**向量实测 recall@1=0.990、@5=1.000、@10=1.000，
在这个规模上与精确检索无差。

> 踩过的坑：第一版用随机高斯向量测 HNSW，得出 recall@5 只有 0.56 的假结论。
> 原因是 1024 维随机向量彼此近乎等距（余弦均值 0.000、标准差 0.031），根本没有近邻结构
> 可供索引；而真实向量的余弦均值 0.322、标准差 0.114，是有结构的。
> **测 ANN 必须用真实分布的向量**——这和本项目反复踩到的「合成夹具掩盖真实数据形态」是同一个坑。

**为什么默认是 Milvus 而不是更快的 numpy**：真实库 451 条向量实测 numpy 8.5ms/次（精确 kNN）、
Milvus 20.1ms/次（HNSW）——**这个规模上 numpy 又快又准**。选 Milvus 是部署形态上的决定：
它的延迟基本不随规模变，而 numpy 的常驻内存与线性扫描到十万条以上会顶不住。
一致性实测 **top-5 30/30 与精确检索完全相同**，所以换过来没有召回损失。
连不上时**不静默退回**（设 `PAPERNEST_VECTOR_FALLBACK=1` 才降级，且在 `store.degraded`
留痕并打到 stderr）——「以为在用向量库、其实一直是 numpy」比直接报错危险得多。

**一个实测出来的性能坑**：Milvus 默认 `consistency_level=Strong` 让每次查询显著变慢，换 `Bounded` 快一个量级。
⚠️ **具体数字暂不引用**：本文档旧版记 399.9ms → 16.1ms（25 倍），而 `milvusstore.py:114` 的注释记 380.3ms → 8.0ms（47 倍），两处互相矛盾且仓库里没有 bench 脚本可以裁决。Strong 换来的只是省掉 300ms 的写后可见延迟——
对「写一次、读上千次」的文献检索是错误的取舍。现在查询走 Bounded，
`rebuild()` 末尾 flush 一次把可见性窗口关掉，两头都要：**328ms → 20ms**，
且「重建完立刻能搜到」的契约实测仍然成立。

**Docker 全套部署（已实测跑通）**：

```bash
docker compose up -d                    # 默认连 Milvus（见下）；想单容器跑：.env 里设 PAPERNEST_VECTOR_BACKEND=numpy
# 想用 Milvus（连机器上已有的那个，不另起）：
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d
# 需要自带一套 Milvus（多 3 个容器）：再加 --profile own-milvus
```

镜像 **613MB**（默认不装 chromadb，比带着它的 1.02GB 省 40%；`--build-arg WITH_CHROMA=1` 可装回去）。
实测验证过的三条：

- **数据持久化**：容器写 `/app/data` 直接落到宿主机 `./data`，重启后 503 篇论文 / 451 条向量完整；
- **密钥不进镜像**：`.dockerignore` 排除 `.env`，实测镜像内 `/app/.env` 不存在，key 只在运行时由 `env_file` 注入；
- **容器 → Milvus 打通**：`host.docker.internal:19530` 连宿主机上已有的 Milvus，
  容器内 `count()=451`、检索 score=1.000（同向向量），HTTP 接口同时正常。

**向量库：Milvus**（`PAPERNEST_VECTOR_BACKEND=milvus|numpy`，默认 milvus）。Chroma 后端代码与测试保留，但**不再随镜像分发**——三种形态都实现过一遍，是后端抽象层「抽得动」的证据，但交付物只带一个。

| 后端 | 容器 | 额外内存 | 真实库延迟 | 检索 | 适用 |
|---|---|---|---|---|---|
| numpy（一行可切） | 0 | 0（矩阵常驻 ~2MB） | **3.3 ms/次** | 精确 kNN，无召回损失 | 十万条以内 |
| chroma（已停用） | 0（嵌入式） | 进程内 | 6.8 ms/次 | HNSW，top-5 一致 39/40 | 代码保留，不随镜像分发 |
| **milvus**（默认） | 1~3 | 2~4 GB | **20.1 ms/次** | HNSW，与精确检索 **top-5 一致 30/30** | 千万级、多应用共享 |

三者在真实库上结果一致：numpy vs Chroma 40 次查询 top-5 顺序完全一致 **39/40**、
集合重合率 **0.995**（唯一差异是 HNSW 漏了一个候选，近似索引的固有代价）。

**Milvus 的部署代价明显不同**，要三个容器（etcd 存元数据、MinIO 存对象、milvus 本体）：

```bash
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d
```

> **Windows 上没有轻量选项**：Milvus Lite（嵌入式）全部历史版本**零个 Windows wheel**，
> `pip install pymilvus[milvus_lite]` 在 Windows 上会静默跳过，实测报
> `ConnectionConfigException: milvus-lite is required for local database connections`。
> 要么跑上面这套服务端，要么在 WSL/Linux 里用 Milvus Lite。
>
> 对**单用户个人库**（本项目约 1200 条向量）来说 Milvus 在速度上是过度配置——
> numpy 后端比它快、且是精确检索。仍把 Milvus 设为默认，是部署形态上的决定
> （延迟不随规模变、可与机器上其他应用共享同一实例），`PAPERNEST_VECTOR_BACKEND=numpy` 一行即可切回。

**Docker**：Chroma 走**嵌入式**（`PersistentClient`），不需要单独的容器，索引落在
`data/chroma/`，沿用既有的 `./data` 卷，**docker-compose 一个字都不用改**。
镜像会因此胖约 163MB（chromadb 依赖树含用不到的 onnxruntime / kubernetes），
不需要就 `docker compose build --build-arg WITH_CHROMA=0` 装精简版。

**为什么曾经不上向量数据库（量出来的决定）**：向量存 SQLite BLOB + NumPy 余弦。
1024 维实测 p50——瓶颈从来不是余弦计算（纯矩阵乘 0.04ms），而是**每次查询重读 BLOB**：

| 向量条数 | 原实现 | 缓存归一化矩阵 | 常驻内存 |
|---|---|---|---|
| 500 | 9.6 ms | **0.19 ms** | 2 MB |
| 10,000 | 203.4 ms | **1.80 ms** | 39 MB |
| 50,000 | 1028.6 ms | **8.87 ms** | 195 MB |
| 200,000 | 4743.2 ms | **41.74 ms** | 781 MB |

只做向量化仅快 1.3 倍，缓存矩阵快 50~113 倍。个人库全量嵌入约 5~6k 条向量、缓存后 1ms 级，
**十万条以内不需要 ANN 索引**；真到十万条以上再换 sqlite-vec（同一个单文件、不引服务进程）。
缓存按「(条数, max(id)) 指纹」自动失效，换嵌入模型或维度不符时**报错而不是静默截断**。

顺带量出一个全项目性的开销：`PRAGMA journal_mode=WAL` 是**写操作**，每次连接要 ~6ms，
而 journal_mode 是数据库文件的持久属性、设一次就够——改成按库文件只设一次后
`db.conn()` 从 **7.06ms 降到 0.91ms**，影响的是每一次数据库访问（一次混合检索开 2~3 个连接）。

**换提供方零成本迁移向量**：`python cli.py embed` 在模型名变更时先用同一文本重嵌入一篇、
与库内旧向量算余弦——≥0.98 判定同一嵌入空间，原地改名完成迁移（0 token）；
否则才全量重建。

**流式对话 + 会话记忆**：`POST /api/chat/stream`——先推 `sources`（检索毫秒级完成），再逐段推正文（`delta`），
异常以 `error` 事件透传，前端边生成边渲染。带 `session_id` 时历史以服务端为准（刷新不丢），
会话内引用编号稳定；断流时**半截回答也会落库**，否则下一轮的历史里会凭空少一轮。

```bash
curl -X POST /api/chat/stream -d '{"messages":[...],"session_id":""}'  # 空串=开新会话
curl /api/chat/sessions            # 历史会话列表（可回放）
```

**迭代问答（自反馈闭环）**：`python cli.py ask "问题" --deep` 或 `POST /api/answer/deep`——
作答 → **机械**找无证据论断（复用评测的论断切分口径，不另发明一套）→ 模型对着缺口提补充查询 →
补检索 → 重答；收敛条件（论断清零 / 无新查询 / 无新文献 / 轮数上限）与每轮的查询、新增文献数
全部记进 trace，可审计。宽上下文的调用方（综述 top_k=10、写作选题 limit=20）也已切到
迭代检索口径——k≥10 实测有净增益、小 k 与单轮逐条相同、0 token。

**RCS 增强问答（重排 + 逐篇定向摘要）**：`python cli.py ask "问题" --rcs` 或
`POST /api/answer/rcs`。朴素 RAG 默认「检索排序就是有用顺序、原文直接塞进去」，
RCS 只改这一段：宽检索候选池（top_k×3，上限 15）→ **LLM 按「能否回答这个问题」重排** →
**对每篇针对问题压成 2-3 句要点 + 原文依据句** → 依据句走机械回取校验 → 作答。

比 PaperQA2 多的一层是**依据句校验**：它只压缩，不验证压缩出来的东西是否真在原文里。
这里的 quote 逐页比对（不拼成一大串，否则前半在第 2 页、后半在第 7 页也能「验过」），
验不过如实标「未通过原文回取校验」并在 prompt 里禁止当事实引用。

代价是每次问答多 `1 + N` 次调用（轻档模型），所以**默认关闭**，默认问答路径不受影响。
重排失败 / 返回垃圾 JSON / 全部被判不相关，一律降级回朴素上下文——增强件不该成为新的故障点。

实测（自建集 QA 子集，`cli.py eval --qa --qa-mode plain|rcs`）：

| 指标 | 朴素 RAG | RCS 第 1 次 | RCS 第 2 次 |
|---|---|---|---|
| gold 引用覆盖 | 0.818 | **1.000** | **1.000** |
| 无证据率（越低越好） | 0.364 | 0.382 | 0.349 |
| 依据句机械回取通过率 | — | 0.912 | 0.929 |

**gold 覆盖是稳定正收益**（重排把真正相关的排上来了）；**无证据率两次跨在基线两边，
说明在这个样本量上它的影响淹没在噪声里**——诚实的结论是「该指标无可靠改善」，
而不是挑好看的那次报。这两次差异也说明：`temperature=0` 不保证跨次一致，
所以 RCS 的数字不像检索指标那样可精确复现，README 里同时列出两次而不是取平均。

另外摘要过滤很激进：11 题共 55 篇入选、被判不相关剔除 39 篇（71%），
平均每题只剩 1.5 篇进上下文。gold 覆盖仍是 1.0 说明留下的是对的，但这个比例值得盯——
**「摘要把有用的论文判成不相关」是 RCS 最危险的失效，且用户看不见**，
所以 trace 里把「入选后被摘要扔掉」与「重排未选中」分开报（混成一个数会严重误导）。

**Obsidian 文献笔记导出**：`python cli.py export-obsidian <vault目录>`——每篇一个 Markdown，
citekey 命名（与 BibTeX 导出同键，Obsidian ↔ LaTeX 对得上号）、YAML frontmatter 供 Dataview
查询、**库内互引自动写成 `[[citekey]]` 双链**（笔记图谱直接长出文献图谱的形状）、
关键结论保留 ✓/? 核验标记与页码；卡片缺什么就不写什么小节，不编造。

**本地模型（Ollama / llama.cpp）**：两者都提供 OpenAI 兼容端点，PaperNest 零改动接入
（`.env.example` 有配置样例）。向量按 model 列隔离不会混维度；换 embedding 模型跑一次
`cli.py embed` 做嵌入空间一致性检测。小模型 JSON 遵从差时走各模块的 markdown 兜底并如实标 degraded。

**存储单元与检索单元分离**：`pages` 按物理页存（精读卡片要报准页码、机械回取按页验），
`chunks` 按**章节**切（**检索**要的是语义完整的块）。

**注意范围：chunks 用于检索排序，不用于上下文组装**——后者实测是负结果，见下节。

曾用 25 篇真实 arXiv PDF 和 69 道 QASPER 带 gold 证据的题做过配对 A/B，结论是
「等上下文预算下按章节切的证据召回是按页切的 2.2~3.2 倍（8000 字预算 0.087→0.275，p=0.0013）」。
⚠️ **这个数字目前不可复现，引用时必须带上以下三条限定**：

1. **脚本与原始数据没有收进仓库**（当时在临时目录里跑的），仓库里没有任何入口能重跑它。
2. **2026-09-03 真去量了一次，结论与它相反**——见下面「上下文组装：一条负结果」。
3. **当时的评测口径对一个已知缺陷不敏感**：`qasper` 的 gold 匹配先删掉全部空白再做子串包含，
   而 `structure.py` 用 `"".join` 拼 span 会吃掉词间空格（真库 435/784 个 chunk 含粘连词）。
   于是 `Thissectionpresents` 与 `This section presents` 在评测里得分相同，
   而线上的 `chunks_fts`（trigram）与 embedding 都对空格敏感。

诚实的说法是「**按章节切在检索排序上有效（块级权重表可复现）；
而「按章节块组装上下文」这一步实测为负收益，已如实记录并保持默认关闭**」。

### 上下文组装：一条负结果（2026-09-03）

审计指出「检索单元与上下文单元不一致」：检索按 chunk 排序，组装上下文却回到 `pages`
按词频重猜一遍。于是把块级命中完整接通（`RetrievalResult.chunk_hits`——词面命中来自
`chunks_fts`，语义命中来自向量矩阵里赢的那一行），然后**真去量了一次**。

三臂配对实验（QASPER 88 道带 gold 证据的题、等上下文预算、纯 FTS 底座、0 token）：

| 口径 | 证据召回（删空白） | 证据召回（保留空白） | 变化的题 | p |
|---|---|---|---|---|
| A 改造前：绝对词频选页 + 页首硬截 | 0.2273 | 0.2273 | - | - |
| B 只改选块：密度归一 + 围绕命中取窗口 | **0.2614** | **0.2614** | 7/88 | 0.452 |
| C 再换成按章节块组装 | 0.2045 | **0.1136** | 17/88 | 0.331 |

```bash
$env:EMBED_MODEL=''; python cli.py qasper ctxab --n-papers 30
```

**结论与预期相反，如实记录**：

- **B 保留**（+0.0341，方向为正、p=0.452 不显著）。两处都是有原理的修正：
  绝对词频让最长的页恒赢（真库 83% 的情况选中的就是该篇最长页，往往是相关工作/参考文献），
  从页首硬截会把页中部的证据切掉（中位数只保留 29%）。
- **C 否决**（-0.0568，且**保留空白口径下腰斩** 0.2614→0.1136）。
  根因不是「章节 vs 页」这个单元本身，而是 **`chunks` 表的文本是 `pages` 的有损再推导**：
  同一篇论文 pages=24 而 chunks=20、pages=19 而 chunks=31，文本已不是原文逐字。
  这会直接打断本项目的机械回取校验（证据句必须能在原文里逐字找到）。
- 「保留空白」是**新加的口径**：旧的 `_norm` 把全部空白删掉再比对，
  对「词间空格被吃掉」完全不敏感。两个口径一起报，才能看见文本保真度的损失。

所以默认上下文单元仍是 `page`（`PAPERNEST_CONTEXT_UNIT=chunk` 可切换）。
**接线已完成并有测试，负结果也钉进了用例**——等 chunk 文本的逐字保真修好之后，
重跑上面那条命令再决定是否切换，而不是凭直觉打开开关。

```bash
python cli.py rechunk        # 存量库重切（有 PDF 走章节切，没有的退化按页，如实计数）
```

**表格感知切分**：表格单独成块（`kind='table'`），**整行切分绝不从行中间截断、表头跨块重复**——
论文结果表被页边界或字数上限砍断后下半张没有表头，检索到也读不懂。识别不到表头时如实标注
「未识别到表头」而不是拿第一行假装。

**多格式入库**：Word / PPT / HTML / Markdown / 纯文本与 PDF 走同一条链路
（PPT 一张幻灯片算一页、演讲者备注一并收；HTML 丢弃 script/style/nav；文本按
utf-8/utf-8-sig/gbk/latin-1 顺序探测编码）。批量导入报**三态计数**（ok / skipped / failed）
并按格式分组——200 份文件的导入要能看清哪些没进来、为什么。

```bash
python cli.py import-doc ./我的资料 --no-cards    # 目录递归；PDF 用 import-pdf
python cli.py sectiontree build && python cli.py sectiontree search "查询"
```

**检索精排（默认关闭，这是一个负结果）**：`rerank.py` 提供 api（Cohere 风格 Cross-Encoder）/
llm / bm25 / off 四个可插拔后端。实测 **BM25 精排把 QASPER@1 从 0.467 打到 0.320，且池越大越差**——
原因不是实现问题而是设计问题：RRF 里已经含了词面排序，再用纯词面打分器重排等于把向量那一路
的信息扔掉，信息量严格变少。**「精排」只有在打分器比召回器掌握更多信息时才成立**，
这正是业界用 Cross-Encoder 而非词面打分器的原因。故默认 `off`，保留两个有语义的后端待接。

**容错**：LLM 调用 4 次指数退避 + 抖动重试（实测中转站 TLS 随机中断场景下，
单次请求 ~60% 成功率时整体成功率 1-0.4⁴ ≈ 97%）；400 立即失败不重试；
流式输出首 chunk 后断流不自动重试（避免重复文本），以 error 事件如实上报。

## 评测：自建集 + QASPER 公开基准

**自建评测集**（`eval_set.json`，82 条，三指标全部机械可算）：

| 指标 | 含义 | 值（FTS 口径） |
|---|---|---|
| Recall@5（关键词串形态，32 条） | 检索 top-5 命中 gold 的比例（宏平均） | **0.7552** |
| Recall@5（自然语言提问形态，32 条） | 同一批 gold，query 改写成用户真会打的提问 | **0.7552** |
| 引用证据核验率 | 推荐候选中证据句通过机械回取校验的比例 | **0.85** |
| 无证据率 | 回答论断句中无 [n] 引用支撑的比例（越低越好） | **0.30** |

> 上表 Recall 是 `EMBED_MODEL=` 关掉向量后的**纯 FTS 消融口径**（0 token、0 外网、精确复现）。
> 两种查询形态**分开报**：合成一个平均值会让「口径变了」被一个没变的指标名盖住，
> 而两者的差距（`form_gap`）本身就是要盯的指标——它衡量「用户按自然语言提问要付多少代价」。
> 2026-09-03 修查询预处理之前，自然语言形态是 0.5312，比关键词串低 0.224（p=0.0071）。
>
> **线上口径**（向量 + FTS）Recall@5 = **0.9427**、@10 = 0.9740（2026-09-03 补齐向量后重测）。
> **这个数字有一个必须一起说的前提**：
>
> 1. **后端是 SQLite BLOB + NumPy 精确 kNN，不是 Milvus。** 检索热路径
>    （`rag.prepare → embeddings.search_hybrid → search_papers → _paper_matrix`）
>    全程不 import `vectorstore` / `milvusstore`——`get_store()` 的调用方只有 `cli.py vec` 子命令。
>    把 Milvus 停掉，这个数字一位都不会变。
> 逐位可复现的只有纯 FTS 口径。线上口径依赖外部接口，瞬态失败会被显式标出并拉低当次数字。

#### 补齐向量前后的配对对比（2026-09-03）

此前向量覆盖只有 104/503，而自建集 42 篇 gold **恰好 100% 有向量**、非 gold 只有约 3.9%——
所以当时的线上口径是有偏的。跑 `cli.py embed` 补到 503/503 后，用**同一份代码、同一批题**
重测（对照臂用补齐前的库备份，唯一变量是覆盖率）：

| 指标 | 覆盖 104/503 | 覆盖 503/503 | 变化的题 | p |
|---|---|---|---|---|
| 自建集 keyword Recall@5 | 0.9115 | **0.9427** | 1/32 | 1.0 |
| 自建集 keyword Recall@10 | 0.9844 | 0.9740 | 1/32 | 1.0 |
| 自建集 natural Recall@5 | 0.9115 | **0.9427** | 1/32 | 1.0 |
| QASPER paper_hit@5 | 0.7614 | **0.7727** | 1/88 | 1.0 |

**两条与预期相反、必须如实说的结论**：

1. **补齐后 Recall 没有掉**，反而各涨了一题。此前的判断是「gold 覆盖 100%、非 gold 只有 3.9%，
   补齐后按 RRF 单调性只可能持平或下降」——**这个预测是错的**。
2. 更值得警惕的是：**把向量覆盖从 20.7% 提到 100%（多了 399 篇可被语义检索命中的干扰项），
   四个指标各只有 1 道题发生变化。** 这说明**现有评测集分辨不出向量覆盖率这么大的变化**——
   题目对当前系统太容易。这不是「覆盖率不重要」的证据，而是评测集灵敏度不足的证据。
   下一步该做的是补难题（跨篇对比、需要正文细节、gold 不在标题里的），而不是继续报这些数字。

顺带：**自然语言提问形态与关键词串形态现在逐位相同**（都是 0.9427）——
查询形态惩罚在混合检索口径下已经归零（纯 FTS 口径下修复前是 −0.224）。

**QASPER 公开基准**（Das et al., AI2 2021，每题带 gold 证据片段）：
`qasper download` → `qasper import`（全文按节入库，零 LLM 零 PDF）→ `qasper eval`，
两层指标——`paper_hit@k`（gold 论文进全库检索 top-k，其余 QASPER 篇互为干扰项）与
`evidence_recall@sec_k`（gold 论文内 top-k 节命中证据片段）。

```bash
python cli.py qasper download && python cli.py qasper import --n-papers 40
python cli.py qasper eval --n-papers 40
```

> 这条链路一度是坏的：三个下载源（GitHub raw / ai2-public-datasets / hf-mirror）全部 404，
> 数据集换了发布位置；即便手工塞进数据也解析不出东西——官方 v0.3 的 `full_text` 是
> list（不是 `{section_names, sections}`），证据埋在 `qas[i].answers[j].answer` 下
> （不是 `qa["evidence"]`）。而四个 QASPER 用例全绿，因为**夹具用的是那个已经不存在的格式**。
> 现在改用 AI2 官方 S3 的 tgz，解析器两种格式都认，夹具按真实 v0.3 结构重写并补了格式回归用例。

## L2 全文进全库检索

`papers_fts` 只索引 title / abstract / keywords，于是「每篇论文只精读一次」拿到的正文
在全库检索里等于不存在——L2 最大的浪费。现在页级全文单独进 `pages_fts`
（不是塞进 papers_fts 的第四列：命中要能定位到页，才对得上「证据带页码」的口径），
以较低权重参与 RRF 融合。

权重是**测出来的**。2026-09-03 在 503 篇库上重测（QASPER 88 题 · 自建 32 条，纯 FTS 底座）：

| 块级权重 | QASPER hit@1 | hit@5 | hit@10 | 自建 Recall@5 | Recall@15 | 自然语言 Recall@5 |
|---|---|---|---|---|---|---|
| 关闭 | 0.4886 | 0.6705 | 0.7386 | 0.7552 | 0.8125 | 0.7552 |
| **0.2（代码默认）** | **0.5455** | 0.7045 | **0.7500** | 0.7552 | 0.8125 | 0.7552 |
| 0.5 | 0.5455 | 0.7045 | 0.7273 | 0.7552 | 0.8125 | 0.7552 |
| 1.0 | 0.5341 | 0.7273 | 0.7614 | 0.7552 | 0.8125 | 0.7552 |
| 1.5 | 0.5227 | **0.7386** | **0.7727** | 0.7552 | 0.8125 | 0.7552 |

**两条必须说清楚的更正**（此前版本的表已过期，不要引用）：

1. **默认值是 0.2，不是 0.5**（`embeddings.CHUNK_WEIGHT`，环境变量
   `PAPERNEST_CHUNK_WEIGHT`，旧名 `PAPERNEST_PAGE_WEIGHT` 仍兼容）。
   旧表测于**按页切**时期；换成**按章节切**之后最优点左移——块变小后单条命中更该「小步加分」。
2. **自建集现在对这个参数完全不敏感**：五档权重下 Recall@5/@15 逐位相同。
   旧表里「1.5 及以上自建集崩到 0.3802」在当前库上**复现不出来**。
   原因是库里只有 45/503 篇有 chunks，自建集的 gold 基本不在其中。
   **所以「两个评测集一起看」这条纪律在这个参数上目前只剩 QASPER 一个集在起作用**——
   这是评测覆盖的缺口，不是「参数已经充分验证」。

按当前数据选 0.2 的理由：hit@1 最高（0.5455，比关闭高 0.057），hit@10 次高。
1.5 在 hit@5/hit@10 上更好但 hit@1 更差，且历史上正是它在按页切时期打崩过自建集
（权重 ≥1.5 时任何一条块级命中的 RRF 分都高于任何一条标题命中，1.5/61 > 1.0/61，
正文噪声直接接管排序）——在自建集重新具备区分力之前，不采纳更激进的权重。

```bash
python cli.py rechunk                        # 老库按章节重切（迁移会自动兜底一次）
$env:EMBED_MODEL=''                          # 纯 FTS 底座，0 token 0 外网
$env:PAPERNEST_CHUNK_SEARCH='0'; python cli.py qasper eval --n-papers 30   # 关掉做对照
$env:PAPERNEST_CHUNK_WEIGHT='1.0'; python cli.py qasper eval --n-papers 30 # 调权重复现上表
```

**规模实验**（`cli.py scale`）：多组关键词批量采集（`--no-cards` 纯元数据，0 LLM），
报告采集吞吐（篇/s）、norm_key 去重率、FTS 查询延迟 p50/p95。

```bash
python cli.py scale "large language model agents;retrieval augmented generation" --limit 100
```

2026-08-30 实测（4 组查询，S2/arXiv/OpenAlex 合并去重）：

- 库 60 → **455 篇**（新增 395，`norm_key` 重复 **0**；来源 s2 158 / arxiv 197 / openalex 100）
- **Recall@5 在 7.6 倍库容下保持 0.6771 不变**（干扰项增长未损伤 gold 召回）
  —— 注：0.6771 是**当时**的纯 FTS 基线；2026-09-03 修查询预处理后同口径为 0.7552。
  这条结论要看的是「库容涨 7.6 倍而召回不掉」，不是那个绝对值。
- FTS 查询延迟 455 篇规模：**p50 4.1ms / p95 5.3ms**
- 全程 **0 次 LLM 调用**（`--no-cards`：元数据入库不建卡）

## 新论文订阅 · 对比矩阵 · 版面结构

**新论文订阅**（`papernest/subscribe.py`）：从库内论文反推兴趣画像 → 拉 arXiv 新提交 →
**确定性打分（0 次模型调用）**→ 三道去重闸（已在库 / 最近 N 天推过 / 标过不感兴趣）。
每条都说明「为什么推给你」：命中画像里哪些词、在标题还是摘要、那个词有没有区分度。

打分口径踩了两个坑，都是拿真实数据打出来才发现的：

- **缺 IDF**。画像里权重最高的是 `learning`(0.93) / `deep`(0.98)——库内很多论文带它们。
  但它们同时出现在当天 arXiv 的**半数**论文里，等于没有区分度，于是推来的是
  「城市交通预测」「核岭回归」「农业 Web 系统」，而库真正的主题 beamforming / MIMO
  一篇没中。现在用**这批论文自己**算 IDF（`log((N+1)/(df+1))/log(N+1)`，不需要外部语料）：
  `learning` 被压到 0.215，`beamforming` 保留 0.744。
- **单词项命中不构成推荐理由**。只命中一个泛词的论文加 0.6 折并标「弱匹配」。
  两条一起上之后，一篇 6G NOMA 和一篇 MIMO Massive Random Access 进了前六，
  只靠一个 `deep` 命中的农业论文从第 1 掉到第 7。

**结构化对比矩阵**（`papernest/matrix.py`）：跨论文按自定义列抽成表格，
**每个格子带来源与机械回取校验结果**——模型给的依据句必须真的出现在它声称的那一页
（口径同 `fulltext._verify_claim`），页码对不上会按原文更正并写进备注。
抽不到的格子**留空不编造**，表尾给出覆盖率与核验率，一眼看出这张表有多少是真有依据的。
展示型导出（Markdown / LaTeX）截断长格子以便横向对比，CSV 保留全文（那是拿去分析的）。

**PDF 版面结构**（`papernest/structure.py`）：字号 / 加粗 / 编号 / 关键词四类判据识别章节，
按章节切 chunk（识别不出就如实降级成按页切），并抽参考文献条目。**零新依赖、不上 GROBID**
（那要跑一个 Java 服务，对个人自用工具太重），做不到的部分如实说明。
判据门槛也是实测定的：允许「单个信号」达标时，一篇 26 页真实综述被识别出 **125 个章节**，
二十多个是分类图里又粗又短的图元标签（Self-Learning / Centralized / …）——它们只命中
`bold` 一个信号。要求至少两个信号后是 **29 个**，层级与真实目录完全吻合。

**联动**：本地上传的 PDF 常常既没有 DOI 也没有 s2_id，联网那条路对它无从下手
（拿标题去瞎搜可能搜到别的论文），于是它永远进不了引文图。但它自己的参考文献段里
就写着几百个 DOI 与 arXiv 编号——那是「它引了谁」的一手证据。
实测一篇综述：**329 条参考文献 → 150 条带标识 → 150 条边**，全程离线 0 token。

```bash
python cli.py digest --profile              # 先看兴趣画像
python cli.py digest --days 3 --top-k 10    # 新论文摘报
python cli.py matrix 1,3,4 --no-llm         # 对比矩阵（0 token）
python cli.py structure 1                   # 章节树 + 参考文献
python cli.py graph local-refs              # 本地 PDF 参考文献 → 引文图
```

## 写作台与汇报 PPT

**文献汇报 PPT（零 LLM，离线可用）**：从库内已核验的结构化卡片直接装配 16:9 幻灯片——
单篇是论文汇报（背景→方法→关键结果带 ✓/？核验标记与页码→局限→课题启发），
多篇是文献汇报（总览→逐篇→参考）；模型只在配置 key 后额外生成一页课题总起，且失败不连累 PPT。

```bash
python cli.py ppt 15,1 --topic "大模型 Agent 的构建与评测"   # 命令行
# 网页：文献详情「生成汇报 PPT」/ 文献库勾选后「生成汇报 PPT」（POST /api/pptx 直接下载）
```

**写作台（网页「写作台」页，粘贴即用，走任务系统 + SSE 进度）**：
- 润色：保留原文事实与 [n] 引用编号，改后全文 + 逐条修改说明（类型/改前/改后/理由）
- 评审：审稿人视角输出总评、完成度评分（1-10）、优点、major/minor/suggestion 分级问题（定位到原文片段 + 可执行修改建议）、下一步清单；评审时勾选库内文献可核对文中引用是否站得住

```bash
curl -X POST /api/jobs -d '{"kind":"polish","params":{"text":"……","instruction":"更凝练"}}'
curl -X POST /api/jobs -d '{"kind":"review","params":{"text":"……","paper_ids":[15]}}'
```


## 三分钟跑起来

```bash
pip install -r requirements.txt
copy .env.example .env      # 需要真卡片就填 LLM_API_*；填 S2_API_KEY 更稳

python cli.py verify                        # 第 0 步复测
python cli.py ingest "你的课题关键词" --limit 20
python cli.py ingest "你的课题关键词" --limit 20   # 再跑一次：缓存命中 N、LLM 0 次
python cli.py import-pdf ~/papers/           # 手上已有的 PDF 也进库（按内容去重）
python cli.py import-bib zotero-export.bib   # 从 Zotero / EndNote 搬家
python cli.py graph fetch --limit 20 && python cli.py graph gaps   # 该读而没读的文献
python cli.py stats                          # 成本与缓存证据（tokens / latency / cost）
python cli.py eval                           # 评测三指标（离线部分不调模型）
python cli.py demo                           # 离线演示：无外网 / 无 key 可跑
python cli.py serve                          # http://127.0.0.1:8765
```

跑测试（1155 条，全部离线确定性，不联网不花钱）：

```bash
PYTHONPATH=. python -m unittest discover -s tests -t tests
```

Docker 一键启动（数据挂卷持久化，key 从宿主机 `.env` 注入、不进镜像）：

```bash
docker compose up --build    # 打开 http://localhost:8765
```

配了 LLM key 之后，把 mock 卡片重刷为真卡片：`python cli.py recard`。

## 评测三指标（`eval_set.json` · 82 条）

gold 标到 DOI/arXiv 归一化主键（`norm_key`），不随重新采集换 id；三个指标全部机械可算：

| 指标 | 含义 | 当前值（FTS 兜底模式） |
|---|---|---|
| Recall@5 | 检索 top-5 命中 gold 的比例（宏平均） | **0.68** |
| 引用证据核验率 | 推荐候选中证据句通过机械回取校验的比例 | **0.85** |
| 无证据率 | 回答论断句中无 [n] 引用支撑的比例（越低越好） | **0.30** |

```bash
python cli.py eval --k 5        # Recall@K + 引用指标（离线可跑）
python cli.py eval --qa         # 加测无证据率（调真模型，约 10 次调用）
```

说明：`EMBED_MODEL=` 置空即强制纯 FTS 口径（0 token、0 外网、精确复现）；线上口径（向量 + FTS）Recall@5 为 **0.9740**；
Recall 的 miss 集中在「中文查询 ↔ 英文论文」的跨语言条目——这正是向量检索的用武之地，
补上 `EMBED_MODEL` 后重跑 `eval` 即可对比两种检索链路的差距。

## 架构

```mermaid
flowchart LR
    U[用户 / 网页八栏<br/>对话 · Agent · 检索 · 引用 · 写作台<br/>写作流水线 · 课题工作区 · 引文网络] -->|goal| API[FastAPI<br/>/api/chat/stream · /api/agent/run<br/>/api/jobs + SSE · /api/pptx<br/>/api/write/* · /api/papers/upload<br/>/api/import/* · /api/graph/*]
    API --> P[Router / Planner<br/>确定性意图路由]
    P --> SM[有界状态机执行器<br/>步骤超时（协作式）· 瞬态故障重试<br/>每事件回调 → SSE / 轨迹落库]
    SM --> TR[Tool Registry]
    TR --> R1[retrieve_library<br/>RRF 混合：向量 + FTS]
    TR --> R2[answer_question<br/>RAG 强制 [n] 角标]
    TR --> R3[recommend_citations<br/>证据句 + 机械回取]
    TR --> R4[generate_survey<br/>逐句核验标 ?]
    TR --> R5[read_paper<br/>OA PDF → 按页 → L2 卡片]
    API --> IN[入库三条路<br/>检索采集 · 本地 PDF 上传<br/>BibTeX/RIS/Zotero 导入<br/>统一按 norm_key 去重]
    API --> G[引文网络 graph<br/>边用 norm_key 存<br/>阅读缺口 · 共被引/文献耦合]
    API --> CH[会话记忆 chat<br/>历史落库 · 编号稳定<br/>追问改写 0 token]
    API --> W[写作台 polish / review<br/>逐条修改说明 · 分级评审意见]
    API --> PPT[pptgen 汇报 PPT<br/>已核验卡片直装 · 零 LLM]
    API --> WP[二期 写作流水线 pipeline<br/>选题→大纲→逐节撰写⇄文献白名单→润色<br/>两人工检查点 · 节级内容寻址缓存<br/>机械终检：引用核验率 / 悬空编号]
    subgraph DB[SQLite · 单文件持久化 · user_version 迁移]
        T1[papers · pages · vectors]
        T2[query_cache · search_runs]
        T3[llm_calls tokens/latency/cost]
        T4[agent_runs 执行轨迹 · jobs 任务进度]
        T5[writing_runs · outlines · sections]
        T6[citation_edges · citation_nodes]
        T7[chat_sessions · chat_messages]
    end
    R1 & R2 & R3 & R4 & R5 --> DB
    IN --> T1
    G --> T6
    CH --> T7
    WP --> T5
    R2 -.-> LLM[OpenAI 兼容 LLM<br/>429/5xx 退避 · 4xx 不重试]
```

**证据链铁律**：回答/综述/引荐/生成稿的每条论断必须带 [n] 角标；
证据句必须在库内存储的原文中机械回取到，取不到就如实降级标注——输出可审计。
写作流水线把同一铁律下移到生成场景：**引用白名单制 + 草稿引用核验率**。

**证据链在界面上闭环**：L2 精读卡片里每条结论都带「✓ 第 N 页」，点页码直接渲染出原文那一页
（服务端 PyMuPDF 出图 + 该页入库纯文本并排显示），肉眼就能核对结论是不是真在那一页上。
不用 `<iframe src=...#page=N>`：浏览器内置 PDF 查看器对 `#page=` 的支持各家不一，
本项目内嵌环境实测根本不跳页——证据链要兑现的动作不能建在一个不保证的行为上。

## 异步任务与进度（W3）

长任务（采集 / L2 精读 / 综述 / Agent run）一律走任务表，不阻塞 HTTP：

```bash
curl -X POST /api/jobs -d '{"kind":"survey","params":{"topic":"XL-MIMO 近场信道估计"}}'
curl /api/jobs/<id>            # 轮询
curl /api/jobs/<id>/events     # SSE 实时进度（15s 心跳，防长 LLM 步骤断流）
```

- 进度持久化在 `jobs` 表，服务重启后仍能看到任务死在哪一步
- **重启恢复**：上个进程留下的 `running` 任务不会再有人推进，启动时如实标 `failed`，
  免得 SSE 客户端对着一个死任务无限轮询
- **有界并发 + 原子认领**：任务在固定大小线程池里跑（`PAPERNEST_MAX_JOBS`，默认 3），
  执行前 `queued → running` 是一条带条件的 UPDATE，重复提交不会跑第二遍
- **同资源去重**：写作流水线的阶段任务按 `dedupe_key=write:<run_id>` 排他，
  连点两次「开写」第二次直接 409——否则两个线程会交错写同一批 `sections` 行
- 每步超时（默认 240s，`timeout_s` 可调）：**协作式取消，不是硬截止**。
  Python 中断不了已开跑的线程（`future.cancel()` 对 running 任务恒为 False），
  所以做两件事：① 调用方不再等待（`ThreadPoolExecutor` 的 `with` 出口默认 `wait=True`，
  会把超时变回「等到天荒地老」）；② 通过 `deadline` 把预算下发给工作线程，
  `llm.chat` 的重试循环与退避 sleep 在每个可中断点自行停手——单次 LLM 调用最坏
  433s（4×90s 超时 + 8/20/45 退避）本来就超过 240s 的步骤预算，不下发预算的话
  被放弃的线程会继续烧钱、继续写库。**单次阻塞的 C 调用（PyMuPDF 解析）仍中断不了**，
  如实标注，不说成硬截止。
- 瞬态故障自动重试一次；`LLMUnavailable`（缺配置）不重试、判 `blocked`
- 成本证据链：`llm_calls` 逐次记录 tokens / latency / cost（价格表在 `.env` 的 `LLM_PRICES_JSON`，未配置的模型 cost 如实记空）

## 工程加固（三期同批）

- **连接不再泄漏**：`db.conn()` 从「返回裸连接」改成真正的 contextmanager。
  `with sqlite3.connect(...)` 只是事务上下文**不关连接**，而全项目有上百处调用点靠 GC 回收句柄，
  WAL 模式下 `-wal`/`-shm` 会持续增长（SSE 每 0.5s 轮询一次就泄一条）。调用点写法不变。
- **schema 版本化迁移**：`PRAGMA user_version` + 幂等迁移函数，替掉原来那串 try/except ALTER；
  `init_db()` 进程内按 DB 路径只跑一次（原来几乎每个请求都 `executescript` 整份 schema）。
- **访问口令：从「可选」改成「暴露时强制」**（2026-09-03）。原来它是可选项，而唯一的部署方式
  `docker-compose` 既把 8765 绑到 0.0.0.0、也没设这个变量——**开箱即用就是同网段任意主机
  可无凭据删项目/笔记、反复触发烧钱的 LLM 任务**。更要命的是：就算设了口令，
  自带前端 30 处 `fetch` 加 1 处 `EventSource` **一个都不带 key**，界面会全线 401，
  于是运维只能撤掉口令退回裸奔——**这道锁事实上不可启用**。现在三处一起收口：
  - `docker-compose.yml` 端口改 `127.0.0.1:8765:8765`，并用 `${PAPERNEST_API_KEY:?…}`
    让缺口令时 compose 直接报错；
  - 应用启动自检：绑定非回环地址却没设口令 → **拒绝启动**（逃生口 `PAPERNEST_ALLOW_NO_AUTH=1`）。
    绑定地址由启动方如实声明（`cli.py serve` 与 Dockerfile 各写 `PAPERNEST_BIND_HOST`），
    因为应用自己问不到 uvicorn 绑在哪；
  - 前端包一层 `window.fetch` 与 `EventSource`：`/api/` 请求自动带 `x-api-key`
    （SSE 不能设请求头，只能拼 `?api_key=`），口令存浏览器 `localStorage`，401 时提示输入。
    **注意 `<img>`/`<iframe>`/`<a href>` 是浏览器自己发起的，包装器碰不到**——
    PDF 阅读视图与页图那三处单独用 `pnUrl()` 把口令拼进 query（漏了这一步的话，
    设了口令 PDF 视图会整条 401，而 `<img onerror>` 会把它显示成「这一页取不到」，
    看起来像渲染 bug）。
  - 口令必须是 **ASCII**：`hmac.compare_digest` 对非 ASCII 的 str 直接 TypeError
    （每个 /api/* 恒 500 而不是 401），且 **HTTP 头本身也装不下非 ASCII**。
    启动时会校验并拒绝，不留「SSE 能过、普通请求全废」的半坏状态。
  测试同时覆盖服务端、**前端契约**与部署配置——此前只有服务端有用例，
  而缺的恰恰是「前端能不能带上口令」，于是「鉴权已覆盖」是一种假信心。
- **出站请求防 SSRF**（`papernest/netguard.py`）：`fulltext.fetch_pdf` 直接拿
  `papers.oa_pdf_url` 发服务端 GET，而那个字段可由用户上传的 `.bib` 写入
  （`POST /api/import/bibliography` → `POST /api/paper/{id}/read` 是一条完整的 SSRF 链路）。
  现在只允许 https + 443，**按解析出的 IP 判**（域名看着正常但解析到 10.x 同样拒绝），
  且**逐跳校验**——原来 `follow_redirects=True`，一个合法外域 302 到 `169.254.169.254`
  就能绕过入口校验。拒绝原因不回吐给调用方：上游状态码是精确的内网探测 oracle。
- **4xx 不重试**：流式那条路径原来走 `raise_for_status()`，`HTTPStatusError` 属于 `httpx.HTTPError`，
  会被下面的 except 接住去退避重试 4 次——与非流式声明的「400 不重试」自相矛盾，401/403 同样白等三分钟。
- **不硬编码私人邮箱**：`OPENALEX_MAILTO` 默认空（不填就不带该参数走匿名池），
  并补了 `.gitignore`——目录还不是 git 仓库，将来 `git init` 极易把含真 key 的 `.env` 一起提交。
- **测试从 60 条加到 257 条**，且**全部离线确定性**。原来的用例会真的去打 embedding 接口
  （开发机 `.env` 配了 `EMBED_MODEL`）——又慢又花钱、结果随网络漂移；摁住后同一批用例
  从 114 秒降到 1.5 秒。新增覆盖：API 层 50 条（错误码、任务生命周期、去重、鉴权、上传、导入、引文图）、
  写作流水线**在线路径**、会话记忆、本地 PDF 入库、BibTeX 解析、引文图算法。

其中一条值得单独说：`pipeline.py` 里 `_write_one` 的在线分支 `return` 引用了一个**从未赋值**的
`final_status`，只要配上真 key，每写完一节就 `NameError`、整个 run 判 failed。
13 个流水线用例全部跑在 `llm.available()=False` 下，在离线分支就提前 return 了，**一个都没碰到那行**——
「二期写作流水线已落地」的说法当时是打了折的。现在 `tests/test_write_online.py` 用假模型把
「撰写⇄评审⇄重写」闭环整个跑起来，专门守着这半边。

## 第 0 步 · 数据源验证（2026-08-29 实测）

查询 `large language model agent evaluation`，每源 20 篇：

| 源 | 结果 | 元数据完整率 | 摘要覆盖率 | OA PDF | 抽样实测下载 |
|---|---|---|---|---|---|
| Semantic Scholar | 20 篇 | **100%** | **100%** | 40% 列出 | **60%**（3/5） |
| OpenAlex | 被本机网络阻断* | - | - | - | - |
| arXiv | 被本机网络阻断* | - | - | - | - |
| **合并去重口径** | 20 篇 | **100%（线 90%，PASS）** | **100%（线 85%，PASS）** | 40% 列出 | **60%（线 40%，PASS）** |

\* 本机网络对 api.openalex.org / export.arxiv.org 直连 RST，且本地代理未开——
代码已支持 `PAPERNEST_PROXY`，开启 Clash 后在 `.env` 填 `http://127.0.0.1:7897` 即启用，无需改代码。
Semantic Scholar 可直连，无 key 共享池按突发拥塞限流（429），客户端已按 20/45/90/150s 退避跨窗口。

报告原件：`data/verify_report.json`。

## 激活真模型（填 key 三步）

```bash
copy .env.example .env   # 填入 LLM_API_KEY；向量检索需同时填 EMBED_MODEL
python cli.py models     # 核对 .env 里的模型 ID 与账号实际可用列表一致（不一致就改）
python cli.py recard && python cli.py embed && python cli.py serve
```

## 合规红线

- 不爬谷歌学术（无官方 API、违反 ToS）；全部走官方学术 API
- 全文精读只对开放获取论文；付费墙 PDF 永不下载（代码硬闸）
- API key 只在 `.env` / `env_file` 注入，不进镜像不进库不进日志；各 API 限速内低频使用，个人自用

## 数字一览（每个数字都附复现方式 · 2026-09-03 重测）

> 报任何 Recall 都要连 **k、检索口径、库容、向量覆盖率**一起报。下表分「线上口径」（向量 + FTS）
> 与「纯 FTS 消融口径」两套，后者 0 token、0 外网、逐位可复现。
> 显著性用符号翻转配对检验（`papernest/stats.py`，20000 次重采样、固定种子），
> **报差值必同时报「多少道题真的变了」和 p 值；不显著就写不显著**。
>
> **向量覆盖率已于 2026-09-03 补齐到 503/503（100%）**（`cli.py embed`，443 篇、约 3.5k 条嵌入）。
> 此前是 104/503（20.7%），而自建集的 42 篇 gold 恰好全部有向量——当时的线上口径数字
> 因此是有偏的。补齐后的重测结果见下面「补齐向量前后的配对对比」。
> `cli.py health --offline` 现在对覆盖率有硬闸门（低于 95% 判失败）。

| 数字 | 值 | 复现方式 |
|---|---|---|
| 代码 / 接口 / 测试 | 20.6k 行 Python（53 个文件）· 83 个 API 端点 · **1155 条离线确定性测试全绿（约 99s）** | `$env:PYTHONPATH='.'; python -m unittest discover -s tests -t tests` |
| 库规模 | 503 篇论文 · 566 页全文 · 784 个章节块 · 1235 条向量（784 chunk / 60 paper / 391 sent）· 2243 条引文边 | `python cli.py stats` · `python cli.py graph stats` · `python cli.py vec status` |
| **向量覆盖率** | **503/503（100%）** 篇能被语义检索命中 | `python cli.py health --offline`（低于 95% 报失败） |
| 自建集 Recall@5 / @10（线上口径 · 关键词串形态） | **0.9427** / 0.9740 | `python cli.py eval --retrieval auto --k 5` |
| 自建集 Recall@5（线上口径 · **自然语言提问形态**） | **0.9427**（与关键词串**逐位相同**） | 同上，看 `retrieval_metric_natural` |
| QASPER paper_hit@5（线上口径） | 0.7727 | `python cli.py qasper eval --n-papers 30` |
| 自建集 Recall@5（纯 FTS 消融 · 关键词串形态） | 0.7552 | `$env:EMBED_MODEL=''; python cli.py eval --retrieval fts --k 5` |
| 自建集 Recall@5（纯 FTS 消融 · 自然语言提问形态） | 0.7552 | 同上，看 `retrieval_metric_natural` |
| 查询预处理修复（自然语言形态） | 0.5312 → **0.7552**，10/32 题，**p=0.0071 显著** | 见「查询预处理」；同一批 gold 只换查询形态 |
| 查询预处理修复（关键词串形态） | 0.6771 → 0.7552，3/32 题，**p=0.256 不显著** | 同上——方向为正但样本量不足以判定 |
| 查询预处理修复（QASPER paper_hit@5） | 0.6364 → 0.7045，10/88 题，**p=0.106 不显著** | 同上 |
| PRF 派生查询「只补位」 | Recall@15 0.8125 → 0.8281（**+0.0156 = 32 题里半道题**） | `--retrieval deep` vs `--retrieval fts` |
| PRF 派生查询 + RRF 融合（负结果） | Recall@5 0.7552 → **0.4740** | `--retrieval deep-rrf` |
| 块级权重选型（QASPER hit@1） | 关闭 0.4886 → 0.2 档 **0.5455** | `$env:PAPERNEST_CHUNK_WEIGHT='...'`，见「L2 全文进全库检索」 |
| BM25 精排（负结果） | QASPER hit@1 0.4672 → 0.3197，池越大越差 | ⚠️ **一次性实验，rerank 模块当前无任何生产调用方**，见下 |
| Cross-Encoder 精排（召回饱和时无收益） | 自建@5 0.9740 → 0.9740，**0/32** 题，延迟 +87% | ⚠️ 同上；且测于 20.7% 覆盖率时期，补齐后未重测 |
| 章节切 vs 按页切（等 8000 字预算的证据召回） | 0.0870 → 0.2754，15/69 题，p=0.0013 | ⚠️ **一次性离线实验，脚本与原始数据未收进仓库，当前无法复现** |
| chunk 向量接入（QASPER hit@1） | 0.4333 → 0.5667，8/60 题，p=0.0073 | ⚠️ 同上（库副本消融，脚本未保留） |
| embedding 配置修复（不是算法） | 自建@5 0.6771 → 0.9427 | 分开 `EMBED_API_BASE`，用 `cli.py models` 核对真实模型 ID |
| 向量覆盖补齐（20.7%→100%） | 自建@5 0.9115 → **0.9427**，**各指标仅 1 题变化** | `cli.py embed` 后重测；配对对比见「评测」一节 |
| RCS 深度问答成本 | 8.7× token / 3.7× 串行延迟，故默认 off | `llm_calls` 表按 purpose 聚合 |
| Milvus 一致性档位 | ⚠️ **两处记录互相矛盾且都无复现脚本**：`milvusstore.py` 注释记 Strong 380.3ms / Bounded 8.0ms（47 倍），本文档旧版记 399.9 / 16.1（25 倍）。**在补上 bench 脚本之前不要引用这组数字。** | — |

简历与面试材料在上级目录（`PaperNest-简历项目经历.md` / `PaperNest-面试问答手册.md`），不随代码分发。

---

*PaperNest 立项 · 2026.08.29 · 第 0 步数据源验证通过后动工；评测与演示材料随四周计划推进更新。
二期（多 Agent 写作系统）· 2026.08.30 立项并落地：选题/大纲/撰写/文献/润色五 Agent 流水线、
引用白名单制、节级内容寻址缓存、机械终检与 md/docx 导出；方案见上级目录二期文档。*
