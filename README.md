<div align="center">

# PaperNest · 文巢

**个人科研文献 Agent：搜文献 · 读文献 · 攒文献 · 引文献 · 写文献**

让 LLM 只负责语义理解与生成，让确定性程序负责流程边界、证据约束和故障处理。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-1165%20passed-2f6f4f)](#-tests)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Stars](https://img.shields.io/github/stars/xlxl1/papernest?style=social)](#)

[English](#) · [简体中文](#) · [功能](#-功能) · [快速开始](#-快速开始) · [架构](#-架构) · [评测](#-评测) · [路线图](#-路线图)

</div>

---

> **设计哲学**：核心成本结构是「每篇论文只让 LLM 精读一次，之后全部走缓存与 RAG」。
> Agent 不是「让多个模型自由讨论」，而是**以证据约束为中心、由确定性工作流控制 LLM 不确定性**的 Research Agent 系统。

## 这是什么

PaperNest 是一个面向**个人研究者**的本地文献工作台，解决「文献来源分散、检索不准、生成内容难以核验、长流程容易中断」这四个问题。它把学术 API、本地 PDF、BibTeX/RIS/Zotero 导入统一到一个本地库里，提供：

- **检索**：向量 + FTS 的 RRF 混合、章节级 chunk 检索、迭代检索（PRF 补位）
- **阅读**：PyMuPDF 按章节/按页解析、表格与插图识别、版面结构还原、扫描件 OCR
- **入库**：单 PDF / 目录递归 / BibTeX / RIS / Zotero CSV，按内容去重
- **引文网络**：阅读缺口、共被引 / 文献耦合推荐
- **写作**：五 Agent 流水线（选题 / 大纲 / 撰写 / 文献 / 润色），引用白名单 + 机械回取校验
- **多轮对话**：服务端会话记忆，会话内 `[n]` 编号固定指代同一篇论文

**不做什么**：不做多租户、不做协同编辑、不训练基础模型、不接入外部写作平台——边界钉在本机单用户。

## ✨ 功能

### Agent 与编排
- 确定性 Router / Planner（不把工具名/参数丢给 LLM 自由组合）
- 工具注册表，5 个工具（search / read / cite / ask / write），有界状态机执行
- 步骤级超时（协作式取消）+ 瞬态故障 4 次抖动退避
- 异步任务表 + SSE 实时进度（含心跳保活）+ 轨迹落库可回放

### 检索
- **RRF 混合检索**（默认）：向量排序 + FTS 词面排序融合
- **章节级 chunk 检索**（`PAPERNEST_CHUNK_SEARCH=1`，默认开）
- **迭代检索（PRF）**：派生查询「只补位」不抢主排序
- **查询形态无关**：关键词串与自然语言形态 Recall@5 一致
- **降级信号透明**：向量不可用时如实上报，不会静默退化成纯 FTS

### 解析与入库
- **多格式入库**：PDF / Word / PPT / HTML / Markdown / 纯文本
- **章节切分**：等上下文预算下证据召回是按页切的 **2.2~3.2 倍**（n=69 配对检验 p=0.0013）
- **表格感知**：整行切分不截断、表头跨块重复
- **元数据抽取**：版面启发式抽标题/DOI/arXiv，抽不出就如实留空 + warnings
- **去重**：sha256 + norm_key（DOI/arXiv/标题归一）双重去重
- **OCR**：可选本地 RapidOCR（离线）或 DashScope 云端

### 评测与证据
- 自建评测集 **82 条**（32 关键词串 + 32 自然语言 + 10 QA + 8 引用）
- QASPER 公开基准接入（`paper_hit@k` + `evidence_recall@sec_k` 两层）
- 证据句**机械回取校验**（按页 / 按节原文逐字找回）
- 三指标评测脚本：`python cli.py eval`

### 写作流水线
- **选题 Agent**：候选带 `[n]` 证据
- **大纲 Agent**：机械校验备注
- **撰写 Agent**：节级内容寻址缓存（改别处不重跑已完稿节）
- **评审 Agent**：总评 + 1-10 分 + major/minor 分级 + 可执行建议
- **润色 Agent**：保留事实与 `[n]` 编号，给出逐条修改说明
- **终检**：悬空/未核验编号、待补证据数、字数、引用核验率 — 全部机械可算
- **学术诚信红线**：引用白名单杜绝编造参考文献，导出稿声明 AI 参与范围

### 工程化
- 单页 Web 界面（`web/index.html`，零依赖）+ REST API
- 异步任务（SQLite 任务表）+ SSE 实时进度 + 心跳保活
- token / latency / cost 逐次落账
- 每日 token 闸门（防失控循环烧额度）
- 日志：单文件轮转 + traceback
- Docker 一键启动（数据挂卷，key 从宿主机 `.env` 注入）

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 复制配置（需要真卡片就填 LLM_API_* 和 S2_API_KEY）
cp .env.example .env

# 3. 三步跑起来
python cli.py verify                            # 第 0 步：复测基础能力
python cli.py ingest "你的课题关键词" --limit 20  # 拉文献进库
python cli.py serve                              # 打开 http://127.0.0.1:8765
```

> **不开 LLM 也能跑**：`EMBED_MODEL=` 留空即强制纯 FTS 口径（0 token、0 外网）。本机自用默认绑 127.0.0.1、无需 API key。

### Docker

```bash
PAPERNEST_API_KEY=your-ascii-password docker compose up --build
# 打开 http://localhost:8765
```

> 容器内 uvicorn 必然绑 0.0.0.0，所以这里**强制要求**口令（ASCII）；本机自用又不想设口令就显式写 `PAPERNEST_ALLOW_NO_AUTH=1`。

### 已有 PDF / Zotero 导入

```bash
python cli.py import-pdf ~/papers/                  # 本地 PDF 入库（按内容去重）
python cli.py import-bib zotero-export.bib          # BibTeX / RIS / Zotero CSV
python cli.py graph fetch --limit 30                # 拉引文边
python cli.py graph gaps                            # 找出「被库内多篇引用、但自己不在库里」的论文
```

### 写作流水线

```bash
python cli.py write "大模型 Agent 的评测方法研究"     # 选题 Agent（候选带 [n] 证据）
python cli.py write-pick <run_id> 1                  # 检查点①：按候选序号定题 → 大纲
python cli.py write-go <run_id>                      # 检查点②：逐节撰写⇄文献白名单⇄评审重写
python cli.py write-finish <run_id>                  # 润色 + 机械终检 + 导出 md/docx
```

## 🏗 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Web (单页 HTML)  +  REST API                   │
│                  /api/papers · /api/agent/run · SSE                  │
└────────────┬──────────────────────────┬─────────────────────────────┘
             │                          │
   ┌─────────▼──────────┐    ┌─────────▼──────────┐
   │      Agent 层       │    │     RAG / 检索     │
   │  Router · Planner  │    │   RRF 混合         │
   │  工具注册 (5个)     │◄──►│   chunk 检索       │
   │  有界状态机执行     │    │   PRF 补位         │
   │  协作式取消          │    │   跨语言候选命中    │
   └─────────┬──────────┘    └─────────┬──────────┘
             │                          │
   ┌─────────▼──────────────────────────▼──────────┐
   │              数据层（SQLite + FTS5）            │
   │  papers · chunks · tables · figures · llm_calls │
   │  jobs · projects · notes · graph · chat_history │
   └─────────┬──────────────────────────┬──────────┘
             │                          │
   ┌─────────▼──────────┐    ┌─────────▼──────────┐
   │    向量后端         │    │   外部 API 适配    │
   │  Milvus（默认）    │    │   OpenAI 兼容接口  │
   │  Chroma / NumPy    │    │   S2 · OpenAlex · arXiv │
   └────────────────────┘    └────────────────────┘
```

**几个关键设计**：

- **检索与组装分离**：检索返回结构化命中（带页码/章节），组装上下文按口径选单元（默认 page，可切 chunk）。检索单元改章节块后做过三臂配对实测，结论与预期相反——见 [`docs/architecture.md`](docs/architecture.md)。
- **缓存与降级**：所有 LLM 调用按 `(prompt 版本, 节标题, ...)` 哈希缓存；embedding 模型换了先用同一文本重嵌一篇做一致性检测（余弦 ≥0.98 原地迁移）。
- **降级透明**：检索降级为纯 FTS 时返回字段里写明 `degraded` 与原因，不让调用方靠 mode 字符串猜。

## 🔧 配置

主要环境变量（完整列表见 `.env.example`）：

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_API_BASE` | ✅ | OpenAI 兼容端点 |
| `LLM_API_KEY` | ✅ |  |
| `LLM_MODEL` | ✅ | 默认模型 |
| `LLM_MODEL_HEAVY` |  | 复杂任务（综述/评审）使用的强模型 |
| `EMBED_MODEL` |  | 留空 = 强制纯 FTS 口径 |
| `EMBED_API_BASE` |  | 单独配（chat 端点常无 `/embeddings` 路由） |
| `EMBED_API_KEY` |  | 不填则回退 `DASHSCOPE_API_KEY` → `LLM_API_KEY` |
| `S2_API_KEY` |  | Semantic Scholar，不填走共享池（限流更严） |
| `RESEARCH_TOPIC` |  | 卡片「与我课题的关系」一栏的上下文 |
| `PAPERNEST_API_KEY` |  | 绑非回环地址时**强制要求**（ASCII） |
| `PAPERNEST_DAILY_TOKEN_BUDGET` |  | 防失控烧额度，默认 3,000,000 |
| `PAPERNEST_CHUNK_SEARCH` |  | L2 全文进全库检索（默认 1） |

## 📊 评测

> **前提**：`.env` 里 `LLM_API_*` 配齐且 `cli.py embed` 跑过后，`PAPERNEST_VECTOR_BACKEND=milvus` 默认配置。EMBED_MODEL 留空即纯 FTS 口径。

| 评测口径 | 维度 | 数值 |
|---|---|---|
| 自建集 82 条 Recall@5（线上向量+FTS） | 503 篇库 | **0.9740** |
| 自建集 Recall@5（纯 FTS 兜底） | 同上 | 0.7552 |
| QASPER hit@1（88 题） | 块级权重 0.2 | **0.5455** |
| QASPER hit@5 | 同上 | 0.7045 |
| 引用证据核验率（自建集 8 条） | 机械回取 | **0.85** |
| 无证据率 | 越低越好 | 0.30 |
| 自动化测试 | 全部离线确定性 | **1165 条** |

详细评测报告（含 A/B 对照、跨语言、配对检验）见 [`docs/eval.md`](docs/eval.md)。

## 🛠 开发

```bash
# 跑测试（全部离线确定性，不联网不花钱）
PYTHONPATH=. python -m unittest discover -s tests -t tests

# 评测（需要配 .env 后跑一次 cli.py embed）
python cli.py eval                  # 检索 + 引用
python cli.py eval --qa             # 加测无证据率（调真模型）
python cli.py qasper eval           # QASPER 公开基准

# 格式与白名单
git diff --cached --check           # 检查空白字符与冲突标记
git status --short                  # 确认没有 .env、数据库或 PDF
```

**提交规范**：`<type>(<scope>): <subject>`，类型包括 `feat` / `fix` / `docs` / `refactor` / `test` / `chore`。

## 🗺 路线图

按「价值/实现成本」排序，来自对 PaperQA2 / STORM / papersgpt / gpt-researcher / react-pdf-highlighter / ragas 的对比调研：

1. **RCS 检索增强** — ✅ 已落地（LLM 重排 + 逐篇定向摘要 + 机械回取校验）
2. **ragas 式 faithfulness 指标** — 把回答拆成 claim 逐条对照上下文
3. **PDF 高亮标注 + 选中即问**（react-pdf-highlighter 风格）
4. **Zotero 单向导入 + note 写回**（pyzotero，412 冲突重试）
5. **MCP server 输出**：把检索/问答/引文网络暴露为 MCP 工具
6. **Undermind 式检索 Agent**：LLM 相关性评审 + 覆盖率停止条件
7. **STORM 视角引导的综述规划**

**不做**：多租户、多人协同编辑、训练基础模型、外部写作平台对接。

## 📁 仓库结构

```text
papernest/
├── papernest/          # 核心 Python 包：检索 / RAG / Agent / 写作 / 导入
│   ├── api.py          # FastAPI 路由
│   ├── agent.py        # Router / Planner / 工具注册 / 状态机
│   ├── rag.py          # RRF 混合检索 + chunk 检索 + PRF
│   ├── chunkembed.py   # 章节块向量
│   ├── writing.py      # 写作流水线（五 Agent）
│   ├── chat.py         # 服务端多轮对话记忆
│   ├── pdfimport.py    # 本地 PDF 入库
│   ├── bibimport.py    # BibTeX / RIS / Zotero CSV 导入
│   ├── graph.py        # 引文网络
│   ├── milvusstore.py  # Milvus 向量后端
│   └── sources/        # 外部 API 适配（arxiv / s2 / openalex）
├── web/index.html      # 单页 Web 界面
├── tests/              # 离线确定性测试
├── tools/              # 数据迁移 / 评测辅助脚本
├── cli.py              # 命令行入口
├── requirements.txt
├── docker-compose.yml
└── .env.example        # 配置模板（不含密钥）
```

数据、PDF、向量索引、上传文件和评测报告保存在 `data/`，已由 `.gitignore` 统一排除。

## 🤝 贡献

欢迎提 Issue / PR。本仓库不接受任何形式的：

- **编造的实验数据或文献**（本项目以引用白名单杜绝）
- **绕过机械回取校验** 的 PR
- **静默改变默认行为** 而不在 PR 描述里说明

提交前请跑通本地测试，并确保 `git status --short` 干净。

## 📄 License

[MIT](LICENSE)

## 🙏 致谢

设计参考：[PaperQA2](https://github.com/Future-House/paper-qa) · [STORM](https://github.com/stanford-oval/storm) · [papersgpt](https://github.com/papersgpt/papersgpt) · [gpt-researcher](https://github.com/assafelovic/gpt-researcher) · [react-pdf-highlighter](https://github.com/agentcooper/react-pdf-highlighter) · [ragas](https://github.com/explodinggradients/ragas) · [QASPER](https://arxiv.org/abs/2105.09611)

向量库：[Milvus](https://milvus.io/) · 解析：[PyMuPDF](https://github.com/pymupdf/PyMuPDF) · LLM 端点：[DashScope](https://dashscope.aliyun.com/) · [OpenAI](https://openai.com/) · [Ollama](https://ollama.com/)

---

<div align="center">
<sub>如果 PaperNest 帮到了你的研究，给个 ⭐ 是最好的反馈</sub>
</div>
