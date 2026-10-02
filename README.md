# PaperNest · 文巢

PaperNest 是一个面向个人研究者的本地科研文献工作台。它把文献搜索、PDF 解析、证据检索、引文网络和受约束写作放在同一个本地项目中。

项目的核心原则是：模型负责语义理解和生成，程序负责文件校验、版面解析、证据定位、状态流转、缓存和错误报告。模型生成的摘要、图像说明和表格说明都会标记为模型内容，不能冒充论文原文。

## 当前能力

- 从 Semantic Scholar、OpenAlex、arXiv 或本地文件导入论文。
- 支持 PDF、Word、PPT、HTML、Markdown、纯文本和 BibTeX/RIS/Zotero CSV。
- 按 DOI、arXiv ID、标题归一化和文件 SHA-256 去重。
- 使用 SQLite、FTS5 和可选 embedding 实现关键词、向量和混合检索。
- 支持章节级检索块、页级证据、跨语言候选召回、PRF 补充检索和 RCS 重排。
- 提供 Plan-Execute-Replan Agent，支持意图分类、澄清问题、有限状态执行、失败重试和过程追踪。
- 支持多轮问答、引用编号固定、引用白名单和机械证据回取。
- 支持选题、列大纲、撰写、评审、润色和导出 Markdown/DOCX 的写作流程。
- 支持引文网络、缺口分析、跨论文矩阵和 Obsidian 导出。
- 提供 Web 界面、REST API、SSE 任务进度和 Docker 启动方式。

## PDF 处理流程

1. 校验 PDF 魔数、文件大小和内容哈希，抽取标题、作者、摘要、DOI 和 arXiv ID。
2. 使用 PyMuPDF 读取文本层，按页保存原文，并按章节生成检索块。
3. 整份扫描 PDF 或混合 PDF 的无文本页面可以使用 OCR。OCR 文本会标记为“非文本层”。
4. 识别摘要单栏、正文双栏、跨栏图表等版面结构；结构失败时会退化为按页切分并报告降级。
5. 检测公式候选并保存页码、bbox、原始文本和置信度。LaTeX 识别需要额外数学 OCR 插件。
6. 识别有图注的矢量图和无图注的嵌入位图。无图注图片先以占位图块进入检索，之后可以补视觉摘要。
7. 识别带框线和无框线表格，支持有明确表题的两列表格、跨页续表和表头重复。复杂合并单元格保留复核标记。
8. 将正文、公式、表格和图片分别写入检索块，保存页码、章节、bbox、来源和块类型。
9. 通过 FTS5 和可选向量索引检索；向量不可用时会明确报告降级状态。

详细说明见 [PDF 处理与 OCR 配置](docs/pdf-backends.md)。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
cp .env.example .env
python cli.py --help
```

本机演示不需要启动 Milvus，可以在 `.env` 中设置：

```env
PAPERNEST_VECTOR_BACKEND=numpy
```

没有 LLM 或 embedding 时，项目仍可以使用 FTS 关键词检索：

```env
EMBED_MODEL=
```

导入本地 PDF：

```bash
python cli.py import-pdf ./papers --no-cards
```

检查配置、索引和检索：

```bash
python cli.py health
```

启动 Web 服务：

```bash
python cli.py serve
```

默认访问地址为 `http://127.0.0.1:8765`。如果绑定非回环地址，必须设置 ASCII 格式的 `PAPERNEST_API_KEY`。

## 常用命令

```bash
# 导入 PDF、目录或多个 PDF
python cli.py import-pdf paper.pdf
python cli.py import-pdf ./papers --topic "大模型 Agent 评测"

# 读取论文并生成精读卡片
python cli.py read <paper_id> --topic "我的研究方向"

# 关键词或自然语言问答
python cli.py ask "这篇论文的主要方法是什么？"
python cli.py ask --deep "不同方法的实验差异是什么？"
python cli.py ask --rcs "哪些论文支持这个结论？"

# 公式、结构和规范化 Markdown
python cli.py formulas <paper_id>
python cli.py structure <paper_id>
python cli.py pdf-markdown <paper_id>

# 表格和图片增强
python cli.py table-summary
python cli.py figure-summary
python cli.py enrich

# 重切块和建立向量
python cli.py rechunk
python cli.py vec status
python cli.py vec embed-chunks --dry-run
python cli.py vec embed-chunks
python cli.py vec rebuild

# BibTeX、RIS、Zotero CSV 和引文网络
python cli.py import-bib export.bib
python cli.py graph fetch --limit 30
python cli.py graph gaps

# 评测
python cli.py eval
python cli.py qasper eval
```

表格摘要和图片摘要默认只试算，不调用模型。确认配置和费用后再加 `--apply`：

```bash
python cli.py table-summary --apply
python cli.py figure-summary --apply
python tools/migrate_chunks.py --apply
python cli.py vec embed-chunks
```

## 配置

完整模板在 [.env.example](.env.example)。常用配置如下：

| 变量 | 作用 |
|---|---|
| `LLM_API_BASE` | OpenAI 兼容的 `/chat/completions` 地址 |
| `LLM_API_KEY` | LLM API 密钥 |
| `LLM_MODEL` | 普通任务模型 |
| `LLM_MODEL_HEAVY` | 精读、评审等复杂任务模型 |
| `EMBED_API_BASE` | embedding API 地址，可与 chat 分开 |
| `EMBED_API_KEY` | embedding API 密钥 |
| `EMBED_MODEL` | embedding 模型；留空则使用 FTS |
| `PAPERNEST_VECTOR_BACKEND` | `numpy` 或 `milvus` |
| `PAPERNEST_VECTOR_FALLBACK` | 设置为 `1` 时，Milvus 不可用可降级到 numpy |
| `PAPERNEST_OCR` | `off`、`local`、`cloud` 或 `auto` |
| `PAPERNEST_OCR_DPI` | OCR 渲染分辨率，默认 200 |
| `PAPERNEST_OCR_CLOUD_MODEL` | 云端 OCR 模型 |
| `PAPERNEST_MATH_OCR_MODULE` | 提供 `recognize_formula()` 的数学 OCR 模块 |
| `PAPERNEST_VISION_MODEL` | 图片视觉摘要模型，默认关闭 |
| `S2_API_KEY` | Semantic Scholar API 密钥，可选 |
| `RESEARCH_TOPIC` | 精读卡片中的研究方向上下文 |
| `PAPERNEST_API_KEY` | 对外绑定地址时的服务访问口令 |

### OCR

本地 OCR 使用 RapidOCR：

```bash
pip install rapidocr_onnxruntime
```

```env
PAPERNEST_OCR=local
```

云端 OCR 需要 `LLM_API_BASE`、`LLM_API_KEY` 和支持图片输入的视觉端点：

```env
PAPERNEST_OCR=cloud
PAPERNEST_OCR_CLOUD_MODEL=qwen-vl-ocr-latest
```

普通 OCR 不能可靠生成公式 LaTeX。数学 OCR 模块的接口是：

```python
def recognize_formula(png_bytes: bytes, item: dict) -> dict:
    return {"latex": r"\frac{a}{b}", "confidence": 0.95}
```

### 复杂 PDF

当前默认解析器是 PyMuPDF，适合普通文本层 PDF。扫描件、公式密集页、复杂跨栏表格和没有图注的纯矢量图可以使用外部解析器辅助：

- [Marker](https://github.com/datalab-to/marker)：适合多栏、公式、图片和表格综合解析。
- [MinerU](https://github.com/opendatalab/MinerU)：适合扫描件、复杂版面和结构化表格。

外部解析器负责生成页面 block、bbox、表格行和置信度，PaperNest 继续负责检索、引用、向量和审计。

## 系统结构

```text
Web / REST API / CLI
          │
          ▼
Agent：意图分类 → 计划 → 工具执行 → 结果检查 → 重规划
          │
          ├── 文献来源：S2 / OpenAlex / arXiv / 本地文件
          ├── 文档处理：PyMuPDF / OCR / 公式 / 图片 / 表格
          ├── 检索：SQLite FTS5 + embedding + RRF + PRF + RCS
          ├── 证据：页码、章节、bbox、机械回取
          └── 写作：选题、提纲、撰写、评审、润色、导出
```

SQLite 是论文、页文本、检索块、公式审计、摘要、任务和向量原始数据的主存储。numpy 适合单机和中小规模数据；Milvus 适合生产部署和大规模向量检索。

## 评测与验证

当前代码库包含离线确定性测试和 PDF 回归测试。最近一次验证结果：

- 完整测试：1557 项通过。
- OCR、PDF 导入、表格和图片专项测试：117 项通过。
- `python -m compileall -q .` 通过。
- `python cli.py --help` 和核心模块导入通过。
- numpy 向量后端健康检查：配置、索引、embedding、检索 4 项通过。

评测命令会读取本地评测集；使用真实 LLM 或 embedding 时会产生网络请求和费用：

```bash
python cli.py eval
python cli.py eval --qa
python cli.py qasper eval
```

## 已知边界

- 默认全文处理前 40 页，单个 PDF 上限 50 MB。
- OCR 文本可能出现错字，引用时应回看原 PDF。
- 公式候选检测已经实现，LaTeX 需要单独配置数学 OCR。
- 无图注的纯矢量图没有 image bbox，需要 Marker/MinerU 等版面模型。
- 复杂合并单元格会保留人工复核标记，不会自动填入未知值。
- 图片视觉摘要和表格自然语言摘要需要额外的 LLM/视觉模型配置。
- 新环境没有 embedding 配置时只能使用 FTS；已有向量数据库不会随源码提交。

## 项目结构

```text
papernest/
├── papernest/          # 核心 Python 包
│   ├── api.py          # FastAPI 路由
│   ├── agent.py        # Agent 状态机和工具编排
│   ├── rag.py          # 混合检索、PRF、上下文组装
│   ├── pdfimport.py    # PDF 校验、元数据、OCR 入库
│   ├── fulltext.py     # 页文本、章节块和规范化 Markdown
│   ├── structure.py    # 章节、单双栏和正文结构
│   ├── tables.py       # 表格识别与行级切分
│   ├── figures.py      # 图注和无图注图片识别
│   ├── formulas.py     # 公式候选和数学 OCR 接口
│   ├── ocr.py          # RapidOCR / 云端 OCR
│   ├── vectorstore.py  # numpy / Chroma / Milvus 抽象
│   └── sources/        # S2、OpenAlex、arXiv 适配器
├── web/                # 单页 Web 界面
├── tests/              # 离线确定性测试
├── tools/              # 迁移、评测和 OCR 工具
├── cli.py              # 命令行入口
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── docker-compose.milvus.yml
└── .env.example        # 不含密钥的配置模板
```

运行数据、数据库、PDF、向量索引、日志和评测报告保存在 `data/`，由 `.gitignore` 排除。上传项目时不要提交 `.env`、`.env.bak*`、`data/`、`.tmp/`、`__pycache__/` 和 `.git/`。

## Docker

本机服务：

```bash
PAPERNEST_API_KEY=your-ascii-password docker compose up --build
```

生产向量后端：

```bash
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up --build -d
```

Docker 会把 `data/` 挂载到容器中。服务默认只绑定 `127.0.0.1`；对外提供服务时必须设置访问口令并自行配置反向代理或端口暴露。

## License

[MIT](LICENSE)
