import contextlib
import json
import re
import sqlite3
import threading
from . import config

# 写并发下等待锁的上限：默认 5s 在「SSE 轮询 + 后台任务写库」并发下太短
BUSY_TIMEOUT_S = 30.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  norm_key TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  abstract TEXT,
  year INTEGER,
  venue TEXT,
  authors TEXT,            -- JSON 数组
  doi TEXT,
  arxiv_id TEXT,
  citation_count INTEGER,
  oa_pdf_url TEXT,
  source TEXT,             -- 首次入库来源: s2 / openalex / arxiv / upload / bibtex
  s2_id TEXT,
  level INTEGER DEFAULT 0, -- 0=L0 元数据 1=L1 卡片 2=L2 全文精读
  card_json TEXT,
  card_model TEXT,         -- 生成卡片的模型（mock-extractive = 摘要直取）
  pdf_path TEXT,           -- 本地 PDF 路径（OA 下载或用户上传）
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now','localtime')),
  purpose TEXT NOT NULL,
  paper_id INTEGER,
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  latency_ms REAL,
  cost_usd REAL,
  model TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
  run_id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  plan_json TEXT NOT NULL DEFAULT '[]',
  events_json TEXT NOT NULL DEFAULT '[]',
  answer TEXT,
  error TEXT,
  total_ms INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  params_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
  stage TEXT DEFAULT '',
  progress REAL DEFAULT 0,                -- 0 ~ 1
  message TEXT DEFAULT '',
  result_json TEXT,
  error TEXT,
  dedupe_key TEXT,                        -- 同一资源同时只允许一个在跑（如 write:<run_id>）
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
-- 注：dedupe_key 上的索引建在迁移 v3 里，不能放这儿——旧库的 jobs 表已存在，
-- CREATE TABLE IF NOT EXISTS 不会补列，索引会先于 ALTER 执行而报「no such column」。
CREATE TABLE IF NOT EXISTS search_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now','localtime')),
  query TEXT NOT NULL,
  source TEXT,             -- s2 / openalex / s2+arxiv ...
  results INTEGER,
  new_papers INTEGER,
  llm_calls INTEGER
);
CREATE TABLE IF NOT EXISTS query_cache (
  query TEXT PRIMARY KEY,        -- 归一化查询（lower + 压缩空白）
  source TEXT,
  keys_json TEXT NOT NULL,       -- 结果集的 norm_key 有序列表
  ts TEXT DEFAULT (datetime('now','localtime'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
  title, abstract, keywords, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS vectors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL,
  kind TEXT NOT NULL,            -- paper | sent
  idx INTEGER DEFAULT 0,
  text TEXT,
  model TEXT,
  vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors ON vectors(paper_id, kind, model);
CREATE TABLE IF NOT EXISTS pages (
  paper_id INTEGER NOT NULL,
  page_no INTEGER NOT NULL,
  text TEXT,
  PRIMARY KEY(paper_id, page_no)
);
-- L2 精读过的全文也要能被全库检索命中。papers_fts 只索引 title/abstract/keywords，
-- 于是「精读一次」拿到的正文在全库检索里等于不存在——这是 L2 最大的浪费。
-- 单开一张页级表而不是把全文塞进 papers_fts 的第四列：命中要能定位到页，
-- 才对得上本项目「证据带页码」的口径。
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  paper_id UNINDEXED, page_no UNINDEXED, text, tokenize='trigram'
);
-- 检索单元与存储单元分离：pages 按物理页存（证据回取要报准页码），
-- chunks 按**章节**切（检索要的是语义完整的块）。实测（25 篇真实 arXiv PDF、
-- 69 道 QASPER 带 gold 证据的题）：等上下文预算下按章节切的证据召回是按页切的
-- 2.2~3.2 倍；按页切要 16000 字才够到的召回，按章节切 8000 字就够到——预算减半。
CREATE TABLE IF NOT EXISTS chunks (
  paper_id INTEGER NOT NULL,
  chunk_no INTEGER NOT NULL,
  section_path TEXT DEFAULT '',   -- 如 "3 Method > 3.2 Training"，检索结果可溯源到章节
  level INTEGER DEFAULT 1,
  start_page INTEGER,             -- 仍然记页码，证据链不断
  end_page INTEGER,
  kind TEXT DEFAULT 'text',       -- text | table（表格块整行不截断、表头跨块重复）
  text TEXT,
  PRIMARY KEY(paper_id, chunk_no)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  paper_id UNINDEXED, chunk_no UNINDEXED, text, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL,
  kind TEXT,
  sym TEXT,
  latex TEXT,
  meaning TEXT,
  page INTEGER
);
CREATE INDEX IF NOT EXISTS idx_symbols ON symbols(paper_id);
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT DEFAULT '',
  topic TEXT DEFAULT '',
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER,
  paper_id INTEGER,
  title TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  note_type TEXT NOT NULL DEFAULT 'insight',
  tags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime')),
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
  FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_project ON notes(project_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_notes_paper ON notes(paper_id, updated_at);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER,
  title TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'draft',
  version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime')),
  FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id, updated_at);
CREATE TABLE IF NOT EXISTS document_sources (
  document_id INTEGER NOT NULL,
  paper_id INTEGER NOT NULL,
  page_no INTEGER,
  quote TEXT DEFAULT '',
  relation TEXT DEFAULT 'support',
  created_at TEXT DEFAULT (datetime('now','localtime')),
  PRIMARY KEY(document_id, paper_id, page_no),
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS writing_runs (
  id TEXT PRIMARY KEY,                    -- uuid hex（写作流水线一次全程）
  project_id INTEGER,
  topic TEXT NOT NULL DEFAULT '',         -- 课题描述（选题 Agent 的输入）
  title TEXT DEFAULT '',                  -- 定题后的题目（检查点①人工选定）
  status TEXT NOT NULL DEFAULT 'topic_running',
  -- topic_running | pending_user_topic | outline_running | pending_user_outline
  -- drafting | drafted | polishing | done | failed
  stage TEXT DEFAULT '',                  -- 当前进度说明（UI/SSE 展示）
  job_id TEXT,                            -- 正在跑（或最近一次）的异步任务
  topics_json TEXT DEFAULT '[]',          -- 选题 Agent 的候选（带 [n] 证据）
  glossary_json TEXT DEFAULT '{}',        -- 全局术语表（长文一致性）
  config_json TEXT DEFAULT '{}',          -- target_words / max_rewrites 等
  result_json TEXT,                       -- 终检报告 + 导出路径（done 后）
  error TEXT,
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS outlines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,     -- 人工改纲 → 版本 +1，旧版保留可回滚
  title TEXT DEFAULT '',
  outline_json TEXT NOT NULL,             -- sections 数组：no/title/points/paper_ids/words
  warnings_json TEXT DEFAULT '[]',        -- 机械校验未过项（如实记录，不静默）
  edited INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now','localtime')),
  FOREIGN KEY(run_id) REFERENCES writing_runs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  outline_version INTEGER NOT NULL,
  sec_no INTEGER NOT NULL,
  title TEXT NOT NULL,
  points_json TEXT DEFAULT '[]',
  whitelist_json TEXT DEFAULT '[]',       -- 文献 Agent 放行的引用白名单（证据句已核验）
  cache_key TEXT DEFAULT '',              -- 内容寻址：hash(prompt 版本,论点,白名单,词表…)
  content TEXT DEFAULT '',                -- 节草稿 Markdown（[n] 为白名单局部编号）
  citations_json TEXT DEFAULT '{}',       -- {"n": {paper_id,title,evidence,verified,…}}
  removed_json TEXT DEFAULT '[]',         -- 生成稿中被剥离的白名单外编号（如实记录）
  status TEXT DEFAULT 'pending',          -- pending|drafting|done|failed|reused
  attempt INTEGER DEFAULT 0,              -- 1=初稿，2/3=带评审意见重写
  score INTEGER,                          -- 评审完成度分（1-10，离线为 NULL）
  review_json TEXT,                       -- 最近一次评审报告
  updated_at TEXT DEFAULT (datetime('now','localtime')),
  FOREIGN KEY(run_id) REFERENCES writing_runs(id) ON DELETE CASCADE,
  UNIQUE(run_id, outline_version, sec_no)
);
CREATE INDEX IF NOT EXISTS idx_sections_cache ON sections(run_id, cache_key);
CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  title TEXT DEFAULT '',
  source_map_json TEXT NOT NULL DEFAULT '{}',  -- paper_id → 会话内固定的 [n] 编号
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,                          -- user | assistant
  content TEXT NOT NULL DEFAULT '',
  sources_json TEXT NOT NULL DEFAULT '[]',     -- 该轮实际进上下文的文献（可回放）
  degraded TEXT,                               -- 给人看的那句中文（前端直接渲染）
  -- 同一批降级信号的结构化形式：[{code,message,critical}]。
  -- degraded 那句中文是拼串，回不去——没法回答「上周有多少次问答是在向量挂掉的
  -- 状态下答的」。code 是可 grep、可 GROUP BY 的稳定标识，词汇表见 degrade.py。
  degraded_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT (datetime('now','localtime')),
  FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chat_messages ON chat_messages(session_id, id);
"""


_WAL_DONE: set[str] = set()
_WAL_LOCK = threading.Lock()


def _ensure_wal(c: sqlite3.Connection):
    """每个库文件只设一次 journal_mode。

    `PRAGMA journal_mode=WAL` 是**写操作**——它要短暂拿排它锁去改数据库头，
    即使已经是 WAL 也照做一遍。实测在 Windows 上每次连接要 6ms，
    而连接本身加其余 PRAGMA 合计不到 1ms：**全项目每一次数据库访问都在白付这 6ms**
    （一次混合检索开 2~3 个连接，SSE 每 0.5s 轮询一次）。

    而 journal_mode 是数据库文件的持久属性，设过一次就永久是 WAL，
    跨连接、跨重启都保持。所以按库文件路径记一次即可，实测 7.06ms → 0.52ms。
    """
    path = str(config.DB_PATH)
    with _WAL_LOCK:
        done = path in _WAL_DONE
    if done:
        return
    c.execute("PRAGMA journal_mode=WAL")
    with _WAL_LOCK:
        _WAL_DONE.add(path)


@contextlib.contextmanager
def conn(immediate: bool = False):
    """事务 + 连接的双重上下文管理器：正常退出提交、异常回滚，无论如何 close。

    坑：`with sqlite3.connect(...) as c` 只是事务上下文，**不关连接**——全项目
    上百处调用点靠 GC 回收句柄，WAL 模式下 -wal/-shm 与文件句柄会持续增长
    （SSE 每 0.5s 轮询一次就泄一条）。所以这里收口成 contextmanager，
    调用点写法 `with db.conn() as c:` 不变，但拿到的是真正会关闭的连接。

    `immediate=True` 立刻取写锁（`BEGIN IMMEDIATE`），用于**读-改-写**这种
    必须整体原子的操作。默认的 `isolation_level=''` 只在 DML 前隐式开事务，
    **SELECT 走 autocommit**——于是「先 SELECT 再算再 UPDATE」中间没有任何隔离，
    是教科书式的 lost update。实测（24 线程并发给不同论文要编号）：
    24 篇只拿到 5 个不同编号、映射表丢掉 19 篇。
    并发下第二个 BEGIN IMMEDIATE 会等锁，最多等 BUSY_TIMEOUT_S。
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, timeout=BUSY_TIMEOUT_S)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    _ensure_wal(c)
    c.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_S * 1000)}")
    if immediate:
        c.execute("BEGIN IMMEDIATE")
    try:
        yield c
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        c.close()


# ── schema 版本与迁移 ──
# 每个版本一个「幂等」迁移函数；PRAGMA user_version 记录已应用到哪一版。
# 旧库（user_version=0）会顺序补齐，新库建表后直接盖到最新版。

SCHEMA_VERSION = 6


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}


def _add_column(c: sqlite3.Connection, table: str, col: str, decl: str):
    if col not in _columns(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _migrate_1(c: sqlite3.Connection):
    """v1：早期库缺的列（新库已在 SCHEMA 里，这里只补旧库）。"""
    _add_column(c, "papers", "pdf_path", "TEXT")
    _add_column(c, "llm_calls", "latency_ms", "REAL")
    _add_column(c, "llm_calls", "cost_usd", "REAL")


def _migrate_2(c: sqlite3.Connection):
    """v2：本地上传 PDF 的去重指纹 + 论文软标签。"""
    _add_column(c, "papers", "file_sha256", "TEXT")
    c.execute("CREATE INDEX IF NOT EXISTS idx_papers_sha ON papers(file_sha256)")


def _migrate_3(c: sqlite3.Connection):
    """v3：任务去重键（防同一写作 run 并发提交两个阶段任务）。"""
    _add_column(c, "jobs", "dedupe_key", "TEXT")
    c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs(dedupe_key, status)")


def _migrate_4(c: sqlite3.Connection):
    """v4：把已有的 pages 全文补建进 pages_fts（新库这步是空跑）。"""
    have = c.execute("SELECT COUNT(*) n FROM pages_fts").fetchone()[0]
    if have:
        return
    c.execute("""INSERT INTO pages_fts(paper_id,page_no,text)
                 SELECT paper_id, page_no, COALESCE(text,'') FROM pages""")


def _migrate_5(c: sqlite3.Connection):
    """v5：老库把 pages 直接搬进 chunks 作为兜底检索单元。

    有 PDF 的论文，主进程之后可以用 `cli.py rechunk` 重切成章节块（更好）；
    这里先保证升级完检索不空窗——一节都不少，只是粒度还是页。
    """
    have = c.execute("SELECT COUNT(*) n FROM chunks").fetchone()[0]
    if have:
        return
    c.execute("""INSERT INTO chunks(paper_id,chunk_no,section_path,level,
                   start_page,end_page,kind,text)
                 SELECT paper_id, page_no, '', 1, page_no, page_no, 'text',
                        COALESCE(text,'') FROM pages""")
    c.execute("""INSERT INTO chunks_fts(paper_id,chunk_no,text)
                 SELECT paper_id, chunk_no, text FROM chunks""")


def _migrate_6(c: sqlite3.Connection):
    """v6：chat_messages 增加结构化降级列。

    纯增量：老库补一列、默认 '[]'，已有的 `degraded` 中文串原样保留不动。
    历史行不做回填——`degrade.from_legacy` 能在读取时把老串兜成 code='unknown'，
    而**猜**出来的 code 入库会污染聚合口径，比留空更糟。
    """
    _add_column(c, "chat_messages", "degraded_json", "TEXT NOT NULL DEFAULT '[]'")


MIGRATIONS = {1: _migrate_1, 2: _migrate_2, 3: _migrate_3, 4: _migrate_4,
              5: _migrate_5, 6: _migrate_6}

_init_lock = threading.Lock()
_initialized: set[str] = set()


def _migrate(c: sqlite3.Connection):
    ver = c.execute("PRAGMA user_version").fetchone()[0]
    if ver >= SCHEMA_VERSION:
        return
    for step in range(ver + 1, SCHEMA_VERSION + 1):
        fn = MIGRATIONS.get(step)
        if fn:
            fn(c)
    c.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def init_db(force: bool = False):
    """建表 + 迁移。进程内按 DB 路径只跑一次（原来每个请求都 executescript 整份 schema）。"""
    key = str(config.DB_PATH)
    if not force and key in _initialized and config.DB_PATH.exists():
        return
    with _init_lock:
        if not force and key in _initialized and config.DB_PATH.exists():
            return
        with conn() as c:
            c.executescript(SCHEMA)
            _migrate(c)
        _initialized.add(key)


def get_by_norm_key(c: sqlite3.Connection, key: str):
    return c.execute("SELECT * FROM papers WHERE norm_key=?", (key,)).fetchone()


def insert_l0(c: sqlite3.Connection, p: dict) -> int:
    cur = c.execute(
        """INSERT INTO papers(norm_key,title,abstract,year,venue,authors,doi,arxiv_id,
             citation_count,oa_pdf_url,source,s2_id,level)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,
             CASE WHEN ? IS NOT NULL THEN 1 ELSE 0 END)""",
        (p["norm_key"], p["title"], p.get("abstract"), p.get("year"), p.get("venue"),
         json.dumps(p.get("authors") or [], ensure_ascii=False), p.get("doi"),
         p.get("arxiv_id"), p.get("citation_count"), p.get("oa_pdf_url"),
         p.get("source"), p.get("s2_id"), p.get("abstract")),
    )
    c.execute("INSERT INTO papers_fts(rowid,title,abstract,keywords) VALUES(?,?,?,?)",
              (cur.lastrowid, p["title"], p.get("abstract") or "", ""))
    return cur.lastrowid


def save_card(c: sqlite3.Connection, paper_id: int, card: dict, model: str):
    row = c.execute("SELECT title, abstract FROM papers WHERE id=?", (paper_id,)).fetchone()
    keywords = json.dumps(card.get("keywords") or [], ensure_ascii=False)
    c.execute(
        """UPDATE papers SET card_json=?, card_model=?, level=MAX(level,1),
             updated_at=datetime('now','localtime') WHERE id=?""",
        (json.dumps(card, ensure_ascii=False), model, paper_id),
    )
    c.execute("DELETE FROM papers_fts WHERE rowid=?", (paper_id,))
    c.execute("INSERT INTO papers_fts(rowid,title,abstract,keywords) VALUES(?,?,?,?)",
              (paper_id, row["title"], row["abstract"] or "", keywords))


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


_CJK_RUN = re.compile(r"[一-鿿]+")
#: 混合词里的 ASCII 段（"XL-MIMO的近场估计" → "XL-MIMO"）
_ASCII_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-/]*")
#: 标点归一：保留字母数字下划线与 CJK（都在 \w 里），外加技术词里有意义的 + # . - /
#: （C++、C#、GPT-4、wav2vec2.0、XL-MIMO 都靠它们）。其余一律换成空格。
_PUNCT = re.compile(r"[^\w+#./-]+", re.UNICODE)
#: 词面首尾要剥掉的符号——'estimation.' 与 'estimation' 在 trigram 下是两个不同的词
_EDGE = ".-/"

#: FTS5 trigram 分词器的硬下限：短于 3 个**字符**的词面命中恒为 0。
#: 这是在本库 papers_fts 上实测出来的，不是按字节推的——'智能'=0、'模型'=0、
#: '信道'=0，连 ASCII 的 'ai' 也是 0；3 字符才有命中（'智能体'=7、'信道估'=11、'LLM'=17）。
#: 所以「放行中文二元词面进 MATCH」这个想法是错的：二元词面永远进不了 trigram 索引。
MIN_FTS_CHARS = 3
#: 单条查询最多展开多少个词面。够长的中文问句也就 20 个上下，48 是宽松上界，
#: 只为兜住异常长的输入；真被截断时丢的是句尾的词面。
MAX_TERMS = 48


def _expand_terms(q: str) -> list[str]:
    """检索词展开。产出的词面同时供 FTS MATCH 与 LIKE 兜底使用。

    做三件事，每一件都对应一类实际打废过查询的输入：

    1. **先剥标点再切**。原来只 `replace('"')`，于是 `What is XL-MIMO?` 切出
       `XL-MIMO?`——而 trigram 下的短语查询等价于**字面子串匹配**，那个问号成了
       必须命中的字符，整个词当场作废；剩下的 `What` 长度 ≥3 反而作为有效词面参与 OR，
       把 Zero-shot Reading Comprehension 这类无关论文顶到前排（实测形态）。
    2. **CJK 长片段切三元窗口，不是二元**。原来切二元，而二元词面既进不了 MATCH
       （见 MIN_FTS_CHARS），在 LIKE 里又太宽——「有哪」「的论」命中一大片。
       三元两头都更合适。
    3. **首尾的 . - / 去掉**，`estimation.` 归一到 `estimation`。

    不额外过滤中文虚词：均匀出现的词面 IDF 低，bm25（FTS5 的 `ORDER BY rank`）
    会自己把它们压下去，多一张停用词表只是多一处要维护的口径。
    返回值保序去重，结果稳定可复现。
    """
    out: list[str] = []
    for raw in _PUNCT.sub(" ", q or "").split():
        t = raw.strip(_EDGE)
        if len(t) < 2:
            continue
        out.append(t)
        for run in _CJK_RUN.findall(t):
            if len(run) > MIN_FTS_CHARS:      # 等长时上面那条已经收了
                out.extend(run[i:i + MIN_FTS_CHARS]
                           for i in range(len(run) - MIN_FTS_CHARS + 1))
            elif len(run) == MIN_FTS_CHARS and run != t:
                out.append(run)
        if _CJK_RE.search(t):                 # 中英混合词：ASCII 段也单独放出来
            # **切出来的段要再剥一次边缘符**：_ASCII_RUN 的字符类含 . - /，
            # 于是 'XL-MIMO-的近场估计' 切出 'XL-MIMO-'（带尾连字符），
            # trigram 短语下要求字面命中那个连字符 → 恒 0 行。
            # 这正是本函数 docstring 里以 'XL-MIMO?' 为例声称修掉的失效模式，
            # 在中英混合词这个入口原样复现过（真库：带横杠 0 行 vs 不带 10 行）。
            for r in _ASCII_RUN.findall(t):
                r = r.strip(_EDGE)
                if len(r) >= 2:
                    out.append(r)
    return list(dict.fromkeys(out))[:MAX_TERMS]


def search_fts(c: sqlite3.Connection, q: str, limit: int = 50):
    """trigram 分词的容错检索：短语引号包裹 + OR 语义，任一词命中即返回。

    教训：裸传用户查询在 trigram 下很脆——'XL-MIMO' 的连字符、'信道' 这种
    <3 字词、中文整句无分词，都会把查询打成 0 行。先试 FTS（相关性排序），
    0 行或语法错误再降级 LIKE（展开词任一命中）。
    """
    expanded = _expand_terms(q)
    terms = [t for t in expanded if len(t) >= MIN_FTS_CHARS]
    if terms:
        try:
            match = " OR ".join(f'"{t}"' for t in terms)
            rows = c.execute(
                """SELECT p.* FROM papers_fts f JOIN papers p ON p.id=f.rowid
                   WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?""",
                (match, limit),
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
    # LIKE 兜底：**按命中的词面数排序，不是按入库顺序**。
    # 原来是 `ORDER BY id DESC`——把「最新入库」当成「最相关」，而这个结果
    # 会以权重 1.0 进 search_hybrid 的 RRF（最高的一路）。实测形态：中文整句提问
    # 在 FTS 上 0 行、退到这里，返回的是库里最新的 N 篇（用户自己的笔记和组会记录），
    # 一篇相关论文都没有，却成了融合分数里最强的一路。
    # 标题命中记 2 分、摘要 1 分：正文/摘要里出现一个词，远不如标题里出现同一个词说明问题。
    like_terms = expanded or [q or ""]
    score = " + ".join(
        "(CASE WHEN title LIKE ? THEN 2 ELSE 0 END)"
        " + (CASE WHEN abstract LIKE ? THEN 1 ELSE 0 END)" for _ in like_terms)
    where = " OR ".join("(title LIKE ? OR abstract LIKE ?)" for _ in like_terms)
    pats = [f"%{t}%" for t in like_terms for _ in range(2)]
    # 并列时按 id 决胜，排序键保持全序（否则同一条问句跨进程会给出不同结果）
    return c.execute(
        f"SELECT p.*, ({score}) AS match_score FROM papers p WHERE {where} "
        f"ORDER BY match_score DESC, p.id DESC LIMIT ?",
        pats + pats + [limit],
    ).fetchall()


def reindex_pages(c: sqlite3.Connection, paper_id: int):
    """把某篇论文的 pages 全文同步进 pages_fts（先删后插，重复精读不会留旧影子）。"""
    c.execute("DELETE FROM pages_fts WHERE paper_id=?", (paper_id,))
    c.execute("""INSERT INTO pages_fts(paper_id,page_no,text)
                 SELECT paper_id, page_no, COALESCE(text,'') FROM pages WHERE paper_id=?""",
              (paper_id,))


def search_pages_fts(c: sqlite3.Connection, q: str, limit: int = 50) -> list:
    """页级全文检索。返回按相关性排序的 (paper_id, page_no, text)。

    与 search_fts 同一套容错口径：trigram 下裸查询很脆，先展开词面再 OR；
    语法错误或 0 行则如实返回空——这里不做 LIKE 兜底，全文表太大，
    全表 LIKE 扫描的代价远超它能捞回来的那点召回。
    """
    terms = [t for t in _expand_terms(q) if len(t) >= MIN_FTS_CHARS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    try:
        return c.execute(
            """SELECT paper_id, page_no, text FROM pages_fts
               WHERE pages_fts MATCH ? ORDER BY rank LIMIT ?""",
            (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return []


def search_pages_ranked(c: sqlite3.Connection, q: str, limit: int = 50) -> list[int]:
    """页级命中聚合到论文级：一篇论文取它最好的那一页的名次，去重后返回 paper_id 序。"""
    seen: dict[int, int] = {}
    for rank, row in enumerate(search_pages_fts(c, q, limit * 4)):
        seen.setdefault(row["paper_id"], rank)
    return sorted(seen, key=lambda p: seen[p])[:limit]


def replace_chunks(c: sqlite3.Connection, paper_id: int, chunks: list[dict]):
    """整篇替换检索单元（先删后插，重切不留旧影子）。

    chunks 每项：{text, section_path?, level?, start_page?, end_page?, kind?}。
    空文本的块直接丢弃——它们只会污染 FTS 索引。

    **同时作废该篇的 chunk 向量**：向量按 idx=chunk_no 定位，重切之后
    chunk_no 的含义全变了——idx=1 的向量对应的是旧文本，而 chunks[1] 已经是
    另一段内容；块数变少时多出来的向量更是指向已不存在的块。留着它们会让检索
    **用旧向量匹配、回取新文本**，向量与文本对不上且不会报错。
    删掉即可，`chunkembed.pending()` 下次会把它们当作待嵌入重新补上。
    """
    c.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
    c.execute("DELETE FROM chunks_fts WHERE paper_id=?", (paper_id,))
    c.execute("DELETE FROM vectors WHERE paper_id=? AND kind='chunk'", (paper_id,))
    n = 0
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        n += 1
        c.execute("""INSERT INTO chunks(paper_id,chunk_no,section_path,level,
                       start_page,end_page,kind,text) VALUES(?,?,?,?,?,?,?,?)""",
                  (paper_id, n, ch.get("section_path") or "", int(ch.get("level") or 1),
                   ch.get("start_page"), ch.get("end_page"),
                   ch.get("kind") or "text", text))
        c.execute("INSERT INTO chunks_fts(paper_id,chunk_no,text) VALUES(?,?,?)",
                  (paper_id, n, text))
    return n


def chunks_from_pages(c: sqlite3.Connection, paper_id: int) -> list[dict]:
    """没有 PDF 可切时的兜底：pages 一行一块（QASPER 这类按节导入的本来就是节）。"""
    return [{"text": r["text"] or "", "section_path": "", "level": 1,
             "start_page": r["page_no"], "end_page": r["page_no"], "kind": "text"}
            for r in c.execute(
                "SELECT page_no,text FROM pages WHERE paper_id=? ORDER BY page_no",
                (paper_id,)).fetchall()]


def search_chunks_fts(c: sqlite3.Connection, q: str, limit: int = 50) -> list:
    """检索单元级全文检索。与 search_fts 同一套容错口径；不做 LIKE 兜底——
    全文表太大，全表扫描的代价远超它能捞回的那点召回。"""
    terms = [t for t in _expand_terms(q) if len(t) >= MIN_FTS_CHARS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    try:
        return c.execute(
            """SELECT f.paper_id, f.chunk_no, f.text,
                      (SELECT section_path FROM chunks
                        WHERE paper_id=f.paper_id AND chunk_no=f.chunk_no) section_path,
                      (SELECT start_page FROM chunks
                        WHERE paper_id=f.paper_id AND chunk_no=f.chunk_no) start_page
               FROM chunks_fts f
               WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
            (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return []


def search_chunks_ranked(c: sqlite3.Connection, q: str, limit: int = 50) -> list[int]:
    """块级命中聚合到论文级：一篇取它最好的那个块的名次，去重后返回 paper_id 序。"""
    seen: dict[int, int] = {}
    for rank, row in enumerate(search_chunks_fts(c, q, limit * 4)):
        seen.setdefault(row["paper_id"], rank)
    return sorted(seen, key=lambda p: seen[p])[:limit]


def search_chunks_hits(c: sqlite3.Connection, q: str, limit: int = 50,
                       per_paper: int = 4) -> dict[int, list[dict]]:
    """块级命中，**保留是哪一块**：{paper_id: [{chunk_no, section_path, start_page, text}]}。

    `search_chunks_ranked` 把命中压成 paper_id 就丢掉了块的身份，于是上下文组装侧
    只能退回 `pages` 表按词频重新猜一遍——项目最硬的那个结论（等预算下按章节切的
    证据召回是按页切的 2.2~3.2 倍）量的恰恰是**进上下文的证据**，压成 paper_id
    之后那份收益一分都到不了模型眼前。这个函数就是把身份带出来。

    每篇最多带 `per_paper` 块（按相关性），避免一篇长论文占满整个上下文预算。
    """
    out: dict[int, list[dict]] = {}
    for rank, row in enumerate(search_chunks_fts(c, q, limit * 4)):
        pid = row["paper_id"]
        bucket = out.setdefault(pid, [])
        if len(bucket) < per_paper:
            bucket.append({"chunk_no": row["chunk_no"],
                           "section_path": row["section_path"] or "",
                           "start_page": row["start_page"],
                           "text": row["text"] or "",
                           "rank": rank})
    return out


def chunks_by_no(c: sqlite3.Connection, paper_id: int,
                 chunk_nos: list[int]) -> list[dict]:
    """按 chunk_no 取回指定块（向量路命中的块要靠它拿正文）。保持传入顺序。"""
    if not chunk_nos:
        return []
    marks = ",".join("?" * len(chunk_nos))
    rows = c.execute(
        f"""SELECT chunk_no, section_path, start_page, text FROM chunks
            WHERE paper_id=? AND chunk_no IN ({marks})""",
        [paper_id, *chunk_nos]).fetchall()
    by_no = {r["chunk_no"]: dict(r) for r in rows}
    return [by_no[n] for n in chunk_nos if n in by_no]


def add_llm_call(c: sqlite3.Connection, purpose: str, paper_id: int | None,
                 ptok: int, ctok: int, model: str,
                 latency_ms: float | None = None, cost_usd: float | None = None):
    c.execute("""INSERT INTO llm_calls(purpose,paper_id,prompt_tokens,completion_tokens,
                 latency_ms,cost_usd,model) VALUES(?,?,?,?,?,?,?)""",
              (purpose, paper_id, ptok, ctok, latency_ms, cost_usd, model))


def new_search_run(c: sqlite3.Connection, query: str, source: str) -> int:
    cur = c.execute("INSERT INTO search_runs(query,source) VALUES(?,?)", (query, source))
    return cur.lastrowid


def norm_query(query: str) -> str:
    return " ".join(query.lower().split())


def get_query_cache(c: sqlite3.Connection, query: str):
    return c.execute("SELECT * FROM query_cache WHERE query=?",
                     (norm_query(query),)).fetchone()


def put_query_cache(c: sqlite3.Connection, query: str, source: str, keys: list[str]):
    c.execute("""INSERT INTO query_cache(query,source,keys_json)
                 VALUES(?,?,?)
                 ON CONFLICT(query) DO UPDATE SET
                   source=excluded.source, keys_json=excluded.keys_json,
                   ts=datetime('now','localtime')""",
              (norm_query(query), source, json.dumps(keys)))


def finish_search_run(c: sqlite3.Connection, run_id: int, results: int,
                      new_papers: int, llm_calls: int):
    c.execute("UPDATE search_runs SET results=?, new_papers=?, llm_calls=? WHERE id=?",
              (results, new_papers, llm_calls, run_id))


def stats(c: sqlite3.Connection) -> dict:
    papers_total = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
    by_level = {r["level"]: r["n"] for r in
                c.execute("SELECT level, COUNT(*) n FROM papers GROUP BY level")}
    llm = c.execute("""SELECT COUNT(*) calls, COALESCE(SUM(prompt_tokens),0) ptok,
                       COALESCE(SUM(completion_tokens),0) ctok,
                       COALESCE(SUM(cost_usd),0) cost_usd,
                       AVG(latency_ms) FILTER (WHERE latency_ms IS NOT NULL) avg_latency_ms
                       FROM llm_calls""").fetchone()
    runs = c.execute("SELECT * FROM search_runs ORDER BY id DESC LIMIT 10").fetchall()
    mock_cards = c.execute(
        "SELECT COUNT(*) n FROM papers WHERE card_model='mock-extractive'").fetchone()["n"]
    return {
        "papers_total": papers_total,
        "level_counts": {"L0": by_level.get(0, 0), "L1": by_level.get(1, 0),
                         "L2": by_level.get(2, 0)},
        "mock_cards": mock_cards,
        "llm_calls": llm["calls"], "prompt_tokens": llm["ptok"],
        "completion_tokens": llm["ctok"],
        "cost_usd": round(llm["cost_usd"], 6),
        "avg_latency_ms": round(llm["avg_latency_ms"]) if llm["avg_latency_ms"] else None,
        "recent_search_runs": [dict(r) for r in runs],
    }


# ── Agent 执行轨迹持久化 ──

def save_agent_run(run_id: str, goal: str, status: str, plan: list, events: list,
                   answer: str | None, error: str | None, total_ms: int):
    with conn() as c:
        c.execute("""INSERT OR REPLACE INTO agent_runs
                     (run_id,goal,status,plan_json,events_json,answer,error,total_ms)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (run_id, goal, status, json.dumps(plan, ensure_ascii=False),
                   json.dumps(events, ensure_ascii=False), answer, error, total_ms))
        c.commit()


def list_agent_runs(limit: int = 20) -> list[dict]:
    with conn() as c:
        rows = c.execute("""SELECT run_id, goal, status, total_ms, created_at
                            FROM agent_runs ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_agent_run(run_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["plan"] = json.loads(d.pop("plan_json") or "[]")
    d["events"] = json.loads(d.pop("events_json") or "[]")
    return d
