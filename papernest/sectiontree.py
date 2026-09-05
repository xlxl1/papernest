"""章节树索引与两级检索：**先选范围，再在范围内精检索**。

现在的全库检索是「一把梭」——一个查询直接扫所有论文的所有页。库小的时候没问题，
库一大就两头受损：噪声进上下文（作答模型自己去淘金），成本随库线性涨。

这里换成两级路由：
- **第一级 `select_scope`**：只在「节级词项表」上算分，选出候选论文与候选章节。
  这一级不碰正文，代价只跟命中的词项条目数有关，跟全文体量无关。
- **第二级 `search_in_scope`**：只在第一级圈定的那几节里做词面/位置打分，取片段。

两级检索唯一的真风险是**第一级误杀**（gold 论文在第一级就被筛掉，第二级再准也没用），
所以 `two_stage_search` 的 `trace` 必须把「库里共 N 节 → 第一级留 m 节 → 第二级命中 k 条」
如实报出来，让「省了多少」和「筛掉了什么」都可审计；`tests/test_sectiontree.py`
里专门有一条反例用例钉这个失效。

设计取舍（都写在这儿，省得日后翻代码猜）：
- **0 token、不联网**：全部是 SQL + 正则打分，离线 demo 能全程跑通。
- **标题命中权重显著高于正文命中**（TITLE_W=6.0 vs 正文一次 1.0）：一节的标题里出现
  「信道估计」，比正文里顺带提十次更能说明这一节就在讲它。
- **词面按 IDF 加权**：不加的话「model」「data」「results」这种每节都有的词会和
  「anisotropy」等价，第一级退化成「谁的正文长谁赢」。这一项在两个评测集上同时涨
  （标题自检索 0.925→1.00、QASPER 真实问题 0.561→0.610），不是单集调参调出来的。
- **node_id 用结构化稳定 id**（`P12/S3.2`），不用中文/英文标题当键：标题会变、会重复、
  带特殊字符，拿它当外键迟早出事。
- **排序全序**：所有排序在分数之后一律用 (paper_id, node_id[, page_no, offset]) 决胜，
  本项目踩过「遍历 set 导致同一评测集连跑两次两个 Recall」的坑。
- **索引可重建**：同一篇重复 build 先删后插，不留旧节点。
"""
from __future__ import annotations

import math
import re
import threading

from . import config, db

# ── 建表：只加两张新表，不动 db.py 的 SCHEMA / MIGRATIONS ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS section_nodes (
  id INTEGER PRIMARY KEY,
  paper_id INTEGER NOT NULL,
  node_id TEXT NOT NULL,          -- 结构化稳定 id，如 P12/S3.2
  parent_id TEXT,                 -- 父节点的 node_id，顶层为 NULL
  level INTEGER NOT NULL DEFAULT 1,
  title TEXT NOT NULL DEFAULT '',
  section_path TEXT NOT NULL DEFAULT '',
  start_page INTEGER,
  end_page INTEGER,
  n_chars INTEGER NOT NULL DEFAULT 0,
  UNIQUE(paper_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_section_nodes_paper ON section_nodes(paper_id);
CREATE TABLE IF NOT EXISTS section_terms (
  paper_id INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  term TEXT NOT NULL,
  tf INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(paper_id, node_id, term)
);
-- 第一级是「按 term 反查节」，主键的最左列是 paper_id，用不上。
-- 不建这个索引，选范围就是全表扫，两级检索省下来的钱全赔在这一步。
CREATE INDEX IF NOT EXISTS idx_section_terms_term ON section_terms(term);
"""

_schema_done: set[str] = set()
_schema_lock = threading.Lock()


def ensure_schema(force: bool = False):
    """幂等建表。按 DB 路径记忆——每个公开函数入口都调它，不加守卫等于每次多开一条
    连接跑一遍 executescript（db.py 已把这个反模式当踩过的坑记下来了）。"""
    key = str(config.DB_PATH)
    if not force and key in _schema_done and config.DB_PATH.exists():
        return
    with _schema_lock:
        if not force and key in _schema_done and config.DB_PATH.exists():
            return
        with db.conn() as c:
            c.executescript(_SCHEMA)
        _schema_done.add(key)


def _boot():
    db.init_db()
    ensure_schema()


# ── 切词：索引侧与查询侧**必须**共用这一个函数 ──

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")

#: 英文功能词。不去掉的话 the/of/and 的 tf 会淹没真正的主题词，
#: 而它们在任何查询里都可能出现，等于给所有节加同一个噪声底分。
STOPWORDS = frozenset("""
a an the and or but if of in on at to for from by with without within into over under
between among as is are was were be been being am do does did doing have has had having
this that these those it its it's we our us they their them he she his her you your i my
not no nor so than then there here when where which who whom whose what how why can could
should would may might must will shall also such same other others more most much many few
some any all both each every only just very too both up down out off about after before
during while because since until per via using used use uses based
""".split())

#: 太短的词面没有区分度：英文单字母、单个数字，切出来只会连接一切。
MIN_TERM_LEN = 2


def terms_of(text: str) -> list[str]:
    """把文本切成可比较的词面（含重复，供统计 tf）。

    为什么自己写一份而不复用 `db._expand_terms`：
    ① 它是私有函数，跨模块 import 私有实现是把别人的自由改动变成自己的故障源；
    ② 更要命的是**它是非对称展开**——保留整词、且只有长度 ≥4 的 CJK 词才拆二元组。
       索引侧和查询侧一旦切法不一致就是静默 0 召回：正文里的「近场信道估计」如果
       只按整词存，查询「信道估计」（3~4 字，可能不拆）就永远对不上。
    这里的口径是**对称**的：CJK 连续片段一律只出二元组、不留整词，英文/数字按
    `[a-z0-9]+` 切；两侧走同一个函数，切出来的词面天然可比。

    代价如实写清楚：中文单字词（「熵」「场」）会被丢掉，二元组也会引入一些跨词边界的
    噪声词面（「道估」）。前者是可接受的召回损失，后者靠 tf 权重稀释。
    """
    low = (text or "").lower()
    out: list[str] = []
    for m in _LATIN_RE.finditer(low):
        t = m.group(0)
        if len(t) >= MIN_TERM_LEN and t not in STOPWORDS:
            out.append(t)
    for m in _CJK_RUN_RE.finditer(low):
        run = m.group(0)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def term_freq(text: str) -> dict[str, int]:
    """词面 → 出现次数。插入顺序即首次出现顺序，遍历它不会引入不确定性。"""
    freq: dict[str, int] = {}
    for t in terms_of(text):
        freq[t] = freq.get(t, 0) + 1
    return freq


# ── 节点编号：稳定、可重建、无孤儿 ──

class _TreeNumberer:
    """按**路径名前缀**发号：1, 2, 2.1, 2.2, 3, 3.1 ……

    这里踩过一个只有拿真库才会暴露的坑，值得写清楚。第一版是「按出现顺序 + level
    计数」，在 data/papernest.db 的 QASPER 论文 466 上建出来的树是**整棵错的**：

        页序：Introduction / Related Work ::: Static Word Embeddings /
              Related Work ::: Probing Tasks / Approach ::: Data / ...
        建成：Introduction(1) → Static Word Embeddings(1.1) → Probing Tasks(1.2)
              → Data(1.3) ……

    「Related Work」「Approach」这两个父节自己没有段落、不单独成页，按层级计数就把
    它们的孩子全挂到了「Introduction」底下——而且错得很安静，节点数、level 都对，
    只有父子关系是假的。所以父子关系一律由**路径前缀**决定：

    - 与当前路径栈求最长公共前缀 L（比名字，不比 level）；
    - 深度 ≤ L 的祖先原样保留，深度 L 这一层的序号在原序号上 +1（同一父下的下一个兄弟），
      更深的层从 1 开始；
    - 路径里出现、但库里没有对应正文的父节点会被**补建**成占位节点（有标题、
      n_chars=0、页区间为空）：它能被第一级按标题选中，但第二级不会去它身上取正文
      （正文都在子节点里），不会造成同一页在父子两个节点下重复出片段。

    路径与当前栈完全相同时（连续两页同名）按「新兄弟」处理，否则会撞 node_id 唯一约束。
    """

    def __init__(self):
        self._names: list[str] = []
        self._counters: list[int] = []

    def next(self, parts: list[str]) -> tuple[str, str | None, int, list[tuple[str, int, str]]]:
        """返回 (本节编号, 父编号|None, 树深, 需要补建的祖先 [(编号, 树深, 标题)])。"""
        parts = [p for p in parts if p] or [""]
        k = len(parts)
        common = 0
        while (common < k and common < len(self._names)
               and self._names[common] == parts[common]):
            common += 1
        if common >= k:
            common = k - 1          # 路径与当前栈完全一致：当成同名兄弟，不能复用同一个号

        counters = list(self._counters[:common])
        for i in range(common, k):
            # 只有分叉的那一层能接着数兄弟；更深的层属于另一个父节点，必须从 1 开始
            if i == common and i < len(self._counters):
                counters.append(self._counters[i] + 1)
            else:
                counters.append(1)
        self._counters = counters
        self._names = list(parts)

        created: list[tuple[str, int, str]] = []
        for i in range(common, k - 1):
            created.append((".".join(str(x) for x in counters[:i + 1]), i + 1, parts[i]))
        num = ".".join(str(x) for x in counters)
        parent = ".".join(str(x) for x in counters[:-1]) or None
        return num, parent, k, created


def node_id_of(paper_id: int, number: str) -> str:
    """结构化节点 id：`P<论文 id>/S<层级编号>`。同一篇重建时同样的结构必得同样的 id。"""
    return f"P{int(paper_id)}/S{number}"


# ── 从 pages 表推断章节（QASPER 导入：一页 = 一节，首行是节名）──

#: 节标题的长度上限（**只管没有 ::: 的裸首行**）。再长的首行基本是正文首句，不是标题。
HEADING_MAX_CHARS = 120
#: 带 ::: 的节路径里，单段标题的长度上限。放得比 HEADING_MAX_CHARS 宽：QASPER 里
#: 确实存在把结论整句当子节名的论文，卡死在 120 会静默丢掉整节。
HEADING_MAX_SEGMENT_CHARS = 200
#: QASPER 的子节分隔符：section_name 形如 "Building a Corpus ::: Primary Data"，
#: 真库里 161 页带这个标记，不认它就会把整棵子树拍平成同级。
SUBSECTION_SEP = ":::"

_SENTENCE_END = "。．.!?！？；;，,、"


def _looks_like_heading(line: str) -> bool:
    """首行像不像节标题。

    钉住两个**真库里实际发生**的反例（拿 data/papernest.db 的副本跑出来的，不是想象的）：

    ① PDF 抽出来的页（fulltext 写入的那种）首行常常是页码 "1"、"2"，再往下是被排版
       切碎的论文题名。这种首行既不是节标题，也不该被当标题存进去——宁可整页当一节、
       如实标 degraded，也不要造一棵假的章节树。
    ② 反过来，只按「短 + 不以句号结尾」判会**误杀真标题**：QASPER 的 `:::` 子节名会
       把父节标题整段重复一遍（560 页里有 14 页首行超 120 字符），而 paper 466 的子节名
       干脆是整句话、带反引号和句号（"… ::: Stopwords (e.g., `the', `of') have …
       representations."）。所以带 `:::` 的首行按「QASPER 明确给出的节路径」直接采信，
       只做逐段的长度体检；长度/句末标点这套保守启发式只用在没有 `:::` 的行上。
    """
    s = (line or "").strip()
    if len(s) < 2:
        return False
    if not re.search(r"[A-Za-z一-鿿]", s):
        return False        # 纯数字/纯符号：页码、公式编号
    if SUBSECTION_SEP in s:
        parts = [p.strip() for p in s.split(SUBSECTION_SEP)]
        return all(2 <= len(p) <= HEADING_MAX_SEGMENT_CHARS for p in parts)
    if len(s) > HEADING_MAX_CHARS:
        return False
    return s[-1] not in _SENTENCE_END


def _split_page(text: str) -> tuple[str, str, bool]:
    """页文本 → (标题, 正文, 是否识别到标题)。识别不到标题时正文取整页。"""
    lines = (text or "").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return "", "", False
    head = lines[i].strip()
    if _looks_like_heading(head):
        return head, "\n".join(lines[i + 1:]).strip(), True
    return "", (text or "").strip(), False


def _nodes_from_pages(c, paper_id: int) -> tuple[list[dict], str | None]:
    rows = c.execute("SELECT page_no, text FROM pages WHERE paper_id=? ORDER BY page_no",
                     (paper_id,)).fetchall()
    out: list[dict] = []
    no_head = 0
    for r in rows:
        title, body, ok = _split_page(r["text"])
        if not (title or body):
            continue
        if not ok:
            no_head += 1
        parts = [p.strip() for p in title.split(SUBSECTION_SEP)] if title else []
        parts = [p for p in parts if p]
        out.append({
            "title": parts[-1] if parts else "",
            # 认不出标题的页用页码占位：每页各成一节，不会被误并进上一节
            "path_parts": parts or [f"（第 {r['page_no']} 页）"],
            "start_page": r["page_no"],
            "end_page": r["page_no"],
            "text": body,
        })
    degraded = None
    if out and no_head:
        degraded = (f"{no_head}/{len(out)} 页首行不像节标题，已整页当作一节"
                    f"（这类论文建议传 structure.section_chunks 的结果）")
    return out, degraded


# ── 从 structure.py 的结构建节点 ──

def _norm_chunk(ch: dict, title_stack: list[tuple[int, str]]) -> dict | None:
    """兼容 structure.py 的两种返回形态。

    - `section_chunks()`：{section_title, section_path, level, start_page, end_page, text, ...}
      —— 首选，有正文，`section_path`（"3 Method > 3.2 Training"）直接就是路径；
    - `detect_sections()`：{title, level, page_no, ...} —— 没有正文也没有 path，只能靠
      `title_stack` 按 level 现场拼路径，且**词项只来自标题**，召回会差一截。
      支持它是为了调用方少做一次转换，但首选前者。
    """
    if not isinstance(ch, dict):
        return None
    if "section_title" in ch or "section_path" in ch:
        path = (ch.get("section_path") or "").strip()
        title = (ch.get("section_title") or "").strip()
        parts = [p.strip() for p in path.split(">")] if path else []
        parts = [p for p in parts if p]
        return {
            "title": title or (parts[-1] if parts else ""),
            "path_parts": parts or ([title] if title else []),
            "start_page": ch.get("start_page"),
            "end_page": ch.get("end_page"),
            "text": ch.get("text") or "",
        }
    title = (ch.get("title") or "").strip()
    if not title:
        return None
    level = max(1, int(ch.get("level") or 1))
    while title_stack and title_stack[-1][0] >= level:
        title_stack.pop()
    title_stack.append((level, title))
    page = ch.get("page_no")
    return {"title": title, "path_parts": [t for _, t in title_stack],
            "start_page": page, "end_page": page, "text": ch.get("text") or ""}


def _nodes_from_chunks(sections) -> tuple[list[dict], str | None]:
    """把 chunk 列表并成节点：同一节被 max_chars 二次切成的多个 part 必须合回一个节点，
    否则一节会在索引里占 n 份，第一级的「选出 m 节」就不再是「m 个章节」。"""
    out: list[dict] = []
    degraded = None
    title_stack: list[tuple[int, str]] = []
    for raw in sections or []:
        if isinstance(raw, dict) and raw.get("degraded") and not degraded:
            degraded = str(raw["degraded"])
        n = _norm_chunk(raw, title_stack)
        if not n or not (n["title"] or n["text"] or n["path_parts"]):
            continue
        key = (tuple(n["path_parts"]), n["title"])
        if out and out[-1]["_key"] == key:
            prev = out[-1]
            prev["text"] = (prev["text"] + "\n\n" + n["text"]).strip()
            prev["start_page"] = _min_page(prev["start_page"], n["start_page"])
            prev["end_page"] = _max_page(prev["end_page"], n["end_page"])
            continue
        n["_key"] = key
        out.append(n)
    for n in out:
        n.pop("_key", None)
    return out, degraded


def _min_page(a, b):
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None


def _max_page(a, b):
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


# ── 建索引 ──

def build_index(paper_id: int, sections=None) -> dict:
    """为一篇论文建章节索引（0 token）。

    `sections` 不传就从 `pages` 表推断（QASPER 导入的论文每页就是一节，首行是节名）；
    传了就直接用 structure.py 的 `section_chunks()`（或 `detect_sections()`）结果。

    先删后插：同一篇重复 build 不会留下旧节点，也不会产生重复行。

    返回 {"paper_id", "nodes", "terms", "source": "pages"|"sections", "degraded": str|None}。
    注意 `nodes` 是**实际入库的行数**，会大于输入的节数——路径里出现但自己没有正文的
    父节点会被补建成占位节点（见 `_TreeNumberer` 的说明）。
    """
    _boot()
    with db.conn() as c:
        if sections is None:
            nodes, degraded = _nodes_from_pages(c, int(paper_id))
            source = "pages"
        else:
            nodes, degraded = _nodes_from_chunks(sections)
            source = "sections"

        # 先删后插。删除必须在同一个事务里，否则中途失败会留下「删了一半」的索引。
        c.execute("DELETE FROM section_terms WHERE paper_id=?", (int(paper_id),))
        c.execute("DELETE FROM section_nodes WHERE paper_id=?", (int(paper_id),))

        numberer = _TreeNumberer()
        n_terms = 0
        n_rows = 0

        def _put(nid, parent_id, depth, title, parts, start, end, body):
            nonlocal n_terms, n_rows
            path = " > ".join(parts)
            c.execute(
                """INSERT INTO section_nodes(paper_id,node_id,parent_id,level,title,
                     section_path,start_page,end_page,n_chars)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (int(paper_id), nid, parent_id, depth, title, path, start, end, len(body)))
            n_rows += 1
            # 标题与**祖先**标题也进词项表：标题命中的节必须能被第一级捞到，
            # 只索引正文的话「标题写着信道估计、正文全用缩写」的节会直接从索引里消失。
            # 只取 parts[:-1] 而不是整条 path——path 的最后一段就是 title，
            # 一起喂进去会让标题词的 tf 白白翻倍，打分公式就不再是手算得出来的了。
            freq = term_freq("{}\n{}\n{}".format(" > ".join(parts[:-1]), title, body))
            if freq:
                c.executemany(
                    "INSERT INTO section_terms(paper_id,node_id,term,tf) VALUES(?,?,?,?)",
                    [(int(paper_id), nid, t, tf) for t, tf in freq.items()])
                n_terms += len(freq)

        for n in nodes:
            number, parent_num, depth, created = numberer.next(n["path_parts"])
            for anc_num, anc_depth, anc_title in created:
                # 补建的父节点：start_page/end_page 留 NULL，第二级据此跳过它，
                # 免得同一页在父子两个节点下各出一次片段
                anc_parent = anc_num.rsplit(".", 1)[0] if "." in anc_num else None
                _put(node_id_of(paper_id, anc_num),
                     node_id_of(paper_id, anc_parent) if anc_parent else None,
                     anc_depth, anc_title, n["path_parts"][:anc_depth], None, None, "")
            _put(node_id_of(paper_id, number),
                 node_id_of(paper_id, parent_num) if parent_num else None,
                 depth, n["title"], n["path_parts"],
                 n["start_page"], n["end_page"], n["text"] or "")
    return {"paper_id": int(paper_id), "nodes": n_rows, "terms": n_terms,
            "source": source, "degraded": degraded}


def build_all(progress=None) -> dict:
    """给全库「有全文」的论文建索引。三态计数，失败不中断整批。

    `progress(frac, stage, message)` 供异步任务接进度，同步调用不传即可。
    返回 {"indexed", "skipped", "failed", "errors": [...]}——skipped 是「有全文但
    推不出任何节点」（整页空白之类），跟 failed（抛异常）分开记，不要混成一个数。
    """
    _boot()
    with db.conn() as c:
        ids = [r["paper_id"] for r in c.execute(
            "SELECT DISTINCT paper_id FROM pages ORDER BY paper_id")]
    indexed = skipped = failed = 0
    errors: list[str] = []
    for i, pid in enumerate(ids):
        if progress:
            progress(min(0.05 + 0.9 * i / max(len(ids), 1), 0.95), "index",
                     f"建章节索引 {i + 1}/{len(ids)}")
        try:
            r = build_index(pid)
        except Exception as e:                      # noqa: BLE001 —— 单篇失败不该拖垮整批
            failed += 1
            errors.append(f"paper {pid}: {type(e).__name__}: {e}")
            continue
        if r["nodes"]:
            indexed += 1
        else:
            skipped += 1
    if progress:
        progress(1.0, "index", f"完成：{indexed} 篇已索引")
    return {"indexed": indexed, "skipped": skipped, "failed": failed, "errors": errors}


# ── 打分权重（两级共用，改这里就能复现实验）──

#: 正文命中一个词面的基础分；tf 每翻十倍再 +1.0（log10），防长节靠堆词碾压短节。
BODY_W = 1.0
#: 标题命中一个词面的加分。取 6.0 的理由：正文里同一个词出现 10 次也才 2.0，
#: 出现 100 次才 3.0——标题命中一次就应当稳压「正文提了很多次但标题没提」的节。
TITLE_W = 6.0
#: 祖先路径（父节标题）命中的加分：比自己的标题弱得多，但强于纯正文。
PATH_W = 1.5
#: 第二级里查询词覆盖率的加分：命中 3 个词的片段应当高于把 1 个词重复 3 次的片段。
COVERAGE_W = 2.0
#: 第二级里的位置加分：同分时靠前的片段更可能是本节的论点句。
POSITION_W = 0.3
#: 第二级里片段命中「本节标题词」的加分。比第一级的 TITLE_W 小一个量级——
#: 这一级已经在选定的节里了，标题只该微调片段顺序，不该再决定谁进谁不进。
PASSAGE_TITLE_W = 1.5
#: 第二级里继承的第一级节点分权重。留一点点，让「标题就在讲这个」的节占先，
#: 但不能大到让第二级变成第一级的复读机。
NODE_PRIOR_W = 0.1

NO_INDEX = "库内没有任何章节索引（先跑 build_all），已退化为全库检索"
NO_TERMS = "查询里没有可用词面（长度<2 或全是停用词）"


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def rank_key(node: dict) -> tuple:
    """节的全序排序键：分数降序，并列时 (paper_id, node_id) 决胜。

    单独抽出来是为了它**可被直接测**。写成 `sort(key=lambda ...)` 内联的话，
    Python 的稳定排序会拿输入顺序当隐形决胜键，测试根本分不出「真的有决胜键」
    和「恰好输入就是有序的」——本项目踩过「同一评测集连跑两次两个 Recall」的坑，
    就是这种看不见的顺序依赖。

    注意 node_id 是按**字符串**比的，所以 "P2/S10" 排在 "P2/S2" 前面。不好看，
    但确定：换成自然序要解析编号，多一处会出错的地方，收益只有观感。
    """
    return (-float(node["score"]), int(node["paper_id"]), str(node["node_id"]))


def passage_rank_key(hit: dict) -> tuple:
    """片段的全序排序键：在 rank_key 之后再用 (page_no, offset) 决胜。"""
    return (-float(hit["score"]), int(hit["paper_id"]), str(hit["node_id"]),
            int(hit["page_no"]), int(hit["offset"]))


def idf_of(df: int, n_nodes: int) -> float:
    """词面的区分度权重。df=命中该词面的节数，n_nodes=库内总节数。

    没有这一项，「model」「data」「results」这种每节都有的词会和「anisotropy」等价，
    第一级就退化成「谁的正文长谁赢」。实测（真库 581 节 / 两个评测集）加上 IDF 后
    两个集的 hit 都涨，没有此消彼长——所以它留下了。
    """
    return math.log10(1.0 + max(int(n_nodes), 1) / max(int(df), 1))


def score_node(matched: dict[str, int], title_terms: set[str],
               path_terms: set[str], idf: dict[str, float] | None = None) -> float:
    """节级分数 = Σ IDF ×（正文 tf 分 + 标题命中加权 / 祖先路径命中加权）。

    公开出来是为了让测试能按同一个公式手算复核——打分函数藏在私有里，
    「标题命中权重显著高于正文命中」这条设计就只能靠「跑通就行」来验证了。
    IDF 对标题项和正文项**同倍**作用，所以标题/正文的权重比不受它影响。
    """
    s = 0.0
    for t, tf in matched.items():
        w = (idf or {}).get(t, 1.0)
        part = BODY_W * (1.0 + math.log10(max(int(tf), 1)))
        if t in title_terms:
            part += TITLE_W
        elif t in path_terms:
            part += PATH_W
        s += w * part
    return s


def select_scope(query: str, top_papers: int = 10, top_nodes: int = 30) -> dict:
    """**第一级：选范围**。只查节级词项表，不碰正文。

    返回 {"paper_ids": [...],
          "nodes": [{"paper_id","node_id","section_path","title","level",
                     "start_page","end_page","score","matched_terms"}],
          "query_terms": [...], "total_nodes": n, "degraded": str|None}

    `matched_terms` 让「凭什么选中这一节」可审计。库内没有任何章节索引时返回空并
    标 degraded=NO_INDEX，调用方据此走全库检索。
    """
    _boot()
    q_terms = sorted(set(terms_of(query)))
    empty = {"paper_ids": [], "nodes": [], "query_terms": q_terms, "total_nodes": 0}
    with db.conn() as c:
        total_nodes = c.execute("SELECT COUNT(*) FROM section_nodes").fetchone()[0]
        empty["total_nodes"] = total_nodes
        if not total_nodes:
            return {**empty, "degraded": NO_INDEX}
        if not q_terms:
            return {**empty, "degraded": NO_TERMS}

        hits: dict[tuple[int, str], dict[str, int]] = {}
        for batch in _batched(q_terms, 400):        # SQLite 变量数上限 999，留足余量
            ph = ",".join("?" * len(batch))
            for r in c.execute(
                    f"SELECT paper_id,node_id,term,tf FROM section_terms "
                    f"WHERE term IN ({ph})", batch):
                hits.setdefault((r["paper_id"], r["node_id"]), {})[r["term"]] = r["tf"]
        if not hits:
            # 有索引但一个词面都没命中：这是「确实没有」，不是降级，别误导调用方去全库重扫
            return {**empty, "degraded": None}

        meta: dict[tuple[int, str], dict] = {}
        paper_ids = sorted({pid for pid, _ in hits})
        for batch in _batched(paper_ids, 400):
            ph = ",".join("?" * len(batch))
            for r in c.execute(
                    f"SELECT * FROM section_nodes WHERE paper_id IN ({ph})", batch):
                key = (r["paper_id"], r["node_id"])
                if key in hits:
                    meta[key] = dict(r)

    # df 是白拿的：上面已经把所有含这些词面的行全取回来了，数一遍就是全局文档频率
    df: dict[str, int] = {}
    for matched in hits.values():
        for t in matched:
            df[t] = df.get(t, 0) + 1
    idf = {t: idf_of(n, total_nodes) for t, n in df.items()}

    scored: list[dict] = []
    for key, matched in hits.items():
        m = meta.get(key)
        if not m:
            continue                                # 词项表有、节点表没有：只能是并发重建的残影
        title_terms = set(terms_of(m["title"]))
        path_terms = set(terms_of(m["section_path"])) - title_terms
        scored.append({
            "paper_id": m["paper_id"], "node_id": m["node_id"],
            "section_path": m["section_path"], "title": m["title"],
            "level": m["level"], "start_page": m["start_page"],
            "end_page": m["end_page"], "n_chars": m["n_chars"],
            "score": round(score_node(matched, title_terms, path_terms, idf), 6),
            "matched_terms": sorted(matched),
        })

    # 全序：分数并列时按 (paper_id, node_id) 决胜，跨进程不漂
    scored.sort(key=rank_key)

    best: dict[int, float] = {}
    for n in scored:
        pid = n["paper_id"]
        if n["score"] > best.get(pid, float("-inf")):
            best[pid] = n["score"]
    ranked_papers = sorted(best, key=lambda p: (-best[p], p))[:max(0, int(top_papers))]
    keep = set(ranked_papers)
    nodes = [n for n in scored if n["paper_id"] in keep][:max(0, int(top_nodes))]
    # idf 一并带出去：第二级要用同一套权重，否则两级对「什么词重要」的判断会打架
    return {"paper_ids": ranked_papers, "nodes": nodes, "query_terms": q_terms,
            "idf": idf, "total_nodes": total_nodes, "degraded": None}


# ── 第二级：范围内精检索 ──

#: 片段长度目标值。太短会把论点句和它的证据切开，太长又等于把整节塞回上下文。
PASSAGE_CHARS = 400


def _passages(text: str) -> list[tuple[int, str]]:
    """节内文本 → [(节内字符偏移, 片段)]。先按段落切，再贪心打包到 ~PASSAGE_CHARS。

    偏移一路带着，既是粗定位信息，也是全序排序的最后一道决胜键。注意它指向的是
    **打包起点那一段的段首**，片段文本本身做过 strip，所以偏移用于定位是近似的、
    用于排序是精确的（同一节内严格递增）。
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    buf_len = 0
    pos = 0
    for para in re.split(r"(\n\s*\n|\n)", text or ""):
        if not para:
            continue
        stripped = para.strip()
        if stripped:
            if not buf:
                buf_start = pos + (len(para) - len(para.lstrip()))
            buf.append(stripped)
            buf_len += len(stripped)
            if buf_len >= PASSAGE_CHARS:
                out.append((buf_start, "\n".join(buf)))
                buf, buf_len = [], 0
        pos += len(para)
    if buf:
        out.append((buf_start, "\n".join(buf)))
    return out


def _score_passage(text: str, q_terms: list[str], title_terms: set[str],
                   idx: int, n: int, idf: dict[str, float]) -> tuple[float, list[str]]:
    freq = term_freq(text)
    matched = {t: freq[t] for t in q_terms if t in freq}
    if not matched:
        return 0.0, []
    s = sum(idf.get(t, 1.0) * BODY_W * (1.0 + math.log10(tf))
            for t, tf in matched.items())
    s += COVERAGE_W * len(matched) / max(len(q_terms), 1)
    s += POSITION_W * (1.0 - idx / max(n, 1))
    s += PASSAGE_TITLE_W * sum(idf.get(t, 1.0) for t in matched if t in title_terms)
    return s, sorted(matched)


def search_in_scope(query: str, scope: dict, top_k: int = 5) -> list[dict]:
    """**第二级：范围内精检索**。只在 scope 圈定的节里打分取片段。

    返回 [{"paper_id","node_id","section_path","section_title","level",
           "start_page","page_no","offset","score","matched_terms","text"}]。

    **一节最多出一个片段**（该节最好的那个）：一是防长节刷屏，二是保证
    `trace` 里「stage2_hits ≤ stage1_nodes」这个不变量恒成立，trace 才说得清筛选比例。

    正文从 `pages` 表按节点的页区间回取。若某篇只在 build_index 时传了 sections、
    库里却没有它的 pages 全文，这一级取不到文本会如实跳过该节（不编造片段）。
    """
    _boot()
    nodes = list((scope or {}).get("nodes") or [])
    q_terms = sorted(set(terms_of(query)))
    idf = (scope or {}).get("idf") or {}
    if not nodes or not q_terms:
        return []

    pages: dict[int, list[tuple[int, str]]] = {}
    paper_ids = sorted({n["paper_id"] for n in nodes})
    with db.conn() as c:
        for batch in _batched(paper_ids, 400):
            ph = ",".join("?" * len(batch))
            for r in c.execute(
                    f"SELECT paper_id,page_no,text FROM pages WHERE paper_id IN ({ph}) "
                    f"ORDER BY paper_id,page_no", batch):
                pages.setdefault(r["paper_id"], []).append((r["page_no"], r["text"] or ""))

    hits: list[dict] = []
    for node in nodes:
        lo, hi = node.get("start_page"), node.get("end_page")
        if lo is None:
            # 没有自有页区间：补建的父节点，或结构里本来就没给页码的 chunk。
            # 不能放它进来「不设上下界地扫全篇」——那等于把第一级的筛选成果丢掉。
            continue
        title_terms = set(terms_of(node.get("title") or ""))
        best: dict | None = None
        for page_no, ptext in pages.get(node["paper_id"], []):
            if lo is not None and page_no < lo:
                continue
            if hi is not None and page_no > hi:
                continue
            chunks = _passages(ptext)
            for i, (off, ctext) in enumerate(chunks):
                s, matched = _score_passage(ctext, q_terms, title_terms, i,
                                            len(chunks), idf)
                if s <= 0:
                    continue
                s += NODE_PRIOR_W * float(node.get("score") or 0.0)
                cand = {
                    "paper_id": node["paper_id"], "node_id": node["node_id"],
                    "section_path": node.get("section_path") or "",
                    "section_title": node.get("title") or "",
                    "level": node.get("level"), "start_page": node.get("start_page"),
                    "page_no": page_no, "offset": off, "score": round(s, 6),
                    "matched_terms": matched, "text": ctext,
                }
                if best is None or (-cand["score"], cand["page_no"], cand["offset"]) < \
                        (-best["score"], best["page_no"], best["offset"]):
                    best = cand
        if best:
            hits.append(best)

    hits.sort(key=passage_rank_key)
    return hits[:max(0, int(top_k))]


# ── 顶层：两级检索 ──

def _degraded_full(query: str, top_k: int, note: str) -> dict:
    """降级路径：库里没有章节索引时走 db 的全库页级检索，如实标 mode/degraded。

    离线可跑是硬要求，所以这条路径不依赖任何模型，只用 pages_fts。
    """
    _boot()
    with db.conn() as c:
        rows = db.search_pages_fts(c, query, max(top_k * 4, 20))
    q_terms = sorted(set(terms_of(query)))
    passages: list[dict] = []
    seen: set[int] = set()
    for r in rows:
        if r["paper_id"] in seen:
            continue                                # 一篇只留最好的一页，与两级路径口径一致
        seen.add(r["paper_id"])
        chunks = _passages(r["text"] or "")
        best = None
        for i, (off, ctext) in enumerate(chunks):
            s, matched = _score_passage(ctext, q_terms, set(), i, len(chunks), {})
            if s > 0 and (best is None or s > best["score"]):
                best = {"paper_id": r["paper_id"], "node_id": None, "section_path": "",
                        "section_title": "", "level": None, "start_page": r["page_no"],
                        "page_no": r["page_no"], "offset": off, "score": round(s, 6),
                        "matched_terms": matched, "text": ctext}
        if best:
            passages.append(best)
        if len(passages) >= top_k:
            break
    return {
        "paper_ids": [p["paper_id"] for p in passages],
        "passages": passages, "mode": "degraded_full", "degraded": note,
        "scope_paper_ids": [],
        "trace": {"stage1_papers": 0, "stage1_nodes": 0, "stage2_hits": len(passages),
                  "total_papers": 0, "total_nodes": 0},
    }


def two_stage_search(query: str, top_k: int = 5, top_papers: int = 10) -> dict:
    """先选范围、再范围内精检索。0 token、离线可跑。

    返回 {"paper_ids": [...], "passages": [...], "mode": "two_stage"|"degraded_full",
          "degraded": str|None, "scope_paper_ids": [...],
          "trace": {"stage1_papers","stage1_nodes","stage2_hits","total_papers","total_nodes"}}

    `paper_ids` 是**第二级真正出了片段**的论文（按最好片段排序），`scope_paper_ids`
    才是第一级圈定的更宽范围——调用方想扩召回就用后者。

    trace 的 total_nodes → stage1_nodes → stage2_hits 三个数就是这个功能的价值证据：
    「库里 N 节，第一级只留 m 节，第二级只在这 m 节里读正文」。
    """
    _boot()
    scope = select_scope(query, top_papers=top_papers,
                         top_nodes=max(int(top_k) * 6, 30))
    if scope.get("degraded") == NO_INDEX:
        return _degraded_full(query, top_k, NO_INDEX)

    passages = search_in_scope(query, scope, top_k=top_k)
    order: dict[int, int] = {}
    for i, p in enumerate(passages):
        order.setdefault(p["paper_id"], i)
    paper_ids = sorted(order, key=lambda p: (order[p], p))
    with db.conn() as c:
        total_papers = c.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM section_nodes").fetchone()[0]
    return {
        "paper_ids": paper_ids, "passages": passages, "mode": "two_stage",
        "degraded": scope.get("degraded"), "scope_paper_ids": scope["paper_ids"],
        "trace": {"stage1_papers": len(scope["paper_ids"]),
                  "stage1_nodes": len(scope["nodes"]),
                  "stage2_hits": len(passages),
                  "total_papers": total_papers,
                  "total_nodes": scope.get("total_nodes", 0)},
    }


def stats() -> dict:
    """索引规模。avg_nodes_per_paper 用来一眼看出「章节抽得太碎还是太粗」。"""
    _boot()
    with db.conn() as c:
        papers = c.execute("SELECT COUNT(DISTINCT paper_id) FROM section_nodes").fetchone()[0]
        n_nodes = c.execute("SELECT COUNT(*) FROM section_nodes").fetchone()[0]
        n_terms = c.execute("SELECT COUNT(*) FROM section_terms").fetchone()[0]
        n_distinct = c.execute("SELECT COUNT(DISTINCT term) FROM section_terms").fetchone()[0]
    return {"papers_indexed": papers, "n_nodes": n_nodes, "n_terms": n_terms,
            "n_distinct_terms": n_distinct,
            "avg_nodes_per_paper": round(n_nodes / papers, 2) if papers else 0.0}
