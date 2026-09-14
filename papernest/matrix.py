"""结构化文献对比矩阵：列由用户自定义，每个单元格都带来源与机械回取校验结论。

与 tablegen.py 的区别：tablegen 是「固定四维 + 交给模型排版」的简版，整张表是模型
一次生成的自由文本；本模块把表拆到**单元格**粒度：

  · 每个 cell 都记 value / source（card:字段 | page:页码 | abstract | llm）/ quote / verified；
  · 离线路径 0 token、完全确定性（卡片字段直取 → 全文关键词定位原句）；
  · 在线路径每篇论文**一次**模型调用抽全部列，模型给的 quote 逐条做机械回取校验，
    验不上的保留内容但标 verified=false，绝不静默丢弃、也绝不拿模型的话当原文；
  · 校验是**按页**做的（口径同 fulltext._verify_claim），不是拿全文合集糊一遍：
    依据句真在第 1 页、模型标 p.9 的，页码会被更正成 p.1 并写进 note；
    整篇都回取不到的，连页码一起丢弃——「验不过的句子 + 它自称的页码」不是出处；
  · 抽不到就留空，并在 cell.note 写明「卡片与全文均未覆盖」——空格是结论，不是遗漏。

coverage / verified_rate 是这张表的自证指标：用户一眼能看出「这表有多少格子真有依据」。
verified_rate 的分母只含**需要**回取校验的格子（模型或卡片写的句子）；离线按关键词从
原文逐字摘录的格子 verified=None、单独计进 cells_excerpted——它们的出处由「切片就是原文」
构造性保证，拿它去做回取是恒真，混进比例里会把这个指标变成空转读数。
"""
import csv
import io
import json
import re
import threading

from . import config, db, llm

# ── 列定义 ──

DEFAULT_COLUMNS: list[dict] = [
    {"key": "method", "label": "方法",
     "hint": "这篇论文提出或使用的核心方法、模型结构、算法"},
    {"key": "dataset", "label": "数据集",
     "hint": "实验所用的数据集 / 基准 / 语料，含规模与来源"},
    {"key": "metric", "label": "评测指标",
     "hint": "用什么指标衡量效果（准确率 / F1 / BLEU / 时延 等）"},
    {"key": "results", "label": "关键结果",
     "hint": "主要实验结论，保留原文的具体数字与对比基线"},
    {"key": "limitations", "label": "局限",
     "hint": "作者自陈的不足、适用边界、未来工作"},
    {"key": "relation", "label": "与课题关系",
     "hint": "这篇文献对我的课题具体能用在哪一环"},
]

# 列 key → 卡片字段候选（按优先级）。卡片字段可能缺失，一律 .get() 容忍。
CARD_FIELDS: dict[str, tuple[str, ...]] = {
    "method": ("method", "method_detail", "methods", "approach"),
    "dataset": ("dataset", "datasets", "data"),
    "metric": ("metric", "metrics", "evaluation"),
    "results": ("results", "result", "key_results", "key_findings"),
    "limitations": ("limitations", "limitation"),
    "relation": ("relation_to_topic", "positioning", "relation"),
    "problem": ("problem", "motivation"),
    "tldr": ("tldr", "summary"),
    "keywords": ("keywords",),
    "findings": ("key_findings",),
}

# 卡片没覆盖时，去 pages 全文里按这些词定位原句。顺序即优先级（全序，跨进程稳定）。
COLUMN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "method": ("we propose", "our method", "our approach", "本文提出", "所提方法", "方法"),
    "dataset": ("dataset", "benchmark", "corpus", "training data", "数据集", "基准", "语料"),
    "metric": ("accuracy", "f1", "bleu", "rouge", "precision", "recall", "metric",
               "评测指标", "准确率", "指标"),
    "results": ("outperform", "achieves", "improves", "state-of-the-art",
                "实验结果", "结果表明", "提升"),
    "limitations": ("limitation", "future work", "we do not", "局限", "不足", "未来工作"),
    "relation": (),  # 「与课题关系」本质是判断，不是原文里能检索到的词——离线只走卡片
    "problem": ("problem", "challenge", "motivation", "研究问题", "挑战", "动机"),
}

# ── 幂等建表：只多一张「列预设」表，不动 db.py 的 MIGRATIONS ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matrix_presets (
  name TEXT PRIMARY KEY,
  columns_json TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now','localtime')),
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
"""

_schema_done: set[str] = set()
_schema_lock = threading.Lock()


def ensure_schema(force: bool = False):
    """幂等建表。按 DB 路径记忆——每个公开函数入口都调它，不加守卫就等于每次
    多开一条连接跑一遍 executescript（db.py 已把这个反模式当踩过的坑记下来了）。"""
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


# ── 文本工具：归一化 / 回取校验 / 断句 ──

def _norm(s: str) -> str:
    """回取校验的口径与 cite._norm 一致：去掉所有空白再小写。

    PDF 抽出来的文本里换行、软连字、双空格到处都是，不压掉空白的话
    任何一句真实原文都对不上。
    """
    return re.sub(r"\s+", "", s or "").lower()


# quote 声称是「逐字原句」，比 fulltext._verify_claim 的 claim（允许改写数字外的表述）
# 口径更严：整串命中，或 2/3 以上探针命中，才算验上。
_PROBE_LEN = 12
_MIN_QUOTE = 8


def verify_quote(quote: str, haystack_norm: str) -> bool:
    """机械回取：quote 归一化后必须真的出现在（归一化后的）原文里。

    haystack_norm 是**单独一段**已归一化的原文（某一页的正文，或摘要），
    由 _context() 预先算好——调用方不要把整篇拼起来传进来（见 _context 的说明）。

    只看前 240 个字符的探针：超长 quote 的尾巴不参与判定，
    所以「真前缀 + 编造后缀」理论上仍能过——这是与 fulltext 一致的已知上界。
    """
    q = _norm(quote)
    if not q or not haystack_norm:
        return False
    if q in haystack_norm:
        return True
    if len(q) < _MIN_QUOTE:
        return False  # 太短，除了整串比对没有可靠判据
    step = max(len(q) // 8, _PROBE_LEN)
    probes = [q[i:i + _PROBE_LEN] for i in range(0, min(len(q), 240), step)]
    probes = [p for p in probes if len(p) >= _MIN_QUOTE]
    if not probes:
        return False
    hit = sum(1 for p in probes if p in haystack_norm)
    return hit * 3 >= len(probes) * 2


_SENT_SPLIT = re.compile(r"(?<=[。！？；?;])|(?<=[.!?])\s+|\n+")
_MAX_SENT = 400


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


_KW_RE_CACHE: dict[str, "re.Pattern"] = {}


def _kw_re(kw: str) -> "re.Pattern":
    """关键词匹配器：只挡**左**边界，不挡右边界。

    裸子串（`k in text.lower()`）会让 'f1' 命中引文占位符 'BIBREF1' / 'TABREF1' /
    'FIGREF1'、'accuracy' 命中 bibkey 'flickinger2011accuracy'、
    'limitation' 命中 'delimitation'。真库 45 篇有全文的论文里 38 篇含 BIBREF 占位符，
    而占位符几乎总在第 1 页 Introduction，`_locate` 又是最先命中即返回——
    于是坏命中系统性地压在 p.1 上（实测 18/172 个离线格子，「评测指标」列 17/42）。

    右边界不能挡：真库 39 个 dataset 格子里 38 个命中的是 'datasets'，
    outperform→outperforms、limitation→limitations 同理，挡了会把真实命中一起打没。
    规则读作「前缀被更长的 token 吞掉 = 噪声；后缀生长 = 词形变化，放行」。
    中日韩关键词不含 [0-9a-z]，这条左边界断言对它们自动放行。
    """
    k = kw.lower()
    pat = _KW_RE_CACHE.get(k)
    if pat is None:
        pat = _KW_RE_CACHE[k] = re.compile(r"(?<![0-9a-z])" + re.escape(k))
    return pat


def _clip(sent: str, kw: str, at: int = -1) -> str:
    """长句（PDF 里常见一整段没有句号）截成关键词附近的窗口。

    只做「取子串」不做改写——截出来的仍是原文逐字片段。
    `at` 是调用方已经算出的命中下标：不传就退回裸 find，但那会把窗口对准句子里第一个
    **词面**巧合（比如 BIBREF1 里的 f1），而不是 `_kw_re` 认可的那一次真命中。
    """
    if len(sent) <= _MAX_SENT:
        return sent
    i = at if at >= 0 else sent.lower().find(kw.lower())
    if i < 0:
        return sent[:_MAX_SENT]
    start = max(0, i - _MAX_SENT // 3)
    return sent[start:start + _MAX_SENT]


# ── 占位串识别：mock 卡片写死的「（mock）」计进覆盖率就是自欺 ──

_PLACEHOLDER_EXACT = {
    "", "-", "--", "—", "–", "n/a", "na", "n.a.", "none", "null", "tbd", "nil",
    "无", "未知", "未提及", "摘要未提及", "不适用", "暂无", "mock", "待补充",
}


def _is_placeholder(s: str) -> bool:
    t = re.sub(r"[\s（）()【】\[\]。.,，]+", "", s or "").lower()
    if t in _PLACEHOLDER_EXACT:
        return True
    if "摘要未提及" in s or "无摘要，仅元数据" in s:
        return True
    return s.strip().startswith(("（mock", "(mock"))


def _page_of(d: dict) -> int | None:
    """页码必须是正整数。3.7 这种非整数不四舍五入成 3——页码现在是会被当成
    出处印进导出表的东西，把一个模型胡诌的小数悄悄截断成「第 3 页」比留空更糟。"""
    raw = d.get("page")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, float) and not raw.is_integer():
        return None
    try:
        p = int(raw)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def _stringify(raw) -> tuple[str | None, int | None]:
    """卡片字段 → (文本, 页码)。字段可能是 str / list / dict / None，一律容忍。"""
    if raw is None or isinstance(raw, bool):
        return None, None
    if isinstance(raw, str):
        return (None, None) if _is_placeholder(raw) else (raw.strip(), None)
    if isinstance(raw, (int, float)):
        return str(raw), None
    if isinstance(raw, dict):
        for k in ("value", "claim", "text", "content"):
            v = raw.get(k)
            if isinstance(v, str) and not _is_placeholder(v):
                return v.strip(), _page_of(raw)
        return None, None
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        pages: list[int | None] = []
        for item in raw:
            s, p = _stringify(item)
            if s:
                parts.append(s)
                pages.append(p)
        # 页码只在**全部**条目都指向同一页时才保留。真实的 L2 卡片长这样：
        # key_findings=[{claim,page:3},{claim,page:7}]——拼成一格后拿第一条的
        # p.3 当整格出处，等于给第二条结论安了个假页码。
        uniq = {p for p in pages if p is not None}
        page = uniq.pop() if len(uniq) == 1 and len(pages) == len(
            [p for p in pages if p is not None]) else None
        return ("；".join(parts) or None), page
    return None, None


# ── 单元格 ──

EMPTY_NOTE = "卡片与全文均未覆盖"


def _cell(value=None, source=None, page=None, quote=None,
          verified=False, note=None) -> dict:
    """`verified` 是**三态**：True=回取通过 / False=回取失败 / None=这条路径上无从断言。

    None 只用在离线 `_locate` 路径：那里的 quote 就是原文切片，回取自己恒真
    （见 `_offline_cell`），报一个恒真的布尔值会把 `verified_rate` 变成 mode 的函数。
    """
    return {"value": value, "source": source, "page": page, "quote": quote,
            "verified": None if verified is None else bool(verified), "note": note}


def empty_cell(note: str = EMPTY_NOTE) -> dict:
    """抽不到就留空——不用「N/A」以外的任何编造内容填充。"""
    return _cell(note=note)


# ── 列 / 论文 ID 校验 ──

def normalize_columns(columns=None) -> list[dict]:
    """列定义归一化。接受 ["method", ...] 或 [{"key","label","hint","keywords"}, ...]。"""
    if columns is None:
        return [dict(c) for c in DEFAULT_COLUMNS]
    if isinstance(columns, (str, dict)):
        raise ValueError("columns 必须是列表")
    out: list[dict] = []
    seen: set[str] = set()
    for col in columns:
        if isinstance(col, str):
            col = {"key": col}
        if not isinstance(col, dict):
            raise ValueError(f"列定义必须是字符串或对象，收到 {type(col).__name__}")
        key = str(col.get("key") or "").strip()
        if not key:
            raise ValueError("列定义缺少 key")
        if key in seen:
            raise ValueError(f"列 key 重复：{key}")
        seen.add(key)
        kws = col.get("keywords") or []
        if isinstance(kws, str):
            kws = [kws]
        out.append({
            "key": key,
            "label": str(col.get("label") or key).strip() or key,
            "hint": str(col.get("hint") or "").strip(),
            "keywords": [str(k).strip() for k in kws if str(k).strip()],
        })
    if not out:
        raise ValueError("至少需要 1 列")
    return out


def _clean_ids(paper_ids) -> list[int]:
    if paper_ids is None or isinstance(paper_ids, (str, bytes, dict)):
        raise ValueError("paper_ids 必须是论文 id 的列表")
    if isinstance(paper_ids, (set, frozenset)):
        # 集合没有顺序，行序就跟着哈希走了——本模块承诺「行序严格跟随调用方」，
        # 与其给一张顺序会飘的表，不如当场报错。
        raise ValueError("paper_ids 不能是集合（行序不可预期）：请传列表")
    out: list[int] = []
    seen: set[int] = set()
    for pid in paper_ids:
        if isinstance(pid, bool):  # True 会被 int() 悄悄变成论文 1
            raise ValueError(f"非法的 paper_id：{pid!r}")
        if isinstance(pid, float) and not pid.is_integer():
            raise ValueError(f"非法的 paper_id：{pid!r}（不是整数）")
        try:
            i = int(pid)
        except (TypeError, ValueError):
            raise ValueError(f"非法的 paper_id：{pid!r}") from None
        if i not in seen:
            seen.add(i)
            out.append(i)  # 保留调用方给的顺序：表格行序必须可预期
    if not out:
        raise ValueError("paper_ids 不能为空：请至少选择 1 篇文献")
    return out


def _keywords_for(col: dict) -> list[str]:
    """列 → 全文兜底检索词。dict.fromkeys 去重且保序；不能用 set，遍历顺序会漂。"""
    kws = [k for k in (col.get("keywords") or []) if len(k) >= 2]
    if kws:
        return list(dict.fromkeys(kws))
    if col["key"] in COLUMN_KEYWORDS:
        # 显式写成空元组 = 该列（如「与课题关系」）本质是判断而非原文检索，不做兜底
        return list(COLUMN_KEYWORDS[col["key"]])
    # 自定义列没给关键词时，拿列名自己当检索词——比什么都不找强，且确定性
    return list(dict.fromkeys(w for w in (col.get("label") or "", col["key"])
                              if len(w) >= 2))


def _derived_keywords(col: dict) -> bool:
    """检索词是不是「拿列名硬凑」出来的（既非用户显式给出，也非 COLUMN_KEYWORDS 精选）。

    这种列命中的只是**词面**：自定义列 cost 会把「The cost function is convex.」
    抓回来当答案。句子确实是原文逐字、页码也对，但「它回答了这一列」这件事
    模块从没验证过。所以要在 cell.note 里说清楚，别让词面巧合冒充语义证据。
    """
    return not [k for k in (col.get("keywords") or []) if len(k) >= 2] \
        and col["key"] not in COLUMN_KEYWORDS


# ── 取数 ──

def _load(ids: list[int]) -> tuple[list[dict], list[dict]]:
    """一次性读齐 papers + pages。返回 (papers, errors)。不存在的 id 记进 errors 并跳过。"""
    papers: list[dict] = []
    errors: list[dict] = []
    with db.conn() as c:
        for pid in ids:
            r = c.execute("SELECT id,title,abstract,year,venue,card_json FROM papers "
                          "WHERE id=?", (pid,)).fetchone()
            if not r:
                errors.append({"paper_id": pid, "error": f"论文 {pid} 不存在（已跳过）"})
                continue
            pages = {row["page_no"]: row["text"] or "" for row in c.execute(
                "SELECT page_no,text FROM pages WHERE paper_id=? ORDER BY page_no",
                (pid,)).fetchall()}
            try:
                card = json.loads(r["card_json"] or "{}")
            except ValueError:
                card = {}
            papers.append({
                "paper_id": r["id"], "title": r["title"] or "", "year": r["year"],
                "venue": r["venue"], "abstract": r["abstract"] or "",
                "card": card if isinstance(card, dict) else {},
                "pages": pages,
            })
    return papers, errors


def _context(p: dict) -> dict:
    """一篇论文的校验上下文：逐页归一化文本 + 归一化摘要（按页序遍历，全序）。

    刻意**不**再合成一个「全文大干草堆」：把所有页拼成一串去比对，会让分散在
    不同页的探针加起来凑够阈值，凭空验出一句原文里其实不存在的连续话。
    每一段证据只跟它自己那一页（或摘要）比。
    """
    pages = [(no, _norm(p["pages"][no])) for no in sorted(p["pages"])]
    return {"pages": pages, "by_page": dict(pages), "abstract": _norm(p["abstract"])}


def _resolve_page(quote: str, ctx: dict, claimed: int | None) -> tuple[int | None, str]:
    """回取校验 + **页码归属**。返回 (可采信的页码, 归属结论)。

    这一步是 fulltext._verify_claim 的核心口径：它校验的是「claim 出现在**它声称的
    那一页**」，而不是「出现在这篇论文的某处」。只查全文合集会让模型编造的页码拿到
    绿标——依据句真在第 1 页、模型写 p.9，照样 verified=true，导出表里就是一条
    编造的出处。所以这里逐页校验：
      claimed    —— 命中模型/卡片声称的那一页，页码可信；
      relocated  —— 那一页没有，但另一页有：采用**真实**页码并写明更正；
      abstract   —— 只在摘要里找到：没有页码，如实置 None；
      none       —— 摘要与全文都回取不到：不采信任何页码。

    逐页校验同时堵死了「探针散落在不同页、加起来凑够 2/3」的跨页拼接假阳性。
    """
    if not quote:
        return None, "none"
    if claimed is not None:
        for no, text in ctx["pages"]:
            if no == claimed:
                if verify_quote(quote, text):
                    return claimed, "claimed"
                break
    for no, text in ctx["pages"]:
        if no != claimed and verify_quote(quote, text):
            return no, "relocated"
    if ctx["abstract"] and verify_quote(quote, ctx["abstract"]):
        return None, "abstract"
    return None, "none"


def _card_lookup(card: dict, key: str) -> tuple[str | None, object]:
    """列 key → 卡片字段。先试同名字段（自定义列直接命中卡片自带字段），再试映射表。"""
    if not isinstance(card, dict):
        return None, None
    for field in (key, *CARD_FIELDS.get(key, ())):
        if field in card and card[field] not in (None, ""):
            return field, card[field]
    return None, None


def _locate(pages: dict[int, str], abstract: str,
            keywords: list[str]) -> tuple[str, int | None, str] | None:
    """按 (关键词序 → 页码序 → 句序) 的全序找第一条含关键词的原句。

    全序是硬要求：任何一步靠 set / dict 的自然顺序，同一份数据跨进程就会给出
    两张不同的表——本项目在评测集上刚踩过这个坑。
    """
    ordered = [(no, pages[no] or "") for no in sorted(pages)]
    for kw in keywords:
        pat = _kw_re(kw)
        for page_no, text in ordered:
            if not pat.search(text.lower()):
                continue
            for sent in _split_sentences(text):
                m = pat.search(sent.lower())
                if m:
                    return _clip(sent, kw, m.start()), page_no, f"page:{page_no}"
    for kw in keywords:  # 没有全文时退到摘要：没有页码，如实标 source=abstract
        pat = _kw_re(kw)
        if not pat.search((abstract or "").lower()):
            continue
        for sent in _split_sentences(abstract):
            m = pat.search(sent.lower())
            if m:
                return _clip(sent, kw, m.start()), None, "abstract"
    return None


# ── 离线抽取（0 token，确定性）──

def _offline_cell(col: dict, paper: dict, ctx: dict) -> dict:
    field, raw = _card_lookup(paper["card"], col["key"])
    if field:
        value, page = _stringify(raw)
        if value:
            real, where = _resolve_page(value, ctx, page)
            if where == "none":
                # 验不上就连页码一起丢掉：卡片里的 page 也是模型写的，
                # 一个「验不过的句子 + 它自称的页码」凑不出可信出处。
                return _cell(value=value, source=f"card:{field}", page=None,
                             quote=None, verified=False,
                             note="取自卡片字段，非原文逐字，未通过回取校验")
            note = (f"卡片标注页码 p.{page} 与原文不符，依据句实际在 p.{real}（已按原文更正）"
                    if where == "relocated" and page is not None else None)
            return _cell(value=value, source=f"card:{field}", page=real,
                         quote=value, verified=True, note=note)
    hit = _locate(paper["pages"], paper["abstract"], _keywords_for(col))
    if hit:
        sent, page_no, source = hit
        # 这句是从库内那一页/摘要里**逐字切出来**的：`_norm` 是逐字符映射（删空白 + 小写），
        # 对拼接同态，而 `_locate` 返回的必然是该页文本的连续切片，所以 _norm(sent) 必然是
        # _norm(page) 的连续子串 —— `verify_quote(sent, 同一页)` **构造上恒为 True**
        # （真库 566 页 13876 句 + 478 篇摘要 3616 句穷举，零反例）。
        # 恒真的断言不是校验：这条路径不报 verified（None = 不适用），出处改由
        # source/page 承载，导出层标【逐字摘录】。
        # 「它是否真的回答了这一列」仍然未经验证 —— 那正是这个标记要提示的事。
        note = (f"全文中按列名「{col['key']}」词面命中的原句（仅词面匹配，非语义判定，请人工确认）"
                if _derived_keywords(col) else None)
        return _cell(value=sent, source=source, page=page_no, quote=sent,
                     verified=None, note=note)
    return empty_cell()


def extract_offline(paper_ids, columns=None, progress=None) -> dict:
    """确定性离线抽取（0 token）：卡片字段直取 → 全文/摘要关键词定位原句 → 留空。

    留空的格子写明「卡片与全文均未覆盖」，绝不编造填充。
    """
    _boot()
    cols = normalize_columns(columns)
    ids = _clean_ids(paper_ids)
    papers, errors = _load(ids)
    rows = []
    for i, p in enumerate(papers):
        if progress:
            progress(min(0.05 + 0.9 * i / max(len(papers), 1), 0.95), "offline",
                     f"离线抽取 {i + 1}/{len(papers)}：{p['title'][:40]}")
        ctx = _context(p)
        rows.append({
            "paper_id": p["paper_id"], "title": p["title"], "year": p["year"],
            "venue": p["venue"],
            "cells": {col["key"]: _offline_cell(col, p, ctx) for col in cols},
        })
    return _finish(cols, rows, errors, mode="offline", degraded=(
        "离线确定性抽取：只做卡片字段直取与全文关键词定位原句，不做语义归纳"
        "（配置 LLM 后可走增强路径）"))


# ── LLM 增强抽取（每篇一次调用）──

MATRIX_SYSTEM = """你是学术文献信息抽取助手。给你一篇论文的标题、摘要与（可能有的）分页全文，
以及一组用户自定义的对比列，请为**每一列**抽取信息。

输出 JSON 对象，且只输出 JSON：
{"cells": {"<列key>": {"value": "该列的抽取结果（简洁，保留原文的数字、指标名与专有名词）",
                       "quote": "支撑该结果的原文依据句，必须逐字复制原文，不得改写、翻译或拼接",
                       "page": 依据句所在页码（整数）；来自摘要则填 null}}}

硬性要求：
1. 只能依据给定材料，严禁编造。原文没有写的列，value 与 quote 都填 null。
2. quote 必须能在给定原文中原样搜到——我们会做机械回取校验，改写过的 quote 会被判为未核验。
3. cells 里必须包含每一个列 key，即使值为 null。
4. 不要输出 JSON 以外的任何文字。"""

_PAGE_CHARS = 2500      # 单页送进上下文的字符上限
_FULL_CHARS = 45000     # 全文总上限：一篇一次调用，别把上下文烧穿


def _llm_user(paper: dict, cols: list[dict], topic: str) -> tuple[str, bool]:
    """返回 (user 消息, 全文是否被截断)。截断必须往上报——被截掉的那几页里
    有没有内容，模型不知道，我们也不知道，不能让它算进「原文未覆盖」。"""
    spec = "\n".join(
        f"- {c['key']}（{c['label']}）：{c['hint'] or '（无补充说明）'}" for c in cols)
    parts = [f"我的课题：{topic}", f"标题：{paper['title']}",
             f"年份/发表处：{paper['year'] or '未知'} / {paper['venue'] or '未知'}",
             f"摘要：{paper['abstract'] or '（无摘要）'}"]
    truncated = False
    if paper["pages"]:
        pieces = []
        for no in sorted(paper["pages"]):
            text = paper["pages"][no] or ""
            truncated = truncated or len(text) > _PAGE_CHARS
            pieces.append(f"【第 {no} 页】\n{text[:_PAGE_CHARS]}")
        full = "\n\n".join(pieces)
        truncated = truncated or len(full) > _FULL_CHARS
        parts.append(f"分页全文：\n{full[:_FULL_CHARS]}")
    else:
        parts.append("分页全文：（该文未做 L2 精读，无全文；依据句只能来自摘要，page 填 null）")
    parts.append(f"需要抽取的列：\n{spec}")
    return "\n\n".join(parts), truncated


def _llm_cell(col: dict, data: dict, ctx: dict) -> dict:
    raw = data.get(col["key"]) if isinstance(data, dict) else None
    if isinstance(raw, str):
        raw = {"value": raw}
    if not isinstance(raw, dict):
        return empty_cell("模型未给出该列（如实留空）")
    value, _ = _stringify(raw.get("value"))
    if not value:
        return empty_cell("模型判定原文未覆盖该列")
    quote = raw.get("quote") if isinstance(raw.get("quote"), str) else None
    page = _page_of(raw)
    if not quote or _is_placeholder(quote):
        # 保留内容但明确「无依据」——不静默丢弃，也不假装有据。
        # 页码一并丢掉：没有依据句，「第几页」就是模型凭空写的一个数。
        return _cell(value=value, source="llm", page=None, quote=None, verified=False,
                     note="模型未给出原文依据句，未通过回取校验"
                          + (f"（模型标注的 p.{page} 无据可依，已丢弃）" if page else ""))
    real, where = _resolve_page(quote, ctx, page)
    if where == "none":
        return _cell(value=value, source="llm", page=None, quote=quote, verified=False,
                     note="模型给出的依据句无法在摘要或全文中回取到（内容保留，标为未核验）")
    if where == "relocated":
        return _cell(value=value, source="llm", page=real, quote=quote, verified=True,
                     note=(f"模型标注页码 p.{page} 与原文不符，依据句实际在 p.{real}（已按原文更正）"
                           if page is not None else None))
    if where == "abstract":
        return _cell(value=value, source="llm", page=None, quote=quote, verified=True,
                     note=(f"依据句出自摘要（无页码）；模型标注的 p.{page} 未采信"
                           if page is not None else None))
    return _cell(value=value, source="llm", page=real, quote=quote, verified=True)


def extract_llm(paper_ids, columns=None, progress=None, topic: str | None = None) -> dict:
    """LLM 增强抽取：**每篇论文一次调用**抽出全部列，再对每个 quote 做机械回取校验。

    每 cell 一次调用会把 token 烧穿（N 篇 × M 列），所以列定义整批塞进同一个 prompt。
    单篇失败不影响其它篇：记进 errors，该篇降级为离线抽取（如实标注 fallback）。
    """
    _boot()
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM，无法走增强抽取路径（请用 extract_offline）")
    cols = normalize_columns(columns)
    ids = _clean_ids(paper_ids)
    papers, errors = _load(ids)
    topic = topic or config.RESEARCH_TOPIC
    rows = []
    fallbacks = 0
    truncated_n = 0
    for i, p in enumerate(papers):
        if progress:
            progress(min(0.05 + 0.9 * i / max(len(papers), 1), 0.95), "llm",
                     f"抽取 {i + 1}/{len(papers)}：{p['title'][:40]}")
        ctx = _context(p)
        row = {"paper_id": p["paper_id"], "title": p["title"], "year": p["year"],
               "venue": p["venue"], "cells": {}}
        user, truncated = _llm_user(p, cols, topic)
        if truncated:
            truncated_n += 1
            # 被截掉的页里可能就写着答案：这一行的空格未必是「原文未覆盖」
            row["context_truncated"] = True
        try:
            text = llm.chat(MATRIX_SYSTEM, user,
                            purpose="matrix", paper_id=p["paper_id"],
                            temperature=0.1, model=config.heavy_model())
            data = llm.extract_json(text)
            cells_raw = data.get("cells")
            if not isinstance(cells_raw, dict):
                # 相当一部分模型直接返回 {"method": {...}}，不套 cells 这层。
                # 列 key 对得上就采信，不然整行白白降级、那次调用的 token 也白烧了。
                if isinstance(data, dict) and any(c["key"] in data for c in cols):
                    cells_raw = data
                else:
                    raise llm.LLMError("返回 JSON 缺少 cells 对象")
            row["cells"] = {c["key"]: _llm_cell(c, cells_raw, ctx) for c in cols}
        except Exception as exc:
            fallbacks += 1
            errors.append({"paper_id": p["paper_id"],
                           "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            row["cells"] = {c["key"]: _offline_cell(c, p, ctx) for c in cols}
            row["fallback"] = "offline"  # 该行是离线兜底出来的，不冒充模型抽取结果
        rows.append(row)
    notes = []
    if fallbacks:
        notes.append(f"{fallbacks}/{len(papers)} 篇模型调用失败，已降级为离线确定性抽取（见 errors）")
    if truncated_n:
        notes.append(f"{truncated_n}/{len(papers)} 篇全文超长被截断进上下文"
                     f"（单页 {_PAGE_CHARS} 字符 / 全文 {_FULL_CHARS} 字符上限），"
                     "这些行的空格不一定代表原文没写")
    return _finish(cols, rows, errors, mode="llm",
                   degraded="；".join(notes) or None)


# ── 顶层 ──

def _finish(cols: list[dict], rows: list[dict], errors: list[dict],
            mode: str, degraded: str | None) -> dict:
    total = len(rows) * len(cols)
    cells = [c for r in rows for c in r["cells"].values() if c.get("value")]
    filled = len(cells)
    # 分母只算「回取校验有内容可断言」的格子。离线逐字摘录的 verified 是 None：
    # 把它算进分子分母，这个比例就成了 mode 的函数、而不是数据质量的函数
    # （真库离线全表 176 个非空格子里，verify_quote 真正做过工作的只有 4 个）。
    checkable = [c for c in cells if c.get("verified") is not None]
    verified = sum(1 for c in checkable if c["verified"])
    return {
        "columns": cols,
        "rows": rows,
        "mode": mode,
        "cells_total": total,
        "cells_filled": filled,
        "cells_verified": verified,
        "cells_checkable": len(checkable),           # 需要回取校验的非空格子
        "cells_excerpted": filled - len(checkable),  # 逐字摘录，校验对其不适用
        "coverage": round(filled / total, 4) if total else 0.0,
        # 分母是「**需要**回取校验的非空单元格」：空格代表原文没写，拿它去摊薄通过率
        # 会把「诚实留空」惩罚成「质量差」；而逐字摘录格子的回取恒真，算进去会把通过率
        # 顶成一个与数据质量无关的常数（真库实测 0.983，其中 171/173 是恒真产物）。
        # 没有可校验格子时给 None（显示「—」）而不是 0.0——一张全是逐字摘录的离线表
        # 报 0% 会被读成「全都没通过校验」，那比报恒真的 100% 更糟。
        "verified_rate": round(verified / len(checkable), 4) if checkable else None,
        "degraded": degraded,
        "errors": errors,
    }


def build(paper_ids: list[int], columns=None, use_llm=None, progress=None,
          topic: str | None = None) -> dict:
    """顶层入口。use_llm=None 时自动择路（有 LLM 走增强，没有走离线）。

    返回 {"columns", "rows", "coverage", "verified_rate", "degraded", "errors", ...}。
    空 paper_ids / 非法列定义直接 ValueError；不存在的 paper_id 跳过并记进 errors。
    """
    _boot()
    cols = normalize_columns(columns)
    ids = _clean_ids(paper_ids)
    if use_llm is None:
        use_llm = llm.available()
    if use_llm and not llm.available():
        raise llm.LLMUnavailable("显式要求 LLM 增强抽取，但未配置 LLM")
    if use_llm:
        return extract_llm(ids, cols, progress=progress, topic=topic)
    return extract_offline(ids, cols, progress=progress)


# ── 列预设（自定义列值得存下来复用）──

def save_preset(name: str, columns) -> dict:
    _boot()
    name = (name or "").strip()
    if not name:
        raise ValueError("预设名不能为空")
    cols = normalize_columns(columns)
    with db.conn() as c:
        c.execute("""INSERT INTO matrix_presets(name,columns_json) VALUES(?,?)
                     ON CONFLICT(name) DO UPDATE SET columns_json=excluded.columns_json,
                       updated_at=datetime('now','localtime')""",
                  (name, json.dumps(cols, ensure_ascii=False)))
    return {"name": name, "columns": cols}


def get_preset(name: str) -> list[dict] | None:
    _boot()
    with db.conn() as c:
        r = c.execute("SELECT columns_json FROM matrix_presets WHERE name=?",
                      ((name or "").strip(),)).fetchone()
    return json.loads(r["columns_json"]) if r else None


def list_presets() -> list[dict]:
    _boot()
    with db.conn() as c:
        rows = c.execute("SELECT name, columns_json, updated_at FROM matrix_presets "
                         "ORDER BY name").fetchall()
    return [{"name": r["name"], "columns": json.loads(r["columns_json"]),
             "updated_at": r["updated_at"]} for r in rows]


def delete_preset(name: str) -> bool:
    _boot()
    with db.conn() as c:
        cur = c.execute("DELETE FROM matrix_presets WHERE name=?",
                        ((name or "").strip(),))
        return cur.rowcount > 0


# ── 导出 ──

LEAD_HEADERS = ["ID", "文献", "年份"]
EMPTY_LEGEND = "空格 = 原文未覆盖，非遗漏（PaperNest 不编造缺失内容）"
_TRUNC_NOTE = ("末尾「…」= 该格已截断至 {n} 字以内便于横向对比；"
               "完整内容见 CSV 导出或 /api/matrix 的 JSON")
MARK_LEGEND = ("【p.N】= 依据句所在页；【未核验】= 未通过原文机械回取校验；"
               "【逐字摘录】= 该格是按关键词从该页原文逐字切出的句子——出处由构造保证，"
               "但「它是否回答了这一列」未经校验，请人工确认")


def _pct(x: float) -> str:
    return f"{round(x * 100, 1)}%"


#: 展示型导出（markdown / latex）的单元格字符上限。
#: 卡片字段常常是几百字的整段（真实库里「与我课题的关系」一格实测 600+ 字），
#: 原样铺进表格会让「一眼横向对比」彻底失效——那正是对比矩阵存在的理由。
#: CSV 不截断：那是拿去做分析的，要全文。
MAX_DISPLAY_CHARS = 160
_CLAUSE_END = "。；;.!?！？"


def _shorten(text: str, limit: int) -> tuple[str, bool]:
    """截到 limit 字以内，尽量落在句读边界上。返回 (文本, 是否截过)。

    不需要截断时**原样返回**，不做空白归一化——换行要留给 markdown 转成 <br>，
    在这里顺手压掉就把「不截断」也变成了一次静默改写。
    """
    raw = str(text)
    if limit <= 0 or len(raw) <= limit:
        return raw, False
    text = re.sub(r"\s+", " ", raw).strip()
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    cut = max(head.rfind(ch) for ch in _CLAUSE_END)
    if cut >= limit * 0.5:          # 句读点太靠前就不用它，宁可硬截
        head = head[:cut + 1]
    return head.rstrip() + "…", True


def _cell_text(cell: dict, with_marks: bool = True, max_chars: int = 0) -> str:
    """单元格 → 展示文本。空 cell 一律给空串——空格本身就是「原文未覆盖」的结论。

    max_chars > 0 时截断并在末尾留「…」；标记（【p.N】/【未核验】）永远追加在
    截断之后，否则截断可能把出处一起切掉，那比长格子更糟。
    """
    value = (cell or {}).get("value")
    if not value:
        return ""
    text, _cut = _shorten(value, max_chars) if max_chars else (str(value), False)
    if not with_marks:
        return text
    marks = []
    page = cell.get("page")
    if page:
        marks.append(f"p.{page}")
    v = cell.get("verified")
    if v is None:      # 逐字摘录：既不能不标（读者会当成已核验），也不能标未核验（那是诬告）
        marks.append("逐字摘录")
    elif not v:
        marks.append("未核验")
    return f"{text}【{' '.join(marks)}】" if marks else text


def _footnotes(matrix: dict, with_marks: bool = True) -> list[str]:
    rate = matrix.get("verified_rate")
    checkable = matrix.get("cells_checkable", matrix.get("cells_filled", 0))
    excerpted = matrix.get("cells_excerpted", 0)
    notes = [
        f"覆盖率 {_pct(matrix.get('coverage', 0.0))}"
        f"（非空 {matrix.get('cells_filled', 0)}/{matrix.get('cells_total', 0)} 格）",
        (f"回取校验通过率 {_pct(rate)}"
         f"（通过 {matrix.get('cells_verified', 0)}/{checkable} 个需要回取校验的格子）"
         if rate is not None else "回取校验通过率 —（本表没有需要回取校验的格子）"),
        EMPTY_LEGEND,
    ]
    if excerpted:
        notes.append(f"另有 {excerpted} 格是按关键词从原文逐字摘录的，出处由构造保证，"
                     "不计入回取校验通过率的分子与分母")
    if with_marks:  # with_marks=False 的表里根本没有标记，再解释一遍标记就是噪声
        notes.append(MARK_LEGEND)
    return notes


def _matrix_rows(matrix: dict, with_marks: bool,
                 max_chars: int = 0) -> tuple[list[str], list[list[str]], bool]:
    cols = matrix.get("columns") or []
    header = LEAD_HEADERS + [c.get("label") or c.get("key") for c in cols]
    body, truncated = [], False
    for r in matrix.get("rows") or []:
        cells = r.get("cells") or {}
        row = [str(r.get("paper_id") or ""), r.get("title") or "", str(r.get("year") or "")]
        for c in cols:
            cell = cells.get(c["key"])
            if max_chars and cell and cell.get("value") \
                    and len(re.sub(r"\s+", " ", str(cell["value"])).strip()) > max_chars:
                truncated = True
            row.append(_cell_text(cell, with_marks, max_chars))
        body.append(row)
    return header, body, truncated


_CSV_RISKY_LEAD = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(v: str) -> str:
    """挡 CSV 公式注入。

    这个导出的卖点就是「Excel 双击打开」，而 Excel / LibreOffice 会把 = + - @ 开头
    的字段当公式执行。库里的标题与原文句子完全可能以 - 或 = 开头（更别说从外部
    BibTeX / RIS 导进来的标题），前置一个单引号让它保持文本；csv.reader 读回时
    只是多一个字面字符，不影响解析。
    """
    return "'" + v if v and v[0] in _CSV_RISKY_LEAD else v


def to_csv(matrix: dict, with_marks: bool = True, max_chars: int = 0) -> str:
    """UTF-8 with BOM 的 CSV：Excel 双击打开中文不乱码（没有 BOM 会按 GBK 解成乱码）。

    默认**不截断**：CSV 是拿去做分析的，截断会静默丢数据。
    要短表就显式传 max_chars。返回的字符串首字符是 \\ufeff；csv.reader 读回时先去掉它。
    """
    header, body, _cut = _matrix_rows(matrix, with_marks, max_chars)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([_csv_safe(h) for h in header])
    for row in body:
        w.writerow([_csv_safe(v) for v in row])
    w.writerow([])
    for note in _footnotes(matrix, with_marks):
        w.writerow(["# " + note])
    return "\ufeff" + buf.getvalue()


def _md_escape(s: str) -> str:
    """单元格里的 | 和换行会把 Markdown 表格拆散架——转义掉。"""
    out = (s or "").replace("\\", "\\\\").replace("|", "\\|")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.replace("\n", "<br>")


def to_markdown(matrix: dict, with_marks: bool = True,
                max_chars: int = MAX_DISPLAY_CHARS) -> str:
    """展示型表格，默认截断长格子（传 max_chars=0 关闭）。"""
    header, body, truncated = _matrix_rows(matrix, with_marks, max_chars)
    lines = ["| " + " | ".join(_md_escape(h) for h in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        lines.append("| " + " | ".join(_md_escape(v) for v in row) + " |")
    lines.append("")
    notes = _footnotes(matrix, with_marks)
    if truncated:
        notes.append(_TRUNC_NOTE.format(n=max_chars))
    lines += [f"> {note}" for note in notes]
    return "\n".join(lines)


_TEX_ESCAPES = {"{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%", "$": r"\$",
                "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}"}


def _tex_escape(s: str) -> str:
    """转义 LaTeX 特殊字符（口径同 cite._bib_escape）。

    反斜杠先挪到哨兵再换回，否则会把刚转义出的反斜杠再转一遍；
    花括号必须排在 ~ / ^ 前面，不然它们替换出来的 {} 会被二次转义。
    换行在 tabular 里直接爆行，统一压成空格。
    """
    out = (s or "").replace("\\", "\x00")
    for ch, rep in _TEX_ESCAPES.items():
        out = out.replace(ch, rep)
    out = out.replace("\x00", r"\textbackslash{}")
    return re.sub(r"\s+", " ", out).strip()


def to_latex(matrix: dict, with_marks: bool = True,
             caption: str = "文献对比矩阵", max_chars: int = MAX_DISPLAY_CHARS) -> str:
    """booktabs 风格 tabular，默认截断长格子。中文内容需要 ctex / xeCJK 编译。"""
    header, body, truncated = _matrix_rows(matrix, with_marks, max_chars)
    n_cols = len(matrix.get("columns") or [])
    width = max(1.6, min(3.2, 12.0 / max(n_cols, 1)))
    spec = "l p{3.0cm} l " + " ".join(f"p{{{width:.1f}cm}}" for _ in range(n_cols))
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
        rf"\caption{{{_tex_escape(caption)}"
        rf"（覆盖率 {_tex_escape(_pct(matrix.get('coverage', 0.0)))}，"
        rf"回取校验通过率 {_tex_escape('—' if matrix.get('verified_rate') is None else _pct(matrix['verified_rate']))}）}}",
        r"\label{tab:papernest-matrix}",
        rf"\begin{{tabular}}{{{spec.strip()}}}",
        r"\toprule",
        " & ".join(_tex_escape(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for row in body:
        lines.append(" & ".join(_tex_escape(v) for v in row) + r" \\")
    notes = _footnotes(matrix, with_marks)
    if truncated:
        notes.append(_TRUNC_NOTE.format(n=max_chars))
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{2pt}",
              r"{\footnotesize 注：" + "；".join(
                  _tex_escape(n) for n in notes) + "。}",
              r"\end{table}"]
    return "\n".join(lines)


EXPORTERS = {"csv": to_csv, "markdown": to_markdown, "latex": to_latex}


def export(matrix: dict, fmt: str, with_marks: bool = True,
           max_chars: int | None = None) -> str:
    """导出。max_chars=None 用各格式的默认（csv 全文、markdown/latex 截断）。"""
    fn = EXPORTERS.get((fmt or "").lower())
    if not fn:
        raise ValueError(f"不支持的导出格式：{fmt}（可选 {sorted(EXPORTERS)}）")
    kw = {} if max_chars is None else {"max_chars": max_chars}
    return fn(matrix, with_marks=with_marks, **kw)
