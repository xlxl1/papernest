"""新论文订阅追踪 —— 每天把「新出的、和你研究方向相关的」论文送到面前。

设计要点（为什么这么做）：

1. **画像不调 LLM，打分不花 token**。兴趣画像从库内已有论文反推（card 的 keywords、
   标题短语、venue），打分是确定性的加权词面命中。订阅是每天都要跑的东西，
   如果每天都要烧一遍模型，用户第三天就会把它关掉；而且离线 demo 必须能跑通。
   LLM 只作为**可选的第二层**（`llm_rerank`），失败或未配置一律原样返回并标 degraded。

2. **每条推荐都必须能回答「为什么推给我」**。`score_papers` 给出的 `reasons`
   逐条列出命中了画像里的哪个词、在标题还是摘要命中、各贡献多少分——
   这是本项目「证据可审计」在推荐场景的落地。分数不是一个说不清来源的黑盒数字。

3. **arXiv 分类不硬猜**。arxiv_id 本身不含分类，库里也没有这一列，所以分类只能从
   词项映射猜。映射表命不中就返回空列表交给调用方决定，绝不硬塞一个可能错的
   `cs.LG` ——推错分类的后果是整个订阅源都是噪音，比没有分类更糟。

4. **同一篇不在连续多天里反复出现**。新 digest 跳过最近 `repeat_after_days` 天已推过的
   norm_key，`dismiss` 过的则永久不再推。没有这道闸，订阅功能第三天就会被关掉。

5. **排序键全序**。分数并列时按 arxiv_id 决胜，否则同一批输入跨进程会排出不同顺序
   （本项目在评测集上踩过这个坑）。

schema 说明：`digest_items` 比原定列清单多两列，都是必要的，不是随手加的——
  · `authors_json`：不存作者，`adopt` 出来的论文就永远是「无作者」的残缺记录，
    而 arXiv 的 Atom 明明给了作者，丢掉它属于自伤；
  · `rank_score`：LLM 重排后的合成排序分。不存它，`digest()` 当场返回的顺序与
    事后 `get_digest()` 看到的顺序会不一致。它与 `score` 分列两栏，
    确定性词面分永远不被模型分覆盖。
"""
import functools
import json
import math
import re
import sqlite3
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from . import config, db, http, llm
from .normalize import norm_key
from .sources import arxiv as arxiv_source


class SubscribeError(Exception):
    """调用方错误（分类为空、digest/条目不存在等）。网络失败不走这个。"""


# ── 建表（幂等，不动 db.py 的 MIGRATIONS）──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS digests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now','localtime')),
  days INTEGER DEFAULT 1,
  n_fetched INTEGER DEFAULT 0,     -- 从 arXiv 实际拉到的条数
  n_new INTEGER DEFAULT 0,         -- 过完「已在库/近期推过/已忽略」三道闸后的候选数
  params_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS digest_items (
  digest_id INTEGER NOT NULL,
  norm_key TEXT NOT NULL,
  arxiv_id TEXT,
  title TEXT,
  abstract TEXT,
  published TEXT,                  -- arXiv 的 published 时间戳（原样保留）
  score REAL DEFAULT 0,            -- 确定性词面分（可审计，永远不被模型覆盖）
  reasons_json TEXT NOT NULL DEFAULT '[]',
  llm_note TEXT,
  -- 以下两列超出 brief 给的列清单，理由写在模块文档字符串的「schema 说明」里：
  authors_json TEXT NOT NULL DEFAULT '[]',   -- 不存作者，adopt 出来的论文就永远没有作者
  rank_score REAL,                 -- LLM 重排后的合成排序分；没重排则为 NULL
  dismissed INTEGER DEFAULT 0,
  PRIMARY KEY(digest_id, norm_key)
);
CREATE INDEX IF NOT EXISTS idx_digest_items_key ON digest_items(norm_key);
"""

_schema_lock = threading.Lock()
_schema_done: set[str] = set()


def ensure_schema(force: bool = False):
    """幂等建表。按 DB 路径记忆，避免每个公开函数入口都跑一遍 executescript
    （db.py 的注释里已经把「每次调用重建 schema」当成踩过的坑记下来了）。"""
    db.init_db(force=force)
    key = str(config.DB_PATH)
    if not force and key in _schema_done and config.DB_PATH.exists():
        return
    with _schema_lock:
        if not force and key in _schema_done and config.DB_PATH.exists():
            return
        with db.conn() as c:
            c.executescript(_SCHEMA)
        _schema_done.add(key)


# ── 画像参数（全部是可解释的常数，不是调出来的魔数）──

# 词项来源的权重档：作者/模型给的 keywords 最能代表方向，标题次之，venue 只作弱信号
W_KEYWORD = 3.0
W_TITLE = 1.0
W_VENUE = 0.5

# 近期加权：一年半衰。研究方向会漂移，两年前的老论文只该占四分之一的话语权。
# 基准点取「库内最新一篇的 created_at」而不是 datetime.now()，
# 这样同一个库任何时候算出的画像都一样（测试与跨进程结果才不会漂）。
RECENCY_HALF_LIFE_DAYS = 365.0
RECENCY_FLOOR = 0.1

MAX_TERMS = 60            # 画像词项上限
MIN_TERM_RATIO = 0.05     # 低于最高权重 5% 的长尾词丢掉（纯噪音）

# 打分参数
HIT_TITLE = 1.0           # 标题命中的加成
HIT_ABSTRACT = 0.4        # 摘要命中的加成
MATCH_CAP = 5             # 归一化基准：满分 = 标题命中画像里权重最高的 5 个词
IDF_MIN_BATCH = 20        # 批内 IDF 的最小样本量：少于这个数就不惩罚（没有统计意义）
SINGLE_CONCEPT_FACTOR = 0.6   # 只命中一个概念时的折扣（见 score_papers 的「广度」一段）

# venue 里这些值不作为词项（对 arXiv 订阅毫无区分度）
_VENUE_BLOCKLIST = {"arxiv", "arxiv.org", "preprint", "corr", "unknown", "none"}

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "are",
    "was", "were", "its", "our", "their", "his", "her", "you", "your", "not",
    "via", "using", "toward", "towards", "into", "onto", "over", "under",
    "can", "does", "how", "what", "why", "when", "where", "which", "who", "whom",
    "new", "novel", "paper", "study", "approach", "based", "use", "used", "uses",
    "results", "result", "case", "cases", "work", "works", "propose", "proposed",
    "proposes", "more", "less", "than", "then", "also", "such", "some",
    "any", "all", "one", "two", "three", "first", "second", "third", "very",
    # 中文二元词面里的高频套话（论文标题几乎人人都有，没有区分度）
    "研究", "方法", "基于", "分析", "一种", "问题", "模型",
}

_CJK_RE = re.compile(r"[一-鿿]")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-]*|[一-鿿]+")

# 词项 → arXiv 分类的写死映射。只列常见几个：宁可猜不出（返回空），
# 也不要把一个错分类塞给调用方，那会让整个订阅源变成噪音。
CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    "cs.CL": ("nlp", "natural language", "language model", "llm", "machine translation",
              "question answering", "summarization", "dialogue", "text generation",
              "tokenizer", "prompt", "instruction tuning", "自然语言", "大模型"),
    "cs.LG": ("machine learning", "deep learning", "neural network", "representation learning",
              "self-supervised", "transformer", "fine-tuning", "finetuning", "pretraining",
              "generalization", "reinforcement learning", "机器学习", "深度学习"),
    "cs.AI": ("agent", "multi-agent", "reasoning", "planning", "knowledge graph",
              "tool use", "autonomous", "智能体"),
    "cs.IR": ("retrieval", "information retrieval", "recommendation", "recommender",
              "ranking", "search engine", "rag", "retrieval-augmented", "检索", "推荐"),
    "cs.CV": ("computer vision", "vision", "image", "video", "segmentation",
              "object detection", "diffusion model", "视觉", "图像"),
    "eess.SP": ("channel estimation", "mimo", "ofdm", "beamforming", "signal processing",
                "wireless", "antenna", "spectrum", "near-field", "信道", "波束"),
}

# 分类入选门槛：低于最高分 25% 的不要（弱信号），最多给 3 个
CATEGORY_KEEP_RATIO = 0.25
MAX_CATEGORIES = 3


# ── 文本小工具 ──

def _norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("‐", "-").lower()).strip()


def _phrase_terms(text: str | None) -> set[str]:
    """从一段文本抽词项：英文一元/二元词，中文二元词面（中文没空格分词）。"""
    out: set[str] = set()
    prev: str | None = None
    for tok in _TOKEN_RE.findall(_norm_text(text)):
        if _CJK_RE.search(tok):
            # 中文整句没有空格分词，切二元词面（与 db._expand_terms 同一口径）
            out.update(g for g in (tok[i:i + 2] for i in range(len(tok) - 1))
                       if g not in _STOPWORDS)
            prev = None
            continue
        tok = tok.strip("-+")
        ok = len(tok) >= 3 and tok not in _STOPWORDS and not tok.isdigit()
        if ok:
            out.add(tok)
            if prev:
                out.add(f"{prev} {tok}")
        prev = tok if ok else None
    return out


def _term_matcher(term: str):
    """返回一个 (text) -> bool 的判定器。

    纯 ASCII 词项走词边界匹配：裸 `in` 会让 'rag' 命中 'storage'、'age' 命中 'language'，
    推荐理由里出现这种命中会直接摧毁用户对分数的信任。中文没有词边界，走子串。
    """
    if _CJK_RE.search(term):
        return lambda text, t=term: t in text
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
    return lambda text, p=pat: bool(p.search(text))


def _parse_dt(s: str | None) -> datetime | None:
    """容忍 'Z' 结尾与空格分隔的 ISO 时间；解析不了返回 None（不猜）。"""
    if not s:
        return None
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _loads(s, default):
    try:
        v = json.loads(s) if s else default
    except (ValueError, TypeError):
        return default
    return v if isinstance(v, type(default)) else default


# ── 1. 兴趣画像 ──

def build_profile(limit: int = 200) -> dict:
    """从库内论文反推兴趣画像。**不调 LLM、不联网**。

    词项权重 = Σ_论文 (来源档位 × 近期系数)，再除以最高权重归一化到 0~1。
    同一篇论文里同一个词只按最高来源档位计一次（df 加权，不是 tf）——
    否则一篇论文标题里重复出现的词会盖过「十篇论文都提到」这种真信号。

    返回 {"terms": [{"term","weight","df","source"}], "arxiv_categories": [...],
          "n_papers": int}。空库返回空词项，不炸。
    """
    ensure_schema()
    limit = max(int(limit), 0)
    if limit == 0:
        return {"terms": [], "arxiv_categories": [], "n_papers": 0}
    with db.conn() as c:
        rows = c.execute(
            """SELECT id, title, venue, card_json, created_at FROM papers
               ORDER BY created_at DESC, id DESC LIMIT ?""", (limit,)).fetchall()
    if not rows:
        return {"terms": [], "arxiv_categories": [], "n_papers": 0}

    dates = [d for d in (_parse_dt(r["created_at"]) for r in rows) if d]
    base = max(dates) if dates else None

    # term -> [累计权重, 文档频次, 最高来源档位]
    acc: dict[str, list] = {}
    for r in rows:
        rec = _recency(_parse_dt(r["created_at"]), base)
        per_paper: dict[str, tuple[float, str]] = {}

        card = _loads(r["card_json"], {})
        for kw in (card.get("keywords") or []) if isinstance(card, dict) else []:
            if not isinstance(kw, str):
                continue
            phrase = _norm_text(kw)
            if len(phrase) >= 2:
                per_paper[phrase] = (W_KEYWORD, "keyword")
            for t in _phrase_terms(kw):
                if per_paper.get(t, (0.0, ""))[0] < W_KEYWORD:
                    per_paper[t] = (W_KEYWORD, "keyword")

        for t in _phrase_terms(r["title"]):
            if per_paper.get(t, (0.0, ""))[0] < W_TITLE:
                per_paper[t] = (W_TITLE, "title")

        venue = _norm_text(r["venue"])
        if venue and venue not in _VENUE_BLOCKLIST and len(venue) >= 2:
            if per_paper.get(venue, (0.0, ""))[0] < W_VENUE:
                per_paper[venue] = (W_VENUE, "venue")

        for term, (w, src) in per_paper.items():
            slot = acc.setdefault(term, [0.0, 0, src])
            slot[0] += w * rec
            slot[1] += 1
            if _SRC_RANK[src] > _SRC_RANK[slot[2]]:
                slot[2] = src

    if not acc:
        return {"terms": [], "arxiv_categories": [], "n_papers": len(rows)}

    top = max(v[0] for v in acc.values()) or 1.0
    terms = [{"term": t, "weight": round(v[0] / top, 6), "df": v[1], "source": v[2]}
             for t, v in acc.items() if v[0] / top >= MIN_TERM_RATIO]
    # 全序：权重降序，并列按词典序——跨进程重跑必须给出同一份画像
    terms.sort(key=lambda d: (-d["weight"], d["term"]))
    terms = terms[:MAX_TERMS]
    return {"terms": terms, "arxiv_categories": guess_categories(terms),
            "n_papers": len(rows)}


_SRC_RANK = {"venue": 0, "title": 1, "keyword": 2}


def _recency(created: datetime | None, base: datetime | None) -> float:
    """越新权重越高（一年半衰）。时间戳解析不了的按最低档，不当成「最新」。"""
    if created is None or base is None:
        return RECENCY_FLOOR
    age_days = max((base - created).total_seconds() / 86400.0, 0.0)
    return max(RECENCY_FLOOR, 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


@functools.lru_cache(maxsize=1024)
def _cat_matcher(phrase: str):
    """分类映射词的判定器：与 `_term_matcher` 同一口径，走词边界（允许复数尾巴）。

    这里曾经是裸 `in` 子串匹配，后果和打分层那个 bug 一模一样，只是更隐蔽：
    cs.IR 的 'rag' 会命中 'average' / 'coverage' / 'storage'，cs.CV 的 'vision'
    会命中 'supervision'——于是一个做无线通信的用户（画像里全是 'average rate'、
    'coverage probability'）会被塞一个 cs.IR，整条订阅源变成噪音。
    模块开头写着「宁可猜不出也不硬塞」，那就不能在这一层用会误命中的匹配。
    """
    if _CJK_RE.search(phrase):
        return lambda term, p=phrase: p in term
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?:e?s)?(?![a-z0-9])")
    return lambda term, p=pat: bool(p.search(term))


def guess_categories(terms: list[dict]) -> list[str]:
    """从画像词项猜 arXiv 分类。猜不出返回 []（由调用方显式指定，绝不硬塞）。"""
    scores: dict[str, float] = {}
    for cat, phrases in CATEGORY_TERMS.items():
        matchers = [_cat_matcher(p) for p in phrases]
        s = 0.0
        for t in terms:
            term = t.get("term") or ""
            w = float(t.get("weight") or 0.0)
            if any(m(term) for m in matchers):
                s += w
        if s > 0:
            scores[cat] = s
    if not scores:
        return []
    best = max(scores.values())
    picked = [(s, cat) for cat, s in scores.items() if s >= best * CATEGORY_KEEP_RATIO]
    picked.sort(key=lambda x: (-x[0], x[1]))   # 全序
    return [cat for _, cat in picked[:MAX_CATEGORIES]]


# ── 2. 拉 arXiv 最近提交 ──

_PAGE_SIZE = 100          # arXiv 单次建议不超过这个量级
_MAX_PAGES = 10
_CAT_RE = re.compile(r"^[A-Za-z][A-Za-z\-]*(\.[A-Za-z\-]+)?$")


def fetch_recent(categories: list[str], days: int = 1, limit: int = 200,
                 now: datetime | None = None) -> list[dict]:
    """拉指定 arXiv 分类最近 `days` 天提交的论文。

    结果 dict 与 `papernest.sources.arxiv.search()` 对齐（多一个 `published` 字段），
    可以直接喂 `db.insert_l0`。

    `now` 只为测试注入固定时钟用；生产调用不传，走 UTC 当前时间。
    单条 entry 解析失败只跳过该条，绝不让一个畸形 entry 毁掉整批。
    """
    cats = [c.strip() for c in (categories or []) if c and c.strip()]
    bad = [c for c in cats if not _CAT_RE.match(c)]
    if bad:
        raise SubscribeError(f"非法 arXiv 分类：{bad}")
    if not cats:
        raise SubscribeError("必须指定至少一个 arXiv 分类（build_profile 猜不出时由调用方决定）")
    days = max(int(days), 1)
    limit = max(int(limit), 0)
    if limit == 0:
        return []
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    query = " OR ".join(f"cat:{c}" for c in cats)

    out: list[dict] = []
    seen: set[str] = set()
    stop = False
    with http.client() as client:
        for page in range(_MAX_PAGES):
            if stop or len(out) >= limit:
                break
            if page:
                arxiv_source.polite_sleep()   # arXiv 官方建议的 3 秒间隔，复用同一口径
            r = client.get(
                arxiv_source.API_URL,
                params={"search_query": query, "start": page * _PAGE_SIZE,
                        "max_results": _PAGE_SIZE, "sortBy": "submittedDate",
                        "sortOrder": "descending"},
                headers={"User-Agent": "PaperNest/0.1 (personal research tool)"})
            r.raise_for_status()
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError as exc:
                raise SubscribeError(f"arXiv 返回的不是合法 Atom XML：{exc}") from exc
            entries = root.findall("a:entry", arxiv_source.NS)
            for e in entries:
                try:
                    p = _parse_entry(e)
                except Exception:
                    continue          # 畸形 entry：跳过这一条，不影响整批
                if p is None:
                    continue
                pub = _parse_dt(p["published"])
                if pub is None:
                    continue          # 没有可信的提交时间就没法判断「是不是最近的」
                if pub < cutoff:
                    stop = True       # 结果按提交时间降序，遇到更早的即可收工
                    break
                if p["norm_key"] in seen:
                    continue
                seen.add(p["norm_key"])
                out.append(p)
                if len(out) >= limit:
                    break
            if len(entries) < _PAGE_SIZE:
                break                 # 不满一页 = 结果已取尽
    return out


# 合法 arXiv id：新式 2608.00011[v3]，旧式 math.GT/0309136[v1] / hep-th/9901001。
# 必须校验：arXiv 在查询非法时返回的是一个**形状完全合法的 Atom feed**，里面一条
# id=http://arxiv.org/api/errors#... 、title=Error 的 entry。不校验的话这条会被当成
# 一篇论文打分、落进 digest_items、再被 adopt 写进 papers——标题「Error」、
# arxiv_id 是一整条 URL、oa_pdf_url 拼成 https://arxiv.org/pdf/http://...。
_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$")


def _parse_entry(e: ET.Element) -> dict | None:
    """单个 Atom entry → 与 sources/arxiv.py 对齐的 dict；缺 id/title 返回 None。"""
    ns = arxiv_source.NS
    raw_id = e.findtext("a:id", "", ns) or ""
    if "/abs/" not in raw_id:
        return None       # 不是论文条目（arXiv 的 error feed 走的就是这条）
    arxiv_id = raw_id.split("/abs/")[-1].strip()
    title = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
    title = re.sub(r"\s+", " ", title)
    if not title or not _ARXIV_ID_RE.match(arxiv_id):
        return None
    published = (e.findtext("a:published", "", ns) or "").strip()
    summary = re.sub(r"\s+", " ", (e.findtext("a:summary", "", ns) or "").strip())
    authors = [n.strip() for n in
               (a.findtext("a:name", "", ns) or "" for a in e.findall("a:author", ns))
               if n and n.strip()]
    return {
        "norm_key": norm_key(title=title, arxiv_id=arxiv_id),
        "title": title,
        "abstract": summary or None,
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": "arXiv",
        "authors": authors,
        "doi": None,
        "arxiv_id": arxiv_id,
        "citation_count": None,
        "oa_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "source": "arxiv",
        "s2_id": None,
        "published": published,
    }


# ── 3. 确定性打分（0 token）──

def _batch_idf(matchers: list, papers: list[dict]) -> dict[str, float]:
    """用**这批论文自己**算每个画像词的区分度（IDF），不需要任何外部语料。

    为什么非加不可——实测（498 篇无线通信库 × 当天 72 篇 arXiv 新文）：
    画像里权重最高的是 `learning`(0.93) / `deep`(0.98)，因为库内很多论文标题带它们。
    但它们同时出现在当天 arXiv 的**半数**论文里，等于没有区分度，于是推荐结果是
    「城市交通预测」「核岭回归」「农业 Web 系统」——命中的全是 learning 一个词，
    而库真正的主题 beamforming / MIMO / 信道估计 一个都没匹配上。
    这是缺 IDF 的教科书症状：库内文档频次（画像权重）只说明「我关心它」，
    批内文档频次才说明「它能不能把这批论文区分开」。

    idf = log((N+1)/(df+1)) / log(N+1)，落在 (0,1]：
    df 占满全批 → 趋近 0（词被压没）；df=1 的罕见词 → 接近 1（原样保留）。
    批内计算 ⇒ 同一批输入必得同一组系数，确定性不受影响。
    """
    n = len(papers)
    if n < IDF_MIN_BATCH:      # 批太小时 IDF 没有统计意义，全给 1.0（不惩罚）
        return {t: 1.0 for t, _w, _h in matchers}
    texts = [(_norm_text(p.get("title")) + " " + _norm_text(p.get("abstract")))
             for p in papers]
    scale = math.log(n + 1)
    out: dict[str, float] = {}
    for term, _weight, hit in matchers:
        df = sum(1 for t in texts if hit(t))
        out[term] = math.log((n + 1) / (df + 1)) / scale
    return out


def score_papers(papers: list[dict], profile: dict) -> list[dict]:
    """按画像给论文打分。纯词面命中，确定性、可手算、0 次模型调用。

    单词项贡献 = 词权重 × (标题命中 HIT_TITLE / 仅摘要命中 HIT_ABSTRACT)
    × **批内 IDF**（见 `_batch_idf`：泛词被压、罕见词保留）；
    标题与摘要都命中只算标题那一份（不重复计分）；随后 `_dedupe_reasons` 把
    「短语 + 成分词」并成一个概念，每个概念只保留最强的一条证据。

    归一化基准 `denom` = 画像里权重最高的 MATCH_CAP 个**词项**权重之和。
    注意这是一把**固定的尺子，不是可达的满分**：分子按「概念」计分而分母按
    「词项」求和，而真实画像的 top-5 往往互相包含（例如 massive / mimo /
    massive mimo 同时在列），所以一篇「命中了全部 top-5」的论文实测也只到
    0.4~0.6，1.0 基本不可达。这不影响排序（同一次 digest 内 denom 是常数），
    但**别把 score 当成百分比读**，也别在 UI 上写「相关度 41%」。

    每条结果带 `reasons`：命中了哪个词、在哪命中、词权重多少、贡献多少分。
    """
    terms = sorted(((t.get("term") or "", float(t.get("weight") or 0.0))
                    for t in (profile or {}).get("terms") or []),
                   key=lambda x: (-x[1], x[0]))
    terms = [(t, w) for t, w in terms if t and w > 0]
    matchers = [(t, w, _term_matcher(t)) for t, w in terms]
    denom = HIT_TITLE * sum(w for _, w in terms[:MATCH_CAP]) or 1.0
    idf = _batch_idf(matchers, papers or [])

    out: list[dict] = []
    for p in papers or []:
        title = _norm_text(p.get("title"))
        abstract = _norm_text(p.get("abstract"))
        reasons: list[dict] = []
        raw = 0.0
        for term, weight, hit in matchers:
            if hit(title):
                where, factor = "title", HIT_TITLE
            elif abstract and hit(abstract):
                where, factor = "abstract", HIT_ABSTRACT
            else:
                continue
            disc = idf.get(term, 1.0)
            contrib = weight * factor * disc
            raw += contrib
            reasons.append({"term": term, "where": where,
                            "weight": round(weight, 6),
                            "idf": round(disc, 4),
                            "contribution": round(contrib, 6)})
        reasons, raw = _dedupe_reasons(reasons)
        reasons.sort(key=lambda d: (-d["contribution"], d["term"]))  # 全序
        # 广度折扣：只命中一个概念的论文打折。实测里排最前的往往是「标题里有
        # deep 或 learning，此外与本库毫无关系」的论文——一个泛词命中不足以
        # 构成推荐理由，而命中两个不同概念（比如 beamforming + power）几乎总是真相关。
        single = len(reasons) == 1
        # 折扣只作用于 score，不动 raw_score——raw_score 必须始终等于各条理由
        # 贡献之和，否则「理由能解释分数」这条就断了，而它是这个模块的立身之本。
        item = dict(p)
        item["raw_score"] = round(raw, 6)
        item["score"] = round(
            min(raw * (SINGLE_CONCEPT_FACTOR if single else 1.0) / denom, 1.0), 6)
        item["reasons"] = reasons
        item["n_concepts"] = len(reasons)
        if single:
            item["weak_match"] = "只命中一个词项，相关性证据较弱"
        out.append(item)
    # 全序：分数降序，并列按 arxiv_id（缺失则退回 norm_key）字典序决胜。
    # 只按分数排会让并列条目的顺序取决于输入顺序/字典遍历顺序，跨进程漂移。
    out.sort(key=lambda d: (-d["score"], _tiebreak(d)))
    return out


def _sub_concepts(term: str) -> list[str]:
    """一个词项「包含」的更小词项：英文短语 → 各成分词；中文词面 → 各二元词面。

    中文这一支不能漏：画像里的「信道估计」会连带产生「信道」「道估」「估计」
    三个二元词面（`_phrase_terms` 对中文只能切二元），一篇中文标题命中一次
    就并排给出四条理由、拿四份分——比英文那边的重复更严重。
    """
    if " " in term:
        return term.split()
    if len(term) > 2 and _CJK_RE.search(term):
        return [term[i:i + 2] for i in range(len(term) - 1)]
    return []


def _reason_rank(r: dict) -> tuple:
    """同一概念内挑「最强证据」的排序键。并列时取更长的词项（信息量更大），
    末位用词面本身兜底以保证全序——否则同分概念的展示会跨进程漂移。"""
    return (r["contribution"], len(r["term"]), r["term"])


def _dedupe_reasons(reasons: list[dict]) -> tuple[list[dict], float]:
    """把「短语 + 它的成分词」并成一个概念，每个概念只留最强的一条证据。
    返回 (去重后的理由, 重算的原始分)。

    为什么要并：画像里「channel estimation」会连带产生「channel」「estimation」
    两个单词词项（keywords 抽取本来就同时收短语和成分词）。不并的话，一个双词
    概念会拿到单词概念三倍的分，排序被短语概念系统性地拉偏；推荐理由里也会并排
    列出三条说的是同一件事的解释——用户看到的是重复，不是证据。

    为什么是「留最强的一条」而不是「无条件留短语」：短语在**摘要**命中、成分词在
    **标题**命中时，无条件留短语会把强证据（标题命中，系数 1.0）压成弱证据
    （摘要命中，系数 0.4）。实测（498 篇真实库的画像）：
      标题「Channel Prediction for Massive Arrays」              → 0.400
      同一标题 + 摘要补上「channel estimation with deep learning」→ 0.326
    补上更多相关内容分数反而掉 18%，这种非单调没有任何人能看懂，而「看得懂」
    正是这个模块的立身之本。取组内最强证据后，分数只会随证据变强而升。
    """
    n = len(reasons)
    if n < 2:
        return list(reasons), sum(r["contribution"] for r in reasons)

    pos = {r["term"]: i for i, r in enumerate(reasons)}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, r in enumerate(reasons):
        for part in _sub_concepts(r["term"]):
            j = pos.get(part)
            if j is None or j == i:
                continue
            ra, rb = find(i), find(j)
            if ra != rb:                    # 按下标合并，与输入顺序一致、可复现
                parent[max(ra, rb)] = min(ra, rb)

    best: dict[int, int] = {}
    for i, r in enumerate(reasons):
        root = find(i)
        cur = best.get(root)
        if cur is None or _reason_rank(r) > _reason_rank(reasons[cur]):
            best[root] = i
    kept = [reasons[i] for i in sorted(best.values())]
    return kept, sum(r["contribution"] for r in kept)


def _tiebreak(d: dict) -> str:
    return str(d.get("arxiv_id") or d.get("norm_key") or d.get("title") or "")


# ── 4. 可选的 LLM 重排 ──

_RERANK_SYS = ("你是科研文献筛选助手。给定研究方向与若干新论文，判断每篇与该方向的相关性。"
               "只输出 JSON，不要解释。")


def llm_rerank(scored: list[dict], topic: str, top_n: int = 20) -> list[dict]:
    """对前 top_n 条做语义重排并给一句话理由。

    未配置 LLM 或调用失败时**原样返回**（degraded 由 `_rerank` 的第二个返回值表达，
    digest 会把它写进结果）。这一层绝不允许把整个 digest 拖挂。
    """
    return _rerank(scored, topic, top_n)[0]


def _rerank(scored: list[dict], topic: str, top_n: int = 20) -> tuple[list[dict], str | None]:
    """返回 (结果, degraded 说明或 None)。"""
    # 逐条浅拷贝：这一层可能给条目挂 llm_note/rank_score，绝不能就地改调用方的 dict
    # （曾经因此让「未配置 LLM」的调用也看到上一次重排留下的理由）。
    items = [dict(p) for p in (scored or [])]
    if not items:
        return items, None
    if not llm.available():
        return items, "未配置 LLM，跳过语义重排，仅用确定性词面打分"
    top_n = max(int(top_n), 0)
    head, tail = items[:top_n], items[top_n:]
    if not head:
        return items, None

    lines = []
    for i, p in enumerate(head):
        abstract = (p.get("abstract") or "")[:400]
        lines.append(f"[{i}] 标题：{p.get('title', '')}\n    摘要：{abstract}")
    user = (f"研究方向：{topic or config.RESEARCH_TOPIC}\n\n候选论文：\n"
            + "\n".join(lines)
            + '\n\n请输出 JSON：{"items":[{"i":序号,"score":0~1的相关性,'
              '"note":"一句话理由（20 字内）"}]}。不相关的也要给出条目，score 给低分。')
    try:
        data = llm.extract_json(llm.chat(_RERANK_SYS, user, purpose="subscribe_rerank",
                                         temperature=0.1))
    except Exception as exc:
        return items, f"LLM 重排失败（{type(exc).__name__}: {str(exc)[:120]}），已退回确定性打分顺序"

    got = 0
    for raw in (data.get("items") or []) if isinstance(data, dict) else []:
        if not isinstance(raw, dict):
            continue
        try:
            idx = int(raw.get("i"))
            ls = float(raw.get("score"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(head):
            continue
        ls = min(max(ls, 0.0), 1.0)
        head[idx]["llm_score"] = round(ls, 6)
        note = raw.get("note")
        head[idx]["llm_note"] = str(note)[:200] if note else None
        # 一半词面一半语义：模型给的分不覆盖可审计的确定性分数，只与它合成排序分
        head[idx]["rank_score"] = round(0.5 * head[idx]["score"] + 0.5 * ls, 6)
        got += 1
    if not got:
        return items, "LLM 重排返回的 JSON 里没有可用条目，已退回确定性打分顺序"

    for p in head:
        p.setdefault("rank_score", p["score"])
    for p in tail:
        p.setdefault("rank_score", p["score"])
    head.sort(key=lambda d: (-d["rank_score"], -d["score"], _tiebreak(d)))
    return head + tail, None


# ── 5. 顶层 digest ──

def digest(days: int = 1, top_k: int = 15, categories: list[str] | None = None,
           use_llm: bool = False, progress=None, repeat_after_days: int = 30,
           fetch_limit: int = 200, topic: str | None = None,
           profile_limit: int = 200, now: datetime | None = None) -> dict:
    """画像 → 拉新 → 去重 → 打分 →（可选重排）→ 落库。

    三道去重闸：① 已在库（norm_key 命中 papers）；② 最近 repeat_after_days 天推过；
    ③ 曾被 dismiss（永久）。第 ② 条是订阅功能能不能活过一周的关键。

    `progress(frac, stage, message)` 供异步任务接进度，同步调用不传即可。
    `degraded` 是一个字符串列表，如实记录每一处降级；空列表表示全功能跑通。
    `now` 只为测试注入固定时钟用，生产调用不传。
    """
    ensure_schema()
    top_k = max(int(top_k), 0)
    repeat_after_days = max(int(repeat_after_days), 0)
    degraded: list[str] = []

    def _tick(frac, stage, message):
        if progress:
            progress(frac, stage, message)

    _tick(0.05, "profile", "从库内论文反推兴趣画像")
    profile = build_profile(profile_limit)
    if not profile["terms"]:
        degraded.append("库内没有可用词项（空库或论文缺标题/关键词），无法个性化打分")

    cats = [c for c in (categories or []) if c] or profile["arxiv_categories"]
    if not cats:
        degraded.append("未指定 arXiv 分类且无法从画像猜出，本次未拉取任何论文"
                        "（请显式传 categories）")
        return _persist(days, 0, 0, [], {"categories": [], "top_k": top_k,
                                         "use_llm": use_llm,
                                         "repeat_after_days": repeat_after_days},
                        degraded)

    _tick(0.2, "fetch", f"拉 arXiv 最近 {days} 天：{'+'.join(cats)}")
    try:
        fetched = fetch_recent(cats, days=days, limit=fetch_limit, now=now)
    except SubscribeError:
        raise
    except Exception as exc:
        degraded.append(f"arXiv 拉取失败（{type(exc).__name__}: {str(exc)[:150]}）")
        return _persist(days, 0, 0, [], {"categories": cats, "top_k": top_k,
                                         "use_llm": use_llm,
                                         "repeat_after_days": repeat_after_days},
                        degraded)

    _tick(0.5, "dedupe", f"拉到 {len(fetched)} 篇，剔除库内已有与近期推过的")
    with db.conn() as c:
        in_lib = {r["norm_key"] for r in c.execute("SELECT norm_key FROM papers")}
        dismissed = {r["norm_key"] for r in
                     c.execute("SELECT DISTINCT norm_key FROM digest_items WHERE dismissed=1")}
        recent = {r["norm_key"] for r in c.execute(
            """SELECT DISTINCT di.norm_key FROM digest_items di
               JOIN digests d ON d.id = di.digest_id
               WHERE d.ts >= datetime('now','localtime', ?)""",
            (f"-{repeat_after_days} days",))} if repeat_after_days else set()

    candidates = [p for p in fetched
                  if p.get("norm_key")
                  and p["norm_key"] not in in_lib
                  and p["norm_key"] not in dismissed
                  and p["norm_key"] not in recent]

    _tick(0.7, "score", f"对 {len(candidates)} 篇候选做确定性打分")
    scored = score_papers(candidates, profile)

    if use_llm:
        _tick(0.85, "rerank", "LLM 语义重排（可选层）")
        scored, note = _rerank(scored, topic or config.RESEARCH_TOPIC, top_n=max(top_k, 1))
        if note:
            degraded.append(note)

    items = scored[:top_k]
    params = {"categories": cats, "top_k": top_k, "use_llm": use_llm,
              "repeat_after_days": repeat_after_days, "days": days,
              "profile_n_papers": profile["n_papers"],
              "profile_top_terms": [t["term"] for t in profile["terms"][:10]]}
    out = _persist(days, len(fetched), len(candidates), items, params, degraded)
    _tick(1.0, "done", f"新论文 {len(candidates)} 篇，推送 {len(items)} 篇")
    return out


def _persist(days: int, n_fetched: int, n_new: int, items: list[dict],
             params: dict, degraded: list[str]) -> dict:
    params = dict(params)
    params["degraded"] = degraded
    with db.conn() as c:
        cur = c.execute(
            "INSERT INTO digests(days,n_fetched,n_new,params_json) VALUES(?,?,?,?)",
            (days, n_fetched, n_new, json.dumps(params, ensure_ascii=False)))
        digest_id = cur.lastrowid
        for p in items:
            c.execute(
                """INSERT OR REPLACE INTO digest_items
                   (digest_id,norm_key,arxiv_id,title,abstract,published,score,
                    reasons_json,llm_note,authors_json,rank_score,dismissed)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,0)""",
                (digest_id, p["norm_key"], p.get("arxiv_id"), p.get("title"),
                 p.get("abstract"), p.get("published"), p.get("score", 0.0),
                 json.dumps(p.get("reasons") or [], ensure_ascii=False),
                 p.get("llm_note"),
                 json.dumps(p.get("authors") or [], ensure_ascii=False),
                 p.get("rank_score")))
        row = c.execute("SELECT ts FROM digests WHERE id=?", (digest_id,)).fetchone()
    return {"digest_id": digest_id, "date": (row["ts"] or "")[:10],
            "n_fetched": n_fetched, "n_new": n_new,
            "items": [_public_item(p) for p in items], "degraded": degraded}


def _public_item(p: dict) -> dict:
    return {"norm_key": p.get("norm_key"), "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"), "abstract": p.get("abstract"),
            "published": p.get("published"), "authors": p.get("authors") or [],
            "oa_pdf_url": p.get("oa_pdf_url"), "score": p.get("score", 0.0),
            "reasons": p.get("reasons") or [], "llm_note": p.get("llm_note"),
            "llm_score": p.get("llm_score"), "rank_score": p.get("rank_score")}


# ── 6/7. 一键入库 / 忽略 / 历史 ──

def adopt(digest_id: int, norm_key: str) -> dict:
    """把 digest 里的某篇一键入库（source="arxiv-digest"）。已在库则返回既有记录。"""
    ensure_schema()
    with db.conn() as c:
        it = c.execute("SELECT * FROM digest_items WHERE digest_id=? AND norm_key=?",
                       (int(digest_id), norm_key)).fetchone()
        if it is None:
            raise SubscribeError(f"digest {digest_id} 里没有 {norm_key}")
        exist = db.get_by_norm_key(c, norm_key)
        if exist:
            return {"status": "exists", "paper_id": exist["id"], "norm_key": norm_key,
                    "title": exist["title"],
                    "note": "该论文已在库中，未重复插入"}
        if not it["title"]:
            raise SubscribeError(f"{norm_key} 没有标题，拒绝入库（不给论文编标题）")
        published = it["published"] or ""
        try:
            pid = _insert_adopted(c, it, norm_key, published)
        except sqlite3.IntegrityError:
            # 「查一下不在 → 插进去」之间有别人抢先插了（前端连点两次、或后台
            # 检索任务同时把这篇捞进来）。papers.norm_key 是 UNIQUE，这里会炸成
            # 500。语义上这就是「已在库」，退回去重读即可，不该让用户看到栈。
            c.rollback()
            exist = db.get_by_norm_key(c, norm_key)
            if exist is None:
                raise
            return {"status": "exists", "paper_id": exist["id"], "norm_key": norm_key,
                    "title": exist["title"],
                    "note": "该论文已在库中（并发写入抢先），未重复插入"}
    return {"status": "adopted", "paper_id": pid, "norm_key": norm_key,
            "title": it["title"], "digest_id": int(digest_id),
            "note": "入库为 arXiv 元数据，未生成卡片（需要卡片请另跑 L1 流程）"}


def _insert_adopted(c, it, norm_key: str, published: str) -> int:
    return db.insert_l0(c, {
        "norm_key": norm_key, "title": it["title"], "abstract": it["abstract"],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": "arXiv", "authors": _loads(it["authors_json"], []),
        "doi": None, "arxiv_id": it["arxiv_id"], "citation_count": None,
        "oa_pdf_url": f"https://arxiv.org/pdf/{it['arxiv_id']}" if it["arxiv_id"] else None,
        "source": "arxiv-digest", "s2_id": None,
    })


def dismiss(digest_id: int, norm_key: str) -> dict:
    """标记「不感兴趣」。之后任何 digest 都不再推这个 norm_key（永久，与时间窗无关）。"""
    ensure_schema()
    with db.conn() as c:
        cur = c.execute("UPDATE digest_items SET dismissed=1 WHERE digest_id=? AND norm_key=?",
                        (int(digest_id), norm_key))
        if cur.rowcount == 0:
            raise SubscribeError(f"digest {digest_id} 里没有 {norm_key}")
    return {"status": "dismissed", "digest_id": int(digest_id), "norm_key": norm_key}


def list_digests(limit: int = 20) -> list[dict]:
    """历史 digest 列表（新的在前）。"""
    ensure_schema()
    limit = max(int(limit), 0)
    if limit == 0:
        return []
    with db.conn() as c:
        rows = c.execute(
            """SELECT d.id, d.ts, d.days, d.n_fetched, d.n_new, d.params_json,
                      (SELECT COUNT(*) FROM digest_items i WHERE i.digest_id=d.id) n_items,
                      (SELECT COUNT(*) FROM digest_items i
                       WHERE i.digest_id=d.id AND i.dismissed=1) n_dismissed
               FROM digests d ORDER BY d.id DESC LIMIT ?""", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        params = _loads(d.pop("params_json"), {})
        d["params"] = params
        d["degraded"] = params.get("degraded") or []
        d["date"] = (d.get("ts") or "")[:10]
        out.append(d)
    return out


def get_digest(digest_id: int) -> dict | None:
    """单个 digest 的完整内容。`in_library` 实时 JOIN papers 判定，不缓存状态。"""
    ensure_schema()
    with db.conn() as c:
        d = c.execute("SELECT * FROM digests WHERE id=?", (int(digest_id),)).fetchone()
        if d is None:
            return None
        rows = c.execute(
            """SELECT i.*, p.id AS paper_id FROM digest_items i
               LEFT JOIN papers p ON p.norm_key = i.norm_key
               WHERE i.digest_id=?
               ORDER BY COALESCE(i.rank_score, i.score) DESC, i.score DESC,
                        COALESCE(NULLIF(i.arxiv_id,''), i.norm_key) ASC""",
            (int(digest_id),)).fetchall()
    params = _loads(d["params_json"], {})
    items = []
    for r in rows:
        items.append({
            "norm_key": r["norm_key"], "arxiv_id": r["arxiv_id"], "title": r["title"],
            "abstract": r["abstract"], "published": r["published"],
            "authors": _loads(r["authors_json"], []),
            "score": r["score"], "reasons": _loads(r["reasons_json"], []),
            "llm_note": r["llm_note"], "rank_score": r["rank_score"],
            "dismissed": bool(r["dismissed"]),
            "in_library": r["paper_id"] is not None, "paper_id": r["paper_id"],
            "oa_pdf_url": f"https://arxiv.org/pdf/{r['arxiv_id']}" if r["arxiv_id"] else None,
        })
    return {"digest_id": d["id"], "ts": d["ts"], "date": (d["ts"] or "")[:10],
            "days": d["days"], "n_fetched": d["n_fetched"], "n_new": d["n_new"],
            "params": params, "degraded": params.get("degraded") or [], "items": items}
