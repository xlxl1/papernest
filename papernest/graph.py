"""引文网络 / 相关论文发现。

设计要点（为什么这样做）：

1. **边用 norm_key 而不是 paper_id**。引文的另一端绝大多数不在库里；如果边只能挂在
   库内 id 上，这些「库外邻居」根本存不下来，图就只剩孤岛。用归一化主键做端点后，
   同一篇论文无论从 S2 引文里冒出来、还是日后被正常检索入库，都落在同一个键上——
   入库那一刻边自动接上，不需要任何回填也能被 `neighbors` 认出来（`in_library`
   一律实时 JOIN papers 判定，绝不信边表里那两个冗余的 *_paper_id 列）。
   冗余列只是给 UI 省一次查询，`adopt` 会顺手回填，但它们不是真相来源。

2. **共被引 / 文献耦合是这个模块的主菜**。直接引用只能顺着一条链走，而
   「被同一批论文一起引用」（co-citation）和「引用了同一批论文」（bibliographic
   coupling）能把领域里真正相邻、但彼此不互引的工作挖出来——这是 Connected Papers
   的核心，纯 SQL + Python 就能算，不联网、不调 LLM。

3. **缺口论文（gap_papers）** 是本模块最实用的产出：库内多篇论文都引了它、它却不在
   库里，几乎必然是该补的经典/基础文献。

抽不到的字段一律留空并在返回里标注（`title` 为 None 就是真没拿到标题），
绝不用占位值糊上去。
"""
import json
import re
import threading
import time
from pathlib import Path

import httpx

from . import config, db, http
from .normalize import norm_key

# ── 常量 ──

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,year,authors,externalIds,citationCount"

DIRECTIONS = ("references", "citations")

# 共被引与文献耦合的加权（合成 related 的 score）。两者互补：
# 共被引反映「后人怎么把它们摆在一起」，耦合反映「作者自己站在谁的肩膀上」。
CO_WEIGHT = 0.5
BC_WEIGHT = 0.5

# S2 的 references/citations 单次上限
S2_MAX_LIMIT = 1000

# 图遍历深度上限：2 跳已经能覆盖上千节点，再深对个人库没有意义且会爆
MAX_DEPTH = 3

_CHUNK = 400  # SQL IN (...) 的分片大小，避免撞 SQLITE_MAX_VARIABLE_NUMBER


class GraphError(Exception):
    """调用方错误（论文不存在、方向非法、缺外部标识）——网络失败不走这个。"""


# ── 建表（幂等，不动 db.py 的 MIGRATIONS）──

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS citation_edges (
  src_key TEXT NOT NULL,        -- 施引论文的 norm_key
  dst_key TEXT NOT NULL,        -- 被引论文的 norm_key
  src_paper_id INTEGER,         -- 库内则填 id，库外为 NULL（冗余缓存，非真相来源）
  dst_paper_id INTEGER,
  fetched_at TEXT DEFAULT (datetime('now','localtime')),
  PRIMARY KEY(src_key, dst_key),
  CHECK (src_key <> dst_key)    -- 自引边（数据源偶尔会给）在库层面也堵死
);
CREATE INDEX IF NOT EXISTS idx_cedges_dst ON citation_edges(dst_key);
CREATE INDEX IF NOT EXISTS idx_cedges_src ON citation_edges(src_key);
CREATE TABLE IF NOT EXISTS citation_nodes (
  norm_key TEXT PRIMARY KEY,
  title TEXT,
  year INTEGER,
  authors_json TEXT DEFAULT '[]',
  doi TEXT,
  arxiv_id TEXT,
  citation_count INTEGER,
  s2_id TEXT,
  updated_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS citation_fetch (
  paper_key TEXT NOT NULL,
  direction TEXT NOT NULL,      -- references | citations
  fetched_at TEXT DEFAULT (datetime('now','localtime')),
  n_edges INTEGER DEFAULT 0,
  PRIMARY KEY(paper_key, direction)
);
"""


_schema_lock = threading.Lock()
_schema_done: set[str] = set()


def ensure_schema(force: bool = False):
    """边表 / 节点缓存 / 抓取日志。每个公开函数入口调用。

    单开 citation_fetch 而不是拿边的 fetched_at 判新鲜度：一篇论文可能真的
    零参考文献（或 S2 就是没有数据），那样边表里永远没有它的行，
    「拉过了」这个事实会丢，每次都重拉。日志表记录的是**尝试**，不是结果。

    进程内按 DB 路径只建一次（守卫写法与 db.init_db 一致）：本模块每个公开
    函数入口都调它，原来每次都要多开一条连接跑一遍 executescript——
    db.py 的注释里已经把这个反模式当成踩过的坑记下来了，别在这儿再种一次。
    """
    db.init_db(force=force)
    key = str(config.DB_PATH)
    if not force and key in _schema_done and config.DB_PATH.exists():
        return
    with _schema_lock:
        if not force and key in _schema_done and config.DB_PATH.exists():
            return
        with db.conn() as c:
            c.executescript(GRAPH_SCHEMA)
        _schema_done.add(key)


# ── 小工具 ──

def _chunks(seq, n: int = _CHUNK):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _grouped(c, sql: str, keys, extra: tuple = ()) -> dict:
    """对 `{ph}` 占位的分组统计做分片执行并累加。

    分片安全的前提：分片切的是 IN 里那一列，而分组键要么与它同列（各片互斥），
    要么统计的是 COUNT(DISTINCT 被切列)（各片的 distinct 集合互不相交），
    两种情况下跨片求和都等于整表结果。本模块的两处调用都满足。
    """
    out: dict[str, int] = {}
    for chunk in _chunks(keys):
        ph = ",".join("?" * len(chunk))
        for r in c.execute(sql.format(ph=ph), tuple(chunk) + extra):
            out[r[0]] = out.get(r[0], 0) + r[1]
    return out


def _rows_in(c, sql: str, keys, extra: tuple = ()) -> list:
    out = []
    for chunk in _chunks(keys):
        ph = ",".join("?" * len(chunk))
        out.extend(c.execute(sql.format(ph=ph), tuple(chunk) + extra).fetchall())
    return out


def _touching(c, keys) -> list[dict]:
    """所有「至少有一端落在 keys 里」的边，已按 (src,dst) 去重。

    去重是必须的：分片后一条边的两端可能分属两个分片，会被查出来两次。
    """
    seen: dict[tuple, dict] = {}
    for chunk in _chunks(keys):
        ph = ",".join("?" * len(chunk))
        for r in c.execute(
                f"""SELECT src_key, dst_key FROM citation_edges
                    WHERE src_key IN ({ph}) OR dst_key IN ({ph})""",
                tuple(chunk) * 2):
            seen[(r["src_key"], r["dst_key"])] = {"src_key": r["src_key"],
                                                  "dst_key": r["dst_key"]}
    return list(seen.values())


def _meta(c, keys) -> dict[str, dict]:
    """一批 norm_key 的展示元数据：库内优先（papers），库外退到 citation_nodes。

    拿不到标题就是 None——上层照实显示「未知标题」，不要在这里编。
    """
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}
    out: dict[str, dict] = {}
    for r in _rows_in(c, """SELECT id, norm_key, title, year, authors, doi, arxiv_id,
                                   citation_count, level
                            FROM papers WHERE norm_key IN ({ph})""", keys):
        out[r["norm_key"]] = {
            "norm_key": r["norm_key"], "title": r["title"], "year": r["year"],
            "authors": _loads_list(r["authors"]), "doi": r["doi"],
            "arxiv_id": r["arxiv_id"], "citation_count": r["citation_count"],
            "in_library": True, "paper_id": r["id"], "level": r["level"],
        }
    missing = [k for k in keys if k not in out]
    for r in _rows_in(c, """SELECT norm_key, title, year, authors_json, doi, arxiv_id,
                                   citation_count
                            FROM citation_nodes WHERE norm_key IN ({ph})""", missing):
        out[r["norm_key"]] = {
            "norm_key": r["norm_key"], "title": r["title"], "year": r["year"],
            "authors": _loads_list(r["authors_json"]), "doi": r["doi"],
            "arxiv_id": r["arxiv_id"], "citation_count": r["citation_count"],
            "in_library": False, "paper_id": None, "level": None,
        }
    for k in keys:  # 边指向的论文连缓存都没有：如实给一个空壳，不丢节点
        out.setdefault(k, {"norm_key": k, "title": None, "year": None, "authors": [],
                           "doi": None, "arxiv_id": None, "citation_count": None,
                           "in_library": False, "paper_id": None, "level": None})
    return out


def _loads_list(s) -> list:
    try:
        v = json.loads(s or "[]")
    except (TypeError, ValueError):
        return []
    return v if isinstance(v, list) else []


def _sort_key(m: dict) -> tuple:
    """节点排序：库内优先 → 被引多优先 → 键序（保证确定性）。"""
    return (0 if m.get("in_library") else 1, -(m.get("citation_count") or 0),
            m.get("norm_key") or "")


# ── 抓取（Semantic Scholar Graph API）──

def _jitter() -> float:
    import random
    return 0.8 + random.random() * 0.4


def _sleep(sec: float):
    time.sleep(sec)


RETRY_DELAYS = (20, 45, 90, 150)


def _backoff(attempt: int):
    """第 attempt 次失败后的退避。最后一次尝试之后**不睡**——后面已经没有下一次
    请求了，睡满这一觉只是把调用方（和 UI 上的这次点击）白白多锁 150 秒；
    `fetch_edges(direction="both")` 撞上彻底挂掉的 S2 会因此阻塞 10 分钟。
    重试之间的间隔仍是 20/45/90s，与 sources/semantic_scholar.py 同一套序列。
    """
    if attempt < len(RETRY_DELAYS) - 1:
        _sleep(RETRY_DELAYS[attempt] * _jitter())


def _get_json(url: str, params: dict) -> dict:
    """退避重试口径与 sources/semantic_scholar.py 一致：
    429/5xx 退避 20/45/90s（±20% 抖动，共 4 次尝试），400 立即失败。

    基数取 10s 级是因为无 key 共享池按 5 分钟窗口限流，秒级退避只会全部
    落回同一个封锁窗口。
    """
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    last = None
    with http.client() as client:
        for attempt in range(len(RETRY_DELAYS)):
            try:
                r = client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                _backoff(attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                _backoff(attempt)
                continue
            if r.status_code == 400:
                raise GraphError(f"Semantic Scholar 请求被拒（400）：{r.text[:200]}")
            if r.status_code == 404:
                raise GraphError(f"Semantic Scholar 查无此论文（404）：{url}")
            r.raise_for_status()
            # 网关插页 / HTML 错误页会带着 200 回来，r.json() 抛的是 ValueError，
            # 既不是 httpx.HTTPError 也不是 GraphError——不在这儿收口的话，
            # 它会一路穿出 fetch_edges，把另一个方向一起掀翻。
            try:
                payload = r.json()
            except ValueError as e:
                raise GraphError(f"Semantic Scholar 返回的不是 JSON"
                                 f"（HTTP {r.status_code}）：{(r.text or '')[:200]}") from e
            if not isinstance(payload, dict):
                raise GraphError(f"Semantic Scholar 返回了 {type(payload).__name__} "
                                 f"而不是 JSON 对象，结构不认识就不猜")
            return payload
    raise RuntimeError(f"Semantic Scholar 连续 {len(RETRY_DELAYS)} 次失败：{last}")


def s2_ref_id(row) -> str | None:
    """S2 接受的论文标识：优先库内已存的 paperId，其次 DOI、arXiv。

    只有标题、没有任何外部标识的论文（如手动上传的 PDF）拉不了引文——
    如实返回 None，让调用方报错，而不是拿标题去瞎搜一个可能不是它的 id。

    DOI 必须先剥掉 `https://doi.org/` 前缀：OpenAlex 返回的就是完整 URL 形式，
    库里 455 篇有一大半是这么存的。直接拼成 `DOI:https://doi.org/10.x` 发过去，
    S2 一律 404——线上实测踩到，而单测用的是合成的裸 DOI，一个都没照出来。
    """
    if row["s2_id"]:
        return str(row["s2_id"])
    if row["doi"]:
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "",
                     str(row["doi"]).strip(), flags=re.I)
        if doi:
            return "DOI:" + doi
    if row["arxiv_id"]:
        aid = re.sub(r"^arxiv:", "", str(row["arxiv_id"]).strip(), flags=re.I)
        if aid:
            return "arXiv:" + aid
    return None


def _node_from_s2(p: dict | None) -> dict | None:
    """S2 的一条 citedPaper/citingPaper → 节点字典。算不出 norm_key 的丢弃。"""
    if not isinstance(p, dict):
        return None
    ext = p.get("externalIds") or {}
    key = norm_key(doi=ext.get("DOI"), arxiv_id=ext.get("ArXiv"), title=p.get("title"))
    if not key:
        return None
    return {
        "norm_key": key,
        "title": (p.get("title") or "").strip() or None,
        "year": p.get("year"),
        "authors": [a.get("name") for a in p.get("authors") or []
                    if isinstance(a, dict) and a.get("name")],
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "citation_count": p.get("citationCount"),
        "s2_id": p.get("paperId"),
    }


def _upsert_node(c, n: dict):
    """库外节点元数据缓存。COALESCE 保证后来的空值不会把先前拿到的字段抹掉。"""
    c.execute(
        """INSERT INTO citation_nodes(norm_key,title,year,authors_json,doi,arxiv_id,
                                      citation_count,s2_id)
           VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(norm_key) DO UPDATE SET
             title=COALESCE(excluded.title, title),
             year=COALESCE(excluded.year, year),
             authors_json=CASE WHEN excluded.authors_json='[]'
                               THEN authors_json ELSE excluded.authors_json END,
             doi=COALESCE(excluded.doi, doi),
             arxiv_id=COALESCE(excluded.arxiv_id, arxiv_id),
             citation_count=COALESCE(excluded.citation_count, citation_count),
             s2_id=COALESCE(excluded.s2_id, s2_id),
             updated_at=datetime('now','localtime')""",
        (n["norm_key"], n.get("title"), n.get("year"),
         json.dumps(n.get("authors") or [], ensure_ascii=False),
         n.get("doi"), n.get("arxiv_id"), n.get("citation_count"), n.get("s2_id")))


def _upsert_edge(c, src: str, dst: str) -> bool:
    """写一条边，返回是否为新增。src 引用 dst。paper_id 现查现填。"""
    exists = c.execute("SELECT 1 FROM citation_edges WHERE src_key=? AND dst_key=?",
                       (src, dst)).fetchone() is not None
    c.execute(
        """INSERT INTO citation_edges(src_key,dst_key,src_paper_id,dst_paper_id)
           VALUES(?,?,(SELECT id FROM papers WHERE norm_key=?),
                      (SELECT id FROM papers WHERE norm_key=?))
           ON CONFLICT(src_key,dst_key) DO UPDATE SET
             fetched_at=datetime('now','localtime'),
             src_paper_id=COALESCE(excluded.src_paper_id, src_paper_id),
             dst_paper_id=COALESCE(excluded.dst_paper_id, dst_paper_id)""",
        (src, dst, src, dst))
    return not exists


def ingest_local_references(paper_id: int) -> dict:
    """从本地 PDF 的参考文献段抽 DOI/arXiv，直接建出边——不联网、0 token。

    补的是这么一个洞：手动上传的 PDF 常常既没有 DOI 也没有 s2_id，`fetch_edges`
    对它无从下手（拿标题去瞎搜可能搜到别的论文），于是它永远进不了引文图。
    但它自己的参考文献段里就写着几百个 DOI 与 arXiv 编号，那是**它引了谁**的
    一手证据，比任何检索都可靠。
    """
    ensure_schema()
    from . import structure
    with db.conn() as c:
        row = c.execute("SELECT norm_key, pdf_path FROM papers WHERE id=?",
                        (paper_id,)).fetchone()
    if not row:
        raise GraphError(f"论文 {paper_id} 不存在")
    if not row["pdf_path"] or not Path(row["pdf_path"]).is_file():
        raise GraphError("这篇论文没有可读的本地 PDF（先 read 或 import-pdf）")

    parsed = structure.extract_references(row["pdf_path"])
    src = row["norm_key"]
    added = skipped = 0
    with db.conn() as c:
        for e in parsed.get("entries") or []:
            key = norm_key(doi=e.get("doi"), arxiv_id=e.get("arxiv_id"))
            if not key or key == src:      # 没有外部标识的条目不入图（标题匹配太不可靠）
                skipped += 1
                continue
            _upsert_node(c, {"norm_key": key, "title": e.get("title_guess"),
                             "year": e.get("year"), "authors": [],
                             "doi": e.get("doi"), "arxiv_id": e.get("arxiv_id"),
                             "citation_count": None, "s2_id": None})
            added += 1 if _upsert_edge(c, src, key) else 0
        c.execute("""INSERT INTO citation_fetch(paper_key,direction,fetched_at)
                     VALUES(?,'references',datetime('now','localtime'))
                     ON CONFLICT(paper_key,direction) DO UPDATE SET
                       fetched_at=datetime('now','localtime')""", (src,))
    return {"paper_id": paper_id, "norm_key": src,
            "n_entries": parsed.get("n_entries", 0),
            "with_id": parsed.get("with_doi", 0) + parsed.get("with_arxiv", 0),
            "edges_added": added, "skipped_no_id": skipped,
            "note": "只收参考文献里带 DOI/arXiv 的条目；纯文本条目不猜标题、不入图"}


def _fetch_age_days(c, key: str, direction: str) -> float | None:
    r = c.execute("""SELECT julianday('now','localtime') - julianday(fetched_at) age
                     FROM citation_fetch WHERE paper_key=? AND direction=?""",
                  (key, direction)).fetchone()
    return None if r is None or r["age"] is None else float(r["age"])


def fetch_edges(paper_id: int, direction: str = "both", limit: int = 100,
                max_age_days: int = 30) -> dict:
    """拉某篇论文的 references（它引了谁）/ citations（谁引了它），落库。

    `max_age_days=0` 表示强制重拉。单个方向失败不会掀翻另一个方向——失败原因
    如实写进返回的 `directions[x]["error"]`，不静默吞。
    """
    ensure_schema()
    dirs = DIRECTIONS if direction == "both" else (direction,)
    for d in dirs:
        if d not in DIRECTIONS:
            raise GraphError(f"direction 只能是 both/references/citations，收到 {direction!r}")
    # 参数在发第一个请求前就校验干净：原来 int(limit) 埋在 _fetch_one 的 try 之外，
    # limit="abc" 会以 ValueError 穿出去（而不是本模块约定的 GraphError），
    # 而且是在第一个方向已经打过网络之后才炸。
    try:
        limit = max(1, min(int(limit), S2_MAX_LIMIT))
    except (TypeError, ValueError):
        raise GraphError(f"limit 必须是整数，收到 {limit!r}") from None

    with db.conn() as c:
        row = _paper_row(c, paper_id)
        center = row["norm_key"]
        ref_id = s2_ref_id(row)
        pending = {}
        for d in dirs:
            age = None if max_age_days <= 0 else _fetch_age_days(c, center, d)
            pending[d] = age if (age is not None and age < max_age_days) else None

    if ref_id is None:
        raise GraphError(
            f"论文 {paper_id}《{row['title']}》没有 DOI / arXiv ID / s2_id，"
            f"无法向 Semantic Scholar 定位，拉不了引文")

    out = {"paper_id": paper_id, "norm_key": center, "ref_id": ref_id,
           "directions": {}, "edges_added": 0, "nodes_seen": 0, "ok": True}

    for d in dirs:
        if pending[d] is not None:
            out["directions"][d] = {"status": "skipped", "age_days": round(pending[d], 3),
                                    "reason": f"{max_age_days} 天内已拉取过",
                                    "fetched": 0, "edges_added": 0, "self_loops": 0,
                                    "no_key": 0, "error": None}
            continue
        out["directions"][d] = _fetch_one(center, ref_id, d, limit)
        st = out["directions"][d]
        out["edges_added"] += st["edges_added"]
        out["nodes_seen"] += st["fetched"]
        if st["status"] == "failed":
            out["ok"] = False
    return out


def _fetch_one(center: str, ref_id: str, direction: str, limit: int) -> dict:
    st = {"status": "fetched", "fetched": 0, "edges_added": 0, "self_loops": 0,
          "no_key": 0, "error": None, "age_days": None}
    url = f"{S2_BASE}/{ref_id}/{direction}"
    params = {"fields": S2_FIELDS, "limit": limit}
    try:
        payload = _get_json(url, params)
    except (GraphError, RuntimeError, httpx.HTTPError) as e:
        st["status"] = "failed"
        st["error"] = f"{type(e).__name__}: {e}"
        return st

    inner = "citedPaper" if direction == "references" else "citingPaper"
    items = payload.get("data") or []
    if not isinstance(items, list):
        # data 不是数组就是我们不认识的格式——如实记为失败，不去猜它的结构
        st["status"] = "failed"
        st["error"] = (f"Semantic Scholar 的 data 字段是 {type(items).__name__}，"
                       f"不是数组")
        return st
    with db.conn() as c:
        for it in items:
            node = _node_from_s2((it or {}).get(inner))
            if node is None:
                st["no_key"] += 1
                continue
            st["fetched"] += 1
            if node["norm_key"] == center:
                st["self_loops"] += 1   # 自引边拒绝写入（S2 偶发，也可能是 norm_key 撞了）
                continue
            _upsert_node(c, node)
            src, dst = ((center, node["norm_key"]) if direction == "references"
                        else (node["norm_key"], center))
            if _upsert_edge(c, src, dst):
                st["edges_added"] += 1
        c.execute("""INSERT INTO citation_fetch(paper_key,direction,n_edges)
                     VALUES(?,?,?)
                     ON CONFLICT(paper_key,direction) DO UPDATE SET
                       fetched_at=datetime('now','localtime'),
                       n_edges=excluded.n_edges""",
                  (center, direction, st["fetched"] - st["self_loops"]))
    return st


def _paper_row(c, paper_id: int):
    row = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if row is None:
        raise GraphError(f"论文 id={paper_id} 不在库里")
    return row


# ── 读图 ──

def neighbors(paper_id: int, depth: int = 1, limit: int = 50) -> dict:
    """以某论文为中心的子图。纯读库、不联网。

    `in_library` 实时 JOIN papers 判定，所以一篇原本在库外的邻居只要日后被检索
    入库（norm_key 对上号），这里立刻就认出来，不需要任何回填动作。
    """
    ensure_schema()
    depth = max(1, min(int(depth), MAX_DEPTH))
    limit = max(1, int(limit))
    with db.conn() as c:
        center = _paper_row(c, paper_id)["norm_key"]
        seen = {center: 0}
        frontier = [center]
        truncated = False
        for d in range(1, depth + 1):
            if not frontier:
                break
            rows = _touching(c, frontier)
            fresh = {k for r in rows for k in (r["src_key"], r["dst_key"])
                     if k not in seen}
            if not fresh:
                break
            # 名额用光但邻居还没展开完 → 如实标 truncated。
            # 原来这个分支写在循环开头的 `len(seen) >= limit` 里直接 break，
            # 上一跳「刚好」填满 limit 时会漏掉整整一跳的节点却报 truncated=False，
            # UI 会当成完整子图画出来。宁可多查一次 _touching 也不能报假话。
            room = limit - len(seen)
            if room <= 0:
                truncated = True
                break
            meta = _meta(c, fresh)
            ordered = sorted(fresh, key=lambda k: _sort_key(meta[k]))
            if len(ordered) > room:
                truncated = True
                ordered = ordered[:room]
            for k in ordered:
                seen[k] = d
            frontier = ordered

        keys = list(seen)
        kset = set(keys)
        meta = _meta(c, keys)
        # 只保留两端都在子图里的边。用「碰到任一端」的结果再过滤，避免
        # 分片的 IN…AND IN… 漏掉跨分片的边（两端落在不同分片就查不出来了）。
        edge_rows = [r for r in _touching(c, keys)
                     if r["src_key"] in kset and r["dst_key"] in kset]

    nodes = []
    for k in keys:
        m = dict(meta[k])
        m["depth"] = seen[k]
        m["is_center"] = (k == center)
        nodes.append(m)
    nodes.sort(key=lambda m: (m["depth"], _sort_key(m)))
    edges = [{"src": r["src_key"], "dst": r["dst_key"]} for r in edge_rows]
    edges.sort(key=lambda e: (e["src"], e["dst"]))
    return {"center": center, "center_paper_id": paper_id, "depth": depth,
            "nodes": nodes, "edges": edges, "truncated": truncated,
            "node_count": len(nodes), "edge_count": len(edges)}


# ── 共被引 / 文献耦合 ──

def related(paper_ids: list[int], top_k: int = 20) -> list[dict]:
    """相关论文发现：共被引 + 文献耦合，Jaccard 归一后加权合并。

    - 共被引 co-citation：候选与种子被同一批论文引用（后人把它们摆在一起）；
    - 文献耦合 bibliographic coupling：候选与种子引用了同一批论文（站在同一批肩膀上）。

    多个种子时按种子数取**均值**（不是取最大）：只跟其中一篇沾边的候选分数会被稀释，
    这正是我们想要的——「跟我这批论文整体相关」比「跟某一篇特别像」更有价值。

    纯 SQL + Python，不联网、不调 LLM。
    """
    ensure_schema()
    top_k = int(top_k)
    if not paper_ids or top_k <= 0:
        return []
    with db.conn() as c:
        seed_keys, missing = [], []
        for pid in paper_ids:
            r = c.execute("SELECT norm_key FROM papers WHERE id=?", (pid,)).fetchone()
            (seed_keys.append(r["norm_key"]) if r else missing.append(pid))
        if missing:
            raise GraphError(f"论文 id 不在库里：{missing}")
        seed_set = set(seed_keys)
        n_seeds = len(seed_keys)

        acc: dict[str, dict] = {}

        def bucket(k: str) -> dict:
            return acc.setdefault(k, {"co_inter": 0, "bc_inter": 0, "co_j": 0.0,
                                      "bc_j": 0.0, "co_seeds": 0, "bc_seeds": 0})

        for key in seed_keys:
            citers = [r["src_key"] for r in c.execute(
                "SELECT src_key FROM citation_edges WHERE dst_key=?", (key,))]
            refs = [r["dst_key"] for r in c.execute(
                "SELECT dst_key FROM citation_edges WHERE src_key=?", (key,))]

            # 共被引：引用了 key 的那批论文，还引用了谁
            if citers:
                inter = _grouped(c, """SELECT dst_key, COUNT(DISTINCT src_key)
                                       FROM citation_edges WHERE src_key IN ({ph})
                                       GROUP BY dst_key""", citers)
                inter.pop(key, None)
                cands = [k for k in inter if k not in seed_set]
                indeg = _grouped(c, """SELECT dst_key, COUNT(DISTINCT src_key)
                                       FROM citation_edges WHERE dst_key IN ({ph})
                                       GROUP BY dst_key""", cands)
                for k in cands:
                    i = inter[k]
                    union = len(set(citers)) + indeg.get(k, i) - i
                    b = bucket(k)
                    b["co_inter"] += i
                    b["co_j"] += i / union if union else 0.0
                    b["co_seeds"] += 1

            # 文献耦合：引用了 key 的参考文献的那批论文
            if refs:
                inter = _grouped(c, """SELECT src_key, COUNT(DISTINCT dst_key)
                                       FROM citation_edges WHERE dst_key IN ({ph})
                                       GROUP BY src_key""", refs)
                inter.pop(key, None)
                cands = [k for k in inter if k not in seed_set]
                outdeg = _grouped(c, """SELECT src_key, COUNT(DISTINCT dst_key)
                                        FROM citation_edges WHERE src_key IN ({ph})
                                        GROUP BY src_key""", cands)
                for k in cands:
                    i = inter[k]
                    union = len(set(refs)) + outdeg.get(k, i) - i
                    b = bucket(k)
                    b["bc_inter"] += i
                    b["bc_j"] += i / union if union else 0.0
                    b["bc_seeds"] += 1

        meta = _meta(c, acc)

    out = []
    for k, b in acc.items():
        score = (CO_WEIGHT * b["co_j"] + BC_WEIGHT * b["bc_j"]) / n_seeds
        if score <= 0:
            continue
        m = dict(meta[k])
        m.update({
            "score": round(score, 6),
            "cocitation": b["co_inter"], "coupling": b["bc_inter"],
            "cocitation_score": round(CO_WEIGHT * b["co_j"] / n_seeds, 6),
            "coupling_score": round(BC_WEIGHT * b["bc_j"] / n_seeds, 6),
            "reason": _reason(b, n_seeds),
        })
        out.append(m)
    out.sort(key=lambda m: (-m["score"], -(m["cocitation"] + m["coupling"]),
                            m["norm_key"]))
    return out[:top_k]


def _reason(b: dict, n_seeds: int) -> str:
    """如实说明命中来源与条数，不写「高度相关」这种无凭据的话。"""
    bits = []
    if b["co_inter"]:
        bits.append(f"共被引 {b['co_inter']} 次"
                    + (f"（与 {b['co_seeds']}/{n_seeds} 篇种子共享施引论文）"
                       if n_seeds > 1 else "（与种子共享施引论文）"))
    if b["bc_inter"]:
        bits.append(f"文献耦合 {b['bc_inter']} 次"
                    + (f"（与 {b['bc_seeds']}/{n_seeds} 篇种子引用了相同文献）"
                       if n_seeds > 1 else "（与种子引用了相同文献）"))
    return "；".join(bits) or "无命中"


# ── 阅读缺口 ──

def gap_papers(top_k: int = 20) -> list[dict]:
    """该补的文献：被库内论文引用最多、自己却不在库里的那些。

    `cited_by_count` 是「库内有几篇引了它」，`cited_by_titles` 给出具体是哪几篇——
    用户一眼能看出为什么该补，而不是只看到一个分数。
    """
    ensure_schema()
    # top_k<=0 直接空手而归。注意别把负数交给 SQL：SQLite 的 `LIMIT -1` 是
    # 「不限」，一个手滑的 -1 会把整张缺口表拖出来。
    top_k = int(top_k)
    if top_k <= 0:
        return []
    with db.conn() as c:
        rows = c.execute(
            """SELECT e.dst_key key, COUNT(DISTINCT e.src_key) n
               FROM citation_edges e
               JOIN papers sp ON sp.norm_key = e.src_key
               LEFT JOIN papers dp ON dp.norm_key = e.dst_key
               WHERE dp.id IS NULL
               GROUP BY e.dst_key
               ORDER BY n DESC, e.dst_key ASC
               LIMIT ?""", (top_k,)).fetchall()
        keys = [r["key"] for r in rows]
        meta = _meta(c, keys)
        titles: dict[str, list[str]] = {}
        for k in keys:
            titles[k] = [r["title"] for r in c.execute(
                """SELECT sp.title FROM citation_edges e
                   JOIN papers sp ON sp.norm_key = e.src_key
                   WHERE e.dst_key=? ORDER BY sp.id LIMIT 5""", (k,))]
    out = []
    for r in rows:
        m = dict(meta[r["key"]])
        m.update({"cited_by_count": r["n"], "cited_by_titles": titles[r["key"]],
                  "in_library": False, "paper_id": None,
                  "has_metadata": m["title"] is not None})
        out.append(m)
    return out


# ── 一键入库 ──

def adopt(norm_key_str: str) -> dict:
    """把图里的库外论文用 citation_nodes 的缓存元数据建成 L0 记录，并回填边表 id。

    缓存里没有摘要（S2 的 references/citations 端点不返回），所以入库就是 L0：
    level=0、abstract 为空。要卡片得再走一次正常的补全流程——这里绝不编摘要。
    """
    ensure_schema()
    with db.conn() as c:
        exist = db.get_by_norm_key(c, norm_key_str)
        if exist:
            n = _backfill(c, norm_key_str, exist["id"])
            return {"status": "exists", "paper_id": exist["id"],
                    "norm_key": norm_key_str, "title": exist["title"],
                    "edges_backfilled": n,
                    "note": "该论文已在库中，只回填了边表的 paper_id"}
        node = c.execute("SELECT * FROM citation_nodes WHERE norm_key=?",
                         (norm_key_str,)).fetchone()
        if node is None:
            raise GraphError(f"citation_nodes 里没有 {norm_key_str}，无法入库"
                             f"（该节点可能只作为边的端点出现过，没有元数据）")
        if not node["title"]:
            raise GraphError(f"{norm_key_str} 的缓存元数据没有标题，拒绝入库"
                             f"（不给论文编标题）")
        pid = db.insert_l0(c, {
            "norm_key": norm_key_str, "title": node["title"],
            "abstract": None, "year": node["year"], "venue": None,
            "authors": _loads_list(node["authors_json"]),
            "doi": node["doi"], "arxiv_id": node["arxiv_id"],
            "citation_count": node["citation_count"], "oa_pdf_url": None,
            "source": "graph", "s2_id": node["s2_id"],
        })
        n = _backfill(c, norm_key_str, pid)
    return {"status": "adopted", "paper_id": pid, "norm_key": norm_key_str,
            "title": node["title"], "edges_backfilled": n, "level": 0,
            "note": "引文端点只带元数据，无摘要，入库为 L0"}


def _backfill(c, key: str, pid: int) -> int:
    a = c.execute("UPDATE citation_edges SET src_paper_id=? WHERE src_key=? AND "
                  "(src_paper_id IS NULL OR src_paper_id<>?)", (pid, key, pid)).rowcount
    b = c.execute("UPDATE citation_edges SET dst_paper_id=? WHERE dst_key=? AND "
                  "(dst_paper_id IS NULL OR dst_paper_id<>?)", (pid, key, pid)).rowcount
    return (a or 0) + (b or 0)


def backfill_ids() -> int:
    """全量回填边表的冗余 paper_id 列（正常检索入库后可以跑一次，纯优化）。

    不跑也不影响任何查询结果——所有 in_library 判定都实时 JOIN papers。
    """
    ensure_schema()
    with db.conn() as c:
        a = c.execute("""UPDATE citation_edges SET src_paper_id=
                         (SELECT id FROM papers WHERE norm_key=src_key)
                         WHERE src_paper_id IS NULL AND EXISTS
                         (SELECT 1 FROM papers WHERE norm_key=src_key)""").rowcount
        b = c.execute("""UPDATE citation_edges SET dst_paper_id=
                         (SELECT id FROM papers WHERE norm_key=dst_key)
                         WHERE dst_paper_id IS NULL AND EXISTS
                         (SELECT 1 FROM papers WHERE norm_key=dst_key)""").rowcount
    return (a or 0) + (b or 0)


# ── 概览 ──

def stats() -> dict:
    """供 UI 展示的图规模概览。"""
    ensure_schema()
    with db.conn() as c:
        edges = c.execute("SELECT COUNT(*) n FROM citation_edges").fetchone()["n"]
        covered = c.execute(
            """SELECT COUNT(*) n FROM papers p WHERE EXISTS
               (SELECT 1 FROM citation_edges e
                WHERE e.src_key=p.norm_key OR e.dst_key=p.norm_key)""").fetchone()["n"]
        papers_total = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
        external = c.execute(
            """SELECT COUNT(*) n FROM citation_nodes cn WHERE NOT EXISTS
               (SELECT 1 FROM papers p WHERE p.norm_key=cn.norm_key)""").fetchone()["n"]
        gaps = c.execute(
            """SELECT COUNT(*) n FROM (
                 SELECT e.dst_key FROM citation_edges e
                 JOIN papers sp ON sp.norm_key=e.src_key
                 LEFT JOIN papers dp ON dp.norm_key=e.dst_key
                 WHERE dp.id IS NULL GROUP BY e.dst_key)""").fetchone()["n"]
        fetched = c.execute(
            "SELECT COUNT(DISTINCT paper_key) n FROM citation_fetch").fetchone()["n"]
    return {"edges": edges, "papers_total": papers_total,
            "papers_with_edges": covered, "papers_fetched": fetched,
            "external_nodes": external, "gap_papers": gaps,
            "coverage": round(covered / papers_total, 4) if papers_total else 0.0}
