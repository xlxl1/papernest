"""Embedding 服务（OpenAI 兼容 /embeddings）与向量存取。

- 向量按 (paper_id, kind, model) 存 SQLite BLOB，numpy 余弦检索。
  **不上向量库是量出来的决定，不是偷懒**（1024 维实测，p50）：

  | 向量条数 | 原实现（每次重读 BLOB 逐行算） | 缓存归一化矩阵 | 常驻内存 |
  |----------|-------------------------------|----------------|----------|
  | 500      | 9.6 ms                        | 0.19 ms        | 2 MB     |
  | 10,000   | 203.4 ms                      | 1.80 ms        | 39 MB    |
  | 50,000   | 1028.6 ms                     | 8.87 ms        | 195 MB   |
  | 200,000  | 4743.2 ms                     | 41.74 ms       | 781 MB   |

  瓶颈从来不是余弦计算（纯矩阵乘 0.04ms），而是**每次查询重读 SQLite BLOB**——
  只做向量化仅快 1.3 倍，缓存矩阵快 50~113 倍。个人库全量嵌入约 5~6k 条向量，
  缓存后 1ms 级；即使 10 倍规模也只有 9ms。**真到 10 万条以上再换 sqlite-vec**
  （同一个单文件、不引服务进程），在那之前引向量库只是多一个依赖和一份数据副本。
- 未配置 EMBED_MODEL 时 available()=False，检索走 FTS/词面重叠兜底。
"""
import json
import os
import threading
from typing import NamedTuple

import numpy as np

from . import config, db, degrade, http, llm


class EmbedUnavailable(Exception):
    pass


def available() -> bool:
    return bool(config.EMBED_API_KEY and config.EMBED_MODEL)


def embed_texts(texts: list[str], purpose: str = "embed") -> list[list[float]]:
    if not available():
        raise EmbedUnavailable("未配置 LLM_API_KEY / EMBED_MODEL")
    import time
    out: list[list[float]] = []
    url = config.EMBED_API_BASE.rstrip("/") + "/embeddings"
    for i in range(0, len(texts), 16):  # 分批，规避单请求条数上限
        t0 = time.perf_counter()
        batch = [t[:6000] for t in texts[i:i + 16]]
        with http.client(timeout=60) as client:
            r = client.post(url, json={"model": config.EMBED_MODEL, "input": batch},
                            headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"})
        if r.status_code != 200:
            raise EmbedUnavailable(f"embeddings 接口失败 HTTP {r.status_code}：{r.text[:200]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        usage = r.json().get("usage") or {}
        with db.conn() as c:
            db.add_llm_call(c, purpose, None, usage.get("prompt_tokens", 0), 0,
                            config.EMBED_MODEL,
                            latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            c.commit()
        out.extend(d["embedding"] for d in data)
    return out


def _to_blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def index_paper(paper_id: int, title: str, abstract: str) -> int:
    """为论文建 paper 级与句子级向量（引用推荐的证据句检索用）。返回向量条数。

    清理只针对本函数自己写的两种 kind。**不能按 (paper_id, model) 一刀切**：
    chunk 向量是 `chunkembed` 另一条路径建的、本函数不会重建，一起删掉就是净丢数据。
    而 `cli.py embed` / `jobs.py` 的选片条件都是「没有 kind='paper' 向量的论文」——
    正文先入库、paper 级向量还没补的论文必然落进这个集合，于是「补建 paper 向量」
    这个动作会顺手抹掉该篇全部 chunk 向量，且没有任何报错或日志。
    同仓库另外两处删向量（chunkembed.py、db.replace_chunks）都是按 kind 收窄的。
    """
    text = f"{title}\n{abstract or ''}".strip()
    sentences = split_sentences(abstract or "")
    inputs = [text] + sentences
    vecs = embed_texts(inputs)
    with db.conn() as c:
        c.execute("DELETE FROM vectors WHERE paper_id=? AND model=? "
                  "AND kind IN ('paper','sent')",
                  (paper_id, config.EMBED_MODEL))
        c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) VALUES(?,?,?,?,?,?)",
                  (paper_id, "paper", 0, text[:400], config.EMBED_MODEL, _to_blob(vecs[0])))
        for j, (s, v) in enumerate(zip(sentences, vecs[1:])):
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) VALUES(?,?,?,?,?,?)",
                      (paper_id, "sent", j, s, config.EMBED_MODEL, _to_blob(v)))
        c.commit()
    invalidate_cache()
    return len(vecs)


_MATRIX_LOCK = threading.Lock()
_MATRIX: dict | None = None      # {"key": ..., "ids": ndarray, "M": 归一化矩阵}


def _matrix_key(c) -> tuple:
    """缓存有效性指纹。(条数, 最大 id) 覆盖本项目所有真实变更路径：
    `index_paper` 是先删后插（autoincrement 让 max(id) 增长）、
    换嵌入模型是 `UPDATE ... SET model=?`（改变该 model 的条数）。
    两个都走索引，成本亚毫秒——比每次重读几百 MB BLOB 便宜四个数量级。"""
    r = c.execute("""SELECT COUNT(1) n, COALESCE(MAX(id),0) m FROM vectors
                     WHERE kind IN ('paper','chunk') AND model=?""",
                  (config.EMBED_MODEL,)).fetchone()
    return (config.EMBED_MODEL, r["n"], r["m"])


def invalidate_cache():
    """向量变更后显式作废（index_paper 会调）。指纹检查已能兜住，这是双保险。"""
    global _MATRIX
    with _MATRIX_LOCK:
        _MATRIX = None


def _paper_matrix():
    """常驻内存的归一化向量矩阵（**paper 级 + chunk 级**）。返回 (ids, M) 或 (None, None)。

    一行一条向量，`ids` 是对应的 paper_id——同一篇论文会出现多行（1 条标题摘要向量
    + N 条章节块向量），调用方按 paper_id 去重取最好的一条即可。

    **为什么必须带 chunk**：只装 kind='paper' 时，语义检索只覆盖标题与摘要，
    正文内容只能靠 chunks_fts 的**关键词**命中——而向量本来就是用来解决
    「中文问、英文正文，词面对不上」这类问题的。实测踩到过：784 条 chunk 向量
    建好之后消融显示贡献恰好 +0.0000（自建集与 QASPER 都是），一查才发现
    向量路根本没读它们——**不是没效果，是没接上**。

    为什么不上向量库：实测 1024 维下**瓶颈是每次查询重读 SQLite BLOB，不是余弦计算**——
    向量化只快 1.3 倍，缓存矩阵快 50~113 倍。缓存之后 1 万条 1.8ms、5 万条 8.9ms、
    20 万条 41.7ms（781MB 内存），个人库场景（全库嵌入约 5~6k 条向量）远没到需要
    ANN 索引的规模。真到 10 万条以上再换 sqlite-vec——同一个单文件、不引服务进程。
    """
    global _MATRIX
    with db.conn() as c:
        key = _matrix_key(c)
        with _MATRIX_LOCK:
            if _MATRIX is not None and _MATRIX["key"] == key:
                return _MATRIX["ids"], _MATRIX["idxs"], _MATRIX["M"]
        rows = c.execute("""SELECT paper_id, kind, idx, vec FROM vectors
                            WHERE kind IN ('paper','chunk') AND model=?
                            ORDER BY paper_id, kind, idx""",
                         (config.EMBED_MODEL,)).fetchall()
    if not rows:
        with _MATRIX_LOCK:
            _MATRIX = {"key": key, "ids": None, "idxs": None, "M": None}
        return None, None, None
    ids = np.fromiter((r["paper_id"] for r in rows), dtype=np.int64, count=len(rows))
    # 每行是哪个 chunk：chunk 行记 chunk_no，paper 行记 -1。
    # **不带上它，语义命中就只能压成 paper_id**——上下文组装侧再也拿不回
    # 「是哪一段匹配上的」，只能退回按页词频重新猜一遍。
    idxs = np.fromiter((r["idx"] if r["kind"] == "chunk" else -1 for r in rows),
                       dtype=np.int64, count=len(rows))
    raw = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32)
    dim = raw.size // len(rows)
    M = raw.reshape(len(rows), dim)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)   # 预归一化：查询期只剩一次矩阵乘
    with _MATRIX_LOCK:
        _MATRIX = {"key": key, "ids": ids, "idxs": idxs, "M": M}
    return ids, idxs, M


def search_papers(query: str, top_k: int = 8) -> list[dict]:
    """论文级语义检索。返回 [{paper_id, score}]，按余弦相似度降序，全序。

    矩阵里一篇论文有多行（标题摘要向量 + 各章节块向量），这里**按 paper_id 取最高分
    那一行**再排序——不去重的话 top-k 会被章节多的那一两篇论文占满，
    「检索到 5 篇」实际只有 1 篇。
    """
    qv = np.asarray(embed_texts([query])[0], dtype=np.float32)
    ids, idxs, M = _paper_matrix()
    if ids is None:
        return []
    if M.shape[1] != qv.size:
        # 换了嵌入模型但库里还是旧维度：如实报错，不做静默截断（截断出来的
        # 余弦是没有意义的数，比报错更危险）。`cli.py embed` 负责迁移或重建。
        raise EmbedUnavailable(
            f"向量维度不一致：库内 {M.shape[1]} 维、当前模型 {qv.size} 维，"
            f"请先跑 `python cli.py embed` 迁移或重建索引")
    scores = M @ (qv / (np.linalg.norm(qv) + 1e-9))
    best: dict[int, float] = {}
    best_chunk: dict[int, int] = {}
    for pid, cno, sc in zip(ids.tolist(), idxs.tolist(), scores.tolist()):
        if sc > best.get(pid, -2.0):
            best[pid] = sc
            # 记下是**哪一段**赢的：-1 表示赢在标题摘要向量上，没有对应的正文块。
            # 这条信息是「语义命中能不能进上下文」的全部依据——中文问句 × 英文正文
            # 时词面命中恒为 0，只有向量这一路知道该把哪段喂给模型。
            best_chunk[pid] = cno
    # 全序：分数并列时按 paper_id 决胜，否则跨进程结果会漂移
    order = sorted(best, key=lambda p: (-best[p], p))[:top_k]
    return [{"paper_id": p, "score": float(best[p]),
             "chunk_no": (best_chunk[p] if best_chunk[p] >= 0 else None)}
            for p in order]


def check_space_compatible(paper_id: int, title: str, abstract: str) -> bool | float:
    """换提供方后验证新旧嵌入空间是否一致：用同一文本重嵌入，与库内旧向量算余弦。

    返回 True（≥0.98，可原地迁移）/ False（需重建）/ False（失败）——异常返回 False。
    """
    try:
        with db.conn() as c:
            row = c.execute("SELECT vec FROM vectors WHERE paper_id=? AND kind='paper' "
                            "AND model != ? ORDER BY id DESC LIMIT 1",
                            (paper_id, config.EMBED_MODEL)).fetchone()
        if not row:
            return False
        old = _from_blob(row["vec"])
        new = np.asarray(embed_texts([f"{title}\n{abstract or ''}".strip()])[0], dtype=np.float32)
        n = min(len(old), len(new))
        if n == 0 or len(old) != len(new):
            return False  # 维度不同必然不同空间
        denom = (np.linalg.norm(old[:n]) * np.linalg.norm(new[:n])) or 1.0
        return True if float(old[:n] @ new[:n] / denom) >= 0.98 else float(
            round(old[:n] @ new[:n] / denom, 4))
    except Exception:
        return False


# L2 全文是否参与全库检索（检索单元 = 按章节切的 chunks，见 db.chunks）。
# `PAPERNEST_CHUNK_SEARCH=0` 关掉做对照，`PAPERNEST_CHUNK_WEIGHT` 调权重。
# 环境变量保留 PAGE_* 旧名兼容（早期这里是按页切的）。
CHUNK_SEARCH = (os.environ.get("PAPERNEST_CHUNK_SEARCH")
                or os.environ.get("PAPERNEST_PAGE_SEARCH", "1")) not in ("0", "false", "False")

# 块级命中的 RRF 权重。**测出来的，不是拍的**——两个评测集必须一起看：
#
# | 权重 | QASPER@1 | @5    | @10   | 自建@5 | @10    | @15    |
# |------|----------|-------|-------|--------|--------|--------|
# | 关闭 | 0.4098   | 0.6066| 0.6639| 0.6771 | 0.7396 | 0.7656 |
# | 0.2  | 0.4672   | 0.6230| 0.6967| 0.7083 | 0.7396 | 0.7500 |
# | 0.5  | 0.4672   | 0.6230| 0.6885| 0.7083 | 0.7083 | 0.7500 |
# | 1.0  | 0.4672   | 0.6639| 0.7377| 0.7083 | 0.7083 | 0.7188 |
#
# 取 0.2：QASPER 每个 k 都涨（@1 +0.057、@10 +0.033），自建集 @5 涨 0.031、@10 持平，
# 代价只有 @15 的 0.0156。只看 QASPER 会选 1.0——那要拿自建集 @10/@15 各三个多点去换。
# （更早按页切时最优点是 0.5；换成按章节切后最优点左移，说明块变小后单条命中更该被"小步加分"。）
CHUNK_WEIGHT = float(os.environ.get("PAPERNEST_CHUNK_WEIGHT")
                     or os.environ.get("PAPERNEST_PAGE_WEIGHT", "0.2"))

# 旧名保留，外部（含测试）仍可读写
PAGE_SEARCH = CHUNK_SEARCH
PAGE_WEIGHT = CHUNK_WEIGHT


class RetrievalResult(NamedTuple):
    """一次检索的完整结果。**降级信号是返回值的一部分，不是可选的旁路。**

    原来这个函数只返回 `(ids, mode)`，把算好的降级说明直接丢掉，逼调用方从 mode
    反推——而 mode 又会被拼成 'fts+chunks'，`mode == "fts"` 的判断当场失效。
    把 degraded 放进返回值，调用方要么用它、要么显式忽略，没有「不小心漏掉」这条路。
    """

    ids: list[int]
    mode: str
    degraded: list[degrade.Degradation]
    #: {paper_id: [{chunk_no, section_path, start_page, text, via}]}
    #: 本次检索里**具体哪些块命中了**。上下文组装靠它把语义/词面命中直接喂给模型，
    #: 而不是拿 paper_id 回到 pages 表按词频重猜一遍。None/空 = 没有块级命中。
    #: 默认写 None 而不是 {}：NamedTuple 的默认值在所有实例间**共享**，
    #: 给可变对象当默认值，谁改一下就污染全部。调用方一律 `res.chunk_hits or {}`。
    chunk_hits: dict | None = None


def _merge_hits(lex: dict, vec: dict) -> dict[int, list[dict]]:
    """合并词面命中与语义命中，**语义那一块排前面**。

    语义命中通常更贴题（跨语言时词面根本没有命中），但词面命中带来多样性——
    两边都留，去重按 chunk_no。
    """
    out: dict[int, list[dict]] = {}
    for pid in set(lex) | set(vec):
        seen, merged = set(), []
        for src, via in ((vec.get(pid) or [], "vector"), (lex.get(pid) or [], "lexical")):
            for h in src:
                if h["chunk_no"] in seen:
                    continue
                seen.add(h["chunk_no"])
                merged.append({**h, "via": h.get("via", via)})
        out[pid] = merged
    return out


def search_hybrid(query: str, top_k: int = 8, rrf_k: int = 60,
                  use_pages: bool | None = None) -> RetrievalResult:
    """RRF 混合检索：向量排序 + 标题/摘要 FTS + 章节块全文 FTS 融合。

    返回 `RetrievalResult(ids, mode, degraded)`；degraded 是结构化的降级记录列表
    （空列表 = 没有降级），要给人看时用 `degrade.render(...)`。

    - 向量未配置 → 不进向量路（正常路径，**不算降级**，degraded 为空）
    - 向量已配置但整路没生效 → degraded 里必有一条 CRITICAL 记录

    **mode 只描述用了哪几路，不承载降级语义**。别再用 `mode == "fts"` 判降级：
    有 chunk 命中时 mode 是 'fts+chunks'，那个等号恒为假，降级会被整个吞掉。
    要判降级看 degraded；要判「向量这一路在不在」用 `mode.startswith("hybrid")`。
    """
    if use_pages is None:
        use_pages = CHUNK_SEARCH
    vec_ids: list[int] | None = None
    notes: list[degrade.Degradation] = []
    vec_chunk: dict[int, int] = {}          # paper_id -> 语义上赢的那个 chunk_no
    if available():
        try:
            hits = search_papers(query, top_k * 3)
            vec_ids = [h["paper_id"] for h in hits]
            vec_chunk = {h["paper_id"]: h["chunk_no"] for h in hits
                         if h.get("chunk_no") is not None}
            if not vec_ids:
                # 配了 EMBED_MODEL 却一条向量都没匹配上，最常见的原因是**模型名对不上**：
                # `.env` 里填的是控制台上的显示名，而建索引时用的是 API 的真实 ID。
                # 这种情况下检索会悄悄退化成纯 FTS 而用户以为混合检索开着——
                # 必须如实上报，不能静默。`cli.py models` 可核对真实 ID。
                notes.append(degrade.Degradation(
                    degrade.VECTOR_INDEX_EMPTY,
                    f"向量索引为空（EMBED_MODEL={config.EMBED_MODEL!r} "
                    f"匹配不到任何向量，检查模型名或先跑 cli.py embed），本次仅 FTS"))
        except Exception as e:
            notes.append(degrade.Degradation(
                degrade.VECTOR_CALL_FAILED,
                f"向量检索不可用（{str(e)[:80]}），已降级 FTS"))
    with db.conn() as c:
        fts_ids = [r["id"] for r in db.search_fts(c, query, top_k * 3)]
        # 检索走 chunks（按章节切）；老库迁移会把 pages 搬过来，所以这里不用兜底
        lex_hits = db.search_chunks_hits(c, query, top_k * 3) if use_pages else {}
        page_ids = sorted(lex_hits, key=lambda p: lex_hits[p][0]["rank"])[:top_k * 3]
        # 向量路赢在某个 chunk 上的论文：把那一块也取回来（词面命中恒为 0 的
        # 跨语言场景下，这是唯一知道该喂哪段的信息来源）
        vec_hits: dict[int, list[dict]] = {}
        for pid, cno in vec_chunk.items():
            got = db.chunks_by_no(c, pid, [cno])
            if got:
                got[0]["via"] = "vector"
                vec_hits[pid] = got

    rankings: list[tuple[list[int], float]] = [(fts_ids, 1.0)]
    mode = "fts"
    if vec_ids:
        rankings.append((vec_ids, 1.0))
        # 走到这里 notes 必为空：两处降级都只在 vec_ids 落空时才记
        # （异常路径下 vec_ids 连赋值都没发生）。原来这行是
        # `"hybrid" if degrade is None else "fts"`，else 分支不可达。
        mode = "hybrid"
    if page_ids:
        rankings.append((page_ids, CHUNK_WEIGHT))
        mode += "+chunks"
    chunk_hits = _merge_hits(lex_hits, vec_hits)
    if len(rankings) == 1:
        return RetrievalResult(fts_ids[:top_k], mode, notes, chunk_hits)

    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    # 分数并列时按 paper_id 决胜——排序键必须全序，否则跨进程结果会漂移
    merged = sorted(scores, key=lambda p: (-scores[p], p))
    return RetrievalResult(merged[:top_k], mode, notes, chunk_hits)


def best_sentence(paragraph_vec: np.ndarray, paper_id: int) -> tuple[str | None, float]:
    """该论文的句子级向量里，与段落最接近的一句。"""
    with db.conn() as c:
        rows = c.execute("SELECT text, vec FROM vectors WHERE paper_id=? AND kind='sent' AND model=?",
                         (paper_id, config.EMBED_MODEL)).fetchall()
    best, best_s = None, -1.0
    for r in rows:
        v = _from_blob(r["vec"])
        denom = (np.linalg.norm(paragraph_vec) * np.linalg.norm(v)) or 1.0
        s = float(paragraph_vec @ v / denom)
        if s > best_s:
            best, best_s = r["text"], s
    return best, (best_s if best_s > 0 else 0.0)


def split_sentences(text: str) -> list[str]:
    """中英混合切句；过滤过短的碎片。"""
    import re
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 12]
