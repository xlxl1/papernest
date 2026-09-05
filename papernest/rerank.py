"""检索结果精排：宽召回 → 精排取前 k。

⚠️ **当前状态：实验/离线模块，未接入任何服务路径。**
全仓搜索 `import rerank` / `retrieve_and_rerank`，命中的只有 `tests/test_rerank.py`——
`api.py` / `cli.py` / `rag.py` / `eval.py` / `deepsearch.py` / `tools.py` 都没有调用方。
也就是说 `PAPERNEST_RERANK` 这个环境变量对运行中的系统**零影响**，README 里那组精排
对照数字来自一次性实验，不是当前线上口径。要接线就在 `tools.retrieve_library` 里
用 `retrieve_and_rerank` 替掉 `search_hybrid`，并把 `used` 后端如实带进 degraded。

为什么要这一层：RRF 融合的是「词面排名」与「向量排名」，它保证的是**捞得全**，
不保证**排得准**。真正含答案的论文常常排在第 4、5 位，而喂给模型的上下文只有前几篇。
把召回池放宽到 k*4，再用一个只干「判相关性」的打分器重排，是检索质量最便宜的一次提升。

四个可插拔后端，按可用性自动降级，**每种都如实上报用了哪个**：

- ``api``      Cohere 风格 ``/rerank`` 端点上的 Cross-Encoder（bge-reranker 等）。
               质量最好、成本最低，但要提供方支持——本项目当前中转端点没有这个路由，
               会自动降级。不装 sentence-transformers 跑本地模型是刻意的：
               为一个个人工具引入 2GB 的 torch 不划算。
- ``llm``      让模型给每个候选打 0-10 分。任何 OpenAI 兼容端点都能用，但一次提问
               多一次调用，且**分数跨次不稳**（同一批候选连跑两次会给出不同排序），
               所以不做默认。
- ``bm25``     0 token 的特征打分：BM25 词频饱和 + 查询词覆盖率 + 标题/正文字段加权。
               确定性、零成本，但**实测是净负收益，不要开**（见下）。
- ``off``      不重排，原样返回——**这是默认**。

实测结论（QASPER 122 题 + 自建集 32 条，两个集一起看）：

| 口径              | QASPER@1 | @5     | 自建@5 | @10    |
|-------------------|----------|--------|--------|--------|
| 原 RRF（不精排）  | 0.4672   | 0.6230 | 0.7083 | 0.7396 |
| BM25 精排 池=2k   | 0.3934   | 0.6148 | 0.7135 | 0.7396 |
| BM25 精排 池=4k   | 0.3852   | 0.5984 | 0.7188 | 0.7292 |
| BM25 精排 池=8k   | 0.3197   | 0.5984 | 0.7188 | 0.7292 |

BM25 精排把 QASPER@1 从 0.467 打到 0.320，且**池越大越差**。原因不是实现问题，
是设计问题：RRF 里已经含了词面排序，再用一个纯词面的打分器重排，等于把向量那一路
的信息**扔掉**，只留词面——信息量严格变少。池越大，被词面相似但语义不对的论文
顶上来的机会越多。

所以「精排」这层只有在**打分器比召回器掌握更多信息**时才成立。

**后来接上真 Cross-Encoder（DashScope `qwen3.7-text-rerank`）做了对照，
结论是「在本项目当前的检索质量下，精排没有可改进的空间」**——
这个结论比预期的更强，也更有意思，因为它是**修完一个 bug 之后才浮出来的**。

第一次测（chunk 向量已建但**未接进检索**，基线 Recall@5=0.9427）：

| 口径              | Recall@5 | 有变化的题 | p      |
|-------------------|----------|-----------|--------|
| BM25 精排         | 0.8750   | 4/32      | 0.6224 |
| Cross-Encoder 精排| 0.9740   | 2/32      | 0.5034 |

看起来 Cross-Encoder「+0.031」，方向对得上推断。但把 chunk 向量真正接进
`_paper_matrix` 之后重测（基线自己涨到 0.9740）：

| 口径              | Recall@5   | 差值    | 有变化的题 | p      |
|-------------------|------------|---------|-----------|--------|
| 不精排（连跑两次一致）| **0.9740** | —       | —         | —      |
| Cross-Encoder 精排| 0.9740     | +0.0000 | **0/32**  | 1.0000 |
| BM25 精排         | 0.8750     | −0.0990 | 4/32      | 0.1285 |

**Cross-Encoder 一道题都改不动了。** 原来那 +0.031 并不是「精排的价值」，
而是它**替检索补上了本该召回、却因为 chunk 向量没接线而漏掉的那两篇**。
根因修掉之后，这份收益就没有了——检索已经把 gold 全捞进 top-5，精排无处可改。

所以「精排只在打分器信息量更大时才成立」这条推断，还要再加一个前提：
**而且召回本身得留有可改进的余地**。召回接近饱和时，再强的打分器也只是重排已经对的结果。
（BM25 的 −0.0990 在 32 条上 p=0.1285 仍不显著；它在 QASPER 122 题上把 hit@1
从 0.4672 打到 0.3197 才是有量级支撑的那份证据。）

延迟数字可信：Cross-Encoder 每次查询多一次接口调用，3398ms → 6336ms（+87%）。
**买不到召回、只买延迟**，所以默认 off。

这正是业界用
Cross-Encoder（query 与 doc 一起进模型，有语义交互）而不是词面打分器的原因。

**``llm`` 后端的实测（有语义，结果确实不一样）**：

| 口径          | QASPER@1 | @5     | 自建@5     | @10    |
|---------------|----------|--------|------------|--------|
| 原 RRF        | 0.4500   | 0.6750 | 0.6771     | 0.7396 |
| LLM 精排 ①    | 0.4250   | 0.6750 | **0.7656** | 0.7656 |
| LLM 精排 ②    | 0.4250   | 0.6750 | **0.7656** | 0.7656 |

自建集 Recall@5 +0.0885，是这个评测集上见过的最大单项提升；QASPER@1 略降、@5 持平。
**但按显著性口径这个提升报不得**：32 条里只有 **4 条**发生变化（4 条全是变好、0 条变差），
符号翻转检验 p=0.1228，没过 0.05。方向为正、样本量不够——如实记成「方向可信、幅度待确认」。

两个值得记的观察：
1. **四条变好的题里两条是中文查询**（跨语言场景），两条是从 0.00 直接到 1.00 的完全漏检。
   语义打分器在「中文问、英文论文」上补的正是词面检索最弱的地方。
2. **连跑两次数字逐位相同**——这与本项目「LLM 参与的指标跨次不稳」的经验相反。
   原因是精排的输出是**排序**（离散且粗粒度），分数的小幅抖动改变不了 top-5 的次序；
   而生成类指标的输出是文本，任何抖动都直接进指标。**LLM 参与不等于不可复现，
   取决于输出经过了多粗的量化。**

代价：每次提问多一次调用（实测 72 条查询多花约 984s，≈13.7s/次）。
综合「不显著 + 一次额外调用」，默认仍是 ``off``，需要时 ``PAPERNEST_RERANK=llm`` 开。

注意 ``bm25`` 不是 Cross-Encoder，别混为一谈：它没有语义理解，只是把词面信号
算得比 RRF 名次细。实测数字见 README。
"""
from __future__ import annotations

import math
import os
import re

from . import config, db, http, llm

BACKEND = os.environ.get("PAPERNEST_RERANK", "off")   # api | llm | bm25 | off
API_MODEL = os.environ.get("PAPERNEST_RERANK_MODEL", "qwen3.7-text-rerank")
# DashScope 的 rerank **不在 OpenAI 兼容路径下**（那里恒 404），走原生地址。
# 留成可配的：换 Cohere/Jina/SiliconFlow 时它们都是 <base>/rerank 的 Cohere 风格。
API_URL = os.environ.get(
    "PAPERNEST_RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")
API_KEY = (os.environ.get("PAPERNEST_RERANK_KEY")
           or os.environ.get("DASHSCOPE_API_KEY", ""))

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]{1,}|\d+(?:\.\d+)?")
_CJK_RUN_RE = re.compile(r"[一-鿿]{2,}")

# BM25 参数（k1 控词频饱和、b 控长度归一）。用文献检索里的常规取值，没有调参空间
# 可言——库只有几百篇，调它属于对噪声过拟合。
_K1, _B = 1.5, 0.75


def _terms(text: str) -> list[str]:
    """英文按词、中文按二元组切。与 db._expand_terms 同一套口径（trigram FTS 下
    中文整句不分词，必须自己拆），这里重写一份是为了不依赖别的模块的私有函数。"""
    out = [w.lower() for w in _WORD_RE.findall(text or "")]
    cjk_runs = _CJK_RUN_RE.findall(text or "")
    for run in cjk_runs:
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def _doc_text(row) -> tuple[str, str]:
    """(标题, 正文)。正文 = 摘要 + 该论文最相关的几个 chunk。"""
    return (row["title"] or "", row["abstract"] or "")


def _gather(paper_ids: list[int], query: str) -> dict[int, dict]:
    """为候选论文取打分素材：标题、摘要、以及命中查询词的章节块。"""
    if not paper_ids:
        return {}
    qs = ",".join("?" * len(paper_ids))
    out: dict[int, dict] = {}
    with db.conn() as c:
        for r in c.execute(
                f"SELECT id,title,abstract FROM papers WHERE id IN ({qs})", paper_ids):
            out[r["id"]] = {"title": r["title"] or "", "abstract": r["abstract"] or "",
                            "chunks": []}
        # 只取命中的块，避免把整篇全文拉进内存
        for row in db.search_chunks_fts(c, query, 200):
            d = out.get(row["paper_id"])
            if d is not None and len(d["chunks"]) < 3:
                d["chunks"].append({"text": row["text"] or "",
                                    "section_path": row["section_path"] or "",
                                    "start_page": row["start_page"]})
    return out


# ── 后端 1：BM25 特征打分（0 token，确定性，默认）──

def _bm25_scores(query: str, docs: dict[int, dict]) -> dict[int, float]:
    q_terms = _terms(query)
    if not q_terms or not docs:
        return {pid: 0.0 for pid in docs}
    uniq = list(dict.fromkeys(q_terms))

    fields: dict[int, dict[str, list[str]]] = {}
    for pid, d in docs.items():
        body = " ".join([d["abstract"]] + [c["text"] for c in d["chunks"]])
        fields[pid] = {"title": _terms(d["title"]), "body": _terms(body)}

    n_docs = len(docs)
    scores: dict[int, float] = {}
    for field, weight in (("title", 3.0), ("body", 1.0)):
        lens = [len(fields[p][field]) for p in fields]
        avg = (sum(lens) / len(lens)) if lens else 1.0
        df = {t: sum(1 for p in fields if t in fields[p][field]) for t in uniq}
        for pid in docs:
            toks = fields[pid][field]
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            s = 0.0
            for t in uniq:
                f = tf.get(t, 0)
                if not f:
                    continue
                # +0.5/+0.5 的 IDF 平滑：库小，df=n 时不能让权重变成负数
                idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / (avg or 1)))
            scores[pid] = scores.get(pid, 0.0) + weight * s

    # 覆盖率加成：命中 5 个查询词中的 4 个，比某一个词出现 20 次更说明相关
    for pid in docs:
        allt = set(fields[pid]["title"]) | set(fields[pid]["body"])
        cov = sum(1 for t in uniq if t in allt) / len(uniq)
        scores[pid] = scores.get(pid, 0.0) * (0.5 + cov)
    return scores


# ── 后端 2：Cross-Encoder API（Cohere 风格 /rerank）──

def api_available() -> bool:
    return bool(config.LLM_API_BASE and config.LLM_API_KEY
                and os.environ.get("PAPERNEST_RERANK_API", "") not in ("0", "false"))


def _api_scores(query: str, docs: dict[int, dict]) -> dict[int, float] | None:
    """调 Cross-Encoder rerank 接口。任何异常都返回 None，让调用方降级并如实上报。

    两种返回格式都认：
    - DashScope 原生：`{"output": {"results": [{"index", "relevance_score"}]}}`
    - Cohere 风格（Jina / SiliconFlow / Voyage / Cohere）：`{"results": [...]}`
    这样换提供方只要改 `PAPERNEST_RERANK_URL`，不用动代码。
    """
    if not api_available() or not docs:
        return None
    ids = sorted(docs)
    texts = [(docs[p]["title"] + chr(10) + docs[p]["abstract"])[:2000] for p in ids]
    key = API_KEY or config.LLM_API_KEY
    dashscope = "dashscope" in API_URL and "compatible-mode" not in API_URL
    if dashscope:
        body = {"model": API_MODEL,
                "input": {"query": query, "documents": texts},
                "parameters": {"top_n": len(texts), "return_documents": False}}
    else:
        body = {"model": API_MODEL, "query": query,
                "documents": texts, "top_n": len(texts)}
    try:
        with http.client(timeout=60) as c:
            r = c.post(API_URL, json=body,
                       headers={"Authorization": f"Bearer {key}",
                                "Content-Type": "application/json"})
        if r.status_code != 200:
            return None
        data = r.json()
        results = (data.get("output") or {}).get("results") or data.get("results") or []
        out: dict[int, float] = {p: 0.0 for p in ids}
        got = 0
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ids):
                out[ids[idx]] = float(item.get("relevance_score") or 0.0)
                got += 1
        # 一条都没解析出来 = 返回结构不是我们认识的那两种。宁可降级也不要
        # 拿一堆 0 分去排序——那等于把召回顺序打乱，比不精排更糟。
        return out if got else None
    except Exception:
        return None


# ── 后端 3：LLM 打分 ──

_LLM_SYSTEM = """你是检索结果相关性评审。对每个候选文献，判断它与用户问题的相关程度。
只输出 JSON：{"scores": [{"id": 候选编号, "score": 0-10 整数}]}
评分口径：10=直接回答了问题；7-9=高度相关；4-6=同一主题但不直接相关；
0-3=无关。**不确定就给低分**，宁可漏也不要把无关的排上来。只输出 JSON。"""


def _llm_scores(query: str, docs: dict[int, dict]) -> dict[int, float] | None:
    if not llm.available() or not docs:
        return None
    ids = sorted(docs)
    lines = []
    for i, p in enumerate(ids, 1):
        d = docs[p]
        lines.append(f"[{i}] {d['title']}\n    {d['abstract'][:300]}")
    try:
        raw = llm.chat(_LLM_SYSTEM,
                       f"问题：{query}\n\n候选文献：\n" + "\n".join(lines),
                       purpose="rerank", temperature=0.0)
        data = llm.extract_json(raw)
    except Exception:
        return None
    out: dict[int, float] = {p: 0.0 for p in ids}
    for item in (data.get("scores") or []):
        try:
            i = int(item["id"])
            if 1 <= i <= len(ids):
                out[ids[i - 1]] = float(item.get("score") or 0)
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ── 顶层 ──

def rerank(query: str, paper_ids: list[int], top_k: int = 5,
           backend: str | None = None) -> tuple[list[int], str]:
    """把召回池重排后取前 top_k。返回 (paper_ids, backend_used)。

    backend_used 如实反映实际生效的后端（api 不可用会降级到 bm25），
    调用方据此决定要不要在 UI 上标注。
    """
    backend = (backend or BACKEND).lower()
    if backend == "off" or not paper_ids:
        return paper_ids[:top_k], "off"

    docs = _gather(paper_ids, query)
    if not docs:
        return paper_ids[:top_k], "off"

    scores = None
    used = backend
    if backend == "api":
        scores = _api_scores(query, docs)
        if scores is None:
            used = "bm25"          # 端点不支持 → 降级，不静默失败
    elif backend == "llm":
        scores = _llm_scores(query, docs)
        if scores is None:
            used = "bm25"
    if scores is None:
        scores = _bm25_scores(query, docs)

    # 排序键全序：分数并列时保持召回池的原始名次，再按 paper_id 决胜。
    # 只按分数排会让并列项的顺序随字典遍历漂移——本项目踩过这个坑。
    order = {p: i for i, p in enumerate(paper_ids)}
    ranked = sorted(docs, key=lambda p: (-scores.get(p, 0.0), order.get(p, 1 << 30), p))
    return ranked[:top_k], used


def retrieve_and_rerank(query: str, top_k: int = 5, pool_mult: int = 4,
                        backend: str | None = None) -> tuple[list[int], str, dict]:
    """宽召回 + 精排的一站式入口。返回 (ids, mode, trace)。"""
    from . import embeddings
    pool = max(top_k * pool_mult, top_k)
    _r = embeddings.search_hybrid(query, pool)
    ids, mode = _r.ids, _r.mode
    ranked, used = rerank(query, ids, top_k, backend)
    moved = sum(1 for i, p in enumerate(ranked) if ids[:top_k].count(p) == 0)
    return ranked, f"{mode}+rerank:{used}", {
        "pool": len(ids), "top_k": top_k, "backend": used,
        "promoted": moved,          # 精排从召回池深处提上来的篇数（这层的直接价值）
    }
