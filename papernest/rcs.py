"""RCS：检索之后、作答之前的两道加工——LLM 重排 + 逐篇定向摘要。

朴素 RAG 有个默认假设：**检索排出来的顺序就是有用的顺序，原文直接塞进去就行**。
两个漏洞：① RRF 融合的是词面/向量的名次，不是「这篇能不能回答这个问题」；
② 每篇几百上千字原样拼接，模型要自己在长上下文里淘金，越长越容易抓错重点。

RCS（借鉴 PaperQA2）只改这一段：
- `rerank`：检索照旧捞更宽的候选池（top_k * POOL_FACTOR），用一次轻档模型调用
  给每篇打相关性分，按新分取前 k。检索负责捞得全，重排负责排得准。
- `summarize_for`：对每篇**针对当前问题**压成两三句，并要求给出原文依据句；
  依据句走一遍机械回取校验（口径同 cite/fulltext），验不过如实标注。
  这一层是 PaperQA2 没有的——它只压缩，不校验压缩出来的东西是否真在原文里。

代价是每次问答多 1 + N 次轻量调用，所以**默认关闭**，做成可开关的深度模式；
`ask --rcs` / `POST /api/answer/rcs` 走它，默认问答路径不受影响。

失败一律降级回原始上下文并如实记 degraded——RCS 是增强件，不该成为新的故障点。
"""
from __future__ import annotations

import json
import re

from . import cite, config, db, degrade, llm

#: 候选池放大倍数：重排要有得挑，池子等于 k 就没什么可排的了
POOL_FACTOR = 3
#: 池子上限，防止小 k 也去打十几次摘要调用
MAX_POOL = 15
#: 单篇喂给重排/摘要的原文上限（重排只需要判相关性，不用看全文）
RERANK_CHARS = 700
SUMMARY_CHARS = 3000

RERANK_SYSTEM = """你在给文献按「能否回答用户的问题」排序。
给你一个问题和若干候选文献（编号 + 标题 + 摘要片段）。
对每篇给 0-10 的相关性分：10=直接回答了这个问题；5=同一主题但没正面回答；0=不相关。
只输出 JSON：{"scores":[{"id":候选编号,"score":整数,"why":"不超过20字的理由"}]}
必须给出**全部**候选的分数，不要遗漏，不要新增不存在的编号。"""

SUMMARY_SYSTEM = """你在为一个问题从单篇文献里提炼相关内容。
只输出 JSON：{"relevant":true/false,
 "summary":"针对该问题的要点，2-3 句；不相关就留空字符串",
 "quotes":["支撑上述要点的原文句子，逐字摘抄，1-3 句"]}
硬性要求：
- quotes 必须**逐字**摘自给定文本，一个字都不能改写、不能拼接、不能翻译；
- 文中没有与问题相关的内容就 relevant=false、summary 留空、quotes 给空数组；
- 不要引入给定文本之外的任何信息。"""


def _light_model() -> str | None:
    """重排/摘要用轻档模型：这一层要跑 N 次，用重档模型成本会翻几倍。"""
    return config.LLM_MODEL or None


def _load(paper_ids: list[int]) -> dict[int, dict]:
    """一次取齐候选的元数据、卡片与页文本，避免 N 次查库。"""
    if not paper_ids:
        return {}
    out: dict[int, dict] = {}
    with db.conn() as c:
        q = ",".join("?" * len(paper_ids))
        for r in c.execute(f"SELECT * FROM papers WHERE id IN ({q})", paper_ids):
            try:
                card = json.loads(r["card_json"] or "{}")
            except json.JSONDecodeError:
                card = {}
            out[r["id"]] = {"row": dict(r), "card": card, "pages": {}}
        for r in c.execute(
                f"SELECT paper_id,page_no,text FROM pages WHERE paper_id IN ({q})"
                " ORDER BY page_no", paper_ids):
            if r["paper_id"] in out:
                out[r["paper_id"]]["pages"][r["page_no"]] = r["text"] or ""
    return out


def _body_text(item: dict, cap: int) -> str:
    """候选的可读文本：摘要 + 卡片要点 + 全文前若干页（有的话）。"""
    row, card = item["row"], item["card"]
    parts = [row.get("abstract") or ""]
    for k in ("tldr", "method", "results", "limitations"):
        v = card.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    for _pno, text in list(item["pages"].items())[:3]:
        parts.append(text)
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()[:cap]


# ── ① 重排 ──

def rerank(question: str, paper_ids: list[int], top_k: int) -> tuple[list[int], dict]:
    """LLM 相关性重排。返回 (重排后的 ids, trace)。失败原样返回输入顺序。"""
    trace: dict = {"pool": len(paper_ids), "used": False, "scores": []}
    if not paper_ids or not llm.available():
        return paper_ids[:top_k], trace
    items = _load(paper_ids)
    lines = []
    for i, pid in enumerate(paper_ids, 1):
        it = items.get(pid)
        if not it:
            continue
        lines.append(f"[{i}] {it['row']['title']}\n{_body_text(it, RERANK_CHARS)}")
    if not lines:
        return paper_ids[:top_k], trace
    user = f"问题：{question}\n\n候选文献：\n" + "\n\n".join(lines)
    try:
        raw = llm.chat(RERANK_SYSTEM, user, purpose="rcs_rerank",
                       temperature=0.0, model=_light_model())
        data = llm.extract_json(raw)
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return paper_ids[:top_k], trace   # 重排失败就用检索原序，不让增强件变成故障点

    scored: dict[int, tuple[int, str]] = {}
    for s in data.get("scores") or []:
        try:
            idx = int(s.get("id"))
            score = int(s.get("score"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(paper_ids):
            scored[paper_ids[idx - 1]] = (max(0, min(score, 10)),
                                          str(s.get("why") or "")[:40])
    if not scored:
        trace["error"] = "重排未返回可用分数"
        return paper_ids[:top_k], trace

    # 模型漏评的候选按检索原序排在已评之后（不猜分数，也不因为漏评就丢掉）。
    # 末位用检索名次决胜 ⇒ 同分时保持检索原序，排序是全序、可复现。
    rank_of = {pid: i for i, pid in enumerate(paper_ids)}
    ordered = sorted(
        paper_ids,
        key=lambda p: (-(scored[p][0] if p in scored else -1), rank_of[p]))
    trace["used"] = True
    trace["missing"] = [p for p in paper_ids if p not in scored]
    # 全部同分 = 重排没有产生任何信息，等于白花一次调用。如实记下来，
    # 否则 trace 里一句「重排生效」会让人误以为顺序被优化过。
    distinct = {v[0] for v in scored.values()}
    if len(distinct) <= 1:
        trace["no_signal"] = (f"模型给全部候选打了同一个分（{distinct.pop() if distinct else '—'}），"
                              "重排未产生任何排序信息，实际仍是检索原序")
    trace["scores"] = [{"paper_id": p, "score": scored[p][0], "why": scored[p][1]}
                       for p in ordered if p in scored]
    return ordered[:top_k], trace


# ── ② 逐篇定向摘要（带依据句机械回取校验）──

#: PDF 抽出来的文本自带一堆排版伪影，逐字比对前必须先抹平，否则模型明明照抄了
#: 也验不过（假阴性比假阳性更隐蔽：它会把真证据标成「未核验」，让人不敢用）。
_SOFT_HYPHEN = dict.fromkeys(map(ord, "­‐‑‒–—-"), None)


def _norm_quote(s: str) -> str:
    """比 cite._norm 更狠一档：空白 + 连字符 + Unicode 兼容分解。

    实测 PDF 里的三类伪影：① 换行处的断词连字符（"estima-\ntion"）；
    ② 连字 ﬁ/ﬂ（NFKC 展开成 fi/fl）；③ 全角/半角标点混用。
    """
    import unicodedata
    text = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", text).translate(_SOFT_HYPHEN).lower()


def _verify(quote: str, item: dict) -> tuple[bool, int | None]:
    """依据句必须能在这篇的原文里回取到。返回 (是否通过, 命中页码)。

    **逐页比对，不拼成一大串**：否则句子前半在第 2 页、后半在第 7 页也能"验过"，
    那是拼接出来的假证据。这条与 cite/fulltext 的口径一致。
    """
    q = _norm_quote(quote)
    if len(q) < 12:
        return False, None
    if q in _norm_quote(item["row"].get("abstract") or ""):
        return True, None
    for pno, text in item["pages"].items():
        if q in _norm_quote(text):
            return True, pno
    return False, None


def summarize_for(question: str, paper_ids: list[int]) -> tuple[dict[int, dict], dict]:
    """对每篇针对问题做定向摘要 + 依据句校验。返回 ({paper_id: 摘要块}, trace)。"""
    trace = {"calls": 0, "kept": 0, "dropped_irrelevant": 0,
             "quotes_total": 0, "quotes_verified": 0, "errors": []}
    out: dict[int, dict] = {}
    if not paper_ids or not llm.available():
        return out, trace
    items = _load(paper_ids)
    for pid in paper_ids:
        it = items.get(pid)
        if not it:
            continue
        body = _body_text(it, SUMMARY_CHARS)
        if not body:
            continue
        user = (f"问题：{question}\n\n"
                f"文献《{it['row']['title']}》的内容：\n{body}")
        try:
            raw = llm.chat(SUMMARY_SYSTEM, user, purpose="rcs_summary",
                           temperature=0.0, model=_light_model())
            data = llm.extract_json(raw)
            trace["calls"] += 1
        except Exception as exc:
            trace["errors"].append(f"#{pid}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if not data.get("relevant"):
            trace["dropped_irrelevant"] += 1
            continue
        summary = str(data.get("summary") or "").strip()
        if not summary:
            trace["dropped_irrelevant"] += 1
            continue
        quotes = []
        for qt in (data.get("quotes") or [])[:3]:
            if not isinstance(qt, str) or not qt.strip():
                continue
            trace["quotes_total"] += 1
            ok, page = _verify(qt, it)
            trace["quotes_verified"] += 1 if ok else 0
            quotes.append({"quote": qt.strip(), "verified": ok, "page": page})
        out[pid] = {"summary": summary, "quotes": quotes,
                    "title": it["row"]["title"], "year": it["row"]["year"],
                    "venue": it["row"]["venue"]}
        trace["kept"] += 1
    return out, trace


# ── ③ 组装上下文 ──

def build_context(question: str, paper_ids: list[int], top_k: int
                  ) -> tuple[str, list[dict], dict]:
    """RCS 全流程：重排 → 逐篇定向摘要 → 组装成浓缩上下文。

    返回 (context_text, sources, trace)。任一步不可用都会退回空 context，
    由调用方降级到朴素 RAG 上下文——增强件不该成为新的故障点。
    """
    trace: dict = {"enabled": True}
    ranked, rtrace = rerank(question, paper_ids, top_k)
    trace["rerank"] = rtrace
    blocks, sources = [], []
    summaries, strace = summarize_for(question, ranked)
    trace["summary"] = strace
    if not summaries:
        trace["degraded"] = "定向摘要没有产出可用内容，已退回原始上下文"
        return "", [], trace

    for i, pid in enumerate([p for p in ranked if p in summaries], 1):
        s = summaries[pid]
        lines = [f"[{i}] {s['title']}（{s.get('venue') or ''} {s.get('year') or ''}）",
                 f"    与问题相关的要点：{s['summary']}"]
        for qt in s["quotes"]:
            mark = ("已核验" + (f"，第 {qt['page']} 页" if qt["page"] else "，摘要内")
                    if qt["verified"] else "未通过原文回取校验")
            lines.append(f"    原文依据（{mark}）：{qt['quote']}")
        blocks.append("\n".join(lines))
        sources.append({"idx": i, "paper_id": pid, "title": s["title"],
                        "year": s.get("year"), "venue": s.get("venue"),
                        "rcs_summary": s["summary"],
                        "quotes": s["quotes"]})
    # 两种「没进上下文」必须分开报，混成一个数会严重误导：
    # - 重排未入选：候选池本来就比 top_k 大，落选是设计如此，不是信息损失；
    # - 摘要判为不相关：这一篇**本来排进了 top_k**，是被模型主动扔掉的——
    #   RCS 最危险的失效就藏在这里（判错就是这篇论文对作答模型彻底消失）。
    trace["not_selected_by_rerank"] = max(len(paper_ids) - len(ranked), 0)
    trace["filtered_out"] = len(ranked) - len(sources)
    return "\n\n".join(blocks), sources, trace


def answer(question: str, top_k: int = 5, progress=None) -> dict:
    """RCS 增强问答：宽检索 → 重排 → 定向摘要 → 作答。

    与 `deepsearch.deep_answer` 互补、可各用各的：deep 解决「检索没捞到」（补检索），
    RCS 解决「捞到了但没用好」（重排 + 提炼）。
    """
    from . import deepsearch, rag
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM——RCS 需要真模型")
    pool = min(max(top_k * POOL_FACTOR, top_k), MAX_POOL)
    if progress:
        progress(0.1, "retrieve", f"宽检索候选池 {pool} 篇")
    ids, mode, rtrace = deepsearch.deep_retrieve(question, pool)
    # 检索这一路的降级必须带下去。原来这里是 `_rt`——算出来了、丢进下划线变量，
    # 正是本项目要消灭的「识别了但没传递」，只是换了个文件。
    retrieval_notes = degrade.from_dicts(rtrace.get("degraded") or [])
    if progress:
        progress(0.3, "rerank", f"LLM 重排 {len(ids)} 篇候选")
    context, sources, trace = build_context(question, ids, top_k)
    trace["retrieval_mode"] = mode

    if not context:      # 降级：退回朴素 RAG 的上下文组装
        ctx = rag.prepare([{"role": "user", "content": question}], top_k=top_k,
                          candidate_ids=ids[:top_k])
        text = llm.chat(ctx.system, ctx.query, purpose="rcs_answer_fallback",
                        temperature=0.3)
        notes = degrade.merge(retrieval_notes, ctx.notes,
                              degrade.from_legacy(trace.get("degraded")))
        return {"answer": text, "sources": ctx.sources, "trace": trace,
                "degraded": degrade.render(notes),
                "degraded_detail": degrade.as_dicts(notes),
                "retrieval_mode": mode}

    if progress:
        progress(0.8, "answer", f"基于 {len(sources)} 篇的定向提炼作答")
    sys_prompt = (rag.SYSTEM
                  + "\n\n下面的上下文已针对本问题做过提炼：每篇给出要点与原文依据句，"
                    "并标明依据句是否通过原文机械回取校验。**标了「未通过原文回取校验」的"
                    "依据句不可当作事实引用**，需要时如实说明证据不足。\n\n"
                  + f"检索到的文献上下文：\n{context}")
    text = llm.chat(sys_prompt, question, purpose="rcs_answer", temperature=0.3)
    st = trace.get("summary") or {}
    return {"answer": text, "sources": sources, "trace": trace,
            "retrieval_mode": mode,
            "verified_rate": (round(st.get("quotes_verified", 0)
                                    / st["quotes_total"], 3)
                              if st.get("quotes_total") else None),
            # 成功路径原来硬编码 None —— 向量整路挂掉（CRITICAL）时也报「无降级」，
            # 是**主动的假阴性**，比字段缺失更糟。
            "degraded": degrade.render(
                degrade.merge(retrieval_notes,
                              degrade.from_legacy(trace.get("degraded")))),
            "degraded_detail": degrade.as_dicts(
                degrade.merge(retrieval_notes,
                              degrade.from_legacy(trace.get("degraded"))))}
