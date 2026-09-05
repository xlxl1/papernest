"""库内 RAG 问答：检索（向量优先 / FTS 兜底）→ 组装带编号上下文 → 严格引用作答。

来源铁律：回答只能基于检索到的文献上下文；每条论断带 [n] 角标；
上下文没覆盖的要直说「库内文献未覆盖」，不许编。
"""
import json
import os
import re
from typing import NamedTuple

from . import chat, config, db, degrade, embeddings, http, llm


SYSTEM = """你是科研文献库的问答助手。仅可依据「检索到的文献上下文」回答问题：
1. 每条来自文献的论断末尾标注来源编号，如 [1]、[2]；编号与上下文一一对应。
2. 上下文不足以回答时，明确说「库内文献未覆盖这个问题」，并可建议检索关键词。
3. 解释术语时优先引用上下文里文献的原句；上下文没有该术语时才允许用通识解释，并注明「（通识解释，非库内来源）」。
4. 润色/写作类请求不强制加角标，但涉及文献事实的部分仍须标注。
5. 有「之前的对话」时，用它理解「它/这篇/第 2 篇」这类指代；但事实仍只能来自文献上下文，
   不要把之前轮次里自己说过的话当成已证实的来源。
用中文回答。"""


def _vector_candidates(q: str, top_k: int
                       ) -> tuple[list[int], list[degrade.Degradation], dict]:
    """检索候选 + 本次检索的降级记录。

    降级判定**不再由这里反推**。原来是 `if mode == "fts" and available()`——
    而 mode 在有 chunk 命中时是 'fts+chunks'，这个等号恒为假，于是向量整路挂掉时
    一句提示都不会出现（默认配置下 chunk 路是开的，正是最常见的情况）。
    现在 search_hybrid 直接把结构化的降级记录返回出来，原样透传即可。
    """
    res = embeddings.search_hybrid(q, top_k)
    return res.ids, list(res.degraded), dict(res.chunk_hits or {})


#: 页面取词：ASCII 按词，CJK 按 2-gram。原来是 `q.split()`——中文问句没有空格，
#: 整句被当成一个「词」，拿去 `t.count()` 数英文正文恒为 0，于是中文提问时
#: L2 全文页**永远**进不了上下文（实测：库内 45 篇有全文的论文里 40 篇正文是英文）。
#: 上下文的字符预算。篇数由调用方的 top_k 决定、单篇由 TL;DR + key_findings + 摘要 +
#: 页块累加，两头原来都没有上限；`llm.chat` 拼 payload 时也不做长度检查，超模型窗口时
#: 服务端返 400，而 400 是「立即失败不重试」——于是整个任务在烧掉前几轮 token 之后
#: 失败，用户只看到一句透传的 provider 报错，看不出是上下文超了。
#: 24000 字符按中文口径 ≈13k token。两个口径的实测：`llm_calls` 里 22 次真实问答的
#: prompt_tokens 区间是 211–6451；而修好 `_page_context`（中文提问下全文页原本恒不进
#: 上下文）之后，合成的最坏情况（12 篇全带 L2 全文）到 12.6k token。
#: 预算与 `deepsearch.MAX_CONTEXT_PAPERS=12` 对齐：12 篇正好是一篇都不丢的点。
MAX_CONTEXT_CHARS = 24000

_ASCII_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CJK_RUN = re.compile(r"[一-鿿]+")


def _query_terms(text: str, limit: int = 12) -> list[str]:
    """切出给页面打分用的检索词。中英分治，去重保序后截断，结果稳定可复现。

    不额外过滤中文虚词：在所有页上均匀出现的词对**页间排序**没有贡献、会自己抵消，
    多一张停用词表反而多一处要维护的口径。
    """
    terms: list[str] = []
    for w in _ASCII_TOK.findall(text or ""):
        if len(w) >= 2:
            terms.append(w.lower())
    for run in _CJK_RUN.findall(text or ""):
        terms += [run[i:i + 2] for i in range(len(run) - 1)]
    return list(dict.fromkeys(terms))[:limit]


def _window(text: str, terms: list[str], cap: int) -> str:
    """截到 cap 字符，但**围绕命中位置取窗口**，不是从页首硬切。

    原来是 `txt[:cap]`：选中一页之后从头截，实测中位数只保留了 29%——
    而命中往往在页中部，于是「选中了正确的页、却把证据切掉了」。
    """
    if len(text) <= cap:
        return text
    low = text.lower()
    pos = min((p for p in (low.find(t) for t in terms) if p >= 0), default=-1)
    if pos < 0:
        return text[:cap]
    start = max(0, pos - cap // 3)
    seg = text[start:start + cap]
    return ("…" + seg) if start else seg


def _pick_pages(pages: dict[int, str], terms: list[str],
                per_paper: int, cap: int) -> list[str]:
    """按词频选页。**分数除以 sqrt(页长)**，否则最长的那页恒赢。

    原来是绝对词频求和，长页天然占优：真库实测 41 篇里 34 篇（83%）打分第一的
    就是该篇字符数最多的页（往往是相关工作/参考文献/附录）。长度归一之后
    比较的是「密度」而不是「体量」。
    """
    if not terms:
        return []
    scored = []
    for pno, txt in pages.items():
        t = (txt or "").lower()
        raw = sum(t.count(k) for k in terms)
        scored.append((raw / ((len(t) or 1) ** 0.5), raw, pno, txt))
    # 排序键必须全序：并列时按页码决胜。原来只按分数排，并列页的先后取决于
    # SELECT 的返回顺序（没有 ORDER BY），同一条问句可能选出不同的页。
    scored.sort(key=lambda x: (-x[0], x[2]))
    return [f"    【第 {pno} 页】{_window(txt, terms, cap // per_paper)}"
            for _d, raw, pno, txt in scored[:per_paper] if raw > 0]


#: 上下文的组装单元：page（默认）| chunk。**默认 page 是量出来的，不是没接线。**
#:
#: 把上下文换成「按章节块组装」是本项目一条**负结果**。三臂配对实验
#: （QASPER 88 道带 gold 证据的题，等上下文预算，纯 FTS 底座，0 token）：
#:
#: | 口径 | 证据召回(删空白) | 证据召回(保留空白) | 变化的题 | p |
#: |---|---|---|---|---|
#: | A 改造前（绝对词频 + 页首硬截） | 0.2273 | 0.2273 | — | — |
#: | B 只改选块（密度归一 + 命中窗口） | **0.2614** | **0.2614** | 7/88 | 0.452 |
#: | C 再换成章节块 | 0.2045 | **0.1136** | 17/88 | 0.331 |
#:
#: B→C 方向为负（−0.0568，不显著），而**保留空白口径下腰斩**（0.2614→0.1136）——
#: 根因不是「章节 vs 页」这个单元本身，而是 `chunks` 表的文本是 pages 的**有损再推导**：
#: 同一篇论文 pages=24 而 chunks=20、pages=19 而 chunks=31，文本已不是原文逐字。
#: 这会直接打断本项目的机械回取校验（证据句要能在原文里逐字找到）。
#:
#: 所以：**在 chunk 文本的逐字保真修好之前，上下文继续按页组装。**
#: 检索侧的 chunk 命中信息已经完整接通（`RetrievalResult.chunk_hits`）并有测试，
#: 换单元只差把这个开关打开——但不能在数字变好之前打开。
#: 复现：`python cli.py qasper ctxab`
CONTEXT_UNIT = os.environ.get("PAPERNEST_CONTEXT_UNIT", "page").strip().lower()

#: 检索命中的块在密度排序上的加成。**是加成，不是闸门**——
#: 第一版把上下文限制成「只有 chunks_fts 命中的那 ≤4 块」，实测证据召回
#: 从 0.2614 掉到 0.1818：命中集之外的块再相关也进不来，而按密度扫全篇能捞到它们。
#: 检索信号该用来「排得更准」，不该用来「把别的候选删掉」。
HIT_BOOST = 1.6


def _rank_chunks(chunks: list[dict], hits: list[dict], terms: list[str],
                 per_paper: int) -> tuple[list[dict], bool]:
    """从**全部**章节块里挑 per_paper 个，检索命中的享加成。

    返回 (选中的块, 是否只能靠检索信号)。后者为 True 时说明问句在这篇正文里
    词面命中恒为 0（典型是中文问句 × 英文正文）——此时检索命中是唯一的信号。
    """
    hit_nos = {h["chunk_no"] for h in hits}
    scored = []
    for ch in chunks:
        t = (ch.get("text") or "").lower()
        raw = sum(t.count(k) for k in terms)
        density = raw / ((len(t) or 1) ** 0.5)
        if ch["chunk_no"] in hit_nos:
            density *= HIT_BOOST
        scored.append((density, raw, ch["chunk_no"], ch))
    if any(s[1] for s in scored):
        scored.sort(key=lambda x: (-x[0], x[2]))   # 全序：并列按 chunk_no 决胜
        return [s[3] for s in scored[:per_paper]], False
    # 词面全 0：只剩检索信号（向量那一路知道哪段语义上贴题）
    return hits[:per_paper], True


def _chunk_context(chunks: list[dict], hits: list[dict], terms: list[str],
                   per_paper: int, cap: int) -> tuple[str, bool]:
    """把章节块铺成上下文。

    这是「按章节切」这条结论真正落地的地方。此前 chunks 只影响「哪篇论文被召回」，
    组装上下文时一行都不读它——`_page_context` 拿 paper_id 回到 pages 表按词频
    重新猜一遍，等于把检索已经算出来的答案扔掉再猜一次。
    带上 section_path 与 start_page：引用要能溯源到章节，页码要能机械回取。
    """
    picked, lexical_blind = _rank_chunks(chunks, hits, terms, per_paper)
    out = []
    for h in picked:
        head = h.get("section_path") or "正文"
        page = h.get("start_page")
        loc = f"{head}·第 {page} 页" if page else head
        out.append(f"    【{loc}】{_window(h.get('text') or '', terms, cap // per_paper)}")
    return ("\n" + "\n".join(out) if out else ""), lexical_blind


def _paper_terms(c, paper_id: int) -> list[str]:
    """这篇论文自己的标题词与卡片关键词——跨语言兜底选页用。"""
    r = c.execute("SELECT title, card_json FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not r:
        return []
    card = cards_safe(r["card_json"])
    kws = [k for k in (card.get("keywords") or []) if isinstance(k, str)]
    return _query_terms(" ".join([r["title"] or ""] + kws))


def _evidence_context(paper_id: int, q: str, hits: list[dict] | None,
                      per_paper: int, cap: int,
                      unit: str = "chunk") -> tuple[str, str | None]:
    """这一篇的证据块。**优先用检索真正命中的 chunk**，没有才退回按页猜。

    口径（会如实进 degraded）：
      None       —— 用了检索命中的章节块，或问句正常选出了页
      "fallback" —— 问句在这篇正文里一个词都没命中，退回按本篇关键词选页
      "miss"     —— 有全文却一页都没选出，只有摘要进上下文
    """
    if unit != "chunk":          # A/B 用：强制走改造前的「按页选块」口径
        return _page_context(paper_id, q, per_paper, cap)
    with db.conn() as c:
        chunks = [dict(r) for r in c.execute(
            "SELECT chunk_no, section_path, start_page, text FROM chunks "
            "WHERE paper_id=? ORDER BY chunk_no", (paper_id,))]
    if chunks:
        block, blind = _chunk_context(chunks, hits or [], _query_terms(q),
                                      per_paper, cap)
        if block:
            # 词面全 0 时选出来的块只由检索信号决定——与按页选块的 "fallback"
            # 是同一类事实，同样要如实上报，不能因为换了单元就假装没降级。
            return block, ("fallback" if blind else None)
    return _page_context(paper_id, q, per_paper, cap)


def _page_context(paper_id: int, q: str, per_paper: int = 2,
                  cap: int = 2400) -> tuple[str, str | None]:
    """论文已有 L2 全文时，挑关键词命中最多的几页，带页码进上下文。

    返回 (页块文本, 选页口径)。口径 None = 按问句正常选出；``"fallback"`` = 问句在
    这篇正文里一个词都没命中（典型是中文问句 × 英文论文），退回按本篇自己的关键词选页；
    ``"miss"`` = 两种口径都没选出页。**不再静默**：原来这三种情况一律返回空串，
    调用方无从区分「这篇没有全文」和「有全文但一个词都没匹配上」。
    """
    with db.conn() as c:
        pages = {r["page_no"]: r["text"] for r in
                 c.execute("SELECT page_no,text FROM pages WHERE paper_id=?", (paper_id,))}
        if not pages:
            return "", None          # 这篇本来就没入 L2 全文，不是异常
        blocks = _pick_pages(pages, _query_terms(q), per_paper, cap)
        if blocks:
            return "\n" + "\n".join(blocks), None
        # 跨语言兜底：中文问句对英文正文的字面匹配必然为 0。退回按本篇关键词选页，
        # 选出来的是「本篇核心」而不是「问句相关」——所以必须标注，让上层如实上报。
        blocks = _pick_pages(pages, _paper_terms(c, paper_id), per_paper, cap)
    if blocks:
        return "\n" + "\n".join(blocks), "fallback"
    return "", "miss"


def _carry_over(session_id: str | None, question: str, ids: list[int],
                top_k: int) -> list[int]:
    """追问回指上一轮的来源时，把被指的论文拉回上下文并排在最前。

    「第 2 篇的方法讲细一点」——如果本轮检索没召回那篇，模型就只能瞎编或者装傻。
    """
    if not session_id:
        return ids
    prior = chat.last_sources(session_id)
    if not prior:
        return ids
    wanted = chat.referenced_indices(question)
    by_idx = {s.get("idx"): s.get("paper_id") for s in prior}
    pinned = [by_idx[i] for i in wanted if by_idx.get(i)]
    if not pinned:
        return ids
    merged = list(dict.fromkeys(pinned + list(ids)))
    return merged[:max(top_k, len(pinned))]


class PreparedContext(NamedTuple):
    """`prepare` 的返回值。

    比原来的 4 元组多一个 `notes`：`degraded` 那句中文是给人看的（前端 10 处消费点
    都按字符串渲染，契约不能动），`notes` 是同一批信号的结构化形式，用于落库、
    做监控聚合、以及判断有没有 CRITICAL 级降级。
    """

    system: str
    query: str
    sources: list[dict]
    degraded: str | None
    notes: list[degrade.Degradation]


def prepare(messages: list[dict], top_k: int = 5,
            candidate_ids: list[int] | None = None,
            session_id: str | None = None,
            max_context_chars: int = MAX_CONTEXT_CHARS,
            _ctx_unit: str | None = None
            ) -> PreparedContext:
    """检索 + 组装上下文，返回 `PreparedContext`。

    供流式回答使用：检索先行可立刻把来源推给前端，再逐 token 出正文。
    传 session_id 时启用服务端会话记忆：历史以库里的为准（前端可以不发），
    追问会把上文问题拼进检索查询，[n] 编号在整个会话内保持指向同一篇论文。
    """
    q = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    turns = chat.history(session_id, 12) if session_id else []
    prior_qs = [t["content"] for t in turns if t["role"] == "user"]
    # 客户端历史只在没有会话时兜底（老前端仍能工作），有会话一律以服务端为准
    if not turns and len(messages) > 1:
        turns = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in messages[:-1]][-12:]
        prior_qs = [t["content"] for t in turns if t["role"] == "user"]

    search_q, rewrite_note = chat.rewrite_query(q, prior_qs)
    if candidate_ids is None:
        ids, notes, chunk_hits = _vector_candidates(search_q, top_k)
        ids = _carry_over(session_id, q, ids, top_k)
    else:
        # 上游（agent / deep_answer）已经检索过并把候选传进来，这里不重复检索。
        # 上游那次检索自己的降级记录由上游负责上报，这里没有可报的。
        # 上游（agent / deep_answer / RCS 兜底）已经检索过并把候选传进来，这里不重复检索。
        # 上游那次检索自己的降级记录由上游负责上报，这里没有可报的。
        # **但块级命中要补一次**：不补的话这几条路径就退回按页猜，
        # 「按章节切」的收益只在直接问答那一条路上生效——同一个功能两条路两种行为。
        # 一次 chunks_fts 查询，0 token。
        ids, notes = list(dict.fromkeys(candidate_ids))[:top_k], []
        with db.conn() as c:
            hits = db.search_chunks_hits(c, search_q, max(top_k * 3, 15))
        chunk_hits = {p: h for p, h in hits.items() if p in set(ids)}

    numbering = chat.assign_indices(session_id, list(ids))
    blocks, sources, page_notes = [], [], []
    used, over_budget = 0, 0
    # 页块配额按篇数摊。宁可让每篇的原文引用短一些，也不要整篇论文连同它的 [n]
    # 一起被预算挤掉——丢一篇是丢掉一条可引用的来源，缩一篇只是证据少一点。
    # 篇数多时把「每篇 2 个半截页」降成「每篇 1 个最相关的页」：同样的字符数下，
    # 一个完整的页比两个各截一半的页更可能包含可直接引用的完整句子。
    # 0.55 是页块能占的份额，其余留给标题 / TL;DR / key_findings / 摘要。
    per_paper = 2 if len(ids) <= 6 else 1
    page_cap = max(600, min(2400, int(max_context_chars * 0.55) // max(len(ids), 1)))
    with db.conn() as c:
        for pid in ids:
            r = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if not r:
                continue
            i = numbering.get(pid, len(sources) + 1)
            card = cards_safe(r["card_json"])
            lines = [f"[{i}] {r['title']}（{r['venue'] or ''} {r['year'] or ''}）"]
            if card.get("tldr"):
                lines.append(f"    TL;DR：{card['tldr']}")
            if card.get("key_findings"):
                for f in card["key_findings"][:4]:
                    lines.append(f"    结论（第{f.get('page','?')}页{'，已核验' if f.get('verified') else ''}）：{f.get('claim','')}")
            if r["abstract"]:
                lines.append(f"    摘要原句：{r['abstract'][:500]}")
            page_block, note = _evidence_context(pid, search_q,
                                                 chunk_hits.get(pid),
                                                 per_paper, page_cap,
                                                 unit=_ctx_unit or CONTEXT_UNIT)
            if page_block:
                lines.append(page_block)
            paper_block = "\n".join(lines)
            # 超预算的论文**同时**不进 blocks 和不进 sources：只丢其一的话，
            # sources 里会有一个上下文里没有对应块的 [n]，模型会去引用一个看不见的编号。
            # 首篇无条件保留——宁可超一点，也不能交出空上下文。
            if blocks and used + len(paper_block) > max_context_chars:
                over_budget += 1
                continue
            used += len(paper_block)
            if note:
                page_notes.append(note)
            blocks.append(paper_block)
            sources.append({"idx": i, "paper_id": pid, "title": r["title"],
                            "year": r["year"], "venue": r["venue"]})
    sources.sort(key=lambda s: s["idx"])
    context = "\n\n".join(blocks) or "（库内没有检索到相关文献）"
    sys = SYSTEM
    hist_block = chat.history_block(turns)
    if hist_block:
        sys += f"\n\n之前的对话（用于理解指代，不是事实来源）：\n{hist_block}"
    sys += f"\n\n检索到的文献上下文：\n{context}"
    if over_budget:
        notes.append(degrade.Degradation(
            degrade.CONTEXT_OVER_BUDGET,
            f"{over_budget} 篇超出上下文预算（{max_context_chars} 字符）未纳入"))
    if page_notes.count("fallback"):
        notes.append(degrade.Degradation(
            degrade.PAGE_PICK_FALLBACK,
            f"{page_notes.count('fallback')} 篇按本篇关键词选页"
            "（问句未命中其正文，常见于中文问句×英文论文）"))
    if page_notes.count("miss"):
        notes.append(degrade.Degradation(
            degrade.PAGE_PICK_MISS,
            f"{page_notes.count('miss')} 篇有全文却未命中任何页，仅摘要进上下文"))
    if rewrite_note:
        notes.append(degrade.Degradation(degrade.QUERY_REWRITTEN, rewrite_note))
    return PreparedContext(sys, q, sources, degrade.render(notes), notes)


def answer(messages: list[dict], top_k: int = 5,
           candidate_ids: list[int] | None = None,
           session_id: str | None = None) -> dict:
    """Answer a question from the local library.

    ``candidate_ids`` lets an orchestrator expose retrieval as a first-class
    tool and pass its result into the writer without a hidden second search.
    Existing callers keep the old behaviour.
    """
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM（.env 的 LLM_API_BASE / LLM_API_KEY / LLM_MODEL）")
    ctx = prepare(messages, top_k, candidate_ids, session_id)
    resp = llm.chat(ctx.system, ctx.query, purpose="chat", temperature=0.4)
    if session_id:
        chat.append(session_id, "user", ctx.query)
        chat.append(session_id, "assistant", resp, ctx.sources,
                    ctx.degraded, ctx.notes)
    return {"answer": resp, "sources": ctx.sources, "degraded": ctx.degraded,
            "degraded_detail": degrade.as_dicts(ctx.notes),
            "session_id": session_id}


def cards_safe(raw):
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}
