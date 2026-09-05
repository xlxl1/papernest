"""迭代检索（agentic retrieval）：单轮 RRF 之外，再按已召回文献扩查询补检索。

一次检索只覆盖问句本身的词面/语义邻域。库里真正相关的论文常常换了一套说法——
README 里记着的那个坑就是典型：中文问、英文论文，「近场信道估计」召不回
"near-field channel estimation"。伪相关反馈（PRF）正是为这个场景发明的：
先信第一轮召回的头部文献，从它们的关键词/标题里派生补充查询，再把多轮排序 RRF 融合。

三层，成本递增、都可独立使用：
- `expand_queries` 确定性派生补充查询（读卡片关键词与标题短语，**0 token**）；
- `deep_retrieve`  多轮检索 + 加权 RRF 融合，**离线可跑**，所以能直接在 eval_set 上
  和单轮口径做 A/B（`cli.py eval --retrieval deep`）——不是「感觉更好」而是有数字；
- `deep_answer`    配了 LLM 时再加一层闭环：作答 → **机械**找无证据论断 → 让模型提补充
  查询 → 补检索 → 重答。每轮都留痕迹，和项目其它部分一样可审计。
"""
from __future__ import annotations

import json
import re

from . import db, degrade, embeddings, llm

# RRF 融合里原始查询的权重高于派生查询：派生词是「猜的」，不该盖过用户原话
ORIGINAL_WEIGHT = 1.0
EXPANSION_WEIGHT = 0.6
RRF_K = 60

#: 进上下文的论文篇数硬闸门。`rag.prepare` 的 `[:top_k]` 是唯一的兜底截断，
#: 而 `deep_answer` 传的是 `top_k=len(seen)`——等于把它恒等化；`max_rounds` 又能从
#: API / CLI 无上界地传进来，于是 seen 可以涨到任意大（max_rounds=20 → 100 篇）。
#: 这个闸门的定位是**防失控**，不是防长度衰减：实测默认口径的上下文只有 5.0k token，
#: 离任何公开的衰减拐点还有 3 倍以上。
MAX_CONTEXT_PAPERS = 12

# 标题里切短语用：这些词自己不构成检索信号
_STOP = {"a", "an", "the", "of", "for", "and", "or", "on", "in", "to", "with",
         "via", "using", "based", "towards", "toward", "from", "by", "at", "is",
         "are", "we", "our", "this", "that", "study", "paper", "novel", "new",
         "approach", "method", "methods", "analysis", "survey", "review"}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
_CJK_RE = re.compile(r"[一-鿿]")


def _norm_term(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _title_phrases(title: str, max_len: int = 4) -> list[str]:
    """把标题切成连续的实词短语（2~max_len 个词），停用词处断开。

    整条标题当查询太长会稀释信号，单个词又太碎；连续实词短语
    （"near-field channel estimation"）是这里最有用的粒度。
    """
    words = [w for w in _WORD_RE.findall(title or "")]
    out, run = [], []
    for w in words + [""]:
        if w and w.lower() not in _STOP:
            run.append(w)
            continue
        for n in range(2, max_len + 1):
            for i in range(len(run) - n + 1):
                out.append(" ".join(run[i:i + n]))
        run = []
    return out


def expand_queries(question: str, paper_ids: list[int], n: int = 3,
                   top_papers: int = 5) -> list[str]:
    """从首轮召回文献派生 n 条补充查询（确定性，不调模型）。

    打分口径：一个词组出现在越多头部文献里越可信（文档频次），
    并排除已经在原问题里出现过的词组——重复它不会带来新的召回。
    """
    q_norm = _norm_term(question)
    df: dict[str, int] = {}
    for pid in paper_ids[:top_papers]:
        with db.conn() as c:
            row = c.execute("SELECT title, card_json FROM papers WHERE id=?",
                            (pid,)).fetchone()
        if not row:
            continue
        try:
            card = json.loads(row["card_json"] or "{}")
        except json.JSONDecodeError:
            card = {}
        terms = [k for k in (card.get("keywords") or []) if isinstance(k, str)]
        terms += _title_phrases(row["title"] or "")
        # sorted() 不能省：直接遍历 set 会让 df 的插入顺序随字符串哈希种子逐进程变化，
        # 下面的稳定排序在同频次时按插入序决胜负 → 同一条查询两次跑出不同结果。
        # （踩过：同一份评测集连跑两次 Recall@15 一次 0.8281 一次 0.7969。）
        for t in sorted({_norm_term(t) for t in terms}):   # 同一篇内去重 = 文档频次
            if len(t) < 4 or t in q_norm:
                continue
            df[t] = df.get(t, 0) + 1
    # 文档频次优先，同频次时长短语优先（更具体），仍并列则按字典序——必须全序，否则不可复现
    ranked = sorted(df, key=lambda t: (-df[t], -len(t), t))
    picked: list[str] = []
    for t in ranked:
        if any(t in p or p in t for p in picked):     # 别选一堆彼此包含的近义词组
            continue
        picked.append(t)
        if len(picked) >= n:
            break
    return picked


def _rrf(rankings: list[tuple[list[int], float]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda p: -scores[p])


def deep_retrieve(query: str, top_k: int = 5, expand_n: int = 2,
                  fusion: str = "append") -> tuple[list[int], str, dict]:
    """单轮检索 + 派生查询补位。返回 (paper_ids, mode, trace)。全程 0 token，离线可跑。

    `fusion="append"`（默认）：**线上单轮的排序原封不动占前排，派生结果只填尾巴**——
    结构上不可能比单轮差，只在单轮召不满 top_k 时才起作用。
    `fusion="rrf"`：把派生查询加权融进主排序（消融口径，见下）。

    为什么默认不是 RRF——实测说话（eval_set 32 条检索题、库内 455 篇、纯 FTS 底座、
    0 次 LLM 调用，`cli.py eval --retrieval {fts,deep-rrf,deep}` 可复现）：

    | 口径            | Recall@5 | @10    | @15        | @20    |
    |-----------------|----------|--------|------------|--------|
    | 单轮（线上）    | 0.6771   | 0.7396 | 0.7656     | 0.7656 |
    | 派生 + RRF 融合 | 0.5990   | 0.6771 | 0.7656     | 0.7969 |
    | 派生只补位      | 0.6771   | 0.7708 | **0.8281** | 0.8281 |

    RRF 融合在小 k 上明显更差：派生词太宽（"large language model" 这种），
    典型的 query drift——PRF 独有候选里 gold 只占 0.9%，把它们提上来就是
    把 gold 挤下去。所以补位是唯一站得住的用法：小 k 与线上逐条相同（不可能变差），
    取宽上下文（k≥15，RAG 喂料常用档）时 Recall 从 0.766 提到 0.828。
    """
    pool = max(top_k, 15)
    _r = embeddings.search_hybrid(query, pool)
    primary, mode, notes = _r.ids, _r.mode, _r.degraded
    # primary_ids 留给 deep_answer 当补检索的闸门用——这次检索本来就要做，
    # 记下来就不用为了闸门再打一次，全程仍是 0 次额外检索。
    trace: dict = {"query": query, "primary_hits": len(primary),
                   "primary_ids": list(primary),
                   # 检索这一路的降级要跟着 trace 走：deep_answer 之后是拿
                   # candidate_ids 调 rag.prepare 的，prepare 不会再检索一次，
                   # 不放这儿这条信号就在这里断掉了。
                   "degraded": degrade.as_dicts(notes),
                   "fusion": fusion, "expansions": [], "added": 0}
    terms = expand_queries(query, primary, expand_n)
    trace["expansions"] = terms
    if not terms:
        return primary[:top_k], mode, trace

    rankings: list[tuple[list[int], float]] = [(primary, ORIGINAL_WEIGHT)]
    appended: list[int] = list(primary)
    for term in terms:
        ids = embeddings.search_hybrid(term, pool).ids
        rankings.append((ids, EXPANSION_WEIGHT))
        for pid in ids:
            if pid not in appended:
                appended.append(pid)
                trace["added"] += 1

    final = _rrf(rankings) if fusion == "rrf" else appended
    trace["mode"] = f"{mode}+prf-{fusion}"
    return final[:top_k], trace["mode"], trace


# ── 带自反馈的迭代问答（需要 LLM）──

CRITIC_SYSTEM = """你在检查一份基于文献的回答还缺哪些证据。
给你：用户问题、当前回答、以及回答里「没有 [n] 引用支撑」的句子。
请判断为了补上这些空缺，还应该去文献库里检索什么。
输出 JSON：{"gaps":["还缺什么证据，一句一条"],"queries":["检索词1","检索词2"]}
queries 最多 3 条，要具体（学术术语优先，中英文都可以），不要重复用户原问题的措辞。
如果现有回答的证据已经够了，queries 给空数组。只输出 JSON。"""


def _unsupported_claims(answer: str) -> list[str]:
    """机械判定「没有 [n] 支撑的论断句」——复用评测里那套口径，不另发明一套。"""
    from .eval import split_claims
    _supported, unsupported = split_claims(answer or "")
    return unsupported


def deep_answer(question: str, top_k: int = 5, max_rounds: int = 2,
                progress=None) -> dict:
    """迭代问答：检索 → 作答 → 机械找证据缺口 → 模型提补充查询 → 补检索 → 重答。

    收敛条件（任一满足即停）：没有无证据论断、模型判定证据已足、补检索没带来新文献
    （含被闸门全部挡下）、达到 max_rounds。每轮的查询 / 新增文献 / 被挡下的条数 /
    无证据论断数都记在 trace 里，可审计。上下文篇数受 MAX_CONTEXT_PAPERS 硬闸门约束。
    """
    from . import rag
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM（.env 的 LLM_API_BASE / LLM_API_KEY / LLM_MODEL）")

    ids, mode, rtrace = deep_retrieve(question, top_k)
    trace = {"retrieval": rtrace, "iterations": []}
    answer, sources, degraded = "", [], None
    notes: list = []
    # deep_retrieve 那一路的降级，往下每一轮都要带着
    retrieval_notes = degrade.from_dicts(rtrace.get("degraded") or [])
    seen = list(ids)
    gaps: list[str] = []
    # 补检索的闸门集合：原问题自己的宽召回。deep_retrieve 里已经算过，直接取。
    base_set = set(rtrace.get("primary_ids") or ids)

    for step in range(1, max_rounds + 1):
        if progress:
            progress(min(0.15 + 0.7 * (step - 1) / max_rounds, 0.9),
                     f"round{step}", f"第 {step} 轮作答（上下文 {len(seen)} 篇）")
        ctx = rag.prepare(
            [{"role": "user", "content": question}],
            top_k=min(len(seen), MAX_CONTEXT_PAPERS), candidate_ids=seen)
        sources = ctx.sources
        notes = degrade.merge(retrieval_notes, ctx.notes)
        degraded = degrade.render(notes)
        answer = llm.chat(ctx.system, ctx.query, purpose=f"deep_answer_r{step}",
                          temperature=0.3)
        gaps = _unsupported_claims(answer)
        entry = {"round": step, "papers_in_context": len(seen),
                 "unsupported_claims": len(gaps), "queries": [], "new_papers": 0}
        trace["iterations"].append(entry)
        if not gaps or step == max_rounds:
            entry["stop"] = "无证据论断已清零" if not gaps else "达到轮数上限"
            break
        queries, reason = _propose_queries(question, answer, gaps)
        if reason == "enough":
            # CRITIC_SYSTEM 约定「证据够了就给空数组」——这才是 docstring 承诺的
            # 收敛条件。原来它和「自评调用失败」共用一个空列表，于是永远不可达。
            entry["stop"] = "模型判定证据已足"
            break
        if reason == "error":
            queries = expand_queries(question, seen, 2)
            entry["queries_from"] = "确定性派生（自评调用失败）"
        if not queries:
            entry["stop"] = "没有可用的补充查询"
            break
        entry["queries"] = queries
        added: list[int] = []
        for query in queries:
            more = embeddings.search_hybrid(query, top_k).ids
            added += [p for p in more if p not in seen and p not in added]
        # 这些 query 来自上一轮**没有证据支撑因而可能是编造的**论断句，召回的论文与
        # 该论断词面相近；无过滤地进上下文，下一轮模型就会把它们当成对该论断的「佐证」。
        # 闸门口径与 deep_retrieve 的 append 纪律一致：派生查询独有的候选里 gold 只占
        # 0.9%（见上面的实测表），所以只放行同时落在原问题宽召回里的那些。0 token。
        kept = [p for p in added if p in base_set]
        entry["gated_out"] = len(added) - len(kept)
        entry["new_papers"] = len(kept)   # 记裁剪后的实际入库数，原来记的是裁剪前
        if not kept:
            entry["stop"] = "补检索没有带来新文献"
            break
        seen += kept[:top_k]
        if len(seen) > MAX_CONTEXT_PAPERS:
            entry["dropped_for_budget"] = len(seen) - MAX_CONTEXT_PAPERS
            seen = seen[:MAX_CONTEXT_PAPERS]   # 首轮召回天然在前，保留；按加入序淘汰

    # 收口时仍有无证据论断，就如实追加在答案末尾——**标注不删改**，口径同 survey。
    # 原来这些句子和有 [n] 支撑的句子形态完全一样，直接混在正文里交付。
    unsupported = [g[:200] for g in gaps]
    if unsupported:
        answer = (answer.rstrip() + "\n\n---\n以下论断在库内未找到支撑，请自行核实：\n"
                  + "\n".join(f"- {g}" for g in unsupported))

    return {"answer": answer, "sources": sources, "degraded": degraded,
            "degraded_detail": degrade.as_dicts(notes),
            "retrieval_mode": mode, "rounds": len(trace["iterations"]),
            "unsupported": unsupported, "trace": trace}


def _propose_queries(question: str, answer: str,
                     gaps: list[str]) -> tuple[list[str], str]:
    """返回 (补充查询, 原因)。原因是 ``"ok"`` / ``"enough"`` / ``"error"`` 三态。

    三态必须分开：CRITIC_SYSTEM 明确约定「证据已经够了就给空数组」，而 except 分支
    也返回空列表。原来两者返回值完全相同、走同一分支，后果有两个——
    docstring 承诺的收敛条件「模型不再提查询」**在代码里不可达**；
    trace 里「自评调用失败」和「模型判定证据已足」记成同一条，事后不可区分。
    """
    user = (f"用户问题：{question}\n\n当前回答：\n{answer[:4000]}\n\n"
            "以下句子没有 [n] 引用支撑：\n- " + "\n- ".join(g[:120] for g in gaps[:8]))
    try:
        raw = llm.chat(CRITIC_SYSTEM, user, purpose="deep_critic", temperature=0.2)
        data = llm.extract_json(raw)
    except Exception:
        return [], "error"   # 自评失败不该让整轮问答失败——上层退回确定性派生
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
    return (queries[:3], "ok") if queries else ([], "enough")
