"""课题综述生成：基于库内结构化卡片撰写带 [n] 引用的综述草稿，并逐句做证据核验。

这是与「网页版 LLM 直接问答」的本质差异所在：
1. 综述只基于用户自己的文献库（跨会话沉淀的私有记忆），不是模型的泛泛而谈；
2. 每条被引用的论断句回到所引文献做相似度核验，核不上的如实标注 ?——输出可审计。
"""
import re

from . import config, db, embeddings, llm

SURVEY_SYSTEM = """你是学术综述撰写助手。仅依据给定的文献上下文，围绕研究课题撰写一篇结构化综述：
- 结构：研究现状 → 方法路线 → 代表工作 → 开放问题（用小标题分隔）；
- 每个来自文献的论断句尾标注来源编号 [n]，编号与上下文一一对应；
- 严禁引用上下文之外的文献，严禁编造文献中没有的结论或数字；
- 上下文确实没有覆盖的方向，在「开放问题」里如实指出「库内文献未覆盖」；
- 中文，总长 400-600 字。"""

CHECK_THRESHOLD = 0.35


def _context_blocks(ids: list[int]) -> tuple[str, list[dict], list[int]]:
    blocks, sources, valid_ids = [], [], []
    with db.conn() as c:
        for i, pid in enumerate(ids, 1):
            r = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if not r:
                continue
            try:
                card = __import__("json").loads(r["card_json"] or "{}")
            except Exception:
                card = {}
            lines = [f"[{i}] {r['title']}（{r['venue'] or ''} {r['year'] or ''}）"]
            for k in ("tldr", "method", "results"):
                if card.get(k):
                    lines.append(f"    {k}：{card[k]}")
            if r["abstract"]:
                lines.append(f"    摘要：{r['abstract'][:400]}")
            blocks.append("\n".join(lines))
            sources.append({"idx": i, "paper_id": pid, "title": r["title"],
                            "year": r["year"], "venue": r["venue"]})
            valid_ids.append(pid)
    return "\n\n".join(blocks), sources, valid_ids


def _check_claims(text: str, sources: list[dict]) -> list[dict]:
    """对每个带 [n] 的句子：与所引文献的句级向量算相似度，低于阈值标 ?。

    ``ok`` 必须是**三态**：True 通过 / False 未通过 / None 未核验（并给 reason）。
    `embeddings.best_sentence` 在一篇论文没有句级向量时返回 ``(None, 0.0)`` 而不抛异常，
    except 接不住，原来算出 ``bool(0.0 >= 0.35) = False``——于是「本库无句级向量」
    被记成「全部核验未通过」，是个把整篇综述判死刑的假警报。
    """
    sents = [s.strip() for s in re.split(r"(?<=[。！？!?])\s*", text) if "[" in s and "]" in s]
    checks = []
    for s in sents[:40]:
        m = re.findall(r"\[(\d{1,2})\]", s)
        if not m:
            continue
        idx = int(m[-1])
        src = next((x for x in sources if x["idx"] == idx), None)
        if not src:
            continue
        try:
            svec = embeddings.embed_texts([s[:500]])[0]
            import numpy as np
            svec = np.asarray(svec, dtype=np.float32)
            ev, score = embeddings.best_sentence(svec, src["paper_id"])
        except Exception as exc:
            checks.append({"claim": s[:80], "full": s, "idx": idx, "ok": None,
                           "score": None, "reason": f"核验调用失败：{type(exc).__name__}"})
            continue
        if ev is None:      # 这篇没有句级向量：**验不了**，不是验不过
            checks.append({"claim": s[:80], "full": s, "idx": idx, "ok": None,
                           "score": None, "reason": "该文献无句级向量"})
            continue
        checks.append({"claim": s[:80], "full": s, "idx": idx,
                       "ok": bool(score >= CHECK_THRESHOLD), "score": round(score, 3)})
    return checks


def _annotate(text: str, checks: list[dict]) -> str:
    """给未通过核验的论断句**句尾追加**标注——标注不删句，口径同 rcs.py:253。

    不这么做的话，相似度低于阈值的句子仍带着它的 [n] 留在正文里，形态与通过核验的
    句子完全一致；而 `agent.py` 的 `final_answer = data.get("survey")` 只取裸文本，
    平行的 checks 列表在下游直接丢失。
    """
    out = text
    for c in checks:
        if c.get("ok") is not False:
            continue
        full = c.get("full") or ""
        if full and full in out:
            out = out.replace(full, f"{full}（未通过证据核验，相似度 {c['score']}）", 1)
    return out


def generate(topic: str, top_k: int = 10, progress=None) -> dict:
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM")
    db.init_db()
    ids, degraded = None, None
    if progress:
        progress(0.1, "retrieve", "库内检索相关文献")
    # 综述取宽上下文（top_k=10），正是迭代检索的受益档：派生查询「只补位」，
    # 实测 Recall@10 0.740→0.771、@15 0.766→0.828，且小 k 与单轮逐条相同、0 token。
    from . import deepsearch
    ids, mode, _trace = deepsearch.deep_retrieve(topic, top_k)
    degraded = None
    if mode.startswith("fts") and embeddings.available():
        degraded = "向量检索调用失败，本次仅 FTS 词面检索"
    context, sources, _ = _context_blocks(ids)
    if progress:
        progress(0.35, "write", f"基于 {len(sources)} 篇文献撰写综述（重档模型）")
    user = (f"研究课题：{topic or config.RESEARCH_TOPIC}\n\n文献上下文：\n{context}")
    text = llm.chat(SURVEY_SYSTEM, user, purpose="survey", temperature=0.4,
                    model=config.heavy_model())
    if progress:
        progress(0.85, "verify", "逐句证据核验")
    checks = _check_claims(text, sources)
    ok_n = sum(1 for c in checks if c["ok"] is True)
    bad_n = sum(1 for c in checks if c["ok"] is False)
    ungraded = [c for c in checks if c["ok"] is None]
    parts = [f"核验：{ok_n} 条通过", f"{bad_n} 条未通过（阈值 {CHECK_THRESHOLD}）"]
    if ungraded:
        why = ungraded[0].get("reason") or "原因未知"
        parts.append(f"{len(ungraded)} 条未核验（{why}）")
    if not checks:
        parts = ["核验：本篇没有带 [n] 的论断句，未做核验"]
    return {
        "topic": topic, "survey": _annotate(text, checks), "survey_raw": text,
        "sources": sources, "checks": checks,
        "verified_summary": "，".join(parts),
        "degraded": degraded,
    }
