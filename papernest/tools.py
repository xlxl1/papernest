"""First-class tools used by the PaperNest Research Agent.

The functions in this module deliberately return plain dictionaries.  They
can therefore be called by the local orchestrator today and later be exposed
to an LLM tool-calling API without changing the domain modules.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from . import cite, db, degrade, embeddings, fulltext, rag, survey


def retrieve_library(query: str, top_k: int = 5) -> dict[str, Any]:
    """Retrieve papers from the local library using RRF hybrid search (vector + FTS)."""
    db.init_db()
    # 降级由 search_hybrid 直接给出。原来这里是 `mode == "fts"` 反推——
    # 有 chunk 命中时 mode 是 'fts+chunks'，等号恒假，降级被静默吞掉。
    _r = embeddings.search_hybrid(query, top_k)
    ids, mode, notes = _r.ids, _r.mode, _r.degraded
    degraded = degrade.render(notes)

    papers: list[dict[str, Any]] = []
    with db.conn() as c:
        for pid in ids:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if not row:
                continue
            try:
                authors = json.loads(row["authors"] or "[]")
            except (TypeError, json.JSONDecodeError):
                authors = []
            papers.append({
                "paper_id": pid,
                "title": row["title"],
                "year": row["year"],
                "venue": row["venue"],
                "score": None,
                "authors": authors,
                "has_fulltext": bool(c.execute(
                    "SELECT 1 FROM pages WHERE paper_id=? LIMIT 1", (pid,)
                ).fetchone()),
            })
    return {"paper_ids": [p["paper_id"] for p in papers],
            "papers": papers, "retrieval_mode": mode, "degraded": degraded,
            "degraded_detail": degrade.as_dicts(notes)}


def answer_question(goal: str, top_k: int = 5,
                    candidate_ids: list[int] | None = None) -> dict[str, Any]:
    """Generate a grounded answer, optionally from a previous retrieval step."""
    messages = [{"role": "user", "content": goal}]
    return rag.answer(messages, top_k=top_k, candidate_ids=candidate_ids)


def recommend_citations(paragraph: str, top_k: int = 5) -> dict[str, Any]:
    return cite.recommend(paragraph, top_k)


def generate_survey(topic: str, top_k: int = 10) -> dict[str, Any]:
    return survey.generate(topic, top_k)


def read_paper(paper_id: int, topic: str | None = None) -> dict[str, Any]:
    return fulltext.read_paper(paper_id, topic)


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "retrieve_library": retrieve_library,
    "answer_question": answer_question,
    "recommend_citations": recommend_citations,
    "generate_survey": generate_survey,
    "read_paper": read_paper,
}

