"""Chapter-level evidence retrieval for the writing pipeline.

The evidence agent is deliberately deterministic.  It retrieves existing
chunks and formula records, then verifies every quote against the stored page
text before binding it to a claim.  An LLM may consume the resulting bundle,
but it is not allowed to manufacture evidence here.
"""
from __future__ import annotations

import re
import json
from collections import defaultdict
from typing import Any

from . import cite, db
from .writing_schemas import Claim, Evidence, EvidenceBundle


_BLOCK_TYPES = {"text", "table", "formula", "figure", "reference"}
_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")


def _as_int(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _claims(section: dict) -> list[Claim]:
    """Normalize old outline points and P0 claim records into Claim objects."""
    raw = section.get("claims")
    if not isinstance(raw, list) or not raw:
        raw = section.get("points") or []
    out: list[Claim] = []
    sec_no = section.get("no") or section.get("section_id") or "1"
    for i, item in enumerate(raw, 1):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("claim") or "").strip()
            cid = str(item.get("claim_id") or f"sec-{sec_no}-claim-{i}")
            ctype = str(item.get("claim_type") or "section_point")
        else:
            text, cid, ctype = str(item).strip(), f"sec-{sec_no}-claim-{i}", "section_point"
        if not text:
            continue
        try:
            out.append(Claim(claim_id=cid[:80], text=text[:5000],
                             claim_type=ctype[:80], status="needs_review"))
        except Exception:
            # A malformed model claim must not prevent the other claims from
            # receiving evidence.
            continue
    return out


def _query(section: dict, claims: list[Claim]) -> str:
    title = str(section.get("title") or "").strip()
    points = [c.text for c in claims]
    return " ".join(x for x in [title, *points] if x)[:5000]


def _candidate_ids(section: dict, candidate_ids: list[int] | None,
                   hits: dict[int, list[dict]]) -> list[int]:
    values = candidate_ids if candidate_ids is not None else section.get("paper_ids")
    if isinstance(values, (str, int)):
        values = [values]
    out: list[int] = []
    for value in values or []:
        pid = _as_int(value)
        if pid and pid not in out:
            out.append(pid)
    if out:
        return out
    # No outline restriction: use the papers that actually contain matching
    # chunks.  Stable ordering makes bundles/cache reproducible.
    return sorted(hits, key=lambda pid: (hits[pid][0].get("rank", 10**9), pid))


def _source_kind(paper_id: int, page: int | None, row_kind: str,
                 model_text: bool) -> str:
    """Classify the provenance of an evidence match.

    The writer boundary only trusts ``source_kind in {"native_page", "imported_fulltext"}``.
    Other kinds are still useful for recall but cannot by themselves verify a
    citation.  ``model_text`` flags content that came from a model-generated
    summary (cards, table/figure summaries) and never counts as primary
    evidence.
    """
    if model_text:
        return "model_summary"
    if row_kind == "imported_fulltext":
        return "imported_fulltext"
    if page is not None:
        return "native_page"
    return "abstract"


def _contains_in_source(c, paper_id: int, quote: str, page: int | None = None,
                        *, require_independent_source: bool = True,
                        source_kind_hint: str | None = None) -> tuple[bool, str]:
    """Return (verified, source_kind) for ``quote`` against the paper.

    When ``require_independent_source`` is True (writer boundary default), the
    match must come from a non-model source.  Model-generated card_json,
    table_summaries and figure_summaries are never eligible.  Native pages win
    over abstract which wins over imported fulltext; imported fulltext beats
    never trusting the quote at all.
    """
    q = cite._norm(quote)
    if len(q) < 2:
        return False, "abstract"
    page_match = False
    params: list[Any] = [paper_id]
    sql = "SELECT text FROM pages WHERE paper_id=?"
    if page is not None:
        sql += " AND page_no=?"
        params.append(page)
    for row in c.execute(sql, params).fetchall():
        if q in cite._norm(row["text"] or ""):
            page_match = True
            break
    if page_match and require_independent_source:
        # A page hit is only as good as its source: native pages and OCR pages
        # both qualify; imported_fulltext is treated as independent because the
        # original importer validates it.  Chunks or model-derived summaries
        # comingled in the row are not promoted here.
        return True, "native_page"
    abstract_match = False
    row = c.execute("SELECT abstract FROM papers WHERE id=?",
                    (paper_id,)).fetchone()
    if row and q in cite._norm(row["abstract"] or ""):
        abstract_match = True
    if abstract_match and not require_independent_source:
        return True, source_kind_hint or "abstract"
    # Imported fulltext rows (e.g. older QASPER imports) provide an independent
    # full text source.  Skip when a real native page is present: we already
    # said the page check failed, but if the page table is empty we still want
    # imported rows to take over.
    has_pages = c.execute("SELECT 1 FROM pages WHERE paper_id=? LIMIT 1",
                          (paper_id,)).fetchone()
    imported = c.execute(
        "SELECT 1 FROM chunks WHERE paper_id=? AND kind='imported_fulltext' LIMIT 1",
        (paper_id,)).fetchone()
    if (not has_pages) and imported:
        for row in c.execute(
                "SELECT text FROM chunks WHERE paper_id=? "
                "AND kind='imported_fulltext'",
                (paper_id,)).fetchall():
            if q in cite._norm(row["text"] or ""):
                return True, "imported_fulltext"
    # Fallback: lenient verification used only for recall.  Never mark
    # ``verified=True`` through this path.  Source kind is preserved so callers
    # can label the evidence as ``unverified_recall``.
    return False, "unverified_recall"


def _score(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z][a-z0-9-]{2,}|[一-鿿]{2,}", (query or "").lower()))
    t = set(re.findall(r"[a-z][a-z0-9-]{2,}|[一-鿿]{2,}", (text or "").lower()))
    return len(q & t) / max(len(q), 1)


class EvidenceAgent:
    """Retrieve and mechanically verify evidence for one outline section."""

    def build_bundle(self, section: dict,
                     candidate_ids: list[int] | None = None) -> EvidenceBundle:
        if hasattr(section, "model_dump"):
            section = section.model_dump()
        section = section if isinstance(section, dict) else {"title": str(section)}
        claims = _claims(section)
        query = _query(section, claims)
        # Retrieve more than the final bundle cap, then rank per claim.  The
        # query is intentionally run once for deterministic, cheap orchestration.
        with db.conn() as c:
            hits = db.search_chunks_hits(c, query, limit=80, per_paper=8)
            pids = _candidate_ids(section, candidate_ids, hits)
            if not pids and candidate_ids is None and not section.get("paper_ids") and query:
                # Fresh L0-only libraries have titles/abstracts but no fulltext
                # chunks yet.  Let those papers participate via the same
                # abstract fallback used for an explicitly named candidate.
                pids = [int(r["id"]) for r in db.search_fts(c, query, 20)]
            rows: list[dict] = []
            for pid in pids:
                for hit in hits.get(pid, []):
                    if (hit.get("block_type") or "text") == "reference":
                        continue
                    rows.append({**hit, "paper_id": pid})
                # A few importers/tests populate ``chunks`` directly without
                # rebuilding chunks_fts.  Keep the agent useful (and honest)
                # by falling back to the canonical chunk table.
                if not any(r["paper_id"] == pid for r in rows):
                    for cr in c.execute(
                            "SELECT chunk_no,section_path,start_page,text,kind,block_type "
                            "FROM chunks WHERE paper_id=? AND kind!='reference' "
                            "ORDER BY chunk_no LIMIT 16", (pid,)).fetchall():
                        rows.append({"paper_id": pid, **dict(cr),
                                     "block_type": cr["block_type"] or cr["kind"] or "text"})
                # Formula rows are independently persisted and can be absent
                # from chunks when parsing was done before rechunking.
                for fr in db.formulas_for_paper(c, pid):
                    raw = str(fr.get("raw_text") or fr.get("latex") or "").strip()
                    if raw:
                        rows.append({"paper_id": pid, "chunk_no": None,
                                     "section_path": "", "start_page": fr.get("page_no"),
                                     "text": raw, "block_type": "formula",
                                     "formula_id": fr.get("block_id")})
                # A model-written table/figure summary is useful for recall,
                # but is never marked verified unless its words occur in source.
                for sr in c.execute("SELECT page_no,summary FROM table_summaries WHERE paper_id=?",
                                     (pid,)).fetchall():
                    if sr["summary"]:
                        rows.append({"paper_id": pid, "chunk_no": None,
                                     "section_path": "", "start_page": sr["page_no"],
                                     "text": str(sr["summary"]), "block_type": "table",
                                     "summary_only": True})
                for sr in c.execute("SELECT page_no,caption,summary FROM figure_summaries WHERE paper_id=?",
                                     (pid,)).fetchall():
                    caption = str(sr["caption"] or "").strip()
                    if caption:
                        rows.append({"paper_id": pid, "chunk_no": None,
                                     "section_path": "", "start_page": sr["page_no"],
                                     "text": caption, "block_type": "figure"})
                    summary = str(sr["summary"] or "").strip()
                    if summary:
                        rows.append({"paper_id": pid, "chunk_no": None,
                                     "section_path": "", "start_page": sr["page_no"],
                                     "text": summary, "block_type": "figure",
                                     "summary_only": True})
            # If an outline names a paper but no full text has been indexed,
            # its abstract is still a real, mechanically verifiable source.
            for pid in pids:
                if not any(r["paper_id"] == pid for r in rows):
                    r = c.execute("SELECT abstract FROM papers WHERE id=?", (pid,)).fetchone()
                    if r and r["abstract"]:
                        rows.append({"paper_id": pid, "chunk_no": None,
                                     "section_path": "摘要", "start_page": None,
                                     "text": r["abstract"], "block_type": "text"})

            evidence: list[Evidence] = []
            by_claim: dict[str, list[str]] = defaultdict(list)
            # Rank candidates for every claim independently.  This prevents a
            # single broad section query from starving a less frequent claim.
            for claim in claims:
                ranked = sorted(rows, key=lambda r: (
                    -_score(claim.text, r.get("text") or ""),
                    r.get("paper_id", 0), r.get("chunk_no") or 0))
                seen: set[tuple] = set()
                for row in ranked:
                    pid = _as_int(row.get("paper_id"))
                    quote = (row.get("text") or "").strip()
                    if not pid or not quote:
                        continue
                    # At most two blocks per claim/paper; a verified quote is
                    # always preferred over an unverified generated summary.
                    key = (pid, row.get("chunk_no"), row.get("block_type"), quote)
                    if key in seen:
                        continue
                    seen.add(key)
                    page = _as_int(row.get("start_page"))
                    is_model_text = bool(row.get("summary_only"))
                    if is_model_text:
                        verified = False
                        source_kind = "model_summary"
                        page_verified = False
                    else:
                        verified, source_kind = _contains_in_source(
                            c, pid, quote, page,
                            require_independent_source=True)
                        # A page-verified match means the citation's claimed
                        # page actually contained the quote.  Absence of pages
                        # or successful page match implies abstract-level only.
                        page_verified = bool(
                            page and verified and source_kind == "native_page")
                    score = _score(claim.text, quote)
                    btype = row.get("block_type") or "text"
                    if btype not in _BLOCK_TYPES:
                        btype = "text"
                    structured = any(term in claim.text.lower() for term in
                                     ("table", "formula", "equation", "figure",
                                      "表", "公式", "方程", "图"))
                    if score <= 0 and not (structured and btype in {"table", "formula", "figure"}):
                        continue
                    eid = f"evidence-sec{section.get('no') or section.get('section_id') or 1}-{len(evidence)+1}"
                    ev = Evidence(evidence_id=eid, paper_id=pid,
                                  quote=quote[:20000], page=page,
                                  section=str(row.get("section_path") or "")[:300],
                                  block_type=btype, verified=verified,
                                  source_kind=source_kind,
                                  page_verified=page_verified,
                                  confidence=round(min(1.0, score + (0.45 if verified else 0.0)), 3))
                    evidence.append(ev)
                    by_claim[claim.claim_id].append(eid)
                    # Keep the bundle compact while covering distinct papers
                    # and structured blocks.
                    if len(by_claim[claim.claim_id]) >= 4:
                        break

            out_claims: list[Claim] = []
            verified_by_id = {e.evidence_id: (e.verified, e.source_kind) for e in evidence}
            for claim in claims:
                ids = by_claim.get(claim.claim_id, [])
                verified_n = 0
                contradicted = False
                for eid in ids:
                    is_verified, source_kind = verified_by_id.get(eid, (False, "unverified_recall"))
                    if is_verified and source_kind in {"native_page", "imported_fulltext"}:
                        verified_n += 1
                    elif source_kind == "model_summary":
                        contradicted = True
                if contradicted and not verified_n:
                    status = "contradicted"
                elif verified_n:
                    status = "supported"
                elif ids:
                    status = "partial"
                else:
                    status = "unsupported"
                out_claims.append(claim.model_copy(update={"evidence_ids": ids,
                                                            "status": status}))
        covered = sum(c.status == "supported" for c in out_claims)
        coverage = covered / len(out_claims) if out_claims else 0.0
        sid = str(section.get("section_id") or section.get("no") or section.get("title") or "section")
        section_id = sid if sid.startswith("section-") else f"section-{sid}"
        return EvidenceBundle(section_id=section_id[:80], query=query,
                              evidence=evidence[:100], claims=out_claims[:100],
                              coverage=round(coverage, 3))


def build_evidence_bundle(section: dict,
                          candidate_ids: list[int] | None = None) -> EvidenceBundle:
    """Convenience API used by tests and the writing pipeline."""
    return EvidenceAgent().build_bundle(section, candidate_ids)


__all__ = ["EvidenceAgent", "build_evidence_bundle"]
