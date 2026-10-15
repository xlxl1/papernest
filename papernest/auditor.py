"""Deterministic checks at the claim/evidence boundary of writing agents.

These checks can establish that a citation belongs to a claim and that a
reported number occurs in its verified quote. They do not establish semantic
entailment: paraphrases and changes of experimental conditions still require
review. No model call, database access, or fact-changing repair occurs here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .writing_schemas import ConsistencyIssue


_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s*(?:[%％]|percent\b|per\s+cent\b))?",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"待补证据|离线模式|离线模板|库内已有|库内暂无|offline[- ]template", re.I)
_SPACE_HYPHEN_RE = re.compile(r"[\s_\-‐‑–—]+")
_WORD_RE = re.compile(r"[a-z][a-z0-9]+|[一-鿿]", re.I)
_STOP_WORDS = {"a", "an", "the", "and", "of", "to", "in", "for", "is", "are",
               "with", "as", "on", "by", "at", "it", "this", "that", "we"}


def _mapping(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n > 0 else None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]", "", text.casefold())


def _sentences(text: str):
    """Yield (line number, assertion) without Markdown structural numbers."""
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip() or re.match(r"^\s*#{1,6}\s", line):
            continue
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)", "", line)
        # A decimal point is not a sentence separator. Keep citations placed
        # after punctuation attached to the preceding sentence as well.
        # Do not detach a citation that follows Chinese punctuation (``。 [1]``
        # or ``。[1]``); the citation is part of the assertion immediately
        # before it. The lookahead still allows punctuation followed directly
        # by the next Chinese sentence to split.
        parts = re.split(r"(?<=[。！？!?；;])(?!\s*\[\d+\])|"
                         r"(?<=\.)\s+(?=[A-Z一-鿿])", line)
        for part in parts:
            if part.strip():
                yield line_no, part.strip()


def _numbers(text: str) -> dict[str, str]:
    """Map normalized values to original spellings; units remain significant."""
    clean = _CITATION_RE.sub("", text)
    # Inline math identifiers and version strings are not quantitative
    # statements. Percentages and plain scalar values are intentionally kept
    # distinct; finding 82 in a quote does not verify a claim of 82%.
    clean = re.sub(r"\b[A-Za-z]+(?:[-_.]\d+)+\b", "", clean)
    result: dict[str, str] = {}
    for match in _NUMBER_RE.finditer(clean):
        raw = match.group(0).strip()
        percentage = bool(re.search(r"[%％]|percent\b|per\s+cent\b", raw, re.I))
        scalar = re.sub(r"\s*(?:[%％]|percent\b|per\s+cent\b)$", "", raw, flags=re.I)
        try:
            number = Decimal(scalar.replace(",", ""))
        except InvalidOperation:
            continue
        if not number.is_finite():
            continue
        normalized = format(number.normalize(), "f")
        if number == 0:
            normalized = "0"
        result[normalized + ("%" if percentage else "")] = raw
    return result


def _location(section_id: str, line_no: int) -> str:
    return f"{section_id}:line-{line_no}"


def _bound_evidence(bundle: dict):
    evidence = {}
    for raw in bundle.get("evidence") or []:
        item = _mapping(raw)
        eid = str(item.get("evidence_id") or "")
        if eid:
            evidence[eid] = item
    claims = [_mapping(c) for c in bundle.get("claims") or []]
    claims = [c for c in claims if c.get("text")]
    by_claim: dict[str, list[dict]] = {}
    bound: dict[str, dict] = {}
    for i, claim in enumerate(claims, 1):
        cid = str(claim.get("claim_id") or f"claim-{i}")
        ids = claim.get("evidence_ids") or []
        if isinstance(ids, (str, int)):
            ids = [ids]
        eligible = []
        for eid in ids:
            ev = evidence.get(str(eid))
            if (ev and ev.get("verified") is True and str(ev.get("quote") or "").strip()
                    and _positive_int(ev.get("paper_id"))
                    and ev.get("block_type") != "reference"):
                eligible.append(ev)
                bound[str(eid)] = ev
        by_claim[cid] = eligible
    # Preserve the bundle ordering, which is also the writer's numbering.
    ordered = [ev for eid, ev in evidence.items() if eid in bound]
    return claims, by_claim, ordered


def _citation_map(citations: Any, evidence: list[dict]) -> dict[int, dict]:
    if citations is None:
        entries = {}
        seen = set()
        for ev in evidence:
            pid = _positive_int(ev.get("paper_id"))
            if pid not in seen:
                seen.add(pid)
                entries[len(entries) + 1] = {"paper_id": pid, "verified": True}
        return entries
    raw = _mapping(citations)
    if isinstance(citations, list):
        raw = {str(i): item for i, item in enumerate(citations, 1)}
    return {no: _mapping(entry) for key, entry in raw.items()
            if (no := _positive_int(key)) is not None}


def _claim_matches(sentence: str, claim: dict) -> bool:
    """Recognize literal claims and close lexical restatements, conservatively."""
    text = str(claim.get("text") or "")
    left, right = _normalize(_CITATION_RE.sub("", sentence)), _normalize(text)
    if len(right) >= 4 and right in left:
        return True
    # Requiring nearly every claim token avoids declaring unrelated outline
    # points a semantic match simply because both mention a broad topic.
    claim_tokens = {t.casefold() for t in _WORD_RE.findall(text)} - _STOP_WORDS
    sentence_tokens = {t.casefold() for t in _WORD_RE.findall(sentence)} - _STOP_WORDS
    return (len(claim_tokens) >= 3 and
            len(claim_tokens & sentence_tokens) / len(claim_tokens) >= 0.85)


def _term_issues(text: str, glossary: Any, section_id: str) -> list[ConsistencyIssue]:
    issues = []
    for canonical, config in _mapping(glossary).items():
        canonical = str(canonical).strip()
        if not canonical:
            continue
        explicit_config = isinstance(config, dict)
        aliases = config.get("aliases", []) if explicit_config else config
        # Existing glossaries store source-count descriptions as bare strings;
        # those are not aliases. A string is accepted only under an explicit
        # ``{"aliases": ...}`` declaration; lists are the preferred form.
        if isinstance(aliases, str):
            aliases = [aliases] if explicit_config else []
        aliases = aliases if isinstance(aliases, (list, tuple)) else []
        variants = [canonical, *(str(a) for a in aliases if a)]
        seen: set[tuple[int, str]] = set()
        for variant in variants:
            pieces = [re.escape(p) for p in _SPACE_HYPHEN_RE.split(variant.strip()) if p]
            if not pieces:
                continue
            pattern = r"[\s_\-‐‑–—]*".join(pieces)
            if variant[0].isascii() and variant[0].isalnum():
                pattern = r"(?<![A-Za-z0-9_])" + pattern
            if variant[-1].isascii() and variant[-1].isalnum():
                pattern += r"(?![A-Za-z0-9_])"
            for match in re.finditer(pattern, text, flags=re.I):
                actual = match.group(0)
                if actual == canonical:
                    continue
                line_no = text.count("\n", 0, match.start()) + 1
                key = (line_no, actual)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(ConsistencyIssue(
                    issue_type="term_conflict", severity="minor",
                    locations=[_location(section_id, line_no)],
                    description=f"术语“{actual}”与术语表规定的“{canonical}”不一致。",
                    fix=f"将“{actual}”统一为“{canonical}”。",
                ))
    return issues


def _unique(issues: Iterable[ConsistencyIssue]) -> list[ConsistencyIssue]:
    seen = set()
    result = []
    for issue in issues:
        key = (issue.issue_type, tuple(issue.locations), issue.description)
        if key not in seen:
            result.append(issue)
            seen.add(key)
    return result


def audit_section(text: str, evidence_bundle: Any = None,
                  citations: Any = None, glossary: Any = None, *,
                  section_id: str | None = None) -> list[ConsistencyIssue]:
    """Check a section with its local citation mapping and evidence bundle.

    Numeric checks use only the verified, claim-bound quotes belonging to the
    papers cited in the same sentence. If a recognizable claim is present,
    the boundary narrows further to that claim's quotes. Historical sections
    without a bundle receive citation/term checks only. An absent outline
    point is not reported as an omitted claim because a paraphrase cannot be
    mechanically ruled out.
    """
    text = str(text or "")
    bundle = _mapping(evidence_bundle)
    sid = str(section_id or bundle.get("section_id") or "section")
    has_bundle = "claims" in bundle or "evidence" in bundle
    claims, by_claim, bound = _bound_evidence(bundle)
    refs = _citation_map(citations, bound)
    issues = _term_issues(text, glossary, sid)
    # Surface provenance risk in the audit: a citation that traces to a
    # verified piece of evidence whose source is not a native page can still
    # be wrong (model summaries, abstract fallbacks).  These are reported
    # as ``minor`` so the coordinator knows they exist without forcing a
    # full rewrite, and the final report can show how many such citations
    # the run actually relied on.
    for ev in bound:
        kind = ev.get("source_kind")
        if kind not in {"model_summary", "unverified_recall", "abstract"}:
            continue
        pid = _positive_int(ev.get("paper_id"))
        if not pid:
            continue
        if not any(_positive_int((entry or {}).get("paper_id")) == pid
                   for entry in refs.values()):
            continue
        issues.append(ConsistencyIssue(
            issue_type="provenance_weak", severity="minor",
            locations=[sid, f"paper:{pid}"],
            description=(f"引用 {pid} 的证据来源级别为 {kind}，未在原生页或导入全文中复核；"
                         "本页可作为草稿交付，但下游引用应尽量替换为 page-verified 证据。"),
            fix=("重新检索该论文的章节块，将该引用替换为 source_kind=native_page 的 quote；"
                 "若来源是摘要/模型卡片，请在文档里明示限制。"),
        ))
    bound_by_paper: dict[int, list[dict]] = {}
    for ev in bound:
        bound_by_paper.setdefault(int(ev["paper_id"]), []).append(ev)
    for line_no, sentence in _sentences(text):
        used = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(sentence)})
        loc = _location(sid, line_no)
        usable: dict[int, list[dict]] = {}
        for no in used:
            entry = refs.get(no)
            pid = _positive_int((entry or {}).get("paper_id"))
            if not entry or not pid:
                issues.append(ConsistencyIssue(
                    issue_type="citation_gap", severity="major",
                    locations=[loc, f"citation:[{no}]"],
                    description=f"引用 [{no}] 没有有效的文献映射。",
                    fix=f"为 [{no}] 补充已核验且绑定到主张的证据，或移除该引用及其无证据论断。",
                ))
            elif entry.get("verified") is not True:
                issues.append(ConsistencyIssue(
                    issue_type="citation_gap", severity="major",
                    locations=[loc, f"citation:[{no}]"],
                    description=f"引用 [{no}] 的来源尚未核验。",
                    fix="回取原文并核验引文；核验前将相关论断标为待补证据。",
                ))
            elif has_bundle and pid not in bound_by_paper:
                issues.append(ConsistencyIssue(
                    issue_type="citation_gap", severity="major",
                    locations=[loc, f"citation:[{no}]"],
                    description=f"引用 [{no}] 虽标记已核验，但没有绑定到任何主张的已核验证据。",
                    fix="将该论断绑定到确实支撑它的已核验证据；不要仅凭文献存在保留引用。",
                ))
            else:
                usable[no] = bound_by_paper.get(pid, [])
        if not has_bundle or _PLACEHOLDER_RE.search(sentence):
            continue
        matched = [claim for claim in claims if _claim_matches(sentence, claim)]
        relevant = []
        for claim in matched:
            cid = str(claim.get("claim_id") or "")
            evidence = by_claim.get(cid, [])
            allowed_papers = {int(ev["paper_id"]) for ev in evidence}
            cited_allowed = [no for no in usable
                             if _positive_int(refs[no].get("paper_id")) in allowed_papers]
            if not evidence:
                issues.append(ConsistencyIssue(
                    issue_type="citation_gap", severity="major",
                    locations=[loc, f"claim:{cid}"],
                    description=f"正文主张“{str(claim['text'])[:300]}”没有绑定到已核验证据。",
                    fix="补充并核验支撑该主张的引文，或将此句改为待补证据。",
                ))
            elif not cited_allowed:
                issues.append(ConsistencyIssue(
                    issue_type="claim_conflict" if used else "citation_gap", severity="major",
                    locations=[loc, f"claim:{cid}", *(f"citation:[{n}]" for n in used)],
                    description=f"正文主张“{str(claim['text'])[:300]}”未引用绑定到该主张的已核验来源。",
                    fix="使用该主张绑定证据所对应的引用编号，或补充能够支撑此句的证据。",
                ))
            relevant.extend(ev for ev in evidence
                            if int(ev["paper_id"]) in {
                                _positive_int(refs[n].get("paper_id")) for n in cited_allowed})
        if not used:
            continue
        # A different claim bound to the same cited paper cannot cover an
        # invented number in a recognized claim.
        candidates = relevant if matched else [ev for evs in usable.values() for ev in evs]
        if not candidates:
            continue  # The preceding citation gap already explains the failure.
        numbers = _numbers(sentence)
        verified_numbers = {value for ev in candidates for value in _numbers(str(ev["quote"]))}
        unsupported = [raw for value, raw in numbers.items() if value not in verified_numbers]
        if unsupported:
            issues.append(ConsistencyIssue(
                issue_type="numeric_conflict", severity="major",
                locations=[loc, *(f"citation:[{n}]" for n in used)],
                description=(f"正文数字 {'、'.join(unsupported)[:500]} 未出现在本句引用所对应的"
                             "主张绑定、已核验引文中；机械检查无法确认其来源。"),
                fix="回到本句绑定的引文核对数值及单位；补充来源或标为待补证据，不猜测替换数字。",
            ))
    return _unique(issues)


def audit_document(sections: Iterable[Any], glossary: Any = None) -> list[ConsistencyIssue]:
    """Audit persisted section rows and flag identical assertions with different numbers."""
    issues = []
    statements: dict[str, tuple[dict[str, str], str]] = {}
    for raw in sections:
        section = _mapping(raw)
        text = str(section.get("content") or section.get("text") or "")
        sid = str(section.get("section_id") or f"section-{section.get('sec_no') or section.get('no') or 'unknown'}")
        bundle = section.get("evidence_bundle", section.get("evidence_json"))
        refs = section.get("citations", section.get("citations_json"))
        issues.extend(audit_section(text, bundle, refs, glossary, section_id=sid))
        for line_no, sentence in _sentences(text):
            if _PLACEHOLDER_RE.search(sentence):
                continue
            values = _numbers(sentence)
            if not values:
                continue
            clean = _CITATION_RE.sub("", sentence)
            key = _normalize(_NUMBER_RE.sub(" NUMBER ", clean))
            # Require a substantial identical phrase. Different methods,
            # datasets, or evaluation conditions must not be conflated.
            if len(key) < 12:
                continue
            loc = _location(sid, line_no)
            previous = statements.get(key)
            if previous and set(previous[0]) != set(values) and previous[1].split(":line-")[0] != sid:
                issues.append(ConsistencyIssue(
                    issue_type="numeric_conflict", severity="minor",
                    locations=[previous[1], loc],
                    description=(f"不同章节的相同表述给出不同数字：{'、'.join(previous[0].values())[:300]}"
                                 f" 与 {'、'.join(values.values())[:300]}。"),
                    fix="核对是否对应不同实验条件、数据集或时期，并在各处明确条件与出处；不得直接统一数值。",
                ))
            else:
                statements[key] = (values, loc)
    return _unique(issues)


def summarize_issues(issues: Iterable[Any]) -> dict:
    items = [_mapping(issue) for issue in issues]
    counts = Counter(item.get("severity") for item in items)
    return {"ok": not counts["major"], "issue_count": len(items),
            "major_count": counts["major"], "minor_count": counts["minor"],
            "suggestion_count": counts["suggestion"], "issues": items}


def issues_to_feedback(issues: Iterable[Any]) -> list[dict]:
    """Adapt structured checks to the existing section-rewrite feedback contract."""
    return [{"severity": issue.get("severity", "minor"),
             "location": "、".join(issue.get("locations") or []),
             "problem": issue.get("description", ""), "fix": issue.get("fix", ""),
             "issue_type": issue.get("issue_type")}
            for raw in issues if (issue := _mapping(raw))]
