"""Formula candidates, optional local Math OCR, and canonical Markdown.

Candidate text is audit evidence, not trusted LaTeX. Math OCR is disabled unless
PAPERNEST_MATH_OCR_MODULE names a local plugin exposing recognize_formula.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re

# Bump when candidate grouping/false-positive rules change so previously
# persisted formula audits and canonical Markdown can be rebuilt explicitly.
PARSER_VERSION = "pdf-canonical-v2"
TRUSTED_LATEX_CONFIDENCE = 0.80
_MATH_FONT = re.compile(r"math|symbol|stix|cambria|cmex|cmmi|cmsy|msam|msbm|euclid", re.I)
_FORMULA_NO = re.compile(r"(?:^|\s)[（(]([A-Za-z]?\d+(?:[.\-]\d+)*)[)）]\s*$")
_MATH_CHARS = set("=<>±×÷∑∏∫√∞≈≠≤≥∂∇∈∉⊂⊆∪∩→←↔⇒⇔∀∃∝⊥∥⊗⊕⋅·")
_GREEK = re.compile(r"[Α-Ωα-ωϑϕϵϱ]")


def _bbox(value):
    vals = list(value or (0, 0, 0, 0))[:4] + [0, 0, 0, 0]
    return tuple(round(float(v or 0), 2) for v in vals[:4])


def _block_id(page_no, bbox, text, source="pdf_text"):
    raw = json.dumps([page_no, _bbox(bbox), text.strip(), source], ensure_ascii=False)
    return "formula-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def _join_spans(spans):
    parts, previous = [], None
    for span in sorted(spans, key=lambda s: _bbox(s.get("bbox"))[0]):
        text = span.get("text") or ""
        if not text:
            continue
        x0, _, x1, _ = _bbox(span.get("bbox"))
        size = float(span.get("size") or 0)
        if parts and previous and x0 - previous[0] > max(previous[1], size, 6) * .35:
            parts.append(" ")
        parts.append(text)
        previous = (x1, size)
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def score_formula_line(text, spans):
    text = (text or "").strip()
    real = [s for s in spans if (s.get("text") or "").strip()]
    sizes = [float(s.get("size") or 0) for s in real]
    bottoms = [_bbox(s.get("bbox"))[3] for s in real]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    median_bottom = sorted(bottoms)[len(bottoms) // 2] if bottoms else 0
    script = any(size <= median_size * .82 and abs(bottom - median_bottom) >= max(1, median_size * .15)
                 for size, bottom in zip(sizes, bottoms)) if median_size else False
    formula_no_match = _FORMULA_NO.search(text)
    formula_no = formula_no_match.group(1) if formula_no_match else None
    bare = _FORMULA_NO.sub("", text).strip()
    math_symbols = sum(c in _MATH_CHARS for c in bare) + len(_GREEK.findall(bare))
    operators = len(re.findall(r"[=<>+*/^_]", bare))
    words = re.findall(r"[A-Za-z]{4,}", bare)
    score, reasons = 0.0, []
    if any(_MATH_FONT.search(str(s.get("font") or "")) for s in real):
        score += .28; reasons.append("math_font")
    if math_symbols >= 2:
        score += .24; reasons.append("math_symbols")
    elif math_symbols == 1:
        score += .11
    if operators >= 2 or (operators and len(bare) <= 90):
        score += .22; reasons.append("operators")
    if script:
        score += .20; reasons.append("super_or_subscript")
    if formula_no:
        score += .18; reasons.append("formula_number")
    if operators and len(bare) <= 120 and len(words) <= 3:
        score += .10; reasons.append("standalone_shape")
    if len(words) >= 8 and math_symbols + operators < 3:
        score -= .30; reasons.append("sentence_penalty")
    return {"confidence": round(max(0, min(score, .99)), 3),
            "formula_no": formula_no, "reasons": reasons,
            "script_detected": script}


def candidates_from_page_dict(raw, page_no):
    found = []
    lines = []
    for block in raw.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            direction = line.get("dir") or (1, 0)
            if abs(float(direction[0])) < .98:
                continue
            spans = line.get("spans", []) or []
            text = _join_spans(spans)
            if not text:
                continue
            bbox = _bbox(line.get("bbox") or block.get("bbox"))
            lines.append({"text": text, "spans": spans, "bbox": bbox,
                          "score": score_formula_line(text, spans)})

    # PDF producers often split one displayed equation into several text blocks
    # (base glyphs, superscripts, and the right-aligned equation number). Merge
    # same-baseline candidates before scoring so `L = x` + superscript `2` is one
    # auditable formula rather than multiple unrelated chunks.
    lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    groups = []
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        placed = None
        for group in reversed(groups[-3:]):
            # Compare with the immediately preceding line, rather than the
            # whole accumulated group.  Using the group's max y made an
            # entire paragraph/column one giant "formula" on PDFs whose text
            # block contains many lines.
            prev = group[-1]
            py0, py1 = prev["bbox"][1], prev["bbox"][3]
            px1 = prev["bbox"][2]
            number_line = bool(_FORMULA_NO.search(line["text"]))
            group_has_number = any(_FORMULA_NO.search(item["text"]) for item in group)
            if (y0 <= py1 + 4 and y1 >= py0 - 4 and
                    (x0 - px1 <= 40 or number_line or group_has_number)):
                placed = group
                break
        if placed is None:
            groups.append([line])
        else:
            placed.append(line)

    for group in groups:
        group.sort(key=lambda x: x["bbox"][0])
        text = _join_spans([span for item in group for span in item["spans"]])
        bbox = (min(item["bbox"][0] for item in group),
                min(item["bbox"][1] for item in group),
                max(item["bbox"][2] for item in group),
                max(item["bbox"][3] for item in group))
        spans = [span for item in group for span in item["spans"]]
        scored = score_formula_line(text, spans)
        # A displayed equation is normally short.  Long prose blocks often
        # contain symbol-font glyphs or superscripts (citations, footnotes,
        # ligatures), which can otherwise cross the confidence threshold.
        if (not scored["formula_no"] and
                len(re.findall(r"[A-Za-z]{4,}", text)) >= 8):
            continue
        # Arrow-heavy table headings (e.g. language-pair columns) are not
        # equations.  They often use symbol fonts and were previously
        # promoted as low-confidence formulas despite having no equation
        # operator or formula number.
        if (not scored["formula_no"] and scored["reasons"] and
                not re.search(r"[=<>+*/^_]", text) and
                len(re.findall(r"[A-Za-z]{4,}", text)) >= 4):
            continue
        if (not scored["formula_no"] and
                any(ch in text for ch in "→←↔⇒⇔") and
                not re.search(r"[=<>+*/^_]", text)):
            continue
        if scored["confidence"] < .48:
            continue
        found.append({"block_id": _block_id(page_no, bbox, text),
                          "page_no": page_no, "bbox": bbox,
                          "raw_text": text, "formula_no": scored["formula_no"],
                          "source": "pdf_text", "confidence": scored["confidence"],
                          "reasons": scored["reasons"], "latex": None,
                          "latex_confidence": None, "status": "unresolved"})
    return found


def detect_pdf(path, max_pages=40):
    import pymupdf
    doc, found, line_count = pymupdf.open(path), [], 0
    try:
        for page_no, page in enumerate(doc, 1):
            if page_no > max_pages:
                break
            raw = page.get_text("dict")
            line_count += sum(len(b.get("lines", []) or []) for b in raw.get("blocks", []) or []
                              if b.get("type") == 0)
            found.extend(candidates_from_page_dict(raw, page_no))
    finally:
        doc.close()
    return {"formulas": found, "source": "pdf_text" if line_count else "none",
            "degraded": None if line_count else
            "PDF has no text geometry; OCR page text cannot provide reliable formula boxes or LaTeX",
            "parser_version": PARSER_VERSION}


def apply_math_ocr(path, candidates):
    module_name = (os.getenv("PAPERNEST_MATH_OCR_MODULE") or "").strip()
    if not module_name:
        return candidates, "Math OCR is disabled by default"
    try:
        recognize = getattr(importlib.import_module(module_name), "recognize_formula")
    except Exception as exc:
        return candidates, f"Math OCR plugin unavailable: {type(exc).__name__}: {exc}"
    import pymupdf
    doc = pymupdf.open(path)
    try:
        for item in candidates:
            try:
                page = doc[item["page_no"] - 1]
                rect = pymupdf.Rect(*item["bbox"]) & page.rect
                if rect.is_empty or not rect.is_valid:
                    continue
                result = recognize(page.get_pixmap(clip=rect, dpi=220, alpha=False).tobytes("png"),
                                   dict(item)) or {}
                latex = (result.get("latex") or "").strip()
                if latex:
                    item["latex"] = latex
                    item["latex_confidence"] = float(result.get("confidence") or 0)
                    item["status"] = ("recognized" if item["latex_confidence"] >= TRUSTED_LATEX_CONFIDENCE
                                      else "low_confidence")
                    item["source"] = "math_ocr:" + module_name
            except Exception as exc:
                item["ocr_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        doc.close()
    return candidates, None


def formula_chunk(item):
    trusted = item.get("status") == "recognized" and item.get("latex") and \
        float(item.get("latex_confidence") or 0) >= TRUSTED_LATEX_CONFIDENCE
    number = f" ({item['formula_no']})" if item.get("formula_no") else ""
    text = (f"Formula{number}\nLaTeX: {item['latex']}" if trusted else
            f"Formula{number} unresolved\n{item.get('raw_text') or ''}")
    x0, y0, x1, y1 = item.get("bbox") or (None, None, None, None)
    return {"text": text, "section_path": f"Formula{number} (page {item['page_no']})",
            "level": 1, "start_page": item["page_no"], "end_page": item["page_no"],
            "kind": "formula", "block_type": "formula",
            "bbox_x1": x0, "bbox_y1": y0, "bbox_x2": x1, "bbox_y2": y1,
            "block_id": item.get("block_id"), "source": item.get("source"),
            "confidence": item.get("confidence"), "status": item.get("status"),
            "latex": item.get("latex"),
            "latex_confidence": item.get("latex_confidence"),
            "raw_text": item.get("raw_text")}


def _anchor(block):
    bbox = block.get("bbox") or [block.get(k) for k in
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
    bbox_text = "null" if not all(v is not None for v in bbox) else ",".join(f"{float(v):.2f}" for v in bbox)
    kind = block.get("type") or block.get("block_type") or block.get("kind") or "text"
    attrs = [("id", block.get("block_id") or block.get("id") or "unknown"),
             ("type", kind), ("page", block.get("page_no") or block.get("start_page") or "null"),
             ("bbox", bbox_text), ("source", block.get("source") or "structured"),
             ("confidence", block.get("confidence") if block.get("confidence") is not None else "null")]
    return "<!-- papernest:block " + " ".join(f"{k}={v}" for k, v in attrs) + " -->"


def canonical_markdown(blocks):
    parts = []
    for block in blocks:
        kind = block.get("type") or block.get("block_type") or block.get("kind") or "text"
        # Formula chunks carry a retrieval-oriented `text` (for example
        # "Formula unresolved ..."). The canonical document must preserve the PDF
        # evidence instead, otherwise rebuilding from chunks serializes our own wrapper
        # rather than the formula candidate that produced it.
        text = ((block.get("raw_text") if kind == "formula" else block.get("text"))
                or block.get("text") or block.get("raw_text") or "").strip()
        parts.append(_anchor(block))
        if kind == "heading":
            parts.append("#" * max(1, min(int(block.get("level") or 1), 6)) + " " + text)
        elif kind == "formula":
            trusted = block.get("status") == "recognized" and block.get("latex") and \
                float(block.get("latex_confidence") or 0) >= TRUSTED_LATEX_CONFIDENCE
            if trusted:
                parts.append("$$\n" + block["latex"].strip() + "\n$$")
            else:
                parts.append("> [!warning] Unresolved formula; raw PDF text is not trusted LaTeX.")
                if text:
                    parts.append("~~~text\n" + text + "\n~~~")
        else:
            parts.append(text)
    return ("\n\n".join(p for p in parts if p).strip() + "\n") if parts else ""
