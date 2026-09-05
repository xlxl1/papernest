"""引用推荐：段落 → 候选文献 → 支撑证据句（机械校验）→ 四格式引用条目。

支撑证据句是本模块的灵魂（QC-Bench 引证校验的平移）：
  推荐一篇文献时必须给出「它里面真正支撑你那段话的原句」，
  且该句必须能在库内存储的摘要原文中回取到（机械校验），取不到就降级标注「相关性弱」。
  —— 防「引了但不支撑」的 AI 引用幻觉。
"""
import json
import re

import numpy as np

from . import db, embeddings


# ── 候选检索 ──

def _paper_vec(paper_id: int) -> np.ndarray | None:
    with db.conn() as c:
        r = c.execute("SELECT vec FROM vectors WHERE paper_id=? AND kind='paper' "
                      "ORDER BY id DESC LIMIT 1", (paper_id,)).fetchone()
    return embeddings._from_blob(r["vec"]) if r else None


def _fallback_scores(text: str) -> list[dict]:
    """无向量时的兜底：中文 bigram + 英文词面的重叠率。"""
    def grams(s: str) -> set:
        s = re.sub(r"\s+", " ", s.lower())
        g = set(s[i:i + 2] for i in range(len(s) - 1) if s[i:i + 2].strip())
        g |= set(w for w in re.findall(r"[a-z][a-z0-9-]{2,}", s))
        return g
    q = grams(text)
    out = []
    with db.conn() as c:
        rows = c.execute("SELECT id, title, abstract FROM papers").fetchall()
    for r in rows:
        g = grams(f"{r['title']} {r['abstract'] or ''}")
        if not q or not g:
            continue
        out.append({"paper_id": r["id"], "score": len(q & g) / len(q)})
    out.sort(key=lambda x: -x["score"])
    return out[:8]


def recommend(paragraph: str, top_k: int = 5) -> dict:
    db.init_db()
    degraded = None
    try:
        pvec = embeddings.embed_texts([paragraph])[0]
        pvec = np.asarray(pvec, dtype=np.float32)
        cands = embeddings.search_papers(paragraph, top_k * 2)
    except Exception as e:
        degraded = f"向量检索不可用（{e}），降级为词面重叠匹配"
        pvec = None
        cands = _fallback_scores(paragraph)

    out = []
    for c in cands[:max(top_k * 2, 8)]:
        pid = c["paper_id"]
        with db.conn() as cdb:
            row = cdb.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        if not row:
            continue
        evidence, escore, verified = None, 0.0, False
        if pvec is not None:
            evidence, escore = embeddings.best_sentence(pvec, pid)
        else:
            evidence, escore = _best_sentence_overlap(paragraph, row["abstract"] or "")
        if evidence:
            # 机械校验：证据句必须能在库内摘要原文中回取到
            verified = _norm(evidence) in _norm(row["abstract"] or "")
        strong = bool(evidence) and verified and (pvec is None or escore >= 0.35)
        out.append({
            "paper_id": pid, "score": round(c["score"], 3),
            "evidence_sentence": evidence, "evidence_score": round(escore, 3),
            "verified": verified, "strong": strong,
            "title": row["title"], "year": row["year"], "venue": row["venue"],
            "authors": json.loads(row["authors"] or "[]"),
            "doi": row["doi"], "arxiv_id": row["arxiv_id"],
        })
    out.sort(key=lambda x: (not x["strong"], -x["score"]))
    return {"candidates": out[:top_k], "degraded": degraded}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _best_sentence_overlap(paragraph: str, abstract: str) -> tuple[str | None, float]:
    q = set(_norm(paragraph)[i:i + 2] for i in range(len(_norm(paragraph)) - 1))
    best, best_s = None, 0.0
    for s in embeddings.split_sentences(abstract):
        g = _norm(s)
        gs = set(g[i:i + 2] for i in range(len(g) - 1))
        if not gs:
            continue
        score = len(q & gs) / max(len(gs), 1)
        if score > best_s:
            best, best_s = s, score
    return best, best_s


# ── 引用条目格式 ──

def _author_list(authors: list[str]) -> list[str]:
    return authors or ["Unknown"]


_BIBTEX_ESCAPES = {"{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%", "$": r"\$",
                   "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}",
                   "^": r"\textasciicircum{}"}


def _bib_escape(s: str) -> str:
    """转义 BibTeX 特殊字符。反斜杠先挪走再换回，否则会把刚转义出的反斜杠再转一遍。"""
    out = (s or "").replace("\\", "\x00")
    for ch, rep in _BIBTEX_ESCAPES.items():
        out = out.replace(ch, rep)
    return out.replace("\x00", r"\textbackslash{}")


def _bib_key(p: dict) -> str:
    """引用键只留 ASCII 字母数字——中文标题拼出来的键很多 BibTeX 工具直接解析失败。"""
    surname = ""
    for a in _author_list(p.get("authors")):
        parts = (a or "").split()
        if parts:
            surname = re.sub(r"[^A-Za-z]", "", parts[-1]).lower()
            if surname:
                break
    slug = re.sub(r"[^A-Za-z0-9]", "", p.get("title") or "")[:20].lower()
    if not slug:  # 纯中文标题：退回论文 id，保证键唯一且合法
        slug = f"p{p.get('paper_id') or ''}"
    return f"{surname or 'anon'}{p.get('year') or ''}{slug}" or "papernestref"


def to_bibtex(p: dict) -> str:
    key = _bib_key(p)
    au = " and ".join(_author_list(p.get("authors")))
    lines = [f"@article{{{key}," if p.get("venue") else f"@misc{{{key},",
             f"  title = {{{_bib_escape(p['title'])}}},",
             f"  author = {{{_bib_escape(au)}}},"]
    if p.get("venue"):
        lines.append(f"  journal = {{{_bib_escape(p['venue'])}}},")
    if p.get("year"):
        lines.append(f"  year = {{{p['year']}}},")
    if p.get("doi"):
        lines.append(f"  doi = {{{_bib_escape(p['doi'])}}},")
    if p.get("arxiv_id"):
        lines.append(f"  eprint = {{{_bib_escape(p['arxiv_id'])}}},")
        lines.append("  archivePrefix = {arXiv},")
    lines.append("}")
    return "\n".join(lines)


def to_ris(p: dict) -> str:
    lines = ["TY  - JOUR", f"TI  - {p['title']}"]
    for a in _author_list(p["authors"]):
        lines.append(f"AU  - {a}")
    if p.get("venue"):
        lines.append(f"JO  - {p['venue']}")
    if p.get("year"):
        lines.append(f"PY  - {p['year']}")
    if p.get("doi"):
        lines.append(f"DO  - {p['doi']}")
    if p.get("arxiv_id"):
        lines.append(f"UR  - https://arxiv.org/abs/{p['arxiv_id']}")
    lines.append("ER  - ")
    return "\n".join(lines)


def to_gbt(p: dict) -> str:
    """GB/T 7714（简化）：作者不超过 3 人全列，超过取前 3 加「等」。"""
    au = _author_list(p["authors"])
    au_s = ", ".join(au[:3]) + ((", 等" if len(au) > 3 else "")) if au else "佚名"
    venue = p.get("venue") or ("arXiv preprint" if p.get("arxiv_id") else "")
    s = f"{au_s}. {p['title']}[J]. {venue}, {p.get('year') or ''}.".replace(" .", ".")
    return re.sub(r"\s+", " ", s)


def to_ieee(p: dict) -> str:
    au = _author_list(p["authors"])
    au_s = ", ".join(au[:6]) + (", et al." if len(au) > 6 else "")
    venue = p.get("venue") or ("arXiv preprint arXiv:" + p["arxiv_id"] if p.get("arxiv_id") else "")
    return f'{au_s}, "{p["title"]}," {venue}, {p.get("year") or ""}.'


FORMATS = {"bibtex": to_bibtex, "ris": to_ris, "gbt7714": to_gbt, "ieee": to_ieee}


def export(paper_ids: list[int], fmt: str) -> str:
    fn = FORMATS.get(fmt)
    if not fn:
        raise ValueError(f"不支持的格式：{fmt}（可选 {list(FORMATS)}）")
    blocks = []
    with db.conn() as c:
        for pid in paper_ids:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if row:
                blocks.append(fn({"paper_id": row["id"], "title": row["title"],
                                  "authors": json.loads(row["authors"] or "[]"),
                                  "year": row["year"], "venue": row["venue"],
                                  "doi": row["doi"], "arxiv_id": row["arxiv_id"]}))
    sep = "\n\n" if fmt != "ris" else "\n"
    return sep.join(blocks)
