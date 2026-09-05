"""导出 Obsidian 文献笔记：每篇一个 Markdown，citekey 命名，双链互联。

Obsidian 生态的主流实践（Zotero Integration / ZotLit）是「模板化生成文献笔记 +
citekey 双链」。PaperNest 不做笔记应用——把已核验的结构化产出（卡片、关键结论带页码、
引文关系）导成规范的 Markdown，让用户接进自己已有的 vault：

- 文件名 = citekey（复用 cite._bib_key，与 BibTeX 导出同键，Obsidian ↔ LaTeX 对得上号）；
- YAML frontmatter 放元数据（title/authors/year/venue/doi/arxiv/tags），
  Dataview 等插件能直接查询；
- **库内互引写成 `[[citekey]]` 双链**：引文网络里 A 引 B 且两篇都在库里，
  A 的笔记「引用了」一节就有 B 的双链——笔记图谱直接长出文献图谱的形状；
- 卡片字段缺什么就不写什么小节，**不编造**；关键结论保留 ✓/? 核验标记与页码。
"""
from __future__ import annotations

import re
from pathlib import Path

from . import cite, db

# Windows/Obsidian 都不接受的文件名字符
_BAD_FN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# OpenAlex 把 DOI 存成完整 URL（https://doi.org/10.x）——引文图那边踩过同一个坑：
# 不剥前缀，frontmatter 是一条 URL、底部链接拼成 https://doi.org/https://doi.org/…
_DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.I)


def _bare_doi(doi) -> str | None:
    d = _DOI_URL_RE.sub("", str(doi or "").strip())
    return d or None


def _citekey(row) -> str:
    import json
    return cite._bib_key({"paper_id": row["id"], "title": row["title"],
                          "authors": json.loads(row["authors"] or "[]"),
                          "year": row["year"]})


def _fn_safe(name: str) -> str:
    return _BAD_FN.sub("_", name)[:120] or "untitled"


def _yaml_escape(s) -> str:
    text = str(s or "").replace('"', r'\"')
    return f'"{text}"'


def _library_links(c, paper_id: int, norm_key_str: str,
                   keys: dict[str, str]) -> tuple[list[str], list[str]]:
    """库内互引 → citekey 双链。keys: norm_key → citekey（仅库内论文）。"""
    cited = [keys[r["dst_key"]] for r in c.execute(
        "SELECT dst_key FROM citation_edges WHERE src_key=? ORDER BY dst_key",
        (norm_key_str,)) if r["dst_key"] in keys]
    cited_by = [keys[r["src_key"]] for r in c.execute(
        "SELECT src_key FROM citation_edges WHERE dst_key=? ORDER BY src_key",
        (norm_key_str,)) if r["src_key"] in keys]
    return cited, cited_by


def note_markdown(paper_id: int, keys: dict[str, str] | None = None) -> tuple[str, str]:
    """单篇 → (文件名, Markdown)。keys 不传就现查（批量导出时传入避免 N 次全表扫）。"""
    import json
    db.init_db()
    with db.conn() as c:
        row = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not row:
            raise LookupError(f"论文 {paper_id} 不存在")
        if keys is None:
            keys = _citekey_map(c)
        has_edges = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='citation_edges'"
        ).fetchone() is not None
        cited, cited_by = (_library_links(c, paper_id, row["norm_key"], keys)
                           if has_edges else ([], []))

    try:
        card = json.loads(row["card_json"] or "{}")
    except json.JSONDecodeError:
        card = {}
    authors = json.loads(row["authors"] or "[]")
    key = keys.get(row["norm_key"]) or _citekey(row)

    fm = ["---",
          f"title: {_yaml_escape(row['title'])}",
          f"citekey: {key}",
          "authors: [" + ", ".join(_yaml_escape(a) for a in authors) + "]"]
    if row["year"]:
        fm.append(f"year: {row['year']}")
    if row["venue"]:
        fm.append(f"venue: {_yaml_escape(row['venue'])}")
    doi = _bare_doi(row["doi"])
    if doi:
        fm.append(f"doi: {_yaml_escape(doi)}")
    if row["arxiv_id"]:
        fm.append(f"arxiv: {_yaml_escape(row['arxiv_id'])}")
    kws = [k for k in (card.get("keywords") or []) if isinstance(k, str)]
    if kws:
        fm.append("tags: [" + ", ".join(_yaml_escape(k.replace(" ", "-")) for k in kws) + "]")
    fm.append("source: PaperNest")
    fm.append("---")

    body = [f"# {row['title']}", ""]
    if authors:
        body.append("**" + ", ".join(authors) + "**"
                    + (f" · {row['venue']}" if row["venue"] else "")
                    + (f" · {row['year']}" if row["year"] else ""))
        body.append("")
    for label, field in (("TL;DR", "tldr"), ("问题", "problem"), ("方法", "method"),
                         ("结果", "results"), ("局限", "limitations"),
                         ("与我课题的关系", "relation_to_topic")):
        v = card.get(field)
        if v and isinstance(v, str) and v.strip():
            body += [f"## {label}", "", v.strip(), ""]
    findings = [f for f in (card.get("key_findings") or []) if isinstance(f, dict)]
    if findings:
        body += ["## 关键结论（✓ = 通过原文机械回取校验）", ""]
        for f in findings:
            mark = "✓" if f.get("verified") else "?"
            page = f" (p.{f['page']})" if f.get("page") else ""
            body.append(f"- {mark}{page} {f.get('claim', '')}")
        body.append("")
    if row["abstract"]:
        body += ["## 摘要", "", row["abstract"].strip(), ""]
    if cited:
        body += ["## 引用了（库内）", ""]
        body += [f"- [[{k}]]" for k in sorted(set(cited))]
        body.append("")
    if cited_by:
        body += ["## 被引于（库内）", ""]
        body += [f"- [[{k}]]" for k in sorted(set(cited_by))]
        body.append("")
    links = []
    if doi:
        links.append(f"[DOI](https://doi.org/{doi})")
    if row["arxiv_id"]:
        links.append(f"[arXiv](https://arxiv.org/abs/{row['arxiv_id']})")
    if links:
        body += ["---", " · ".join(links), ""]

    return _fn_safe(key) + ".md", "\n".join(fm) + "\n\n" + "\n".join(body)


def _citekey_map(c) -> dict[str, str]:
    return {r["norm_key"]: _citekey(r)
            for r in c.execute("SELECT id,norm_key,title,authors,year FROM papers")}


def export_vault(out_dir: str | Path, paper_ids: list[int] | None = None) -> dict:
    """批量导出到目录（不存在则建）。同名文件直接覆盖——笔记以库内数据为准，
    要保留手写内容请导到独立子目录再由 Obsidian 合并。"""
    db.init_db()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with db.conn() as c:
        keys = _citekey_map(c)
        if paper_ids is None:
            paper_ids = [r["id"] for r in
                         c.execute("SELECT id FROM papers ORDER BY id").fetchall()]
    written, errors = [], []
    for pid in paper_ids:
        try:
            fn, text = note_markdown(pid, keys)
        except Exception as exc:
            errors.append(f"#{pid}: {type(exc).__name__}: {exc}")
            continue
        (out / fn).write_text(text, encoding="utf-8")
        written.append(fn)
    return {"out_dir": str(out), "written": len(written), "files": written[:50],
            "errors": errors}
