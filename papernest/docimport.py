"""多格式文档入库：Word / PPT / HTML / Markdown / 纯文本 → 与 PDF 同一条入库链路。

PDF 走 `pdfimport`（有版面信息可用于抽元数据），其余格式走这里：
`docparse` 负责把各种格式解析成统一的 blocks，本模块负责**把它接进论文库**——
元数据推断、归一化主键去重、正文入 pages/chunks、卡片生成。

元数据只用能确凿拿到的：docx 的 core_properties.title、HTML 的 <title>、
Markdown 的一级标题、PPT 的首页标题；拿不到就退回文件名，**不猜作者、不猜年份**
（PDF 那边能靠字号猜标题是因为有版面信息，这些格式没有，猜就是编）。

三态计数沿用 docparse 的口径：ok / skipped（重复、格式不支持、空文件）/ failed。
"""
from __future__ import annotations

import re
from pathlib import Path

from . import cards, config, db, docparse, llm
from .normalize import norm_key

MIN_CHARS = 200          # 少于这个字数的文档进库没有检索价值，跳过
MAX_CHUNK_CHARS = db.MAX_CHUNK_CHARS   # 与 structure.section_chunks 同一口径


class DocImportError(ValueError):
    pass


def _title_of(parsed: dict) -> str:
    """标题来源优先级：解析器给的 → 一级标题块 → 文件名（去扩展名）。"""
    t = (parsed.get("title") or "").strip()
    if t:
        return t[:300]
    for b in parsed.get("blocks") or []:
        if b.get("kind") == "heading" and (b.get("text") or "").strip():
            return b["text"].strip()[:300]
    return Path(parsed["path"]).stem[:300] or "未命名文档"


def _abstract_of(parsed: dict) -> str:
    """摘要：正文前 1200 字（跳过标题块）。不去猜哪段是「摘要」——
    这些格式没有稳定的结构信号，猜错了比留空更糟。"""
    parts = []
    for b in parsed.get("blocks") or []:
        if b.get("kind") in ("heading", "notes"):
            continue
        s = (b.get("text") or "").strip()
        if s:
            parts.append(s)
        if sum(len(x) for x in parts) > 1200:
            break
    return " ".join(parts)[:1200]


def _chunks_of(parsed: dict) -> list[dict]:
    """blocks → 检索块。标题块作为 section_path 前缀带给后续块，
    这样检索命中一个块时仍知道它属于哪一节（与 PDF 的章节块对齐）。"""
    out: list[dict] = []
    current = ""
    buf: list[str] = []

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            out.append({"text": text, "section_path": current, "level": 1,
                        "start_page": None, "end_page": None, "kind": "text"})
        buf.clear()

    for b in parsed.get("blocks") or []:
        kind, text = b.get("kind"), (b.get("text") or "").strip()
        if not text:
            continue
        if kind == "heading":
            flush()
            current = text[:200]
            continue
        if kind == "slide":
            flush()
            current = text.split("\n", 1)[0][:200] or current
        if kind == "table":
            flush()
            out.append({"text": text, "section_path": current, "level": 1,
                        "start_page": None, "end_page": None, "kind": "table"})
            continue
        buf.append(text)
        if sum(len(x) for x in buf) >= MAX_CHUNK_CHARS:
            flush()
    flush()
    # 缓冲是「先 append 再判」，flush 出来的块最多能到 (MAX_CHUNK_CHARS-1) + 最后
    # 一段的长度；单段本身就超限时更是一刀都切不开。出口统一过 cap_chunks，
    # 与 db.chunks_from_pages 同一层收口。
    return db.cap_chunks(out, MAX_CHUNK_CHARS)


def _pages_of(parsed: dict) -> list[str]:
    """非 PDF 没有物理页。PPT 一张幻灯片算一「页」（用户就是这么指的），
    其余格式按 MAX_CHUNK_CHARS 顺序切——页码在这里是「第几段」，
    卡片里的页码引用因此仍然指得回原文。"""
    blocks = parsed.get("blocks") or []
    if any(b.get("kind") == "slide" for b in blocks):
        pages, cur = [], []
        for b in blocks:
            if b.get("kind") == "slide" and cur:
                pages.append("\n".join(cur))
                cur = []
            if b.get("text"):
                cur.append(b["text"])
        if cur:
            pages.append("\n".join(cur))
        return pages
    text = parsed.get("text") or ""
    return [text[i:i + MAX_CHUNK_CHARS]
            for i in range(0, len(text), MAX_CHUNK_CHARS)] or []


def import_document(path, topic: str | None = None,
                    make_card: bool = True) -> dict:
    """单份非 PDF 文档入库。返回 dict（含 status，与 docparse 三态一致）。

    重复判定：按标题归一化主键。这些格式没有 DOI/arXiv id，
    `norm_key(title=...)` 是唯一可用的稳定键。
    """
    db.init_db()
    parsed = docparse.parse(path)
    if parsed["status"] != "ok":
        return {"status": parsed["status"], "reason": parsed["reason"],
                "path": str(path), "format": parsed.get("format")}
    if parsed["n_chars"] < MIN_CHARS:
        return {"status": "skipped", "path": str(path), "format": parsed["format"],
                "reason": f"正文仅 {parsed['n_chars']} 字，低于入库下限 {MIN_CHARS}"}

    title = _title_of(parsed)
    abstract = _abstract_of(parsed)
    key = norm_key(title=title)
    warnings = list(parsed.get("warnings") or [])

    with db.conn() as c:
        exist = db.get_by_norm_key(c, key)
        if exist:
            return {"status": "skipped", "reason": "库内已有同名文档",
                    "paper_id": exist["id"], "title": exist["title"],
                    "norm_key": key, "path": str(path),
                    "format": parsed["format"], "duplicate": True}
        meta = parsed.get("meta") or {}
        year = meta.get("year") if isinstance(meta.get("year"), int) else None
        authors = [a for a in (meta.get("authors") or []) if isinstance(a, str)]
        pid = db.insert_l0(c, {
            "norm_key": key, "title": title, "abstract": abstract,
            "year": year, "venue": None, "authors": authors,
            "doi": None, "arxiv_id": None, "citation_count": None,
            "oa_pdf_url": None, "source": f"upload-{parsed['format']}", "s2_id": None,
        })
        pages = _pages_of(parsed)
        for i, text in enumerate(pages, 1):
            if text.strip():
                c.execute("INSERT OR REPLACE INTO pages(paper_id,page_no,text) "
                          "VALUES(?,?,?)", (pid, i, text))
        db.reindex_pages(c, pid)
        chunks = _chunks_of(parsed) or db.chunks_from_pages(c, pid)
        n_chunks = db.replace_chunks(c, pid, chunks)
        if pages:
            c.execute("UPDATE papers SET level=2 WHERE id=?", (pid,))

    card_model = None
    if make_card:
        try:
            card, model = cards.make_card(
                {"title": title, "abstract": abstract, "_db_id": pid},
                topic or config.RESEARCH_TOPIC)
            with db.conn() as c:
                db.save_card(c, pid, card, model)
            card_model = model
        except llm.LLMUnavailable:
            warnings.append("未配置 LLM，本次未生成卡片（配好后跑 cli.py recard 补）")
        except Exception as e:      # 卡片失败不该让入库回滚——正文已经有价值了
            warnings.append(f"卡片生成失败（{type(e).__name__}），正文已入库")

    return {"status": "ok", "paper_id": pid, "title": title, "norm_key": key,
            "format": parsed["format"], "path": str(path),
            "pages_stored": len(pages), "chunks": n_chunks,
            "n_chars": parsed["n_chars"], "card_model": card_model,
            "warnings": warnings}


def import_batch(paths, topic: str | None = None, make_card: bool = True,
                 progress=None) -> dict:
    """批量入库，三态计数。单份失败不中断整批。"""
    items, counts = [], {"ok": 0, "skipped": 0, "failed": 0}
    by_format: dict[str, dict[str, int]] = {}
    total = max(len(paths), 1)
    for i, p in enumerate(paths, 1):
        if progress:
            progress(i / total, "import", f"[{i}/{total}] {Path(p).name}")
        try:
            r = import_document(p, topic, make_card)
        except Exception as e:      # 兜底：任何未预期异常都记成 failed，不炸整批
            r = {"status": "failed", "path": str(p),
                 "reason": f"{type(e).__name__}: {e}", "format": None}
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        fmt = r.get("format") or "unknown"
        by_format.setdefault(fmt, {"ok": 0, "skipped": 0, "failed": 0})
        by_format[fmt][r["status"]] += 1
        items.append(r)
    return {**counts, "items": items, "by_format": by_format, "total": len(paths)}


def collect_paths(patterns) -> list[Path]:
    """展开目录/通配，只留 docparse 支持的非 PDF 格式（PDF 走 import-pdf）。"""
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*")
                              if f.is_file() and docparse.is_supported(f)))
        elif p.is_file():
            out.append(p)
        else:
            out.extend(sorted(f for f in Path().glob(str(pat))
                              if f.is_file() and docparse.is_supported(f)))
    return list(dict.fromkeys(out))
