"""文献 → 汇报 PPT：从库内已核验的结构化卡片直接装配幻灯片。

设计取向：模型零参与、离线可用——PPT 是「汇报容器」，内容只来自已核验的
L1/L2 卡片（含 ✓/？核验标记与页码），不现场生成任何未经库内证据的事实。
单篇是论文汇报结构（背景→方法→结果→局限→启发），多篇是文献汇报结构
（总览→逐篇→参考）；配了 LLM 时额外生成一页课题叙事总起，没配则跳过。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from . import config, db

ACCENT = RGBColor(0x9A, 0x6B, 0x35)
DEEP = RGBColor(0x7C, 0x53, 0x26)
FG = RGBColor(0x20, 0x1D, 0x1A)
MUTED = RGBColor(0x77, 0x72, 0x6A)
LINE = RGBColor(0xE6, 0xE2, 0xD8)
CREAM = RGBColor(0xF6, 0xF4, 0xEF)

EXPORT_DIR = config.DATA_DIR / "exports" / "ppt"

NARRATIVE_SYSTEM = """你是学术汇报助手。基于给定文献卡片，为主题汇报写一段 150 字以内的「总起」：
这两句话要能交代研究背景、这些文献共同回答的问题与不同路线。只能使用卡片中的信息，不得编造。直接输出正文，不要标题。"""


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _box(slide, l, t, w, h):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def _style(tf, text, size, color=FG, bold=False, align=PP_ALIGN.LEFT, space_after=6):
    p = tf.paragraphs[0] if not tf.paragraphs[0].runs and not tf.paragraphs[0].text else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "微软雅黑"
    return p


def _header(slide, prs_w, title, sub=""):
    bar = slide.shapes.add_shape(1, Inches(0.6), Inches(0.55), Inches(0.09), Inches(0.42))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    tf = _box(slide, 0.85, 0.42, prs_w.inches - 1.6, 0.8)
    _style(tf, title, 26, DEEP, bold=True)
    if sub:
        tf2 = _box(slide, 0.87, 1.08, prs_w.inches - 1.6, 0.4)
        _style(tf2, sub, 12, MUTED)


def _bullets(tf, items, size=16):
    first = True
    for it in items:
        text, level = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = ("• " if level == 0 else "– ") + text
        r.font.size = Pt(size if level == 0 else size - 2)
        r.font.color.rgb = FG
        r.font.name = "微软雅黑"


def _note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def _card(row) -> dict:
    try:
        return json.loads(row["card_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _paper(row) -> dict:
    card = _card(row)
    abstract = (row["abstract"] or "").strip()
    return {
        "title": row["title"],
        "meta": f"{row['venue'] or '预印本'} · {row['year'] or ''}",
        "authors": (json.loads(row["authors"] or "[]") or [])[:6],
        "problem": card.get("problem") or (abstract.split("。")[0] if abstract else "卡片中未记录"),
        "method": card.get("method") or card.get("method_detail") or "",
        "results": card.get("results") or "",
        "findings": card.get("key_findings") or [],
        "limitations": card.get("limitations") or "",
        "relation": card.get("relation_to_topic") or "",
        "tldr": card.get("tldr") or "",
        "doi": row["doi"] or "",
        "arxiv": row["arxiv_id"] or "",
        "level": row["level"],
    }


def _source_line(p: dict) -> str:
    if p["doi"]:
        return f"doi: {p['doi']}"
    if p["arxiv"]:
        return f"arXiv: {p['arxiv']}"
    return "来源：PaperNest 文献库"


def _paper_slide(prs, prs_w, p: dict, idx: int):
    s = _blank(prs)
    _header(s, prs_w, f"{idx}. {p['title'][:60]}", p["meta"])
    tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, prs_w.inches * 9 / 16 - 2.0)
    items = []
    if p["tldr"]:
        items.append(f"TL;DR：{p['tldr'][:180]}")
    items.append(f"问题：{p['problem'][:180]}")
    if p["method"]:
        items.append(f"方法：{p['method'][:220]}")
    if p["results"]:
        items.append(f"结果：{p['results'][:220]}")
    _bullets(tf, items, size=14)
    _note(s, f"汇报提示：用一句话讲清这篇与前面文献的差异；数据细节见库内精读卡片（L{p['level']}）。")


def deck(paper_ids: list[int], topic: str = "", out_dir: Path | None = None,
         narrative: bool = True) -> dict:
    """生成汇报 PPT（单篇=论文汇报，多篇=文献汇报）。返回 {path, slides}。

    narrative：多篇时是否让模型生成一页「课题总起」（需 LLM；离线/测试可关）。"""
    db.init_db()
    ids = list(dict.fromkeys(int(x) for x in paper_ids if int(x) > 0))
    if not ids:
        raise ValueError("未指定文献")
    papers = []
    with db.conn() as c:
        for pid in ids:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if row:
                papers.append((row, _paper(row)))
    if not papers:
        raise ValueError("文献不存在")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs_w = prs.slide_width
    topic_text = topic or config.RESEARCH_TOPIC

    # 封面
    s = _blank(prs)
    bg = s.shapes.add_shape(1, 0, 0, prs_w, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = CREAM
    bg.line.fill.background()
    tf = _box(s, 1.0, 2.2, prs_w.inches - 2.0, 2.2)
    _style(tf, papers[0][1]["title"] if len(papers) == 1 else f"{topic_text} · 文献汇报",
           30 if len(papers) == 1 else 28, DEEP, bold=True, space_after=14)
    if len(papers) == 1:
        p0 = papers[0][1]
        _style(tf, " · ".join(p0["authors"]) if p0["authors"] else "", 15, FG, space_after=4)
        _style(tf, p0["meta"], 14, MUTED)
    tf2 = _box(s, 1.0, 4.6, prs_w.inches - 2.0, 1.2)
    _style(tf2, f"汇报课题：{topic_text}", 15, ACCENT, bold=True, space_after=4)
    _style(tf2, f"PaperNest 生成 · {time.strftime('%Y-%m-%d')} · 共 {len(papers)} 篇文献",
           11, MUTED)
    _note(s, "开场：先讲课题为什么关心这几篇文献，再进入正文。")

    single = len(papers) == 1
    if single:
        p = papers[0][1]
        s = _blank(prs)
        _header(s, prs_w, "研究背景与问题")
        tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.6)
        _bullets(tf, [f"问题：{p['problem'][:300]}", f"定位：{p['tldr'][:220] or '见下页方法'}"], size=17)
        _note(s, "汇报提示：交代这篇论文出现的场景与已有方法的不足。")

        s = _blank(prs)
        _header(s, prs_w, "方法")
        tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.6)
        _bullets(tf, [p["method"][:400] or "（卡片未记录方法，请补 L2 精读）"], size=17)
        _note(s, "汇报提示：强调方法的创新点与可迁移的部分。")

        s = _blank(prs)
        marks = []
        for f in p["findings"][:6]:
            mark = "✓" if f.get("verified") else "？"
            page = f"（p{f.get('page', '?')}）" if f.get("page") else ""
            marks.append(f"[{mark}] {str(f.get('claim', ''))[:160]}{page}")
        _header(s, prs_w, "关键结果", "✓ = 机械回取校验通过 · ？ = 待核验（页码为库内证据所在页）")
        tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.6)
        _bullets(tf, marks or ["（无已抽取的关键结论，建议先做 L2 全文精读）"], size=16)
        _note(s, "汇报提示：优先讲 ✓ 的结论；被质疑数据出处时给出对应页码。")

        s = _blank(prs)
        _header(s, prs_w, "局限与对我的课题的启发")
        tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.6)
        _bullets(tf, [f"局限：{p['limitations'][:260] or '卡片未记录'}",
                      f"启发：{p['relation'][:260] or '结合课题自行阐述'}"], size=16)
        _note(s, "汇报提示：主动讲局限能显著提升可信度。")

        s = _blank(prs)
        _header(s, prs_w, "谢谢", _source_line(p))
        _note(s, "备查：全文页码级证据在 PaperNest 文献库中可回放。")
    else:
        s = _blank(prs)
        _header(s, prs_w, "文献总览", f"{len(papers)} 篇 · {topic_text}")
        tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.8)
        _bullets(tf, [(f"[{i}] {p['title'][:70]}（{p['meta']}）", 0)
                      for i, (_, p) in enumerate(papers, 1)], size=13)
        _note(s, "汇报提示：按路线/时间线给文献分组，再逐篇展开。")

        if narrative and _llm_available():
            try:
                from . import llm
                blocks = "\n\n".join(
                    f"[{i}] {p['title']}：问题 {p['problem'][:80]}；方法 {p['method'][:80]}；结果 {p['results'][:80]}"
                    for i, (_, p) in enumerate(papers, 1))
                text = llm.chat(NARRATIVE_SYSTEM,
                                f"课题：{topic_text}\n\n文献卡片：\n{blocks}",
                                purpose="ppt_narrative", temperature=0.3)
                s = _blank(prs)
                _header(s, prs_w, "为什么是这几篇", topic_text)
                tf = _box(s, 0.9, 1.5, prs_w.inches - 1.8, 4.4)
                _style(tf, text[:400], 17)
                _note(s, "总起页：模型基于卡片生成，汇报前请人工确认表述。")
            except Exception:
                pass  # 叙事页失败不连累 PPT 生成

        for i, (_, p) in enumerate(papers, 1):
            _paper_slide(prs, prs_w, p, i)

        s = _blank(prs)
        _header(s, prs_w, "参考文献")
        tf = _box(s, 0.9, 1.4, prs_w.inches - 1.8, 5.2)
        _bullets(tf, [f"[{i}] {p['title']} · {_source_line(p)}"
                      for i, (_, p) in enumerate(papers, 1)], size=12)
        _note(s, "引用格式可在 PaperNest 中导出 BibTeX / RIS / GB-T 7714 / IEEE。")

    out_dir = Path(out_dir) if out_dir else EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if single:
        path = out_dir / f"paper_{ids[0]}.pptx"
    else:
        path = out_dir / f"deck_{'-'.join(str(i) for i in ids[:5])}{'_etc' if len(ids) > 5 else ''}.pptx"
    prs.save(path)
    return {"path": str(path), "slides": len(prs.slides._sldIdLst), "papers": len(papers)}


def _llm_available() -> bool:
    try:
        from . import llm
        return llm.available()
    except Exception:
        return False
