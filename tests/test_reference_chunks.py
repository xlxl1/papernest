# -*- coding: utf-8 -*-
"""参考文献段不能当成正文检索单元进 chunks_fts / 向量 / 回答上下文。

`_section_chunks` 只按 sections 的 block_index 划边界，而 References 在
`_detect_sections` 眼里就是普通一章（'references'/'bibliography' 本来就在关键词表里），
于是整段按 4000 字符切成正常 chunk。同一个文件里的 `_ref_lines()` 能准确定位该段，
但只被 `_extract_references()` 调用——抽条目和切块对「哪里开始是参考文献」有两套
不连通的判据，其中一套是空的。

真库实测（data/papernest.db，503 篇 / 804 chunk / 4816 向量）：
  · 13.9% 的可检索 chunk 文本是他人的文献条目（按块数是 7.7%）；
    集中在 25 篇有 PDF 的论文，它们自己的文本里 17.2% 是参考文献，paper 1 单篇 33.2%；
  · 64 条真实评测查询里 **16 条**的块级 top-5 含参考文献块，共 31/320 个位次，
    最坏的一条 "web navigating agent built on large language models" 是 **5/5**；
  · 6.8% 的 chunk 向量是参考文献，却拿下 15.7% 的 top-1（2.3 倍过表达）。

这条必须用真 PDF：合成夹具的 References 只有两三条，既压不出 `_pack` 的多块切分，
也复现不出「参考文献块在 FTS 里排到 top-1」——合成夹具正好会掩盖这条缺陷。
真 PDF 不在就跳过，不让干净 checkout 变红（与 tests/test_span_join.py 同一口径）。
"""
import re
import unittest
from pathlib import Path

from papernest import chunkembed, db, fulltext, rag, structure

try:                       # `python -m unittest tests.x` vs `discover -s tests`
    from .support import TempDbTestCase
except ImportError:
    from support import TempDbTestCase

ROOT = Path(__file__).resolve().parents[1]
_WS = re.compile(r"\s+")


def _n(s: str) -> str:
    return _WS.sub("", (s or "")).lower()


def _ref_blob(path):
    """真参考文献段的归一化文本。

    定位用模块自己的 `_ref_lines`（切块本来就该用它），终点取「References 之后
    第一个不是参考文献标题的已识别章节」，把附录排除掉。
    """
    blocks, n_pages = structure._read(path)
    if not blocks:
        return None
    seg, _, _ = structure._ref_lines(blocks)
    if not seg:
        return None
    secs = structure._detect_sections(blocks, structure.body_font_size(blocks), n_pages)
    start = seg[0]["block_index"]
    later = [s["block_index"] for s in secs
             if s["block_index"] >= start and not structure._REF_HEAD_RE.match(s["title"])]
    if later:
        seg = [l for l in seg if l["block_index"] < min(later)]
    return _n(" ".join(l["text"] for l in seg))


def _inside(text: str, blob: str) -> bool:
    """这块文本是不是「基本上就是参考文献段」：40 字符探针命中过半。"""
    t = _n(text)
    if len(t) < 200:
        return False
    step = max(len(t) // 20, 40)
    probes = [t[i:i + 40] for i in range(0, len(t) - 40, step)]
    return bool(probes) and sum(1 for p in probes if p in blob) / len(probes) >= 0.5


def _pick_real_pdf():
    """挑一篇真有参考文献段的真 PDF（按文件名定序，结果可复现）。"""
    for p in sorted((ROOT / "data" / "pdf").glob("*.pdf")):
        blob = _ref_blob(str(p))
        if blob and len(blob) >= 4000:
            return str(p), blob
    return None, None


class ReferenceChunksAreNotBodyText(TempDbTestCase):
    prefix = "papernest_refchunk"

    def setUp(self):
        super().setUp()
        try:
            import pymupdf                              # noqa: F401
        except ImportError:
            self.skipTest("未安装 pymupdf")
        db.init_db()
        self.pdf, self.blob = _pick_real_pdf()
        if not self.pdf:
            self.skipTest("data/pdf 下没有带参考文献段的真实 PDF 样本")

    # ── ① 切块侧：参考文献必须是另一种 kind，不能混进正文块 ──

    def test_reference_section_gets_its_own_kind(self):
        chunks = structure.section_chunks(self.pdf)
        self.assertTrue(chunks, "真 PDF 一个 chunk 都没切出来，用例没有说服力")

        refs = [c for c in chunks if c.get("kind") == "reference"]
        body = [c for c in chunks if c.get("kind") != "reference"]
        leaked = [c for c in body if _inside(c["text"], self.blob)]

        self.assertGreater(len(refs), 0,
                           "参考文献段没有被标成独立 kind：它会作为 kind='text' 的"
                           "正文块进 chunks_fts 与向量索引")
        self.assertEqual([], [c["section_path"] for c in leaked],
                         f"{len(leaked)} 个正文块的内容其实是参考文献条目")
        self.assertGreater(len(body), 5, "正文块被误伤：修法不能把正文一起标成参考文献")

    # ── ② 索引侧：参考文献块不进 FTS、不进待嵌入队列 ──

    def test_reference_chunks_are_not_indexed_or_embedded(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "test:refchunk", "title": "T", "abstract": "A",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": None, "source": "s2"})
            db.replace_chunks(c, pid, fulltext.build_chunks(self.pdf, c, pid))
            n_ref = c.execute("SELECT COUNT(*) n FROM chunks WHERE paper_id=? "
                              "AND kind='reference'", (pid,)).fetchone()["n"]
            n_other = c.execute("SELECT COUNT(*) n FROM chunks WHERE paper_id=? "
                                "AND kind<>'reference'", (pid,)).fetchone()["n"]
            n_fts = c.execute("SELECT COUNT(*) n FROM chunks_fts "
                              "WHERE paper_id=?", (pid,)).fetchone()["n"]

        self.assertGreater(n_ref, 0, "build_chunks 没把参考文献块的 kind 透传下来")
        self.assertGreater(n_other, 5, "正文块被误伤")
        self.assertEqual(n_fts, n_other,
                         f"chunks_fts 里有 {n_fts - n_other} 个参考文献块，"
                         f"它们会被 search_chunks_fts 当成本篇证据检索出来")

        pending = {(x["paper_id"], x["chunk_no"]) for x in chunkembed.pending()}
        with db.conn() as c:
            ref_nos = {(pid, r["chunk_no"]) for r in c.execute(
                "SELECT chunk_no FROM chunks WHERE paper_id=? AND kind='reference'",
                (pid,))}
        self.assertEqual(set(), pending & ref_nos,
                         "参考文献块进了待嵌入队列：额度会花在他人的文献条目上")

    # ── ③ 上下文侧：参考文献块不能当成证据喂给模型 ──

    def test_reference_text_never_reaches_the_answer_context(self):
        """把它们排除出 chunks_fts 还不够。

        `rag._rank_chunks` 是直接从 chunks 表拉**全部**块按词频密度排的，不靠检索命中，
        所以上下文那条路必须自己再挡一道。
        """
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "test:refctx", "title": "T", "abstract": "A",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": None, "source": "s2"})
            db.replace_chunks(c, pid, fulltext.build_chunks(self.pdf, c, pid))
            rows = c.execute("SELECT text FROM chunks WHERE paper_id=? "
                             "ORDER BY length(text) DESC", (pid,)).fetchall()
        ref_text = next((r["text"] for r in rows if _inside(r["text"], self.blob)), None)
        self.assertIsNotNone(ref_text, "这篇 PDF 里没找到参考文献块，用例没有说服力")
        # 问句直接由参考文献条目里的词构成：修好之前它必然把参考文献块顶成密度第一
        q = " ".join(re.findall(r"[A-Za-z]{6,}", ref_text)[:12])

        with db.conn() as c:
            hits = db.search_chunks_hits(c, q, 15).get(pid, [])
        block, _ = rag._evidence_context(pid, q, hits, per_paper=2, cap=2400,
                                         unit="chunk")
        # 上下文里每个【章节·第 n 页】是一个证据块，逐块查
        bad = [b.split("】")[0] + "】" for b in re.split(r"(?=【)", block)
               if _inside(b, self.blob)]
        self.assertEqual([], bad, "参考文献条目被当成本篇证据写进了 LLM 上下文")


if __name__ == "__main__":
    unittest.main()
