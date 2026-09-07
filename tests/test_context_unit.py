"""上下文组装：块级命中的接线，以及「换成章节块」这条**负结果**。

背景：审计指出「按章节切，等预算证据召回 2.2~3.2 倍」这条招牌结论在生产链路上
没有落地——`rag.prepare` 组装上下文时一行都不读 `chunks`，检索算出来的块级命中
被压成 paper_id 就丢了。

于是做了两件事：
1. **把块级命中接通**（`RetrievalResult.chunk_hits`）：词面命中来自 chunks_fts，
   语义命中来自向量矩阵里赢的那一行（`search_papers` 现在带 `chunk_no`）。
2. **真去量了一次**，结果是负的：三臂配对实验（QASPER 88 题、等预算、0 token）

   | 口径 | 证据召回(删空白) | (保留空白) | 变化的题 | p |
   |---|---|---|---|---|
   | A 改造前 | 0.2273 | 0.2273 | — | — |
   | B 只改选块 | **0.2614** | **0.2614** | 7/88 | 0.452 |
   | C 换成章节块（修 span 之前） | 0.2045 | **0.1136** | 17/88 | 0.331 |
   | C 换成章节块（修 span + 重切之后） | 0.2500 | **0.2500** | 17/88 | 1.0 |

   修掉 span 拼接吃空格并重切全库之后，**保真缺口归零**（两个口径逐位相同），
   C 与 B 的差距从 -0.0568 收敛到 -0.0114（1 道题，p=1.0）——**实质持平**。
   没有可测量的收益就不改主路径默认值，所以默认单元仍是 `page`。
   **这些用例的作用是：把这个结论钉住，别让人凭直觉把开关打开。**
   复现：`python cli.py qasper ctxab`
"""
from __future__ import annotations

import unittest
from unittest import mock

from papernest import db, embeddings, rag

try:
    from .support import TempDbTestCase
except ImportError:                          # noqa: F401
    from support import TempDbTestCase


class ChunkHitsAreCarriedThrough(TempDbTestCase):
    """接线本身必须是对的——即使默认不用它，它也是下一步的前提。"""

    prefix = "papernest_ctxunit"

    def setUp(self):
        super().setUp()
        db.init_db()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:4001", "title": "Quantization for Transformers",
                "abstract": "We study quantization.", "year": 2024, "venue": "V",
                "authors": [], "doi": None, "arxiv_id": "4001", "source": "s2"})
            db.replace_chunks(c, self.pid, [
                {"text": "Introduction. Transformers are large.",
                 "section_path": "1 Introduction", "start_page": 1},
                {"text": "We apply quantization to reduce memory footprint.",
                 "section_path": "3 Method", "start_page": 5},
                {"text": "Related work on pruning and distillation.",
                 "section_path": "2 Related Work", "start_page": 3}])
            c.commit()

    def test_search_returns_which_chunk_matched(self):
        res = embeddings.search_hybrid("quantization memory", 5)
        self.assertIn(self.pid, res.chunk_hits or {},
                      "块级命中没有被带出来——上下文组装就拿不回「是哪一段匹配的」")
        hit = res.chunk_hits[self.pid][0]
        self.assertIn("chunk_no", hit)
        self.assertIn("section_path", hit)
        self.assertIn("start_page", hit)

    def test_hits_point_at_the_right_section(self):
        res = embeddings.search_hybrid("quantization memory footprint", 5)
        nos = [h["chunk_no"] for h in (res.chunk_hits or {}).get(self.pid, [])]
        with db.conn() as c:
            method = c.execute(
                "SELECT chunk_no FROM chunks WHERE paper_id=? AND section_path LIKE '3%'",
                (self.pid,)).fetchone()["chunk_no"]
        self.assertIn(method, nos, "命中的不是含该词的那一节")

    def test_chunks_by_no_preserves_order(self):
        """chunk_no 是 1 基（replace_chunks 先自增再插）。"""
        with db.conn() as c:
            got = db.chunks_by_no(c, self.pid, [3, 1])
        self.assertEqual([g["chunk_no"] for g in got], [3, 1])

    def test_missing_chunk_no_is_skipped_not_crashed(self):
        with db.conn() as c:
            self.assertEqual(db.chunks_by_no(c, self.pid, [999]), [])
            self.assertEqual(db.chunks_by_no(c, self.pid, []), [])

    def test_candidate_ids_path_also_gets_hits(self):
        """agent / deep_answer / RCS 兜底走的是 candidate_ids——不补的话
        同一个功能两条路两种行为。"""
        with mock.patch.object(rag, "CONTEXT_UNIT", "chunk"):
            ctx = rag.prepare([{"role": "user", "content": "quantization memory"}],
                              top_k=1, candidate_ids=[self.pid])
        self.assertIn("3 Method", ctx.system,
                      "candidate_ids 路径没拿到块级命中，退回按页猜了")


class RankingUsesHitsAsBoostNotGate(TempDbTestCase):
    """第一版把上下文限制成「只有命中的那几块」，实测证据召回 0.2614→0.1818。
    检索信号该用来排序，不该用来删候选。"""

    prefix = "papernest_boost"

    def setUp(self):
        super().setUp()
        db.init_db()

    def _chunks(self, n):
        return [{"chunk_no": i, "section_path": f"S{i}", "start_page": i,
                 "text": f"section {i} " + ("alpha " * (i + 1))} for i in range(n)]

    def test_non_hit_chunks_can_still_win_on_density(self):
        chunks = self._chunks(4)
        hits = [{"chunk_no": 0, "section_path": "S0", "start_page": 0,
                 "text": chunks[0]["text"]}]
        picked, blind = rag._rank_chunks(chunks, hits, ["alpha"], per_paper=2)
        self.assertFalse(blind)
        nos = [p["chunk_no"] for p in picked]
        self.assertTrue(any(n != 0 for n in nos),
                        "命中集之外的块一个都进不来——那是闸门，不是加成")

    def test_hit_gets_a_boost_on_a_tie(self):
        chunks = [{"chunk_no": 0, "section_path": "A", "start_page": 1, "text": "alpha x"},
                  {"chunk_no": 1, "section_path": "B", "start_page": 2, "text": "alpha x"}]
        hits = [{"chunk_no": 1, "section_path": "B", "start_page": 2, "text": "alpha x"}]
        picked, _ = rag._rank_chunks(chunks, hits, ["alpha"], per_paper=1)
        self.assertEqual(picked[0]["chunk_no"], 1, "并列时命中的块没有享到加成")

    def test_zero_lexical_hits_falls_back_to_retrieval_signal(self):
        """中文问句 × 英文正文：词面全 0，只剩检索信号。"""
        chunks = self._chunks(3)
        hits = [{"chunk_no": 2, "section_path": "S2", "start_page": 2, "text": "x"}]
        picked, blind = rag._rank_chunks(chunks, hits, ["近场信道"], per_paper=1)
        self.assertTrue(blind, "词面全 0 却没标成只靠检索信号")
        self.assertEqual(picked[0]["chunk_no"], 2)

    def test_blind_selection_is_reported_as_degraded(self):
        """只靠检索信号选出来的块，与按页选块的 fallback 是同一类事实，要如实上报。

        中文问句 × 英文正文：词面命中恒为 0，选出来的块只由检索信号决定。
        """
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:4200", "title": "English Only Paper",
                "abstract": "", "year": 2024, "venue": "V", "authors": [],
                "doi": None, "arxiv_id": "4200", "source": "s2"})
            db.replace_chunks(c, pid, [
                {"text": "Introduction to beamforming.", "section_path": "1",
                 "start_page": 1},
                {"text": "Channel estimation in the near field.",
                 "section_path": "3", "start_page": 3}])
            c.commit()
        hits = [{"chunk_no": 2, "section_path": "3", "start_page": 3,
                 "text": "Channel estimation in the near field."}]
        _block, note = rag._evidence_context(
            pid, "近场信道估计怎么做", hits, 1, 600, unit="chunk")
        self.assertEqual(note, "fallback",
                         "词面全 0、只靠检索信号选块，却没有如实上报")


class PickPagesImprovements(unittest.TestCase):
    """A→B 那一步：密度归一 + 命中窗口。这两个是保留下来的改动。"""

    def test_long_page_no_longer_wins_by_size_alone(self):
        pages = {1: "alpha " * 3 + "filler " * 200,      # 命中多但极长
                 2: "alpha alpha"}                        # 命中少但极短、密度高
        picked = rag._pick_pages(pages, ["alpha"], per_paper=1, cap=4000)
        self.assertIn("第 2 页", picked[0],
                      "绝对词频让最长的页恒赢——真库 83% 的情况选中的就是最长页")

    def test_truncation_keeps_the_match_not_the_head(self):
        text = "x" * 3000 + " NEEDLE " + "y" * 3000
        out = rag._window(text, ["needle"], 600)
        self.assertIn("NEEDLE", out, "从页首硬截把证据切掉了")
        self.assertLessEqual(len(out), 601)

    def test_window_falls_back_to_head_when_no_match(self):
        self.assertTrue(rag._window("abc" * 500, ["zzz"], 100).startswith("abc"))

    def test_short_text_is_untouched(self):
        self.assertEqual(rag._window("short", ["s"], 100), "short")


class NegativeResultIsPinned(unittest.TestCase):
    """默认单元必须是 page。改这个默认值前请先跑 `cli.py qasper ctxab`。"""

    def test_default_context_unit_is_page(self):
        self.assertEqual(
            rag.CONTEXT_UNIT, "page",
            "2026-09-03 修完 span 拼接吃空格、重切全库之后重测："
            "保真缺口已归零（strip 与保留空白两个口径逐位相同），"
            "但 page→chunk 仍是 0.2614→0.2500（-0.0114，17/88 题变化，p=1.0）——"
            "**实质持平，没有可测量的收益**。没有收益就不改主路径的默认值。"
            "要改先跑 `python cli.py qasper ctxab` 拿出新数字。")

    def test_the_switch_exists_and_works(self):
        """负结果不等于代码没接通——开关要能打开，否则下次没法重测。"""
        self.assertIn(rag.CONTEXT_UNIT, ("page", "chunk"))
        self.assertTrue(hasattr(rag, "_rank_chunks"))
        self.assertTrue(hasattr(rag, "_chunk_context"))


if __name__ == "__main__":
    unittest.main()
