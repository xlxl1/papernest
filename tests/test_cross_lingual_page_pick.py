# -*- coding: utf-8 -*-
"""中文提问 × 英文正文：选页不能只靠字面匹配。

`_pick_pages` 用的是纯字面 `t.count(k)`，而 `_query_terms` 对中文切的是二元词面——
中文问句在英文正文里命中恒为 0。真库实测（45 篇有 L2 全文的论文 × eval_set 里
50 条中文问句 / 32 条英文问句，共 2250 与 1440 个「问句×论文」组合）：

    中文提问：选得出页 34/2250   → **fallback 率 98.5%**
    英文提问：选得出页 1040/1440 → fallback 率 27.8%

也就是说在项目自称的**主路径**上，`PAGE_PICK_FALLBACK` 不是边界情况而是稳态：
进上下文的是「本篇核心页」，与用户这一次问什么无关。

而信号一直都在：`search_hybrid` 早就把向量赢下的那一块取回来了
（`chunk_hits` 里 via='vector' 的项，`db.chunks_by_no` 带 `start_page`），
`embeddings.py` 那句注释写着「词面命中恒为 0 的跨语言场景下，这是唯一知道该喂
哪段的信息来源」——只是这条信号从来没传到按页组装这条路上（`_evidence_context`
调 `_page_context` 时把 hits 丢了）。

真库结构上限：45 篇有全文的论文**全部**有 chunk 向量，且 `chunk.start_page`
都能落回 pages 表，覆盖 408 个不同页。所以只要向量路出得来命中，这条路就有信号。

这里的 hits 是**注入**的（形状与 `db.chunks_by_no` 返回的一致），所以用例不依赖
网络、不花 embedding 额度，也不受额度耗尽影响。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, rag

CN_QUESTION = "这个方法在信道估计上的误差有多大"      # 纯中文，英文正文里字面恒 0
BODY_PAGES = {
    1: "Introduction. " + ("This paper studies neural machine translation. " * 30),
    2: "Related work. " + ("Prior systems rely on phrase based models. " * 30),
    7: "Experiments. We report a channel estimation NMSE of 0.031 on the test split. "
       + ("Additional discussion follows. " * 25),
}


class CrossLingualPagePickTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_xling_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        for p in (mock.patch("papernest.embeddings.available", return_value=False),
                  mock.patch("papernest.llm.available", return_value=False)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._restore)
        db.init_db()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:9100", "title": "Neural Machine Translation Study",
                "abstract": "We study neural machine translation.", "year": 2023,
                "venue": "ACL", "authors": ["A B"], "doi": None,
                "arxiv_id": "9100", "source": "s2"})
            for pno, text in BODY_PAGES.items():
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (self.pid, pno, text))
            db.reindex_pages(c, self.pid)

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    #: 向量路赢下的那一块，形状与 db.chunks_by_no 的返回一致
    VEC_HIT = [{"chunk_no": 5, "section_path": "Experiments", "start_page": 7,
                "text": "We report a channel estimation NMSE of 0.031", "via": "vector"}]

    def test_literal_match_really_is_zero_for_a_chinese_question(self):
        """先把前提坐实：这个中文问句在这篇英文正文里字面命中确实是 0。

        修法前后都必须绿——它保证下面那条用例真的咬在跨语言这件事上。
        """
        terms = rag._query_terms(CN_QUESTION)
        self.assertTrue(terms, "中文问句一个检索词都没切出来，用例前提不成立")
        self.assertEqual([], rag._pick_pages(BODY_PAGES, terms, 2, 2400),
                         "字面匹配居然选出了页，这个夹具不是跨语言场景")

    def test_semantic_hit_locates_the_page_instead_of_falling_back(self):
        """有向量命中时，要按命中块的页选页，而不是退回「本篇核心页」。"""
        block, note = rag._page_context(self.pid, CN_QUESTION, per_paper=2,
                                        cap=2400, hits=self.VEC_HIT)
        self.assertIn("【第 7 页】", block,
                      f"没有用语义命中的页，进上下文的是：{block[:120]!r}")
        self.assertIsNone(
            note,
            "按语义命中选出的页是**因为这次提问**才被选中的，不该记成 fallback")

    def test_without_hits_it_still_reports_the_fallback_honestly(self):
        """没有语义命中时行为不变：退回本篇关键词选页，并如实标 fallback。"""
        block, note = rag._page_context(self.pid, CN_QUESTION, per_paper=2, cap=2400)
        self.assertEqual(note, "fallback",
                         "没有任何信号却没报 fallback——那正是这条修复要消灭的静默")
        self.assertTrue(block, "退化路径也该给出页块")

    def test_literal_hits_still_win_over_semantic_ones(self):
        """英文提问照旧走字面选页——修法只补跨语言那条空缺，不改已经有效的判据。"""
        block, note = rag._page_context(
            self.pid, "channel estimation NMSE experiments", per_paper=1,
            cap=2400, hits=[{"chunk_no": 1, "section_path": "Intro",
                             "start_page": 1, "text": "x", "via": "vector"}])
        self.assertIn("【第 7 页】", block, "字面命中的页被语义命中顶掉了")
        self.assertIsNone(note)

    def test_hits_are_threaded_through_evidence_context(self):
        """接线本身也要钉住：`_evidence_context` 不能再把 hits 丢掉。"""
        block, note = rag._evidence_context(self.pid, CN_QUESTION, self.VEC_HIT,
                                            per_paper=2, cap=2400, unit="page")
        self.assertIn("【第 7 页】", block, "hits 没有传到 _page_context")
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
