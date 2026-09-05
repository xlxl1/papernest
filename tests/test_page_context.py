# -*- coding: utf-8 -*-
"""L2 全文页进上下文的选页口径：中英取词、跨语言兜底、以及「不再静默」。

背景（实测，不是推测）：改之前 `_page_context` 用 `q.split()` 取词——中文问句没有空格，
整句被当成一个「词」，拿去 `t.count()` 数英文正文恒为 0，于是**中文提问时全文页永远
进不了上下文**。库内 45 篇有全文的论文里 40 篇（88.9%）正文是英文，而项目的主语言是中文，
所以默认问答路径下模型只剩 abstract[:500]，却被 SYSTEM 要求「优先引用文献原句」。
这些用例把三种口径（正常 / 跨语言兜底 / 完全未命中）分别钉住。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, rag

EN_BODY = ("We study near-field channel estimation for extremely large aperture arrays. "
           "The proposed method uses a polar-domain dictionary to capture spherical wavefronts. ")
CN_BODY = "本文研究近场信道估计问题，提出一种极化域字典方法来刻画球面波前。"


class PageContextTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_pagectx_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 向量按住：开发机 .env 真配了 EMBED_MODEL，不摁住会真发请求
        self._patches = [mock.patch("papernest.embeddings.available", return_value=False),
                         mock.patch("papernest.llm.available", return_value=False)]
        for p in self._patches:
            p.start()
        db.init_db()
        with db.conn() as c:
            self.p_en = db.insert_l0(c, {
                "norm_key": "arxiv:9001",
                "title": "Near-Field Channel Estimation for Extremely Large Aperture Arrays",
                "abstract": "Polar-domain dictionary for spherical wavefronts.",
                "year": 2023, "venue": "TWC", "authors": ["A B"],
                "doi": None, "arxiv_id": "9001", "source": "s2"})
            self.p_cn = db.insert_l0(c, {
                "norm_key": "arxiv:9002", "title": "近场信道估计综述",
                "abstract": "综述近场信道估计的主流方法。",
                "year": 2024, "venue": "学报", "authors": ["丙 丁"],
                "doi": None, "arxiv_id": "9002", "source": "s2"})
            self.p_bare = db.insert_l0(c, {
                "norm_key": "arxiv:9003", "title": "Unrelated Work On Compilers",
                "abstract": "Nothing to do with wireless.",
                "year": 2020, "venue": "PLDI", "authors": ["C D"],
                "doi": None, "arxiv_id": "9003", "source": "s2"})
            for pno in (1, 2, 3):
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (self.p_en, pno, EN_BODY * (4 - pno)))   # 第 1 页命中最多
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (self.p_cn, pno, CN_BODY * (4 - pno)))

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 取词 ──

    def test_chinese_question_is_not_one_giant_token(self):
        """这是原 bug 的根：中文问句 split() 只切出整句，永远匹配不到任何正文。"""
        terms = rag._query_terms("近场信道估计有哪些主流方法")
        self.assertIn("近场", terms)
        self.assertIn("信道", terms)
        self.assertTrue(all(len(t) == 2 for t in terms), terms)

    def test_ascii_and_cjk_are_tokenized_together(self):
        terms = rag._query_terms("multi-modal 大模型的 limitations")
        self.assertIn("multi-modal", terms)      # 连字符不拆开
        self.assertIn("limitations", terms)
        self.assertIn("大模", terms)

    def test_terms_are_deduped_and_order_stable(self):
        """同一条问句必须每次切出同样的词，顺序也一样——否则选页结果会跨进程漂移。"""
        q = "信道信道估计 channel Channel"
        self.assertEqual(rag._query_terms(q), rag._query_terms(q))
        self.assertEqual(len(rag._query_terms(q)), len(set(rag._query_terms(q))))

    # ── 三种选页口径 ──

    def test_matching_language_picks_pages_normally(self):
        block, note = rag._page_context(self.p_cn, "近场信道估计有哪些主流方法")
        self.assertIsNone(note)
        self.assertIn("【第 1 页】", block)

    def test_english_question_on_english_body_still_works(self):
        block, note = rag._page_context(self.p_en, "near-field channel estimation method")
        self.assertIsNone(note)
        self.assertIn("【第 1 页】", block)

    def test_chinese_question_on_english_body_falls_back_not_silent(self):
        """原来这里返回空串：中文问 + 英文论文，全文页永远进不了上下文，且无任何提示。"""
        block, note = rag._page_context(self.p_en, "近场信道估计有哪些主流方法")
        self.assertEqual(note, "fallback")
        self.assertIn("【第", block)

    def test_paper_without_fulltext_is_not_an_anomaly(self):
        block, note = rag._page_context(self.p_bare, "任何问题")
        self.assertEqual(block, "")
        self.assertIsNone(note)          # 「本来就没有全文」不是降级

    def test_tie_break_is_total_order(self):
        """两页分数相同时必须按页码决胜，否则同一条问句可能选出不同的页。"""
        pages = {7: "alpha beta", 3: "alpha beta", 5: "alpha beta"}
        picked = rag._pick_pages(pages, ["alpha"], per_paper=2, cap=100)
        self.assertEqual(["    【第 3 页】alpha beta", "    【第 5 页】alpha beta"], picked)

    # ── 上层如实上报 ──

    def test_prepare_reports_cross_language_fallback_in_degraded(self):
        _sys, _q, _src, degraded, _n = rag.prepare(
            [{"role": "user", "content": "近场信道估计有哪些主流方法"}],
            top_k=1, candidate_ids=[self.p_en])
        self.assertIsNotNone(degraded)
        self.assertIn("按本篇关键词选页", degraded)

    def test_prepare_is_quiet_when_pages_were_picked_by_the_question(self):
        _sys, _q, _src, degraded, _n = rag.prepare(
            [{"role": "user", "content": "近场信道估计有哪些主流方法"}],
            top_k=1, candidate_ids=[self.p_cn])
        self.assertIsNone(degraded)      # 正常命中不该报降级

    def test_prepare_puts_page_text_into_the_prompt(self):
        sys_prompt, _q, _src, _deg, _n = rag.prepare(
            [{"role": "user", "content": "近场信道估计有哪些主流方法"}],
            top_k=1, candidate_ids=[self.p_en])
        self.assertIn("【第", sys_prompt)


if __name__ == "__main__":
    unittest.main()
