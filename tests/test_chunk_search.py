"""L2 全文进全库检索：存储单元（pages，按页）与检索单元（chunks，按章节）分离。

背景：papers_fts 只索引 title/abstract/keywords，「每篇论文只精读一次」拿到的正文
在全库检索里等于不存在——L2 最大的浪费。而检索单元按页切会把章节拦腰砍断：
实测（25 篇真实 arXiv PDF、69 道 QASPER 带 gold 证据的题）等上下文预算下
按章节切的证据召回是按页切的 2.2~3.2 倍，所以两个单元必须分开。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, embeddings


class ChunkSearchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_chunks_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._embed = mock.patch("papernest.embeddings.available", return_value=False)
        self._embed.start()
        self._old_weight = embeddings.PAGE_WEIGHT
        db.init_db()
        with db.conn() as c:
            # 标题/摘要里完全没有 "quantization"，只有正文里有
            self.p_body = db.insert_l0(c, {
                "norm_key": "arxiv:5001", "title": "Efficient Inference for Language Models",
                "abstract": "We study inference efficiency in production settings.",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "5001", "source": "s2"})
            self.p_title = db.insert_l0(c, {
                "norm_key": "arxiv:5002", "title": "A Survey of Retrieval Systems",
                "abstract": "Retrieval systems overview.",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "5002", "source": "s2"})
            for no, txt in ((1, "Introduction\nWe apply eight bit quantization "
                                "to the attention projections."),
                            (2, "Results\nQuantization reduces latency by 40 percent.")):
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (self.p_body, no, txt))
            db.reindex_pages(c, self.p_body)
            db.replace_chunks(c, self.p_body, [
                {"text": "We apply eight bit quantization to the attention projections.",
                 "section_path": "1 Introduction", "level": 1,
                 "start_page": 1, "end_page": 1},
                {"text": "Quantization reduces latency by 40 percent.",
                 "section_path": "4 Results", "level": 1,
                 "start_page": 2, "end_page": 2}])

    def tearDown(self):
        embeddings.PAGE_WEIGHT = self._old_weight
        self._embed.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 索引同步 ──

    def test_replace_chunks_populates_both_table_and_index(self):
        with db.conn() as c:
            self.assertEqual(c.execute("SELECT COUNT(1) n FROM chunks").fetchone()["n"], 2)
            self.assertEqual(c.execute("SELECT COUNT(1) n FROM chunks_fts").fetchone()["n"], 2)

    def test_replace_is_idempotent_and_clears_stale_rows(self):
        """重切不该留旧影子。"""
        with db.conn() as c:
            db.replace_chunks(c, self.p_body, [{"text": "only one chunk now"}])
            self.assertEqual(c.execute("SELECT COUNT(1) n FROM chunks").fetchone()["n"], 1)
            self.assertEqual(c.execute("SELECT COUNT(1) n FROM chunks_fts").fetchone()["n"], 1)

    def test_empty_chunks_are_dropped_not_indexed(self):
        """空块只会污染 FTS 索引。"""
        with db.conn() as c:
            n = db.replace_chunks(c, self.p_body,
                                  [{"text": "real"}, {"text": "   "}, {"text": ""}])
        self.assertEqual(n, 1)

    def test_replacing_one_paper_does_not_touch_another(self):
        with db.conn() as c:
            db.replace_chunks(c, self.p_title, [{"text": "unrelated"}])
            n = c.execute("SELECT COUNT(1) n FROM chunks WHERE paper_id=?",
                          (self.p_body,)).fetchone()["n"]
        self.assertEqual(n, 2)

    def test_chunks_from_pages_fallback_preserves_page_numbers(self):
        """章节识别失败时退化按页——页码不能丢，证据链要接得上。"""
        with db.conn() as c:
            cs = db.chunks_from_pages(c, self.p_body)
        self.assertEqual([c["start_page"] for c in cs], [1, 2])

    def test_replacing_chunks_invalidates_their_vectors(self):
        """重切之后旧的 chunk 向量必须作废。

        向量按 idx=chunk_no 定位，重切后 chunk_no 的含义全变了——idx=1 的向量
        对应的是旧文本、而 chunks[1] 已是另一段内容；块数变少时多出来的向量更是
        指向已不存在的块。留着会让检索**用旧向量匹配、回取新文本**，不报错但结果错。
        """
        with db.conn() as c:
            for i in (1, 2):
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,?,?,?,?,?)",
                          (self.p_body, "chunk", i, f"旧第{i}块", "m", bytes(8)))
            # 同篇的 paper 级向量与别篇的 chunk 向量都不该被误删
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.p_body, "paper", 0, "论文级", "m", bytes(8)))
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.p_title, "chunk", 1, "别篇的", "m", bytes(8)))

            db.replace_chunks(c, self.p_body, [{"text": "第二版：完全不同的内容"}])

            def n(where, *a):
                return c.execute("SELECT COUNT(1) n FROM vectors WHERE " + where,
                                 a).fetchone()["n"]
            self.assertEqual(n("paper_id=? AND kind='chunk'", self.p_body), 0,
                             "本篇的旧 chunk 向量应被作废")
            self.assertEqual(n("paper_id=? AND kind='paper'", self.p_body), 1,
                             "paper 级向量与 chunk 无关，不能误删")
            self.assertEqual(n("paper_id=?", self.p_title), 1,
                             "别篇的向量不能误删")


    # ── 检索 ──

    def test_body_only_term_is_findable(self):
        with db.conn() as c:
            self.assertEqual(db.search_fts(c, "quantization", 5), [])   # 标题/摘要路搜不到
            hits = db.search_chunks_fts(c, "quantization", 5)
        self.assertTrue(hits)
        self.assertEqual({h["paper_id"] for h in hits}, {self.p_body})

    def test_hits_carry_section_path_and_page(self):
        """命中要能溯源到章节与页码——这是本项目「证据可审计」的口径。"""
        with db.conn() as c:
            hits = db.search_chunks_fts(c, "latency", 5)
        self.assertEqual(hits[0]["section_path"], "4 Results")
        self.assertEqual(hits[0]["start_page"], 2)

    def test_ranked_aggregation_dedupes_to_papers(self):
        with db.conn() as c:
            self.assertEqual(db.search_chunks_ranked(c, "quantization", 10), [self.p_body])

    def test_short_query_returns_nothing_rather_than_scanning(self):
        with db.conn() as c:
            self.assertEqual(db.search_chunks_fts(c, "a", 5), [])
            self.assertEqual(db.search_chunks_fts(c, "", 5), [])

    # ── 融合行为 ──

    def test_hybrid_finds_body_only_paper_when_enabled(self):
        _r = embeddings.search_hybrid("quantization", 5, use_pages=True)
        ids, mode = _r.ids, _r.mode
        self.assertIn(self.p_body, ids)
        self.assertIn("chunks", mode)

    def test_hybrid_misses_it_when_disabled(self):
        _r = embeddings.search_hybrid("quantization", 5, use_pages=False)
        ids, mode = _r.ids, _r.mode
        self.assertNotIn(self.p_body, ids)
        self.assertNotIn("chunks", mode)

    def test_title_match_still_outranks_body_match_at_default_weight(self):
        """默认权重 0.2 下正文命中不能盖过标题命中——权重调高会反转，实测让自建集掉分。"""
        embeddings.PAGE_WEIGHT = 0.2
        with db.conn() as c:
            db.replace_chunks(c, self.p_body, [
                {"text": "We also discuss retrieval systems at length.",
                 "section_path": "2 Related Work"}])
        ids = embeddings.search_hybrid("retrieval systems", 5, use_pages=True).ids
        self.assertEqual(ids[0], self.p_title)

    def test_ordering_is_deterministic(self):
        """排序键必须全序：并列时按 paper_id 决胜，否则跨进程漂移。"""
        runs = {tuple(embeddings.search_hybrid("quantization systems", 5,
                                               use_pages=True)[0]) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_no_chunks_means_mode_has_no_suffix(self):
        with db.conn() as c:
            c.execute("DELETE FROM chunks_fts")
        mode = embeddings.search_hybrid("quantization", 5, use_pages=True).mode
        self.assertNotIn("chunks", mode)


class ChunkMigrationTests(unittest.TestCase):
    def test_migration_backfills_chunks_from_pages(self):
        """老库升级：pages 里已有的全文要搬进 chunks，升级完检索不空窗。"""
        tmp = tempfile.mkdtemp(prefix="papernest_mig5_")
        self.addCleanup(shutil.rmtree, tmp, True)
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        try:
            config.DB_PATH = Path(tmp) / "old.db"
            config.DATA_DIR = Path(tmp) / "data"
            db.init_db(force=True)
            with db.conn() as c:
                pid = db.insert_l0(c, {
                    "norm_key": "arxiv:4001", "title": "Legacy Paper", "abstract": "x",
                    "year": 2020, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": "4001", "source": "s2"})
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (pid, 1, "legacy body text about differential privacy"))
                c.execute("DELETE FROM chunks")
                c.execute("DELETE FROM chunks_fts")
                c.execute("PRAGMA user_version=4")
            db.init_db(force=True)
            with db.conn() as c:
                self.assertEqual(c.execute("SELECT COUNT(1) n FROM chunks").fetchone()["n"], 1)
                self.assertTrue(db.search_chunks_fts(c, "differential privacy", 5))
        finally:
            config.DB_PATH, config.DATA_DIR = old_db, old_dir


if __name__ == "__main__":
    unittest.main()
