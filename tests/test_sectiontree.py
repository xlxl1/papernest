"""章节树索引与两级检索：重点钉住「两级路由」独有的失败模式。

两级检索比全库一把梭多了一道**筛选**，也就多了一种朴素检索不会有的错法：
第一级把 gold 论文筛掉了，第二级排得再准也白搭，而且失败是静默的——
结果看上去干干净净，只是少了一篇。所以这个文件测的重点不是 happy path：

- 第一级误杀（`test_two_stage_not_worse_than_full_search`，本文件最重要的一条）；
- 章节树的父子关系被建错（`test_qasper_subsection_hierarchy_*`）——这是拿真库
  data/papernest.db 跑出来的真实回归：QASPER 论文 466 的 "Related Work ::: X" 曾被
  挂到 "Introduction" 底下，节点数和 level 全对，只有父子关系是假的；
- 「标题命中权重高于正文命中」这条设计能不能被**手算**验证，而不是「跑通就行」；
- 重复建索引留下影子节点；
- 空库 / 无索引 / 首行不是标题 → 要降级，不要崩。

夹具按真实数据的形态造：QASPER 导入的论文是「一页一节、首行是节名、子节用 :::
分隔」，PDF 抽出来的页则是「首行是页码数字」——两种都造进来了。
"""
import shutil
import itertools
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, sectiontree as st


def _page(title: str, body: str) -> str:
    """按 qasper.import_papers 的真实写法拼页文本：首行节名，其后正文。"""
    return f"{title}\n{body}"


class SectionTreeBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_sectree_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 开发机 .env 真配了 EMBED_MODEL / LLM_*，不摁住会真发请求
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            mock.patch("papernest.llm.available", return_value=False),
        ]
        for p in self._patches:
            p.start()
        db.init_db()
        st.ensure_schema()
        self.ids = {}
        with db.conn() as c:
            self.ids["ce"] = self._paper(c, "ce", "Channel Estimation for XL-MIMO Systems", [
                _page("Introduction",
                      "Massive antenna arrays promise high spectral efficiency. "
                      "We study the near-field regime in this work."),
                # 标题里有查询词、正文里一次都没有 —— 与下面那节构成手算对照
                _page("Channel Estimation for Large Arrays",
                      "We propose a scheme for very large antenna arrays "
                      "under pilot contamination and hardware impairment."),
                # 正文里堆了 10 次查询词、标题完全无关
                _page("Related Work",
                      " ".join(["The channel estimation problem is discussed."] * 10)),
                _page("Conclusion",
                      "We summarised a scheme for large antenna arrays."),
            ])
            self.ids["ocean"] = self._paper(c, "ocean", "Ocean Buoy Temperature Records", [
                _page("Introduction",
                      "Buoy networks record sea surface temperature every hour."),
                _page("Measurements",
                      "Salinity and temperature drift are calibrated yearly."),
            ])
            # QASPER 真实形态：子节用 ::: 分隔，父节自己没有段落、不单独成页
            self.ids["qasper"] = self._paper(c, "qasper", "Contextual Embedding Study", [
                _page("Introduction", "We study contextualized representations."),
                _page("Related Work ::: Static Word Embeddings",
                      "GloVe and word2vec produce one vector per word type."),
                _page("Related Work ::: Probing Tasks",
                      "Probing classifiers reveal syntactic information."),
                _page("Approach ::: Data",
                      "We use the SemEval Semantic Textual Similarity corpus."),
                _page("Approach ::: Measures ::: Definition 1",
                      "Self-similarity is the average cosine similarity."),
            ])
            # PDF 抽出来的页：首行是页码，不是节标题。一位数和两位数各造一页——
            # 挡掉它们靠的是两条不同的判据（长度 < 2 / 不含字母），少一页就漏测一条。
            self.ids["pdf"] = self._paper(c, "pdf", "A Survey Rendered From PDF", [
                "1\nLarge Language Model Agent: A Survey on\nMethodology and Challenges",
                "2\nAgent Methodology Construction Evolution Collaboration",
                "12\nDecentralized Collaboration And Hybrid Architecture",
            ])
            # gold 只出现在深层子节的正文里，不在任何标题里 —— 第一级最容易误杀的形态
            self.ids["deep"] = self._paper(c, "deep", "Speech Corpus Construction", [
                _page("Introduction", "We build a spoken language resource."),
                _page("Corpus ::: Annotation Protocol",
                      "Two annotators labelled every utterance for prosodic boundary "
                      "tone, and disagreements were resolved by a third annotator."),
            ])

    def _paper(self, c, key: str, title: str, pages: list[str]) -> int:
        pid = db.insert_l0(c, {"norm_key": f"qasper:{key}", "title": title,
                               "abstract": "", "year": 2021, "venue": "QASPER-dev",
                               "authors": [], "doi": None, "arxiv_id": None,
                               "source": "qasper"})
        for no, text in enumerate(pages, 1):
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (pid, no, text))
        db.reindex_pages(c, pid)        # 全库检索的对照组要能用
        return pid

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _build_all(self):
        return st.build_all()

    def _nodes(self, paper_id: int) -> list[dict]:
        with db.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM section_nodes WHERE paper_id=? ORDER BY id", (paper_id,))]


# ── ① 建索引 ──

class BuildIndexTests(SectionTreeBase):
    def test_nodes_levels_and_paths(self):
        r = st.build_index(self.ids["ce"])
        self.assertEqual(r["source"], "pages")
        self.assertIsNone(r["degraded"])
        nodes = self._nodes(self.ids["ce"])
        self.assertEqual([n["title"] for n in nodes],
                         ["Introduction", "Channel Estimation for Large Arrays",
                          "Related Work", "Conclusion"])
        self.assertEqual([n["level"] for n in nodes], [1, 1, 1, 1])
        self.assertEqual([n["node_id"] for n in nodes],
                         [f"P{self.ids['ce']}/S{i}" for i in (1, 2, 3, 4)])
        self.assertEqual([n["parent_id"] for n in nodes], [None] * 4)
        self.assertEqual([n["start_page"] for n in nodes], [1, 2, 3, 4])
        self.assertEqual(nodes[0]["section_path"], "Introduction")
        for n in nodes:
            self.assertRegex(n["node_id"], r"^P\d+/S[\d.]+$")
            self.assertGreater(n["n_chars"], 0)

    def test_qasper_subsection_hierarchy_uses_path_not_position(self):
        """真库回归：'Related Work ::: X' 必须挂在 'Related Work' 下，不是 'Introduction' 下。

        旧实现按「出现顺序 + level」发号，父节没有独立成页时就会把整棵子树挂错父亲，
        而节点数、level 全都对得上——只有父子关系是假的，靠计数断言根本发现不了。
        """
        st.build_index(self.ids["qasper"])
        nodes = {n["node_id"]: n for n in self._nodes(self.ids["qasper"])}
        by_title = {n["title"]: n for n in nodes.values()}

        self.assertIn("Related Work", by_title)          # 补建出来的父节点
        self.assertIn("Approach", by_title)
        static = by_title["Static Word Embeddings"]
        self.assertEqual(nodes[static["parent_id"]]["title"], "Related Work")
        self.assertEqual(by_title["Probing Tasks"]["parent_id"], static["parent_id"])
        self.assertEqual(nodes[by_title["Data"]["parent_id"]]["title"], "Approach")

        # 补建的父节点：有标题、无正文、无页区间（第二级据此跳过它，不重复出片段）
        rel = by_title["Related Work"]
        self.assertEqual(rel["n_chars"], 0)
        self.assertIsNone(rel["start_page"])
        self.assertEqual(rel["level"], 1)

        # 三层路径也要正确
        d1 = by_title["Definition 1"]
        self.assertEqual(d1["level"], 3)
        self.assertEqual(d1["section_path"], "Approach > Measures > Definition 1")
        self.assertEqual(nodes[d1["parent_id"]]["title"], "Measures")

    def test_no_dangling_parent_anywhere(self):
        self._build_all()
        with db.conn() as c:
            dangling = c.execute(
                "SELECT COUNT(*) FROM section_nodes n WHERE n.parent_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM section_nodes p "
                "                WHERE p.paper_id=n.paper_id AND p.node_id=n.parent_id)"
            ).fetchone()[0]
        self.assertEqual(dangling, 0)

    def test_pdf_like_pages_degrade_instead_of_faking_titles(self):
        """PDF 页的首行是页码 '1'/'2'，不能被当成节标题存进去。"""
        r = st.build_index(self.ids["pdf"])
        self.assertIsNotNone(r["degraded"])
        self.assertIn("首行不像节标题", r["degraded"])
        nodes = self._nodes(self.ids["pdf"])
        self.assertEqual([n["title"] for n in nodes], ["", "", ""])
        self.assertEqual([n["section_path"] for n in nodes],
                         ["（第 1 页）", "（第 2 页）", "（第 3 页）"])
        # 标题认不出来，正文仍然要整页入索引，不能连正文一起丢
        self.assertTrue(all(n["n_chars"] > 0 for n in nodes))

    def test_rebuild_leaves_no_duplicate_or_stale_rows(self):
        """核心断言：同一篇重复 build 不产生重复行，也不留旧节点。"""
        first = st.build_index(self.ids["ce"])
        before = self._nodes(self.ids["ce"])
        with db.conn() as c:
            terms_before = c.execute(
                "SELECT COUNT(*) FROM section_terms WHERE paper_id=?",
                (self.ids["ce"],)).fetchone()[0]

        second = st.build_index(self.ids["ce"])
        after = self._nodes(self.ids["ce"])
        with db.conn() as c:
            terms_after = c.execute(
                "SELECT COUNT(*) FROM section_terms WHERE paper_id=?",
                (self.ids["ce"],)).fetchone()[0]

        self.assertEqual(first["nodes"], second["nodes"])
        self.assertEqual(len(before), len(after))
        self.assertEqual(terms_before, terms_after)
        self.assertEqual([n["node_id"] for n in before], [n["node_id"] for n in after])
        self.assertEqual(len({n["node_id"] for n in after}), len(after))

    def test_rebuild_drops_nodes_that_no_longer_exist(self):
        """先用 5 节的结构建，再回落到 pages（4 节）：多出来的那一节必须消失。"""
        fake = [{"section_title": f"S{i}", "section_path": f"S{i}", "level": 1,
                 "start_page": i, "end_page": i, "text": f"body {i}"}
                for i in range(1, 6)]
        st.build_index(self.ids["ce"], sections=fake)
        self.assertEqual(len(self._nodes(self.ids["ce"])), 5)
        st.build_index(self.ids["ce"])
        nodes = self._nodes(self.ids["ce"])
        self.assertEqual(len(nodes), 4)
        self.assertNotIn("S5", [n["title"] for n in nodes])
        with db.conn() as c:
            stale = c.execute(
                "SELECT COUNT(*) FROM section_terms t WHERE t.paper_id=? AND NOT EXISTS"
                "(SELECT 1 FROM section_nodes n WHERE n.paper_id=t.paper_id "
                "                                 AND n.node_id=t.node_id)",
                (self.ids["ce"],)).fetchone()[0]
        self.assertEqual(stale, 0)

    def test_structure_chunks_parts_merge_into_one_node(self):
        """structure.section_chunks 把长章节切成多个 part，索引里必须并回一个节点。"""
        chunks = [
            {"section_title": "Method", "section_path": "Method", "level": 1,
             "start_page": 3, "end_page": 3, "text": "part one", "part": 1, "n_parts": 2},
            {"section_title": "Method", "section_path": "Method", "level": 1,
             "start_page": 4, "end_page": 4, "text": "part two", "part": 2, "n_parts": 2},
            {"section_title": "Training", "section_path": "Method > Training", "level": 2,
             "start_page": 5, "end_page": 5, "text": "sgd details", "part": 1, "n_parts": 1},
        ]
        r = st.build_index(self.ids["ce"], sections=chunks)
        self.assertEqual(r["source"], "sections")
        nodes = self._nodes(self.ids["ce"])
        self.assertEqual([n["title"] for n in nodes], ["Method", "Training"])
        self.assertEqual(nodes[0]["start_page"], 3)
        self.assertEqual(nodes[0]["end_page"], 4)          # 两个 part 的页区间合并
        self.assertEqual(nodes[1]["parent_id"], nodes[0]["node_id"])

    def test_build_all_three_state_counts_and_progress(self):
        with db.conn() as c:
            blank = db.insert_l0(c, {"norm_key": "qasper:blank", "title": "Blank",
                                     "abstract": "", "year": 2021, "venue": "v",
                                     "authors": [], "doi": None, "arxiv_id": None,
                                     "source": "qasper"})
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (blank, 1, "   "))
        seen = []
        r = st.build_all(progress=lambda f, s, m: seen.append((round(f, 3), s)))
        self.assertEqual(r["indexed"], 5)
        self.assertEqual(r["skipped"], 1)                  # 有 pages 但推不出节点
        self.assertEqual(r["failed"], 0)
        self.assertEqual(r["errors"], [])
        self.assertTrue(seen and seen[-1][0] == 1.0)

    def test_build_all_counts_failures_without_aborting(self):
        real = st.build_index
        bad = self.ids["ocean"]

        def flaky(pid, sections=None):
            if pid == bad:
                raise ValueError("boom")
            return real(pid, sections)

        with mock.patch("papernest.sectiontree.build_index", side_effect=flaky):
            r = st.build_all()
        self.assertEqual(r["failed"], 1)
        self.assertEqual(r["indexed"], 4)
        self.assertTrue(any("boom" in e for e in r["errors"]))


# ── ② 打分：标题命中必须显著压过正文命中 ──

class ScoringTests(SectionTreeBase):
    def test_title_weight_beats_a_hundred_body_hits_by_construction(self):
        """把设计意图写成数字：一次标题命中要压过同一个词在正文里出现 100 次。"""
        self.assertGreater(st.TITLE_W, st.BODY_W * (1.0 + math.log10(100)))

    def test_score_formula_is_hand_computable(self):
        """两节各命中 channel/estimation 两个词：

        标题节：每词 (1 + log10(1)) + TITLE_W = 1 + 6 = 7 → 共 14
        正文节：每词 (1 + log10(10))          = 1 + 1 = 2 → 共  4
        """
        title_hit = st.score_node({"channel": 1, "estimation": 1},
                                  {"channel", "estimation"}, set())
        body_hit = st.score_node({"channel": 10, "estimation": 10},
                                 {"related", "work"}, set())
        self.assertAlmostEqual(title_hit, 14.0, places=9)
        self.assertAlmostEqual(body_hit, 4.0, places=9)
        self.assertAlmostEqual(title_hit / body_hit, 3.5, places=9)

    def test_ancestor_path_hit_sits_between_title_and_body(self):
        only_body = st.score_node({"method": 1}, set(), set())
        via_path = st.score_node({"method": 1}, set(), {"method"})
        via_title = st.score_node({"method": 1}, {"method"}, set())
        self.assertLess(only_body, via_path)
        self.assertLess(via_path, via_title)

    def test_title_hit_node_outranks_body_heavy_node_in_real_scope(self):
        """同一篇论文里，标题含查询词的节要排在「正文提了 10 次但标题无关」的节前面。

        两节命中的词面完全相同，IDF 对两者同倍作用会整体约掉，所以分数比恒等于
        14 / 4 = 3.5——这条断言是手算得出来的，不是「跑通就行」。
        """
        st.build_index(self.ids["ce"])
        scope = st.select_scope("channel estimation", top_papers=5, top_nodes=10)
        mine = [n for n in scope["nodes"] if n["paper_id"] == self.ids["ce"]]
        titled = next(n for n in mine if n["title"] == "Channel Estimation for Large Arrays")
        bodyish = next(n for n in mine if n["title"] == "Related Work")
        self.assertLess(mine.index(titled), mine.index(bodyish))
        self.assertAlmostEqual(titled["score"] / bodyish["score"], 3.5, places=6)

    def test_terms_are_symmetric_for_chinese(self):
        """索引侧与查询侧共用一个切词函数：中文查询必须能命中中文正文。

        本项目的老坑是「索引按整词、查询按二元组」这类不对称展开导致静默 0 召回。
        """
        doc = set(st.terms_of("近场信道估计方法"))
        self.assertTrue(set(st.terms_of("信道估计")).issubset(doc))
        self.assertIn("信道", doc)
        self.assertNotIn("近", doc)          # 单字不成词面

    def test_stopwords_do_not_become_terms(self):
        self.assertEqual(st.terms_of("the of and a"), [])
        self.assertEqual(st.term_freq("model model data"), {"model": 2, "data": 1})


# ── ③ 第一级：选范围 ──

class SelectScopeTests(SectionTreeBase):
    def test_unrelated_paper_is_filtered_out(self):
        self._build_all()
        scope = st.select_scope("channel estimation pilot contamination",
                                top_papers=3, top_nodes=10)
        self.assertIn(self.ids["ce"], scope["paper_ids"])
        self.assertNotIn(self.ids["ocean"], scope["paper_ids"])
        self.assertTrue(all(n["paper_id"] in scope["paper_ids"] for n in scope["nodes"]))

    def test_matched_terms_are_honest(self):
        self._build_all()
        scope = st.select_scope("channel estimation buoy", top_papers=5, top_nodes=20)
        for n in scope["nodes"]:
            self.assertTrue(n["matched_terms"])
            self.assertEqual(n["matched_terms"], sorted(set(n["matched_terms"])))
            for t in n["matched_terms"]:
                self.assertIn(t, scope["query_terms"])
            with db.conn() as c:
                stored = {r["term"] for r in c.execute(
                    "SELECT term FROM section_terms WHERE paper_id=? AND node_id=?",
                    (n["paper_id"], n["node_id"]))}
            self.assertTrue(set(n["matched_terms"]).issubset(stored))

    def test_ties_are_broken_by_paper_id_then_node_id(self):
        """排序全序：内容完全相同的两篇论文分数一致，顺序必须由 (paper_id, node_id) 决定。"""
        with db.conn() as c:
            body = _page("Duplicate Section", "identical wording about widgets everywhere")
            a = self._paper(c, "twin_a", "Twin A", [body])
            b = self._paper(c, "twin_b", "Twin B", [body])
        st.build_index(a)
        st.build_index(b)
        scope = st.select_scope("widgets", top_papers=10, top_nodes=10)
        twins = [n for n in scope["nodes"] if n["paper_id"] in (a, b)]
        self.assertEqual(len(twins), 2)
        self.assertEqual(twins[0]["score"], twins[1]["score"])
        self.assertEqual([n["paper_id"] for n in twins], sorted([a, b]))

    def test_rank_key_is_a_total_order_under_any_input_order(self):
        """上面那条用例其实证明不了「有决胜键」——Python 的排序是稳定的，输入本来
        就有序时，把决胜键删掉结果也一样（这条断言是被变异测试抓出来补上的）。
        所以直接对排序键本身下手：穷举所有输入顺序，输出必须唯一。
        """
        rows = [{"score": 1.0, "paper_id": 9, "node_id": "P9/S2"},
                {"score": 1.0, "paper_id": 2, "node_id": "P2/S10"},
                {"score": 1.0, "paper_id": 2, "node_id": "P2/S2"},
                {"score": 2.0, "paper_id": 7, "node_id": "P7/S1"}]
        # node_id 按字符串比，所以 "P2/S10" 在 "P2/S2" 之前——难看但确定
        want = [(7, "P7/S1"), (2, "P2/S10"), (2, "P2/S2"), (9, "P9/S2")]
        for perm in itertools.permutations(rows):
            got = [(r["paper_id"], r["node_id"])
                   for r in sorted(perm, key=st.rank_key)]
            self.assertEqual(got, want)

    def test_passage_rank_key_breaks_ties_down_to_offset(self):
        rows = [{"score": 1.0, "paper_id": 3, "node_id": "P3/S1", "page_no": 2, "offset": 0},
                {"score": 1.0, "paper_id": 3, "node_id": "P3/S1", "page_no": 1, "offset": 9},
                {"score": 1.0, "paper_id": 3, "node_id": "P3/S1", "page_no": 1, "offset": 4}]
        want = [(1, 4), (1, 9), (2, 0)]
        for perm in itertools.permutations(rows):
            got = [(r["page_no"], r["offset"])
                   for r in sorted(perm, key=st.passage_rank_key)]
            self.assertEqual(got, want)

    def test_rare_term_outweighs_a_term_every_section_has(self):
        """IDF 是真在起作用的：只命中一个稀有词的节，要压过只命中「到处都有的词」的节。

        两节的 tf 都是 1、都没有标题命中，分差只可能来自 IDF。去掉 IDF 这条就挂。
        """
        with db.conn() as c:
            pid = self._paper(c, "idf", "IDF Probe", [
                _page("Alpha", "commonword shows up right here."),
                _page("Beta", "commonword shows up right here."),
                _page("Gamma", "commonword shows up right here."),
                _page("Delta", "commonword shows up right here."),
                _page("Zeta", "rareword shows up right here."),
            ])
        st.build_index(pid)
        scope = st.select_scope("commonword rareword", top_papers=5, top_nodes=20)
        mine = [n for n in scope["nodes"] if n["paper_id"] == pid]
        self.assertEqual(mine[0]["title"], "Zeta")
        self.assertEqual(mine[0]["matched_terms"], ["rareword"])
        common = next(n for n in mine if n["title"] == "Alpha")
        self.assertEqual(common["matched_terms"], ["commonword"])
        self.assertGreater(mine[0]["score"], common["score"])

    def test_empty_index_returns_degraded_not_crash(self):
        scope = st.select_scope("anything at all")
        self.assertEqual(scope["paper_ids"], [])
        self.assertEqual(scope["nodes"], [])
        self.assertEqual(scope["degraded"], st.NO_INDEX)

    def test_query_without_usable_terms_is_degraded_not_a_crash(self):
        self._build_all()
        scope = st.select_scope("the of a")
        self.assertEqual(scope["nodes"], [])
        self.assertEqual(scope["degraded"], st.NO_TERMS)

    def test_indexed_but_no_hit_is_not_degraded(self):
        """有索引却一个词都没命中：这是「确实没有」，不能标 degraded 骗调用方去全库重扫。"""
        self._build_all()
        scope = st.select_scope("quantum chromodynamics lattice")
        self.assertEqual(scope["nodes"], [])
        self.assertIsNone(scope["degraded"])
        self.assertGreater(scope["total_nodes"], 0)


# ── ④ 第二级 + 顶层 ──

class TwoStageTests(SectionTreeBase):
    def test_search_in_scope_carries_path_and_page(self):
        self._build_all()
        scope = st.select_scope("channel estimation", top_papers=5, top_nodes=20)
        hits = st.search_in_scope("channel estimation", scope, top_k=5)
        self.assertTrue(hits)
        for h in hits:
            self.assertIn("section_path", h)
            self.assertIsInstance(h["page_no"], int)
            self.assertTrue(h["matched_terms"])
            self.assertTrue(h["text"].strip())

    def test_at_most_one_passage_per_node(self):
        """一节只出最好的一个片段：既防长节刷屏，也保证 stage2_hits <= stage1_nodes。"""
        with db.conn() as c:
            long_body = "\n\n".join(
                [f"Paragraph {i} discusses widget calibration in depth. " * 6
                 for i in range(8)])
            pid = self._paper(c, "long", "Long Section Paper",
                              [_page("Calibration", long_body)])
        st.build_index(pid)
        scope = st.select_scope("widget calibration", top_papers=5, top_nodes=20)
        hits = st.search_in_scope("widget calibration", scope, top_k=10)
        node_ids = [h["node_id"] for h in hits]
        self.assertEqual(len(node_ids), len(set(node_ids)))

    def test_synthesised_parent_nodes_do_not_duplicate_their_children_pages(self):
        """补建的父节点没有自己的正文，第二级必须跳过它。

        不跳过的话它的页区间是空的（lo/hi 都是 NULL），过滤条件形同虚设，它会把
        整篇的页都扫一遍，于是同一页在「父节点」和「子节点」下各出一次片段——
        top_k 的名额被自己人吃掉，而且看上去像两条独立证据。
        """
        st.build_index(self.ids["qasper"])
        scope = st.select_scope("related work static embeddings",
                                top_papers=5, top_nodes=20)
        parents = {n["node_id"] for n in scope["nodes"] if n["start_page"] is None}
        self.assertTrue(parents, "夹具要能选中补建的父节点，否则这条用例是空跑")

        hits = st.search_in_scope("related work static embeddings", scope, top_k=10)
        self.assertTrue(hits)
        self.assertFalse(parents & {h["node_id"] for h in hits})
        pages = [(h["paper_id"], h["page_no"]) for h in hits]
        self.assertEqual(len(pages), len(set(pages)))

    def test_two_stage_not_worse_than_full_search(self):
        """两级检索最大的风险是第一级误杀，这条用例就是那个反例。

        查询词只出现在 deep 那篇的**深层子节正文**里，标题里一个都没有——
        正是第一级最容易漏掉的形态。要求它既进第一级范围，也进最终结果，
        并且不比全库页级检索差。
        """
        self._build_all()
        q = "prosodic boundary tone annotators"
        gold = self.ids["deep"]
        with db.conn() as c:
            full = db.search_pages_ranked(c, q, 5)
        self.assertIn(gold, full, "对照组本身要能召回，否则这条用例证明不了任何事")

        res = st.two_stage_search(q, top_k=5, top_papers=5)
        self.assertEqual(res["mode"], "two_stage")
        self.assertIn(gold, res["scope_paper_ids"])
        self.assertIn(gold, res["paper_ids"])
        self.assertTrue(any(h["paper_id"] == gold for h in res["passages"]))

    def test_trace_numbers_are_consistent(self):
        self._build_all()
        res = st.two_stage_search("channel estimation", top_k=5, top_papers=3)
        t = res["trace"]
        with db.conn() as c:
            total = c.execute("SELECT COUNT(*) FROM section_nodes").fetchone()[0]
            papers = c.execute(
                "SELECT COUNT(DISTINCT paper_id) FROM section_nodes").fetchone()[0]
        self.assertEqual(t["total_nodes"], total)
        self.assertEqual(t["total_papers"], papers)
        self.assertEqual(t["stage1_papers"], len(res["scope_paper_ids"]))
        self.assertGreaterEqual(t["stage1_nodes"], t["stage2_hits"])   # 一节最多出一片段
        self.assertLessEqual(t["stage1_nodes"], t["total_nodes"])      # 第一级确实在筛
        self.assertLessEqual(t["stage1_papers"], 3)
        self.assertEqual(t["stage2_hits"], len(res["passages"]))

    def test_trace_shows_the_scope_actually_shrinks(self):
        self._build_all()
        res = st.two_stage_search("prosodic boundary tone", top_k=5, top_papers=2)
        t = res["trace"]
        self.assertLess(t["stage1_papers"], t["total_papers"])
        self.assertLess(t["stage1_nodes"], t["total_nodes"])

    def test_no_index_falls_back_to_full_library(self):
        """一个节点都没建时不能空手而归，要走全库检索并如实标 mode/degraded。"""
        res = st.two_stage_search("temperature buoy salinity", top_k=5)
        self.assertEqual(res["mode"], "degraded_full")
        self.assertEqual(res["degraded"], st.NO_INDEX)
        self.assertEqual(res["trace"]["stage1_nodes"], 0)
        self.assertIn(self.ids["ocean"], res["paper_ids"])

    def test_completely_empty_db_does_not_crash(self):
        with db.conn() as c:
            c.execute("DELETE FROM pages")
            c.execute("DELETE FROM pages_fts")
            c.execute("DELETE FROM papers")
        res = st.two_stage_search("anything", top_k=5)
        self.assertEqual(res["mode"], "degraded_full")
        self.assertEqual(res["paper_ids"], [])
        self.assertEqual(res["passages"], [])
        self.assertEqual(
            st.build_all(),
            # 新增的两个键是纯增量：build_all 原来把 build_index 的 degraded
            # 整个丢在栈里，cli 打出来是「建索引 45 篇 · 跳过 0 · 失败 0」一片祥和。
            {"indexed": 0, "skipped": 0, "failed": 0, "errors": [],
             "degraded": None, "degraded_detail": []})

    def test_two_runs_give_identical_results(self):
        """跨次确定性：本项目踩过「同一评测集连跑两次两个 Recall」的坑。"""
        self._build_all()
        a = st.two_stage_search("channel estimation large arrays", top_k=5)
        b = st.two_stage_search("channel estimation large arrays", top_k=5)
        self.assertEqual(json.dumps(a, sort_keys=True, ensure_ascii=False),
                         json.dumps(b, sort_keys=True, ensure_ascii=False))

    def test_rebuild_reproduces_identical_index_and_results(self):
        """索引可重建：重建一遍之后，节点 id 与检索结果必须逐字一致。"""
        self._build_all()
        before_nodes = self._nodes(self.ids["qasper"])
        before = st.two_stage_search("static word embeddings", top_k=5)
        self._build_all()
        after_nodes = self._nodes(self.ids["qasper"])
        after = st.two_stage_search("static word embeddings", top_k=5)
        self.assertEqual([(n["node_id"], n["parent_id"], n["section_path"])
                          for n in before_nodes],
                         [(n["node_id"], n["parent_id"], n["section_path"])
                          for n in after_nodes])
        self.assertEqual(json.dumps(before, sort_keys=True, ensure_ascii=False),
                         json.dumps(after, sort_keys=True, ensure_ascii=False))


class StatsTests(SectionTreeBase):
    def test_stats_counts(self):
        self.assertEqual(st.stats()["n_nodes"], 0)
        self._build_all()
        s = st.stats()
        with db.conn() as c:
            n_nodes = c.execute("SELECT COUNT(*) FROM section_nodes").fetchone()[0]
            n_terms = c.execute("SELECT COUNT(*) FROM section_terms").fetchone()[0]
        self.assertEqual(s["n_nodes"], n_nodes)
        self.assertEqual(s["n_terms"], n_terms)
        self.assertEqual(s["papers_indexed"], 5)
        self.assertAlmostEqual(s["avg_nodes_per_paper"], round(n_nodes / 5, 2))


if __name__ == "__main__":
    unittest.main()
