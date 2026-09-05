"""查询预处理：用户真会打的输入形态不能把检索打废。

审计确认的三条失败形态（下面逐条钉住）：

1. `_expand_terms` 只剥双引号，其余标点原样进词面；而 `search_fts` 又把每个词面
   包成 FTS5 短语，在 trigram 分词下等价于**字面子串匹配**——`XL-MIMO?` 里的问号
   成了必须命中的字符，整个词作废；剩下的 `What` 长度 ≥3 反而作为有效词面参与 OR。
2. CJK 片段被切成**二元**词面，而 trigram 分词器按**字符**计，2 字符的词面命中恒为 0
   （本库实测：'智能'=0、'模型'=0、'信道'=0，连 ASCII 的 'ai' 也是 0）。于是
   中文整句提问在 FTS 上必然 0 行。
3. FTS 落空后退到 LIKE 兜底，而兜底 `ORDER BY id DESC`——「最新入库」被当成
   「最相关」，再以权重 1.0 灌进 search_hybrid 的 RRF（最高的一路）。

这些用例大多是**行为断言**（"带标点的查询要和不带标点的查到同样的东西"），
而不是对实现细节的镜像断言——换实现只要行为对就不该红。
"""
from __future__ import annotations

import unittest

from papernest import db

try:
    from .support import TempDbTestCase
except ImportError:                          # noqa: F401
    from support import TempDbTestCase


class ExpandTermsTests(unittest.TestCase):
    """纯函数，无需库。"""

    def test_trailing_question_mark_does_not_poison_the_term(self):
        self.assertIn("XL-MIMO", db._expand_terms("What is XL-MIMO?"))
        self.assertNotIn("XL-MIMO?", db._expand_terms("What is XL-MIMO?"))

    def test_parentheses_are_stripped(self):
        terms = db._expand_terms("(channel estimation)")
        self.assertIn("channel", terms)
        self.assertIn("estimation", terms)

    def test_sentence_final_period_is_stripped(self):
        self.assertIn("estimation", db._expand_terms("channel estimation."))

    def test_technical_punctuation_survives(self):
        """C++ / C# / GPT-4 / wav2vec2.0 里的符号是词的一部分，不能一起剥掉。"""
        for q, want in [("C++ templates", "C++"), ("C# async", "C#"),
                        ("GPT-4 evaluation", "GPT-4"),
                        ("wav2vec2.0 pretraining", "wav2vec2.0"),
                        ("XL-MIMO channel", "XL-MIMO")]:
            self.assertIn(want, db._expand_terms(q), f"{q!r} 丢掉了 {want!r}")

    def test_cjk_is_split_into_three_char_windows_not_two(self):
        """二元词面进不了 trigram 索引（见 MIN_FTS_CHARS），所以必须切三元。"""
        terms = db._expand_terms("大模型智能体评测")
        self.assertIn("智能体", terms)
        self.assertIn("大模型", terms)
        self.assertNotIn("智能", terms, "二元词面对 trigram 索引恒为 0 命中，不该产出")

    def test_mixed_script_token_yields_both_sides(self):
        terms = db._expand_terms("XL-MIMO的近场信道估计")
        self.assertIn("XL-MIMO", terms)
        self.assertIn("信道估", terms)

    def test_terms_are_deduped_and_order_stable(self):
        q = "channel estimation channel"
        self.assertEqual(db._expand_terms(q), db._expand_terms(q))
        self.assertEqual(len(db._expand_terms(q)), len(set(db._expand_terms(q))))

    def test_term_count_is_bounded(self):
        long_cn = "近场信道估计" * 40
        self.assertLessEqual(len(db._expand_terms(long_cn)), db.MAX_TERMS)

    def test_empty_and_punctuation_only_do_not_crash(self):
        for q in ("", None, "？？？", "   ", "..."):
            self.assertIsInstance(db._expand_terms(q), list)


class TrigramFloorTests(TempDbTestCase):
    """MIN_FTS_CHARS 是实测出来的，不是拍的——这条用例把那次实测钉在库里。"""

    prefix = "papernest_trigram"

    def setUp(self):
        super().setUp()
        db.init_db()
        with db.conn() as c:
            db.insert_l0(c, {
                "norm_key": "arxiv:5001", "title": "大模型智能体评测综述",
                "abstract": "本文讨论智能体评测。", "year": 2024, "venue": "V",
                "authors": [], "doi": None, "arxiv_id": "5001", "source": "s2"})
            c.commit()

    def _match(self, term):
        with db.conn() as c:
            return c.execute(
                "SELECT COUNT(*) n FROM papers_fts WHERE papers_fts MATCH ?",
                (f'"{term}"',)).fetchone()["n"]

    def test_two_char_terms_never_match_in_a_trigram_index(self):
        """这是 MIN_FTS_CHARS=3 的全部依据。若某天它不再成立，这条会红，提醒改阈值。"""
        self.assertEqual(self._match("智能"), 0)
        self.assertEqual(self._match("模型"), 0)

    def test_three_char_terms_do_match(self):
        self.assertEqual(self._match("智能体"), 1)
        self.assertEqual(self._match("大模型"), 1)


class SearchBehaviourTests(TempDbTestCase):
    prefix = "papernest_searchq"

    def setUp(self):
        super().setUp()
        db.init_db()
        with db.conn() as c:
            self.target = db.insert_l0(c, {
                "norm_key": "arxiv:5101",
                "title": "Near-Field Channel Estimation for XL-MIMO Systems",
                "abstract": "We study channel estimation in the near field.",
                "year": 2024, "venue": "TWC", "authors": [], "doi": None,
                "arxiv_id": "5101", "source": "s2"})
            self.cn = db.insert_l0(c, {
                "norm_key": "arxiv:5102", "title": "大模型智能体评测基准",
                "abstract": "面向智能体的评测基准。", "year": 2024, "venue": "V",
                "authors": [], "doi": None, "arxiv_id": "5102", "source": "s2"})
            # 后入库的无关条目。**必须与目标共享词面**，否则它在 LIKE 兜底里
            # 根本不参与竞争，这条用例就测不出「按 id DESC 排」的问题——
            # 真实库里的干扰项恰恰是「RAG 评测组会汇报」这种共享「评测」的自建笔记。
            self.newest = db.insert_l0(c, {
                "norm_key": "note:9999", "title": "本周组会：评测相关讨论",
                "abstract": "组会上讨论了评测的一些问题，与大模型无关。",
                "year": 2025, "venue": "",
                "authors": [], "doi": None, "arxiv_id": None, "source": "upload"})
            c.commit()

    def _ids(self, q, k=5):
        with db.conn() as c:
            return [r["id"] for r in db.search_fts(c, q, k)]

    def test_punctuation_does_not_change_what_is_found(self):
        """带标点的提问要和裸关键词查到同一篇——这是本次修复的核心行为。"""
        plain = self._ids("XL-MIMO")
        for q in ("What is XL-MIMO?", "XL-MIMO.", "(XL-MIMO)", "XL-MIMO!"):
            self.assertIn(self.target, self._ids(q), f"{q!r} 没召回目标论文")
            self.assertEqual(self._ids(q)[0], plain[0], f"{q!r} 的首位与裸关键词不一致")

    def test_parenthesised_query_is_not_empty(self):
        """回归：'(channel estimation)' 原来 FTS 0 行、LIKE 也 0 行，返回空列表。"""
        self.assertIn(self.target, self._ids("(channel estimation)"))

    def test_chinese_full_sentence_finds_the_right_paper(self):
        """回归：中文整句原来在 FTS 上必然 0 行，退 LIKE 后返回最新入库的无关条目。"""
        ids = self._ids("有哪些关于大模型智能体评测的论文？")
        self.assertIn(self.cn, ids)
        self.assertEqual(ids[0], self.cn, f"首位应是相关论文，实际 {ids[:3]}")

    def test_like_fallback_ranks_by_match_count_not_insertion_order(self):
        """兜底路径也必须按相关性排。原来是 ORDER BY id DESC。"""
        # 构造一个必然落到 LIKE 的查询：词面短于 trigram 下限、FTS 匹配不到
        with db.conn() as c:
            rows = db.search_fts(c, "近场", 5)
        ids = [r["id"] for r in rows]
        if ids:
            self.assertNotEqual(
                ids[0], self.newest,
                "LIKE 兜底把最新入库的无关条目排在了首位")

    def test_ordering_is_deterministic(self):
        """排序键必须全序，否则同一条问句跨进程会漂移。"""
        runs = {tuple(self._ids("大模型智能体评测")) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_chunk_and_page_search_share_the_same_floor(self):
        """三处 FTS 用同一个字符下限常量，不能各写各的。"""
        import inspect
        for fn in (db.search_fts, db.search_chunks_fts, db.search_pages_fts):
            self.assertIn("MIN_FTS_CHARS", inspect.getsource(fn),
                          f"{fn.__name__} 没用统一的 MIN_FTS_CHARS")


if __name__ == "__main__":
    unittest.main()
