"""多轮对话记忆：会话落库、追问改写、会话内 [n] 编号稳定、回指拉回上一轮来源。

背景：改之前「多轮」是假的——前端把最近 12 条历史发上来，rag.prepare 只取最后一条
user 消息，其余全丢。所以「它和刚才那篇比呢」必然失效。这些用例把新行为钉住。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import chat, config, db, degrade, embeddings, rag


class ChatMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_chat_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 向量按住：否则开发机配了 EMBED_MODEL 时测试会真发请求
        self._patches = [mock.patch("papernest.embeddings.available", return_value=False),
                         mock.patch("papernest.llm.available", return_value=False)]
        for p in self._patches:
            p.start()
        db.init_db()
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "arxiv:7001", "title": "Retrieval Augmented Generation",
                "abstract": "RAG combines retrieval with generation for knowledge tasks.",
                "year": 2023, "venue": "NeurIPS", "authors": ["Ann Ray"],
                "doi": None, "arxiv_id": "7001", "source": "s2"})
            self.p2 = db.insert_l0(c, {
                "norm_key": "arxiv:7002", "title": "Agent Tool Use Benchmark",
                "abstract": "A benchmark for measuring tool use reliability of agents.",
                "year": 2024, "venue": "ICLR", "authors": ["Bo Chen"],
                "doi": None, "arxiv_id": "7002", "source": "s2"})

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 会话存储 ──

    def test_session_roundtrip(self):
        sid = chat.ensure_session(None, "RAG 是什么")
        chat.append(sid, "user", "RAG 是什么")
        chat.append(sid, "assistant", "RAG 是检索增强生成 [1]。",
                    [{"idx": 1, "paper_id": self.p1, "title": "RAG"}])
        turns = chat.history(sid)
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])
        self.assertEqual(turns[1]["sources"][0]["paper_id"], self.p1)
        self.assertEqual(chat.get_session(sid)["title"], "RAG 是什么")
        self.assertEqual(len(chat.list_sessions()), 1)

    def test_unknown_session_id_does_not_raise(self):
        """对话不该因为一个过期 id 就中断——换成新会话继续。"""
        sid = chat.ensure_session("不存在的会话", "问题")
        self.assertTrue(chat.get_session(sid))

    def test_delete_session_cascades_messages(self):
        sid = chat.ensure_session(None, "x")
        chat.append(sid, "user", "x")
        self.assertTrue(chat.delete_session(sid))
        self.assertEqual(chat.history(sid), [])
        self.assertFalse(chat.delete_session(sid))

    def test_history_is_capped_and_chronological(self):
        sid = chat.ensure_session(None, "x")
        for i in range(20):
            chat.append(sid, "user", f"问题{i}")
        turns = chat.history(sid, 5)
        self.assertEqual(len(turns), 5)
        self.assertEqual([t["content"] for t in turns],
                         [f"问题{i}" for i in range(15, 20)])   # 最近 5 条、正序

    # ── 追问识别与改写 ──

    def test_followup_detection(self):
        self.assertFalse(chat.is_followup("RAG 是什么", []))
        self.assertTrue(chat.is_followup("它的局限是什么", ["RAG 是什么"]))
        self.assertTrue(chat.is_followup("为什么？", ["RAG 是什么"]))       # 太短
        self.assertTrue(chat.is_followup("第二篇讲得更细一点", ["RAG 是什么"]))
        self.assertFalse(chat.is_followup("大模型 Agent 的评测方法有哪些", ["RAG 是什么"]))

    def test_rewrite_merges_prior_question(self):
        q, note = chat.rewrite_query("它的局限是什么", ["RAG 检索增强生成"])
        self.assertIn("RAG", q)              # 空查询被补上了检索信号
        self.assertIn("局限", q)
        self.assertIsNotNone(note)

    def test_rewrite_leaves_standalone_question_alone(self):
        q, note = chat.rewrite_query("大模型 Agent 的评测方法有哪些", ["RAG 是什么"])
        self.assertEqual(q, "大模型 Agent 的评测方法有哪些")
        self.assertIsNone(note)

    def test_referenced_indices_parsing(self):
        self.assertEqual(chat.referenced_indices("第 2 篇怎么做的"), [2])
        self.assertEqual(chat.referenced_indices("第二篇怎么做的"), [2])
        self.assertEqual(chat.referenced_indices("[3] 和 [1] 有什么区别"), [3, 1])
        self.assertEqual(chat.referenced_indices("完全没有回指"), [])

    # ── 会话内编号稳定 ──

    def test_indices_are_stable_across_turns(self):
        sid = chat.ensure_session(None, "x")
        first = chat.assign_indices(sid, [self.p1, self.p2])
        self.assertEqual(sorted(first.values()), [1, 2])
        # 第二轮只召回了 p2：它必须还是原来那个号，不能变成 [1]
        second = chat.assign_indices(sid, [self.p2])
        self.assertEqual(second[self.p2], first[self.p2])

    def test_new_paper_gets_next_free_index(self):
        sid = chat.ensure_session(None, "x")
        chat.assign_indices(sid, [self.p1])
        second = chat.assign_indices(sid, [self.p1, self.p2])
        self.assertEqual(second[self.p1], 1)
        self.assertEqual(second[self.p2], 2)

    def test_without_session_numbering_is_positional(self):
        m = chat.assign_indices(None, [self.p2, self.p1])
        self.assertEqual(m, {self.p2: 1, self.p1: 2})

    # ── prepare 端到端 ──

    def _prepare(self, question, sid=None, ids=None):
        with mock.patch("papernest.embeddings.search_hybrid",
                        return_value=embeddings.RetrievalResult(
                            ids if ids is not None else [self.p1, self.p2],
                            "fts", [])):
            return rag.prepare([{"role": "user", "content": question}],
                               top_k=5, session_id=sid)

    def test_prepare_persists_nothing_by_itself(self):
        """prepare 只读不写：落库发生在 answer/流式结束时，避免半截轮次污染历史。"""
        sid = chat.ensure_session(None, "x")
        self._prepare("RAG 是什么", sid)
        self.assertEqual(chat.history(sid), [])

    def test_prepare_includes_history_block(self):
        sid = chat.ensure_session(None, "RAG 是什么")
        chat.append(sid, "user", "RAG 是什么")
        chat.append(sid, "assistant", "检索增强生成 [1]。",
                    [{"idx": 1, "paper_id": self.p1, "title": "RAG"}])
        sys, q, _sources, _deg, _n = self._prepare("它的局限是什么", sid)
        self.assertIn("之前的对话", sys)
        self.assertIn("RAG 是什么", sys)
        self.assertEqual(q, "它的局限是什么")      # 送给模型的仍是用户原话

    def test_followup_pulls_back_referenced_source(self):
        """「第 2 篇…」即使本轮没召回那篇，也要被拉回上下文——否则模型只能瞎编。"""
        sid = chat.ensure_session(None, "x")
        chat.append(sid, "user", "有哪些相关文献")
        chat.append(sid, "assistant", "见 [1][2]。",
                    [{"idx": 1, "paper_id": self.p1, "title": "RAG"},
                     {"idx": 2, "paper_id": self.p2, "title": "Bench"}])
        chat.assign_indices(sid, [self.p1, self.p2])
        # 本轮检索只召回 p1，但用户点名要第 2 篇
        _sys, _q, sources, _deg, _n = self._prepare("第 2 篇怎么做的", sid, ids=[self.p1])
        self.assertIn(self.p2, [s["paper_id"] for s in sources])
        self.assertEqual(next(s["idx"] for s in sources if s["paper_id"] == self.p2), 2)

    def test_reference_survives_an_intervening_turn(self):
        """第 3 轮的「第 2 篇」也要指到会话里的那个 [2]，哪怕第 2 轮换了话题。

        编号是**会话全局**的（`chat_sessions.source_map_json`），而回指原来只在
        `last_sources()`——最近一条有来源的 assistant 消息——里反查。中间隔一轮
        换话题，[2] 就不在那份 sources 里，回指静默失效。
        上面那条 `test_followup_pulls_back_referenced_source` 只覆盖第 2 轮
        （那一轮两者恰好相等），正是这个缺陷的盲区。
        """
        with db.conn() as c:
            p3 = db.insert_l0(c, {
                "norm_key": "arxiv:7003", "title": "Graph Neural Network Survey",
                "abstract": "A survey of graph neural networks.", "year": 2022,
                "venue": "TPAMI", "authors": ["Cui Li"], "doi": None,
                "arxiv_id": "7003", "source": "s2"})
        sid = chat.ensure_session(None, "x")

        n1 = chat.assign_indices(sid, [self.p1, self.p2])       # [1] [2]
        chat.append(sid, "user", "有哪些相关文献")
        chat.append(sid, "assistant", "见 [1][2]。",
                    [{"idx": n1[self.p1], "paper_id": self.p1, "title": "RAG"},
                     {"idx": n1[self.p2], "paper_id": self.p2, "title": "Bench"}])

        # 第 2 轮换话题，这一轮的 sources 里没有 [2]
        n2 = chat.assign_indices(sid, [p3])                      # [3]
        chat.append(sid, "user", "换个话题，图神经网络怎么做的")
        chat.append(sid, "assistant", "见 [3]。",
                    [{"idx": n2[p3], "paper_id": p3, "title": "GNN"}])

        # 第 3 轮回指第 1 轮的 [2]，而本轮检索只召回 p3
        _sys, _q, sources, _deg, _n = self._prepare("第 2 篇怎么做的", sid, ids=[p3])
        got = [s["paper_id"] for s in sources]
        self.assertIn(self.p2, got, f"「第 2 篇」没有被拉回上下文：sources={got}")
        self.assertEqual(
            next(s["idx"] for s in sources if s["paper_id"] == self.p2), 2)

    def test_pins_never_evict_the_whole_turn_retrieval(self):
        """回指 pin 不许把本轮检索整体挤掉，篇数也不许由用户输入决定。

        `merged[:max(top_k, len(pinned))]` 里 pin 是无条件插队的，而 `len(pinned)`
        由问句里出现多少个 [n] 决定。会话全局反查一上线，
        「[1][2][3][4][5] 和 X 比起来怎么样」就会把为 X 检索到的论文一篇不剩地
        挤出上下文，粘一段带 [1]..[99] 的相关工作还会把篇数和 token 撑到预算上限——
        两种都完全静默。所以 pin 要封顶，并且至少给本轮检索留一个名额。
        """
        extra = []
        with db.conn() as c:
            for k in range(6):
                extra.append(db.insert_l0(c, {
                    "norm_key": f"arxiv:81{k:02d}", "title": f"Pinned Paper {k}",
                    "abstract": f"Abstract of pinned paper {k}.", "year": 2020 + k,
                    "venue": "X", "authors": ["A B"], "doi": None,
                    "arxiv_id": f"81{k:02d}", "source": "s2"}))
        sid = chat.ensure_session(None, "x")
        chat.assign_indices(sid, extra)                          # [1]..[6]
        fresh = self.p1
        _sys, _q, sources, _deg, notes = self._prepare(
            "[1] [2] [3] [4] [5] 和 RAG 检索增强生成比起来怎么样", sid, ids=[fresh])
        got = [s["paper_id"] for s in sources]
        self.assertLessEqual(len(sources), 5,
                             f"进上下文的篇数被问句里的 [n] 个数撑大了：{len(sources)} 篇")
        self.assertIn(fresh, got, f"本轮检索到的论文被 pin 整体挤掉了：sources={got}")
        self.assertIn(degrade.REF_PIN_TRUNCATED, [n.code for n in notes],
                      "回指被截断却没上报——静默失败正是这条修复要消灭的形状")

    def test_unresolvable_reference_is_reported_not_silent(self):
        """用户点名的编号不在会话编号表里 → 必须留痕；全新会话不报（那是引文角标）。"""
        sid = chat.ensure_session(None, "x")
        chat.assign_indices(sid, [self.p1, self.p2])             # [1] [2]
        _s, _q, _src, _d, notes = self._prepare("第 9 篇讲了什么", sid, ids=[self.p1])
        self.assertIn(degrade.REF_INDEX_UNRESOLVED, [n.code for n in notes])

        fresh_sid = chat.ensure_session(None, "y")               # 一个编号都还没有
        _s, _q, _src, _d, notes2 = self._prepare(
            "论文里的 [12] 指的是什么", fresh_sid, ids=[self.p1])
        self.assertNotIn(degrade.REF_INDEX_UNRESOLVED, [n.code for n in notes2],
                         "全新会话里的 [12] 是正文引文角标，不该报回指失败")

    def test_client_history_used_when_no_session(self):
        """没有会话时退回用前端发来的历史，老前端不受影响。"""
        with mock.patch("papernest.embeddings.search_hybrid",
                        return_value=embeddings.RetrievalResult([self.p1], "fts", [])):
            sys, _q, _s, _d, _n = rag.prepare(
                [{"role": "user", "content": "RAG 是什么"},
                 {"role": "assistant", "content": "检索增强生成。"},
                 {"role": "user", "content": "它的局限是什么"}], top_k=5)
        self.assertIn("之前的对话", sys)
        self.assertIn("RAG 是什么", sys)

    def test_degraded_reports_rewrite(self):
        sid = chat.ensure_session(None, "x")
        chat.append(sid, "user", "RAG 检索增强生成")
        _sys, _q, _sources, degraded, _n = self._prepare("它的局限是什么", sid)
        self.assertIsNotNone(degraded)
        self.assertIn("追问改写", degraded)


if __name__ == "__main__":
    unittest.main()
