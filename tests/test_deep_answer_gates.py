# -*- coding: utf-8 -*-
"""迭代问答的四道闸门：补检索交集、篇数预算、自评三态、无证据论断上产物。

背景（这些都是改之前真实存在的行为）：
- 补充查询来自「上一轮没有证据支撑因而可能是编造的」论断句，召回的论文无过滤地进
  上下文，还和首轮 gold 论文完全等权——下一轮模型会把它们当成对该论断的「佐证」。
  项目在检索层已经实测过派生查询的 query drift（独有候选里 gold 只占 0.9%）并限制成
  「只补位」，生成层却没跟上。
- `top_k=len(seen)` 把 rag.prepare 里唯一的截断闸门恒等化，而 max_rounds 无上界。
- CRITIC_SYSTEM 约定「证据够了给空数组」，但它和 except 分支返回值相同，于是
  docstring 承诺的收敛条件不可达，trace 里两个相反状态记成同一条。
- 命中轮数上限时，明知哪几句没有证据，仍照原样混在正文里交付。
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, deepsearch, embeddings, llm

QUESTION = "近场信道估计有哪些主流方法"
# 长度过 15 字且不含 [n]，会被 eval.split_claims 判为「无证据论断」
UNSUPPORTED_ANSWER = "近场信道估计在超大孔径阵列下会出现球面波前效应从而使远场假设失效。"


class DeepAnswerGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_deepgate_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._patches = [mock.patch("papernest.embeddings.available", return_value=False),
                         mock.patch("papernest.llm.available", return_value=True)]
        for p in self._patches:
            p.start()
        db.init_db()
        self.pids = []
        with db.conn() as c:
            for i in range(25):
                self.pids.append(db.insert_l0(c, {
                    "norm_key": f"arxiv:80{i:02d}", "title": f"Channel Estimation Study {i}",
                    "abstract": f"Abstract number {i} about arrays.",
                    "year": 2020 + i % 5, "venue": "TWC", "authors": ["X Y"],
                    "doi": None, "arxiv_id": f"80{i:02d}", "source": "s2"}))
        self.base = self.pids[:15]        # 原问题的宽召回 = 闸门集合
        self.outside = self.pids[20:]     # 闸门集合之外
        self._critic_reply = json.dumps({"gaps": ["缺"], "queries": ["CRITIC_IN_1"]})
        self._critic_exc = None

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 测试替身 ──

    def _search(self, query, k=8, *a, **kw):
        if query.startswith("CRITIC_OUT"):
            return embeddings.RetrievalResult(list(self.outside), "fts", [])
        # CRITIC_IN 与 expand_queries 的派生词都落回原问题的宽召回
        return embeddings.RetrievalResult(list(self.base), "fts", [])

    def _chat(self, system, user, purpose="", **kw):
        if purpose == "deep_critic":
            if self._critic_exc:
                raise self._critic_exc
            return self._critic_reply
        return UNSUPPORTED_ANSWER

    def _run(self, **kw):
        with mock.patch("papernest.embeddings.search_hybrid", side_effect=self._search), \
             mock.patch("papernest.llm.chat", side_effect=self._chat):
            return deepsearch.deep_answer(QUESTION, **kw)

    # ── 自评三态 ──

    def test_empty_queries_means_evidence_is_enough_not_a_failure(self):
        """CRITIC_SYSTEM 说「够了就给空数组」。原来这条收敛路径在代码里不可达。"""
        self._critic_reply = json.dumps({"gaps": [], "queries": []})
        r = self._run(top_k=5, max_rounds=3)
        self.assertEqual(r["rounds"], 1)
        self.assertEqual(r["trace"]["iterations"][0]["stop"], "模型判定证据已足")

    def test_critic_failure_is_recorded_separately_from_enough(self):
        """自评调用炸了和「证据已足」原来记成同一条，事后不可区分。"""
        self._critic_exc = llm.LLMError("boom")
        r = self._run(top_k=5, max_rounds=2)
        first = r["trace"]["iterations"][0]
        self.assertIn("自评调用失败", first.get("queries_from", ""))
        self.assertNotEqual(first.get("stop"), "模型判定证据已足")

    # ── 补检索交集闸门 ──

    def test_supplementary_hits_outside_the_base_recall_are_gated_out(self):
        """闸门的正题：派生召回里原问题够不着的那些，不准进上下文。"""
        self._critic_reply = json.dumps({"gaps": ["缺"], "queries": ["CRITIC_OUT_1"]})
        r = self._run(top_k=5, max_rounds=3)
        first = r["trace"]["iterations"][0]
        self.assertEqual(first["gated_out"], len(self.outside))
        self.assertEqual(first["new_papers"], 0)
        self.assertEqual(first["stop"], "补检索没有带来新文献")
        self.assertEqual(r["trace"]["iterations"][-1]["papers_in_context"], 5)

    def test_supplementary_hits_inside_the_base_recall_still_get_through(self):
        """闸门不能把补检索堵死——原问题自己够得着的，照常进上下文。"""
        r = self._run(top_k=5, max_rounds=2)
        first = r["trace"]["iterations"][0]
        self.assertEqual(first["gated_out"], 0)
        self.assertGreater(first["new_papers"], 0)
        self.assertGreater(r["trace"]["iterations"][1]["papers_in_context"], 5)

    def test_new_papers_counts_what_actually_entered_the_context(self):
        """原来记的是裁剪前的数量，trace 会比实际入库的多。"""
        r = self._run(top_k=5, max_rounds=2)
        it = r["trace"]["iterations"]
        entered = it[1]["papers_in_context"] - it[0]["papers_in_context"]
        self.assertEqual(entered, min(it[0]["new_papers"], 5))

    # ── 篇数预算 ──

    def test_context_papers_never_exceed_the_budget(self):
        r = self._run(top_k=5, max_rounds=5)
        sizes = [it["papers_in_context"] for it in r["trace"]["iterations"]]
        self.assertLessEqual(max(sizes), deepsearch.MAX_CONTEXT_PAPERS, sizes)
        self.assertTrue(any(it.get("dropped_for_budget") for it in r["trace"]["iterations"]),
                        "预算闸门没被触发，这条用例就没有在测它想测的东西")

    def test_budget_holds_even_with_an_absurd_round_count(self):
        """max_rounds 从 API/CLI 无上界地传进来时，上下文也不该跟着无上界地涨。"""
        r = self._run(top_k=5, max_rounds=20)
        for it in r["trace"]["iterations"]:
            self.assertLessEqual(it["papers_in_context"], deepsearch.MAX_CONTEXT_PAPERS)

    # ── 无证据论断上产物 ──

    def test_unsupported_claims_are_annotated_on_the_answer(self):
        """明知哪几句没有证据，就不能让它们和有 [n] 支撑的句子形态相同地交付。"""
        r = self._run(top_k=5, max_rounds=1)
        self.assertTrue(r["unsupported"])
        self.assertIn("未找到支撑", r["answer"])
        self.assertIn(r["unsupported"][0][:20], r["answer"])

    def test_answer_is_annotated_not_truncated(self):
        """标注不删改：原文必须原样保留在答案里。"""
        r = self._run(top_k=5, max_rounds=1)
        self.assertTrue(r["answer"].startswith(UNSUPPORTED_ANSWER))

    def test_clean_answer_gets_no_annotation(self):
        cited = "近场信道估计在超大孔径阵列下会出现球面波前效应从而使远场假设失效[1]。"
        with mock.patch("papernest.embeddings.search_hybrid", side_effect=self._search), \
             mock.patch("papernest.llm.chat",
                        side_effect=lambda s, u, purpose="", **kw: cited):
            r = deepsearch.deep_answer(QUESTION, top_k=5, max_rounds=2)
        self.assertEqual(r["unsupported"], [])
        self.assertNotIn("未找到支撑", r["answer"])
        self.assertEqual(r["trace"]["iterations"][0]["stop"], "无证据论断已清零")


if __name__ == "__main__":
    unittest.main()
