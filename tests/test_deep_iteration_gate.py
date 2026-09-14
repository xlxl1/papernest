# -*- coding: utf-8 -*-
"""补检索闸门的准入判据：挡「原问题够不着的」，不是挡「首轮上下文里没有的」。

原来的闸门是 `kept = [p for p in added if p in base_set]`，而
`base_set = rtrace["primary_ids"]` 正是 `deep_retrieve` 里 `pool = max(top_k, 15)`
那一次召回——也就是首轮上下文自己的来源池。`added` 又已经排除了 `seen`，
于是补检索结构上无法引入首轮池之外的任何文献（真库 82 条评测问题实测 100% 被挡）；
`top_k >= 15` 时 `base_set == seen`，`kept` 成为空集，退化到恒 1 轮 + 恒白付一次
`deep_critic`。

口径提醒（别在材料里写过头）：这是一条**功能与 docstring 不符**的正确性修复，
不是检索质量收益——修完 82 条题的 gold_in_context 净 +1（102 篇里），
远低于本仓的显著性门槛，而 LLM 调用多约 50%。
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, deepsearch, embeddings

QUESTION = "近场信道估计有哪些主流方法"
UNSUPPORTED_ANSWER = "近场信道估计在超大孔径阵列下会出现球面波前效应从而使远场假设失效。"


class DeepIterationGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_iter_")
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
                    "norm_key": f"arxiv:81{i:02d}", "title": f"Channel Estimation Study {i}",
                    "abstract": f"Abstract number {i} about arrays.",
                    "year": 2020 + i % 5, "venue": "TWC", "authors": ["X Y"],
                    "doi": None, "arxiv_id": f"81{i:02d}", "source": "s2"}))
        # 原问题 depth-15 就够得着的（= 首轮上下文的来源池，deep_retrieve 的 pool）
        self.base = self.pids[:15]
        # 要更深的召回池才够得着的：真实场景里排在第 16~50 名的那些论文
        self.wide_only = self.pids[15:20]
        # 任何深度都够不着的：模型自由发挥的产物，必须继续挡
        self.unreachable = self.pids[20:]
        self._critic_reply = json.dumps({"gaps": ["缺"], "queries": ["CRITIC_WIDE_1"]})

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _search(self, query, k=8, *a, **kw):
        """真库上的形态：原问题召得越深，够得着的论文越多。"""
        if query == QUESTION:
            ids = list(self.base) + (list(self.wide_only) if k > 15 else [])
            return embeddings.RetrievalResult(ids, "fts", [])
        if query.startswith("CRITIC_WIDE"):      # 补充查询把「排 16~50 名」的顶上来
            return embeddings.RetrievalResult(list(self.wide_only), "fts", [])
        if query.startswith("CRITIC_OUT"):       # 补充查询把不相关的顶上来
            return embeddings.RetrievalResult(list(self.unreachable), "fts", [])
        return embeddings.RetrievalResult(list(self.base), "fts", [])

    def _chat(self, system, user, purpose="", **kw):
        if purpose == "deep_critic":
            return self._critic_reply
        return UNSUPPORTED_ANSWER

    def _run(self, **kw):
        with mock.patch("papernest.embeddings.search_hybrid", side_effect=self._search), \
             mock.patch("papernest.llm.chat", side_effect=self._chat):
            return deepsearch.deep_answer(QUESTION, **kw)

    # ── 红：闸门取错了集合 ──

    def test_iteration_can_still_add_papers_when_top_k_reaches_the_recall_pool(self):
        """top_k>=15 时闸门集合与首轮上下文恒等，补检索的产出恒为空集。"""
        r = self._run(top_k=15, max_rounds=3)
        first = r["trace"]["iterations"][0]
        self.assertGreater(first["new_papers"], 0, f"补检索一篇都没进来：{first}")
        self.assertGreater(r["rounds"], 1, "迭代退化成 1 轮，critic 调用白付")

    def test_admitted_paper_actually_reaches_the_model_context(self):
        """放行还不够——它得真的出现在下一轮喂给模型的上下文里。

        `seen[:MAX_CONTEXT_PAPERS]` 砍掉的正是刚放行的新文献（首轮召回天然在前），
        所以只修闸门而不修淘汰方向，等于修了个寂寞。
        """
        r = self._run(top_k=15, max_rounds=2)
        got = [s["paper_id"] for s in r["sources"]]
        self.assertTrue(set(got) & set(self.wide_only), f"放行的论文没进上下文：sources={got}")

    def test_gate_pool_is_always_deeper_than_the_first_round_pool(self):
        """闸门池必须比首轮召回池深——否则 top_k >= GATE_POOL 时闸门又变回恒空集。

        `cli.py` 的 `--top-k` 没有上界（只有 API 的 DeepBody 夹了 le=20），
        `ask --deep --top-k 60` 会直接踩到写死 50 的那一版。
        """
        depths = []

        def spy(query, k=8, *a, **kw):
            if query == QUESTION:
                depths.append(k)
            return self._search(query, k, *a, **kw)

        with mock.patch("papernest.embeddings.search_hybrid", side_effect=spy), \
             mock.patch("papernest.llm.chat", side_effect=self._chat):
            deepsearch.deep_answer(QUESTION, top_k=60, max_rounds=2)
        first_round_pool = max(60, 15)          # deep_retrieve 里的 pool
        self.assertTrue(depths, "原问题一次都没被检索")
        self.assertGreater(max(depths), first_round_pool,
                           f"闸门池 {max(depths)} 没有超过首轮池 {first_round_pool}，"
                           f"闸门集合又成了首轮上下文的子集")

    # ── 绿：闸门的正题不能被改坏 ──

    def test_papers_the_question_cannot_reach_at_any_depth_are_still_gated_out(self):
        """补充查询来自可能是编造的论断句，原问题够不着的候选一律不准进上下文。"""
        self._critic_reply = json.dumps({"gaps": ["缺"], "queries": ["CRITIC_OUT_1"]})
        r = self._run(top_k=15, max_rounds=3)
        first = r["trace"]["iterations"][0]
        self.assertEqual(first["new_papers"], 0)
        self.assertEqual(first["gated_out"], len(self.unreachable))
        self.assertEqual(first["stop"], "补检索没有带来新文献")
        self.assertFalse(set(s["paper_id"] for s in r["sources"]) & set(self.unreachable))

    def test_new_papers_per_round_are_bounded(self):
        """放行不是放开：单轮新增有上限，一次补检索灌不满上下文。"""
        r = self._run(top_k=15, max_rounds=3)
        for it in r["trace"]["iterations"]:
            self.assertLessEqual(it["new_papers"], deepsearch.MAX_NEW_PER_ROUND, it)
        self.assertLessEqual(len(r["sources"]), deepsearch.MAX_CONTEXT_PAPERS)


if __name__ == "__main__":
    unittest.main()
