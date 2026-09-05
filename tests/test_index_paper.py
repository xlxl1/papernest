"""`embeddings.index_paper` 的删除边界：它只能动自己写的那两种 kind。

**为什么单开一个文件**：`index_paper` 此前在 tests/ 里一条用例都没有（grep 无结果），
而它的清理语句是 `DELETE FROM vectors WHERE paper_id=? AND model=?`——按 (论文, 模型)
一刀切，把 `chunkembed` 建的 chunk 向量一起删掉，然后只补回 paper + sent。

这条 bug 在真实库上的杀伤力来自选片条件的配合：`cli.py embed`（cli.py:143）与
`jobs.py:181` 挑的都是「没有 kind='paper' 向量的论文」，而**正文先入库、paper 级
向量还没补**的论文必然全部落进这个集合——它们恰好就是 chunk 向量最多的那批。
审计当时的真实库：784 条 chunk 向量里 715 条（91%）属于这类论文，
一次 `cli.py embed` 就会全部消失，且没有任何报错、日志或降级提示。

所以这里断言的不是「函数返回值对」，而是**它没有碰不该碰的数据**。
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from papernest import config, db, embeddings

try:                                        # 与 test_milvusstore 同一套：tests/ 不是包，
    from .support import TempDbTestCase     # discover 与直接运行两种方式都要能导入
except ImportError:                         # noqa: F401
    from support import TempDbTestCase

MODEL = "test-embed"
DIM = 4


def _blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


class IndexPaperDeleteScopeTests(TempDbTestCase):
    prefix = "papernest_index_paper"

    def setUp(self):
        super().setUp()
        self._old_model = config.EMBED_MODEL
        config.EMBED_MODEL = MODEL
        self.addCleanup(self._restore_model)
        embeddings.invalidate_cache()
        self.addCleanup(embeddings.invalidate_cache)
        db.init_db()

        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:9001", "title": "Attention Is All You Need",
                "abstract": "We propose a new architecture. It relies on attention.",
                "year": 2017, "venue": "NeurIPS", "authors": [], "doi": None,
                "arxiv_id": "9001", "source": "s2"})
            # 另一篇：用来确认删除没有越过 paper_id 边界
            self.other = db.insert_l0(c, {
                "norm_key": "arxiv:9002", "title": "Other", "abstract": "b",
                "year": 2018, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "9002", "source": "s2"})
            for pid in (self.pid, self.other):
                for n in range(3):
                    c.execute(
                        "INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                        "VALUES(?,?,?,?,?,?)",
                        (pid, "chunk", n, f"chunk-{pid}-{n}", MODEL,
                         _blob([n, 0, 0, 1])))
            # 换模型的旧向量：model 维度的隔离必须继续成立
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pid, "chunk", 0, "old-model-chunk", "legacy-embed",
                       _blob([9, 9, 9, 9])))
            c.commit()

    def _restore_model(self):
        config.EMBED_MODEL = self._old_model

    def _rows(self, kind, model=MODEL, paper_id=None):
        with db.conn() as c:
            return c.execute(
                "SELECT idx, text FROM vectors WHERE paper_id=? AND kind=? AND model=? "
                "ORDER BY idx", (paper_id or self.pid, kind, model)).fetchall()

    def _run_index(self):
        """跑 index_paper，embedding 侧给确定性假向量（不发网络请求）。"""
        def fake_embed(texts, purpose="embed"):
            return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

        with mock.patch.object(embeddings, "embed_texts", side_effect=fake_embed):
            with db.conn() as c:
                r = c.execute("SELECT title, abstract FROM papers WHERE id=?",
                              (self.pid,)).fetchone()
            return embeddings.index_paper(self.pid, r["title"], r["abstract"])

    # ── 核心回归：补建 paper 向量不得动 chunk 向量 ──

    def test_rebuilding_paper_vectors_keeps_chunk_vectors(self):
        before = [(r["idx"], r["text"]) for r in self._rows("chunk")]
        self.assertEqual(len(before), 3, "前置条件：这篇应有 3 条 chunk 向量")

        self._run_index()

        after = [(r["idx"], r["text"]) for r in self._rows("chunk")]
        self.assertEqual(
            after, before,
            "index_paper 删掉了 chunk 向量——它只写 paper/sent，就不能删别人的。"
            "真实库上这一删会抹掉 91% 的 chunk 向量且不报错")

    def test_chunk_vectors_of_other_papers_untouched(self):
        self._run_index()
        self.assertEqual(len(self._rows("chunk", paper_id=self.other)), 3,
                         "删除越过了 paper_id 边界")

    def test_other_model_vectors_untouched(self):
        self._run_index()
        self.assertEqual(len(self._rows("chunk", model="legacy-embed")), 1,
                         "删除越过了 model 边界，换嵌入模型时会误删旧空间的向量")

    # ── 同时确认：它该重建的那两种 kind 仍然是「先删后插」，不留旧影子 ──

    def test_paper_and_sent_vectors_are_replaced_not_appended(self):
        self._run_index()
        self._run_index()          # 连跑两次，重复执行不该累积

        self.assertEqual(len(self._rows("paper")), 1,
                         "paper 级向量应当被替换而不是追加")
        sents = self._rows("sent")
        self.assertEqual([r["idx"] for r in sents], list(range(len(sents))),
                         "sent 向量的 idx 应当从 0 连续重排，不能留旧编号")

    def test_stale_sent_vectors_from_longer_abstract_are_dropped(self):
        """摘要改短后，多出来的 sent 向量必须消失——它们指向已不存在的句子。"""
        with db.conn() as c:
            for n in range(5, 9):
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,?,?,?,?,?)",
                          (self.pid, "sent", n, f"stale-{n}", MODEL, _blob([1, 1, 1, 1])))
            c.commit()

        self._run_index()

        idxs = [r["idx"] for r in self._rows("sent")]
        self.assertNotIn(8, idxs, "旧的 sent 向量没有被清掉")
        self.assertEqual(len(self._rows("chunk")), 3, "清 sent 时又误伤了 chunk")

    # ── 复现真实事故路径：cli.py embed 的选片条件 + index_paper ──

    def test_cli_embed_selection_then_index_preserves_chunks(self):
        """`cli.py embed` 挑「没有 paper 向量的论文」，正是 chunk 向量最多的那批。"""
        with db.conn() as c:
            rows = c.execute(
                """SELECT id, title, abstract FROM papers
                   WHERE id NOT IN (SELECT DISTINCT paper_id FROM vectors
                                    WHERE kind='paper' AND model=?)""",
                (MODEL,)).fetchall()
        picked = {r["id"] for r in rows}
        self.assertIn(self.pid, picked,
                      "前置条件：只有 chunk 向量的论文会被 embed 选中")

        with db.conn() as c:
            total_before = c.execute(
                "SELECT COUNT(*) n FROM vectors WHERE kind='chunk' AND model=?",
                (MODEL,)).fetchone()["n"]

        def fake_embed(texts, purpose="embed"):
            return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]

        with mock.patch.object(embeddings, "embed_texts", side_effect=fake_embed):
            for r in rows:
                embeddings.index_paper(r["id"], r["title"], r["abstract"] or "")

        with db.conn() as c:
            total_after = c.execute(
                "SELECT COUNT(*) n FROM vectors WHERE kind='chunk' AND model=?",
                (MODEL,)).fetchone()["n"]
        self.assertEqual(total_after, total_before,
                         f"跑一遍 embed 之后 chunk 向量从 {total_before} 掉到 {total_after}")

    def test_index_paper_invalidates_matrix_cache(self):
        """检索矩阵含 paper+chunk 两种向量，重建之后缓存必须失效，否则读到旧矩阵。"""
        ids, _idxs, _m = embeddings._paper_matrix()
        self.assertIsNotNone(ids, "前置条件：矩阵里应当已有向量")
        self._run_index()
        self.assertIsNone(embeddings._MATRIX, "index_paper 之后矩阵缓存没有被作废")


if __name__ == "__main__":
    unittest.main()
