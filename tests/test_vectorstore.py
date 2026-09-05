"""向量存储后端抽象层：numpy 与 Chroma 跑同一组契约断言。

两件事这个文件必须钉死：

1. **两个后端行为一致。** 契约类 `VectorStoreContract` 只写一遍断言，
   `NumpyStoreTests` / `ChromaStoreTests` 各绑一个后端跑一遍；再加一个
   `CrossBackendTests` 直接比较两边返回的 top-k。否则 A/B 换后端时，
   「换完结果变了」分不清是后端差异还是实现 bug。
2. **夹具用有结构的向量，不用随机高斯。** 本项目量过：1024 维随机高斯向量彼此
   近乎等距、没有近邻结构可索引，拿它测 HNSW 会得到 recall@5≈0.56 的假结论
   （真实向量测出来是 1.000）。所以这里用「主题簇」向量：4 个主题中心 + 确定性抖动，
   簇内余弦 ≈0.80、簇间 ≈0.10——这才是 embedding 的真实形态。
   抖动用 `math.sin` 的确定性公式而不是 np.random，跨 numpy 版本也不会漂。

全程离线：config.DB_PATH / DATA_DIR 指到 tempfile，`embeddings.embed_texts`
被换成会抛断言错的桩——真调一次就红，绝不可能花到钱。
"""
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papernest import config, db, vectorstore as vs

DIM = 32
TOPICS = 4
PAPERS = 12
CHUNKS_PER_PAPER = 3


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _center(k: int) -> np.ndarray:
    """主题中心：在若干相邻维度上有一个平滑的「话题峰」。

    真实 embedding 就是这种形态——少数维度强、相邻主题部分重叠（这里中心两两余弦
    0.26/0.00），而不是各向同性的随机噪声。
    """
    return _unit([math.exp(-((j - (k * 7 + 3)) ** 2) / 18.0) for j in range(DIM)])


def _doc(k: int, n: int) -> np.ndarray:
    """主题 k 下第 n 篇文档。抖动是 n 的确定性函数，跨进程/跨版本完全可复现。"""
    jit = _unit([math.sin((n + 1) * (j + 1) * 0.7 + k) for j in range(DIM)])
    return _unit(_center(k) + 0.5 * jit)


def _query(k: int) -> np.ndarray:
    """落在主题 k 里的查询（不等于任何一篇文档，但明显更靠近该簇）。"""
    return _unit(_center(k) + 0.15 * _unit([math.sin(j * 1.3 + k) for j in range(DIM)]))


def _axis(*pairs) -> np.ndarray:
    """按 (维度, 权重) 造轴向量，用来手算余弦。"""
    v = np.zeros(DIM, dtype=np.float32)
    for j, w in pairs:
        v[j] = w
    return v


def _rmtree(path):
    """删掉本用例的临时目录。

    **不删是会积起来的**：每个用例一个 `mkdtemp`，Chroma 后端的那份还带一个
    PersistentClient 的 sqlite（约 400KB）。原来 tearDown 里没有这一步，
    本机 %TEMP% 下已经攒了 3376 个 `papernest_vs_*` 目录、1.37GB。
    `ignore_errors` 是必须的：Windows 上 Chroma 的句柄要等 `vs.reset_clients()`
    清掉 SharedSystemClient 缓存后才放，个别文件仍可能占着，删不掉也不该让用例红。
    """
    shutil.rmtree(path, ignore_errors=True)


class VectorStoreContract:
    """两个后端共用的契约断言。子类只绑 `backend`，一行都不重复写。"""

    backend = "numpy"

    # ── 夹具 ──

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix=f"papernest_vs_{self.backend}_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = "test-embed"
        vs.reset_clients()
        # 真调一次 embedding 接口就红：开发机 .env 是真 key，会花钱
        self._guard = mock.patch(
            "papernest.embeddings.embed_texts",
            side_effect=AssertionError("测试绝不能真调 embedding 接口"))
        self._guard.start()
        self._avail = mock.patch("papernest.embeddings.available", return_value=False)
        self._avail.start()
        db.init_db()
        self.topic_of: dict[int, int] = {}
        self.pids: list[int] = []
        with db.conn() as c:
            for n in range(PAPERS):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:vs{1000 + n}", "title": f"Paper {n}",
                    "abstract": "abs", "year": 2020 + n % 5, "venue": "V",
                    "authors": [], "doi": None, "arxiv_id": str(1000 + n),
                    "source": "s2"})
                self.pids.append(pid)
                self.topic_of[pid] = n % TOPICS
            self.axis_pids = []
            for n in range(3):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:vsx{n}", "title": f"Axis {n}", "abstract": "abs",
                    "year": 2024, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": f"9{n}", "source": "s2"})
                self.axis_pids.append(pid)
        self.store = vs.get_store(self.backend)
        self.assertEqual(self.store.name, self.backend)
        self.store.upsert(self._corpus())

    def tearDown(self):
        self._avail.stop()
        self._guard.stop()
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old
        vs.reset_clients()
        _rmtree(self._tmp)

    def _corpus(self) -> list[dict]:
        """12 篇 × (1 条 paper 向量 + 3 条 chunk 向量) + 3 条手算用的 sent 轴向量。"""
        items = []
        for n, pid in enumerate(self.pids):
            k = self.topic_of[pid]
            items.append({"paper_id": pid, "kind": "paper", "idx": 0,
                          "text": f"paper-{pid} 标题与摘要", "vec": _doc(k, n * 10),
                          "meta": {"year": 2020 + n % 5, "topic": k}})
            for i in range(CHUNKS_PER_PAPER):
                items.append({"paper_id": pid, "kind": "chunk", "idx": i,
                              "text": f"paper-{pid} 第 {i} 块正文",
                              "vec": _doc(k, n * 10 + 1 + i),
                              "meta": {"year": 2020 + n % 5, "topic": k}})
        for pid, vec in zip(self.axis_pids,
                            (_axis((0, 1.0)), _axis((0, 0.8), (1, 0.6)), _axis((1, 1.0)))):
            items.append({"paper_id": pid, "kind": "sent", "idx": 0,
                          "text": f"axis-{pid}", "vec": vec, "meta": {"axis": True}})
        return items

    def _keys(self, hits) -> list[tuple]:
        return [(h["paper_id"], h["kind"], h["idx"]) for h in hits]

    # ── score 是相似度，不是距离 ──

    def test_score_is_cosine_similarity_not_distance(self):
        """同向≈1.0、夹角居中=0.8、正交=0.0。Chroma 返回的是距离，必须转过来。"""
        hits = self.store.search(_axis((0, 1.0)), top_k=3, kind="sent")
        self.assertEqual([h["paper_id"] for h in hits], self.axis_pids)
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)
        self.assertAlmostEqual(hits[1]["score"], 0.8, places=5)
        self.assertAlmostEqual(hits[2]["score"], 0.0, places=5)

    def test_ranking_is_hand_computable(self):
        """查询 = 0.6·e0 + 0.8·e1：手算余弦 0.6 / 0.96 / 0.8，顺序应为 中间→e1→e0。"""
        hits = self.store.search(_axis((0, 0.6), (1, 0.8)), top_k=3, kind="sent")
        self.assertEqual([h["paper_id"] for h in hits],
                         [self.axis_pids[1], self.axis_pids[2], self.axis_pids[0]])
        for got, want in zip(hits, (0.96, 0.8, 0.6)):
            self.assertAlmostEqual(got["score"], want, places=5)

    def test_result_shape(self):
        hit = self.store.search(_query(0), top_k=1)[0]
        self.assertEqual(set(hit), {"paper_id", "kind", "idx", "score", "text"})
        self.assertIsInstance(hit["paper_id"], int)
        self.assertIsInstance(hit["idx"], int)
        self.assertIsInstance(hit["score"], float)
        # text 从 SQLite 回取（真相来源），不是从派生索引里读的
        self.assertIn(str(hit["paper_id"]), hit["text"])

    # ── 检索质量：有结构的向量才测得出近邻 ──

    def test_query_returns_its_own_topic_cluster(self):
        """簇内余弦 0.80 / 簇间 0.10：top-8 应该全落在查询所属主题里。"""
        for k in range(TOPICS):
            hits = self.store.search(_query(k), top_k=8, kind="chunk")
            self.assertEqual(len(hits), 8)
            self.assertEqual({self.topic_of[h["paper_id"]] for h in hits}, {k},
                             f"主题 {k} 的查询召回了别的簇：{self._keys(hits)}")

    def test_scores_are_descending(self):
        hits = self.store.search(_query(2), top_k=10)
        self.assertEqual([h["score"] for h in hits],
                         sorted((h["score"] for h in hits), reverse=True))

    # ── kind / where 过滤 ──

    def test_kind_filter_excludes_other_kinds(self):
        hits = self.store.search(_query(0), top_k=20, kind="chunk")
        self.assertTrue(hits)
        self.assertEqual({h["kind"] for h in hits}, {"chunk"})
        self.assertEqual({h["kind"] for h in self.store.search(_query(0), 5, kind="paper")},
                         {"paper"})

    def test_kind_filter_unknown_kind_returns_empty(self):
        self.assertEqual(self.store.search(_query(0), 5, kind="figure"), [])

    def test_where_filters_on_user_meta(self):
        hits = self.store.search(_query(1), top_k=20, where={"topic": 1})
        self.assertTrue(hits)
        self.assertEqual({self.topic_of[h["paper_id"]] for h in hits}, {1})

    def test_where_supports_range_operator(self):
        hits = self.store.search(_query(0), top_k=50, where={"year": {"$gte": 2023}})
        self.assertTrue(hits)
        with db.conn() as c:
            years = {r["id"]: r["year"] for r in c.execute("SELECT id, year FROM papers")}
        self.assertTrue(all(years[h["paper_id"]] >= 2023 for h in hits))

    def test_where_can_filter_on_reserved_paper_id(self):
        pid = self.pids[3]
        hits = self.store.search(_query(3), top_k=20, where={"paper_id": pid})
        self.assertTrue(hits)
        self.assertEqual({h["paper_id"] for h in hits}, {pid})

    def test_kind_and_where_combine(self):
        hits = self.store.search(_query(1), top_k=20, kind="chunk", where={"topic": 1})
        self.assertTrue(hits)
        self.assertEqual({h["kind"] for h in hits}, {"chunk"})
        self.assertEqual({self.topic_of[h["paper_id"]] for h in hits}, {1})

    def test_where_and_or(self):
        hits = self.store.search(_query(0), top_k=50,
                                 where={"$or": [{"topic": 0}, {"topic": 2}]})
        self.assertTrue(hits)
        self.assertLessEqual({self.topic_of[h["paper_id"]] for h in hits}, {0, 2})

    def test_where_rejects_multiple_top_level_keys(self):
        """故意与 Chroma 一样严：宽松了就会出现「一个后端跑通、另一个炸」。"""
        with self.assertRaises(vs.VectorStoreError):
            self.store.search(_query(0), 5, where={"topic": 1, "year": 2024})

    def test_where_rejects_unknown_operator(self):
        with self.assertRaises(vs.VectorStoreError):
            self.store.search(_query(0), 5, where={"year": {"$like": 2024}})

    # ── top_k 边界 ──

    def test_top_k_is_respected(self):
        self.assertEqual(len(self.store.search(_query(0), top_k=5)), 5)

    def test_top_k_larger_than_corpus(self):
        total = self.store.count()
        self.assertEqual(len(self.store.search(_query(0), top_k=total + 50)), total)

    def test_top_k_zero_returns_empty(self):
        self.assertEqual(self.store.search(_query(0), top_k=0), [])

    # ── upsert 语义 ──

    def test_upsert_same_id_twice_does_not_duplicate(self):
        before = self.store.count()
        self.store.upsert(self._corpus())
        self.assertEqual(self.store.count(), before)
        keys = self._keys(self.store.search(_query(0), top_k=8, kind="chunk"))
        self.assertEqual(len(keys), len(set(keys)))

    def test_upsert_overwrites_vector_and_text(self):
        pid = self.pids[0]
        self.store.upsert([{"paper_id": pid, "kind": "sent", "idx": 0,
                            "text": "改写后的证据句", "vec": _axis((0, 1.0)),
                            "meta": {"axis": True}}])
        hits = self.store.search(_axis((0, 1.0)), top_k=4, kind="sent")
        mine = [h for h in hits if h["paper_id"] == pid]
        self.assertEqual(len(mine), 1)
        self.assertAlmostEqual(mine[0]["score"], 1.0, places=5)
        self.assertEqual(mine[0]["text"], "改写后的证据句")

    def test_upsert_returns_written_count(self):
        self.assertEqual(self.store.upsert([]), 0)
        self.assertEqual(self.store.upsert(
            [{"paper_id": self.pids[0], "kind": "chunk", "idx": 99,
              "text": "新块", "vec": _doc(0, 999)}]), 1)

    def test_meta_cannot_override_reserved_keys(self):
        with self.assertRaises(vs.VectorStoreError):
            self.store.upsert([{"paper_id": 1, "kind": "chunk", "idx": 0, "text": "x",
                                "vec": _doc(0, 1), "meta": {"paper_id": 7}}])

    def test_meta_values_must_be_scalar(self):
        """Chroma metadata 只收标量；numpy 后端也拒——不然换后端才炸。"""
        with self.assertRaises(vs.VectorStoreError):
            self.store.upsert([{"paper_id": 1, "kind": "chunk", "idx": 0, "text": "x",
                                "vec": _doc(0, 1), "meta": {"tags": ["a", "b"]}}])

    def test_kind_with_colon_rejected(self):
        """id 编码是 kind:paper_id:idx，kind 里带冒号就反解不回来了。"""
        with self.assertRaises(vs.VectorStoreError):
            self.store.upsert([{"paper_id": 1, "kind": "a:b", "idx": 0, "text": "x",
                                "vec": _doc(0, 1)}])

    # ── 维度：报错，不静默截断 ──

    def test_search_with_wrong_dimension_raises(self):
        with self.assertRaises(vs.DimensionMismatch):
            self.store.search(np.ones(DIM - 1, dtype=np.float32), top_k=3)

    def test_upsert_with_wrong_dimension_raises(self):
        before = self.store.count()
        with self.assertRaises(vs.DimensionMismatch):
            self.store.upsert([{"paper_id": self.pids[0], "kind": "chunk", "idx": 50,
                                "text": "x", "vec": np.ones(DIM + 8, dtype=np.float32)}])
        self.assertEqual(self.store.count(), before, "维度错的写入不能留下半条记录")

    def test_mixed_dimensions_in_one_batch_raises(self):
        before = self.store.count()
        with self.assertRaises(vs.DimensionMismatch):
            self.store.upsert([
                {"paper_id": self.pids[0], "kind": "chunk", "idx": 60,
                 "text": "a", "vec": np.ones(DIM, dtype=np.float32)},
                {"paper_id": self.pids[0], "kind": "chunk", "idx": 61,
                 "text": "b", "vec": np.ones(DIM + 1, dtype=np.float32)}])
        self.assertEqual(self.store.count(), before, "整批应原子拒绝，不能写一半")

    def test_empty_vector_rejected(self):
        with self.assertRaises(vs.VectorStoreError):
            self.store.upsert([{"paper_id": 1, "kind": "chunk", "idx": 0,
                                "text": "x", "vec": []}])

    def test_nan_vector_rejected(self):
        bad = np.ones(DIM, dtype=np.float32)
        bad[3] = np.nan
        with self.assertRaises(vs.VectorStoreError):
            self.store.upsert([{"paper_id": 1, "kind": "chunk", "idx": 0,
                                "text": "x", "vec": bad}])

    # ── 删除 ──

    def test_delete_paper_removes_every_kind(self):
        pid = self.pids[0]
        n = self.store.delete_paper(pid)
        self.assertEqual(n, 1 + CHUNKS_PER_PAPER)
        for kind in (None, "paper", "chunk", "sent"):
            hits = self.store.search(_query(self.topic_of[pid]), top_k=60, kind=kind)
            self.assertNotIn(pid, {h["paper_id"] for h in hits},
                             f"kind={kind} 仍能搜到已删除的 paper_id={pid}")

    def test_delete_paper_is_idempotent(self):
        pid = self.pids[1]
        self.assertGreater(self.store.delete_paper(pid), 0)
        self.assertEqual(self.store.delete_paper(pid), 0)

    def test_delete_all_then_empty_store(self):
        for pid in self.pids + self.axis_pids:
            self.store.delete_paper(pid)
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.search(_query(0), top_k=5), [])
        self.assertEqual(self.store.search(_query(0), top_k=5, kind="chunk"), [])

    # ── 计数 ──

    def test_count_by_kind(self):
        self.assertEqual(self.store.count("paper"), PAPERS)
        self.assertEqual(self.store.count("chunk"), PAPERS * CHUNKS_PER_PAPER)
        self.assertEqual(self.store.count("sent"), 3)
        self.assertEqual(self.store.count(), PAPERS * (1 + CHUNKS_PER_PAPER) + 3)
        self.assertEqual(self.store.count("figure"), 0)

    # ── rebuild ──

    def test_rebuild_keeps_results_identical(self):
        queries = [(_query(k), kind) for k in range(TOPICS)
                   for kind in (None, "chunk", "paper")]
        before = [self.store.search(q, top_k=6, kind=kd) for q, kd in queries]
        report = self.store.rebuild()
        self.assertEqual(report["errors"], [])
        after = [self.store.search(q, top_k=6, kind=kd) for q, kd in queries]
        self.assertEqual(before, after)

    def test_rebuild_report_shape(self):
        seen = []
        report = self.store.rebuild(progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(set(report), {"backend", "indexed", "elapsed_s", "errors"})
        self.assertEqual(report["backend"], self.backend)
        self.assertEqual(report["indexed"], self.store.count())
        self.assertIsInstance(report["elapsed_s"], float)
        self.assertEqual(report["errors"], [])
        self.assertTrue(seen, "progress 回调没被调用过")
        self.assertEqual(seen[-1][0], seen[-1][1])

    # ── 确定性 ──

    def test_two_runs_are_byte_identical(self):
        """连跑两次必须完全一致——排序键不全序时这里会红。"""
        for k in range(TOPICS):
            a = self.store.search(_query(k), top_k=9, kind="chunk")
            b = self.store.search(_query(k), top_k=9, kind="chunk")
            self.assertEqual(a, b)

    def test_search_opens_exactly_one_connection(self):
        """一次 search 只开一条 db.conn()。

        这不是洁癖，是量出来的：真库（33MB、sqlite_master 82 个对象）上新连接的
        **第一条**语句要 2.48ms（要把整库 schema 读进来），同一连接上再查两次几乎免费。
        开两条连接就是把这 2.5ms 再付一遍——端到端从 3.3ms 变 9.4ms。
        """
        self.store.search(_query(0), top_k=5)          # 先把矩阵缓存热起来
        real, opened = db.conn, []

        def counting(*a, **k):
            opened.append(1)
            return real(*a, **k)

        with mock.patch.object(db, "conn", counting):
            self.store.search(_query(0), top_k=5)
        self.assertEqual(len(opened), 1, f"一次 search 开了 {len(opened)} 条连接")

    def test_ties_broken_by_paper_kind_idx(self):
        """分数并列时按 (paper_id, kind, idx) 决胜，否则跨进程会漂移（本项目踩过）。"""
        tied = _axis((7, 1.0))
        plan = [(self.pids[2], 91), (self.pids[0], 92), (self.pids[0], 90),
                (self.pids[1], 90)]
        self.store.upsert([{"paper_id": pid, "kind": "chunk", "idx": idx,
                            "text": f"tie-{pid}-{idx}", "vec": tied,
                            "meta": {"tie": 1}} for pid, idx in plan])
        hits = self.store.search(tied, top_k=4, kind="chunk", where={"tie": 1})
        self.assertEqual([(h["paper_id"], h["idx"]) for h in hits], sorted(plan))
        self.assertEqual(len({round(h["score"], 5) for h in hits}), 1, "这四条本该同分")

    def test_tie_group_larger_than_top_k_picks_smallest_keys(self):
        """并列条数**多于** top_k 时，进 top-k 的必须是键最小的那几条。

        上一条用例里并列数恰好等于 top_k，两个后端的「并列展宽」代码根本执行不到：
        numpy 走的是 `k == scores.size` 的分支、Chroma 的过取窗口也没起作用。
        把 numpy 的 argpartition 展宽删掉、把 Chroma 的过取窗口砍成 top_k，
        原来的用例全都照绿——所以补这一条。
        （numpy 2.4.2 上 argpartition 对全等分数恰好返回前 k 个，换版本就不一定；
        Chroma 侧则是 HNSW 只保证近似 top-k。两边都必须把并列行全捞回来再全序排。）
        """
        q = _axis((9, 1.0), (10, 1.0))          # 与 tied 的余弦 0.7071，与 winner 1.0
        tied, winner = _axis((9, 1.0)), _axis((9, 1.0), (10, 1.0))
        plan = [(self.pids[i], 300 + j) for i in range(4) for j in range(3)][:11]
        top = (self.pids[3], 399)               # 键最大、分数最高
        items = [{"paper_id": pid, "kind": "chunk", "idx": idx, "text": f"tg-{pid}-{idx}",
                  "vec": tied, "meta": {"tiebig": 1}} for pid, idx in plan]
        items.append({"paper_id": top[0], "kind": "chunk", "idx": top[1],
                      "text": "tg-winner", "vec": winner, "meta": {"tiebig": 1}})
        # 倒序写入：让「按插入序截断」与正确答案必然不同，Chroma 侧才测得出来
        self.store.upsert(list(reversed(items)))
        hits = self.store.search(q, top_k=3, kind="chunk", where={"tiebig": 1})
        self.assertEqual([(h["paper_id"], h["idx"]) for h in hits], [top] + sorted(plan)[:2])
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)
        self.assertAlmostEqual(hits[1]["score"], hits[2]["score"], places=6)

    def test_unnormalized_vectors_still_score_as_cosine(self):
        """入库向量与查询向量**都不是单位长度**时，score 仍必须是余弦。

        原夹具每一条向量都是 `_unit(...)` 出来的单位向量，于是「矩阵按行归一化」
        和「查询向量归一化」两段代码从来没被执行到——把它们整段删掉，
        原来的 99 个用例照样全绿（本轮变异检验实测如此）。而 embedding 接口
        并不保证返回单位向量，一旦不是，score 就不再是余弦、跨后端也对不上。
        """
        pid = self.axis_pids[0]
        self.store.upsert([{"paper_id": pid, "kind": "sent", "idx": 5,
                            "text": "长度 5 的 e0", "vec": _axis((0, 5.0)),
                            "meta": {"axis": True}}])
        # 查询 = 3·(0.6·e0 + 0.8·e1)，长度 3；余弦仍应是 0.96 / 0.8 / 0.6 / 0.6
        hits = self.store.search(_axis((0, 1.8), (1, 2.4)), top_k=4, kind="sent")
        self.assertEqual([(h["paper_id"], h["idx"]) for h in hits],
                         [(self.axis_pids[1], 0), (self.axis_pids[2], 0),
                          (self.axis_pids[0], 0), (self.axis_pids[0], 5)])
        for got, want in zip(hits, (0.96, 0.8, 0.6, 0.6)):
            self.assertAlmostEqual(got["score"], want, places=5)

    def test_delete_paper_removes_vectors_of_every_model(self):
        """「这篇没了」不该只删当前模型那一份。

        本项目换嵌入提供方时旧模型向量会与新模型共存
        （`embeddings.check_space_compatible` 就是靠 `model != ?` 找旧向量的）。
        只删当前模型会留下一批既搜不到、也再删不掉的孤儿行。
        """
        pid = self.pids[2]
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (pid, "paper", 0, "旧模型向量", "legacy-embed",
                       np.ones(DIM, dtype=np.float32).tobytes()))
        self.store.delete_paper(pid)
        with db.conn() as c:
            left = c.execute("SELECT COUNT(1) n FROM vectors WHERE paper_id=?",
                             (pid,)).fetchone()["n"]
        self.assertEqual(left, 0, "旧嵌入模型的向量没被删掉，成了删不掉的孤儿行")

    def test_write_invalidates_embeddings_paper_matrix(self):
        """写入/删除要连带作废 embeddings 那份常驻矩阵——两边是同一张 vectors 表。

        不作废就会出现「这里删干净了、那里还能搜到」的幽灵结果。
        模块 docstring 明写了这一条，但原来没有任何用例覆盖：把
        `embeddings.invalidate_cache()` 那一句删掉，测试全绿。
        """
        from papernest import embeddings
        stale = {"key": ("stale",), "ids": None, "M": None}
        embeddings._MATRIX = dict(stale)
        self.store.upsert([{"paper_id": self.pids[0], "kind": "chunk", "idx": 78,
                            "text": "x", "vec": _doc(0, 3)}])
        self.assertIsNone(embeddings._MATRIX, "upsert 后没作废 embeddings 的常驻矩阵")
        embeddings._MATRIX = dict(stale)
        self.store.delete_paper(self.pids[0])
        self.assertIsNone(embeddings._MATRIX, "delete_paper 后没作废 embeddings 的常驻矩阵")


class NumpyStoreTests(VectorStoreContract, unittest.TestCase):
    backend = "numpy"

    def test_matrix_cache_invalidated_by_write(self):
        """常驻矩阵按 (条数, max(id)) 指纹失效：写完立刻能搜到，不用手动清缓存。"""
        self.store.search(_query(0), top_k=3)          # 先把缓存热起来
        pid = self.pids[0]
        self.store.upsert([{"paper_id": pid, "kind": "chunk", "idx": 77,
                            "text": "刚写进来的块", "vec": _axis((11, 1.0))}])
        hits = self.store.search(_axis((11, 1.0)), top_k=1, kind="chunk")
        self.assertEqual((hits[0]["paper_id"], hits[0]["idx"]), (pid, 77))
        self.assertEqual(hits[0]["text"], "刚写进来的块")

    def test_mixed_dimension_library_raises_instead_of_reshaping(self):
        """库里混了两种维度时 reshape 会「成功」但每行错位——必须报错。"""
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pids[0], "chunk", 88, "坏行", config.EMBED_MODEL,
                       np.ones(DIM + 4, dtype=np.float32).tobytes()))
        vs.reset_clients()
        with self.assertRaises(vs.DimensionMismatch):
            self.store.search(_query(0), top_k=3)


class ChromaStoreTests(VectorStoreContract, unittest.TestCase):
    backend = "chroma"

    def test_index_count_tracks_truth_count(self):
        self.assertEqual(self.store.index_count(), self.store.count())
        self.store.delete_paper(self.pids[0])
        self.assertEqual(self.store.index_count(), self.store.count())

    def test_ids_are_stable_and_reversible(self):
        self.assertEqual(vs.ChromaStore._id(12, "chunk", 3), "chunk:12:3")
        self.assertEqual(vs.ChromaStore._parse_id("chunk:12:3"), (12, "chunk", 3))
        self.assertIsNone(vs.ChromaStore._parse_id("坏id"))

    def test_sqlite_wins_when_index_is_stale(self):
        """绕过 store 直接删 SQLite：Chroma 还留着那几行，但检索结果里不能出现。

        这条就是「SQLite 是唯一真相来源」的可执行版本。丢弃要留痕（last_stale_hits），
        不能静默——静默的话，索引漂了都没人知道。
        """
        pid = self.pids[0]
        with db.conn() as c:
            c.execute("DELETE FROM vectors WHERE paper_id=?", (pid,))
        vs.reset_clients()
        store = vs.get_store("chroma")
        self.assertGreater(store.index_count(), store.count(), "夹具没制造出索引漂移")
        hits = store.search(_query(self.topic_of[pid]), top_k=60)
        self.assertNotIn(pid, {h["paper_id"] for h in hits})
        self.assertGreater(store.last_stale_hits, 0, "丢弃了陈旧命中却没留痕")

    def test_rebuild_restores_index_wiped_behind_our_back(self):
        """有人手工删了 collection：rebuild 从 SQLite 全量重放，一条不少。"""
        before = self.store.search(_query(1), top_k=6, kind="chunk")
        self.store._client.delete_collection(vs.collection_name())
        self.store._col = self.store._collection()
        self.assertEqual(self.store.index_count(), 0)
        report = self.store.rebuild()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["indexed"], self.store.count())
        self.assertEqual(self.store.search(_query(1), top_k=6, kind="chunk"), before)

    def test_rebuild_restores_user_meta(self):
        """meta 也在 SQLite 里（vector_meta 表），否则重建完 where 过滤就废了。"""
        self.store.rebuild()
        hits = self.store.search(_query(1), top_k=20, where={"topic": 1})
        self.assertTrue(hits)
        self.assertEqual({self.topic_of[h["paper_id"]] for h in hits}, {1})


class CrossBackendTests(unittest.TestCase):
    """两个后端在同一份 SQLite 上必须给出同样的 top-k。

    写入走 ChromaStore（它会同时落 SQLite 与 Chroma），再用 NumpyStore 读同一份
    SQLite——这正是主进程 A/B 时的真实姿势。
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_vs_cross_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = "test-embed"
        vs.reset_clients()
        self._guard = mock.patch(
            "papernest.embeddings.embed_texts",
            side_effect=AssertionError("测试绝不能真调 embedding 接口"))
        self._guard.start()
        db.init_db()
        self.topic_of, self.pids = {}, []
        with db.conn() as c:
            for n in range(PAPERS):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:cross{n}", "title": f"P{n}", "abstract": "a",
                    "year": 2020 + n % 5, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": str(2000 + n), "source": "s2"})
                self.pids.append(pid)
                self.topic_of[pid] = n % TOPICS
        items = []
        for n, pid in enumerate(self.pids):
            k = self.topic_of[pid]
            items.append({"paper_id": pid, "kind": "paper", "idx": 0,
                          "text": f"p{pid}", "vec": _doc(k, n * 10),
                          "meta": {"year": 2020 + n % 5, "topic": k}})
            for i in range(CHUNKS_PER_PAPER):
                items.append({"paper_id": pid, "kind": "chunk", "idx": i,
                              "text": f"c{pid}-{i}", "vec": _doc(k, n * 10 + 1 + i),
                              "meta": {"year": 2020 + n % 5, "topic": k}})
        self.chroma = vs.get_store("chroma")
        self.chroma.upsert(items)
        self.numpy = vs.get_store("numpy")

    def tearDown(self):
        self._guard.stop()
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old
        vs.reset_clients()
        _rmtree(self._tmp)

    def _assert_same(self, qvec, top_k, **kw):
        a = self.numpy.search(qvec, top_k, **kw)
        b = self.chroma.search(qvec, top_k, **kw)
        self.assertEqual([(h["paper_id"], h["kind"], h["idx"]) for h in a],
                         [(h["paper_id"], h["kind"], h["idx"]) for h in b],
                         f"两后端 top-{top_k} 不一致（kw={kw}）")
        for x, y in zip(a, b):
            self.assertAlmostEqual(x["score"], y["score"], places=5)
            self.assertEqual(x["text"], y["text"])

    def test_top_k_identical_for_every_topic(self):
        for k in range(TOPICS):
            for top_k in (1, 5, 10):
                self._assert_same(_query(k), top_k)

    def test_top_k_identical_with_kind_filter(self):
        for k in range(TOPICS):
            self._assert_same(_query(k), 6, kind="chunk")
            self._assert_same(_query(k), 6, kind="paper")

    def test_top_k_identical_with_where_filter(self):
        self._assert_same(_query(0), 8, where={"topic": 0})
        self._assert_same(_query(1), 8, kind="chunk", where={"year": {"$gte": 2022}})

    def test_counts_agree(self):
        self.assertEqual(self.numpy.count(), self.chroma.count())
        self.assertEqual(self.numpy.count("chunk"), self.chroma.count("chunk"))

    def test_delete_through_numpy_is_visible_to_chroma_after_rebuild(self):
        """numpy 后端删除只动了真相；Chroma 索引要 rebuild 才跟上——这正是设计意图。"""
        pid = self.pids[0]
        self.numpy.delete_paper(pid)
        self.assertNotIn(pid, {h["paper_id"]
                               for h in self.numpy.search(_query(self.topic_of[pid]), 60)})
        self.chroma.rebuild()
        self.assertEqual(self.chroma.index_count(), self.chroma.count())
        self._assert_same(_query(self.topic_of[pid]), 8)


class WhereSyntaxTests(unittest.TestCase):
    """where 求值器单独测——它是两个后端一致性的地基，不该只靠端到端覆盖。"""

    def test_validate_accepts_supported_forms(self):
        for w in (None, {"year": 2024}, {"year": {"$gte": 2023}},
                  {"kind": {"$in": ["chunk", "paper"]}},
                  {"$and": [{"topic": 1}, {"year": {"$lt": 2024}}]},
                  {"$or": [{"topic": 1}, {"topic": 2}]}):
            vs.validate_where(w)

    def test_validate_rejects_bad_forms(self):
        for w in ({"a": 1, "b": 2}, {"$nope": [{"a": 1}]}, {"a": {"$like": 1}},
                  {"a": {"$gt": 1, "$lt": 3}}, {"$and": []}, {"a": {"$in": "abc"}},
                  {"a": {"$in": []}}, "not a dict", {"a": [1, 2]}):
            with self.assertRaises(vs.VectorStoreError, msg=f"没拒绝 {w!r}"):
                vs.validate_where(w)

    def test_match_semantics(self):
        meta = {"paper_id": 3, "kind": "chunk", "idx": 0, "year": 2023, "tag": "nlp"}
        self.assertTrue(vs.match_where(meta, None))
        self.assertTrue(vs.match_where(meta, {"year": 2023}))
        self.assertFalse(vs.match_where(meta, {"year": 2024}))
        self.assertTrue(vs.match_where(meta, {"year": {"$gte": 2023}}))
        self.assertFalse(vs.match_where(meta, {"year": {"$gt": 2023}}))
        self.assertTrue(vs.match_where(meta, {"tag": {"$in": ["nlp", "cv"]}}))
        self.assertFalse(vs.match_where(meta, {"tag": {"$nin": ["nlp"]}}))
        self.assertTrue(vs.match_where(meta, {"$and": [{"year": 2023}, {"tag": "nlp"}]}))
        self.assertFalse(vs.match_where(meta, {"$and": [{"year": 2023}, {"tag": "cv"}]}))
        self.assertTrue(vs.match_where(meta, {"$or": [{"year": 1999}, {"tag": "nlp"}]}))

    def test_missing_key_never_matches(self):
        """缺失的键一律不匹配（不当 NULL 参与比较）——语义对齐 Chroma。"""
        meta = {"paper_id": 1, "kind": "chunk", "idx": 0}
        self.assertFalse(vs.match_where(meta, {"year": {"$gte": 0}}))
        self.assertFalse(vs.match_where(meta, {"year": {"$ne": 2024}}))

    def test_incomparable_types_do_not_crash(self):
        meta = {"tag": "nlp"}
        self.assertFalse(vs.match_where(meta, {"tag": {"$gt": 3}}))


class GetStoreTests(unittest.TestCase):
    """工厂与降级口径：Chroma 不可用时的行为必须是明确的、被测过的。"""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_vs_factory_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = "test-embed"
        vs.reset_clients()
        db.init_db()

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old
        vs.reset_clients()
        _rmtree(self._tmp)

    def test_default_comes_from_module_constant(self):
        with mock.patch.object(vs, "BACKEND", "numpy"):
            self.assertEqual(vs.get_store().name, "numpy")
        with mock.patch.object(vs, "BACKEND", "chroma"):
            self.assertEqual(vs.get_store().name, "chroma")

    def test_explicit_backend_beats_constant(self):
        with mock.patch.object(vs, "BACKEND", "chroma"):
            self.assertEqual(vs.get_store("numpy").name, "numpy")

    def test_unknown_backend_raises_with_options(self):
        with self.assertRaises(vs.VectorStoreError) as cm:
            vs.get_store("faiss")
        self.assertIn("faiss", str(cm.exception))
        self.assertIn("numpy", str(cm.exception))

    def test_both_stores_satisfy_protocol(self):
        for name in ("numpy", "chroma"):
            self.assertIsInstance(vs.get_store(name), vs.VectorStore)

    def test_chroma_unavailable_raises_not_silent_fallback(self):
        """默认**抛错**。静默退回 numpy 会让 A/B 报告写着 Chroma、实际跑的是 numpy。"""
        with mock.patch.dict(sys.modules, {"chromadb": None}):
            vs.reset_clients()
            with self.assertRaises(vs.VectorStoreUnavailable) as cm:
                vs.get_store("chroma")
        self.assertIn("chromadb", str(cm.exception))

    def test_degrade_helper_returns_reason(self):
        with mock.patch.dict(sys.modules, {"chromadb": None}):
            vs.reset_clients()
            store, degrade = vs.get_store_or_degrade("chroma")
        self.assertEqual(store.name, "numpy")
        self.assertIsInstance(degrade, str)
        self.assertIn("chroma", degrade)
        self.assertEqual(store.degraded, degrade)

    def test_degrade_helper_reports_none_when_backend_is_available(self):
        store, degrade = vs.get_store_or_degrade("chroma")
        self.assertEqual(store.name, "chroma")
        self.assertIsNone(degrade)

    def test_env_opt_in_fallback_still_leaves_a_trace(self):
        with mock.patch.dict(os.environ, {vs.FALLBACK_ENV: "1"}), \
                mock.patch.dict(sys.modules, {"chromadb": None}):
            vs.reset_clients()
            store = vs.get_store("chroma")
        self.assertEqual(store.name, "numpy")
        self.assertIn("chroma", store.degraded or "")



class MultiModelIsolationTests(unittest.TestCase):
    """多模型共存：两个 embedding 模型的向量必须互不干扰。

    修复前的真实缺陷：Chroma 的 id 是 `kind:paper_id:idx` **不含 model**，
    同一篇论文在两个模型下撞成同一条，后写的覆盖先写的。之后用 A 模型的查询向量
    去比 B 模型的向量——**文本是对的、分数全错**（同向向量 score 应为 1.0，实测 0.0），
    而且 count() 与 index_count() 仍然相等，自检判据一起失效。
    修法：每个模型一个 collection（不同模型维度本就不同，塞进同一个 HNSW 索引从根上就不对）。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="papernest_vs_mm_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(vs.reset_clients)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        self.addCleanup(self._restore)
        config.DATA_DIR = self.tmp / "data"
        config.DB_PATH = config.DATA_DIR / "t.db"
        db.init_db()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:mm1", "title": "T", "abstract": "a", "year": 2024,
                "venue": None, "authors": [], "doi": None, "arxiv_id": "mm1",
                "source": "s2"})

    def _restore(self):
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old

    def _write(self, model, vec):
        config.EMBED_MODEL = model
        vs.reset_clients()
        vs.get_store("chroma").upsert(
            [{"paper_id": self.pid, "kind": "paper", "idx": 0,
              "text": f"{model} 的向量", "vec": vec}])

    def _search(self, model, q):
        config.EMBED_MODEL = model
        vs.reset_clients()
        return vs.get_store("chroma").search(q, 3)

    def test_two_models_do_not_overwrite_each_other(self):
        self._write("model-A", [1.0, 0.0])
        self._write("model-B", [0.0, 1.0])
        for model, q in (("model-A", [1.0, 0.0]), ("model-B", [0.0, 1.0])):
            hits = self._search(model, q)
            self.assertTrue(hits, f"{model} 查不到自己的向量")
            self.assertAlmostEqual(hits[0]["score"], 1.0, places=3,
                                   msg=f"{model} 的同向向量 score 应为 1.0")
            self.assertIn(model, hits[0]["text"])

    def test_collection_name_differs_per_model(self):
        a = vs.collection_name("model-A")
        b = vs.collection_name("model-B")
        self.assertNotEqual(a, b)

    def test_collection_name_passes_chroma_validation(self):
        """Chroma 校验：3-512 字符、只能 [a-zA-Z0-9._-]、首尾须字母数字。
        模型名可能含中文，sanitize 后要仍然合法且互不相同。"""
        for model in ("Qwen3.7-通用文本向量", "全中文模型名", "nomic-embed-text", ""):
            n = vs.collection_name(model)
            self.assertRegex(n, r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$",
                             f"{model!r} 生成的名字非法：{n}")
        # 两个中文名 sanitize 后都会退化成同样的横线串，靠哈希区分
        self.assertNotEqual(vs.collection_name("中文甲"),
                            vs.collection_name("中文乙"))

    def test_switching_model_without_reset_still_isolates(self):
        """进程内直接改 EMBED_MODEL（测试与 cli.py embed 迁移都会这么干），
        不调 reset_clients 也不能拿旧模型的索引回答新模型的查询。"""
        self._write("model-A", [1.0, 0.0])
        config.EMBED_MODEL = "model-A"
        vs.reset_clients()
        store = vs.get_store("chroma")
        self.assertTrue(store.search([1.0, 0.0], 3))
        config.EMBED_MODEL = "model-B"          # 同一个 store 实例，只换模型
        self.assertEqual(store.search([1.0, 0.0], 3), [],
                         "换模型后不该看到旧模型的向量")

if __name__ == "__main__":
    unittest.main()
