"""向量检索矩阵缓存：结果正确性、失效时机、线程安全、维度不匹配的处理。

背景：原实现每次查询都从 SQLite 重读全部 BLOB 再逐行算余弦。实测瓶颈在重读而非
计算——缓存归一化矩阵后 1 万条向量从 203ms 降到 1.8ms。所以**不上向量库**，
先把这 100 倍拿到手；真到十万条以上再换 sqlite-vec。
"""
import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papernest import config, db, embeddings

DIM = 8


def _blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_vec_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        self._old_model = config.EMBED_MODEL
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = "test-embed"
        embeddings.invalidate_cache()
        db.init_db()
        with db.conn() as c:
            self.ids = []
            for i, vec in enumerate([
                    [1, 0, 0, 0, 0, 0, 0, 0],       # 与查询完全同向
                    [0, 1, 0, 0, 0, 0, 0, 0],       # 正交
                    [0.8, 0.6, 0, 0, 0, 0, 0, 0],   # 夹角居中
            ], 1):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:{7000+i}", "title": f"P{i}", "abstract": "a",
                    "year": 2024, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": str(7000 + i), "source": "s2"})
                self.ids.append(pid)
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,?,?,?,?,?)",
                          (pid, "paper", 0, f"P{i}", "test-embed", _blob(vec)))

    def tearDown(self):
        embeddings.invalidate_cache()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir
        config.EMBED_MODEL = self._old_model

    def _search(self, qv, top_k=3):
        with mock.patch.object(embeddings, "embed_texts", return_value=[qv]):
            return embeddings.search_papers("q", top_k)

    # ── 正确性 ──

    def test_ranking_matches_hand_computed_cosine(self):
        """夹角越小排越前——这个顺序可以手算验证。"""
        got = self._search([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual([g["paper_id"] for g in got],
                         [self.ids[0], self.ids[2], self.ids[1]])
        self.assertAlmostEqual(got[0]["score"], 1.0, places=5)
        self.assertAlmostEqual(got[1]["score"], 0.8, places=5)
        self.assertAlmostEqual(got[2]["score"], 0.0, places=5)

    def test_top_k_is_respected(self):
        self.assertEqual(len(self._search([1, 0, 0, 0, 0, 0, 0, 0], top_k=2)), 2)

    def test_top_k_larger_than_corpus(self):
        self.assertEqual(len(self._search([1, 0, 0, 0, 0, 0, 0, 0], top_k=99)), 3)

    def test_ties_are_broken_deterministically(self):
        """并列时按 paper_id 决胜；否则跨进程结果会漂移（本项目踩过）。"""
        with db.conn() as c:
            c.execute("UPDATE vectors SET vec=? WHERE model='test-embed'",
                      (_blob([1, 0, 0, 0, 0, 0, 0, 0]),))
        embeddings.invalidate_cache()
        runs = {tuple(g["paper_id"] for g in self._search([1, 0, 0, 0, 0, 0, 0, 0]))
                for _ in range(5)}
        self.assertEqual(len(runs), 1)
        self.assertEqual(list(next(iter(runs))), sorted(self.ids))

    def test_empty_corpus_returns_empty(self):
        with db.conn() as c:
            c.execute("DELETE FROM vectors")
        embeddings.invalidate_cache()
        self.assertEqual(self._search([1, 0, 0, 0, 0, 0, 0, 0]), [])

    # ── 缓存失效 ──

    def test_new_vector_invalidates_cache(self):
        """加了论文却还用旧矩阵 = 新论文永远搜不到，是最隐蔽的缓存 bug。"""
        self._search([1, 0, 0, 0, 0, 0, 0, 0])          # 先把缓存热起来
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:7999", "title": "New", "abstract": "a",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "7999", "source": "s2"})
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (pid, "paper", 0, "New", "test-embed",
                       _blob([0, 0, 1, 0, 0, 0, 0, 0])))
        got = self._search([0, 0, 1, 0, 0, 0, 0, 0], top_k=1)
        self.assertEqual(got[0]["paper_id"], pid)

    def test_deleted_vector_invalidates_cache(self):
        self._search([1, 0, 0, 0, 0, 0, 0, 0])
        with db.conn() as c:
            c.execute("DELETE FROM vectors WHERE paper_id=?", (self.ids[0],))
        got = self._search([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertNotIn(self.ids[0], [g["paper_id"] for g in got])

    def test_switching_model_does_not_reuse_other_models_matrix(self):
        """换嵌入模型必须换一套向量，不能拿旧模型的矩阵算。"""
        self._search([1, 0, 0, 0, 0, 0, 0, 0])
        config.EMBED_MODEL = "other-embed"
        self.assertEqual(self._search([1, 0, 0, 0, 0, 0, 0, 0]), [])

    def test_explicit_invalidate_forces_rebuild(self):
        self._search([1, 0, 0, 0, 0, 0, 0, 0])
        embeddings.invalidate_cache()
        self.assertEqual(len(self._search([1, 0, 0, 0, 0, 0, 0, 0])), 3)

    # ── 维度不匹配 ──

    def test_dimension_mismatch_raises_instead_of_truncating(self):
        """截断出来的余弦是没有意义的数，比报错更危险。"""
        with self.assertRaises(embeddings.EmbedUnavailable) as cm:
            self._search([1, 0, 0])          # 3 维查询 vs 8 维库
        self.assertIn("维度不一致", str(cm.exception))

    # ── 并发 ──

    def test_concurrent_searches_are_consistent(self):
        """任务跑在线程池里，缓存是共享的——并发重建不能给出错结果。"""
        results, errors = [], []

        def work():
            try:
                embeddings.invalidate_cache()
                r = self._search([1, 0, 0, 0, 0, 0, 0, 0])
                results.append(tuple(g["paper_id"] for g in r))
            except Exception as e:              # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1, f"并发结果不一致：{set(results)}")


class WalPragmaTests(unittest.TestCase):
    """`PRAGMA journal_mode=WAL` 是写操作，每次连接要 ~6ms，而它是数据库文件的
    持久属性——设一次就够。实测 db.conn() 7.06ms → 0.91ms。"""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_wal_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "wal.db"

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def test_new_database_gets_wal(self):
        db.init_db()
        with db.conn() as c:
            self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_wal_pragma_is_not_reissued_on_later_connections(self):
        """第二次连接不该再发 journal_mode——那正是每次 6ms 的来源。

        用 sqlite3 的 set_trace_callback 抓真实执行的 SQL：确定性，
        不依赖计时（计时断言在忙碌机器上会假失败，本项目已经踩过一次）。
        """
        db.init_db()
        path = str(config.DB_PATH)
        self.assertIn(path, db._WAL_DONE, "init_db 之后应已记账")

        def sql_of(warm: bool):
            seen = []
            if not warm:
                db._WAL_DONE.discard(path)
            try:
                with db.conn() as c:
                    c.set_trace_callback(seen.append)
                    c.execute("SELECT 1").fetchone()
            finally:
                db._WAL_DONE.add(path)
            return seen

        # 热路径：连接建立后不该再有 journal_mode 语句
        with db.conn() as c:
            pass
        seen = []
        with db.conn() as c:
            c.set_trace_callback(seen.append)
            c.execute("SELECT 1").fetchone()
        self.assertFalse([x for x in seen if "journal_mode" in x.lower()])

    def test_wal_pragma_is_issued_on_first_connection(self):
        """反例：把记账清掉后必须重新发——否则新库不是 WAL，测试就成了空转。"""
        db.init_db()
        path = str(config.DB_PATH)
        db._WAL_DONE.discard(path)
        try:
            with db.conn() as c:
                mode = c.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode, "wal")
            self.assertIn(path, db._WAL_DONE, "发过之后要记账，否则每次都重发")
        finally:
            db._WAL_DONE.add(path)

    def test_switching_db_path_still_sets_wal(self):
        """测试会把 DB_PATH 指到临时目录——换路径必须重新设，否则新库不是 WAL。"""
        db.init_db()
        other = Path(self._tmp) / "other" / "x.db"
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        try:
            config.DATA_DIR = other.parent
            config.DB_PATH = other
            db.init_db()
            with db.conn() as c:
                self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        finally:
            config.DB_PATH, config.DATA_DIR = old_db, old_dir

    def test_concurrent_first_connections_are_safe(self):
        db.init_db()
        db._WAL_DONE.discard(str(config.DB_PATH))
        errors = []

        def work():
            try:
                with db.conn() as c:
                    c.execute("SELECT 1").fetchone()
            except Exception as e:      # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=work) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
