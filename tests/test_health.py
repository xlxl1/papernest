# -*- coding: utf-8 -*-
"""检索健康探针：把「静默退化成纯 FTS」变成一条会报错的断言。

这是项目里唯一一个「失效会让所有数字变差、却不会让任何测试变红」的地方——
EMBED_MODEL 填成控制台的中文显示名而非 API 真实 ID，检索匹配不到任何向量、
悄悄退回纯 FTS，Recall@5 长期停在 0.6771（修好配置后 0.9427）。
这些用例把那次失效的形态钉住，让它下次撞上门禁。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, embeddings, health


class HealthProbeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_health_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL,
                     config.EMBED_API_BASE)
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()

    def tearDown(self):
        (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL,
         config.EMBED_API_BASE) = self._old

    def _store_vector(self, model: str, dim: int = 4):
        import numpy as np
        blob = embeddings._to_blob(list(np.zeros(dim, dtype=np.float32)))
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)", (1, "paper", 0, "t", model, blob))

    def _named(self, r, name):
        return next(c for c in r["checks"] if c["name"] == name)

    # ── 那次真实失效的形态 ──

    def test_model_name_mismatch_is_caught(self):
        """配置里的模型名与建索引时用的对不上——检索会匹配不到任何向量、静默退回 FTS。"""
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        self._store_vector("通用文本向量-v3")          # 控制台显示名，不是 API 真实 ID
        r = health.probe(live=False)
        self.assertFalse(r["ok"])
        idx = self._named(r, "索引")
        self.assertIs(idx["ok"], False)
        self.assertIn("不一致", idx["detail"])

    def test_configured_but_empty_index_is_a_failure(self):
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        r = health.probe(live=False)
        self.assertFalse(r["ok"])
        self.assertIs(self._named(r, "索引")["ok"], False)

    def test_model_missing_api_base_is_a_failure(self):
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = ""
        self._store_vector("text-embedding-v3")
        r = health.probe(live=False)
        self.assertFalse(r["ok"])
        self.assertIs(self._named(r, "配置")["ok"], False)

    # ── 「明确不用向量」不是故障 ──

    def test_deliberately_no_vectors_is_not_a_failure(self):
        """『明确选择不用向量』和『以为在用其实没用上』必须区分开——这是探针存在的理由。"""
        config.EMBED_MODEL = ""
        config.EMBED_API_BASE = ""
        r = health.probe(live=False)
        self.assertTrue(r["ok"])
        self.assertIsNone(self._named(r, "配置")["ok"])
        self.assertIsNone(self._named(r, "索引")["ok"])

    def test_healthy_config_and_index_pass(self):
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        self._store_vector("text-embedding-v3")
        r = health.probe(live=False)
        self.assertTrue(r["ok"], r["checks"])
        self.assertIs(self._named(r, "索引")["ok"], True)

    # ── 契约 ──

    def test_offline_makes_no_network_calls(self):
        """--offline 的全部意义就是 0 次网络调用；检索检查内部会 embed，必须一起跳过。"""
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        self._store_vector("text-embedding-v3")
        boom = mock.Mock(side_effect=AssertionError("离线模式不该发起网络调用"))
        with mock.patch("papernest.embeddings.embed_texts", boom), \
             mock.patch("papernest.embeddings.search_hybrid", boom):
            r = health.probe(live=False)
        boom.assert_not_called()
        self.assertIsNone(self._named(r, "检索")["ok"])

    def test_probe_never_raises_even_when_retrieval_explodes(self):
        """探针自己不能成为新的故障点。"""
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        self._store_vector("text-embedding-v3")
        with mock.patch("papernest.embeddings.available", return_value=True), \
             mock.patch("papernest.embeddings.embed_texts",
                        side_effect=RuntimeError("connection reset")), \
             mock.patch("papernest.embeddings.search_hybrid",
                        side_effect=RuntimeError("connection reset")):
            r = health.probe(live=True)
        self.assertFalse(r["ok"])
        self.assertIs(self._named(r, "活体")["ok"], False)
        self.assertIs(self._named(r, "检索")["ok"], False)

    def test_silent_fts_degradation_is_caught(self):
        """配了向量、检索却走了纯 FTS——这正是那次失效在检索层的表征。"""
        config.EMBED_MODEL = "text-embedding-v3"
        config.EMBED_API_BASE = "https://example.invalid/v1"
        self._store_vector("text-embedding-v3")
        with mock.patch("papernest.embeddings.available", return_value=True), \
             mock.patch("papernest.embeddings.embed_texts", return_value=[[0.0] * 4]), \
             mock.patch("papernest.embeddings.search_hybrid",
                        return_value=embeddings.RetrievalResult([1, 2], "fts", [])):
            r = health.probe(live=True)
        self.assertFalse(r["ok"])
        self.assertIn("没生效", self._named(r, "检索")["detail"])


if __name__ == "__main__":
    unittest.main()
