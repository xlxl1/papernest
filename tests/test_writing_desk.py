import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from papernest import config, db, writing

SAMPLE = "本文提出了一种新的信道估计方法，效果很好。如 [1] 所示，MSE 下降了 30%。"


class WritingDeskOfflineTests(unittest.TestCase):
    """写作台润色/评审的离线路径：不调模型，结构必须稳定可用。"""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_wt_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "test.db"
        db.init_db()
        with db.conn() as c:
            db.insert_l0(c, {"norm_key": "arxiv:2501.00003", "title": "MIMO Channel Estimation",
                             "abstract": "channel estimation study", "year": 2024,
                             "venue": "", "authors": [], "doi": None,
                             "arxiv_id": "2501.00003", "source": "s2"})
            c.commit()

    def tearDown(self):
        config.DB_PATH = self._old_db

    def test_polish_offline_returns_original_with_degraded(self):
        with mock.patch.object(writing.llm, "available", return_value=False):
            r = writing.polish_text(SAMPLE)
        self.assertEqual(r["revised"], SAMPLE)
        self.assertEqual(r["model"], "offline")
        self.assertIn("未配置 LLM", r["degraded"])

    def test_review_offline_returns_checklist(self):
        with mock.patch.object(writing.llm, "available", return_value=False):
            r = writing.review_text(SAMPLE)
        self.assertIsNone(r["score"])
        self.assertEqual(r["issues"], [])
        self.assertTrue(r["checklist"])
        self.assertIn("未配置 LLM", r["degraded"])

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            writing.polish_text("   ")
        with self.assertRaises(ValueError):
            writing.review_text("")


class PolishPromptContractTests(unittest.TestCase):
    def test_polish_system_requires_json_and_no_new_facts(self):
        self.assertIn("JSON", writing.POLISH_SYSTEM)
        self.assertIn("保留原文事实", writing.POLISH_SYSTEM)
        self.assertIn("severity", writing.REVIEW_SYSTEM)
        self.assertIn("只输出 JSON", writing.REVIEW_SYSTEM)


class HybridSearchTests(unittest.TestCase):
    """RRF 混合检索：向量与 FTS 两路融合，单路不可用时正确退化。"""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_hs_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "test.db"
        db.init_db()
        with db.conn() as c:
            for i, (key, title) in enumerate([
                ("arxiv:1", "Near-field channel estimation with deep learning"),
                ("arxiv:2", "Beam training codebook design for XL-MIMO"),
                ("arxiv:3", "Rehabilitation training efficacy study"),
            ]):
                pid = db.insert_l0(c, {"norm_key": key, "title": title,
                                       "abstract": f"abstract about {title.lower()}",
                                       "year": 2025, "venue": "", "authors": [],
                                       "doi": None, "arxiv_id": key.split(":")[1],
                                       "source": "s2"})
                c.commit()
                if i < 2:  # 第三篇不建向量：验证 FTS 路仍能召回它
                    c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) VALUES(?,?,?,?,?,?)",
                              (pid, "paper", 0, title, "test-model", b"\x00" * 4))
            c.commit()

    def tearDown(self):
        config.DB_PATH = self._old_db

    def test_hybrid_when_both_available(self):
        from papernest import embeddings
        with mock.patch.object(embeddings, "available", return_value=True), \
             mock.patch.object(embeddings, "search_papers",
                               return_value=[{"paper_id": 2, "score": 0.9},
                                             {"paper_id": 1, "score": 0.8}]):
            _r = embeddings.search_hybrid("near-field channel", 2)
            ids, mode = _r.ids, _r.mode
        self.assertEqual(mode, "hybrid")
        # 论文 1 在向量路排第 2、FTS 路第 1；论文 2 在向量路第 1、FTS 路未命中
        # RRF 下 1 的两路加权应高于 2 → 融合把双路命中的排前
        self.assertEqual(ids[0], 1)

    def test_fts_only_when_embeddings_unconfigured(self):
        from papernest import embeddings
        with mock.patch.object(embeddings, "available", return_value=False):
            _r = embeddings.search_hybrid("near-field", 3)
            ids, mode = _r.ids, _r.mode
        self.assertEqual(mode, "fts")
        self.assertIn(1, ids)

    def test_fts_fallback_when_vector_call_fails(self):
        from papernest import embeddings
        with mock.patch.object(embeddings, "available", return_value=True), \
             mock.patch.object(embeddings, "search_papers", side_effect=RuntimeError("网络断了")):
            _r = embeddings.search_hybrid("near-field", 3)
            ids, mode = _r.ids, _r.mode
        self.assertEqual(mode, "fts")
        self.assertIn(1, ids)


class ChatStreamTests(unittest.TestCase):
    """llm.chat_stream 的 SSE 解析与落账：MockTransport 全程无网络。"""

    def test_stream_yields_chunks_and_records_usage(self):
        import json as _json
        import tempfile
        from papernest import db as db_mod
        from papernest import llm, http

        sse = (
            'data: ' + _json.dumps({"choices": [{"delta": {"content": "你好"}}]}) + "\n\n"
            'data: ' + _json.dumps({"choices": [{"delta": {"content": "！[1]"}}]}) + "\n\n"
            'data: ' + _json.dumps({"choices": [], "usage": {"prompt_tokens": 10,
                                                             "completion_tokens": 3}}) + "\n\n"
            "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(200, content=sse.encode())

        tmp = tempfile.mkdtemp(prefix="papernest_cs_")
        self.addCleanup(shutil.rmtree, tmp, True)
        old_db = config.DB_PATH
        config.DB_PATH = Path(tmp) / "t.db"
        db_mod.init_db()
        try:
            def fake_client(timeout=None, **kw):
                return httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)

            with mock.patch.object(llm.http, "client", fake_client):
                got = list(llm.chat_stream("sys", "user", purpose="test_stream"))
            self.assertEqual(got, ["你好", "！[1]"])
            with db_mod.conn() as c:
                row = c.execute("SELECT purpose, prompt_tokens, completion_tokens, latency_ms "
                                "FROM llm_calls WHERE purpose='test_stream'").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (10, 3))
            self.assertGreater(row["latency_ms"], 0)
        finally:
            config.DB_PATH = old_db

    def test_stream_400_without_usage_support_retries_clean(self):
        import json as _json
        import tempfile
        from papernest import db as db_mod
        from papernest import llm, http

        sse = ('data: ' + _json.dumps({"choices": [{"delta": {"content": "ok"}}]}) + "\n\n"
               "data: [DONE]\n\n")
        seen = {"stream_options": 0}

        def handler(request):
            body = _json.loads(request.content)
            if body.get("stream_options"):
                seen["stream_options"] += 1
                return httpx.Response(400, text="unknown field")
            return httpx.Response(200, content=sse.encode())

        tmp = tempfile.mkdtemp(prefix="papernest_cs2_")
        self.addCleanup(shutil.rmtree, tmp, True)
        old_db = config.DB_PATH
        config.DB_PATH = Path(tmp) / "t.db"
        db_mod.init_db()
        try:
            def fake_client(timeout=None, **kw):
                return httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)

            with mock.patch.object(llm.http, "client", fake_client):
                got = list(llm.chat_stream("sys", "user", purpose="test_stream2"))
            self.assertEqual(got, ["ok"])
            self.assertEqual(seen["stream_options"], 1)  # 首次带、被 400 后剔除重试
        finally:
            config.DB_PATH = old_db


if __name__ == "__main__":
    unittest.main()
