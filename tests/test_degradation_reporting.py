"""降级信号必须**传得出去**：从 search_hybrid 一路到用户和库里。

审计确认的形态（这些用例逐条钉住它）：

1. `search_hybrid` 里两段写得很细的降级文案算完就丢——函数只返回 `(ids, mode)`，
   `degrade` 变量全仓没有任何读取方，是死代码。
2. 调用方只能从 mode 反推，判的是 `mode == "fts"`；而有 chunk 命中时 mode 被拼成
   `"fts+chunks"`，等号恒假 → **向量整路挂掉时一句提示都不会出现**。
   默认配置 `PAPERNEST_CHUNK_SEARCH=1` 且库里有全文时，这就是最常见的那条路径。
3. health 探针只查向量总条数与模型名，不查覆盖率——审计时真实库 104/503（20.7%）
   的论文能被语义检索到，而四项检查全绿。

所以这里的断言重心不是「函数返回值长什么样」，而是「**在旧代码上必然为 None 的地方，
现在必须有值**」。
"""
from __future__ import annotations

import json
import sqlite3
import unittest
from unittest import mock

from papernest import chat, config, db, degrade, embeddings, health, rag, tools

try:
    from .support import TempDbTestCase
except ImportError:                          # noqa: F401
    from support import TempDbTestCase

MODEL = "test-embed"


class _Base(TempDbTestCase):
    prefix = "papernest_degrade"

    def setUp(self):
        super().setUp()
        self._old_model = config.EMBED_MODEL
        config.EMBED_MODEL = MODEL
        self.addCleanup(self._restore)
        embeddings.invalidate_cache()
        self.addCleanup(embeddings.invalidate_cache)
        db.init_db()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:6001", "title": "Quantization for Transformers",
                "abstract": "We study quantization.", "year": 2023, "venue": "V",
                "authors": [], "doi": None, "arxiv_id": "6001", "source": "s2"})
            # 有全文 → chunks 路会命中 → mode 会被拼成 "...+chunks"，
            # 这正是让旧的 `mode == "fts"` 判断失效的那个条件
            db.replace_chunks(c, self.pid, [
                {"text": "Quantization reduces memory for transformer inference.",
                 "section_path": "3 Method"}])
            c.commit()

    def _restore(self):
        config.EMBED_MODEL = self._old_model

    def _vector_broken(self):
        """向量路配了但整条挂掉——审计里那条真实故障的形态。"""
        return mock.patch.multiple(
            "papernest.embeddings",
            available=mock.DEFAULT, search_papers=mock.DEFAULT)


class SearchHybridReportsDegradation(_Base):

    def test_vector_failure_is_reported_even_when_chunks_hit(self):
        """**核心回归**：mode 是 'fts+chunks' 时，降级不能被吞掉。

        旧代码：调用方判 `mode == "fts"` → 恒假 → degraded 为 None，全程无提示。
        """
        with mock.patch("papernest.embeddings.available", return_value=True), \
             mock.patch("papernest.embeddings.search_papers",
                        side_effect=RuntimeError("connection reset")):
            res = embeddings.search_hybrid("quantization", 5)

        self.assertIn("chunks", res.mode,
                      "前置条件：本用例要的就是 mode 被拼上 '+chunks' 的那种情况")
        self.assertTrue(res.degraded,
                        f"向量整路挂掉却没有任何降级记录（mode={res.mode!r}）"
                        "——这正是旧代码静默吞掉信号的地方")
        self.assertEqual(res.degraded[0].code, degrade.VECTOR_CALL_FAILED)
        self.assertTrue(res.degraded[0].critical)

    def test_empty_vector_index_is_reported(self):
        """配了 EMBED_MODEL 却一条向量都匹配不上（典型成因：模型名填成了显示名）。"""
        with mock.patch("papernest.embeddings.available", return_value=True), \
             mock.patch("papernest.embeddings.search_papers", return_value=[]):
            res = embeddings.search_hybrid("quantization", 5)
        self.assertEqual([n.code for n in res.degraded],
                         [degrade.VECTOR_INDEX_EMPTY])
        self.assertIn("检查模型名", res.degraded[0].message)

    def test_not_configuring_vectors_is_not_a_degradation(self):
        """明确选择不用向量 ≠ 以为在用其实没用上。前者不该报降级。"""
        with mock.patch("papernest.embeddings.available", return_value=False):
            res = embeddings.search_hybrid("quantization", 5)
        self.assertEqual(res.degraded, [])

    def test_healthy_vector_path_reports_nothing(self):
        with mock.patch("papernest.embeddings.available", return_value=True), \
             mock.patch("papernest.embeddings.search_papers",
                        return_value=[{"paper_id": self.pid, "score": 0.9}]):
            res = embeddings.search_hybrid("quantization", 5)
        self.assertEqual(res.degraded, [])
        self.assertTrue(res.mode.startswith("hybrid"))


class CallersSurfaceDegradation(_Base):

    def _broken(self):
        return (mock.patch("papernest.embeddings.available", return_value=True),
                mock.patch("papernest.embeddings.search_papers",
                           side_effect=RuntimeError("boom")))

    def test_rag_prepare_surfaces_it(self):
        a, b = self._broken()
        with a, b:
            ctx = rag.prepare([{"role": "user", "content": "quantization"}], top_k=3)
        self.assertIsNotNone(ctx.degraded, "rag.prepare 把检索降级吞掉了")
        self.assertIn(degrade.VECTOR_CALL_FAILED, [n.code for n in ctx.notes])

    def test_tools_retrieve_library_surfaces_it(self):
        a, b = self._broken()
        with a, b:
            out = tools.retrieve_library("quantization", top_k=3)
        self.assertIsNotNone(out["degraded"], "tools.retrieve_library 把降级吞掉了")
        self.assertIn(degrade.VECTOR_CALL_FAILED,
                      [d["code"] for d in out["degraded_detail"]])

    def test_rendered_text_matches_structured_records(self):
        """给人看的那句中文与结构化记录必须同源，不能各说各的。"""
        a, b = self._broken()
        with a, b:
            ctx = rag.prepare([{"role": "user", "content": "quantization"}], top_k=3)
        self.assertEqual(ctx.degraded, degrade.render(ctx.notes))


class DegradationPersistence(_Base):

    def test_append_and_history_round_trip_structured_form(self):
        sid = chat.ensure_session(None, "q")
        notes = [degrade.Degradation(degrade.VECTOR_CALL_FAILED, "向量挂了"),
                 degrade.Degradation(degrade.CONTEXT_OVER_BUDGET, "2 篇超预算")]
        chat.append(sid, "assistant", "答案", [], degrade.render(notes), notes)

        msgs = chat.history(sid, 10)
        self.assertEqual(msgs[-1]["degraded"], "向量挂了；2 篇超预算")
        self.assertEqual([d["code"] for d in msgs[-1]["degraded_detail"]],
                         [degrade.VECTOR_CALL_FAILED, degrade.CONTEXT_OVER_BUDGET])
        self.assertTrue(msgs[-1]["degraded_detail"][0]["critical"])

    def test_history_now_returns_degraded_at_all(self):
        """回归：degraded 一直在落库，但 history 从没把它读出来，会话回放里看不见。"""
        sid = chat.ensure_session(None, "q")
        chat.append(sid, "assistant", "答案", [], "向量挂了",
                    [degrade.Degradation(degrade.VECTOR_CALL_FAILED, "向量挂了")])
        self.assertIn("degraded", chat.history(sid, 10)[-1])

    def test_legacy_callers_do_not_lose_the_signal(self):
        """没传 notes 的老调用方，信号也不能整条丢——兜成 code='unknown'。"""
        sid = chat.ensure_session(None, "q")
        chat.append(sid, "assistant", "答案", [], "某种降级")
        detail = chat.history(sid, 10)[-1]["degraded_detail"]
        self.assertEqual([d["code"] for d in detail], ["unknown"])
        self.assertEqual(detail[0]["message"], "某种降级")

    def test_degraded_json_is_queryable_by_code(self):
        """结构化的意义就在这里：能按 code 聚合，中文串做不到。"""
        sid = chat.ensure_session(None, "q")
        for msg in ("a", "b"):
            chat.append(sid, "assistant", msg, [], "向量挂了",
                        [degrade.Degradation(degrade.VECTOR_CALL_FAILED, "向量挂了")])
        chat.append(sid, "assistant", "c", [], None, [])
        with db.conn() as c:
            n = c.execute(
                "SELECT COUNT(*) n FROM chat_messages WHERE degraded_json LIKE ?",
                (f'%"{degrade.VECTOR_CALL_FAILED}"%',)).fetchone()["n"]
        self.assertEqual(n, 2)


class HealthCoverageGate(_Base):
    """探针必须照出「条数正常、覆盖率极低」这种退化——审计时它是全绿的。"""

    def _add_papers(self, n, with_vector):
        import numpy as np
        blob = embeddings._to_blob(list(np.zeros(4, dtype=np.float32)))
        with db.conn() as c:
            for i in range(n):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:70{i:03d}", "title": f"P{i}", "abstract": "a",
                    "year": 2024, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": f"70{i:03d}", "source": "s2"})
                if i < with_vector:
                    c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec)"
                              " VALUES(?,?,?,?,?,?)",
                              (pid, "paper", 0, "t", MODEL, blob))
            c.commit()

    def _index_check(self):
        config.EMBED_API_BASE = "https://example.invalid/v1"
        counts, stored, dim = health._index_stats()
        return health._check_index(counts, stored, dim)

    def test_low_vector_coverage_fails_the_index_check(self):
        self._add_papers(20, with_vector=4)          # 20% 覆盖，与审计时真库同量级
        chk = self._index_check()
        self.assertIs(chk["ok"], False,
                      "覆盖率 20% 却报 ok——这正是审计时探针全绿的原因")
        self.assertIn("进得了向量索引", chk["detail"])
        self.assertIn("cli.py embed", chk["detail"])

    def test_full_coverage_passes(self):
        self._add_papers(20, with_vector=20)
        chk = self._index_check()
        self.assertIs(chk["ok"], True)
        self.assertIn("向量覆盖", chk["detail"])

    def test_chunk_only_papers_count_as_covered(self):
        """口径必须与 _paper_matrix 一致：只有 chunk 向量的论文也是能被检索到的。"""
        covered, total = health._coverage()
        self.assertEqual((covered, total), (0, 1),
                         "前置条件：setUp 那篇有 chunks 但还没有 chunk 向量")
        import numpy as np
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec)"
                      " VALUES(?,?,?,?,?,?)",
                      (self.pid, "chunk", 0, "t", MODEL,
                       embeddings._to_blob(list(np.zeros(4, dtype=np.float32)))))
            c.commit()
        self.assertEqual(health._coverage(), (1, 1))


class DegradeVocabulary(unittest.TestCase):

    def test_merge_dedups_by_code_and_keeps_order(self):
        a = [degrade.Degradation(degrade.VECTOR_CALL_FAILED, "第一次")]
        b = [degrade.Degradation(degrade.VECTOR_CALL_FAILED, "又一次"),
             degrade.Degradation(degrade.CONTEXT_OVER_BUDGET, "超预算")]
        merged = degrade.merge(a, b)
        self.assertEqual([n.code for n in merged],
                         [degrade.VECTOR_CALL_FAILED, degrade.CONTEXT_OVER_BUDGET])
        self.assertEqual(merged[0].message, "第一次", "去重应保留先出现的那条")

    def test_render_is_none_when_empty(self):
        self.assertIsNone(degrade.render([]))
        self.assertIsNone(degrade.render(None))

    def test_dict_round_trip(self):
        notes = [degrade.Degradation(degrade.PAGE_PICK_FALLBACK, "退回关键词选页")]
        self.assertEqual(degrade.from_dicts(degrade.as_dicts(notes)), notes)

    def test_only_answer_invalidating_codes_are_critical(self):
        """CRITICAL 是闸门口径，不能随手加——「少了一篇上下文」不该和「向量挂了」同权。"""
        self.assertTrue(degrade.Degradation(degrade.VECTOR_INDEX_EMPTY, "x").critical)
        self.assertFalse(degrade.Degradation(degrade.CONTEXT_OVER_BUDGET, "x").critical)
        self.assertFalse(degrade.Degradation(degrade.QUERY_REWRITTEN, "x").critical)


class MigrationV6(unittest.TestCase):

    def test_old_db_gains_the_column_without_touching_existing_rows(self):
        """纯增量迁移：老库补一列，已有的中文串原样留着。"""
        import shutil
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="papernest_mig6_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        self.addCleanup(lambda: (setattr(config, "DB_PATH", old_db),
                                 setattr(config, "DATA_DIR", old_dir)))
        config.DATA_DIR = tmp
        config.DB_PATH = tmp / "old.db"

        # 造一个 v5 的库：没有 degraded_json 列
        raw = sqlite3.connect(config.DB_PATH)
        raw.executescript("""
            CREATE TABLE chat_sessions(id TEXT PRIMARY KEY, title TEXT,
              source_map_json TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE chat_messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL, role TEXT NOT NULL,
              content TEXT NOT NULL DEFAULT '',
              sources_json TEXT NOT NULL DEFAULT '[]', degraded TEXT,
              created_at TEXT);
            INSERT INTO chat_sessions(id,title) VALUES('s1','t');
            INSERT INTO chat_messages(session_id,role,content,degraded)
              VALUES('s1','assistant','旧答案','旧的降级说明');
            PRAGMA user_version=5;
        """)
        raw.commit()
        raw.close()

        db.init_db(force=True)

        with db.conn() as c:
            self.assertIn("degraded_json",
                          {r["name"] for r in
                           c.execute("PRAGMA table_info(chat_messages)")})
            row = c.execute("SELECT degraded, degraded_json FROM chat_messages"
                            " WHERE session_id='s1'").fetchone()
        self.assertEqual(row["degraded"], "旧的降级说明", "历史行被改动了")
        self.assertEqual(json.loads(row["degraded_json"]), [],
                         "历史行不该被猜出来的 code 回填污染")


if __name__ == "__main__":
    unittest.main()
