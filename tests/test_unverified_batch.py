# -*- coding: utf-8 -*-
"""原审计 41 条 unverified 里核出来的 5 条——它们从没被任何人验证过。

审计当时因为预算耗尽把 41 条留成了「未核验」。逐条对当前代码复核后，
下面这 5 条**是真的**（其余多数已被前两轮顺带修掉，或核不成立）：

① `jobs.create_job` 的去重闸是 check-then-insert，而连接默认 autocommit、
   `idx_jobs_active` 又不是 UNIQUE。**实测 16 线程并发建同一 dedupe_key 的任务
   建成了 2 个**——注释里说它要防的「前端连点两次开写让两个线程交错写同一批
   sections」从来没被真正防住。
② `/api/chat` 与 `/api/survey` 的 `top_k` 完全无上下界，而相邻的 `/api/answer/deep`
   有 `ge=1, le=20`。**SQLite 的 `LIMIT -1` 是「不限」**（实测 10 行的表返回 10 行），
   传 -1 会把全库拉回来拼进 prompt。
③ `deepsearch` 的 trace 报 `len(seen)`，而真正进上下文的是
   `min(len(seen), MAX_CONTEXT_PAPERS)`——top_k>12 时虚报。cli 与前端都读这个字段。
④ `cli.py rechunk` 会作废全部 chunk 向量，命令输出一个字都不提。
⑤ **只有 `ingest` 那条路会建向量**：上传 PDF / 导入 .bib / 多格式入库都不建，
   而且不吭声——真库 443 篇无向量就是这么攒出来的，而向量路权重 1.0，
   这些论文等于被系统性排在后面。
"""
import pathlib
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from papernest import api, config, db, jobs


class _TempDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_unver_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)
        db.init_db()

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def _paper(self, i):
        with db.conn() as c:
            return db.insert_l0(c, {
                "norm_key": "x:%d" % i, "title": "T", "abstract": "a", "year": 2024,
                "venue": "v", "authors": [], "doi": None, "arxiv_id": None,
                "source": "s2"})


class JobDedupeIsEnforcedByTheDbTests(_TempDb):
    def test_concurrent_create_yields_exactly_one_active_job(self):
        """16 线程抢同一个 dedupe_key：修复前建成 2 个。"""
        made, conflicts, other = [], [], []

        def go():
            try:
                made.append(jobs.create_job("write", {"x": 1}, dedupe_key="write:r1"))
            except jobs.JobConflict:
                conflicts.append(1)
            except Exception as exc:                    # noqa: BLE001
                other.append(type(exc).__name__)

        ts = [threading.Thread(target=go) for _ in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM jobs WHERE dedupe_key='write:r1' "
                          "AND status IN ('queued','running')").fetchone()["n"]
        self.assertEqual(n, 1, f"同键同时有 {n} 个任务在跑")
        self.assertEqual(len(made), 1)
        self.assertEqual(other, [],
                         f"应当是干净的 JobConflict，实际抛了 {other}——"
                         f"裸 IntegrityError 会以 500 冒到用户面前")

    def test_jobs_without_a_dedupe_key_still_coexist(self):
        """不带 dedupe_key 的任务互不相干（UNIQUE 索引里 NULL 互不相等）。"""
        ids = [jobs.create_job("read", {"i": i}) for i in range(5)]
        self.assertEqual(len(set(ids)), 5)

    def test_a_finished_job_frees_the_key(self):
        """部分唯一索引只约束在跑的任务——同一个 run 可以先后写很多次。"""
        first = jobs.create_job("write", {"x": 1}, dedupe_key="write:r2")
        with db.conn() as c:
            c.execute("UPDATE jobs SET status='done' WHERE id=?", (first,))
        second = jobs.create_job("write", {"x": 2}, dedupe_key="write:r2")
        self.assertNotEqual(first, second)


class TopKIsBoundedTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def test_negative_top_k_is_rejected(self):
        """SQLite 的 `LIMIT -1` 是「不限」——一次请求能把全库拉回来拼进 prompt。"""
        r = self.client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "x"}], "top_k": -1})
        self.assertEqual(r.status_code, 422)

    def test_absurd_top_k_is_rejected(self):
        """`survey._context_blocks` 没有字符预算，top_k 大到一定程度就拼出十几万字符。"""
        r = self.client.post("/api/survey", json={"topic": "x", "top_k": 1000})
        self.assertEqual(r.status_code, 422)

    def test_the_bounds_match_the_neighbouring_endpoint(self):
        """同一个参数在相邻端点上两套口径本身就是坑。"""
        chat = api.ChatBody.model_fields["top_k"]
        deep = api.DeepBody.model_fields["top_k"]
        def bound(f, kind):
            return next((getattr(m, kind) for m in f.metadata
                         if getattr(m, kind, None) is not None), None)
        self.assertEqual(bound(chat, "le"), bound(deep, "le"),
                         "/api/chat 与 /api/answer/deep 的 top_k 上界对不上")


class TraceReportsTheRealContextSizeTests(unittest.TestCase):
    def test_papers_in_context_is_capped(self):
        """`seen` 会超过 MAX_CONTEXT_PAPERS，而 `rag.prepare` 是按上限取的。"""
        import inspect
        import io
        import tokenize
        from papernest import deepsearch
        code = "".join(
            t.string for t in
            tokenize.generate_tokens(
                io.StringIO(inspect.getsource(deepsearch.deep_answer)).readline)
            if t.type != tokenize.COMMENT).replace(" ", "")
        self.assertIn('"papers_in_context":min(len(seen),MAX_CONTEXT_PAPERS)', code,
                      "trace 又开始报 len(seen) 了——top_k>12 时它和真实上下文对不上")


class ExpensiveActionsAnnounceTheirCostTests(_TempDb):
    def test_rechunk_warns_before_dropping_vectors(self):
        """`db.replace_chunks` 会作废整篇的 chunk 向量，命令必须先把账说清楚。"""
        src = (pathlib.Path(config.ROOT) / "cli.py").read_text(encoding="utf-8")
        i = src.index("def cmd_rechunk(")
        body = src[i:src.index("\ndef ", i + 10)]
        self.assertIn("作废", body, "rechunk 没有提示它会作废向量")
        self.assertIn("embed-chunks", body, "没告诉人怎么补回来")
        self.assertIn("migrate_chunks", body, "没指出更省的那条路")

    def test_import_paths_report_papers_left_without_vectors(self):
        """只有 ingest 会建向量；其余入库路径不建**是对的**（花钱要用户同意），
        但不说就不对——真库 443 篇无向量正是这么攒出来的。"""
        for i in range(3):
            self._paper(i)
        out = {}
        jobs._note_missing_vectors(out)
        self.assertEqual(out.get("papers_without_vectors"), 3)
        self.assertIn("cli.py embed", out.get("hint", ""))
        self.assertIn("费用", out.get("hint", ""), "没说明补建是要花钱的")

    def test_no_hint_when_everything_is_indexed(self):
        for i in range(2):
            pid = self._paper(i)
            with db.conn() as c:
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,'paper',0,'t',?,x'00')", (pid, config.EMBED_MODEL))
        out = {}
        jobs._note_missing_vectors(out)
        self.assertEqual(out, {}, "全都有向量时不该再唠叨")


if __name__ == "__main__":
    unittest.main()
