"""并发与阻塞：三处只有真并发才会暴露的问题。

审计确认、且都在真环境下复现过：

1. **`chat.assign_indices` 的读-改-写竞态**。默认隔离级别下 SELECT 走 autocommit，
   两个并发请求读到同一份旧 mapping、各自从同一个 `nxt` 起编号再互相覆盖。
   实测 24 线程并发：**24 篇只拿到 5 个不同编号、映射表丢掉 19 篇**，
   而这个函数存在的全部意义就是「同一个 [n] 始终指向同一篇论文」。
2. **`upload_pdfs` / `import_bibliography_file` 是 `async def` 却跑同步 IO**，
   把 PyMuPDF 解析、SQLite 写、带 sleep 退避的 LLM 调用搬到了事件循环上。
   对照实验（同一个阻塞函数）：`async def` 期间只处理 2 次请求、最大延迟 6.4s；
   改成 `def` 后处理 7 次、最大延迟 1.4s。
3. **agent 的「硬截止」只截断了调用方**。Python 中断不了已开跑的线程，
   被放弃的线程继续跑完整个工具。现在改成协作式取消（`deadline`），
   长耗时组件在重试/退避之间自行停手。

`tests/` 此前 14k 行里 `threading` 零出现——并发问题一条用例都没有。
"""
from __future__ import annotations

import inspect
import io
import json
import threading
import time
import unittest
from collections import Counter
from unittest import mock

from papernest import api, chat, db, deadline

try:
    from .support import TempDbTestCase
except ImportError:                          # noqa: F401
    from support import TempDbTestCase


class AssignIndicesIsAtomic(TempDbTestCase):
    prefix = "papernest_race"

    def setUp(self):
        super().setUp()
        db.init_db()
        self.sid = chat.ensure_session(None, "race")

    def test_concurrent_callers_never_share_an_index(self):
        """N 篇不同论文并发要编号 → 必须拿到 N 个互不相同的编号。"""
        N = 16
        got, errors = {}, []
        barrier = threading.Barrier(N)

        def worker(pid):
            try:
                barrier.wait(timeout=10)     # 让读尽量发生在同一瞬间
                got[pid] = chat.assign_indices(self.sid, [pid])[pid]
            except Exception as e:           # noqa: BLE001
                errors.append(repr(e))

        ths = [threading.Thread(target=worker, args=(9000 + i,)) for i in range(N)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=60)

        self.assertEqual(errors, [], f"并发调用抛异常：{errors[:3]}")
        dupes = {i: n for i, n in Counter(got.values()).items() if n > 1}
        self.assertFalse(dupes, f"编号重复——同一个 [n] 指向多篇论文：{dupes}")
        self.assertEqual(len(got), N)

    def test_no_paper_is_lost_from_the_stored_map(self):
        """丢更新的另一面：映射表里少了论文，后续回指就查不到它。"""
        N = 16
        barrier = threading.Barrier(N)

        def worker(pid):
            barrier.wait(timeout=10)
            chat.assign_indices(self.sid, [pid])

        ths = [threading.Thread(target=worker, args=(9100 + i,)) for i in range(N)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=60)

        with db.conn() as c:
            row = c.execute("SELECT source_map_json FROM chat_sessions WHERE id=?",
                            (self.sid,)).fetchone()
        stored = json.loads(row["source_map_json"] or "{}")
        self.assertEqual(len(stored), N, f"映射表丢了论文：只存下 {len(stored)}/{N}")

    def test_repeated_calls_are_stable(self):
        """同一篇论文反复要编号，必须始终是同一个号（本函数的立身之本）。"""
        first = chat.assign_indices(self.sid, [1, 2, 3])
        for _ in range(5):
            self.assertEqual(chat.assign_indices(self.sid, [1, 2, 3]), first)


class ImmediateTransaction(TempDbTestCase):
    prefix = "papernest_immediate"

    def setUp(self):
        super().setUp()
        db.init_db()

    def test_immediate_takes_a_write_lock_up_front(self):
        """`BEGIN IMMEDIATE` 之后另一个写事务必须等锁，而不是并行读到旧值。"""
        started = threading.Event()
        second_done = threading.Event()

        def holder():
            with db.conn(immediate=True) as c:
                c.execute("INSERT INTO chat_sessions(id,title) VALUES('a','x')")
                started.set()
                time.sleep(0.6)

        def second():
            started.wait(timeout=5)
            with db.conn(immediate=True) as c:
                c.execute("INSERT INTO chat_sessions(id,title) VALUES('b','y')")
            second_done.set()

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=second)
        t1.start(); t2.start()
        # 第一个还持锁时，第二个不该已经完成
        started.wait(timeout=5)
        self.assertFalse(second_done.wait(timeout=0.2),
                         "第二个写事务没有等锁，BEGIN IMMEDIATE 没生效")
        t1.join(timeout=10); t2.join(timeout=10)
        self.assertTrue(second_done.is_set(), "等锁之后应当能完成")

    def test_default_conn_still_works_unchanged(self):
        with db.conn() as c:
            c.execute("INSERT INTO chat_sessions(id,title) VALUES('z','t')")
        with db.conn() as c:
            self.assertEqual(
                c.execute("SELECT COUNT(*) n FROM chat_sessions").fetchone()["n"], 1)

    def test_rollback_on_error(self):
        with self.assertRaises(ValueError):
            with db.conn(immediate=True) as c:
                c.execute("INSERT INTO chat_sessions(id,title) VALUES('r','t')")
                raise ValueError("boom")
        with db.conn() as c:
            self.assertEqual(
                c.execute("SELECT COUNT(*) n FROM chat_sessions").fetchone()["n"], 0)


class EndpointsDoNotBlockTheEventLoop(unittest.TestCase):
    """会做同步 IO 的端点必须是 `def`，让 Starlette 丢线程池。"""

    def test_upload_and_import_are_sync(self):
        for name in ("upload_pdfs", "import_bibliography_file"):
            fn = getattr(api, name)
            self.assertFalse(
                inspect.iscoroutinefunction(fn),
                f"{name} 是 async def，函数体里的 PyMuPDF 解析 / SQLite 写 / LLM 调用"
                f"会跑在事件循环上，一次批量上传就能占死整个服务")

    def test_upload_bodies_contain_no_await(self):
        """按 AST 判，不按字符串判——注释里出现 `await f.read()`（在解释旧写法）
        不该让这条用例变红，而真代码里出现必须变红。"""
        import ast
        with io.open(api.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        wanted = {"upload_pdfs", "import_bibliography_file"}
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in wanted:
                found.add(node.name)
                self.assertNotIsInstance(
                    node, ast.AsyncFunctionDef, f"{node.name} 仍是 async def")
                awaits = [n for n in ast.walk(node) if isinstance(n, ast.Await)]
                self.assertEqual(awaits, [], f"{node.name} 里仍有 await")
        self.assertEqual(found, wanted, f"没找到端点：{wanted - found}")


class ReadCapped(unittest.TestCase):
    """上限检查必须发生在**读进内存之前**，否则防线形同虚设。"""

    def test_returns_none_past_the_limit(self):
        self.assertIsNone(api._read_capped(io.BytesIO(b"x" * 5000), 1000))

    def test_returns_content_under_the_limit(self):
        self.assertEqual(api._read_capped(io.BytesIO(b"abc"), 1000), b"abc")

    def test_does_not_read_the_whole_oversized_stream(self):
        """峰值内存应是 limit + chunk，而不是文件大小。"""
        class Counting(io.BytesIO):
            def __init__(self, n):
                super().__init__(b"x" * n)
                self.read_bytes = 0

            def read(self, n=-1):
                b = super().read(n)
                self.read_bytes += len(b)
                return b

        fp = Counting(20 << 20)                       # 20MB
        self.assertIsNone(api._read_capped(fp, 1 << 20, chunk=1 << 16))
        self.assertLess(fp.read_bytes, 4 << 20, "把整个超限文件都读进来了")

    def test_empty_stream(self):
        self.assertEqual(api._read_capped(io.BytesIO(b""), 100), b"")


class DeadlineSemantics(unittest.TestCase):

    def tearDown(self):
        deadline.clear()

    def test_no_deadline_means_unlimited(self):
        deadline.clear()
        self.assertEqual(deadline.remaining(), float("inf"))
        self.assertFalse(deadline.expired())
        deadline.check()                     # 不该抛

    def test_expires_and_raises(self):
        deadline.set(0.01)
        time.sleep(0.05)
        self.assertTrue(deadline.expired())
        with self.assertRaises(deadline.DeadlineExceeded):
            deadline.check()

    def test_sleep_never_overshoots_the_deadline(self):
        """回归：退避原来是裸 time.sleep(80)——步骤早超时了还要再睡 80 秒。"""
        deadline.set(0.2)
        t0 = time.monotonic()
        with self.assertRaises(deadline.DeadlineExceeded):
            deadline.sleep(30)
        self.assertLess(time.monotonic() - t0, 2.0, "睡过了截止时间")

    def test_scope_restores_the_previous_budget(self):
        deadline.set(100)
        outer = deadline.at()
        with deadline.scope(5):
            self.assertLess(deadline.remaining(), 6)
        self.assertEqual(deadline.at(), outer)

    def test_budget_is_per_thread(self):
        """预算必须是线程本地的，否则一个请求的截止会影响到别的请求。"""
        seen = {}

        def other():
            seen["remaining"] = deadline.remaining()

        deadline.set(5)
        t = threading.Thread(target=other)
        t.start(); t.join()
        self.assertEqual(seen["remaining"], float("inf"),
                         "预算泄漏到了别的线程")


class LlmStopsRetryingOnDeadline(unittest.TestCase):
    """时间大头在重试与退避（最坏 433s 里 73s 是纯 sleep），预算必须能截断它。"""

    def tearDown(self):
        deadline.clear()

    def _run(self, budget):
        import httpx
        from papernest import config, llm
        calls = {"n": 0}

        def fail(*a, **kw):
            calls["n"] += 1
            raise httpx.ConnectError("boom")

        fake = mock.MagicMock()
        fake.__enter__.return_value.post = fail
        fake.__exit__.return_value = False
        with mock.patch.object(config, "LLM_API_BASE", "https://x.invalid/v1"), \
             mock.patch.object(config, "LLM_API_KEY", "k"), \
             mock.patch.object(config, "LLM_MODEL", "m"), \
             mock.patch.object(llm.http, "client", return_value=fake), \
             mock.patch.object(llm.db, "conn"), \
             deadline.scope(budget):
            t0 = time.monotonic()
            with self.assertRaises(Exception) as cm:
                llm.chat("s", "u", purpose="t")
            return time.monotonic() - t0, calls["n"], cm.exception

    def test_short_budget_stops_after_one_attempt(self):
        el, n, exc = self._run(0.05)
        self.assertIsInstance(exc, deadline.DeadlineExceeded)
        self.assertEqual(n, 1, f"预算耗尽后还发了 {n} 次请求")
        self.assertLess(el, 3, f"耗时 {el:.1f}s——退避没有被预算截断")


class AgentPushesDeadlineToTheWorker(unittest.TestCase):

    def test_tool_runs_with_a_budget(self):
        """回归：原来工具线程拿不到任何预算，超时后一路跑到底。"""
        from papernest import agent, tools
        seen = {}

        def probe(**kw):
            seen["remaining"] = deadline.remaining()
            return {"paper_ids": []}

        orig = tools.TOOL_REGISTRY["retrieve_library"]
        tools.TOOL_REGISTRY["retrieve_library"] = probe
        try:
            agent.run_agent("测试", timeout_s=7, persist=False, max_steps=1)
        finally:
            tools.TOOL_REGISTRY["retrieve_library"] = orig
        self.assertIn("remaining", seen, "工具没有被调用")
        self.assertNotEqual(seen["remaining"], float("inf"),
                            "工具线程里没有预算，超时形同虚设")
        self.assertLessEqual(seen["remaining"], 7.0)


if __name__ == "__main__":
    unittest.main()
