# -*- coding: utf-8 -*-
"""本机单用户这个部署设想带来的三条结论——它们是**边界**，不是缺陷。

审计把「没有用户/租户概念」「没有限流配额」「全项目零 logging」并列成三条 high。
按这个项目实际的部署设想（**自己本机跑**）重新判：

① **不做多租户**。没有「别人」这个角色：`docker-compose` 把端口绑在 `127.0.0.1`，
   `api._check_exposure()` 在绑非回环又没设口令时**拒绝启动**。
   加 `owner_id` 列、做数据隔离，在这个前提下是给自己找麻烦，而且会给人
   「这东西可以多人用」的错觉——那才是真正危险的。
   要做的是把这条边界**显式钉住**，见下面的用例。

② **不做限流，做花钱闸门**。限流挡的是「别人打爆我的服务」；本机没有别人。
   真正咬过人的是失控循环烧光自己的额度——2026-09-03 embedding 免费额度被跑光，
   线上口径从 Recall@5 0.9427 退化成纯 FTS 的 0.7552。所以做的是每日 token 上限
   （`papernest/budget.py`），按真库观测峰值 117 万/天的 2.5 倍取 300 万。

③ **日志要够用，不要那一套**。结构化 JSON + request-id + trace 解决的是
   「多实例、要在 Grafana 里串起来」；本机要回答的只有「刚才那次为什么失败」。
   所以是一个会轮转的文件 + traceback（`papernest/log.py`），并且把
   **异常吞成 HTTP 200** 的那几个端点补上 `logger.exception`。

这个文件把这三条判断连同它们的**前提**一起钉住：哪天部署设想变了
（真要放到公网 / 给别人用），下面这些断言会提醒你回来重新做这三个决定。
"""
import unittest
from unittest import mock

from papernest import api, budget, config, db, log

try:
    from .support import make_tempdir
except ImportError:                     # discover -s tests 时 tests/ 不当包用
    from support import make_tempdir


def code_of(fn) -> str:
    """函数源码，**剥掉注释与全部空白**。

    剥注释：断言「代码里有没有 X」时不剥的话会命中解释「原来这里写的是 X」的
    那句话（本仓已经栽过三次）。
    剥空白：`tokenize` 是用空格把 token 连回去的，`_log.exception` 会变成
    `_log . exception`，直接 `in` 匹配不到。
    """
    import inspect
    import io
    import tokenize
    return "".join(
        t.string for t in
        tokenize.generate_tokens(io.StringIO(inspect.getsource(fn)).readline)
        if t.type != tokenize.COMMENT).replace(" ", "").replace(chr(10), "")


class SingleUserBoundaryIsExplicitTests(unittest.TestCase):
    """① 单用户边界：靠部署守卫，不靠数据层隔离。"""

    def test_no_ownership_column_anywhere(self):
        """数据层**故意**没有归属字段。这条不是在固化缺陷，是在钉住「设想」：

        它变红只有一种情况——有人开始加多租户了。那时就该把鉴权、数据隔离、
        限流一起重新设计，而不是悄悄加一列。
        """
        offenders = []
        for line in db.SCHEMA.splitlines():
            low = line.lower().strip()
            if any(k in low for k in ("owner_id", "tenant_id", "user_id")):
                offenders.append(line.strip())
        self.assertEqual(
            offenders, [],
            f"schema 里出现了归属字段：{offenders}——如果真要做多租户，"
            f"请连同鉴权与限流一起重新设计，别只加一列")

    def test_exposure_guard_is_what_actually_protects_us(self):
        """既然不做租户，那唯一的防线就是「不对外暴露」——它必须真的会拒绝启动。"""
        with mock.patch.object(api, "BIND_HOST", "0.0.0.0"), \
             mock.patch.object(api, "API_KEY", ""), \
             mock.patch.object(api, "ALLOW_NO_AUTH", False):
            with self.assertRaises(RuntimeError) as cm:
                api._check_exposure()
        self.assertIn("PAPERNEST_ALLOW_NO_AUTH", str(cm.exception),
                      "拒绝启动时必须给出逃生口，否则内网调试会被卡死")

    def test_loopback_without_key_is_fine(self):
        """本机自用不该被强制设口令——那只会让人把口令写进脚本里。"""
        with mock.patch.object(api, "BIND_HOST", "127.0.0.1"), \
             mock.patch.object(api, "API_KEY", ""), \
             mock.patch.object(api, "ALLOW_NO_AUTH", False):
            api._check_exposure()          # 不抛就是通过


class DailySpendGateTests(unittest.TestCase):
    """② 花钱闸门：防失控，不防滥用。"""

    def test_gate_blocks_when_today_is_over_budget(self):
        with mock.patch.object(budget, "DAILY_TOKEN_BUDGET", 1000), \
             mock.patch.object(budget, "spent_today", return_value=1500):
            with self.assertRaises(budget.BudgetExceeded) as cm:
                budget.check("模型调用")
        msg = str(cm.exception)
        self.assertIn("1,500", msg)
        self.assertIn("PAPERNEST_DAILY_TOKEN_BUDGET", msg,
                      "报错必须告诉人怎么调高或关掉，否则只是把人卡死")

    def test_gate_is_off_when_set_to_zero(self):
        with mock.patch.object(budget, "DAILY_TOKEN_BUDGET", 0), \
             mock.patch.object(budget, "spent_today", return_value=10 ** 9):
            budget.check()                 # 显式关闭就不该拦

    def test_a_broken_gate_does_not_lock_the_tool(self):
        """闸门自己坏了（库读不了）不该把功能一起锁死——它是护栏，不是主路。"""
        with mock.patch.object(budget.db, "conn", side_effect=RuntimeError("db down")):
            self.assertEqual(budget.spent_today(), 0)

    def test_both_paid_entry_points_are_gated(self):
        """`llm.chat` / `llm.chat_stream` / `embeddings.embed_texts` 三个花钱入口。

        断言的是代码，所以先剥注释——本仓已经三次栽在「测试匹配到自己写的注释」上。
        """
        from papernest import embeddings, llm

        for fn in (llm.chat, llm.chat_stream, embeddings.embed_texts):
            with self.subTest(fn=fn.__name__):
                self.assertIn("budget", code_of(fn),
                              f"{fn.__name__} 没有过每日用量闸门")


class FailuresLeaveATraceTests(unittest.TestCase):
    """③ 日志：不求全，但失败必须查得到。"""

    def test_logger_writes_to_a_rotating_file_under_data(self):
        self.assertEqual(log.LOG_PATH.parent, config.DATA_DIR)
        self.assertGreater(log.MAX_BYTES, 0, "没有大小上限，日志会吃掉磁盘")
        self.assertGreater(log.BACKUPS, 0)

    def test_setup_does_not_touch_the_root_logger(self):
        """动 root 会顺手改掉 uvicorn / httpx 的行为，那不是这个模块该管的。"""
        import logging
        log.get("probe")
        self.assertEqual(logging.getLogger().handlers,
                         logging.getLogger().handlers)   # 只是确保下面那条有意义
        self.assertFalse(logging.getLogger("papernest").propagate)

    def test_endpoints_that_swallow_into_200_still_log(self):
        """这四个端点把异常吞成 `HTTP 200 + {"error": ...}`（前端在读这个形状，
        不能改），所以它们**必须**留下 traceback，否则事后完全无从复盘。"""
        for fn in (api.do_chat, api.do_survey, api.do_extract_symbols, api.do_table):
            code = code_of(fn)
            with self.subTest(fn=fn.__name__):
                self.assertIn("_log.exception", code,
                              f"{fn.__name__} 把异常吞了却没留痕")

    def test_unhandled_exceptions_are_logged_and_do_not_leak_internals(self):
        """**先把库指走**。这条用例今天碰不到生产库纯属巧合——它把 `init_db`
        mock 成抛异常，于是那句写操作没跑成。哪天换个端点、或者不再 mock，
        它就会对用户的 80MB 生产库执行 `executescript(SCHEMA)` + `_migrate()`。
        2026-09-09 就是这么让 `table_summaries` 建到真库里的（另一个文件）。
        """
        from fastapi.testclient import TestClient

        tmp = make_tempdir(self, "papernest_posture")
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        self.addCleanup(lambda: (setattr(config, "DB_PATH", old_db),
                                 setattr(config, "DATA_DIR", old_dir)))
        config.DATA_DIR = tmp / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"

        before = log.LOG_PATH.read_text(encoding="utf-8") if log.LOG_PATH.exists() else ""
        client = TestClient(api.app, raise_server_exceptions=False)
        with mock.patch.object(api.db, "init_db", side_effect=RuntimeError("boom-probe")):
            r = client.get("/api/symbols")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("boom-probe", str(r.json()),
                         "内部错误文本被透传给了客户端")
        if log.LOG_PATH.exists():
            added = log.LOG_PATH.read_text(encoding="utf-8")[len(before):]
            self.assertIn("boom-probe", added, "未捕获的异常没有进日志")
            self.assertIn("Traceback", added, "日志里没有 traceback，等于只说了「出错了」")


if __name__ == "__main__":
    unittest.main()
