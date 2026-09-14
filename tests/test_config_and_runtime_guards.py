# -*- coding: utf-8 -*-
"""几条运维口子：.env 解析、import 期崩、白睡的退避、靠中文文案猜状态。

这些单独看都很小，共同点是**失败方式都很安静**：
  · `.env` 里 `KEY=value  # 注释` 会把整串注释塞进 `os.environ`，而项目自己的
    `.env.example` 就是这么写的、README 又让人直接 copy——「照文档做」必产出脏配置；
  · 模块级裸 `float(os.environ[...])` 让一个排序权重配错就把整个包（连同 uvicorn
    和 cli）在 import 期炸掉；
  · S2 在**最后一次**尝试失败后仍然睡满 150s，纯粹把调用方多锁 150 秒；
  · agent 已经用 `except llm.LLMUnavailable` 精确捕到「缺配置」，40 行外又用
    `"未配置" in tool_error` 这个中文子串重新猜一遍。
"""
import unittest
import warnings
from unittest import mock

from papernest import agent, api, config, embeddings, graph, jobs
from papernest.sources import semantic_scholar as s2


def code_of(fn) -> str:
    """函数源码，**剥掉注释**。

    断言「代码里有没有 X」时必须先剥注释，否则会命中解释「原来这里写的是 X」的
    那句话。本仓已经栽过三次：`test_no_endpoint_awaits_upload_read` 一次、
    本文件里 `t_start` 一次、`GraphError` 一次——三次都是作者（我）自己写的注释。
    """
    import inspect
    import io
    import tokenize
    return " ".join(t.string for t in
                    tokenize.generate_tokens(io.StringIO(inspect.getsource(fn)).readline)
                    if t.type != tokenize.COMMENT)


class EnvValueParsingTests(unittest.TestCase):
    def test_inline_comment_is_stripped(self):
        """`.env.example` 里就有这种写法（PAPERNEST_RERANK=off  # api|llm|...）。"""
        self.assertEqual(config._strip_env_value("off          # api | llm | bm25"), "off")

    def test_quoted_value_is_taken_verbatim(self):
        """引号是「我要字面量」的显式声明：里面的 # 不是注释（口令里可能就有）。"""
        self.assertEqual(config._strip_env_value('"pa#ss word"'), "pa#ss word")
        self.assertEqual(config._strip_env_value("'x  # y'"), "x  # y")

    def test_hash_without_leading_space_is_not_a_comment(self):
        """URL 片段里的 # 不该被当注释砍掉。"""
        self.assertEqual(config._strip_env_value("https://x/y#frag"), "https://x/y#frag")

    def test_example_file_would_survive_a_copy(self):
        """README 让人 `copy .env.example .env`，那条路径必须产出干净的值。"""
        from pathlib import Path
        for line in (Path(config.ROOT) / ".env.example").read_text(
                encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            v = config._strip_env_value(line.split("=", 1)[1])
            self.assertNotIn("#", v.split()[0] if v.split() else "",
                             f"{line[:40]!r} 复制过去会被污染成 {v!r}")


    def test_load_env_actually_uses_the_stripper(self):
        """守的是**调用点**：光有 `_strip_env_value` 而 `_load_env` 不调它等于没修。"""
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env").write_text(
                "PAPERNEST_PROBE_X=off          # api | llm | bm25" + chr(10),
                encoding="utf-8")
            with mock.patch.object(config, "ROOT", Path(d)),                  mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PAPERNEST_PROBE_X", None)
                config._load_env()
                got = os.environ.pop("PAPERNEST_PROBE_X", None)
        self.assertEqual(got, "off", f"_load_env 把整串注释塞进了环境变量：{got!r}")


class BadEnvValuesDoNotKillImportTests(unittest.TestCase):
    def test_illegal_float_falls_back_to_the_default(self):
        with mock.patch.dict("os.environ", {"X_W": "abc"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(config._env_float(0.2, "X_W"), 0.2)

    def test_empty_string_does_not_silently_skip_to_the_next_name(self):
        """原来的 `A or B` 链让「设成空」跳过 A 去拿 B，看着像生效其实没有。"""
        with mock.patch.dict("os.environ", {"A_W": "", "B_W": "9"}):
            self.assertEqual(config._env_float(0.2, "A_W", "B_W"), 9.0)

    def test_int_is_clamped_to_a_sane_minimum(self):
        """批大小 / 并发数是 0 或负数会让循环空转或崩掉。"""
        with mock.patch.dict("os.environ", {"X_N": "0"}):
            self.assertEqual(config._env_int(8, "X_N"), 1)

    def test_live_constants_use_the_safe_reader(self):
        self.assertIsInstance(embeddings.CHUNK_WEIGHT, float)
        self.assertGreaterEqual(jobs.MAX_CONCURRENT, 1)

    def test_the_dead_alias_is_gone(self):
        """`PAGE_WEIGHT = CHUNK_WEIGHT` 是副本，写它是 no-op——留着就是给人挖坑，
        而唯一守护默认权重的那条测试正是靠写它，等于什么都没断言。"""
        self.assertFalse(hasattr(embeddings, "PAGE_WEIGHT"))
        self.assertFalse(hasattr(embeddings, "PAGE_SEARCH"))


class NoWastedSleepAfterTheLastAttemptTests(unittest.TestCase):
    def test_search_does_not_sleep_after_the_final_failure(self):
        """最后一次尝试之后已经没有下一次请求了，那一觉纯粹是把调用方多锁 150 秒。

        单次查询最坏耗时因此从 305s 降到 155s。graph.py 的 `_backoff` 早就写对了，
        而这个文件的注释还声称「与 graph.py 同一套序列」。
        """
        slept = []

        class _Boom:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, *a, **kw):
                import httpx
                raise httpx.ConnectError("down")

        with mock.patch.object(s2.http, "client", lambda *a, **k: _Boom()), \
             mock.patch.object(s2, "_sleep", slept.append), \
             mock.patch.object(s2, "_jitter", lambda: 1.0):
            with self.assertRaises(s2.SourceError):
                s2.search("anything")
        self.assertEqual(slept, [20, 45, 90],
                         f"最后一次失败后还睡了：{slept}（最长那 150s 是白睡的）")


class AgentStatusComesFromTypesNotChineseTextTests(unittest.TestCase):
    """这两条断言的是**代码**，所以必须先把注释剥掉。

    本仓踩过一次：`test_no_endpoint_awaits_upload_read` 匹配到了作者自己解释旧写法的
    注释而变红。写这两条时又踩了一次——注释里写着「原来这里用的是 t_start」，
    断言就命中了。剥注释是最小的正确做法（AST 也行，但这里只需要一层）。
    """

    def test_blocked_is_decided_by_the_exception_type(self):
        """异常文案改一个字、或换个抛英文消息的 provider，blocked 就不该静默变 failed。"""
        code = code_of(agent.run_agent)
        self.assertIn("unconfigured", code)
        self.assertNotIn("未配置", code,
                         "又在用中文子串猜状态：中转站返回的中文错误体会把真故障误记成缺配置")

    def test_failure_event_reports_this_step_not_the_whole_run(self):
        code = code_of(agent.run_agent)
        i = code.index('"failed"')
        self.assertNotIn("t_start", code[i:i + 260],
                         "失败事件的 latency 又记成整个 run 的累计耗时了")


class GraphEndpointsMapTheirOwnErrorTests(unittest.TestCase):
    def test_both_graph_routes_catch_graph_error(self):
        """同一个 GraphError，紧邻的四个路由映射成 400，这两个漏了就是 500 + 裸 traceback。"""
        for fn in (api.graph_neighbors, api.graph_related):
            with self.subTest(fn=fn.__name__):
                self.assertIn("GraphError", code_of(fn))


if __name__ == "__main__":
    unittest.main()
