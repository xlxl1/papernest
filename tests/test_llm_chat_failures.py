# -*- coding: utf-8 -*-
"""`llm.chat` 的失败形态——这个函数体过去**一行都没测过**。

整套测试里唯一被真正打过的 LLM 路径是 `chat_stream`（test_writing_desk 那 2 条）。
而非流式的 `llm.chat` 是卡片生成 / survey / rerank / matrix / deep_critic /
polish-review 全部要走的那一条，`grep "llm.chat(" tests/` 过去零命中——
所有用例都是 `mock.patch("papernest.llm.chat")` 把它整个换掉。

于是这条最常见的失败形态一直没人管：**中转站过载时回 HTTP 200 + 错误体**。
原来的收尾是 `return data["choices"][0]["message"]["content"]`，链上每一环都可能缺，
抛出来的是 KeyError / IndexError / TypeError——**它们都不是 `LLMError`**，
于是绕过上层所有 `except LLMError` 的降级分支，以 500 + 裸 traceback 冒到用户面前。

这里用替身 `http.client` 直接打 `llm.chat` 的函数体，0 次真实请求、0 token。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from papernest import config, db, llm


class _Resp:
    def __init__(self, payload, status=200, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    """按脚本逐次返回响应或抛异常。脚本**跨调用共享**，否则测不到重试。"""

    def __init__(self, script, calls):
        self.script, self.calls = script, calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append(url)
        step = self.script.pop(0) if self.script else _Resp(_ok("默认"))
        if isinstance(step, Exception):
            raise step
        return step


def _ok(text="正文"):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class LlmChatFailureModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_llmchat_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.LLM_API_BASE,
                     config.LLM_API_KEY, config.LLM_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.LLM_API_BASE = "https://stub.invalid/v1"
        config.LLM_API_KEY = "stub"
        config.LLM_MODEL = "stub-model"
        self.addCleanup(self._restore)
        db.init_db()

    def _restore(self):
        (config.DB_PATH, config.DATA_DIR, config.LLM_API_BASE,
         config.LLM_API_KEY, config.LLM_MODEL) = self._old

    def _run(self, script):
        calls = []
        shared = list(script)
        with mock.patch.object(llm.http, "client",
                               lambda *a, **k: _Client(shared, calls)), \
             mock.patch.object(llm, "deadline", mock.MagicMock(
                 remaining=lambda: 90.0, check=lambda: None, sleep=lambda s: None)):
            try:
                return llm.chat("s", "u", purpose="t"), calls, None
            except Exception as exc:                 # noqa: BLE001
                return None, calls, exc

    # ── 正常路径先立住，否则下面的断言都没有意义 ──

    def test_happy_path_returns_the_content(self):
        out, calls, exc = self._run([_Resp(_ok("答案"))])
        self.assertIsNone(exc)
        self.assertEqual(out, "答案")
        self.assertEqual(len(calls), 1)

    # ── HTTP 200 + 错误体：中转站最常见的失败形态 ──

    def test_error_body_with_http_200_raises_llm_error_not_key_error(self):
        _out, _calls, exc = self._run([_Resp({"error": {"message": "model not found"}})])
        self.assertIsInstance(exc, llm.LLMError,
                              f"抛的是 {type(exc).__name__}，不是 LLMError——"
                              f"上层所有 except LLMError 的降级分支都会被绕过")
        self.assertIn("model not found", str(exc))

    def test_provider_side_error_is_not_retried(self):
        """provider 明确说了错在哪（模型名错、余额不足），重试只是白等 73 秒。"""
        _out, calls, _exc = self._run(
            [_Resp({"error": {"message": "insufficient balance"}})] * 4)
        self.assertEqual(len(calls), 1, f"重试了 {len(calls)} 次")

    def test_empty_choices_is_retried_then_reported(self):
        """没有 error 只是 choices 空——那更像瞬态，值得重试；耗尽后仍要是 LLMError。"""
        _out, calls, exc = self._run([_Resp({"choices": []})] * 4)
        self.assertIsInstance(exc, llm.LLMError)
        self.assertEqual(len(calls), 4)

    def test_recovers_when_a_later_attempt_succeeds(self):
        _out, calls, exc = self._run([_Resp({"choices": []}), _Resp(_ok("救回来了"))])
        self.assertIsNone(exc)
        self.assertEqual(_out, "救回来了")
        self.assertEqual(len(calls), 2)

    def test_non_json_body_with_http_200_is_handled(self):
        """过载时也可能回 HTTP 200 + 一段 HTML 错误页。"""
        _out, _calls, exc = self._run(
            [_Resp(ValueError("no json"), text="<html>502</html>")] * 4)
        self.assertIsInstance(exc, llm.LLMError)

    # ── 既有的状态码分支同样没被测过，一并钉住 ──

    def test_400_is_not_retried(self):
        _out, calls, exc = self._run([_Resp({}, status=400, text="bad request")] * 4)
        self.assertIsInstance(exc, llm.LLMError)
        self.assertEqual(len(calls), 1)

    def test_401_is_not_retried(self):
        _out, calls, exc = self._run([_Resp({}, status=401, text="unauthorized")] * 4)
        self.assertIsInstance(exc, llm.LLMError)
        self.assertEqual(len(calls), 1)

    def test_503_is_retried(self):
        _out, calls, exc = self._run([_Resp({}, status=503, text="busy")] * 4)
        self.assertIsInstance(exc, llm.LLMError)
        self.assertEqual(len(calls), 4)

    def test_network_error_is_retried(self):
        _out, calls, exc = self._run([httpx.ConnectError("boom")] * 4)
        self.assertIsInstance(exc, llm.LLMError)
        self.assertEqual(len(calls), 4)

    # ── 落账：成本证据链是这个项目对外声称的东西，成功路径必须真写进去 ──

    def test_successful_call_is_recorded(self):
        out, _calls, exc = self._run([_Resp(_ok("答案"))])
        self.assertIsNone(exc)
        self.assertEqual(out, "答案")
        with db.conn() as c:
            rows = c.execute("SELECT purpose, prompt_tokens FROM llm_calls").fetchall()
        self.assertEqual([(r["purpose"], r["prompt_tokens"]) for r in rows], [("t", 1)])

    def test_failed_call_is_not_recorded_as_a_success(self):
        """失败不该在成本证据链里留下一条「成功调用」。"""
        self._run([_Resp({"error": {"message": "nope"}})])
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM llm_calls").fetchone()["n"]
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()


class HttpFourHundredIsExplainedTests(unittest.TestCase):
    """400 是个筐：参数错、模型名错、内容被拦、**账户欠费**，全在里面。

    2026-09-09 实测：一批表格摘要跑到第 61 张时账户欠费，剩下 23 张全炸成
    `{"type":"Arrearage", "message":"Access denied, please make sure your account
    is in good standing..."}`。原来一律报「LLM 请求被拒（400，不重试）」再跟一段
    JSON——批量任务里刷出几十条一模一样的长英文，人得自己去读 JSON 才知道
    是钱的问题，而不是代码的问题。
    """

    def test_arrearage_is_named_in_chinese(self):
        from papernest.llm import _explain_400
        body = ('{"error":{"message":"Access denied, please make sure your account '
                'is in good standing.","type":"Arrearage"}}')
        msg = _explain_400(body)
        self.assertIn("欠费", msg)
        self.assertIn("充值", msg, "没告诉人该做什么，等于只是把英文抄了一遍")
        self.assertIn("跳过", msg,
                      "没说重跑会跳过已完成的部分，人会不敢重跑")

    def test_the_raw_body_is_still_there(self):
        """翻译不能把原文吞掉——判断错了的时候还得看得见原始响应。"""
        from papernest.llm import _explain_400
        msg = _explain_400('{"error":{"type":"Arrearage"}}')
        self.assertIn("Arrearage", msg)

    def test_a_wrong_model_name_points_at_the_checker(self):
        from papernest.llm import _explain_400
        msg = _explain_400('{"error":{"message":"model not found: gpt-9"}}')
        self.assertIn("check_models", msg)

    def test_an_unrecognised_400_still_reports_the_body(self):
        from papernest.llm import _explain_400
        msg = _explain_400('{"error":{"message":"something new"}}')
        self.assertIn("something new", msg)


class ExtractJsonIsDefinedOnceTests(unittest.TestCase):
    """`extract_json` 一度在 llm.py 里定义了两次。

    合并 JSON 解析时我把新版加在了文件**最开头**、却没删掉旧版。后果两条：
    ① 后定义的（旧实现，自己抄了一份 `_scan_balanced` 的括号计数）才生效，
       新版是死代码——两份实现修一处漏一处，正是这次合并要消灭的东西；
    ② 排在模块 docstring 前面的那个 `def` 让 `llm.__doc__` 变成了 None。
    """

    def test_only_one_definition(self):
        from papernest import config
        src = (config.ROOT / "papernest" / "llm.py").read_text(encoding="utf-8")
        self.assertEqual(src.count("def extract_json(text"), 1)

    def test_module_docstring_survives(self):
        from papernest import llm
        self.assertTrue(llm.__doc__, "模块 docstring 又被一个 def 挡在后面了")

    def test_it_delegates_instead_of_reimplementing(self):
        import inspect

        from papernest import llm
        src = inspect.getsource(llm.extract_json)
        self.assertIn("_scan_balanced", src)
        self.assertNotIn("depth", src, "又把括号计数抄了一份进来")

    def test_a_brace_inside_a_string_does_not_truncate(self):
        from papernest import llm
        got = llm.extract_json('```json\n{"a": "含 } 的字符串", "b": 2}\n```')
        self.assertEqual(got, {"a": "含 } 的字符串", "b": 2})
