# -*- coding: utf-8 -*-
"""`llm.extract_json`：模型返回的 JSON 出毛病时，**必须干净失败**而不是崩。

这是「真接上模型之后功能能不能跑」的一个具体卡点。原来的实现有两个洞：

① `json.loads` 对「括号配平但格式非法」抛的是 `json.JSONDecodeError`
   （`ValueError` 子类，**不是 `LLMError`**）。而全仓 6 个调用点只 `except llm.LLMError`：
   `cards.py:31`（L1 卡片）· `pipeline.py:251/375`（写作选题、大纲）·
   `tablegen.py:40`（对比表）· `writing.py:61/93`（润色、审阅）。
   模型只要犯一次**单引号**或**尾逗号**——这是 LLM 最常见的两种 JSON 毛病——
   这些功能就直接崩，而它们旁边就摆着写好的离线兜底（`_offline_topics` /
   `_offline_outline`）。兜底进不去，等于白写。

② 花括号扫描器**不认字符串**：`{"s": "a }"}` 会在字符串内部的 `}` 处提前截断。
   模型回 LaTeX（`\frac{a}{b}`）、代码片段时很容易踩到，而症状看起来像
   「模型没按格式输出」，实际是我们自己切错了。

真库佐证：22 个 LLM 调用点里 **7 个从没被真模型跑过**（symbols / matrix 在线路径 /
ppt_narrative / subscribe_rerank / write_topic / write_outline / rcs_answer_fallback），
所以这类问题在换真 key 之前根本暴露不出来。
"""
import unittest

from papernest import llm


class ToleratesCommonModelMistakesTests(unittest.TestCase):
    def test_strict_json_still_works(self):
        self.assertEqual(llm.extract_json('{"a": 1}'), {"a": 1})

    def test_code_fence_is_stripped(self):
        self.assertEqual(llm.extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_single_quotes(self):
        """模型最常犯的毛病之一。走 `ast.literal_eval`——它只认字面量、不执行代码。"""
        self.assertEqual(llm.extract_json("{'topics': [1]}"), {"topics": [1]})

    def test_trailing_comma(self):
        self.assertEqual(llm.extract_json('{"a": [1, 2,], "b": {"c": 1,},}'),
                         {"a": [1, 2], "b": {"c": 1}})

    def test_brace_inside_a_string_does_not_truncate(self):
        """扫描器要认字符串，否则字符串里的 `}` 会把 JSON 切成半截。"""
        self.assertEqual(llm.extract_json('{"s": "a, }"}'), {"s": "a, }"})

    def test_escaped_quote_inside_a_string(self):
        self.assertEqual(llm.extract_json(r'{"s": "he said \"hi\" }"}'),
                         {"s": 'he said "hi" }'})


class FailsCleanlySoCallersCanFallBackTests(unittest.TestCase):
    """真解析不了时必须抛 `LLMError`——调用方只 catch 这一个。"""

    BAD = ("sorry, I cannot do that",          # 完全没有 JSON
           '{"a": 1',                          # 未闭合
           "{ this is not json at all }",      # 括号配平但内容不是 JSON
           '{"a": undefined}')                 # JS 的 undefined，不是合法 JSON

    def test_every_failure_is_an_llm_error(self):
        for raw in self.BAD:
            with self.subTest(raw=raw[:24]):
                with self.assertRaises(llm.LLMError):
                    llm.extract_json(raw)

    def test_no_other_exception_type_escapes(self):
        """具体到那 6 个只 catch LLMError 的调用点：别的异常类型逃出去就是崩。"""
        for raw in self.BAD:
            with self.subTest(raw=raw[:24]):
                try:
                    llm.extract_json(raw)
                except llm.LLMError:
                    pass
                except Exception as exc:                        # noqa: BLE001
                    self.fail(f"{type(exc).__name__} 逃出去了：{raw[:40]!r}")

    def test_the_callers_that_only_catch_llmerror_are_still_there(self):
        """这条是**存档**：只要还有调用点只 catch LLMError，上面那些断言就必须成立。

        它不是在固化缺陷——收口在 `extract_json` 里是对的（改一处胜过补六处），
        这条只是让「为什么必须收口」在代码里留个记号。
        """
        import ast
        import pathlib
        from papernest import config
        narrow = []
        for name in ("cards.py", "pipeline.py", "tablegen.py", "writing.py"):
            path = pathlib.Path(config.ROOT) / "papernest" / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                body = ast.Module(body=node.body, type_ignores=[])
                if not any(getattr(n, "attr", "").startswith("extract_json")
                           for n in ast.walk(body)):
                    continue
                names = [getattr(h.type, "attr", getattr(h.type, "id", None))
                         for h in node.handlers if h.type is not None]
                if names and all(n == "LLMError" for n in names):
                    narrow.append(f"{name}:{node.lineno}")
        self.assertTrue(narrow,
                        "所有调用点都改成宽 catch 了——那这个文件的前提变了，"
                        "可以重新评估是否还需要 extract_json 里的容错")


if __name__ == "__main__":
    unittest.main()
