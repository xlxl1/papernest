# -*- coding: utf-8 -*-
r"""模型返回的**形状会漂**——功能不能因此崩掉或静默毁数据。

2026-09-09 第一次拿真模型（glm-5.2）跑那 7 个从没被验证过的 LLM 路径，同一段代码
跑两次结果不同：第一次 `write_topic` 崩在
`TypeError: object of type 'int' has no len()`（烧掉 9657 token），
`symbols` 抽出 0 条（烧掉 10025 token）；第二次两个都正常
（4 个选题 / 29 个符号）。**功能是实现了的，脆的是对返回形状的假设。**

这个文件把两类脆弱钉住：

① **形状漂移**。`{"topics": [...]}` 这次是对象列表，下次可能是数字（把数量当答案）、
   单个对象、或列表里混进字符串。而下游是 `for t in topics` + `len(t["evidence"])`。
   规整成 list[dict] 之后，坏形状会落进「空了就走离线兜底」那条既有的路，
   而不是把整个阶段炸掉——那条兜底就写在旁边，崩了等于白写。

② **解析失败不许毁数据**。`symbols.extract_for_paper` 是「先 DELETE 再 INSERT」：
   模型返回一次垃圾 → 解析出 0 条 → **用户已有的符号表被清空换成空的**，
   而返回值只说 `count: 0`，不报错也不报降级。这是静默的破坏性失败。

③ 顺带收口了 `extract_json_array`：它原来住在 `symbols.py` 里，是 `extract_json`
   的平行实现，带着同样两个洞——括号扫描不认字符串（而本模块返回的正是
   `$\mathcal{T}_\mathrm{LM}$` 这种满是括号的 LaTeX）、解析失败静默返回 `[]`。
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, llm, symbols
from papernest.pipeline import _as_list_of_dicts


class ShapeNormalisationTests(unittest.TestCase):
    def test_normalises_every_shape_the_model_has_produced(self):
        self.assertEqual(_as_list_of_dicts([{"a": 1}]), [{"a": 1}])
        self.assertEqual(_as_list_of_dicts({"a": 1}), [{"a": 1}], "单个对象要能收")
        self.assertEqual(_as_list_of_dicts(4), [], "模型把数量当答案返回过")
        self.assertEqual(_as_list_of_dicts("topics"), [])
        self.assertEqual(_as_list_of_dicts(None), [])
        self.assertEqual(_as_list_of_dicts([{"a": 1}, "junk", 3]), [{"a": 1}],
                         "混进来的非对象要丢掉，不能让它流到下游")


class ArrayParserIsAsHardenedAsTheObjectOneTests(unittest.TestCase):
    def test_latex_braces_inside_strings_do_not_truncate(self):
        """符号抽取返回的就是满是括号的 LaTeX，这是它最常见的输入。"""
        got = llm.extract_json_array(
            r'[{"latex": "\mathcal{T}_\mathrm{LM}", "sym": "T"}]')
        self.assertEqual(got[0]["sym"], "T")

    def test_bracket_inside_a_string_does_not_truncate(self):
        self.assertEqual(llm.extract_json_array('[{"sym": "a]b"}]'),
                         [{"sym": "a]b"}])

    def test_single_quotes_and_trailing_commas(self):
        self.assertEqual(llm.extract_json_array("[{'sym': 'a'}]"), [{"sym": "a"}])
        self.assertEqual(llm.extract_json_array('[{"sym":"a"},]'), [{"sym": "a"}])

    def test_failure_raises_instead_of_returning_empty(self):
        """静默的空列表会让调用方把「解析失败」当成「模型说没有」。"""
        for raw in ("sorry, no", '{"not": "an array"}', "[ this is junk }"):
            with self.subTest(raw=raw[:20]):
                with self.assertRaises(llm.LLMError):
                    llm.extract_json_array(raw)

    def test_symbols_module_uses_the_shared_parser(self):
        self.assertIs(symbols.extract_json_array, llm.extract_json_array,
                      "又出现平行实现了——两份解析器会各自长出各自的洞")


class ParseFailureDoesNotDestroyExistingDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_shape_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)
        db.init_db()
        symbols.ensure_schema() if hasattr(symbols, "ensure_schema") else None
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "x:1", "title": "T", "abstract": "a", "year": 2024,
                "venue": "v", "authors": [], "doi": None, "arxiv_id": None,
                "source": "s2"})
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (self.pid, 1, "The model uses $\alpha$ as the learning rate."))
            c.execute("""INSERT INTO symbols(paper_id,kind,sym,latex,meaning,page)
                         VALUES(?,?,?,?,?,?)""",
                      (self.pid, "symbol", "α", "\alpha", "学习率", 1))

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def _count(self):
        with db.conn() as c:
            return c.execute("SELECT COUNT(*) n FROM symbols WHERE paper_id=?",
                             (self.pid,)).fetchone()["n"]

    def test_garbage_response_keeps_the_existing_symbols(self):
        """模型返回垃圾时，原有符号必须原封不动。"""
        with mock.patch.object(symbols.llm, "available", return_value=True), \
             mock.patch.object(symbols.llm, "chat", return_value="抱歉，我无法完成"):
            r = symbols.extract_for_paper(self.pid)
        self.assertEqual(self._count(), 1, "已有的符号被清空了")
        self.assertTrue(r.get("kept_existing"))
        self.assertIn("error", r, "毁没毁数据先不说，至少要说清这次失败了")

    def test_wellformed_but_empty_also_keeps_existing(self):
        """返回了合法数组但一条 sym 都没有——同样不该覆盖。"""
        with mock.patch.object(symbols.llm, "available", return_value=True), \
             mock.patch.object(symbols.llm, "chat", return_value='[{"note":"none"}]'):
            r = symbols.extract_for_paper(self.pid)
        self.assertEqual(self._count(), 1)
        self.assertTrue(r.get("kept_existing"))

    def test_a_good_response_does_replace(self):
        """正常返回时该覆盖还是要覆盖——不能为了安全把功能也关掉。"""
        good = '[{"kind":"symbol","sym":"β","latex":"\\beta","meaning":"动量","page":2}]'
        with mock.patch.object(symbols.llm, "available", return_value=True), \
             mock.patch.object(symbols.llm, "chat", return_value=good):
            r = symbols.extract_for_paper(self.pid)
        self.assertEqual(r["count"], 1)
        self.assertFalse(r.get("kept_existing"))
        with db.conn() as c:
            syms = [x["sym"] for x in c.execute(
                "SELECT sym FROM symbols WHERE paper_id=?", (self.pid,))]
        self.assertEqual(syms, ["β"], "新结果没有替换掉旧的")


class StageTopicSurvivesEveryShapeTests(unittest.TestCase):
    """选题阶段：模型返回什么形状都不能崩——坏了就走离线兜底并如实报降级。

    这些形状不是想象出来的，是拿真模型（glm-5.2）跑同一个 prompt 反复跑出来的。
    实测崩点有两个，都**不是**一眼能想到的那种：
      · `pipeline.py:115` `(quote or "").strip()` —— 模型给了 `quote: 123`
      · `pipeline.py:288` `int(ev["paper_id"])` —— 模型给了 `paper_id: "abc"`
    所以后处理是**整段兜住**的，而不是逐个字段打补丁：每加一个字段就多一处可能崩的
    地方，而旁边就摆着写好的 `_offline_topics`——被一个 TypeError 炸掉等于白写。
    """

    SHAPES = {
        "topics 是数字（把数量当答案）": {"topics": 4},
        "topics 是单个对象": {"topics": {"title": "x", "evidence": []}},
        "evidence 是数字": {"topics": [{"title": "x", "evidence": 3}]},
        "quote 是数字": {"topics": [{"title": "x",
                                   "evidence": [{"paper_id": 1, "quote": 123}]}]},
        "paper_id 是乱码": {"topics": [{"title": "x",
                                     "evidence": [{"paper_id": "abc", "quote": "q"}]}]},
        "title 是数字": {"topics": [{"title": 5, "evidence": []}]},
        "整个返回是数组": [{"title": "x"}],
        "列表里混进字符串": {"topics": [{"title": "ok", "evidence": []}, "junk", 7]},
    }

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_topic_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)
        db.init_db()
        self.papers = []
        with db.conn() as c:
            for i in range(3):
                pid = db.insert_l0(c, {
                    "norm_key": f"x:{i}", "title": f"LLM Agent eval {i}",
                    "abstract": "a" * 80, "year": 2024, "venue": "v", "authors": [],
                    "doi": None, "arxiv_id": None, "source": "s2"})
                db.save_card(c, pid, {"tldr": "t",
                                      "limitations": "这里是一段足够长的局限描述用于离线兜底"}, "m")
                self.papers.append({"id": pid, "title": f"P{pid}",
                                    "card": {"tldr": "t", "limitations": "x" * 40},
                                    "abstract": "a" * 60})

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def _run(self, payload):
        from papernest import pipeline
        rid = pipeline.create_run("LLM Agent 的评测方法")["id"]
        with mock.patch.object(pipeline, "_topic_context",
                               return_value=(self.papers, "ctx")),              mock.patch.object(pipeline.llm, "available", return_value=True),              mock.patch.object(pipeline.llm, "chat",
                               return_value=json.dumps(payload, ensure_ascii=False)):
            return pipeline.stage_topic(rid)

    def test_no_shape_crashes_the_stage(self):
        for name, payload in self.SHAPES.items():
            with self.subTest(shape=name):
                r = self._run(payload)
                self.assertGreater(r["topics"], 0,
                                   f"{name}: 一个候选都没产出，离线兜底也没接上")

    def test_unusable_shapes_fall_back_and_say_so(self):
        """完全用不了的形状要走离线兜底，并且**留下降级说明**。"""
        r = self._run({"topics": 4})
        self.assertGreater(r["topics"], 0)
        self.assertTrue(r.get("degraded"), "走了兜底却没报降级，用户不知道这批是离线产的")

    def test_a_good_response_is_not_downgraded(self):
        """正常返回不该被这层防御误伤。"""
        good = {"topics": [{"title": "一个正常选题",
                            "evidence": [{"paper_id": self.papers[0]["id"],
                                          "quote": "a" * 20}]}]}
        r = self._run(good)
        self.assertEqual(r["topics"], 1)
        self.assertIsNone(r.get("degraded"))


if __name__ == "__main__":
    unittest.main()
