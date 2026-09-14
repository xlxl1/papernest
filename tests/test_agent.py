import sqlite3
import time
import unittest

from papernest import config, tools
from papernest.agent import build_plan, run_agent
from papernest.llm import LLMUnavailable
from papernest.schemas import AgentRequest


class AgentPlanningTests(unittest.TestCase):
    def test_default_plan_is_retrieve_then_answer(self):
        plan = build_plan("比较库内的 Agent 评测方法", top_k=3)
        self.assertEqual([step.tool for step in plan],
                         ["retrieve_library", "answer_question"])
        self.assertEqual(plan[0].args["top_k"], 3)

    def test_citation_intent_selects_citation_tool(self):
        plan = build_plan("推荐支持这段话的参考文献")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].tool, "recommend_citations")

    def test_read_intent_extracts_paper_id(self):
        plan = build_plan("请精读论文 #42")
        self.assertEqual(plan[0].tool, "read_paper")
        self.assertEqual(plan[0].args["paper_id"], 42)

    def test_dry_run_never_calls_tools(self):
        result = run_agent("生成一份文献综述", dry_run=True)
        self.assertEqual(result.status, "planned")
        self.assertEqual(result.events, [])
        self.assertEqual(result.plan[0].tool, "generate_survey")

    def test_request_limits_are_validated(self):
        request = AgentRequest(goal="hello", top_k=10, max_steps=4)
        self.assertEqual(request.top_k, 10)
        with self.assertRaises(ValueError):
            AgentRequest(goal="hello", top_k=0)


class AgentExecutionTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(tools.TOOL_REGISTRY)

    def tearDown(self):
        tools.TOOL_REGISTRY.clear()
        tools.TOOL_REGISTRY.update(self._saved)

    def test_step_timeout_hard_stops(self):
        tools.TOOL_REGISTRY["retrieve_library"] = (
            lambda query, top_k=5: (time.sleep(2), {"paper_ids": []})[1])
        r = run_agent("测试超时", timeout_s=1, persist=False)
        self.assertEqual(r.status, "failed")
        self.assertEqual([e.status for e in r.events], ["timeout"])
        self.assertIn("超过", r.error)

    def test_transient_failure_is_retried_once(self):
        calls = {"n": 0}

        def flaky(query, top_k=5):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("瞬态故障")
            return {"paper_ids": [1], "papers": []}

        tools.TOOL_REGISTRY["retrieve_library"] = flaky
        tools.TOOL_REGISTRY["answer_question"] = (
            lambda goal, top_k=5, candidate_ids=None: {"answer": "ok [1]", "sources": []})
        r = run_agent("测试重试", persist=False)
        self.assertEqual(r.status, "completed")
        self.assertEqual(calls["n"], 2)
        self.assertEqual([e.status for e in r.events if e.tool == "retrieve_library"],
                         ["running", "completed"])

    def test_llm_unavailable_is_not_retried_and_blocks(self):
        tools.TOOL_REGISTRY["retrieve_library"] = lambda query, top_k=5: {"paper_ids": []}
        def no_llm(goal, top_k=5, candidate_ids=None):
            raise LLMUnavailable("未配置 LLM")
        tools.TOOL_REGISTRY["answer_question"] = no_llm
        r = run_agent("测试 blocked", persist=False)
        self.assertEqual(r.status, "blocked")
        self.assertEqual(len(r.events), 2)  # 检索完成 + 生成失败，无重试事件

    def test_on_event_receives_every_event(self):
        tools.TOOL_REGISTRY["recommend_citations"] = (
            lambda paragraph, top_k=5: {"candidates": []})
        seen = []
        run_agent("推荐文献 for on_event test", persist=False,
                  on_event=lambda ev: seen.append(ev["status"]))
        self.assertEqual(seen, ["completed"])

    def test_total_ms_is_recorded(self):
        tools.TOOL_REGISTRY["recommend_citations"] = (
            lambda paragraph, top_k=5: {"candidates": []})
        r = run_agent("推荐文献 timing", persist=False)
        self.assertGreaterEqual(r.total_ms, 0)


if __name__ == "__main__":
    unittest.main()



class PlanRoutingGroundingTests(unittest.TestCase):
    """路由必须分清「用户的意图」和「被用户引述的库内容」。

    下面这些标题是 data/papernest.db 里**真实存在**的论文，不是构造的夹具：
    库主题正是 LLM Agent，503 篇里有 5 篇标题带 Survey/综述，而「贴一段标题来提问」
    是最常见的用法。裸子串路由会把「问这篇」读成「写一篇全库综述」——
    真库实测：贴标题问「被引用了多少次」503/503 全部被劫持去做引用推荐，
    而答案（209）就在 papers.citation_count 里。

    加词边界修不掉这一层——"A Survey" 本来就是独立单词。判据必须是
    「领域词 AND 本分支的祈使动词」。
    """

    # 逐字取自真库 papers.title（id=1 / id=10）
    REAL_TITLES = (
        "Large Language Model Agent: A Survey on Methodology, Applications and Challenges",
        "Large Language Model-based Data Science Agent: A Survey",
    )

    def test_quoted_survey_title_is_not_a_request_to_write_a_survey(self):
        for title in self.REAL_TITLES:
            for goal in (f"库里有哪些和《{title}》相关的文献？",
                         f"{title} 这篇论文的方法是什么？"):
                with self.subTest(goal=goal):
                    self.assertEqual(build_plan(goal)[0].tool, "retrieve_library")

    def test_read_intent_survives_a_survey_word_inside_the_title(self):
        plan = build_plan(f"精读论文 1：{self.REAL_TITLES[0]}")
        self.assertEqual(plan[0].tool, "read_paper")
        self.assertEqual(plan[0].args["paper_id"], 1)

    def test_asking_the_citation_count_is_not_a_citation_recommendation(self):
        """papers.citation_count 就在库里，这是问答/检索，不是「推荐能支持这段话的文献」。"""
        goal = f"《{self.REAL_TITLES[0]}》被引用了多少次？"
        self.assertEqual(build_plan(goal)[0].tool, "retrieve_library")

    def test_cite_matches_a_word_not_a_substring(self):
        """真库 pages 里出现过 elicited（paper 495 第 10 页），chunks 里有 citeseer。"""
        for goal in ("找一篇讲 elicited emotions 的论文",
                     "recommend a method for excited-state simulation"):
            with self.subTest(goal=goal):
                self.assertEqual(build_plan(goal)[0].tool, "retrieve_library")

    def test_real_intent_words_still_route(self):
        """反向闸门：修法不许把正例一起砍掉。修前修后都必须绿。"""
        self.assertEqual(build_plan("写一篇关于 LLM Agent 评测的综述")[0].tool,
                         "generate_survey")
        self.assertEqual(
            build_plan("推荐支持这段话的参考文献：near-field channel estimation")[0].tool,
            "recommend_citations")

    def test_no_real_library_title_hijacks_the_router(self):
        """真库全量：贴任何一篇真实标题问「相关文献」都不该跳去写综述。

        允许 1 篇残留——《检索增强生成的评测方法综述》里的「生成」既是祈使动词
        又是 RAG 的领域词，中文没有词边界，确定性规则解不掉。
        **这个阈值不许往上调**：以后库里再进「生成式/写作」主题的中文论文而它变红时，
        该做的是重新评估判据，不是放宽断言。
        """
        if not config.DB_PATH.exists():
            self.skipTest(f"本机没有文献库（{config.DB_PATH}），跳过全量路由校验")
        conn = sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
        try:
            titles = [t for (t,) in conn.execute(
                "SELECT title FROM papers WHERE title IS NOT NULL")]
        finally:
            conn.close()
        bad = [t for t in titles
               if build_plan(f"库里有哪些和《{t}》相关的文献？")[0].tool != "retrieve_library"]
        self.assertLessEqual(
            len(bad), 1,
            f"{len(bad)}/{len(titles)} 篇真实标题劫持了路由，例如：{bad[:5]}")
