import time
import unittest

from papernest import tools
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

