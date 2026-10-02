"""前端回传 intent_hint 的测试：用户答完 clarification 后重跑 agent 的契约。"""

import unittest

from papernest import schemas
from papernest.agent import build_plan, run_agent


class AgentRequestIntentHintTests(unittest.TestCase):
    """AgentRequest 加 intent_hint 字段，向后兼容。"""

    def test_default_none(self):
        r = schemas.AgentRequest(goal="g")
        self.assertIsNone(r.intent_hint)

    def test_accept_known_label(self):
        for label in ("cite", "survey", "read", "qa"):
            r = schemas.AgentRequest(goal="g", intent_hint=label)
            self.assertEqual(r.intent_hint, label)

    def test_accept_garbage(self):
        """非法 label 也接受——agent 端会静默忽略。"""
        r = schemas.AgentRequest(goal="g", intent_hint="garbage")
        self.assertEqual(r.intent_hint, "garbage")

    def test_too_long_rejected(self):
        with self.assertRaises(Exception):
            schemas.AgentRequest(goal="g", intent_hint="x" * 201)


class IntentHintOverridesPlanTests(unittest.TestCase):
    """intent_hint 应该接管意图分类与 plan 生成，不再走启发式。"""

    def test_cite_hint_changes_plan(self):
        r = run_agent("帮我处理一下", dry_run=True, persist=False, intent_hint="cite")
        tools_used = [s.tool for s in r.plan]
        self.assertEqual(tools_used[-1], "recommend_citations")
        self.assertEqual(r.intent.primary, "cite")
        self.assertEqual(r.intent.confidence, 1.0)

    def test_survey_hint_changes_plan(self):
        r = run_agent("帮我处理一下", dry_run=True, persist=False, intent_hint="survey")
        tools_used = [s.tool for s in r.plan]
        self.assertEqual(tools_used[-1], "generate_survey")
        self.assertEqual(r.intent.primary, "survey")

    def test_qa_hint_keeps_qa_plan(self):
        r = run_agent("帮我处理一下", dry_run=True, persist=False, intent_hint="qa")
        tools_used = [s.tool for s in r.plan]
        self.assertEqual(tools_used[-1], "answer_question")
        self.assertEqual(r.intent.primary, "qa")

    def test_garbage_hint_falls_back_to_heuristic(self):
        """非法 label 不影响——启发式照常工作。"""
        r = run_agent("帮我处理一下", dry_run=True, persist=False, intent_hint="garbage")
        # 启发式会落到 qa
        self.assertEqual(r.intent.primary, "qa")
        # 没加 user_hint 信号
        self.assertFalse(any("user_hint" in s for s in r.intent.signals))

    def test_hint_signal_appears_in_signals(self):
        r = run_agent("帮我处理一下", dry_run=True, persist=False, intent_hint="cite")
        self.assertTrue(any("user_hint:cite" in s for s in r.intent.signals))

    def test_hint_prevents_ask_user_loop(self):
        """如果用户已经答过 cite，conf=1.0，Replanner rule 0 不会再 ASK_USER。

        这是关键防御——回答澄清问题后必须能直接跑，不能再次问同样问题。
        """
        # 一个默认会触发 ASK_USER 的 goal（无领域词、无动词）
        r = run_agent("asdf ghjk", dry_run=True, persist=False, intent_hint="qa")
        # 状态应该是 planned（而不是 needs_clarification）
        self.assertEqual(r.status, "planned")
        # plan 应该是 qa 路径
        tools_used = [s.tool for s in r.plan]
        self.assertIn("answer_question", tools_used)


class BuildPlanBackwardsCompatTests(unittest.TestCase):
    """build_plan 行为不变——intent_hint 不传时走它。"""

    def test_no_hint_uses_build_plan(self):
        r = run_agent("推荐几篇引用", dry_run=True, persist=False)
        tools_used = [s.tool for s in r.plan]
        # build_plan: cite → recommend_citations（单步）
        self.assertEqual(tools_used, ["recommend_citations"])

    def test_build_plan_direct(self):
        p = build_plan("推荐几篇引用")
        self.assertEqual([s.tool for s in p], ["recommend_citations"])


if __name__ == "__main__":
    unittest.main()
