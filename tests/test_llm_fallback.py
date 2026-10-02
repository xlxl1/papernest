"""Plan-Execute-Replan LLM 兜底路径的测试。

PAPERNEST_INTENT=llm / PAPERNEST_REPLANNER=llm 时，启发式信心不足会升级到 LLM。
LLM 不可用时**必须**退回启发式，不能抛异常。

测两类：
1. **LLM 不可用**（LLMUnavailable）→ 退回启发式，不影响主流程
2. **LLM 可用** + 各种响应 → 正确解析 / 正确比较 / 正确合并

用 mock 替掉 llm.chat / llm.available / llm.extract_json，0 token 真实调用。
"""

import json
import os
import unittest
from unittest import mock

from papernest import intent, replan


def _patch_llm(monkeypatch, *, available=True, chat_returns=None, extract_returns=None):
    """monkeypatch llm 模块的三个外部依赖。"""
    monkeypatch.setattr(intent.llm, "available", lambda: available)
    if chat_returns is not None:
        monkeypatch.setattr(intent.llm, "chat",
                            lambda *a, **kw: chat_returns)
    if extract_returns is not None:
        monkeypatch.setattr(intent.llm, "extract_json",
                            lambda text: extract_returns)


class IntentLLMFallbackTests(unittest.TestCase):

    def setUp(self):
        # 默认开启 LLM 兜底；setUp 不写，单个用例自己 env
        self._env_backup = os.environ.get("PAPERNEST_INTENT")
        os.environ["PAPERNEST_INTENT"] = "llm"
        # 重新加载 intent 模块以让模块级常量重新读取 env
        import importlib
        importlib.reload(intent)
        # reload 后 _LLM_MODE 应该已经是 True
        self.assertTrue(intent._LLM_MODE)

    def tearDown(self):
        if self._env_backup is None:
            os.environ.pop("PAPERNEST_INTENT", None)
        else:
            os.environ["PAPERNEST_INTENT"] = self._env_backup
        # 还原 intent 模块状态
        import importlib
        importlib.reload(intent)

    def test_llm_unavailable_falls_back_to_heuristic(self):
        """LLM 不可用（available()=False）→ 退回启发式，不抛异常。"""
        with mock.patch.object(intent.llm, "available", return_value=False):
            result = intent.classify_with_llm_fallback("精读论文 12")
            # 应该走启发式，read + paper_id=12
            self.assertEqual(result.primary, "read")
            self.assertEqual(result.paper_id, 12)

    def test_llm_unavailable_no_chat_call(self):
        """LLM 不可用时不调 chat——节省 token。"""
        with mock.patch.object(intent.llm, "available", return_value=False), \
             mock.patch.object(intent.llm, "chat") as mock_chat:
            intent.classify_with_llm_fallback("LLM Agent 方法论")
            mock_chat.assert_not_called()

    def test_high_heuristic_confidence_skips_llm(self):
        """启发式 confidence ≥ 0.7 时不调 LLM——节省 token。"""
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat") as mock_chat:
            # 精读论文 12 → confidence 0.9（强 read 信号）
            result = intent.classify_with_llm_fallback("精读论文 12")
            mock_chat.assert_not_called()
            self.assertEqual(result.primary, "read")

    def test_low_heuristic_confidence_calls_llm(self):
        """启发式 confidence < 0.7 且 LLM 可用时调一次 LLM。"""
        llm_response = json.dumps({
            "primary": "cite", "confidence": 0.95,
            "reason": "用户提到要补充引用"
        })
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat", return_value=llm_response) as mock_chat, \
             mock.patch.object(intent.llm, "extract_json",
                               return_value={"primary": "cite", "confidence": 0.95,
                                             "reason": "用户提到要补充引用"}):
            # default qa 路径 confidence=0.5 < 0.7，会升级到 LLM
            result = intent.classify_with_llm_fallback("补几篇引用")
            mock_chat.assert_called_once()
            self.assertEqual(result.primary, "cite")
            self.assertGreaterEqual(result.confidence, 0.9)
            # signals 里要能追到 LLM 的判据
            self.assertTrue(any("llm:" in s for s in result.signals))

    def test_llm_lower_confidence_rejected(self):
        """LLM 给的 confidence 不如启发式 → 用启发式。

        这种场景下 LLM 反而是不确定的那个，强行用它的结果会回退。
        """
        llm_response = json.dumps({
            "primary": "qa", "confidence": 0.4, "reason": "我不确定"
        })
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat", return_value=llm_response), \
             mock.patch.object(intent.llm, "extract_json",
                               return_value={"primary": "qa", "confidence": 0.4,
                                             "reason": "我不确定"}):
            # default qa confidence 0.5 > LLM 的 0.4 → 用启发式
            result = intent.classify_with_llm_fallback("LLM Agent 方法论")
            self.assertEqual(result.primary, "qa")
            self.assertEqual(result.confidence, 0.5)  # 启发式，不是 LLM 的 0.4

    def test_llm_invalid_primary_rejected(self):
        """LLM 返回的 primary 不在合法集合里 → 退回启发式。"""
        llm_response = json.dumps({
            "primary": "thinking", "confidence": 0.9, "reason": "I'm thinking"
        })
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat", return_value=llm_response), \
             mock.patch.object(intent.llm, "extract_json",
                               return_value={"primary": "thinking",
                                             "confidence": 0.9,
                                             "reason": "I'm thinking"}):
            result = intent.classify_with_llm_fallback("补几篇引用")
            self.assertEqual(result.primary, "qa")  # 退回 default qa

    def test_llm_chat_raises_llmerror_falls_back(self):
        """llm.chat 抛 LLMError → 退回启发式。"""
        from papernest.llm import LLMError
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat",
                               side_effect=LLMError("rate limit")):
            result = intent.classify_with_llm_fallback("补几篇引用")
            # 启发式结果
            self.assertEqual(result.primary, "qa")

    def test_llm_extract_json_raises_falls_back(self):
        """llm.extract_json 抛异常 → 退回启发式。"""
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat", return_value="garbled"), \
             mock.patch.object(intent.llm, "extract_json",
                               side_effect=ValueError("bad json")):
            result = intent.classify_with_llm_fallback("补几篇引用")
            self.assertEqual(result.primary, "qa")

    def test_off_mode_does_not_call_llm(self):
        """PAPERNEST_INTENT != llm → 完全不调 LLM（默认行为）。"""
        import importlib
        os.environ["PAPERNEST_INTENT"] = "off"
        importlib.reload(intent)
        self.assertFalse(intent._LLM_MODE)
        with mock.patch.object(intent.llm, "available", return_value=True), \
             mock.patch.object(intent.llm, "chat") as mock_chat:
            result = intent.classify_with_llm_fallback("补几篇引用")
            mock_chat.assert_not_called()
            self.assertEqual(result.primary, "qa")


class ReplanLLMFallbackTests(unittest.TestCase):

    def setUp(self):
        self._env_backup = os.environ.get("PAPERNEST_REPLANNER")
        os.environ["PAPERNEST_REPLANNER"] = "llm"
        import importlib
        importlib.reload(replan)
        self.assertTrue(replan._LLM_MODE)

    def tearDown(self):
        if self._env_backup is None:
            os.environ.pop("PAPERNEST_REPLANNER", None)
        else:
            os.environ["PAPERNEST_REPLANNER"] = self._env_backup
        import importlib
        importlib.reload(replan)

    def _state(self, **overrides):
        """构造一个 fallback 测试用的最小 state。"""
        s = {
            "goal": "推荐支持某段的引用",
            "intent": intent.Intent(primary="cite", confidence=0.5, signals=[]),
            "top_k": 5,
            "steps_done": [],
            "last_step": None,
            "last_observation": None,
            "last_status": "completed",
            "has_answer": False,
            "timeout_count": 0,
            "failed_tools": [],
            "plan_remaining": 0,
        }
        s.update(overrides)
        return s

    def test_high_confidence_decision_skips_llm(self):
        """rule 1 已生成 answer → 置信度 1.0，不调 LLM。"""
        from papernest.schemas import PlanStep
        s = self._state(has_answer=True)
        s["steps_done"] = [(PlanStep(id=1, tool="retrieve_library", reason="", args={}),
                            {"paper_ids": [1], "papers": [{"id": 1}]})]
        with mock.patch.object(replan.llm, "available", return_value=True), \
             mock.patch.object(replan.llm, "chat") as mock_chat:
            d = replan.decide_with_llm_fallback(s)
            mock_chat.assert_not_called()
            self.assertEqual(d.action, "finish")

    def test_ask_user_decision_skips_llm(self):
        """ask_user 是终端决策，LLM 也救不回来——直接用规则结果。"""
        s = self._state(intent=intent.Intent(primary="unknown", confidence=0.2,
                                            signals=[]))
        with mock.patch.object(replan.llm, "available", return_value=True), \
             mock.patch.object(replan.llm, "chat") as mock_chat:
            d = replan.decide_with_llm_fallback(s)
            mock_chat.assert_not_called()
            self.assertEqual(d.action, "ask_user")

    def test_low_confidence_continue_calls_llm(self):
        """heuristic 给 continue 但 conf=0.6（< 0.7）→ 调 LLM。"""
        s = self._state()
        # 先让规则返回一个 conf=0.6 的结果——办法：传入一个 conf=0.6 但确实是 continue 的场景
        # 最简单的办法：构造一个完全空的 state，让规则继续走 default 但故意降低 confidence
        # 但 default rule confidence=1.0，所以得构造一个能拿到 conf=0.6 的场景
        # 这里用 mock 替掉 decide 让它返回指定值
        with mock.patch.object(replan, "decide") as mock_decide, \
             mock.patch.object(replan.llm, "available", return_value=True), \
             mock.patch.object(replan.llm, "chat", return_value='{"action":"continue","reason":"llm说继续","confidence":0.8}'), \
             mock.patch.object(replan.llm, "extract_json",
                               return_value={"action": "continue",
                                             "reason": "llm说继续",
                                             "confidence": 0.8}):
            from papernest.replan import ReplanDecision
            mock_decide.return_value = ReplanDecision(action="continue", reason="heuristic", confidence=0.6)
            d = replan.decide_with_llm_fallback(s)
            self.assertEqual(d.action, "continue")
            self.assertIn("llm", d.reason.lower())

    def test_llm_error_falls_back_to_rule(self):
        """LLM 抛错 → 用规则结果，不抛异常。"""
        from papernest.replan import ReplanDecision
        from papernest.llm import LLMError
        with mock.patch.object(replan, "decide") as mock_decide, \
             mock.patch.object(replan.llm, "available", return_value=True), \
             mock.patch.object(replan.llm, "chat", side_effect=LLMError("rate limit")):
            rule_result = ReplanDecision(action="continue", reason="rule", confidence=0.6)
            mock_decide.return_value = rule_result
            d = replan.decide_with_llm_fallback(self._state())
            # 应该用规则结果
            self.assertEqual(d.reason, "rule")

    def test_llm_invalid_action_falls_back(self):
        """LLM 返回非法 action → 退回规则结果。"""
        from papernest.replan import ReplanDecision
        with mock.patch.object(replan, "decide") as mock_decide, \
             mock.patch.object(replan.llm, "available", return_value=True), \
             mock.patch.object(replan.llm, "chat", return_value='{"action":"dance","confidence":0.8}'), \
             mock.patch.object(replan.llm, "extract_json",
                               return_value={"action": "dance", "confidence": 0.8}):
            rule_result = ReplanDecision(action="continue", reason="rule", confidence=0.6)
            mock_decide.return_value = rule_result
            d = replan.decide_with_llm_fallback(self._state())
            self.assertEqual(d.reason, "rule")


if __name__ == "__main__":
    unittest.main()
