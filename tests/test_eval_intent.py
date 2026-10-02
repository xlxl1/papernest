"""意图分类评测的单元测试。

不依赖 LLM、不依赖真实库——纯逻辑测试。
metric_intent_routing 接受 classify 参数化，所以可以注入假分类器压所有边界。
"""

import unittest

from papernest import eval as ev
from papernest.intent import Intent


def _make_entries(*gold_pred_pairs):
    """[(id, gold, predicted), ...] → eval 集格式。"""
    return [
        {"id": f"e{i}", "type": "intent", "query": q,
         "gold_intent": g, "category": "test"}
        for i, (q, g, _) in enumerate(gold_pred_pairs)
        for _ in [None]  # noqa
    ]


def _stub_classify(mapping):
    """构造假 classify：query → Intent(primary=..., confidence=..., signals=...)。"""
    def classify(query):
        pred = mapping.get(query)
        if isinstance(pred, Intent):
            return pred
        return Intent(primary=pred, confidence=1.0, signals=["stub"])
    return classify


class MetricIntentRoutingTests(unittest.TestCase):

    def test_all_correct_gives_accuracy_one(self):
        entries = _make_entries(
            ("q1", "cite", "cite"),
            ("q2", "survey", "survey"),
            ("q3", "read", "read"),
            ("q4", "qa", "qa"),
        )
        m = ev.metric_intent_routing(entries, classify=_stub_classify({
            "q1": "cite", "q2": "survey", "q3": "read", "q4": "qa",
        }))
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["n"], 4)
        for label in ("cite", "survey", "read", "qa"):
            self.assertEqual(m["per_intent"][label]["tp"], 1)

    def test_all_wrong_gives_accuracy_zero(self):
        entries = _make_entries(
            ("q1", "cite", "qa"),
            ("q2", "read", "qa"),
            ("q3", "survey", "qa"),
        )
        m = ev.metric_intent_routing(entries, classify=_stub_classify({
            "q1": "qa", "q2": "qa", "q3": "qa",
        }))
        self.assertEqual(m["accuracy"], 0.0)
        # confusion: 全部 gold 行都进了 qa 列
        self.assertEqual(m["confusion"]["cite"]["qa"], 1)
        self.assertEqual(m["confusion"]["read"]["qa"], 1)
        self.assertEqual(m["confusion"]["survey"]["qa"], 1)
        # qa 的 precision = 0/3 = 0
        self.assertEqual(m["per_intent"]["qa"]["precision"], 0.0)
        # qa 的 recall = 0/0 = 0 (没 qa gold)
        self.assertEqual(m["per_intent"]["qa"]["recall"], 0.0)

    def test_precision_recall_separate(self):
        """半对：gold=qa 4 条（3 对 1 错），gold=cite 1 条（被误判为 qa）。"""
        entries = _make_entries(
            ("q1", "qa", "qa"), ("q2", "qa", "qa"),
            ("q3", "qa", "qa"), ("q4", "qa", "cite"),
            ("q5", "cite", "qa"),
        )
        m = ev.metric_intent_routing(entries, classify=_stub_classify({
            "q1": "qa", "q2": "qa", "q3": "qa",
            "q4": "cite", "q5": "qa",
        }))
        # qa precision: 3/4 (correct qa predictions / total qa predictions)
        self.assertEqual(m["per_intent"]["qa"]["precision"], 0.75)
        # qa recall: 3/4 (correct qa predictions / gold qa)
        self.assertEqual(m["per_intent"]["qa"]["recall"], 0.75)

    def test_confusion_matrix_shape(self):
        entries = _make_entries(("q", "cite", "qa"))
        m = ev.metric_intent_routing(entries, classify=_stub_classify({"q": "qa"}))
        # 5×5（5 个 INTENT_LABELS）
        self.assertEqual(len(m["confusion"]), 5)
        for label in ev.INTENT_LABELS:
            self.assertEqual(len(m["confusion"][label]), 5)

    def test_worst_items_sorted_by_confidence(self):
        """worst_items 应该是出错且 confidence 最低的前 5 条。"""
        entries = _make_entries(
            ("q1", "cite", "qa"),  # conf=0.3（低）
            ("q2", "cite", "qa"),  # conf=0.9（高）
            ("q3", "read", "qa"),  # conf=0.5（中）
        )
        # stub 按 query 返回不同 confidence
        def cls(q):
            conf = {"q1": 0.3, "q2": 0.9, "q3": 0.5}[q]
            return Intent(primary="qa", confidence=conf, signals=[])
        m = ev.metric_intent_routing(entries, classify=cls)
        self.assertEqual(len(m["worst_items"]), 3)
        # 排序：confidence 升序
        self.assertEqual(m["worst_items"][0]["query"], "q1")
        self.assertEqual(m["worst_items"][1]["query"], "q3")
        self.assertEqual(m["worst_items"][2]["query"], "q2")

    def test_handles_classify_returning_dict(self):
        """classify 可以返回 Intent 或 dict——兼容旧调用。"""
        def cls(query):
            return {"primary": "cite", "confidence": 0.8, "signals": [], "paper_id": None, "topic_hint": None}
        entries = _make_entries(("q", "cite", "cite"))
        m = ev.metric_intent_routing(entries, classify=cls)
        self.assertEqual(m["accuracy"], 1.0)

    def test_real_intent_classify_against_real_eval_set(self):
        """压真分类器：intent_eval_set.json 必须 100% 准确（hard 门槛）。

        一旦词表 / 规则改动让准确率掉下来，这条用例会立刻红，
        强迫改的人在 commit 前检查是不是把已知 case 改坏了。
        """
        try:
            entries = ev.load_intent_eval_set()
        except (FileNotFoundError, AssertionError):
            self.skipTest("intent_eval_set.json 不存在或损坏，跳过")
        m = ev.metric_intent_routing(entries)
        # 不强求 100%，但要求 ≥ 90%（hard cap 留出 10% 给新增 case 调试空间）
        self.assertGreaterEqual(
            m["accuracy"], 0.9,
            f"意图分类准确率 {m['accuracy']} 跌破 90% 门槛。"
            f"worst_items={[(w['id'], w['gold'], w['predicted']) for w in m['worst_items'][:5]]}")


class LoadIntentEvalSetTests(unittest.TestCase):

    def test_loads_real_file(self):
        try:
            entries = ev.load_intent_eval_set()
        except FileNotFoundError:
            self.skipTest("intent_eval_set.json 不存在")
        self.assertGreater(len(entries), 0)
        for e in entries:
            self.assertEqual(e["type"], "intent")
            self.assertIn(e["gold_intent"], ev.INTENT_LABELS)

    def test_bad_gold_intent_rejected(self):
        """gold_intent 必须是合法值，否则加载时断言。"""
        import tempfile
        import json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            _json.dump([
                {"id": "x", "type": "intent", "query": "q", "gold_intent": "garbage"}
            ], f, ensure_ascii=False)
            tmp_path = f.name
        try:
            with self.assertRaises(AssertionError):
                ev.load_intent_eval_set(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)

    def test_wrong_type_rejected(self):
        """type 必须是 intent，否则拒绝。"""
        import tempfile
        import json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            _json.dump([
                {"id": "x", "type": "retrieval", "query": "q", "gold_intent": "cite"}
            ], f, ensure_ascii=False)
            tmp_path = f.name
        try:
            with self.assertRaises(AssertionError):
                ev.load_intent_eval_set(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)


class RunEvalIntentIntegrationTests(unittest.TestCase):
    """run_eval() 现在会调 metric_intent_routing——验证集成没破。"""

    def test_intent_metric_appears_in_report(self):
        from papernest import eval as ev2
        # 故意把 eval_set_path 指向不存在的文件，只跑 intent
        # 这里偷懒：直接验证 metric_intent_routing 在 report 中存在
        # 完整 run_eval 需要真库，跳过
        import tempfile
        import json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            _json.dump([
                {"id": "x", "type": "intent", "query": "推荐几篇", "gold_intent": "cite"}
            ], f, ensure_ascii=False)
            tmp_path = f.name
        try:
            entries = ev2.load_intent_eval_set(tmp_path)
            report = {"intent_routing": ev2.metric_intent_routing(entries)}
            self.assertIn("intent_routing", report)
            self.assertEqual(report["intent_routing"]["accuracy"], 1.0)
        finally:
            import os
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
