"""Budget guard pre-flight and stop-reason tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from papernest import config
from papernest.writing_benchmark import (
    _check_budget_guard_feasibility,
    _enforce_budget,
    run_benchmark,
)


def _dataset():
    return {"schema_version": 1, "name": "fixture", "topics": [
        {"id": "t1", "query": "fixture", "title": "Fixture",
         "sections": [{"id": "t1-s1", "no": 1, "title": "Results",
                        "points": ["Report the result."],
                        "source_keys": ["s1"],
                        "claims": [{"claim_id": "t1-c1",
                                     "text": "method 82% accuracy",
                                     "source_key": "s1",
                                     "quote": "method 82% accuracy"}]}]}],
            "sources": [{"source_key": "s1", "norm_key": "arxiv:0001",
                         "title": "S1", "abstract": "method 82% accuracy"}]}


class BudgetGuardTests(unittest.TestCase):
    def setUp(self):
        self._old_prices = config.LLM_PRICES_JSON
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        tmp = Path(tempfile.mkdtemp(prefix="papernest_budget_"))
        self._tmp = tmp
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        config.DB_PATH = tmp / "test.db"
        config.DATA_DIR = tmp / "data"

    def tearDown(self):
        config.LLM_PRICES_JSON = self._old_prices
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def test_cost_guard_requires_prices(self):
        config.LLM_PRICES_JSON = ""
        guard = {"max_total_cost_usd": 0.5}
        msg = _check_budget_guard_feasibility(guard)
        self.assertIsNotNone(msg)
        self.assertIn("LLM_PRICES_JSON", msg)

    def test_cost_guard_passes_when_prices_set(self):
        config.LLM_PRICES_JSON = json.dumps({"default": [0.0, 0.0]})
        msg = _check_budget_guard_feasibility({"max_total_cost_usd": 0.5})
        self.assertIsNone(msg)

    def test_no_guard_is_feasible(self):
        self.assertIsNone(_check_budget_guard_feasibility({}))
        self.assertIsNone(_check_budget_guard_feasibility(None))

    def test_budget_state_enforcement(self):
        import time as _time
        state = {"started_at": _time.perf_counter(), "cost_usd": 0.0, "llm_calls": 0,
                 "prompt_tokens": 0, "completion_tokens": 0,
                 "max_total_cost_usd": 1.0, "max_llm_calls": 2,
                 "max_total_tokens": 1000, "max_wallclock_s": 3600}
        # Cost under the cap ⇒ keep going
        self.assertFalse(_enforce_budget(state))
        # Cost above the cap ⇒ stop
        state["cost_usd"] = 1.5
        self.assertTrue(_enforce_budget(state))
        self.assertIn("cost_usd", state["stop_reason"])
        # Reset cost, push LLM calls over the cap ⇒ stop on calls
        state["cost_usd"] = 0.0
        state["llm_calls"] = 5
        self.assertTrue(_enforce_budget(state))
        self.assertIn("llm_calls", state["stop_reason"])

    def test_run_benchmark_rejects_cost_guard_without_prices(self):
        config.LLM_PRICES_JSON = ""
        with TemporaryDirectory() as td:
            path = Path(td) / "eval.json"
            path.write_text(json.dumps(_dataset()), encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                run_benchmark(path, modes=("single",), offline=True,
                              budget_guard={"max_total_cost_usd": 0.5})
        self.assertIn("LLM_PRICES_JSON", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
