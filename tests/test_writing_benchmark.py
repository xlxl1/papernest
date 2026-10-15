"""Offline runner, mode comparison and failure-isolation contract tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from papernest import config
from papernest.writing_benchmark import run_benchmark

from .support import TempDbTestCase


def _source(key="s1"):
    return {"source_key": key, "norm_key": f"fixture:{key}", "title": "Fixture",
            "abstract": "The method achieved 82% accuracy on 4 tasks.",
            "year": 2025, "authors": ["Author"]}


def _topic(topic_id="ok", source_key="s1"):
    return {"id": topic_id, "query": "fixture topic", "title": "Fixture",
            "sections": [{"id": topic_id + "-s1", "no": 1, "title": "Results",
                           "points": ["Report the result."], "source_keys": [source_key],
                           "claims": [{"claim_id": topic_id + "-c1",
                                        "text": "The method achieved 82% accuracy on 4 tasks.",
                                        "source_key": source_key,
                                        "quote": "The method achieved 82% accuracy on 4 tasks.",
                                        "numbers": ["82%", "4"]}]}]}


def _dataset():
    return {"schema_version": 1, "name": "fixture", "topics": [_topic("ok"),
            _topic("second")], "sources": [_source()]}


class WritingBenchmarkTests(TempDbTestCase):
    prefix = "papernest_writing_benchmark"

    def test_runner_is_offline_and_restores_config(self):
        original_db, original_dir = config.DB_PATH, config.DATA_DIR
        with TemporaryDirectory() as td:
            path = Path(td) / "eval.json"
            path.write_text(json.dumps(_dataset()), encoding="utf-8")
            with mock.patch("papernest.llm.chat", side_effect=AssertionError("LLM forbidden")), \
                 mock.patch("papernest.embeddings.embed_texts", side_effect=AssertionError("vectors forbidden")), \
                 mock.patch("papernest.http.client", side_effect=AssertionError("network forbidden")):
                report = run_benchmark(path, modes=("single",), offline=True)
        self.assertEqual((config.DB_PATH, config.DATA_DIR), (original_db, original_dir))
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(len(report["items"]), 2)
        self.assertTrue(all(item["error"] is None for item in report["items"]))

    def test_all_modes_are_reported_and_are_comparable(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "eval.json"
            path.write_text(json.dumps(_dataset()), encoding="utf-8")
            report = run_benchmark(path, modes=("single", "serial", "multi"),
                                   limit=1, offline=True)
        self.assertEqual(set(report["by_mode"]), {"single", "serial", "multi"})
        self.assertEqual(len(report["items"]), 3)
        self.assertTrue(all(item["metrics"]["llm_calls"] == 0 for item in report["items"]))
        for aggregate in report["by_mode"].values():
            self.assertIn("evidence_coverage", aggregate)
            self.assertIn("citation_precision", aggregate)
            self.assertIn("citation_string_match", aggregate)
            self.assertIn("usage_known", aggregate)
            self.assertIn("cost_usd", aggregate)

    def test_one_bad_topic_does_not_hide_other_topic_and_writes_report(self):
        data = _dataset()
        with TemporaryDirectory() as td:
            path = Path(td) / "eval.json"
            out = Path(td) / "report.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            # Inject one runtime failure after schema validation.  The runner
            # must preserve the successful topic and persist both outcomes.
            import papernest.writing_benchmark as wb
            real = wb.benchmark_topic
            def flaky(topic, sources, mode, offline=True, max_rewrites=2,
                      budget_state=None):
                if topic.get("id") == "ok":
                    raise RuntimeError("fixture failure")
                return real(topic, sources, mode, offline=offline,
                            max_rewrites=max_rewrites,
                            budget_state=budget_state)
            with mock.patch.object(wb, "benchmark_topic", side_effect=flaky):
                report = run_benchmark(path, modes=("multi",), offline=True, out_path=out)
            persisted = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(report["items"]), 2)
        self.assertEqual(report["failure_count"], 1)
        self.assertTrue(any(item["error"] for item in report["items"]))
        self.assertTrue(any(item["error"] is None for item in report["items"]))
        self.assertEqual(persisted["failure_count"], 1)

    def test_malformed_dataset_is_rejected_before_running_agents(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "malformed.json"
            path.write_text(json.dumps({"schema_version": 1, "topics": []}), encoding="utf-8")
            with self.assertRaises((ValueError, TypeError, KeyError)):
                run_benchmark(path, modes=("single",), offline=True)


if __name__ == "__main__":
    unittest.main()
