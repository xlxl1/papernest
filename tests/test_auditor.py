"""Offline regression tests for claim, citation and consistency auditing."""
from __future__ import annotations

import copy
import json
import unittest
from unittest import mock

from papernest import auditor, pipeline
from papernest.writing_schemas import ConsistencyIssue, EvidenceBundle


def _bundle() -> dict:
    return {
        "section_id": "section-1",
        "evidence": [
            {"evidence_id": "e1", "paper_id": 1,
             "quote": "Success rate was 82% at 4 steps.",
             "page": 5, "verified": True, "confidence": 1.0},
        ],
        "claims": [
            {"claim_id": "c1", "text": "Success rate was 82% at 4 steps.",
             "evidence_ids": ["e1"], "status": "supported"},
        ],
        "coverage": 1.0,
    }


def _citations() -> dict:
    return {"1": {"paper_id": 1, "verified": True}}


class AuditorTests(unittest.TestCase):
    def audit(self, text: str, bundle=None, citations=None, glossary=None):
        return auditor.audit_section(
            text, _bundle() if bundle is None else bundle,
            _citations() if citations is None else citations, glossary,
            section_id="section-1")

    def test_verified_claim_bound_citation_passes_without_external_calls(self):
        with mock.patch("papernest.llm.chat",
                        side_effect=AssertionError("audit must remain offline")), \
             mock.patch("papernest.embeddings.embed_texts",
                        side_effect=AssertionError("audit must remain offline")):
            issues = self.audit("Success rate was 82% at 4 steps [1].")
        self.assertEqual(issues, [])

    def test_verified_but_unbound_paper_is_a_citation_gap(self):
        bundle = _bundle()
        bundle["evidence"].append({
            "evidence_id": "unbound", "paper_id": 2,
            "quote": "Ocean temperatures are increasing.", "verified": True})
        issues = self.audit(
            "Ocean temperatures are increasing [2].", bundle,
            {**_citations(), "2": {"paper_id": 2, "verified": True}})
        self.assertIn("citation_gap", {i.issue_type for i in issues})
        self.assertTrue(any("2" in i.description for i in issues))

    def test_unverified_bound_evidence_cannot_support_a_citation(self):
        bundle = _bundle()
        bundle["evidence"][0]["verified"] = False
        issues = self.audit("Success rate was 82% at 4 steps [1].", bundle)
        self.assertIn("citation_gap", {i.issue_type for i in issues})

    def test_recognized_claim_without_a_citation_is_a_citation_gap(self):
        issues = self.audit("Success rate was 82% at 4 steps.")
        gaps = [i for i in issues if i.issue_type == "citation_gap"]
        self.assertTrue(gaps)
        self.assertTrue(any("c1" in i.description or "主张" in i.description
                            for i in gaps))

    def test_dangling_citation_has_a_structured_location_and_fix(self):
        issues = self.audit("Unsupported statement [9].")
        gaps = [i for i in issues if i.issue_type == "citation_gap"]
        self.assertTrue(gaps)
        self.assertTrue(all(isinstance(i, ConsistencyIssue) for i in gaps))
        self.assertTrue(all(i.locations for i in gaps))
        self.assertTrue(any("section-1" in " ".join(i.locations) for i in gaps))
        self.assertTrue(all(i.fix for i in gaps))

    def test_identified_claim_cannot_cite_a_different_bound_paper(self):
        bundle = _bundle()
        bundle["evidence"].append({
            "evidence_id": "e2", "paper_id": 2,
            "quote": "Ocean temperatures are increasing.", "verified": True})
        bundle["claims"].append({
            "claim_id": "c2", "text": "Ocean temperatures are increasing.",
            "evidence_ids": ["e2"], "status": "supported"})
        issues = self.audit("Success rate was 82% at 4 steps [2].", bundle,
                            {**_citations(), "2": {"paper_id": 2, "verified": True}})
        self.assertIn("claim_conflict", {i.issue_type for i in issues})

    def test_changed_percentage_is_numeric_conflict(self):
        issues = self.audit("Success rate was 91% at 4 steps [1].")
        conflicts = [i for i in issues if i.issue_type == "numeric_conflict"]
        self.assertTrue(conflicts)
        self.assertTrue(any("91%" in i.description for i in conflicts))
        self.assertTrue(all(i.severity == "major" for i in conflicts))

    def test_other_papers_numbers_cannot_support_the_current_citation(self):
        bundle = _bundle()
        bundle["evidence"].append({
            "evidence_id": "e2", "paper_id": 2,
            "quote": "A separate benchmark reported a success rate of 91%.",
            "verified": True})
        bundle["claims"].append({
            "claim_id": "c2", "text": "A separate benchmark reported 91%.",
            "evidence_ids": ["e2"], "status": "supported"})
        issues = self.audit("Success rate was 91% at 4 steps [1].", bundle,
                            {**_citations(), "2": {"paper_id": 2, "verified": True}})
        self.assertIn("numeric_conflict", {i.issue_type for i in issues})

    def test_unbound_numbers_cannot_launder_a_changed_value(self):
        bundle = _bundle()
        bundle["evidence"].append({
            "evidence_id": "unused", "paper_id": 1,
            "quote": "Success rate was 91% on a different task.", "verified": True})
        issues = self.audit("Success rate was 91% at 4 steps [1].", bundle)
        self.assertIn("numeric_conflict", {i.issue_type for i in issues})

    def test_plain_number_does_not_justify_a_percentage(self):
        bundle = _bundle()
        bundle["evidence"][0]["quote"] = "There were 82 trials at 4 steps."
        issues = self.audit("Success rate was 82% at 4 steps [1].", bundle)
        self.assertIn("numeric_conflict", {i.issue_type for i in issues})

    def test_headings_citation_indices_and_uncited_planning_dates_are_not_numbers_to_audit(self):
        text = ("## 1. Results\n\n"
                "Success rate was 82% at 4 steps [1].\n\n"
                "In 2026, the following topics are planned.")
        self.assertEqual(self.audit(text), [])

    def test_explicit_term_alias_is_reported_with_canonical_fix(self):
        issues = self.audit("We discuss large language models.", glossary={
            "LLM": {"aliases": ["large language models"]}})
        terms = [i for i in issues if i.issue_type == "term_conflict"]
        self.assertTrue(terms)
        self.assertTrue(any("large language models" in i.description for i in terms))
        self.assertTrue(any("LLM" in i.fix for i in terms))

    def test_spacing_and_case_term_variants_are_reported(self):
        issues = self.audit("Tool Use reliability requires evaluation.",
                            glossary={"tool-use": "3 篇文献的关键词"})
        self.assertIn("term_conflict", {i.issue_type for i in issues})

    def test_canonical_terms_and_legacy_glossary_descriptions_do_not_trigger_conflicts(self):
        glossary = {"tool-use": "3 篇文献的关键词"}
        self.assertEqual(self.audit("We discuss tool-use reliability.", glossary=glossary), [])
        self.assertEqual(self.audit("这里包含3 篇文献的关键词。", glossary=glossary), [])

    def test_document_accepts_persisted_json_rows_and_preserves_section_location(self):
        sections = [
            {"sec_no": 1, "content": "Success rate was 82% at 4 steps [1].",
             "evidence_json": json.dumps(_bundle()),
             "citations_json": json.dumps(_citations())},
            {"sec_no": 2, "content": "Success rate was 91% at 4 steps [1].",
             "evidence_json": json.dumps(_bundle()),
             "citations_json": json.dumps(_citations())},
        ]
        issues = auditor.audit_document(sections)
        conflicts = [i for i in issues if i.issue_type == "numeric_conflict"]
        self.assertTrue(conflicts)
        self.assertTrue(all("section-2" in " ".join(i.locations) for i in conflicts))

    def test_contract_models_and_inferred_citation_mapping_are_supported(self):
        bundle = EvidenceBundle.model_validate(_bundle())
        issues = auditor.audit_section("Success rate was 82% at 4 steps [1].", bundle)
        self.assertEqual(issues, [])

    def test_audit_does_not_mutate_shared_evidence_or_citation_inputs(self):
        bundle, citations = _bundle(), _citations()
        original = copy.deepcopy((bundle, citations))
        self.audit("Success rate was 91% at 4 steps [1].", bundle, citations)
        self.assertEqual((bundle, citations), original)

    def test_final_check_exposes_machine_audit_in_the_report(self):
        report = pipeline.final_check(
            "Success rate was 91% at 4 steps [1].",
            {1: {"paper_id": 1, "verified": True}},
            sections=[{"sec_no": 1, "content": "Success rate was 91% at 4 steps [1].",
                        "evidence_bundle": _bundle(), "citations": _citations()}])
        self.assertIn("audit", report)
        self.assertFalse(report["audit_ok"])
        self.assertGreaterEqual(report["audit"]["major_count"], 1)


if __name__ == "__main__":
    unittest.main()
