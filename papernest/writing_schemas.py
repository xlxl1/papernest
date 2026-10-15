"""Structured contracts shared by the writing agents.

Keeping claims and evidence explicit is what turns the writing flow from a
sequence of prompts into an auditable multi-agent protocol.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=80)
    paper_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=20_000)
    page: int | None = Field(default=None, ge=1)
    section: str = Field(default="", max_length=300)
    block_type: Literal["text", "table", "formula", "figure", "reference"] = "text"
    verified: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_kind: Literal["native_page", "imported_fulltext", "abstract",
                        "model_summary", "unverified_recall"] = "unverified_recall"
    page_verified: bool = False


class Claim(BaseModel):
    claim_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=5000)
    claim_type: str = Field(default="factual", max_length=80)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    status: Literal["supported", "partial", "unsupported", "needs_review"] = "needs_review"


class EvidenceBundle(BaseModel):
    section_id: str = Field(min_length=1, max_length=80)
    query: str = Field(default="", max_length=5000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    claims: list[Claim] = Field(default_factory=list, max_length=100)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    agent: str = "evidence_agent"


class AgentMessage(BaseModel):
    message_id: str = Field(min_length=1, max_length=80)
    sender: str = Field(min_length=1, max_length=80)
    recipient: str = Field(min_length=1, max_length=80)
    message_type: str = Field(min_length=1, max_length=100)
    payload: dict = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_review: bool = False


class ConsistencyIssue(BaseModel):
    issue_type: Literal["numeric_conflict", "term_conflict", "claim_conflict",
                         "citation_gap", "provenance_weak"] = "citation_gap"
    locations: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1, max_length=2000)
    severity: Literal["major", "minor", "suggestion"] = "minor"
    fix: str = Field(default="", max_length=2000)
