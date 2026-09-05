"""Typed request/response models shared by the Agent API and orchestrator.

Keeping these models separate from the FastAPI module makes the workflow easy
to test without starting a web server and gives every tool a stable contract.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_steps: int = Field(default=5, ge=1, le=10)
    topic: str | None = Field(default=None, max_length=500)
    timeout_s: int = Field(default=240, ge=5, le=900)
    dry_run: bool = False


class PlanStep(BaseModel):
    id: int = Field(ge=1)
    tool: str
    reason: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolEvent(BaseModel):
    step_id: int
    tool: str
    status: Literal["running", "completed", "failed", "skipped", "timeout"]
    latency_ms: int = 0
    attempt: int = 1
    summary: str = ""
    error: str | None = None


class AgentResponse(BaseModel):
    run_id: str
    status: Literal["planned", "completed", "blocked", "failed"]
    goal: str
    plan: list[PlanStep]
    events: list[ToolEvent] = Field(default_factory=list)
    answer: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    degraded: str | None = None
    error: str | None = None
    total_ms: int = 0


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    topic: str = Field(default="", max_length=500)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    topic: str | None = Field(default=None, max_length=500)


class NoteCreate(BaseModel):
    title: str = Field(default="未命名笔记", max_length=200)
    content: str = Field(default="", max_length=100_000)
    note_type: str = Field(default="insight", max_length=40)
    tags: list[str] = Field(default_factory=list, max_length=20)
    paper_id: int | None = Field(default=None, ge=1)


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, max_length=100_000)
    note_type: str | None = Field(default=None, max_length=40)
    tags: list[str] | None = Field(default=None, max_length=20)
    paper_id: int | None = Field(default=None, ge=1)


class DocumentCreate(BaseModel):
    title: str = Field(default="未命名文档", max_length=200)
    content: str = Field(default="", max_length=300_000)
    status: str = Field(default="draft", max_length=30)


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, max_length=300_000)
    status: str | None = Field(default=None, max_length=30)


class SourceAttach(BaseModel):
    paper_id: int = Field(ge=1)
    page_no: int | None = Field(default=None, ge=1)
    quote: str = Field(default="", max_length=10_000)
    relation: str = Field(default="support", max_length=40)


class WritingRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=10_000)
    mode: Literal["outline", "draft", "polish", "review", "expand", "summarize"] = "draft"
    paper_ids: list[int] = Field(default_factory=list, max_length=30)
    save: bool = True


class WriteRunCreate(BaseModel):
    """二期写作流水线：创建 run 并启动选题 Agent。"""
    topic: str = Field(min_length=4, max_length=2000)
    project_id: int | None = Field(default=None, ge=1)
    target_words: int = Field(default=3000, ge=500, le=20_000)
    max_rewrites: int = Field(default=2, ge=0, le=3)


class WriteTopicSelect(BaseModel):
    """检查点①：人工定题（可直接采用候选题目或改写）。"""
    title: str = Field(min_length=4, max_length=300)


class WriteOutlineSection(BaseModel):
    no: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    points: list[str] = Field(default_factory=list, max_length=10)
    paper_ids: list[int] = Field(default_factory=list, max_length=30)
    words: int = Field(default=600, ge=100, le=3000)
    no_support: bool = False


class WriteOutlineUpdate(BaseModel):
    """检查点②：人工改纲（版本 +1；不传 sections 表示不改直接开写）。"""
    sections: list[WriteOutlineSection] = Field(min_length=2, max_length=8)
