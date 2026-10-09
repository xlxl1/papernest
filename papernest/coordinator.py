"""Persistent coordinator for the multi-agent writing workflow.

The existing ``jobs`` module owns process-level execution, retries and
cancellation.  This module owns the *writing run* protocol: per-agent state,
structured hand-off messages, deterministic next-step planning, and bounded
parallel evidence preparation.  Keeping those concerns separate makes a run
replayable after a worker restart without coupling the database task runner to
the writing agents.
"""
from __future__ import annotations

import concurrent.futures
import json
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from . import db
from .writing_schemas import AgentMessage


DEFAULT_AGENTS = ("topic", "outline", "evidence", "writer", "reviewer", "auditor", "polisher")
_ORDER = DEFAULT_AGENTS


def _dump(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False,
                      sort_keys=True, default=str)


class Coordinator:
    """State and message facade for one ``writing_runs`` record."""

    def __init__(self, run_id: str):
        self.run_id = str(run_id)
        db.init_db()

    def initialize(self, agents: Iterable[str] = DEFAULT_AGENTS) -> dict[str, dict]:
        for agent in agents:
            self.set_state(str(agent), "pending")
        return self.states()

    def set_state(self, agent: str, status: str, *, input: Any = None,
                  output: Any = None, error: str | None = None,
                  increment_attempt: bool = False) -> dict:
        allowed = {"pending", "running", "done", "failed", "paused"}
        if status not in allowed:
            raise ValueError(f"未知 Agent 状态：{status}")
        agent = str(agent).strip()
        if not agent or len(agent) > 80:
            raise ValueError("Agent 名称不能为空且不能超过 80 个字符")
        with db.conn(immediate=True) as c:
            old = c.execute("SELECT attempt,input_json,output_json FROM writing_agent_states "
                            "WHERE run_id=? AND agent=?", (self.run_id, agent)).fetchone()
            attempt = int(old["attempt"] or 0) if old else 0
            if increment_attempt:
                attempt += 1
            in_json = _dump(input) if input is not None else (old["input_json"] if old else "{}")
            out_json = _dump(output) if output is not None else (old["output_json"] if old else "{}")
            c.execute("""INSERT INTO writing_agent_states
                         (run_id,agent,status,attempt,input_json,output_json,error)
                         VALUES(?,?,?,?,?,?,?)
                         ON CONFLICT(run_id,agent) DO UPDATE SET
                           status=excluded.status, attempt=excluded.attempt,
                           input_json=excluded.input_json, output_json=excluded.output_json,
                           error=excluded.error, updated_at=datetime('now','localtime')""",
                      (self.run_id, agent, status, attempt, in_json, out_json, error))
            row = c.execute("SELECT * FROM writing_agent_states WHERE run_id=? AND agent=?",
                            (self.run_id, agent)).fetchone()
        return self._state_row(row)

    @staticmethod
    def _state_row(row) -> dict:
        d = dict(row)
        for key in ("input_json", "output_json"):
            try:
                d[key[:-5]] = json.loads(d.pop(key) or "{}")
            except json.JSONDecodeError:
                d[key[:-5]] = {}
        return d

    def states(self) -> dict[str, dict]:
        with db.conn() as c:
            rows = c.execute("SELECT * FROM writing_agent_states WHERE run_id=? "
                             "ORDER BY agent", (self.run_id,)).fetchall()
        return {r["agent"]: self._state_row(r) for r in rows}

    def send(self, sender: str, recipient: str, message_type: str,
             payload: dict | None = None, *, confidence: float = 0.0,
             requires_review: bool = False, message_id: str | None = None) -> dict:
        msg = AgentMessage(message_id=message_id or uuid.uuid4().hex[:16],
                           sender=str(sender), recipient=str(recipient),
                           message_type=str(message_type), payload=payload or {},
                           confidence=confidence, requires_review=requires_review)
        data = msg.model_dump()
        with db.conn() as c:
            c.execute("""INSERT OR IGNORE INTO writing_messages
                         (run_id,message_id,sender,recipient,message_type,payload_json,
                          confidence,requires_review)
                         VALUES(?,?,?,?,?,?,?,?)""",
                      (self.run_id, msg.message_id, msg.sender, msg.recipient,
                       msg.message_type, _dump(msg.payload), msg.confidence,
                       int(msg.requires_review)))
        return data

    def messages(self, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        with db.conn() as c:
            rows = c.execute("""SELECT * FROM writing_messages WHERE run_id=?
                               ORDER BY id DESC LIMIT ?""", (self.run_id, limit)).fetchall()
        out = []
        for row in reversed(rows):
            d = dict(row)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            d["requires_review"] = bool(d["requires_review"])
            out.append(d)
        return out

    def plan(self) -> dict:
        """Return the deterministic next action and all persisted states."""
        states = self.states()
        # A running stage already owns the next action.  Do not invite a
        # concurrent rewrite while it is still reviewing or auditing a draft.
        running = next((agent for agent in _ORDER
                        if states.get(agent, {}).get("status") == "running"), None)
        if running:
            return {"next": running, "action": "wait", "states": states,
                    "resumable": False}
        for agent in _ORDER:
            state = states.get(agent, {})
            if state.get("status") in (None, "pending", "failed", "paused"):
                return {"next": agent, "states": states,
                        "resumable": state.get("status") in {"failed", "paused"}}
            if agent == "auditor" and state.get("status") == "done":
                output = state.get("output") or {}
                issues = output.get("issues") or []
                if any(isinstance(issue, dict) and issue.get("severity") == "major"
                       for issue in issues):
                    sections = sorted({int(no) for no in output.get("rewrite_sections") or []
                                       if str(no).isdigit() and int(no) > 0})
                    return {"next": "writer" if sections else "auditor",
                            "action": "rewrite_sections" if sections else "review_audit",
                            "sections": sections, "requires_review": True,
                            "states": states, "resumable": True}
        return {"next": None, "states": states, "resumable": False}

    def parallel_evidence(self, sections: Iterable[dict], builder: Callable[[dict], Any],
                          max_workers: int = 4) -> tuple[dict[int, Any], dict[int, str]]:
        """Build section evidence concurrently while preserving section order.

        Each worker has its own SQLite connection through the evidence agent.
        Exceptions are captured per section so one bad paper does not discard
        successful bundles; the caller can retry only the failed section.
        """
        ordered = sorted(list(sections), key=lambda s: int(s.get("no") or 0))
        if not ordered:
            return {}, {}
        workers = max(1, min(int(max_workers), len(ordered), 8))
        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                    thread_name_prefix="papernest-evidence") as pool:
            futures = {}
            for section in ordered:
                no = int(section.get("no") or 0)
                self.set_state(f"evidence:{no}", "running", input=section,
                               increment_attempt=True)
                futures[pool.submit(builder, section)] = no
            for future in concurrent.futures.as_completed(futures):
                no = futures[future]
                try:
                    result = future.result()
                    results[no] = result
                    self.set_state(f"evidence:{no}", "done", output=result.model_dump()
                                   if hasattr(result, "model_dump") else result)
                except Exception as exc:  # noqa: BLE001
                    errors[no] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    self.set_state(f"evidence:{no}", "failed", error=errors[no])
                    self.send("evidence_agent", "coordinator", "evidence_failed",
                              {"sec_no": no, "error": errors[no]}, requires_review=True)
        return ({no: results[no] for no in sorted(results)},
                {no: errors[no] for no in sorted(errors)})


def for_run(run_id: str) -> Coordinator:
    return Coordinator(run_id)


__all__ = ["Coordinator", "DEFAULT_AGENTS", "for_run"]
