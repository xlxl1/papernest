"""异步任务：SQLite 落库的任务表 + 后台执行 + SSE 可订阅的进度事件。

长任务（采集 / L2 精读 / 综述 / Agent run）不再阻塞 HTTP 请求：
POST /api/jobs 提交 → 后台跑 → GET /api/jobs/{id} 轮询 或
GET /api/jobs/{id}/events 以 SSE 收实时进度。进度持久化，
服务重启后也能看到任务死在哪一步。
"""
import concurrent.futures
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from . import db

PROGRESS = Callable[[float, str, str], None]  # (0~1, stage, message)

# 同时在跑的任务上限。任务多是 LLM 密集型，无上限地铺开只会一起超时；
# 也避免占满 Starlette 线程池导致 HTTP 请求饿死。
MAX_CONCURRENT = max(1, int(os.environ.get("PAPERNEST_MAX_JOBS", "3")))

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


class JobConflict(Exception):
    """同一资源已有任务在跑（dedupe_key 冲突）。"""


def _pool() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT, thread_name_prefix="papernest-job")
    return _executor


def create_job(kind: str, params: dict, dedupe_key: str | None = None) -> str:
    """建任务。dedupe_key 非空时，同键已有 queued/running 任务则抛 JobConflict。

    没有这道闸时，前端连点两次「开写」会让两个线程交错写同一批 sections 行。
    """
    job_id = uuid.uuid4().hex[:12]
    db.init_db()
    with db.conn() as c:
        if dedupe_key:
            busy = c.execute(
                """SELECT id FROM jobs WHERE dedupe_key=? AND status IN ('queued','running')
                   ORDER BY created_at DESC LIMIT 1""", (dedupe_key,)).fetchone()
            if busy:
                raise JobConflict(f"已有任务 {busy['id']} 在处理该对象，请等它结束或先取消")
        c.execute("INSERT INTO jobs(id,kind,params_json,dedupe_key) VALUES(?,?,?,?)",
                  (job_id, kind, json.dumps(params, ensure_ascii=False), dedupe_key))
    return job_id


def submit(job_id: str):
    """把任务丢进有界线程池（立即返回）。执行体自己做原子认领，重复提交无害。"""
    _pool().submit(execute_job, job_id)


def claim_job(job_id: str) -> bool:
    """原子认领：queued → running。返回 False 表示别人已经领走（防重复执行）。"""
    with db.conn() as c:
        cur = c.execute(
            """UPDATE jobs SET status='running', stage='start', message='任务开始',
               progress=0.01, updated_at=datetime('now','localtime')
               WHERE id=? AND status='queued'""", (job_id,))
        return cur.rowcount > 0


def recover_stale_jobs() -> int:
    """服务启动时调用：上次进程留下的 queued/running 任务永远不会再被推进，
    如实标 failed，免得 SSE 客户端对着一个死任务无限轮询。"""
    db.init_db()
    with db.conn() as c:
        cur = c.execute(
            """UPDATE jobs SET status='failed',
               error=COALESCE(NULLIF(error,''), '服务重启，任务被中断（未完成的步骤需重新提交）'),
               updated_at=datetime('now','localtime')
               WHERE status IN ('queued','running')""")
        return cur.rowcount


def get_job(job_id: str) -> dict | None:
    with db.conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    if d.get("result_json"):
        d["result"] = json.loads(d.pop("result_json"))
    else:
        d.pop("result_json")
    return d


def list_jobs(limit: int = 20) -> list[dict]:
    with db.conn() as c:
        rows = c.execute("""SELECT id, kind, status, stage, progress, message, error, created_at
                            FROM jobs ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def set_progress(job_id: str, frac: float, stage: str, message: str):
    with db.conn() as c:
        c.execute("""UPDATE jobs SET status='running', progress=?, stage=?, message=?,
                     updated_at=datetime('now','localtime') WHERE id=?""",
                  (round(min(max(frac, 0.0), 1.0), 3), stage[:60], message[:300], job_id))
        c.commit()


def finish_job(job_id: str, result: dict):
    with db.conn() as c:
        c.execute("""UPDATE jobs SET status='done', progress=1.0, stage='done',
                     result_json=?, updated_at=datetime('now','localtime') WHERE id=?""",
                  (json.dumps(result, ensure_ascii=False), job_id))
        c.commit()


def fail_job(job_id: str, error: str):
    with db.conn() as c:
        c.execute("""UPDATE jobs SET status='failed', error=?,
                     updated_at=datetime('now','localtime') WHERE id=?""",
                  (error[:1000], job_id))
        c.commit()


def execute_job(job_id: str):
    """后台任务入口（线程池里跑）：原子认领 → 按 kind 分发 → 推进度 → 落结果。"""
    job = get_job(job_id)
    if not job:
        return
    runner = RUNNERS.get(job["kind"])
    if not runner:
        fail_job(job_id, f"未知任务类型：{job['kind']}")
        return
    if not claim_job(job_id):
        return  # 已被认领或已结束：重复提交不会跑第二遍
    started = time.perf_counter()

    def progress(frac: float, stage: str = "", message: str = ""):
        set_progress(job_id, frac, stage, message)

    try:
        result = runner(job["params"], progress)
        result["_elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        finish_job(job_id, result)
    except Exception as exc:
        fail_job(job_id, f"{type(exc).__name__}: {exc}")


# ── 各任务的执行体：params dict + progress 回调 → 结果 dict ──

def _ingest_runner(params: dict, progress: PROGRESS) -> dict:
    from . import config, embeddings, ingest
    progress(0.05, "search", f"检索：{params.get('query', '')[:50]}")
    result = ingest.ingest(params.get("query", ""), params.get("limit", 20),
                           topic=params.get("topic"),
                           year_from=params.get("year_from"),
                           year_to=params.get("year_to"),
                           progress=lambda f, s, m: progress(0.05 + f * 0.85, s, m))
    # 新论文建向量（与同步端点同一逻辑）
    if embeddings.available() and result["new_papers"]:
        progress(0.92, "embed", "为无向量论文补建索引")
        from . import db as _db
        with _db.conn() as c:
            rows = c.execute("""SELECT id, title, abstract FROM papers
                                WHERE id NOT IN (SELECT DISTINCT paper_id FROM vectors
                                                 WHERE kind='paper' AND model=?)""",
                             (config.EMBED_MODEL,)).fetchall()
        n = 0
        for r in rows:
            try:
                n += embeddings.index_paper(r["id"], r["title"], r["abstract"] or "")
            except Exception:
                break
        result["vectors_indexed"] = n
    progress(1.0, "done", "完成")
    return result


def _read_runner(params: dict, progress: PROGRESS) -> dict:
    from . import fulltext
    return fulltext.read_paper(params["paper_id"], topic=params.get("topic"),
                               progress=lambda f, s, m: progress(f, s, m))


def _survey_runner(params: dict, progress: PROGRESS) -> dict:
    from . import survey
    return survey.generate(params["topic"], params.get("top_k", 10),
                           progress=lambda f, s, m: progress(f, s, m))


def _agent_runner(params: dict, progress: PROGRESS) -> dict:
    from . import agent
    ev_count = {"n": 0}

    def on_event(ev: dict):
        ev_count["n"] += 1
        progress(min(ev_count["n"] / max(params.get("max_steps", 5), 1), 0.95),
                 f"step{ev.get('step_id')}:{ev.get('tool')}",
                 f"{ev.get('tool')} → {ev.get('status')} {ev.get('summary', '')}")

    return agent.run_agent(params.get("goal", ""), params.get("top_k", 5),
                           params.get("max_steps", 5), params.get("topic"),
                           dry_run=params.get("dry_run", False),
                           timeout_s=params.get("timeout_s", 240),
                           on_event=on_event).model_dump()


def _polish_runner(params: dict, progress: PROGRESS) -> dict:
    from . import writing
    progress(0.2, "context", "收集关联文献作为术语/引用参照")
    result = writing.polish_text(params.get("text", ""), params.get("instruction", ""),
                                 params.get("paper_ids"))
    progress(0.95, "done", "润色完成")
    return result


def _review_runner(params: dict, progress: PROGRESS) -> dict:
    from . import writing
    progress(0.2, "context", "收集库内文献上下文（核对引用与相关工作表述）")
    result = writing.review_text(params.get("text", ""), params.get("paper_ids"))
    progress(0.95, "done", "评审完成")
    return result


def _bibimport_runner(params: dict, progress: PROGRESS) -> dict:
    """BibTeX/RIS/CSV 导入。开 enrich 时逐条查 Semantic Scholar 补摘要——
    受共享池限流，几百条能跑到分钟级，所以这条路必须走任务系统而不是阻塞 HTTP 线程。"""
    from . import bibimport
    progress(0.05, "parse", "解析条目")
    result = bibimport.import_text(params.get("text", ""), params.get("fmt", "auto"),
                                   enrich=bool(params.get("enrich")),
                                   source_name=params.get("source_name", ""))
    progress(1.0, "done",
             f"新增 {result['imported']} · 跳过 {result['skipped']} · 失败 {result['failed']}")
    return result


def _graph_runner(params: dict, progress: PROGRESS) -> dict:
    """批量拉引文边（联网、受限流，天然是长任务）。单篇失败不影响其余。"""
    from . import db as _db
    from . import graph
    ids = params.get("paper_ids")
    if not ids:
        with _db.conn() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM papers ORDER BY citation_count DESC NULLS LAST, id DESC "
                "LIMIT ?", (int(params.get("limit", 20)),)).fetchall()]
    added = ok = failed = 0
    errors: list[str] = []
    for i, pid in enumerate(ids):
        progress(0.02 + 0.95 * i / max(len(ids), 1), "fetch",
                 f"拉第 {i + 1}/{len(ids)} 篇的引文边（已新增 {added} 条）")
        try:
            r = graph.fetch_edges(pid, params.get("direction", "both"),
                                  int(params.get("per_paper", 100)))
            added += r.get("edges_added", 0)
            ok += 1
        except Exception as exc:
            failed += 1
            errors.append(f"#{pid}: {type(exc).__name__}: {str(exc)[:120]}")
    return {"papers": len(ids), "ok": ok, "failed": failed, "edges_added": added,
            "errors": errors[:20], "stats": graph.stats()}


def _deep_answer_runner(params: dict, progress: PROGRESS) -> dict:
    """迭代问答：作答 → 机械找无证据论断 → 补检索 → 重答，最多 max_rounds 轮。
    一轮 = 一次作答 + 一次自评，两轮就是四次模型调用起步——必须走任务系统。"""
    from . import deepsearch
    return deepsearch.deep_answer(params.get("question", ""),
                                  top_k=int(params.get("top_k", 5)),
                                  max_rounds=int(params.get("max_rounds", 2)),
                                  progress=progress)


def _rcs_answer_runner(params: dict, progress: PROGRESS) -> dict:
    """RCS 问答：1 次重排 + N 次逐篇摘要 + 1 次作答，篇数多了就是分钟级。"""
    from . import rcs
    return rcs.answer(params.get("question", ""),
                      top_k=int(params.get("top_k", 5)), progress=progress)


def _digest_runner(params: dict, progress: PROGRESS) -> dict:
    """新论文摘报：拉 arXiv 天然是长任务（受礼貌间隔约束），必须走任务系统。"""
    from . import subscribe
    return subscribe.digest(
        days=int(params.get("days", 1)), top_k=int(params.get("top_k", 15)),
        categories=params.get("categories"), use_llm=bool(params.get("use_llm")),
        repeat_after_days=int(params.get("repeat_after_days", 30)),
        topic=params.get("topic"), progress=progress)


def _matrix_runner(params: dict, progress: PROGRESS) -> dict:
    """对比矩阵：开 LLM 增强时每篇一次调用，篇数多了就是分钟级。"""
    from . import matrix
    return matrix.build(params.get("paper_ids") or [], params.get("columns"),
                        params.get("use_llm"), progress=progress,
                        topic=params.get("topic"))


def _write_runner(params: dict, progress: PROGRESS) -> dict:
    """二期写作流水线：stage = topic | outline | sections | section | polish。

    检查点即「任务结束、等用户指令」：选题后等定题、大纲后等确认；
    每次确认由 API/CLI 提交下一个 stage 任务，断点续跑天然成立。
    """
    from . import pipeline
    return pipeline.run_stage(params["run_id"], params, progress)


RUNNERS: dict[str, Callable[[dict, PROGRESS], dict]] = {
    "ingest": _ingest_runner,
    "read": _read_runner,
    "survey": _survey_runner,
    "agent": _agent_runner,
    "polish": _polish_runner,
    "review": _review_runner,
    "write": _write_runner,
    "bibimport": _bibimport_runner,
    "graph": _graph_runner,
    "digest": _digest_runner,
    "matrix": _matrix_runner,
    "deep_answer": _deep_answer_runner,
    "rcs_answer": _rcs_answer_runner,
}
