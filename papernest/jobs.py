"""异步任务：SQLite 落库的任务表 + 后台执行 + SSE 可订阅的进度事件。

长任务（采集 / L2 精读 / 综述 / Agent run）不再阻塞 HTTP 请求：
POST /api/jobs 提交 → 后台跑 → GET /api/jobs/{id} 轮询 或
GET /api/jobs/{id}/events 以 SSE 收实时进度。进度持久化，
服务重启后也能看到任务死在哪一步。
"""
import concurrent.futures
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from . import config, db

PROGRESS = Callable[[float, str, str], None]  # (0~1, stage, message)

# 同时在跑的任务上限。任务多是 LLM 密集型，无上限地铺开只会一起超时；
# 也避免占满 Starlette 线程池导致 HTTP 请求饿死。
# 非法值降级到默认，不在 import 期抛 ValueError 把整个包（连同 uvicorn / cli）拖死。
MAX_CONCURRENT = config._env_int(3, "PAPERNEST_MAX_JOBS")

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


class JobConflict(Exception):
    """同一资源已有任务在跑（dedupe_key 冲突）。"""


class JobCancelled(Exception):
    """任务被用户取消，执行器应停止继续推进并丢弃部分结果。"""


_recovered_ids: set[str] = set()
_recovered_lock = threading.Lock()


def _pool() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT, thread_name_prefix="papernest-job")
    return _executor


def create_job(kind: str, params: dict, dedupe_key: str | None = None,
               idempotency_key: str | None = None, max_attempts: int = 1) -> str:
    """建任务。支持资源去重、客户端幂等和持久化重试上限。

    没有这道闸时，前端连点两次「开写」会让两个线程交错写同一批 sections 行。
    """
    if not isinstance(params, dict):
        raise ValueError("任务参数必须是对象")
    try:
        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("任务参数必须可序列化为 JSON") from exc
    if len(params_json.encode("utf-8")) > 1_000_000:
        raise ValueError("任务参数不能超过 1MB")
    max_attempts = max(1, min(int(max_attempts), 5))
    job_id = uuid.uuid4().hex[:12]
    db.init_db()
    # **必须是写事务**：这是一次读-改-写（先查同键有没有在跑、再插）。
    # 默认连接是 autocommit，SELECT 在锁外做，两个线程会同时看到「没人在跑」。
    # 实测 16 线程并发建同一 dedupe_key 的任务：建成 2 个（应为 1）。
    # 真正的保证在 `_migrate_8` 的部分唯一索引上，这里的事务只是让错误变成
    # 干净的 JobConflict 而不是 IntegrityError。
    with db.conn(immediate=True) as c:
        if idempotency_key:
            existing = c.execute(
                "SELECT id, kind, params_json FROM jobs WHERE idempotency_key=?",
                (idempotency_key,)).fetchone()
            if existing:
                if existing["kind"] != kind:
                    raise JobConflict("幂等键已被另一种任务类型使用")
                if existing["params_json"] != params_json:
                    raise JobConflict("幂等键已存在，但任务参数不同")
                return existing["id"]
        if dedupe_key:
            busy = c.execute(
                """SELECT id FROM jobs WHERE dedupe_key=? AND status IN ('queued','running')
                   ORDER BY created_at DESC LIMIT 1""", (dedupe_key,)).fetchone()
            if busy:
                raise JobConflict(f"已有任务 {busy['id']} 在处理该对象，请等它结束或先取消")
        try:
            c.execute("""INSERT INTO jobs(
                       id,kind,params_json,dedupe_key,idempotency_key,max_attempts)
                       VALUES(?,?,?,?,?,?)""",
                      (job_id, kind, params_json, dedupe_key,
                       idempotency_key, max_attempts))
        except sqlite3.IntegrityError as exc:
            # 并发请求可能同时通过上面的 SELECT，唯一索引是最终闸门。
            if idempotency_key:
                existing = c.execute(
                    "SELECT id, kind, params_json FROM jobs WHERE idempotency_key=?",
                    (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing["kind"] != kind or existing["params_json"] != params_json:
                        raise JobConflict("幂等键已存在，但任务参数不同") from exc
                    return existing["id"]
            if dedupe_key:
                raise JobConflict("已有任务在处理该对象，请等它结束或先取消") from exc
            raise
    return job_id


def submit(job_id: str):
    """把任务丢进有界线程池（立即返回）。执行体自己做原子认领，重复提交无害。"""
    _pool().submit(execute_job, job_id)


def claim_job(job_id: str) -> bool:
    """原子认领：queued → running。

    并发上限在 SQLite 写事务中再次校验，因此多个 uvicorn worker / 多个容器
    共享同一 SQLite 库时也不会各自突破本进程的 ``ThreadPoolExecutor`` 上限。
    """
    with db.conn() as c:
        cur = c.execute(
            """UPDATE jobs SET status='running', stage='start', message='任务开始', error=NULL,
               progress=0.01, attempt=attempt+1, cancel_requested=0,
               updated_at=datetime('now','localtime')
               WHERE id=? AND status='queued' AND cancel_requested=0
                 AND (SELECT COUNT(*) FROM jobs WHERE status='running') < ?""",
            (job_id, MAX_CONCURRENT))
        return cur.rowcount > 0


def recover_stale_jobs() -> int:
    """服务启动时恢复可重试任务，达到上限的任务才标记 failed。

    running 任务无法知道进程中断前执行到了哪一步，因此恢复时从 runner
    起点重新执行；runner 自己负责幂等写入，且 attempt 持久化避免无限重试。
    """
    db.init_db()
    with db.conn(immediate=True) as c:
        rows = c.execute("""SELECT id, attempt, max_attempts FROM jobs
                           WHERE status IN ('queued','running')
                           ORDER BY created_at""").fetchall()
        recover, failed = [], []
        for row in rows:
            if int(row["attempt"] or 0) < int(row["max_attempts"] or 1):
                recover.append(row["id"])
            else:
                failed.append(row["id"])
        if recover:
            c.executemany("""UPDATE jobs SET status='queued', stage='recovered',
                           message='服务重启后自动恢复，任务将重新执行',
                           error=COALESCE(NULLIF(error,''), '服务重启后恢复'),
                           updated_at=datetime('now','localtime') WHERE id=?""",
                          [(job_id,) for job_id in recover])
        if failed:
            c.executemany("""UPDATE jobs SET status='failed',
                           error=COALESCE(NULLIF(error,''), '服务重启，重试次数已耗尽'),
                           updated_at=datetime('now','localtime') WHERE id=?""",
                          [(job_id,) for job_id in failed])
    with _recovered_lock:
        _recovered_ids.update(recover)
    return len(rows)


def drain_recovered() -> list[str]:
    with _recovered_lock:
        ids = list(_recovered_ids)
        _recovered_ids.clear()
    return ids


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
        rows = c.execute("""SELECT id, kind, status, stage, progress, message, error,
                                   attempt, max_attempts, cancel_requested, created_at
                            FROM jobs ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def set_progress(job_id: str, frac: float, stage: str, message: str):
    if is_cancel_requested(job_id):
        raise JobCancelled("任务已请求取消")
    with db.conn() as c:
        c.execute("""UPDATE jobs SET status='running', progress=?, stage=?, message=?,
                     updated_at=datetime('now','localtime') WHERE id=?""",
                  (round(min(max(frac, 0.0), 1.0), 3), stage[:60], message[:300], job_id))
        c.commit()


def finish_job(job_id: str, result: dict):
    with db.conn(immediate=True) as c:
        cur = c.execute("""UPDATE jobs SET status='done', progress=1.0, stage='done', message='完成',
                     error=NULL,
                     result_json=?, updated_at=datetime('now','localtime')
                     WHERE id=? AND cancel_requested=0""",
                  (json.dumps(result, ensure_ascii=False), job_id))
        if cur.rowcount == 0:
            c.execute("""UPDATE jobs SET status='cancelled', stage='cancelled',
                       message='用户取消', updated_at=datetime('now','localtime') WHERE id=?""",
                      (job_id,))
        c.commit()


def fail_job(job_id: str, error: str):
    with db.conn(immediate=True) as c:
        cur = c.execute("""UPDATE jobs SET status='failed', error=?,
                     updated_at=datetime('now','localtime')
                     WHERE id=? AND cancel_requested=0""",
                        (error[:1000], job_id))
        if cur.rowcount == 0:
            c.execute("""UPDATE jobs SET status='cancelled', stage='cancelled',
                       message='用户取消', updated_at=datetime('now','localtime') WHERE id=?""",
                      (job_id,))
        c.commit()


def is_cancel_requested(job_id: str) -> bool:
    with db.conn() as c:
        row = c.execute("SELECT status, cancel_requested FROM jobs WHERE id=?",
                        (job_id,)).fetchone()
    return not row or row["status"] == "cancelled" or bool(row["cancel_requested"])


def cancel_job(job_id: str) -> dict | None:
    """取消 queued 任务，或给 running 任务设置取消标记。"""
    with db.conn(immediate=True) as c:
        row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] in ("done", "failed", "cancelled"):
            return None
        if row["status"] == "queued":
            c.execute("""UPDATE jobs SET status='cancelled', stage='cancelled',
                       message='用户取消', cancel_requested=1,
                       updated_at=datetime('now','localtime') WHERE id=?""", (job_id,))
        else:
            c.execute("""UPDATE jobs SET cancel_requested=1, message='等待当前步骤结束后取消',
                       updated_at=datetime('now','localtime') WHERE id=?""", (job_id,))
    return get_job(job_id)


def _retry_job(job_id: str, error: str) -> bool:
    """异常后把任务重新排队；返回 True 表示还会继续执行。"""
    with db.conn(immediate=True) as c:
        row = c.execute("SELECT status, attempt, max_attempts, cancel_requested FROM jobs WHERE id=?",
                        (job_id,)).fetchone()
        if not row or row["status"] != "running" or row["cancel_requested"]:
            return False
        if int(row["attempt"] or 0) >= int(row["max_attempts"] or 1):
            return False
        c.execute("""UPDATE jobs SET status='queued', stage='retrying',
                   message='本次执行失败，准备重试', error=?, updated_at=datetime('now','localtime')
                   WHERE id=?""", (error[:1000], job_id))
    submit(job_id)
    return True


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
        # 已被认领/结束时无需再排队；只是跨进程达到并发上限时，延迟重试，
        # 避免把线程池 worker 阻塞在 sleep 上导致所有任务互相饿死。
        state = get_job(job_id)
        if state and state["status"] == "queued" and not state.get("cancel_requested"):
            timer = threading.Timer(0.5, submit, args=(job_id,))
            timer.daemon = True
            timer.start()
        return
    started = time.perf_counter()

    def progress(frac: float, stage: str = "", message: str = ""):
        set_progress(job_id, frac, stage, message)

    try:
        result = runner(job["params"], progress)
        result["_elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        if is_cancel_requested(job_id):
            with db.conn() as c:
                c.execute("""UPDATE jobs SET status='cancelled', stage='cancelled',
                           message='用户取消', updated_at=datetime('now','localtime') WHERE id=?""",
                          (job_id,))
                c.commit()
        else:
            finish_job(job_id, result)
    except JobCancelled as exc:
        with db.conn() as c:
            c.execute("""UPDATE jobs SET status='cancelled', stage='cancelled',
                       message=?, error=NULL, updated_at=datetime('now','localtime') WHERE id=?""",
                      (str(exc)[:300] or "用户取消", job_id))
            c.commit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not _retry_job(job_id, error):
            fail_job(job_id, error)


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
    _note_missing_vectors(result)
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


def _note_missing_vectors(result: dict) -> None:
    """把「还有多少篇没有向量」写进任务结果。

    只有 `ingest` 那条路会补建向量，其余入库路径（上传 PDF / 导入 .bib /
    多格式入库）都不建。**不建是对的**——建向量要花钱，不该在用户没同意时偷偷花；
    但**不说**就不对了：真库 443 篇无向量正是这么攒出来的，而向量路权重 1.0，
    这些论文等于被系统性地排在后面。
    """
    try:
        with db.conn() as c:
            n = db.papers_without_vectors(c, config.EMBED_MODEL)
    except Exception:                                   # noqa: BLE001
        return
    if n:
        result["papers_without_vectors"] = n
        result["hint"] = (f"库里还有 {n} 篇没有向量索引（只有「检索入库」那条路会自动补建）。"
                          f"跑 `python cli.py embed` 补上——**会产生嵌入调用费用**。"
                          f"在那之前这些论文只能靠关键词被检索到。")


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
