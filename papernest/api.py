"""FastAPI 服务：库管理 + Agent + 异步任务/SSE + RAG 问答 + 引用推荐 + L2 精读。"""
import asyncio
import hmac
import json
import math
import os
import re
import uuid
from pathlib import Path

from fastapi import (BackgroundTasks, File, FastAPI, HTTPException, Query, Request,
                     Response, UploadFile)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from . import (agent, bibimport, budget, cards, chat, cite, config, coordinator, db, deepsearch, degrade,
               embeddings, fulltext, graph, jobs, llm, log, matrix, pdfimport,
               pipeline, pptgen, rag, structure, subscribe, survey, symbols,
               tablegen, workspace, writing)
from .schemas import (
    AgentRequest,
    DocumentCreate,
    DocumentUpdate,
    NoteCreate,
    NoteUpdate,
    ProjectCreate,
    ProjectUpdate,
    SourceAttach,
    WriteOutlineUpdate,
    WriteRunCreate,
    WriteTopicSelect,
    WritingRequest,
)

_log = log.get("api")

# 文档端点默认关闭。生产环境如果需要 OpenAPI，应通过受保护的独立网关暴露，
# 避免把完整接口契约和模型字段泄露给未认证请求。
app = FastAPI(title="PaperNest", docs_url=None, redoc_url=None, openapi_url=None)
WEB = Path(__file__).resolve().parent.parent / "web"

# 可选访问口令：设了 PAPERNEST_API_KEY 才启用（本机自用默认不设，行为不变）。
# 部署到公网/内网可达的机器时务必设——否则删项目、跑花钱的 LLM 任务都是裸奔。
API_KEY = os.environ.get("PAPERNEST_API_KEY", "")

#: 服务实际绑定的地址。`cli.py serve` 与 Dockerfile 各自如实写进来——
#: 应用自己问不到 uvicorn 绑在哪，而「有没有对外暴露」恰恰是要不要强制口令的唯一依据。
BIND_HOST = os.environ.get("PAPERNEST_BIND_HOST", "127.0.0.1")
#: 明确承担裸奔风险的逃生口（本机自用、CI、内网隔离环境）
ALLOW_NO_AUTH = os.environ.get("PAPERNEST_ALLOW_NO_AUTH", "") in ("1", "true", "True")

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _exposed_without_auth() -> bool:
    """绑了非回环地址却没设口令 —— 这是「开箱即用即裸奔」的那个状态。"""
    return bool(BIND_HOST) and BIND_HOST not in _LOOPBACK and not API_KEY


def _check_key_transportable():
    """口令必须是 HTTP 头能装下的字符（ASCII）。

    两层原因，缺一不可：
    1. `hmac.compare_digest` 对两个 str 只支持 ASCII，非 ASCII 直接 TypeError——
       中间件里没有兜底，结果是每个 /api/* 恒定 500 而不是 401；
    2. 就算修掉第 1 点，**HTTP 头本身也装不下非 ASCII**（latin-1 编码），
       客户端在发出请求前就会 UnicodeEncodeError。只有 `?api_key=` 那条路能走，
       于是变成「SSE 能过、普通请求全废」的半坏状态。
    与其留一个半坏的配置，不如启动时说清楚。docker-compose 的提示语也已改成
    「ASCII 口令」——中文提示很容易诱导人填中文口令。
    """
    if API_KEY and not API_KEY.isascii():
        raise RuntimeError(
            "拒绝启动：PAPERNEST_API_KEY 含非 ASCII 字符。\n"
            "  HTTP 请求头装不下非 ASCII，客户端发不出去；口令请用 ASCII"
            "（字母数字与常见符号）。")


def _check_exposure():
    """启动自检：对外暴露 + 无口令 = 拒绝启动。

    原来这是一个「建议」（README 写着「部署时务必设」），而唯一的部署方式
    docker-compose 既没设它、也没提示。默认形态就是同网段任何人都能删数据、
    烧 LLM 额度。安全默认值不能靠文档提醒来兜。
    """
    if _exposed_without_auth() and not ALLOW_NO_AUTH:
        raise RuntimeError(
            f"拒绝启动：服务绑在 {BIND_HOST}（非回环）却没有设 PAPERNEST_API_KEY。\n"
            f"  这会让同网段任意主机无凭据调用全部 /api/*（删项目/笔记、触发烧钱的 LLM 任务）。\n"
            f"  设置访问口令：PAPERNEST_API_KEY=<你的口令>\n"
            f"  确实要裸奔（本机自用/内网隔离）：PAPERNEST_ALLOW_NO_AUTH=1")


@app.middleware("http")
async def _auth(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        # 浏览器原生 EventSource / img / iframe 无法自定义请求头，SSE 才允许
        # query 传 key；普通 API 即使带 api_key 也必须走请求头，避免密钥进入
        # 代理访问日志、浏览器历史和 Referer。
        is_sse = (request.url.path.endswith("/events")
                  or request.url.path.endswith("/stream")
                  or "text/event-stream" in request.headers.get("accept", ""))
        got = request.headers.get("x-api-key", "")
        authenticated_by_header = bool(got)
        if not got:
            got = request.cookies.get("papernest_auth", "")
        if is_sse and not got:
            got = request.query_params.get("api_key", "")
        # **必须先 encode**：`hmac.compare_digest` 对两个 str 只支持 ASCII，
        # 含中文的口令直接 TypeError —— 中间件里没有兜底、也没有全局异常处理器，
        # 结果是每个 /api/* 恒定 500 而不是 401，前端只在 401 时才弹口令输入框，
        # 用户连「口令错了」都看不到。而 docker-compose 现在用中文提示强制要求口令，
        # 填中文口令是完全正常的选择。encode 之后仍是常数时间比较。
        if not hmac.compare_digest(got.encode("utf-8"), API_KEY.encode("utf-8")):
            return JSONResponse({"detail": "缺少或错误的 API key"}, status_code=401)
        response = await call_next(request)
        # 浏览器的 img/iframe 无法设置请求头。首个 fetch 用 header 鉴权成功后，
        # 写入 HttpOnly cookie，后续 PDF/PNG 原生请求无需把密钥放进 URL。
        if authenticated_by_header:
            response.set_cookie("papernest_auth", API_KEY, httponly=True,
                                samesite="strict", secure=request.url.scheme == "https",
                                max_age=86_400, path="/")
        return response
    return await call_next(request)


@app.middleware("http")
async def _request_context(request: Request, call_next):
    """为每个请求返回可关联的 request id，并补齐低风险安全响应头。"""
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied[:128] if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.exception_handler(Exception)
async def _log_unhandled(request: Request, exc: Exception):
    """未捕获的异常：**先记下来**，再返回 500。

    没有这个处理器时，traceback 只会被 uvicorn 打到控制台——后台起服务
    （`serve &`、docker、开机自启）时那份输出根本没人接，出了事只剩一句
    「500 Internal Server Error」。本机单用户不需要告警，但需要**事后查得到**。
    响应体保持 FastAPI 默认的形状，不把内部错误文本透给客户端。
    """
    _log.exception("未处理的异常 request_id=%s %s %s",
                   getattr(request.state, "request_id", "-"),
                   request.method, request.url.path)
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse({"detail": "服务器内部错误，详见 data/papernest.log"},
                        status_code=500,
                        headers={"X-Request-ID": rid,
                                 "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "strict-origin-when-cross-origin"})


@app.exception_handler(budget.BudgetExceeded)
async def _budget_exceeded(request: Request, exc: budget.BudgetExceeded):
    _log.warning("预算闸门拒绝请求 request_id=%s path=%s",
                 getattr(request.state, "request_id", "-"), request.url.path)
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse({"detail": str(exc), "code": "daily_token_budget_exceeded"},
                        status_code=429, headers={"Retry-After": "86400",
                                                   "X-Request-ID": rid,
                                                   "X-Content-Type-Options": "nosniff"})


@app.exception_handler(budget.BudgetUnavailable)
async def _budget_unavailable(request: Request, exc: budget.BudgetUnavailable):
    _log.error("预算数据不可用 request_id=%s path=%s",
               getattr(request.state, "request_id", "-"), request.url.path,
               exc_info=True)
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse({"detail": "预算服务暂时不可用，请稍后重试",
                         "code": "budget_unavailable"},
                        status_code=503,
                        headers={"X-Request-ID": rid,
                                 "X-Content-Type-Options": "nosniff"})


@app.on_event("startup")
def _on_startup():
    """上个进程留下的 running 任务不会再有人推进——如实标失败，别让 SSE 空转。"""
    _check_key_transportable()
    _check_exposure()
    if not API_KEY:
        print(f"[papernest] 未设 PAPERNEST_API_KEY：/api/* 无鉴权"
              f"（绑定 {BIND_HOST}）。对外提供服务前请设置口令。")
    n = jobs.recover_stale_jobs()
    if n:
        recovered = jobs.drain_recovered()
        for job_id in recovered:
            jobs.submit(job_id)
        print(f"[papernest] 已检查 {n} 个中断任务，自动恢复 {len(recovered)} 个")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


# ── 库 ──

@app.get("/api/papers")
def papers(q: str = Query("", max_length=500),
           limit: int = Query(50, ge=1, le=200),
           year_from: int | None = Query(None, ge=1800, le=2200),
           year_to: int | None = Query(None, ge=1800, le=2200)):
    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(status_code=422, detail="year_from 不能晚于 year_to")
    db.init_db()
    with db.conn() as c:
        rows = (db.search_fts(c, q, limit * 2) if q else
                c.execute("SELECT * FROM papers ORDER BY id DESC LIMIT ?",
                          (limit * 2,)).fetchall())
        out = []
        for r in rows:
            if year_from and (r["year"] or 0) < year_from:
                continue
            if year_to and (r["year"] or 9999) > year_to:
                continue
            d = dict(r)
            d["card"] = cards.card_json(d.pop("card_json") or "{}")
            d["authors"] = json.loads(d.get("authors") or "[]")
            d["has_pages"] = bool(c.execute(
                "SELECT 1 FROM pages WHERE paper_id=? LIMIT 1", (r["id"],)).fetchone())
            d.pop("pdf_path", None)
            out.append(d)
            if len(out) >= limit:
                break
    return {"count": len(out), "papers": out, "embed_ready": embeddings.available()}


class IngestBody(BaseModel):
    query: str = Field(default="", max_length=1000)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=50, ge=1, le=100)
    year_from: int | None = Field(default=None, ge=1800, le=2200)
    year_to: int | None = Field(default=None, ge=1800, le=2200)

    @field_validator("keywords")
    @classmethod
    def _keyword_lengths(cls, values: list[str]) -> list[str]:
        if any(len(v) > 200 for v in values):
            raise ValueError("单个关键词不能超过 200 个字符")
        return values

    @model_validator(mode="after")
    def _year_range(self):
        if (self.year_from is not None and self.year_to is not None
                and self.year_from > self.year_to):
            raise ValueError("year_from 不能晚于 year_to")
        return self


# ── Research Agent ──

@app.post("/api/agent/plan")
def agent_plan(body: AgentRequest):
    """Return the selected tool plan without calling external services."""
    return agent.run_agent(body.goal, body.top_k, body.max_steps,
                           body.topic, dry_run=True,
                           intent_hint=body.intent_hint).model_dump()


@app.post("/api/agent/run")
def agent_run(body: AgentRequest):
    """Run a bounded tool workflow and return its execution timeline."""
    return agent.run_agent(body.goal, body.top_k, body.max_steps,
                           body.topic, dry_run=body.dry_run,
                           timeout_s=body.timeout_s,
                           intent_hint=body.intent_hint).model_dump()


@app.get("/api/agent/runs")
def agent_runs(limit: int = Query(20, ge=1, le=100)):
    """历史执行轨迹（时间线 UI 的回放数据）。"""
    db.init_db()
    return {"runs": db.list_agent_runs(limit)}


@app.get("/api/agent/runs/{run_id}")
def agent_run_detail(run_id: str):
    run = db.get_agent_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="执行轨迹不存在")
    return run


# ── 异步任务：提交 / 轮询 / SSE 进度 ──

class JobCreate(BaseModel):
    kind: str = Field(min_length=1, max_length=40)  # ingest | read | survey | agent | polish | review | write
    params: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=120)
    max_attempts: int = Field(default=1, ge=1, le=5)


@app.post("/api/jobs")
def create_job(body: JobCreate, background: BackgroundTasks):
    if body.kind not in jobs.RUNNERS:
        raise HTTPException(status_code=400,
                            detail=f"未知任务类型 {body.kind}（可选 {sorted(jobs.RUNNERS)}）")
    # 写作流水线的阶段任务按 run 去重，避免两个阶段并发改同一批 sections
    dedupe = (f"write:{body.params.get('run_id')}"
              if body.kind == "write" and body.params.get("run_id") else None)
    try:
        job_id = jobs.create_job(body.kind, body.params, dedupe_key=dedupe,
                                 idempotency_key=body.idempotency_key,
                                 max_attempts=body.max_attempts)
    except jobs.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "kind": body.kind, "status": "queued"}


@app.get("/api/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=100)):
    db.init_db()
    return {"jobs": jobs.list_jobs(limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    """SSE：实时推送任务进度，直到 done/failed 关流。"""
    if not jobs.get_job(job_id):
        raise HTTPException(status_code=404, detail="任务不存在")

    async def stream():
        last = None
        idle = 0.0
        while True:
            job = jobs.get_job(job_id)
            if not job:
                yield f"event: job\ndata: {json.dumps({'status': 'failed', 'error': '任务丢失'})}\n\n"
                return
            snap = json.dumps(job, ensure_ascii=False, default=str)
            if snap != last:  # 只在有变化时推，省带宽
                last = snap
                idle = 0.0
                yield f"event: job\ndata: {snap}\n\n"
            else:
                # LLM 长步骤期间状态不变：SSE 注释行心跳，防止客户端/代理闲置断流
                idle += 0.5
                if idle >= 15:
                    idle = 0.0
                    yield ": keepalive\n\n"
            if job["status"] in ("done", "failed", "cancelled"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                     "X-Accel-Buffering": "no"})


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not jobs.get_job(job_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    job = jobs.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=409, detail="任务已完成，无法取消")
    return job


# ── Research workspace: projects, notes and writing documents ──

@app.get("/api/projects")
def list_projects():
    return {"projects": workspace.list_projects()}


@app.post("/api/projects")
def create_project(body: ProjectCreate):
    try:
        return workspace.create_project(body.name, body.description, body.topic)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    project = workspace.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@app.patch("/api/projects/{project_id}")
def update_project(project_id: int, body: ProjectUpdate):
    try:
        project = workspace.update_project(project_id, **body.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int):
    if not workspace.delete_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"deleted": True, "project_id": project_id}


@app.get("/api/projects/{project_id}/notes")
def list_notes(project_id: int, paper_id: int | None = None):
    if not workspace.get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"notes": workspace.list_notes(project_id, paper_id)}


@app.post("/api/projects/{project_id}/notes")
def create_note(project_id: int, body: NoteCreate):
    try:
        return workspace.create_note(project_id, body.title, body.content,
                                     body.note_type, body.tags, body.paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/notes/{note_id}")
def get_note(note_id: int):
    note = workspace.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return note


@app.patch("/api/notes/{note_id}")
def update_note(note_id: int, body: NoteUpdate):
    try:
        note = workspace.update_note(note_id, **body.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not note:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return note


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    if not workspace.delete_note(note_id):
        raise HTTPException(status_code=404, detail="笔记不存在")
    return {"deleted": True, "note_id": note_id}


@app.get("/api/projects/{project_id}/documents")
def list_documents(project_id: int):
    if not workspace.get_project(project_id):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"documents": workspace.list_documents(project_id)}


@app.post("/api/projects/{project_id}/documents")
def create_document(project_id: int, body: DocumentCreate):
    try:
        return workspace.create_document(project_id, body.title, body.content, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/documents/{document_id}")
def get_document(document_id: int):
    document = workspace.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@app.patch("/api/documents/{document_id}")
def update_document(document_id: int, body: DocumentUpdate):
    update = body.model_dump(exclude_unset=True)
    version = update.pop("version", None)
    if update.get("expected_version") is None and version is not None:
        update["expected_version"] = version
    try:
        document = workspace.update_document(document_id,
                                             **update)
    except workspace.VersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int):
    if not workspace.delete_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": True, "document_id": document_id}


@app.post("/api/documents/{document_id}/sources")
def attach_document_source(document_id: int, body: SourceAttach):
    try:
        return workspace.attach_source(document_id, body.paper_id, body.page_no,
                                       body.quote, body.relation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/documents/{document_id}/write")
def write_document(document_id: int, body: WritingRequest):
    try:
        return writing.run(document_id, body.instruction, body.mode,
                           body.paper_ids, body.save)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ingest")
def do_ingest(body: IngestBody):
    """同步采集（老接口，长任务建议改用 POST /api/jobs kind=ingest）。

    逻辑与异步 job 版共用同一个 runner——原来这里是复制的第二份实现，
    两边已经开始各走各的了（补向量的分支只在其中一边改过）。
    """
    kws = [k.strip() for k in body.keywords if k.strip()]
    query = " ".join(kws) if kws else body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="关键词为空")
    return jobs.RUNNERS["ingest"](
        {"query": query, "limit": body.limit,
         "year_from": body.year_from, "year_to": body.year_to},
        lambda *_a, **_kw: None)


# ── 本地入库：上传 PDF / 导入 BibTeX·RIS·Zotero CSV ──

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_FILES = 20
MAX_UPLOAD_TOTAL_BYTES = 200 * 1024 * 1024


def _read_capped(fp, limit: int, chunk: int = 1 << 20) -> bytes | None:
    """分块读，超过 limit 立刻放弃并返回 None（不把超限文件整个读进内存）。

    多读一个字节用于判「是否恰好超限」，因此峰值内存是 limit + chunk 而不是文件大小。
    """
    buf = bytearray()
    while True:
        b = fp.read(chunk)
        if not b:
            return bytes(buf)
        buf.extend(b)
        if len(buf) > limit:
            return None


@app.post("/api/papers/upload")
def upload_pdfs(files: list[UploadFile] = File(...), topic: str = "",
                make_card: bool = True):
    """本地 PDF 入库：一次可传多份，逐份独立成败（一份坏的不该拖垮整批）。

    **同步 `def`，不是 `async def`**——函数体里是 PyMuPDF 解析、SQLite 写，
    `make_card=True` 时还会走到 `llm.chat`（含 8/20/45/80s 的同步退避）。
    写成 `async def` 会把这些全部跑在 uvicorn 的事件循环线程上，一次多份上传
    就能占死整个服务几分钟：期间 SSE 不出 token、任务进度卡死、所有请求排队。
    交给 Starlette 丢线程池才是对的（本项目约 60 个端点里唯二写错的就是这个和
    `import_bibliography_file`）。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有收到文件")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413,
                            detail=f"单批最多上传 {MAX_UPLOAD_FILES} 个文件")
    if len(topic) > 500:
        raise HTTPException(status_code=422, detail="topic 不能超过 500 个字符")
    declared_sizes = [getattr(f, "size", None) for f in files]
    if all(size is not None for size in declared_sizes):
        declared_total = sum(int(size) for size in declared_sizes)
        if declared_total > MAX_UPLOAD_TOTAL_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"单批文件总大小不能超过 {MAX_UPLOAD_TOTAL_BYTES // 1024 // 1024}MB")
    results, failed = [], []
    total_read = 0
    for f in files:
        if getattr(f, "size", None) is not None and int(f.size) > MAX_UPLOAD_BYTES:
            failed.append({"filename": f.filename,
                           "error": f"超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 单文件上限"})
            continue
        # 分块读到上限就停：原来是先 `await f.read()` 整个读进内存、**再**判大小，
        # 防线设在内存已经被占满之后，等于没有。
        remaining = MAX_UPLOAD_TOTAL_BYTES - total_read
        if remaining <= 0:
            failed.append({"filename": f.filename, "error": "超过单批总大小上限"})
            continue
        data = _read_capped(f.file, min(MAX_UPLOAD_BYTES, remaining))
        if data is None:
            reason = (f"超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 单文件上限"
                      if remaining >= MAX_UPLOAD_BYTES else "超过单批总大小上限")
            failed.append({"filename": f.filename, "error": reason})
            continue
        total_read += len(data)
        try:
            results.append(pdfimport.ingest_pdf(data, f.filename or "upload.pdf",
                                                topic=topic or None,
                                                make_card=make_card))
        except pdfimport.PdfImportError as exc:
            failed.append({"filename": f.filename, "error": str(exc)})
        except Exception as exc:                      # 解析层的意外不该 500 掉整批
            failed.append({"filename": f.filename,
                           "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    out = {"imported": len(results), "failed": len(failed),
           "papers": results, "errors": failed}
    # 上传这条路**不建向量**（建向量要花钱，不该在用户没同意时偷偷花），
    # 但必须说出来——真库 443 篇无向量正是这么攒出来的，而向量路权重 1.0。
    jobs._note_missing_vectors(out)
    return out


@app.get("/api/papers/uploads")
def list_uploads(limit: int = Query(50, ge=1, le=200)):
    return {"uploads": pdfimport.list_uploads(limit)}


class MetadataFix(BaseModel):
    title: str | None = Field(default=None, max_length=1000)
    authors: list[str] | None = Field(default=None, max_length=100)
    year: int | None = Field(default=None, ge=1800, le=2200)
    venue: str | None = Field(default=None, max_length=500)
    doi: str | None = Field(default=None, max_length=300)
    arxiv_id: str | None = Field(default=None, max_length=200)
    abstract: str | None = Field(default=None, max_length=100_000)

    @field_validator("authors")
    @classmethod
    def _author_lengths(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(len(author) > 300 for author in values):
            raise ValueError("单个作者名不能超过 300 个字符")
        return values


@app.patch("/api/paper/{paper_id}/metadata")
def fix_metadata(paper_id: int, body: MetadataFix):
    """人工修正抽错的元数据（改标题/DOI 会重算 norm_key，撞车则报错不覆盖）。"""
    with db.conn() as c:
        if not c.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise HTTPException(status_code=404, detail="论文不存在")
    try:
        return pdfimport.update_metadata(
            paper_id, **body.model_dump(exclude_unset=True, exclude_none=True))
    except pdfimport.PdfImportError as exc:
        # norm_key 撞车是「冲突」，字段名写错是「请求错」——分开报，前端才好提示
        code = 409 if "norm_key 冲突" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


class BibImportBody(BaseModel):
    text: str = Field(min_length=1, max_length=5_000_000)
    fmt: str = Field(default="auto", max_length=20)  # auto | bibtex | ris | csv
    enrich: bool = False       # 用 Semantic Scholar 补缺失摘要（会联网）
    source_name: str = Field(default="", max_length=500)


@app.post("/api/import/bibliography")
def import_bibliography(body: BibImportBody, background: BackgroundTasks):
    """导入 BibTeX / RIS / Zotero CSV。已存在的按 norm_key 跳过，只补空不覆盖。

    enrich=True 要逐条查 Semantic Scholar 补摘要，几百条能跑到分钟级——
    那条路改走任务系统返回 job_id，不阻塞 HTTP 线程（用 /api/jobs/{id}/events 看进度）。
    """
    if body.enrich:
        job_id = jobs.create_job("bibimport", {
            "text": body.text, "fmt": body.fmt, "enrich": True,
            "source_name": body.source_name})
        background.add_task(jobs.submit, job_id)
        return {"async": True, "job_id": job_id,
                "note": "开启了联网补全，已转为后台任务；用 /api/jobs/{job_id} 查进度"}
    try:
        return bibimport.import_text(body.text, body.fmt, False, body.source_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/import/bibliography/file")
def import_bibliography_file(background: BackgroundTasks,
                             file: UploadFile = File(...),
                             fmt: str = Query("auto", max_length=20),
                             enrich: bool = False):
    """同步 `def`：函数体里是解析与 SQLite 写，写成 async 会占死事件循环（同 upload_pdfs）。"""
    data = _read_capped(file.file, 20 * 1024 * 1024)
    if data is None:
        raise HTTPException(status_code=400, detail="文件超过 20MB 上限")
    text = data.decode("utf-8-sig", "replace")      # Zotero 导出常带 BOM
    if enrich:
        job_id = jobs.create_job("bibimport", {"text": text, "fmt": fmt, "enrich": True,
                                               "source_name": file.filename or ""})
        background.add_task(jobs.submit, job_id)
        return {"async": True, "job_id": job_id,
                "note": "开启了联网补全，已转为后台任务；用 /api/jobs/{job_id} 查进度"}
    try:
        return bibimport.import_text(text, fmt, False, file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/import/runs")
def import_runs(limit: int = Query(20, ge=1, le=100)):
    return {"runs": bibimport.list_imports(limit)}


# ── 引文网络：邻居图 / 相关推荐 / 阅读缺口 ──

@app.get("/api/graph/stats")
def graph_stats():
    return graph.stats()


@app.post("/api/paper/{paper_id}/edges")
def graph_fetch_edges(paper_id: int, direction: str = Query("both", max_length=20),
                      limit: int = Query(100, ge=1, le=500),
                      max_age_days: int = Query(30, ge=0, le=3650)):
    """拉该论文的 references / citations 入边表（会联网，已拉过的按 max_age_days 跳过）。"""
    try:
        return graph.fetch_edges(paper_id, direction, limit, max_age_days=max_age_days)
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/paper/{paper_id}/edges/local")
def graph_local_references(paper_id: int):
    """从本地 PDF 的参考文献段建出边（离线、0 token）。

    手动上传的 PDF 常常没有 DOI/s2_id，联网那条路对它无从下手；
    但它自己的参考文献段就写着「它引了谁」。
    """
    try:
        return graph.ingest_local_references(paper_id)
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/paper/{paper_id}/neighbors")
def graph_neighbors(paper_id: int, depth: int = Query(1, ge=1, le=5),
                    limit: int = Query(50, ge=1, le=500)):
    # graph.neighbors 对不在库的 paper_id 抛 GraphError。不接就是 500 + 裸 traceback，
    # 而同一模块紧邻的另外两个路由都规规矩矩映射成 400——同一类错误两种口径。
    try:
        return graph.neighbors(paper_id, depth, limit)
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class _PaperIdsBody(BaseModel):
    paper_ids: list[int] = Field(min_length=1, max_length=100)

    @field_validator("paper_ids")
    @classmethod
    def _positive_ids(cls, values: list[int]) -> list[int]:
        if any(pid < 1 for pid in values):
            raise ValueError("paper_ids 必须是正整数")
        return values


class RelatedBody(_PaperIdsBody):
    top_k: int = Field(default=20, ge=1, le=100)


@app.post("/api/graph/related")
def graph_related(body: RelatedBody):
    """共被引 + 文献耦合推荐（纯读库，不联网、不调 LLM）。"""
    try:
        return {"related": graph.related(body.paper_ids, body.top_k)}
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/graph/gaps")
def graph_gaps(top_k: int = Query(20, ge=1, le=100)):
    """被库内论文频繁引用、但自己不在库里的论文——最明确的阅读缺口。"""
    return {"gaps": graph.gap_papers(top_k)}


@app.post("/api/graph/adopt")
def graph_adopt(norm_key: str = Query(..., min_length=1, max_length=300)):
    """把图里的库外论文一键入库。"""
    try:
        return graph.adopt(norm_key)
    except graph.GraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── 新论文订阅推送 ──

class DigestBody(BaseModel):
    days: int = Field(default=3, ge=1, le=30)
    top_k: int = Field(default=15, ge=1, le=50)
    categories: list[str] | None = Field(default=None, max_length=20)
    use_llm: bool = False
    repeat_after_days: int = Field(default=30, ge=1, le=365)

    @field_validator("categories")
    @classmethod
    def _category_lengths(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(len(category) > 100 for category in values):
            raise ValueError("单个分类不能超过 100 个字符")
        return values


@app.get("/api/subscribe/profile")
def subscribe_profile(limit: int = Query(200, ge=1, le=500)):
    """从库内论文反推的兴趣画像（可解释：每个词项带权重）。"""
    return subscribe.build_profile(limit)


@app.post("/api/subscribe/digest")
def subscribe_digest(body: DigestBody, background: BackgroundTasks):
    """生成新论文摘报。要联网拉 arXiv，一律走任务系统，不阻塞 HTTP 线程。"""
    job_id = jobs.create_job("digest", body.model_dump(exclude_none=True))
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/subscribe/digests")
def subscribe_digests(limit: int = Query(20, ge=1, le=100)):
    return {"digests": subscribe.list_digests(limit)}


@app.get("/api/subscribe/digests/{digest_id}")
def subscribe_digest_detail(digest_id: int):
    d = subscribe.get_digest(digest_id)
    if not d:
        raise HTTPException(status_code=404, detail="摘报不存在")
    return d


@app.post("/api/subscribe/digests/{digest_id}/adopt")
def subscribe_adopt(digest_id: int, norm_key: str = Query(..., min_length=1, max_length=300)):
    try:
        return subscribe.adopt(digest_id, norm_key)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/subscribe/digests/{digest_id}/dismiss")
def subscribe_dismiss(digest_id: int, norm_key: str = Query(..., min_length=1, max_length=300)):
    try:
        return subscribe.dismiss(digest_id, norm_key)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── 结构化文献对比矩阵 ──

class MatrixBody(_PaperIdsBody):
    paper_ids: list[int] = Field(min_length=1, max_length=50)
    columns: list[dict] | None = Field(default=None, max_length=30)
    use_llm: bool | None = None
    topic: str | None = Field(default=None, max_length=1000)
    fmt: str | None = Field(default=None, max_length=20)

    @field_validator("columns")
    @classmethod
    def _column_payload_size(cls, values: list[dict] | None) -> list[dict] | None:
        if values is not None and len(json.dumps(values, ensure_ascii=False)) > 200_000:
            raise ValueError("columns 参数不能超过 200KB")
        return values


@app.post("/api/matrix")
def build_matrix(body: MatrixBody):
    """跨论文按自定义列抽成对比表。每个格子带来源与机械回取校验结果，
    抽不到的如实留空——coverage / verified_rate 让人一眼看出这张表有多少是真有依据的。"""
    try:
        m = matrix.build(body.paper_ids, body.columns, body.use_llm, topic=body.topic)
    except llm.LLMUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.fmt:
        try:
            return PlainTextResponse(matrix.export(m, body.fmt))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return m


@app.get("/api/matrix/presets")
def matrix_presets():
    return {"presets": matrix.list_presets(), "default": matrix.DEFAULT_COLUMNS}


class PresetBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    columns: list[dict] = Field(min_length=1, max_length=30)

    @field_validator("columns")
    @classmethod
    def _column_payload_size(cls, values: list[dict]) -> list[dict]:
        if len(json.dumps(values, ensure_ascii=False)) > 200_000:
            raise ValueError("columns 参数不能超过 200KB")
        return values


@app.post("/api/matrix/presets")
def matrix_save_preset(body: PresetBody):
    try:
        return matrix.save_preset(body.name, body.columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/matrix/presets/{name}")
def matrix_delete_preset(name: str):
    if not matrix.delete_preset(name):
        raise HTTPException(status_code=404, detail="预设不存在")
    return {"deleted": True, "name": name}


# ── PDF 版面结构（章节切分 + 参考文献抽取）──

@app.get("/api/paper/{paper_id}/structure")
def paper_structure(paper_id: int, max_chars: int = Query(4000, ge=100, le=50_000)):
    """按字号/编号启发式识别章节，并抽参考文献条目。0 token、不联网。"""
    return structure.summarize(str(_local_pdf(paper_id)), max_chars=max_chars)


# ── 迭代检索（agentic retrieval）──

class DeepBody(BaseModel):
    """迭代检索 / 迭代问答 / RCS 共用的入参。

    上下界不是装饰：`deep_answer` 的上下文篇数是 top_k × max_rounds，两个字段原来都是
    裸 int 无约束，`POST max_rounds=50` 就是 50 次 LLM 调用 + 250 篇论文的上下文，
    一个请求打穿配额。max_rounds 上限与 deepsearch.MAX_CONTEXT_PAPERS 对齐
    （5 × top_k 上限 20 已远超篇数闸门，闸门兜底、这里挡住明显的误用）。
    """
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(5, ge=1, le=20)
    max_rounds: int = Field(2, ge=1, le=5)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value


@app.post("/api/search/deep")
def deep_search(body: DeepBody):
    """派生查询补位的迭代检索（0 token，纯检索，不作答）。"""
    ids, mode, trace = deepsearch.deep_retrieve(body.question, body.top_k)
    with db.conn() as c:
        papers = [dict(r) for pid in ids
                  if (r := c.execute(
                      "SELECT id,title,year,venue FROM papers WHERE id=?",
                      (pid,)).fetchone())]
    return {"paper_ids": ids, "papers": papers, "mode": mode, "trace": trace}


@app.post("/api/answer/rcs")
def rcs_answer(body: DeepBody, background: BackgroundTasks):
    """RCS 增强问答：宽检索 → LLM 重排 → 逐篇定向摘要（依据句机械校验）→ 作答。

    与 /api/answer/deep 互补：deep 解决「检索没捞到」，RCS 解决「捞到了但没用好」。
    """
    if not llm.available():
        raise HTTPException(status_code=400,
                            detail="未配置 LLM——RCS 需要真模型（默认问答用 /api/chat/stream）")
    job_id = jobs.create_job("rcs_answer",
                             {"question": body.question, "top_k": body.top_k})
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/answer/deep")
def deep_answer(body: DeepBody, background: BackgroundTasks):
    """带自反馈的迭代问答：作答 → 机械找无证据论断 → 模型提补充查询 → 补检索 → 重答。

    收敛与每轮的查询/新增文献都记在结果的 trace 里，可审计。
    多轮 LLM 调用天然是长任务，走任务系统（结果从 /api/jobs/{id} 取）。
    """
    if not llm.available():
        raise HTTPException(status_code=400,
                            detail="未配置 LLM——迭代问答需要真模型（纯检索用 /api/search/deep）")
    job_id = jobs.create_job("deep_answer", {
        "question": body.question, "top_k": body.top_k,
        "max_rounds": body.max_rounds})
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/health/retrieval")
def retrieval_health(live: bool = False):
    """检索健康探针。默认 live=False（0 次网络调用），传 live=true 才打真实 embedding。

    `search_hybrid` 里本来就有降级说明，但那是单次调用的即时反馈、散落在各处的
    degraded 字段里，没人会去看。这个端点把它变成一条可以被监控轮询的断言。
    """
    from . import health
    return health.probe(live=live)


@app.get("/api/stats")
def stats():
    db.init_db()
    with db.conn() as c:
        s = db.stats(c)
        s["vectors"] = c.execute("SELECT COUNT(*) n FROM vectors").fetchone()["n"]
    return s


@app.get("/api/paper/{paper_id}")
def paper_detail(paper_id: int):
    db.init_db()
    with db.conn() as c:
        r = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="论文不存在")
        pages = [p["page_no"] for p in c.execute(
            "SELECT page_no FROM pages WHERE paper_id=? ORDER BY page_no",
            (paper_id,)).fetchall()]
    d = dict(r)
    d["card"] = cards.card_json(d.pop("card_json") or "{}")
    d["authors"] = json.loads(d.get("authors") or "[]")
    d["pages"] = pages
    # 本机绝对路径不出网（列表端点已 pop，详情端点原来漏了）；只暴露「有无本地 PDF」
    d["has_pdf"] = bool(d.pop("pdf_path", None))
    return d


# ── 问答 ──

class ChatBody(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=50)
    # 必须有上下界。`top_k` 一路传到 SQL 的 LIMIT，而 **SQLite 的 `LIMIT -1` 是
    # 「不限」**（实测：10 行的表 `LIMIT -1` 返回 10 行）——传 -1 会让一次请求
    # 把全库拉回来拼进 prompt。上界与 `/api/answer/deep` 的 DeepBody 对齐（le=20）：
    # 同一个参数在相邻端点上两套口径本身就是坑。
    top_k: int = Field(5, ge=1, le=20)
    session_id: str | None = None   # 传了就启用服务端会话记忆；不传行为与以前一致

    @field_validator("messages")
    @classmethod
    def _message_bounds(cls, values: list[dict]) -> list[dict]:
        allowed = {"system", "user", "assistant"}
        total = 0
        for message in values:
            if not isinstance(message, dict) or message.get("role") not in allowed:
                raise ValueError("messages 中每项必须包含合法 role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("消息内容不能为空")
            if len(content) > 20_000:
                raise ValueError("单条消息不能超过 20000 个字符")
            total += len(content)
        if total > 100_000:
            raise ValueError("对话总长度不能超过 100000 个字符")
        return values


@app.post("/api/chat")
def do_chat(body: ChatBody):
    try:
        sid = chat.ensure_session(body.session_id, _last_user(body.messages)) \
            if body.session_id is not None else None
        return rag.answer(body.messages, body.top_k, session_id=sid)
    except (budget.BudgetExceeded, budget.BudgetUnavailable):
        raise
    except Exception as e:
        _log.exception("chat 失败 session=%s", body.session_id)
        raise HTTPException(status_code=502, detail="问答服务暂时不可用，请稍后重试") from e


def _last_user(messages: list[dict]) -> str:
    return next((m.get("content", "") for m in reversed(messages)
                 if m.get("role") == "user"), "")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
def do_chat_stream(body: ChatBody):
    """流式问答：先推 sources（检索毫秒级），再逐段推正文（SSE delta）。

    用同步生成器——StreamingResponse 会在线程池里迭代，不阻塞事件循环。
    带 session_id 时，问答两侧都会落库：历史以服务端为准，刷新页面不丢。
    """
    q_text = _last_user(body.messages)
    # session_id 传 "" 或任意串都会拿到一个可用会话；传 None 表示不要会话（老行为）
    sid = chat.ensure_session(body.session_id, q_text) if body.session_id is not None else None
    try:
        ctx = rag.prepare(body.messages, body.top_k, session_id=sid)
    except (budget.BudgetExceeded, budget.BudgetUnavailable):
        raise
    except Exception as e:
        _log.exception("流式问答检索失败 session=%s", sid)
        raise HTTPException(status_code=502, detail="检索服务暂时不可用，请稍后重试") from e
    sys, q, sources, degraded, notes = ctx

    def gen():
        # degraded_detail 与 degraded 一同下发：前端渲染那句中文，
        # 监控/脚本按 code 聚合，两边看的是同一批信号。
        yield _sse("sources", {"sources": sources, "degraded": degraded,
                               "degraded_detail": degrade.as_dicts(notes),
                               "session_id": sid})
        acc: list[str] = []
        try:
            for chunk in llm.chat_stream(sys, q, purpose="chat", temperature=0.4):
                acc.append(chunk)
                yield _sse("delta", {"t": chunk})
        except Exception as e:
            # 半截回答也要入库，否则下一轮的历史里会凭空少一轮。
            # 中断本身也是一条降级，且要和本轮检索的降级一起留下——原来这里
            # 直接用一句「输出中断」覆盖掉了 degraded，检索侧的信号就此丢失。
            broken = degrade.merge(notes, [degrade.Degradation(
                degrade.STREAM_INTERRUPTED, f"输出中断：{type(e).__name__}")])
            if sid:
                chat.append(sid, "user", q)
                chat.append(sid, "assistant", "".join(acc), sources,
                            degrade.render(broken), broken)
            _log.exception("流式问答输出失败 session=%s", sid)
            yield _sse("error", {"error": "输出服务暂时不可用，请稍后重试",
                                 "degraded": degrade.render(broken),
                                 "degraded_detail": degrade.as_dicts(broken)})
            return
        if sid:
            chat.append(sid, "user", q)
            chat.append(sid, "assistant", "".join(acc), sources, degraded, notes)
        yield _sse("done", {"answer": "".join(acc), "degraded": degraded,
                            "degraded_detail": degrade.as_dicts(notes),
                            "session_id": sid})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── 对话会话（历史可回放）──

@app.get("/api/chat/sessions")
def chat_sessions(limit: int = Query(30, ge=1, le=100)):
    return {"sessions": chat.list_sessions(limit)}


@app.get("/api/chat/sessions/{session_id}")
def chat_session_detail(session_id: str):
    s = chat.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    return s


@app.delete("/api/chat/sessions/{session_id}")
def chat_session_delete(session_id: str):
    if not chat.delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True, "session_id": session_id}


class SurveyBody(BaseModel):
    topic: str = Field(min_length=1, max_length=2000)
    # 综述比问答吃更多篇，但同样不能无界：`survey._context_blocks` 没有字符预算，
    # top_k 大到一定程度就会拼出十几万字符的 prompt，而 `llm.chat` 对 400 不重试
    # ——用户看到的是一句透传的 provider 英文报错。
    top_k: int = Field(10, ge=1, le=30)

    @field_validator("topic")
    @classmethod
    def _topic_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("主题不能为空")
        return value


@app.post("/api/survey")
def do_survey(body: SurveyBody):
    try:
        return survey.generate(body.topic, body.top_k)
    except (budget.BudgetExceeded, budget.BudgetUnavailable):
        raise
    except Exception as e:
        _log.exception("survey 生成失败 topic=%s", body.topic)
        raise HTTPException(status_code=502, detail="综述服务暂时不可用，请稍后重试") from e


# ── 引用推荐 ──

class CiteBody(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("引用上下文不能为空")
        return value


@app.post("/api/cite")
def do_cite(body: CiteBody):
    return cite.recommend(body.text, body.top_k)


class ExportBody(_PaperIdsBody):
    paper_ids: list[int] = Field(min_length=1, max_length=100)
    fmt: str = Field(default="bibtex", max_length=20)


@app.post("/api/export", response_class=PlainTextResponse)
def do_export(body: ExportBody):
    try:
        return cite.export(body.paper_ids, body.fmt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ── 写作流水线（选题/大纲/撰写/文献/润色 五 Agent 确定性流水线）──
# 两个人工检查点：定题（选题后）、改纲（大纲后）。检查点 = 任务结束等指令，
# 确认即提交下一阶段任务——断点续跑天然成立。文献 Agent 实施引用白名单制。

def _start_write_job(run_id: str, stage: str, background: BackgroundTasks,
                     extra: dict | None = None, max_attempts: int = 2) -> str:
    params = {"run_id": run_id, "stage": stage} | (extra or {})
    try:
        # A transient provider/database failure should replay the idempotent
        # stage once. The job table persists the attempt count across restarts.
        job_id = jobs.create_job("write", params, dedupe_key=f"write:{run_id}",
                                 max_attempts=max_attempts)
    except jobs.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    pipeline.attach_job(run_id, job_id)
    background.add_task(jobs.submit, job_id)
    return job_id


@app.post("/api/write/runs")
def write_create_run(body: WriteRunCreate, background: BackgroundTasks):
    try:
        run = pipeline.create_run(body.topic, body.project_id,
                                  body.target_words, body.max_rewrites,
                                  body.pause_after_section)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = _start_write_job(run["id"], "topic", background)
    return {"run": run, "job_id": job_id}


@app.get("/api/write/runs")
def write_runs(limit: int = Query(20, ge=1, le=100)):
    db.init_db()
    return {"runs": pipeline.list_runs(limit)}


@app.get("/api/write/runs/{run_id}")
def write_state(run_id: str):
    state = pipeline.get_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    return state


@app.post("/api/write/runs/{run_id}/topic")
def write_pick_topic(run_id: str, body: WriteTopicSelect, background: BackgroundTasks):
    try:
        run = pipeline.select_topic(run_id, body.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = _start_write_job(run_id, "outline", background)
    return {"run": run, "job_id": job_id}


@app.post("/api/write/runs/{run_id}/outline")
def write_confirm_outline(run_id: str, background: BackgroundTasks,
                          body: WriteOutlineUpdate | None = None):
    """body 缺省 = 不改纲直接开写；带 sections = 人工改纲（版本 +1）。"""
    try:
        pipeline.save_outline(run_id,
                              [s.model_dump() for s in body.sections] if body else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = _start_write_job(run_id, "sections", background)
    return {"run": pipeline.get_run(run_id), "job_id": job_id}


@app.post("/api/write/runs/{run_id}/sections/{sec_no}/regenerate")
def write_regenerate_section(run_id: str, sec_no: int, background: BackgroundTasks):
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    if run["status"] not in ("drafted", "done", "failed", "drafting"):
        raise HTTPException(status_code=400,
                            detail=f"当前状态 {run['status']} 不能重跑章节")
    job_id = _start_write_job(run_id, "section", background, {"sec_no": sec_no})
    return {"run": run, "job_id": job_id}


@app.post("/api/write/runs/{run_id}/polish")
def write_polish(run_id: str, background: BackgroundTasks):
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    if run["status"] not in ("drafted", "done", "failed"):
        raise HTTPException(status_code=400,
                            detail=f"当前状态 {run['status']} 不能润色（先完成逐节撰写）")
    # 先建任务再改状态：建任务可能因同 run 已有任务在跑而 409，
    # 反过来写会把 run 卡在「polishing」而其实没有任务在推进它
    job_id = _start_write_job(run_id, "polish", background)
    pipeline._update_run(run_id, status="polishing", stage="润色与机械终检中", error=None)
    return {"run": pipeline.get_run(run_id), "job_id": job_id}


@app.post("/api/write/runs/{run_id}/audit")
def write_audit(run_id: str, background: BackgroundTasks):
    """Re-run the deterministic claim/citation/number/term audit."""
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    if run["status"] not in ("drafted", "done", "failed", "polishing"):
        raise HTTPException(status_code=400,
                            detail=f"当前状态 {run['status']} 不能审计（先完成逐节撰写）")
    job_id = _start_write_job(run_id, "audit", background)
    pipeline._update_run(run_id, stage="机械一致性审计中", error=None)
    return {"run": pipeline.get_run(run_id), "job_id": job_id}


@app.get("/api/write/runs/{run_id}/report")
def write_report(run_id: str):
    """引用一致性报告（纯机械，任意阶段可算）。"""
    if not pipeline.get_run(run_id):
        raise HTTPException(status_code=404, detail="写作任务不存在")
    try:
        return pipeline.report(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/write/runs/{run_id}/coordinator")
def write_coordinator_state(run_id: str):
    """Return the replayable Agent states, next action and hand-off messages."""
    if not pipeline.get_run(run_id):
        raise HTTPException(status_code=404, detail="写作任务不存在")
    c = coordinator.Coordinator(run_id)
    return {"plan": c.plan(), "states": c.states(), "messages": c.messages(200)}


@app.get("/api/write/runs/{run_id}/export")
def write_export(run_id: str, fmt: str = "md", mode: str = "final"):
    """Final-gated export.

    ``mode=final`` only succeeds when ``status == "done"``.  ``mode=draft`` is
    always available after a polish run and serves the last exported files with
    a ``X-Papernest-Export-Mode`` header so the front-end can show the
    ``needs_review`` banner instead of a clean final document.
    """
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    files = (run.get("result") or {}).get("files") or {}
    if not files:
        raise HTTPException(status_code=400, detail="终稿还没导出（先完成润色终检阶段）")
    export_mode = (run.get("result") or {}).get("export_mode")
    if mode == "final":
        if run["status"] != "done":
            raise HTTPException(
                status_code=409,
                detail=f"run 状态 {run['status']}，未通过审计闸门，禁止以 final 模式导出；"
                       "改用 mode=draft 下载未通过审计的稿件")
        effective_mode = "final"
    else:
        effective_mode = export_mode or "draft"
    path = files.get(fmt)
    if not path:
        raise HTTPException(status_code=404, detail=f"没有 {fmt} 格式的导出文件")
    resolved = Path(path).resolve()
    if (not resolved.is_relative_to(Path(config.DATA_DIR).resolve())
            or not resolved.is_file()):
        raise HTTPException(status_code=404, detail="导出文件不存在或不在数据目录内")
    media = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
             if fmt == "docx" else "text/markdown; charset=utf-8")
    headers = {"X-Papernest-Export-Mode": effective_mode}
    return FileResponse(resolved, filename=resolved.name, media_type=media,
                        headers=headers)


# ── L2 精读 ──

@app.post("/api/paper/{paper_id}/read")
def do_read(paper_id: int):
    return fulltext.read_paper(paper_id)


@app.get("/api/paper/{paper_id}/pdf")
def paper_pdf(paper_id: int):
    """把本地 PDF 喂给网页内嵌阅读器（连续阅读用；跳页走 /page/{n}.png）。

    只服务 data/ 目录下的文件：pdf_path 虽然是本项目自己写进去的，但它是一列
    普通 TEXT，任何一条写错/被改过的记录都会变成任意文件读取。这里按解析后的
    真实路径做一次归属校验，不依赖「写进去的时候应该是对的」。
    """
    return FileResponse(_local_pdf(paper_id), media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


def _local_pdf(paper_id: int) -> Path:
    """取该论文的本地 PDF 真实路径，并校验它确实在库目录内。"""
    with db.conn() as c:
        row = c.execute("SELECT pdf_path FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="论文不存在")
    if not row["pdf_path"]:
        raise HTTPException(status_code=404, detail="这篇论文没有本地 PDF（未精读或未上传）")
    path = Path(row["pdf_path"]).resolve()
    if not path.is_relative_to(Path(config.DATA_DIR).resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF 文件已不在库目录内或已被删除")
    return path


@app.get("/api/paper/{paper_id}/page/{page_no}.png")
def paper_page_image(paper_id: int, page_no: int, dpi: int = 110,
                     bbox_x1: float | None = None, bbox_y1: float | None = None,
                     bbox_x2: float | None = None, bbox_y2: float | None = None):
    """把指定页渲染成 PNG；传完整 bbox 时在原 PDF 坐标上高亮该区域。

    为什么不靠 `<iframe src=...#page=N>`：浏览器内置 PDF 查看器对 `#page=` 的支持
    各家不一，实测在本项目的内嵌环境里根本不跳页。而「点结论上的页码 → 看到那一页」
    正是证据链要兑现的动作，不能建立在一个不保证的行为上。服务端渲染是确定的。
    """
    import pymupdf
    path = _local_pdf(paper_id)
    dpi = max(60, min(dpi, 300))
    doc = pymupdf.open(path)
    try:
        if not (1 <= page_no <= doc.page_count):
            raise HTTPException(status_code=404,
                                detail=f"页码超出范围（本文共 {doc.page_count} 页）")
        page = doc.load_page(page_no - 1)
        bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        if all(v is not None and math.isfinite(v) for v in bbox):
            rect = pymupdf.Rect(*bbox) & page.rect
            if rect.is_valid and not rect.is_empty:
                rect += (-3, -2, 3, 2)
                rect &= page.rect
                page.draw_rect(rect, color=(0.95, 0.58, 0.05),
                               fill=(1.0, 0.85, 0.25), fill_opacity=0.24,
                               width=1.4, overlay=True)
        pix = page.get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
    finally:
        doc.close()
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/paper/{paper_id}/pages")
def paper_pages(paper_id: int, offset: int = Query(0, ge=0),
                limit: int = Query(50, ge=1, le=200)):
    with db.conn() as c:
        if not c.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise HTTPException(status_code=404, detail="论文不存在")
        total = c.execute("SELECT COUNT(*) n FROM pages WHERE paper_id=?",
                          (paper_id,)).fetchone()["n"]
        rows = c.execute("""SELECT page_no, text FROM pages WHERE paper_id=?
                           ORDER BY page_no LIMIT ? OFFSET ?""",
                         (paper_id, limit, offset)).fetchall()
    return {"count": total, "returned": len(rows), "offset": offset, "limit": limit,
            "pages": [dict(r) for r in rows]}


@app.get("/api/paper/{paper_id}/formulas")
def paper_formulas(paper_id: int):
    with db.conn() as c:
        items = db.formulas_for_paper(c, paper_id)
        audit = db.document_artifact(c, paper_id, "formula_audit")
    return {"count": len(items), "items": items,
            "audit": dict(audit) if audit else None}


@app.get("/api/paper/{paper_id}/markdown")
def paper_markdown(paper_id: int, rebuild: bool = False):
    from . import formulas, fulltext
    pdf_path = _local_pdf(paper_id) if rebuild else None
    with db.conn() as c:
        if rebuild:
            if pdf_path:
                chunks = fulltext.build_chunks(str(pdf_path), c, paper_id)
                db.replace_chunks(c, paper_id, chunks)
                content = fulltext.rebuild_canonical_markdown(c, paper_id, chunks)
            else:
                content = fulltext.rebuild_canonical_markdown(c, paper_id)
        else:
            artifact = db.document_artifact(c, paper_id, "canonical_markdown",
                                            formulas.PARSER_VERSION)
            if not artifact:
                content = fulltext.rebuild_canonical_markdown(c, paper_id)
            else:
                content = artifact["content"]
    return Response(content=content, media_type="text/markdown; charset=utf-8")


# ── 符号大全 ──

@app.post("/api/paper/{paper_id}/symbols")
def do_extract_symbols(paper_id: int):
    try:
        return symbols.extract_for_paper(paper_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log.exception("符号抽取失败 paper_id=%s", paper_id)
        raise HTTPException(status_code=500, detail="符号抽取失败，请查看服务日志") from e


@app.get("/api/symbols")
def get_symbols(q: str = Query("", max_length=500), kind: str = Query("", max_length=100),
                limit: int = Query(200, ge=1, le=500)):
    db.init_db()
    return {"count": 0, "items": symbols.library_symbols(q, kind, limit)}


# ── 方法对比表 ──

class TableBody(_PaperIdsBody):
    paper_ids: list[int] = Field(min_length=1, max_length=50)
    aspect: str = Field(default="", max_length=1000)


@app.post("/api/table")
def do_table(body: TableBody):
    try:
        return tablegen.compare(body.paper_ids, body.aspect)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        _log.exception("对比表生成失败 paper_ids=%s", body.paper_ids)
        raise HTTPException(status_code=500, detail="对比表生成失败，请查看服务日志") from e


# ── 汇报 PPT（从已核验卡片零 LLM 装配；单篇=论文汇报，多篇=文献汇报）──

class PptxBody(_PaperIdsBody):
    paper_ids: list[int] = Field(min_length=1, max_length=50)
    topic: str = Field(default="", max_length=1000)


@app.post("/api/pptx")
def do_pptx(body: PptxBody):
    try:
        result = pptgen.deck(body.paper_ids, body.topic)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    path = Path(result["path"])
    return FileResponse(
        path, filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")
