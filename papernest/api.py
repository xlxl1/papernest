"""FastAPI 服务：库管理 + Agent + 异步任务/SSE + RAG 问答 + 引用推荐 + L2 精读。"""
import asyncio
import hmac
import json
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import (agent, bibimport, cards, chat, cite, config, db, deepsearch, degrade,
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

app = FastAPI(title="PaperNest", docs_url=None, redoc_url=None)
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
        got = (request.headers.get("x-api-key")
               or request.query_params.get("api_key") or "")
        # **必须先 encode**：`hmac.compare_digest` 对两个 str 只支持 ASCII，
        # 含中文的口令直接 TypeError —— 中间件里没有兜底、也没有全局异常处理器，
        # 结果是每个 /api/* 恒定 500 而不是 401，前端只在 401 时才弹口令输入框，
        # 用户连「口令错了」都看不到。而 docker-compose 现在用中文提示强制要求口令，
        # 填中文口令是完全正常的选择。encode 之后仍是常数时间比较。
        if not hmac.compare_digest(got.encode("utf-8"), API_KEY.encode("utf-8")):
            return JSONResponse({"detail": "缺少或错误的 API key"}, status_code=401)
    return await call_next(request)


@app.exception_handler(Exception)
async def _log_unhandled(request: Request, exc: Exception):
    """未捕获的异常：**先记下来**，再返回 500。

    没有这个处理器时，traceback 只会被 uvicorn 打到控制台——后台起服务
    （`serve &`、docker、开机自启）时那份输出根本没人接，出了事只剩一句
    「500 Internal Server Error」。本机单用户不需要告警，但需要**事后查得到**。
    响应体保持 FastAPI 默认的形状，不把内部错误文本透给客户端。
    """
    _log.exception("未处理的异常 %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "服务器内部错误，详见 data/papernest.log"},
                        status_code=500)


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
        print(f"[papernest] 已把 {n} 个中断任务标记为 failed（服务重启）")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


# ── 库 ──

@app.get("/api/papers")
def papers(q: str = "", limit: int = 50,
           year_from: int | None = None, year_to: int | None = None):
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
    query: str = ""
    keywords: list[str] = []
    limit: int = 50
    year_from: int | None = None
    year_to: int | None = None


# ── Research Agent ──

@app.post("/api/agent/plan")
def agent_plan(body: AgentRequest):
    """Return the selected tool plan without calling external services."""
    return agent.run_agent(body.goal, body.top_k, body.max_steps,
                           body.topic, dry_run=True).model_dump()


@app.post("/api/agent/run")
def agent_run(body: AgentRequest):
    """Run a bounded tool workflow and return its execution timeline."""
    return agent.run_agent(body.goal, body.top_k, body.max_steps,
                           body.topic, dry_run=body.dry_run,
                           timeout_s=body.timeout_s).model_dump()


@app.get("/api/agent/runs")
def agent_runs(limit: int = 20):
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
    kind: str  # ingest | read | survey | agent | polish | review | write
    params: dict = {}


@app.post("/api/jobs")
def create_job(body: JobCreate, background: BackgroundTasks):
    if body.kind not in jobs.RUNNERS:
        raise HTTPException(status_code=400,
                            detail=f"未知任务类型 {body.kind}（可选 {sorted(jobs.RUNNERS)}）")
    # 写作流水线的阶段任务按 run 去重，避免两个阶段并发改同一批 sections
    dedupe = (f"write:{body.params.get('run_id')}"
              if body.kind == "write" and body.params.get("run_id") else None)
    try:
        job_id = jobs.create_job(body.kind, body.params, dedupe_key=dedupe)
    except jobs.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "kind": body.kind, "status": "queued"}


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
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
            if job["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


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
    document = workspace.update_document(document_id, **body.model_dump(exclude_unset=True))
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
    results, failed = [], []
    for f in files:
        # 分块读到上限就停：原来是先 `await f.read()` 整个读进内存、**再**判大小，
        # 防线设在内存已经被占满之后，等于没有。
        data = _read_capped(f.file, MAX_UPLOAD_BYTES)
        if data is None:
            failed.append({"filename": f.filename,
                           "error": f"超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限"})
            continue
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
def list_uploads(limit: int = 50):
    return {"uploads": pdfimport.list_uploads(limit)}


class MetadataFix(BaseModel):
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None


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
    text: str
    fmt: str = "auto"          # auto | bibtex | ris | csv
    enrich: bool = False       # 用 Semantic Scholar 补缺失摘要（会联网）
    source_name: str = ""


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
                             fmt: str = "auto", enrich: bool = False):
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
def import_runs(limit: int = 20):
    return {"runs": bibimport.list_imports(limit)}


# ── 引文网络：邻居图 / 相关推荐 / 阅读缺口 ──

@app.get("/api/graph/stats")
def graph_stats():
    return graph.stats()


@app.post("/api/paper/{paper_id}/edges")
def graph_fetch_edges(paper_id: int, direction: str = "both", limit: int = 100,
                      max_age_days: int = 30):
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
def graph_neighbors(paper_id: int, depth: int = 1, limit: int = 50):
    # graph.neighbors 对不在库的 paper_id 抛 GraphError。不接就是 500 + 裸 traceback，
    # 而同一模块紧邻的另外两个路由都规规矩矩映射成 400——同一类错误两种口径。
    try:
        return graph.neighbors(paper_id, depth, limit)
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class RelatedBody(BaseModel):
    paper_ids: list[int]
    top_k: int = 20


@app.post("/api/graph/related")
def graph_related(body: RelatedBody):
    """共被引 + 文献耦合推荐（纯读库，不联网、不调 LLM）。"""
    try:
        return {"related": graph.related(body.paper_ids, body.top_k)}
    except graph.GraphError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/graph/gaps")
def graph_gaps(top_k: int = 20):
    """被库内论文频繁引用、但自己不在库里的论文——最明确的阅读缺口。"""
    return {"gaps": graph.gap_papers(top_k)}


@app.post("/api/graph/adopt")
def graph_adopt(norm_key: str):
    """把图里的库外论文一键入库。"""
    try:
        return graph.adopt(norm_key)
    except graph.GraphError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── 新论文订阅推送 ──

class DigestBody(BaseModel):
    days: int = 3
    top_k: int = 15
    categories: list[str] | None = None
    use_llm: bool = False
    repeat_after_days: int = 30


@app.get("/api/subscribe/profile")
def subscribe_profile(limit: int = 200):
    """从库内论文反推的兴趣画像（可解释：每个词项带权重）。"""
    return subscribe.build_profile(limit)


@app.post("/api/subscribe/digest")
def subscribe_digest(body: DigestBody, background: BackgroundTasks):
    """生成新论文摘报。要联网拉 arXiv，一律走任务系统，不阻塞 HTTP 线程。"""
    job_id = jobs.create_job("digest", body.model_dump(exclude_none=True))
    background.add_task(jobs.submit, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/subscribe/digests")
def subscribe_digests(limit: int = 20):
    return {"digests": subscribe.list_digests(limit)}


@app.get("/api/subscribe/digests/{digest_id}")
def subscribe_digest_detail(digest_id: int):
    d = subscribe.get_digest(digest_id)
    if not d:
        raise HTTPException(status_code=404, detail="摘报不存在")
    return d


@app.post("/api/subscribe/digests/{digest_id}/adopt")
def subscribe_adopt(digest_id: int, norm_key: str):
    try:
        return subscribe.adopt(digest_id, norm_key)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/subscribe/digests/{digest_id}/dismiss")
def subscribe_dismiss(digest_id: int, norm_key: str):
    try:
        return subscribe.dismiss(digest_id, norm_key)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── 结构化文献对比矩阵 ──

class MatrixBody(BaseModel):
    paper_ids: list[int]
    columns: list[dict] | None = None
    use_llm: bool | None = None
    topic: str | None = None
    fmt: str | None = None       # 给了就直接返回导出文本（csv / markdown / latex）


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
    name: str
    columns: list[dict]


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
def paper_structure(paper_id: int, max_chars: int = 4000):
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
    question: str
    top_k: int = Field(5, ge=1, le=20)
    max_rounds: int = Field(2, ge=1, le=5)


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
    messages: list[dict]
    # 必须有上下界。`top_k` 一路传到 SQL 的 LIMIT，而 **SQLite 的 `LIMIT -1` 是
    # 「不限」**（实测：10 行的表 `LIMIT -1` 返回 10 行）——传 -1 会让一次请求
    # 把全库拉回来拼进 prompt。上界与 `/api/answer/deep` 的 DeepBody 对齐（le=20）：
    # 同一个参数在相邻端点上两套口径本身就是坑。
    top_k: int = Field(5, ge=1, le=20)
    session_id: str | None = None   # 传了就启用服务端会话记忆；不传行为与以前一致


@app.post("/api/chat")
def do_chat(body: ChatBody):
    try:
        sid = chat.ensure_session(body.session_id, _last_user(body.messages)) \
            if body.session_id is not None else None
        return rag.answer(body.messages, body.top_k, session_id=sid)
    except Exception as e:
        # 形状不改：前端把 answer 渲染成气泡里的 ⚠，改成非 2xx 会打坏它。
        # 但异常必须留痕——否则 traceback 连同栈帧一起消失，事后无从复盘。
        _log.exception("chat 失败 session=%s", body.session_id)
        return {"answer": f"⚠ {e}", "sources": [], "error": True}


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}") from e
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
            yield _sse("error", {"error": f"{type(e).__name__}: {str(e)[:200]}",
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
def chat_sessions(limit: int = 30):
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
    topic: str
    # 综述比问答吃更多篇，但同样不能无界：`survey._context_blocks` 没有字符预算，
    # top_k 大到一定程度就会拼出十几万字符的 prompt，而 `llm.chat` 对 400 不重试
    # ——用户看到的是一句透传的 provider 英文报错。
    top_k: int = Field(10, ge=1, le=30)


@app.post("/api/survey")
def do_survey(body: SurveyBody):
    try:
        return survey.generate(body.topic, body.top_k)
    except Exception as e:
        _log.exception("survey 生成失败 topic=%s", body.topic)
        return {"survey": f"⚠ {e}", "sources": [], "checks": [], "error": True}


# ── 引用推荐 ──

class CiteBody(BaseModel):
    text: str
    top_k: int = 5


@app.post("/api/cite")
def do_cite(body: CiteBody):
    return cite.recommend(body.text, body.top_k)


class ExportBody(BaseModel):
    paper_ids: list[int]
    fmt: str = "bibtex"


@app.post("/api/export", response_class=PlainTextResponse)
def do_export(body: ExportBody):
    try:
        return cite.export(body.paper_ids, body.fmt)
    except ValueError as e:
        return str(e)


# ── 写作流水线（选题/大纲/撰写/文献/润色 五 Agent 确定性流水线）──
# 两个人工检查点：定题（选题后）、改纲（大纲后）。检查点 = 任务结束等指令，
# 确认即提交下一阶段任务——断点续跑天然成立。文献 Agent 实施引用白名单制。

def _start_write_job(run_id: str, stage: str, background: BackgroundTasks,
                     extra: dict | None = None) -> str:
    params = {"run_id": run_id, "stage": stage} | (extra or {})
    try:
        job_id = jobs.create_job("write", params, dedupe_key=f"write:{run_id}")
    except jobs.JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    pipeline.attach_job(run_id, job_id)
    background.add_task(jobs.submit, job_id)
    return job_id


@app.post("/api/write/runs")
def write_create_run(body: WriteRunCreate, background: BackgroundTasks):
    try:
        run = pipeline.create_run(body.topic, body.project_id,
                                  body.target_words, body.max_rewrites)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = _start_write_job(run["id"], "topic", background)
    return {"run": run, "job_id": job_id}


@app.get("/api/write/runs")
def write_runs(limit: int = 20):
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


@app.get("/api/write/runs/{run_id}/report")
def write_report(run_id: str):
    """引用一致性报告（纯机械，任意阶段可算）。"""
    if not pipeline.get_run(run_id):
        raise HTTPException(status_code=404, detail="写作任务不存在")
    try:
        return pipeline.report(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/write/runs/{run_id}/export")
def write_export(run_id: str, fmt: str = "md"):
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="写作任务不存在")
    if run["status"] != "done" or not (run.get("result") or {}).get("files"):
        raise HTTPException(status_code=400, detail="终稿还没导出（先完成润色终检阶段）")
    path = (run["result"]["files"] or {}).get(fmt)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"没有 {fmt} 格式的导出文件")
    media = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
             if fmt == "docx" else "text/markdown; charset=utf-8")
    return FileResponse(path, filename=Path(path).name, media_type=media)


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
def paper_page_image(paper_id: int, page_no: int, dpi: int = 110):
    """把指定页渲染成 PNG。

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
        pix = doc.load_page(page_no - 1).get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
    finally:
        doc.close()
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/paper/{paper_id}/pages")
def paper_pages(paper_id: int):
    with db.conn() as c:
        rows = c.execute("SELECT page_no, text FROM pages WHERE paper_id=? ORDER BY page_no",
                         (paper_id,)).fetchall()
    return {"count": len(rows), "pages": [dict(r) for r in rows]}


# ── 符号大全 ──

@app.post("/api/paper/{paper_id}/symbols")
def do_extract_symbols(paper_id: int):
    try:
        return symbols.extract_for_paper(paper_id)
    except Exception as e:
        _log.exception("符号抽取失败 paper_id=%s", paper_id)
        return {"error": str(e)}


@app.get("/api/symbols")
def get_symbols(q: str = "", kind: str = "", limit: int = 200):
    db.init_db()
    return {"count": 0, "items": symbols.library_symbols(q, kind, limit)}


# ── 方法对比表 ──

class TableBody(BaseModel):
    paper_ids: list[int]
    aspect: str = ""


@app.post("/api/table")
def do_table(body: TableBody):
    try:
        return tablegen.compare(body.paper_ids, body.aspect)
    except Exception as e:
        _log.exception("对比表生成失败 paper_ids=%s", body.paper_ids)
        return {"error": str(e)}


# ── 汇报 PPT（从已核验卡片零 LLM 装配；单篇=论文汇报，多篇=文献汇报）──

class PptxBody(BaseModel):
    paper_ids: list[int]
    topic: str = ""


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
