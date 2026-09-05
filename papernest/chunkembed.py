"""章节块向量化：给 `chunks` 表补上 kind='chunk' 的向量。

**要补的是一个真实缺口，不是锦上添花。** 库里 784 个章节全文块一条向量都没有，
`vectors` 表只有 kind='paper'（标题+摘要）60 条和 kind='sent'（摘要句）391 条。
后果是**语义检索只覆盖标题和摘要，正文只能靠关键词字面命中**——中文问
「近场信道估计怎么做的」而正文是英文，词面对不上，可向量本来正是解决这个的。

三条约束贯穿本模块：

- **花钱前先看账**。`estimate()` 在动手之前给出「多少块、多少字、大概多少 token」，
  `embed_chunks(dry_run=True)` 一次接口都不调。全库 784 块约 129 万字符，
  按 chars/4 粗估约 32 万 token——这个数应该在按下回车之前就看到。
- **降级不静默**。未配置 embedding 时 `embed_chunks()` 直接抛 `EmbedUnavailable`
  而不是返回 0：这是要花钱的操作，静默跳过会让人以为已经做完了。截断了几条、
  清洗了几条、哪一批失败了（失败的是哪几块），全部进返回值，不吞。
- **重复调用不重复花钱**。`pending()` 用左连接排除已有向量的块；写入前再按
  (paper_id, kind='chunk', idx, model) 删一次，并发下也写不出两条。

不新建表：`chunks` 与 `vectors` 都在 db.py 的 SCHEMA 里，本模块只是往
`vectors` 里多写一种 kind，所以没有自己的 ensure_schema()。
"""
import re
import time
from collections.abc import Callable, Iterable

import numpy as np

from . import config, db, embeddings

#: 单条文本送进 embedding 前的字符上限。与 `embeddings.embed_texts` 内部的
#: `t[:6000]` 保持一致——我们自己先截是为了**能计数**：交给下游静默截断的话，
#: 「这批里有 4 条正文被砍掉了尾巴」这个事实就丢了，而它直接影响这些块的召回质量。
MAX_CHARS = 6000

#: `vectors.text` 只存前 400 字，与现有 paper/sent 级一致。这一列是给人看的锚点，
#: 不是回取正文的地方——正文按 (paper_id, chunk_no) 回 `chunks` 表取，不留第二份。
TEXT_PREVIEW = 400

#: token 粗估系数。**是粗估不是分词**：中文一个字往往就是 1 个 token（比 1/4 贵得多），
#: 英文一个 token 约 4 个字符。混合语料下这个数只用来判断「量级对不对」。
CHARS_PER_TOKEN = 4

#: `embeddings.embed_texts` 内部**固定按 16 条拆 HTTP 请求**，一段失败就整个调用抛异常，
#: 前面已经成功（已计费、已进 llm_calls）的那几段被丢弃。实测 batch=32、第 2 个 HTTP
#: 失败时：付了 1600 prompt_tokens，库里 0 条向量。所以本模块的 batch 上限就是 16——
#: 传更大只会加宽失败的爆炸半径，一次 HTTP 请求都省不下来。
API_BATCH = 16


def _boot():
    """chunks / vectors 都在 db.py 的 SCHEMA 里，本模块不建表，只保证库已初始化。"""
    db.init_db()


def _to_blob(vec) -> bytes:
    """必须与 `embeddings._from_blob` 的 np.float32 一致，否则读出来是乱码浮点。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


#: PDF 抽取残留的 C0 控制字符（保留 \t \n \r，它们是正常排版）。
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str) -> str:
    """去掉控制字符。**这是从真库里量出来的**，不是想象的边界：784 块里有 10 块
    含 C0 控制字符（共 43 个，其中 1 块含 3 个 \\x00），来自 PDF 字体编码残留——
    合成夹具永远碰不到这条（db.replace_chunks 的 strip() 去不掉 \\x00）。

    为什么非做不可：JSON 里的 \\u0000 会被不少 embedding 端点判成非法入参，
    **整批 16 条一起失败**，而且每次重跑都会在同一条上再失败——这一批会永远补不上。
    控制字符对语义没有贡献，去掉不损失信息；但去掉了要计数上报（结果里的 sanitized）。
    """
    return _CTRL_RE.sub("", text)


def _store_dim(model: str) -> int | None:
    """库内该模型**已有**向量的维度（float32，4 字节一个分量）；没有则 None。

    口径与 `vectorstore._store_dim` 一致，且**不限 kind**：同一个 model 下混进两种维度，
    `vectorstore._snapshot()` 会对整个库抛 DimensionMismatch（不只是 chunk 那部分不能用，
    是 paper/sent 一起废），所以这个基准必须跨 kind 取。
    """
    with db.conn() as c:
        row = c.execute("SELECT LENGTH(vec) n FROM vectors WHERE model=? LIMIT 1",
                        (model,)).fetchone()
    return (row["n"] // 4) if row else None


def _n_chars(item: dict) -> int:
    """pending() 给的项自带 n_chars；外部手搓的项按 text 现算，不假设调用方填了。"""
    n = item.get("n_chars")
    return int(n) if n is not None else len(item.get("text") or "")


def pending(model: str | None = None) -> list[dict]:
    """列出**还没有向量**的章节块，按 (paper_id, chunk_no) 升序。

    判定口径：`chunks` 左连 `vectors`，连接条件是
    kind='chunk' AND idx=chunk_no AND model=当前模型——三个条件缺一不可。
    少了 model 就会把「换模型之后旧维度的向量」当成已完成（那批向量与新模型不同空间，
    用它检索出来的余弦是没有意义的数）。

    空文本块被排除：它们嵌不出有意义的向量，留在 pending 里只会让这个列表永远排不空。
    `db.replace_chunks` 本来就不写空块，所以真实库里不该出现——这是防御性的。

    排序键是 chunks 的主键 (paper_id, chunk_no)，**全序**，跨进程不漂移。
    """
    _boot()
    m = config.EMBED_MODEL if model is None else model
    with db.conn() as c:
        rows = c.execute(
            """SELECT c.paper_id, c.chunk_no, c.section_path, c.text
                 FROM chunks c
                 LEFT JOIN vectors v
                   ON v.paper_id = c.paper_id AND v.kind = 'chunk'
                      AND v.idx = c.chunk_no AND v.model = ?
                WHERE v.id IS NULL AND c.text IS NOT NULL
                ORDER BY c.paper_id, c.chunk_no""", (m,)).fetchall()
    out = []
    for r in rows:
        text = r["text"]
        if not text.strip():          # SQLite 的 TRIM 只去空格，去不掉 \n\t，这里在 Python 侧判
            continue
        out.append({"paper_id": r["paper_id"], "chunk_no": r["chunk_no"],
                    "text": text, "section_path": r["section_path"] or "",
                    "n_chars": len(text)})
    return out


def estimate(items: Iterable[dict]) -> dict:
    """动手前的成本估算。返回 n_chunks / total_chars / est_tokens / note。

    `est_tokens` 按**截断后**的字符数算（`est_chars`），因为那才是真正会付钱的量；
    `total_chars` 仍报原始字符数，两个数一起看才知道有多少内容被砍掉了。
    额外的 est_chars / n_over_limit 是为了让这个估算能手算复核。
    """
    items = list(items)
    total_chars = sum(_n_chars(it) for it in items)
    est_chars = sum(min(_n_chars(it), MAX_CHARS) for it in items)
    n_over = sum(1 for it in items if _n_chars(it) > MAX_CHARS)
    note = (f"est_tokens = est_chars/{CHARS_PER_TOKEN} 的**粗估**，不是分词结果："
            f"中文一个字常常就是 1 个 token（比这个估算贵数倍），英文才接近 4 字符/token。"
            f"只用来判断量级。est_chars 是截断到 {MAX_CHARS} 字后的字符数，"
            f"本次有 {n_over} 条超限会被截断。")
    return {"n_chunks": len(items), "total_chars": total_chars,
            "est_chars": est_chars, "n_over_limit": n_over,
            "est_tokens": int(round(est_chars / CHARS_PER_TOKEN)), "note": note}



CONSECUTIVE_FAIL_LIMIT = 3
#: 这些迹象说明「端点本身不提供这个能力」，不是瞬时抖动——重试多少次都一样。
_PERMANENT_MARKS = ("HTTP 404", "HTTP 401", "HTTP 403", "HTTP 400",
                    "未配置 LLM_API_KEY", "未配置")


def _permanent_reason(exc: BaseException) -> str:
    msg = str(exc)
    for mark in _PERMANENT_MARKS:
        if mark in msg:
            return mark
    return type(exc).__name__


def _is_permanent(exc: BaseException) -> bool:
    """区分「端点没这个能力」与「这次网络抖了一下」。

    实测踩到：中转端点只有 chat 路由，`/embeddings` 恒 404，而原实现会把 49 个批次
    全部跑完（100 秒）才收工——每一批都在验证同一件已知的事。4xx 与缺配置都属于
    重试无意义的永久错误，第一批就该停。5xx / 超时 / 连接错误不在此列，那些值得重试。
    """
    return any(m in str(exc) for m in _PERMANENT_MARKS)


def embed_chunks(limit: int | None = None, batch: int = 16,
                 progress: Callable[[dict], None] | None = None,
                 dry_run: bool = False) -> dict:
    """把待办的章节块分批嵌入并写进 `vectors`（kind='chunk'、idx=chunk_no）。

    参数
    - limit：只处理 pending 的前 N 条（按全序截断，所以「先跑 50 条看看」是可复现的）。
    - batch：每批条数，默认 16，**上限也是 16**（见 `API_BATCH`：embed_texts 内部按 16
      拆 HTTP，传更大只是让一次失败连坐更多已付费的请求，省不下任何一次调用）。
    - progress：每批结束回调一次，收到
      {"batch","batches","embedded","total","errors"}。回调自己抛异常不会中断嵌入
      （钱已经花了，不能因为一个打印函数丢掉结果），但会记进 errors 并停止后续回调。
    - dry_run：只算账、一次接口都不调。**未配置 embedding 时 dry_run 仍可用**，
      因为它不花钱。

    返回 embedded / skipped / truncated / sanitized / batches / errors /
    est_tokens / elapsed_s。
    - `skipped` = 选中但没写进库的条数（= 失败批次里的条数）。
    - `truncated` 只统计**成功写入**的截断条数——它描述的是「库里有几条向量的原文被砍过」，
      失败批次没进库，不该算在里面。
    - `sanitized` = 送出前被去掉控制字符的条数（见 `_sanitize`，真实库里有 10 条）。
    - `blank` = 清洗后变成空串、**根本没送出去**的条数（见下）。
    - `errors` 每项 {"batch","n","chunks","error"}，chunks 是 (paper_id, chunk_no) 列表，
      失败的到底是哪几块要能查出来，不能只给一句「有 1 批失败」。
    - `batches` 是**实际跑过**的批数、`planned_batches` 是计划的批数；维度不一致会提前
      收工，两个数不等就说明中止了（`aborted` 里是原因）。

    单批失败不中断整体：每批一个独立事务，前面成功的不回滚，后面的继续跑。
    **唯一的例外是维度不一致**：那是全局性的（库内 model 已经是 A 维、端点给 B 维），
    每一批都会以同样的理由失败，继续跑只是继续付钱，所以第一次撞上就收工。
    """
    _boot()
    # 未配置就明确报错。**不能返回 0 了事**：这是花钱的操作，静默"完成"会让人
    # 以为正文已经可检索了，而实际上一条向量都没有——本项目最忌讳的那种假成功。
    if not dry_run and not embeddings.available():
        raise embeddings.EmbedUnavailable(
            "未配置 LLM_API_KEY / EMBED_MODEL，无法嵌入章节块。"
            "这是要花钱的操作，静默跳过会让人误以为已经做完；"
            "先配置好再跑，或用 embed_chunks(dry_run=True) 看估算。")

    t0 = time.perf_counter()
    items = pending()
    if limit is not None:
        items = items[:max(0, int(limit))]
    est = estimate(items)
    batch = min(max(1, int(batch)), API_BATCH)
    # 清洗后变成空串的块**一条都不送**：OpenAI 兼容端点会把空字符串判成非法入参，
    # 而入参非法是**整个请求**失败——一条全是控制字符的块能把同批 15 条正常块一起拖下水，
    # 且每次重跑都在同一批上再失败。这正是 _sanitize 想避免的"永远补不上的一批"，
    # 不能在清洗这一步自己造出来。真实库当前 0 条，属于防御性过滤，但要计数上报。
    live = [it for it in items if _sanitize(it["text"]).strip()]
    blank = len(items) - len(live)
    groups = [live[i:i + batch] for i in range(0, len(live), batch)]

    if dry_run:
        # 返回值是 estimate 的超集：run 侧的键全给 0，并显式带 dry_run=True。
        # 这样调用方读 result["embedded"] 不会 KeyError，也不会把 0 误读成"跑过但没嵌上"。
        return {**est, "dry_run": True, "model": config.EMBED_MODEL,
                "embedded": 0, "skipped": 0, "truncated": 0, "sanitized": 0,
                "blank": blank, "batches": 0, "planned_batches": len(groups),
                "aborted": None, "errors": [],
                "elapsed_s": round(time.perf_counter() - t0, 3)}

    embedded = 0
    truncated = 0
    sanitized = 0
    errors: list[dict] = []
    aborted: str | None = None
    consecutive = 0     # 连续失败批次数，见 CONSECUTIVE_FAIL_LIMIT
    # 维度基准**先从库里取**，不是等第一批跑完才定：库内该 model 已经是 1024 维、
    # 端点这次给 768 维的话，本模块要是照写不误，`vectors` 里同一个 model 就混了两种维度，
    # 于是 `vectorstore._snapshot()` 对**整个库**抛 DimensionMismatch——paper/sent 的检索
    # 一起废掉，而不只是新写的 chunk 不能用。vectorstore 自己的写路径有这道守卫
    # （`_write_items` 的 `_store_dim` 比对），本模块直写 vectors，必须自己带上同一道。
    dim: int | None = _store_dim(config.EMBED_MODEL)

    def fail(bi: int, group: list[dict], msg: str) -> None:
        errors.append({"batch": bi, "n": len(group),
                       "chunks": [(it["paper_id"], it["chunk_no"]) for it in group],
                       "error": msg[:300]})

    def do_batch(bi: int, group: list[dict]) -> tuple[int, int] | None:
        """跑一批。成功返回 (截断条数, 清洗条数)，失败返回 None（错误已记进 errors）。"""
        nonlocal dim, aborted
        cleaned = [_sanitize(it["text"]) for it in group]
        texts = [t[:MAX_CHARS] for t in cleaned]
        try:
            vecs = embeddings.embed_texts(texts, purpose="embed_chunk")
        except Exception as e:                                  # noqa: BLE001
            fail(bi, group, f"embed_texts 失败 {type(e).__name__}: {e}")
            if _is_permanent(e):
                aborted = (f"嵌入端点不可用（{_permanent_reason(e)}）：这类错误之后每批"
                           f"都会同样失败。已在第 {bi}/{len(groups)} 批停手，未写入任何向量。"
                           f"请检查 LLM_API_BASE 是否提供 /embeddings 路由、"
                           f"以及 EMBED_MODEL 是否为该端点的真实模型 ID。")
            return None
        if len(vecs) != len(group):
            # 条数对不上就没法保证 zip 的对应关系，宁可整批丢弃也不能错位存——
            # 错位的向量比没有向量更糟：检索会给出看似有理、实则张冠李戴的证据。
            fail(bi, group, f"返回向量条数不匹配：期望 {len(group)}、实到 {len(vecs)}")
            return None
        sizes = {len(v) for v in vecs}
        if len(sizes) != 1 or 0 in sizes:
            fail(bi, group, f"向量维度异常：本批 {sorted(sizes)}、基准 {dim}")
            return None
        if dim is not None and sizes != {dim}:
            # 全局性错误：库内该 model 就是 dim 维，端点给的是另一种，之后每批都会同样失败。
            # 记一次、收工，不要用 49 批的钱去验证同一件事。
            aborted = (f"向量维度不一致：库内 model={config.EMBED_MODEL!r} 是 {dim} 维、"
                       f"本次端点返回 {sorted(sizes)[0]} 维。混写会让 vectorstore 对整个库"
                       f"抛 DimensionMismatch（paper/sent 一起不可用）。已在第 {bi} 批停手，"
                       f"未写入任何向量；请先确认模型名与维度，或重建索引。")
            fail(bi, group, aborted)
            return None
        with np.errstate(over="ignore"):
            # 超出 float32 范围的分量会溢出成 inf——那正是下面要抓的东西，
            # 不需要 numpy 再警告一次（BLOB 本来就是 float32，溢出不可避免）。
            arr = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(arr).all():
            # NaN/Inf 会**静默**毁掉排序：归一化后整行是 NaN，与之比较的一切都是 False，
            # 检索分数直接变 nan（实测 vectorstore 返回 [1.0, nan, 1.0, ...]）。
            # 与"条数不匹配"同样处理：整批丢弃——坏向量比没有向量更糟。
            # （vectorstore._norm_item 也拒 NaN/Inf，本模块直写 vectors，守卫要自己带。）
            bad = int(np.count_nonzero(~np.isfinite(arr).all(axis=1)))
            fail(bi, group, f"返回向量含 NaN/Inf：{bad}/{len(group)} 条，整批丢弃")
            return None
        vecs = arr
        try:
            with db.conn() as c:        # 一批一个事务：失败批次不牵连已成功的批次
                for it, clean, v in zip(group, cleaned, vecs):
                    # pending 已排除有向量的块，这一删是并发下的兜底：两个进程同时跑
                    # 也只会剩一条，不会把同一块存两遍（存两遍等于同一证据在检索里投两票）。
                    c.execute("DELETE FROM vectors WHERE paper_id=? AND kind='chunk' "
                              "AND idx=? AND model=?",
                              (it["paper_id"], it["chunk_no"], config.EMBED_MODEL))
                    c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                              "VALUES(?,?,?,?,?,?)",
                              (it["paper_id"], "chunk", it["chunk_no"],
                               clean[:TEXT_PREVIEW], config.EMBED_MODEL, _to_blob(v)))
        except Exception as e:                                  # noqa: BLE001
            fail(bi, group, f"写库失败 {type(e).__name__}: {e}")
            return None
        dim = next(iter(sizes))
        # 截断按**清洗后**的长度判，否则一条靠控制字符撑过 6000 的块会被误报成截断
        return (sum(1 for t in cleaned if len(t) > MAX_CHARS),
                sum(1 for it, t in zip(group, cleaned) if t != it["text"]))

    ran = 0
    for bi, group in enumerate(groups, 1):
        ran = bi
        done = do_batch(bi, group)
        if done is not None:
            embedded += len(group)
            truncated += done[0]
            sanitized += done[1]
            consecutive = 0
        else:
            consecutive += 1
            if aborted is None and consecutive >= CONSECUTIVE_FAIL_LIMIT:
                # 单批失败可能是瞬时抖动，连着失败就不是了。继续跑只是拿
                # 剩下几十批的时间和钱去验证同一件事（实测端点 404 时
                # 跑满 49 批白耗 100 秒）。
                aborted = (f"连续 {consecutive} 批失败，已在第 {bi}/{len(groups)} 批停手。"
                           f"最后一条错误：{errors[-1]['error'][:160]}")
        # 失败的批次也要报进度：进度条卡住不动是"还在跑"还是"在连环失败"，
        # 调用方必须能分得出来（errors 计数就在回调里）。
        if progress is not None:
            try:
                progress({"batch": bi, "batches": len(groups), "embedded": embedded,
                          "total": len(items), "errors": len(errors)})
            except Exception as e:                              # noqa: BLE001
                errors.append({"batch": bi, "n": 0, "chunks": [],
                               "error": f"progress 回调抛异常，已停止回调："
                                        f"{type(e).__name__}: {e}"[:300]})
                progress = None
        if aborted:
            break

    return {"embedded": embedded, "skipped": len(items) - embedded,
            "truncated": truncated, "sanitized": sanitized, "blank": blank,
            "batches": ran, "planned_batches": len(groups),
            "aborted": aborted, "errors": errors,
            "est_tokens": est["est_tokens"],
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "model": config.EMBED_MODEL, "dry_run": False}


def coverage(model: str | None = None) -> dict:
    """一眼看出正文向量补到什么程度。

    返回 chunks_total / chunks_embedded / coverage / papers_with_chunks /
    papers_fully_embedded（外加 model，因为覆盖率是**按模型**算的——换模型就归零）。

    `chunks_total` 是 chunks 表的原始条数，不排除空文本块：空块永远嵌不上，
    如实留在分母里比悄悄把分母改小诚实（`db.replace_chunks` 不写空块，真实库里为 0）。

    用 EXISTS 而不是 JOIN 计数：万一同一块因为并发写出了两条向量，JOIN 会把覆盖率
    算超过 100%，EXISTS 不会——统计口径不该被数据异常带偏。
    """
    _boot()
    m = config.EMBED_MODEL if model is None else model
    has_vec = ("EXISTS(SELECT 1 FROM vectors v WHERE v.paper_id=c.paper_id "
               "AND v.kind='chunk' AND v.idx=c.chunk_no AND v.model=?)")
    with db.conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        done = c.execute(f"SELECT COUNT(*) n FROM chunks c WHERE {has_vec}", (m,)).fetchone()["n"]
        papers = c.execute("SELECT COUNT(DISTINCT paper_id) n FROM chunks").fetchone()["n"]
        full = c.execute(
            f"""SELECT COUNT(*) n FROM (
                  SELECT c.paper_id FROM chunks c GROUP BY c.paper_id
                   HAVING SUM(CASE WHEN {has_vec} THEN 0 ELSE 1 END) = 0)""",
            (m,)).fetchone()["n"]
    return {"chunks_total": total, "chunks_embedded": done,
            "coverage": round(done / total, 4) if total else 0.0,
            "papers_with_chunks": papers, "papers_fully_embedded": full,
            "model": m}
