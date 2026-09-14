"""Embedding 服务（OpenAI 兼容 /embeddings）与向量存取。

- 向量按 (paper_id, kind, model) 存 SQLite BLOB，numpy 余弦检索。
  **不上向量库是量出来的决定，不是偷懒**（1024 维实测，p50）：

  | 向量条数 | 原实现（每次重读 BLOB 逐行算） | 缓存归一化矩阵 | 常驻内存 |
  |----------|-------------------------------|----------------|----------|
  | 500      | 9.6 ms                        | 0.19 ms        | 2 MB     |
  | 10,000   | 203.4 ms                      | 1.80 ms        | 39 MB    |
  | 50,000   | 1028.6 ms                     | 8.87 ms        | 195 MB   |
  | 200,000  | 4743.2 ms                     | 41.74 ms       | 781 MB   |

  瓶颈从来不是余弦计算（纯矩阵乘 0.04ms），而是**每次查询重读 SQLite BLOB**——
  只做向量化仅快 1.3 倍，缓存矩阵快 50~113 倍。个人库全量嵌入约 5~6k 条向量，
  缓存后 1ms 级；即使 10 倍规模也只有 9ms。**真到 10 万条以上再换 sqlite-vec**
  （同一个单文件、不引服务进程），在那之前引向量库只是多一个依赖和一份数据副本。
- 未配置 EMBED_MODEL 时 available()=False，检索走 FTS/词面重叠兜底。
"""
import json
import os
import warnings
import threading
from typing import NamedTuple

import numpy as np

from . import budget, config, db, deadline, degrade, http, llm, vectorstore


class EmbedUnavailable(Exception):
    pass


def available() -> bool:
    return bool(config.EMBED_API_KEY and config.EMBED_MODEL)


#: 单请求条数。批越大越省往返，但也越容易读超时（实测长文本 16 条一批会 ReadTimeout）。
BATCH_SIZE = config._env_int(8, "PAPERNEST_EMBED_BATCH")
#: 瞬态失败的退避（秒）。**必须有**：原来一次读超时就让整条 `cli.py embed` 挂掉，
#: 而这条命令要跑几百次调用——没有重试等于「跑到哪算哪，全靠运气」。
_EMBED_DELAYS = (3, 8, 20)
#: 单请求读超时。成功的调用实测 0.6~1.0s，20s 已经非常宽松——
#: 而超时在这里是**探测信号**（端点对超限输入是挂起而不是报错），
#: 所以它必须便宜：60s 一次的话，几百条超限文本就是几小时。
_EMBED_TIMEOUT = config._env_int(20, "PAPERNEST_EMBED_TIMEOUT")


#: 单条输入的初始字符上限，以及自适应缩短时的下限。
#: **不能只靠一个固定上限**：token 与字符的比例随语言差好几倍（英文约 4 字符/token、
#: 中文常常 1 字符/token），同一个字符数在两种文本下的 token 数完全不同。
INPUT_CHARS = config._env_int(6000, "PAPERNEST_EMBED_CHARS", minimum=100)
MIN_INPUT_CHARS = 400

#: 超时被当成「可能是输入过长」的信号：某些端点（实测 qwen3.7-text-embedding-flash）
#: 在输入超过其 token 上限时**挂起而不是返回 400**——重试多少次都是同一个超时。
#: 实测：同一篇摘要截到 1870 字符正常返回、1878 字符必定超时（连测 4 次），
#: 而另外两篇在 2100 字符处完全正常。所以这不是固定的字符阈值，只能自适应。
_SHRINK_ON_TIMEOUT = True

#: 进程内「学到的」输入上限。一旦某个上限被证明可用，**后续批次直接从它开始**——
#: 否则每批都要重新从 6000 往下踩一遍超时，几百条长文本就是几小时。
#: 这是 16/804 条 chunk 跑了十几分钟的直接原因。
_LEARNED_CAP: list[int] = [INPUT_CHARS]


def _post_once(url: str, batch: list[str]):
    with http.client(timeout=_EMBED_TIMEOUT) as client:
        return client.post(
            url, json={"model": config.EMBED_MODEL, "input": batch},
            headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"})


def _post_with_retry(url: str, batch: list[str], bi: int, total: int
                     ) -> tuple[object, int | None]:
    """发一批嵌入请求。返回 (response, 生效的字符上限或 None)。

    三层处理，对应三类不同的失败：
    - **永久错误立即失败**：400（模型名错、额度耗尽）、401/403 —— 重试只是把同一个
      错再犯几遍。口径与 `llm.chat` 一致。
    - **瞬态错误退避重试**：429/5xx 与网络抖动。
    - **超时则缩短输入再试**：某些端点在输入超限时挂起而不是报错，此时退避多久都没用，
      只能把输入截短。截短会改变语义，所以**必须如实返回生效的上限**，由调用方上报。

    每一步都尊重 `deadline`：超时的步骤不该继续烧钱。
    """
    cap = _LEARNED_CAP[0]
    last = None
    while True:
        cur = [t[:cap] for t in batch]
        for delay in (*_EMBED_DELAYS, None):
            deadline.check()
            try:
                r = _post_once(url, cur)
            except Exception as e:                              # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:80]}"
                if "Timeout" in type(e).__name__ and _SHRINK_ON_TIMEOUT:
                    break                                       # 交给外层缩短输入
                if delay is None:
                    break
                deadline.sleep(delay)
                continue
            if r.status_code == 200:
                _LEARNED_CAP[0] = cap        # 记住这个上限，后续批次直接用
                return r, (cap if cap < INPUT_CHARS else None)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                if delay is None:
                    break
                deadline.sleep(delay)
                continue
            raise EmbedUnavailable(
                f"embeddings 接口失败 HTTP {r.status_code}：{r.text[:200]}")
        if not (_SHRINK_ON_TIMEOUT and cap > MIN_INPUT_CHARS
                and max((len(t) for t in batch), default=0) > MIN_INPUT_CHARS):
            break
        cap = max(MIN_INPUT_CHARS, cap // 2)                     # 缩短后重来一轮
    raise EmbedUnavailable(
        f"embeddings 第 {bi + 1} 批（共 {-(-total // BATCH_SIZE)} 批）失败："
        f"{last}（已尝试把单条输入缩到 {cap} 字符）")


#: 最近一次 embed_texts 里发生的截断：[最小生效上限, 被截短的批数]。空 = 没有截断。
#: 调用方（如 cli.py embed）据此如实提示，而不是让「向量只覆盖了半篇」悄悄发生。
LAST_TRUNCATION: list[int] = []


def embed_texts(texts: list[str], purpose: str = "embed") -> list[list[float]]:
    if not available():
        raise EmbedUnavailable("未配置 LLM_API_KEY / EMBED_MODEL")
    # 每日用量闸门。这条路径正是 2026-09-03 把免费额度跑光的那一条
    # （`cli.py embed` 一次几千条调用），所以它比 chat 更需要这道闸。
    budget.check("嵌入调用")
    import time
    out: list[list[float]] = []
    url = config.EMBED_API_BASE.rstrip("/") + "/embeddings"
    shrunk: list[int] = []
    for i in range(0, len(texts), BATCH_SIZE):  # 分批，规避单请求条数上限
        t0 = time.perf_counter()
        batch = [t[:INPUT_CHARS] for t in texts[i:i + BATCH_SIZE]]
        r, cap = _post_with_retry(url, batch, i // BATCH_SIZE, len(texts))
        if cap is not None:
            # 被截短过就要留痕：向量代表的不再是完整文本，静默截断会让人
            # 以为「这条向量覆盖了整篇」。
            shrunk.append(cap)
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        usage = r.json().get("usage") or {}
        with db.conn() as c:
            db.add_llm_call(c, purpose, None, usage.get("prompt_tokens", 0), 0,
                            config.EMBED_MODEL,
                            latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            c.commit()
        out.extend(d["embedding"] for d in data)
    if shrunk:
        LAST_TRUNCATION[:] = [min(shrunk), len(shrunk)]
    else:
        LAST_TRUNCATION.clear()
    return out


def _to_blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def index_paper(paper_id: int, title: str, abstract: str) -> int:
    """为论文建 paper 级与句子级向量（引用推荐的证据句检索用）。返回向量条数。

    清理只针对本函数自己写的两种 kind。**不能按 (paper_id, model) 一刀切**：
    chunk 向量是 `chunkembed` 另一条路径建的、本函数不会重建，一起删掉就是净丢数据。
    而 `cli.py embed` / `jobs.py` 的选片条件都是「没有 kind='paper' 向量的论文」——
    正文先入库、paper 级向量还没补的论文必然落进这个集合，于是「补建 paper 向量」
    这个动作会顺手抹掉该篇全部 chunk 向量，且没有任何报错或日志。
    同仓库另外两处删向量（chunkembed.py、db.replace_chunks）都是按 kind 收窄的。
    """
    text = f"{title}\n{abstract or ''}".strip()
    sentences = split_sentences(abstract or "")
    inputs = [text] + sentences
    vecs = embed_texts(inputs)
    with db.conn() as c:
        c.execute("DELETE FROM vectors WHERE paper_id=? AND model=? "
                  "AND kind IN ('paper','sent')",
                  (paper_id, config.EMBED_MODEL))
        c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) VALUES(?,?,?,?,?,?)",
                  (paper_id, "paper", 0, text[:400], config.EMBED_MODEL, _to_blob(vecs[0])))
        for j, (s, v) in enumerate(zip(sentences, vecs[1:])):
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) VALUES(?,?,?,?,?,?)",
                      (paper_id, "sent", j, s, config.EMBED_MODEL, _to_blob(v)))
        c.commit()
    invalidate_cache()
    return len(vecs)


_MATRIX_LOCK = threading.Lock()
_MATRIX: dict | None = None      # {"key": ..., "ids": ndarray, "M": 归一化矩阵}


def _matrix_key(c) -> tuple:
    """缓存有效性指纹。(条数, 最大 id) 覆盖本项目所有真实变更路径：
    `index_paper` 是先删后插（autoincrement 让 max(id) 增长）、
    换嵌入模型是 `UPDATE ... SET model=?`（改变该 model 的条数）。
    两个都走索引，成本亚毫秒——比每次重读几百 MB BLOB 便宜四个数量级。"""
    r = c.execute("""SELECT COUNT(1) n, COALESCE(MAX(id),0) m FROM vectors
                     WHERE kind IN ('paper','chunk') AND model=?""",
                  (config.EMBED_MODEL,)).fetchone()
    return (config.EMBED_MODEL, r["n"], r["m"])


def invalidate_cache():
    """向量变更后显式作废（index_paper 会调）。指纹检查已能兜住，这是双保险。"""
    global _MATRIX
    with _MATRIX_LOCK:
        _MATRIX = None


def _paper_matrix():
    """常驻内存的归一化向量矩阵（**paper 级 + chunk 级**）。返回 (ids, M) 或 (None, None)。

    一行一条向量，`ids` 是对应的 paper_id——同一篇论文会出现多行（1 条标题摘要向量
    + N 条章节块向量），调用方按 paper_id 去重取最好的一条即可。

    **为什么必须带 chunk**：只装 kind='paper' 时，语义检索只覆盖标题与摘要，
    正文内容只能靠 chunks_fts 的**关键词**命中——而向量本来就是用来解决
    「中文问、英文正文，词面对不上」这类问题的。实测踩到过：784 条 chunk 向量
    建好之后消融显示贡献恰好 +0.0000（自建集与 QASPER 都是），一查才发现
    向量路根本没读它们——**不是没效果，是没接上**。

    为什么不上向量库：实测 1024 维下**瓶颈是每次查询重读 SQLite BLOB，不是余弦计算**——
    向量化只快 1.3 倍，缓存矩阵快 50~113 倍。缓存之后 1 万条 1.8ms、5 万条 8.9ms、
    20 万条 41.7ms（781MB 内存），个人库场景（全库嵌入约 5~6k 条向量）远没到需要
    ANN 索引的规模。真到 10 万条以上再换 sqlite-vec——同一个单文件、不引服务进程。
    """
    global _MATRIX
    with db.conn() as c:
        key = _matrix_key(c)
        with _MATRIX_LOCK:
            if _MATRIX is not None and _MATRIX["key"] == key:
                return _MATRIX["ids"], _MATRIX["idxs"], _MATRIX["M"]
        rows = c.execute("""SELECT paper_id, kind, idx, vec FROM vectors
                            WHERE kind IN ('paper','chunk') AND model=?
                            ORDER BY paper_id, kind, idx""",
                         (config.EMBED_MODEL,)).fetchall()
    if not rows:
        with _MATRIX_LOCK:
            _MATRIX = {"key": key, "ids": None, "idxs": None, "M": None}
        return None, None, None
    ids = np.fromiter((r["paper_id"] for r in rows), dtype=np.int64, count=len(rows))
    # 每行是哪个 chunk：chunk 行记 chunk_no，paper 行记 -1。
    # **不带上它，语义命中就只能压成 paper_id**——上下文组装侧再也拿不回
    # 「是哪一段匹配上的」，只能退回按页词频重新猜一遍。
    idxs = np.fromiter((r["idx"] if r["kind"] == "chunk" else -1 for r in rows),
                       dtype=np.int64, count=len(rows))
    raw = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32)
    dim = raw.size // len(rows)
    M = raw.reshape(len(rows), dim)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)   # 预归一化：查询期只剩一次矩阵乘
    with _MATRIX_LOCK:
        _MATRIX = {"key": key, "ids": ids, "idxs": idxs, "M": M}
    return ids, idxs, M


#: 语义检索只认这两类向量：标题摘要（kind='paper'）与章节块（kind='chunk'）。
#: 句级向量（kind='sent'）是引用推荐那条路用的，混进论文级检索会让同一篇论文刷屏。
_PAPER_KINDS = ("paper", "chunk")
_KIND_WHERE = {"kind": {"$in": list(_PAPER_KINDS)}}

#: 向量后端的降级原因。检索是同步函数、任务跑在线程池里，所以按线程存，
#: 免得两个并发查询互相盖掉对方的降级记录。
_STORE_NOTE = threading.local()


def take_store_note() -> str | None:
    """取出并清空本线程上一次向量检索的降级原因。

    做成「取走即清空」是为了让调用点不会把上一轮的降级当成本轮的：降级记录一旦
    读漏或读串，界面上就会出现「这次好好的却报着降级」或反过来的假象。
    """
    note = getattr(_STORE_NOTE, "value", None)
    _STORE_NOTE.value = None
    return note


def _candidate_rows(top_k: int) -> int:
    """向后端要多少行候选。

    后端返回的是**向量行**，而这里要的是**论文**：一篇论文有 1 条标题摘要向量加
    N 条章节块向量，只按 top_k 取行的话，章节多的一两篇论文就能把名额占满，
    「检索到 5 篇」实际只有 1 篇。所以按行超取，去重之后再截到 top_k。
    真库一篇论文平均两三条向量，20 倍是留足余量的经验值；下限 64 是为了
    top_k 很小时也不至于取得太紧。
    """
    return max(int(top_k) * 20, 64)


def search_papers(query: str, top_k: int = 8) -> list[dict]:
    """论文级语义检索。返回 [{paper_id, score, chunk_no}]，按余弦相似度降序，全序。

    检索走 `vectorstore.get_store()` 选出来的后端（默认 Milvus，单容器形态设
    `PAPERNEST_VECTOR_BACKEND=numpy` 就地算），本函数不再自己读 BLOB 建矩阵——
    同一份 SQLite 真相，由谁来算是部署形态的选择，不该写死在检索链路里。

    一篇论文在后端里有多行（标题摘要向量 + 各章节块向量），这里**按 paper_id 取
    最高分那一行**再排序。降级原因不随返回值走，用 `take_store_note()` 取，
    这样上层（`search_hybrid`）能把它记成一条降级，而这个函数的签名不用变。
    """
    qv = np.asarray(embed_texts([query])[0], dtype=np.float32)
    store, note = vectorstore.get_store_or_degrade()
    want = _candidate_rows(top_k)
    try:
        rows = store.search(qv, want, where=_KIND_WHERE)
        if not rows and store.name != "numpy" and store.count() > 0:
            # 真相里有向量、派生索引却一行都不返回，说明索引落后（collection 还没建、
            # rebuild 没跑完）。这时退回真相来源现算，而不是把空结果交上去——
            # 空结果和「确实没有相关论文」在上层长得一模一样，那才是真正危险的静默失败。
            rows = vectorstore.NumpyStore().search(qv, want, where=_KIND_WHERE)
            if rows:
                note = (f"{store.name} 的索引落后于 SQLite 真相，本次退回本地计算；"
                        f"跑 `python cli.py vec rebuild` 重建索引")
    except vectorstore.DimensionMismatch as e:
        # 换了嵌入模型但库里还是旧维度：如实报错，不做静默截断（截断出来的
        # 余弦是没有意义的数，比报错更危险）。`cli.py embed` 负责迁移或重建。
        raise EmbedUnavailable(
            f"{e}；请先跑 `python cli.py embed` 迁移或重建索引") from e
    _STORE_NOTE.value = note
    best: dict[int, float] = {}
    best_chunk: dict[int, int] = {}
    for r in rows:
        pid, sc = int(r["paper_id"]), float(r["score"])
        if sc > best.get(pid, -2.0):
            best[pid] = sc
            # 记下是**哪一段**赢的：kind 是 'paper' 说明赢在标题摘要向量上，没有
            # 对应的正文块。这条信息是「语义命中能不能进上下文」的全部依据——
            # 中文问句配英文正文时词面命中恒为 0，只有向量这一路知道该喂哪段给模型。
            best_chunk[pid] = int(r["idx"]) if r.get("kind") == "chunk" else -1
    # 全序：分数并列时按 paper_id 决胜，否则跨进程结果会漂移
    order = sorted(best, key=lambda p: (-best[p], p))[:top_k]
    return [{"paper_id": p, "score": float(best[p]),
             "chunk_no": (best_chunk[p] if best_chunk[p] >= 0 else None)}
            for p in order]


def check_space_compatible(paper_id: int, title: str, abstract: str) -> bool | float:
    """换提供方后验证新旧嵌入空间是否一致：用同一文本重嵌入，与库内旧向量算余弦。

    返回 True（≥0.98，可原地迁移）/ False（需重建）/ False（失败）——异常返回 False。
    """
    try:
        with db.conn() as c:
            row = c.execute("SELECT vec FROM vectors WHERE paper_id=? AND kind='paper' "
                            "AND model != ? ORDER BY id DESC LIMIT 1",
                            (paper_id, config.EMBED_MODEL)).fetchone()
        if not row:
            return False
        old = _from_blob(row["vec"])
        new = np.asarray(embed_texts([f"{title}\n{abstract or ''}".strip()])[0], dtype=np.float32)
        n = min(len(old), len(new))
        if n == 0 or len(old) != len(new):
            return False  # 维度不同必然不同空间
        denom = (np.linalg.norm(old[:n]) * np.linalg.norm(new[:n])) or 1.0
        return True if float(old[:n] @ new[:n] / denom) >= 0.98 else float(
            round(old[:n] @ new[:n] / denom, 4))
    except Exception:
        return False


# L2 全文是否参与全库检索（检索单元 = 按章节切的 chunks，见 db.chunks）。
# `PAPERNEST_CHUNK_SEARCH=0` 关掉做对照，`PAPERNEST_CHUNK_WEIGHT` 调权重。
# 环境变量保留 PAGE_* 旧名兼容（早期这里是按页切的）。
CHUNK_SEARCH = (os.environ.get("PAPERNEST_CHUNK_SEARCH")
                or os.environ.get("PAPERNEST_PAGE_SEARCH", "1")) not in ("0", "false", "False")

# 块级命中的 RRF 权重。**测出来的，不是拍的**——两个评测集必须一起看：
#
# | 权重 | QASPER@1 | @5    | @10   | 自建@5 | @10    | @15    |
# |------|----------|-------|-------|--------|--------|--------|
# | 关闭 | 0.4098   | 0.6066| 0.6639| 0.6771 | 0.7396 | 0.7656 |
# | 0.2  | 0.4672   | 0.6230| 0.6967| 0.7083 | 0.7396 | 0.7500 |
# | 0.5  | 0.4672   | 0.6230| 0.6885| 0.7083 | 0.7083 | 0.7500 |
# | 1.0  | 0.4672   | 0.6639| 0.7377| 0.7083 | 0.7083 | 0.7188 |
#
# 取 0.2：QASPER 每个 k 都涨（@1 +0.057、@10 +0.033），自建集 @5 涨 0.031、@10 持平，
# 代价只有 @15 的 0.0156。只看 QASPER 会选 1.0——那要拿自建集 @10/@15 各三个多点去换。
# （更早按页切时最优点是 0.5；换成按章节切后最优点左移，说明块变小后单条命中更该被"小步加分"。）
#
# ⚠️ **「0.2 是低权重」这个直觉在默认档位上不成立，别照它推理。**
# RRF 的分数是 `weight / (rrf_k + rank + 1)`，而 rrf_k=60 远大于候选深度，
# 所以整条主路的**全部动态范围**都被压得很扁：
#   top_k=5（默认，候选深度 15）：权重 1.0 的主路从第 1 名到第 15 名总共只值
#     1/61 − 1/75 = 0.00306，而 chunk 路第 1 名就贡献 0.2/61 = 0.00328（1.07 倍）
#     —— 也就是说 chunk 路命中一次，足以抹平主路里任意名次差。
#   top_k=8  → 主路跨度 0.00449，chunk 首位 0.00328（0.73 倍，主路更大）
#   top_k=15 → 主路跨度 0.00687，chunk 首位 0.00328（0.48 倍）
# 结论：这个「权重」控制的是**跨路相对影响**，而它的实际量级由 rrf_k 与候选深度
# 共同决定、随 top_k 变化。上面那张表仍然有效（它是在默认档位上实测的），
# 但不要再把 0.2 解释成「显著低于主路」。要改 rrf_k 得两个评测集一起重测。
CHUNK_WEIGHT = config._env_float(0.2, "PAPERNEST_CHUNK_WEIGHT", "PAPERNEST_PAGE_WEIGHT")

# 旧名 PAGE_SEARCH / PAGE_WEIGHT 已删除。它们是 `= CHUNK_*` 的**副本**，而
# `search_hybrid` 读的是 CHUNK_*——写旧名是纯 no-op，却看着像生效了。
# 曾经唯一守护默认权重 0.2 的那条测试正是靠写 PAGE_WEIGHT，等于什么都没断言。
# 环境变量 `PAPERNEST_PAGE_WEIGHT` 作为用户侧的旧名仍然认（见上面的 _env_float）。


class RetrievalResult(NamedTuple):
    """一次检索的完整结果。**降级信号是返回值的一部分，不是可选的旁路。**

    原来这个函数只返回 `(ids, mode)`，把算好的降级说明直接丢掉，逼调用方从 mode
    反推——而 mode 又会被拼成 'fts+chunks'，`mode == "fts"` 的判断当场失效。
    把 degraded 放进返回值，调用方要么用它、要么显式忽略，没有「不小心漏掉」这条路。
    """

    ids: list[int]
    mode: str
    degraded: list[degrade.Degradation]
    #: {paper_id: [{chunk_no, section_path, start_page, text, via}]}
    #: 本次检索里**具体哪些块命中了**。上下文组装靠它把语义/词面命中直接喂给模型，
    #: 而不是拿 paper_id 回到 pages 表按词频重猜一遍。None/空 = 没有块级命中。
    #: 默认写 None 而不是 {}：NamedTuple 的默认值在所有实例间**共享**，
    #: 给可变对象当默认值，谁改一下就污染全部。调用方一律 `res.chunk_hits or {}`。
    chunk_hits: dict | None = None


def _merge_hits(lex: dict, vec: dict) -> dict[int, list[dict]]:
    """合并词面命中与语义命中，**语义那一块排前面**。

    语义命中通常更贴题（跨语言时词面根本没有命中），但词面命中带来多样性——
    两边都留，去重按 chunk_no。
    """
    out: dict[int, list[dict]] = {}
    for pid in set(lex) | set(vec):
        seen, merged = set(), []
        for src, via in ((vec.get(pid) or [], "vector"), (lex.get(pid) or [], "lexical")):
            for h in src:
                if h["chunk_no"] in seen:
                    continue
                seen.add(h["chunk_no"])
                merged.append({**h, "via": h.get("via", via)})
        out[pid] = merged
    return out


def search_hybrid(query: str, top_k: int = 8, rrf_k: int = 60,
                  use_pages: bool | None = None) -> RetrievalResult:
    """RRF 混合检索：向量排序 + 标题/摘要 FTS + 章节块全文 FTS 融合。

    返回 `RetrievalResult(ids, mode, degraded)`；degraded 是结构化的降级记录列表
    （空列表 = 没有降级），要给人看时用 `degrade.render(...)`。

    - 向量未配置 → 不进向量路（正常路径，**不算降级**，degraded 为空）
    - 向量已配置但整路没生效 → degraded 里必有一条 CRITICAL 记录

    **mode 只描述用了哪几路，不承载降级语义**。别再用 `mode == "fts"` 判降级：
    有 chunk 命中时 mode 是 'fts+chunks'，那个等号恒为假，降级会被整个吞掉。
    要判降级看 degraded；要判「向量这一路在不在」用 `mode.startswith("hybrid")`。
    """
    if use_pages is None:
        use_pages = CHUNK_SEARCH
    vec_ids: list[int] | None = None
    notes: list[degrade.Degradation] = []
    vec_chunk: dict[int, int] = {}          # paper_id -> 语义上赢的那个 chunk_no
    if available():
        try:
            take_store_note()           # 清掉上一轮的，免得把旧降级算到这轮头上
            hits = search_papers(query, top_k * 3)
            store_note = take_store_note()
            if store_note:
                notes.append(degrade.Degradation(
                    degrade.VECTOR_BACKEND_DEGRADED,
                    f"向量后端降级：{store_note}"))
            vec_ids = [h["paper_id"] for h in hits]
            vec_chunk = {h["paper_id"]: h["chunk_no"] for h in hits
                         if h.get("chunk_no") is not None}
            if not vec_ids:
                # 配了 EMBED_MODEL 却一条向量都没匹配上，最常见的原因是**模型名对不上**：
                # `.env` 里填的是控制台上的显示名，而建索引时用的是 API 的真实 ID。
                # 这种情况下检索会悄悄退化成纯 FTS 而用户以为混合检索开着——
                # 必须如实上报，不能静默。`cli.py models` 可核对真实 ID。
                notes.append(degrade.Degradation(
                    degrade.VECTOR_INDEX_EMPTY,
                    f"向量索引为空（EMBED_MODEL={config.EMBED_MODEL!r} "
                    f"匹配不到任何向量，检查模型名或先跑 cli.py embed），本次仅 FTS"))
        except Exception as e:
            notes.append(degrade.Degradation(
                degrade.VECTOR_CALL_FAILED,
                f"向量检索不可用（{str(e)[:80]}），已降级 FTS"))
    with db.conn() as c:
        fts_ids = [r["id"] for r in db.search_fts(c, query, top_k * 3)]
        # 检索走 chunks（按章节切）；老库迁移会把 pages 搬过来，所以这里不用兜底
        lex_hits = db.search_chunks_hits(c, query, top_k * 3) if use_pages else {}
        page_ids = sorted(lex_hits, key=lambda p: lex_hits[p][0]["rank"])[:top_k * 3]
        # 向量路赢在某个 chunk 上的论文：把那一块也取回来（词面命中恒为 0 的
        # 跨语言场景下，这是唯一知道该喂哪段的信息来源）
        vec_hits: dict[int, list[dict]] = {}
        for pid, cno in vec_chunk.items():
            got = db.chunks_by_no(c, pid, [cno])
            if got:
                got[0]["via"] = "vector"
                vec_hits[pid] = got

    rankings: list[tuple[list[int], float]] = [(fts_ids, 1.0)]
    mode = "fts"
    if vec_ids:
        rankings.append((vec_ids, 1.0))
        # 走到这里 notes 必为空：两处降级都只在 vec_ids 落空时才记
        # （异常路径下 vec_ids 连赋值都没发生）。原来这行是
        # `"hybrid" if degrade is None else "fts"`，else 分支不可达。
        mode = "hybrid"
    if page_ids:
        rankings.append((page_ids, CHUNK_WEIGHT))
        mode += "+chunks"
    chunk_hits = _merge_hits(lex_hits, vec_hits)
    if len(rankings) == 1:
        return RetrievalResult(fts_ids[:top_k], mode, notes, chunk_hits)

    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    # 分数并列时按 paper_id 决胜——排序键必须全序，否则跨进程结果会漂移
    merged = sorted(scores, key=lambda p: (-scores[p], p))
    return RetrievalResult(merged[:top_k], mode, notes, chunk_hits)


def best_sentence(paragraph_vec: np.ndarray, paper_id: int) -> tuple[str | None, float]:
    """该论文的句子级向量里，与段落最接近的一句。"""
    with db.conn() as c:
        rows = c.execute("SELECT text, vec FROM vectors WHERE paper_id=? AND kind='sent' AND model=?",
                         (paper_id, config.EMBED_MODEL)).fetchall()
    best, best_s = None, -1.0
    for r in rows:
        v = _from_blob(r["vec"])
        denom = (np.linalg.norm(paragraph_vec) * np.linalg.norm(v)) or 1.0
        s = float(paragraph_vec @ v / denom)
        if s > best_s:
            best, best_s = r["text"], s
    return best, (best_s if best_s > 0 else 0.0)


def split_sentences(text: str) -> list[str]:
    """中英混合切句；过滤过短的碎片。"""
    import re
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 12]
