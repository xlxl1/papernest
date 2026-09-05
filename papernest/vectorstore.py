"""向量存储后端抽象层：numpy 常驻归一化矩阵与 Milvus / Chroma 并存、可 A/B。

⚠️ **当前状态：这一层没有接进检索热路径。**
问答走的是 `rag.prepare → embeddings.search_hybrid → search_papers → embeddings._paper_matrix`，
直接从 SQLite `vectors` 表读 BLOB 建 numpy 矩阵，全程不 import 本模块。
`get_store()` 在全仓只有 `cli.py` 的 `vec status` / `vec rebuild` 两个调用点，
真库 `vector_meta` 表 0 行可佐证。写向量的三条路径（`embeddings.index_paper`、
`chunkembed`、`db.replace_chunks`）也都是自己拼 SQL 直写，不经过 `_write_items()`。

后果之一已经发生过：抽象层的删除语义是 per-(paper_id, kind, idx) 的，
而主写路径曾是 per-(paper_id, model) 的——粒度一宽就把同篇的 chunk 向量连带删了
（已于 2026-09-03 修复，见 `embeddings.index_paper` 与 `tests/test_index_paper.py`）。
**所以：不要说「线上检索由 Milvus 承载」；`PAPERNEST_VECTOR_BACKEND` 目前只影响
`cli.py vec` 子命令的行为。** 接线是待办项。

## 为什么 SQLite `vectors` 表是唯一真相来源，Chroma 只是派生索引

这是整个设计的关键约束，不是实现细节：

1. **一份数据两个副本必然漂移。** 进程崩在「Chroma add 成功、SQLite 事务回滚」之间、
   有人手工删了 `data/chroma/`、换嵌入模型时只重写了一半——每一种都会让两边对不上。
   规定了谁是真相，不一致时就只有一个正确解法：`rebuild()` 从 SQLite 重放。
   反过来（以 Chroma 为准）SQLite 就重建不出来了，一次目录损坏＝永久数据丢失。
2. **证据链要求引用能追到原文。** `vectors.text` 与 `chunks` / `pages` 同库同事务；
   Chroma 是另一个目录另一套文件。把证据文本的权威放进派生索引，等于让本项目
   「证据可审计」这条核心价值观依赖一个随时可以被删掉的缓存。所以两个后端的
   `search()` 都从 SQLite 回取 text——这也顺带保证了 A/B 时两边返回的文本逐字相同。
3. **后端可切换的前提**就是各自的索引都能从同一份真相重建，否则每 A/B 一次
   就要重新花钱嵌入一遍。

## 后端选型是量出来的，不是拍的（1024 维，p50 查询延迟）

| 规模   | Chroma(HNSW) | numpy 缓存矩阵 |
|--------|--------------|----------------|
| 1,000  | 1.99 ms      | 0.14 ms        |
| 5,000  | 2.66 ms      | 0.91 ms        |
| 20,000 | 2.91 ms      | 3.21 ms        |

Chroma 查询时间几乎不随规模增长（HNSW 次线性），**拐点约 2 万条**。库内当前约
1,200 条向量 → numpy 更快，所以 **numpy 是默认后端**，Chroma 是可选后端。
HNSW 近似召回用库里 451 条真实向量测得 recall@1=0.990、@5/@10=1.000，
在这个规模上与精确检索基本无差。

（测 ANN 必须用真实分布的向量：1024 维随机高斯向量彼此近乎等距、没有近邻结构可
索引，拿它测会得到 recall@5≈0.56 的假结论。本模块的测试夹具因此用聚类结构向量。）

## 上面那张表是「纯算分」的耗时，端到端不是这个数——别拿它汇报

拿真库（451 条 1024 维真实向量）量的端到端 `search()`：**numpy 3.3ms / chroma 6.5ms**。
拆开看，绝大部分既不是矩阵乘也不是 HNSW：

| 环节                          | 耗时    |
|-------------------------------|---------|
| `db.conn()` 后的**第一条**语句 | 2.48 ms |
| 指纹查询（同一连接内）         | 0.06 ms |
| 纯矩阵乘（451×1024）           | 0.08 ms |
| 按行号回取 10 条正文           | 0.24 ms |
| 纯 Chroma query                | 1.68 ms |

第一条语句贵是因为新连接要把整库 schema 读进来（这个库 sqlite_master 有 82 个对象）；
同一条连接上再查两次几乎免费（2.73→2.77ms），换成复用连接后指纹只要 0.061ms。
**所以本模块每次 search 只开一条连接**——开两条就是把 2.5ms 再付一遍（改之前正是如此，
端到端 9.4ms）。剩下的 2.5ms 是全项目 `with db.conn() as c:` 约定的固有成本，
`embeddings.search_papers` 今天也一样在付；要再降只能在项目层面讨论连接复用，
不是本模块能单方面决定的（db.py 把「不关连接导致句柄泄漏」记成过踩过的坑）。

## 降级口径

Chroma 导入/初始化失败时 `get_store("chroma")` **抛 `VectorStoreUnavailable`**，
不静默 fallback。需要兜底的调用点用 `get_store_or_degrade()`，它返回
`(store, degrade)`，`degrade` 是给用户看的实话，由调用点负责上报。
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from . import config, db

# 后端选择。主进程按环境变量接线；`get_store(backend)` 显式传参优先。
# 默认向量后端。milvus 是**部署形态**上的选择而不是性能上的：真实库 451 条向量实测
# numpy 8.5ms/次（精确 kNN）、milvus 20.1ms/次（HNSW），numpy 在这个规模上更快也更准。
# 选 milvus 是因为它能跟着数据量长——十万条以上 numpy 的常驻内存与线性扫描会顶不住，
# 而 milvus 的延迟基本不随规模变。**与精确检索的一致性实测 top-5 30/30**，没有召回损失。
# 连不上 Milvus 时不会静默退回：设 PAPERNEST_VECTOR_FALLBACK=1 才降级，且会在
# store.degraded 上留痕并打到 stderr——「以为在用向量库、其实一直是 numpy」比报错危险。
BACKEND = os.environ.get("PAPERNEST_VECTOR_BACKEND", "milvus")

# collection 名字有严格校验：3-512 字符、只能 [a-zA-Z0-9._-]、首尾必须是字母数字。
def collection_name(model: str | None = None) -> str:
    """**每个 embedding 模型一个 collection。**

    为什么不共用一个：Chroma 的 id 是 `kind:paper_id:idx`，**不含 model**——
    同一篇论文在两个模型下会撞成同一条，后写的把先写的覆盖掉。之后拿 A 模型的查询
    向量去比 B 模型的向量，**文本是对的、分数全错**（实测同向向量的 score 应为 1.0
    却返回 0.0），而且 count() 与 index_count() 仍然相等，「两者不等即需 rebuild」
    这个自检判据也一起失效——最隐蔽的一类错误。

    本项目确有多模型共存路径（`embeddings.check_space_compatible` 就是靠
    `WHERE model != ?` 找旧向量来判断能否原地迁移的），所以必须隔离。
    何况不同模型维度不同，本来就不该塞进同一个 HNSW 索引。

    命名要过 Chroma 的校验：3-512 字符、只能 [a-zA-Z0-9._-]、首尾须字母数字。
    模型名可能含中文（如 "Qwen3.7-通用文本向量"），sanitize 后可能彼此相同，
    所以补 8 位哈希保证唯一。
    """
    model = model if model is not None else (config.EMBED_MODEL or "")
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", model).strip("._-")[:64] or "default"
    h = hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]
    return f"pn-{slug}-{h}"
CHROMA_DIR_NAME = "chroma"
CHROMA_BATCH = 2000          # 单次 add 上限，Chroma 官方建议 ≤2000

# `get_store_or_degrade` 之外，允许运维用环境变量强制「Chroma 不可用就退回 numpy」。
# 默认关闭：静默降级会让人以为 ANN 后端在跑，而实际上一直是 numpy。
FALLBACK_ENV = "PAPERNEST_VECTOR_FALLBACK"

# 这三个键由存储层自己写入 metadata，供 where 过滤；用户 meta 不许覆盖它们，
# 否则 `where={"paper_id": 3}` 的语义会随数据而变。
RESERVED_META = ("paper_id", "kind", "idx")

_CMP_OPS = frozenset(("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"))
_SET_OPS = frozenset(("$in", "$nin"))
_BOOL_OPS = frozenset(("$and", "$or"))


class VectorStoreError(Exception):
    """向量存储层的通用错误（where 语法、meta 非法值等）。"""


class VectorStoreUnavailable(VectorStoreError):
    """后端不可用（依赖缺失 / 初始化失败）。消息里必须写清原因，调用点据此上报。"""


class DimensionMismatch(VectorStoreError):
    """维度不一致。**绝不静默截断**——截断出来的余弦是没有意义的数，比报错更危险。"""


# ── schema：只加自己的表和索引，不碰 db.py 的 SCHEMA / MIGRATIONS ──

_SCHEMA = """
-- 派生索引（Chroma）要能被完整重建，就必须连元数据一起放进真相来源。
-- db.py 的 vectors 表没有 meta 列且不归本模块管，所以旁开一张 1:1 的附表；
-- 没有 meta 的向量不写这张表（LEFT JOIN 出来是 NULL，按 {} 处理）。
CREATE TABLE IF NOT EXISTS vector_meta (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  idx INTEGER NOT NULL DEFAULT 0,
  model TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_meta_key
  ON vector_meta(paper_id, kind, idx, model);
-- vectors 上原有的 idx_vectors 是 (paper_id, kind, model)，最左列是 paper_id；
-- 本模块的热路径都不带 paper_id，用不上它。加两条互补索引——只加索引不改表结构，
-- 幂等且无数据变更。
--
-- **这两条索引是量出来的，别删。** 真库（451 条 1024 维向量）实测：
-- 指纹查询 `COUNT(1), MAX(id) WHERE model=?` 在没有覆盖索引时要 3.5ms——因为
-- MAX(id) 逼着 SQLite 回表，而每行都拖着 4KB 的 vec BLOB，一次查询就要读 1.8MB。
-- 这正是 embeddings.py 记下的那个坑（瓶颈是重读 BLOB，不是余弦计算）换了个地方复发。
-- 加上 (model, id) 覆盖索引后降到 0.066ms，快 53 倍；同理正文回取从 3.5ms 到 0.04ms。
CREATE INDEX IF NOT EXISTS idx_vectors_model_id ON vectors(model, id);
CREATE INDEX IF NOT EXISTS idx_vectors_model_pid ON vectors(model, paper_id, kind, idx);
"""

_schema_done: set[str] = set()
_schema_lock = threading.Lock()


def ensure_schema(force: bool = False):
    """幂等建表。按 DB 路径记忆——不加守卫等于每个公开函数入口都多开一条连接
    跑一遍 executescript（db.py 已把这个反模式当踩过的坑记下来了）。"""
    key = str(config.DB_PATH)
    if not force and key in _schema_done and config.DB_PATH.exists():
        return
    with _schema_lock:
        if not force and key in _schema_done and config.DB_PATH.exists():
            return
        with db.conn() as c:
            c.executescript(_SCHEMA)
        _schema_done.add(key)


def _boot():
    db.init_db()
    ensure_schema()


# ── where 过滤：两个后端共用同一套语法与校验，行为才可能一致 ──

def validate_where(where: dict | None):
    """校验 where 语法。**故意与 Chroma 一样严**：顶层只允许一个 key。

    Chroma 1.5.5 对 `{"kind":"chunk","paper_id":2}` 直接抛
    `Expected where to have exactly one operator`。如果 numpy 后端在这里比 Chroma
    宽松，A/B 时同一份调用代码会在一个后端上跑通、在另一个上炸——那这层抽象就白做了。
    多条件请显式写 `{"$and": [{...}, {...}]}`。
    """
    if where is None:
        return
    if not isinstance(where, dict):
        raise VectorStoreError(f"where 必须是 dict，收到 {type(where).__name__}")
    if len(where) != 1:
        raise VectorStoreError(
            f"where 顶层只允许一个条件，收到 {len(where)} 个：{sorted(where)}；"
            f"多条件请写 {{'$and': [...]}}")
    key, val = next(iter(where.items()))
    if key in _BOOL_OPS:
        if not isinstance(val, (list, tuple)) or not val:
            raise VectorStoreError(f"{key} 的值必须是非空 list")
        for sub in val:
            validate_where(sub)
        return
    if key.startswith("$"):
        raise VectorStoreError(f"不支持的顶层操作符 {key!r}")
    if isinstance(val, dict):
        if len(val) != 1:
            raise VectorStoreError(f"{key!r} 的条件只允许一个操作符，收到 {sorted(val)}")
        op, operand = next(iter(val.items()))
        if op in _SET_OPS:
            if not isinstance(operand, (list, tuple)) or not operand:
                raise VectorStoreError(f"{op} 的值必须是非空 list")
        elif op not in _CMP_OPS:
            raise VectorStoreError(f"不支持的操作符 {op!r}")
    elif not isinstance(val, (str, int, float, bool)):
        raise VectorStoreError(f"where 的值只能是标量，{key!r} 收到 {type(val).__name__}")


def match_where(meta: dict, where: dict | None) -> bool:
    """numpy 后端的 where 求值。语义对齐 Chroma：缺失的键一律不匹配（不当成 NULL 参与比较）。"""
    if not where:
        return True
    key, val = next(iter(where.items()))
    if key == "$and":
        return all(match_where(meta, s) for s in val)
    if key == "$or":
        return any(match_where(meta, s) for s in val)
    if key not in meta:
        return False
    cur = meta[key]
    if not isinstance(val, dict):
        return bool(cur == val)
    op, operand = next(iter(val.items()))
    try:
        if op == "$eq":
            return bool(cur == operand)
        if op == "$ne":
            return bool(cur != operand)
        if op == "$in":
            return cur in operand
        if op == "$nin":
            return cur not in operand
        if op == "$gt":
            return cur > operand
        if op == "$gte":
            return cur >= operand
        if op == "$lt":
            return cur < operand
        return cur <= operand
    except TypeError:
        # 类型不可比（如字符串 > 数字）。Chroma 在这种行上也是不匹配，保持一致。
        return False


def _combine_where(kind: str | None, where: dict | None) -> dict | None:
    """把 kind 过滤与用户 where 合成一个合法条件（顶层仍只有一个 key）。"""
    clauses = [c for c in ({"kind": kind} if kind is not None else None, where) if c]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ── SQLite 真相层：两个后端共用 ──

def _model() -> str:
    """当前嵌入模型。向量按 model 分区——混着不同模型的向量算余弦是没有意义的。

    运行期读 `config.EMBED_MODEL`（不是 import 期），换模型或测试改配置立刻生效。
    """
    return config.EMBED_MODEL


def _norm_item(it: dict) -> tuple[int, str, int, str, np.ndarray, dict]:
    """把 items 的一项规整成 (paper_id, kind, idx, text, vec, meta)，非法输入立刻报错。"""
    try:
        paper_id = int(it["paper_id"])
        kind = str(it["kind"])
    except (KeyError, TypeError, ValueError) as e:
        raise VectorStoreError(f"item 缺少 paper_id/kind 或类型不对：{e}") from e
    if not kind or ":" in kind:
        # id 编码是 f"{kind}:{paper_id}:{idx}"，kind 里带冒号就反解不回来了
        raise VectorStoreError(f"kind 不能为空且不能含冒号，收到 {kind!r}")
    idx = int(it.get("idx") or 0)
    text = it.get("text") or ""
    vec = np.asarray(it.get("vec"), dtype=np.float32).ravel()
    if vec.size == 0:
        raise VectorStoreError(f"({paper_id},{kind},{idx}) 的 vec 为空")
    if not np.all(np.isfinite(vec)):
        raise VectorStoreError(f"({paper_id},{kind},{idx}) 的 vec 含 NaN/Inf")
    meta = dict(it.get("meta") or {})
    bad = [k for k in meta if k in RESERVED_META]
    if bad:
        raise VectorStoreError(
            f"meta 不能覆盖存储层保留键 {bad}（它们由 paper_id/kind/idx 直接决定）")
    bad_val = [k for k, v in meta.items() if not isinstance(v, (str, int, float, bool))]
    if bad_val:
        raise VectorStoreError(
            f"meta 的值只能是 str/int/float/bool（Chroma metadata 的限制），"
            f"非法键：{bad_val}")
    return paper_id, kind, idx, text, vec, meta


def _store_dim(c, model: str) -> int | None:
    """库内该模型的向量维度；没有向量则 None。"""
    row = c.execute("SELECT LENGTH(vec) n FROM vectors WHERE model=? LIMIT 1",
                    (model,)).fetchone()
    return (row["n"] // 4) if row else None


def _write_items(items: list[dict]) -> list[tuple]:
    """把 items 写进真相来源，返回规整后的行（先删后插＝覆盖语义，同一 id 写两次不重复）。

    返回规整结果而不是条数，是为了让 ChromaStore 不必把 `_norm_item` 再跑一遍——
    校验只做一次，两个后端看到的也就必然是同一份数据。
    """
    if not items:
        return []
    model = _model()
    prepared = [_norm_item(it) for it in items]
    dims = {p[4].size for p in prepared}
    if len(dims) > 1:
        raise DimensionMismatch(f"同一批 items 里混了多种维度：{sorted(dims)}")
    dim = dims.pop()
    _boot()
    with db.conn() as c:
        exist = _store_dim(c, model)
        if exist is not None and exist != dim:
            raise DimensionMismatch(
                f"向量维度不一致：库内 {exist} 维、本次写入 {dim} 维（model={model!r}）；"
                f"换嵌入模型请先重建索引，不要混写")
        for paper_id, kind, idx, text, vec, meta in prepared:
            c.execute("DELETE FROM vectors WHERE paper_id=? AND kind=? AND idx=? AND model=?",
                      (paper_id, kind, idx, model))
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (paper_id, kind, idx, text, model, vec.tobytes()))
            c.execute("DELETE FROM vector_meta WHERE paper_id=? AND kind=? AND idx=? "
                      "AND model=?", (paper_id, kind, idx, model))
            if meta:
                c.execute("INSERT INTO vector_meta(paper_id,kind,idx,model,meta_json) "
                          "VALUES(?,?,?,?,?)",
                          (paper_id, kind, idx, model,
                           json.dumps(meta, ensure_ascii=False, sort_keys=True)))
    _invalidate()
    return prepared


def _delete_paper_rows(paper_id: int) -> int:
    """删掉该论文的全部向量（**所有模型**——「这篇没了」不该只删当前模型那一份）。"""
    _boot()
    with db.conn() as c:
        n = c.execute("DELETE FROM vectors WHERE paper_id=?", (paper_id,)).rowcount
        c.execute("DELETE FROM vector_meta WHERE paper_id=?", (paper_id,))
    _invalidate()
    return int(n or 0)


def _count_rows(kind: str | None) -> int:
    _boot()
    model = _model()
    with db.conn() as c:
        if kind is None:
            r = c.execute("SELECT COUNT(1) n FROM vectors WHERE model=?", (model,)).fetchone()
        else:
            r = c.execute("SELECT COUNT(1) n FROM vectors WHERE model=? AND kind=?",
                          (model, kind)).fetchone()
    return int(r["n"])


def _iter_truth(model: str, c=None):
    """按全序读出该模型的全部向量（含 meta）。rebuild 与 numpy 装载共用。"""
    if c is None:
        with db.conn() as own:
            return _iter_truth(model, own)
    return c.execute(
        """SELECT v.id, v.paper_id, v.kind, v.idx, v.text, v.vec, m.meta_json
             FROM vectors v
             LEFT JOIN vector_meta m
               ON m.paper_id=v.paper_id AND m.kind=v.kind
              AND m.idx=v.idx AND m.model=v.model
            WHERE v.model=?
            ORDER BY v.paper_id, v.kind, v.idx, v.id""", (model,)).fetchall()


def _texts_by_rowid(c, rowids: list[int]) -> dict:
    """numpy 后端的正文回取：走 INTEGER PRIMARY KEY，是 SQLite 最快的一条路。

    常驻矩阵里存了行号就是为了这一步——只回取选中的 k 行，不把全库正文塞进内存。
    """
    if not rowids:
        return {}
    ph = ",".join("?" * len(rowids))
    return {int(r["id"]): (r["text"] or "")
            for r in c.execute(f"SELECT id, text FROM vectors WHERE id IN ({ph})", rowids)}


def _texts_for(keys: list[tuple[int, str, int]], model: str, c=None) -> dict:
    """Chroma 后端的正文回取：它只能从 id 反解出 (paper_id,kind,idx)，拿不到行号。

    text 的权威永远在 SQLite——派生索引里也存了一份 documents，但只用于人工排查。
    """
    if not keys:
        return {}
    if c is None:
        with db.conn() as own:
            return _texts_for(keys, model, own)
    pids = sorted({k[0] for k in keys})
    ph = ",".join("?" * len(pids))
    rows = c.execute(
        f"SELECT paper_id,kind,idx,text FROM vectors WHERE model=? AND paper_id IN ({ph})",
        (model, *pids)).fetchall()
    want = set(keys)
    out = {}
    for r in rows:
        k = (int(r["paper_id"]), r["kind"], int(r["idx"]))
        if k in want:
            out[k] = r["text"] or ""
    return out


# ── numpy 后端的常驻矩阵缓存（模块级：同一份 DB 只装载一次，多个 store 实例共享） ──

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple, dict] = {}


def _invalidate():
    """向量变更后作废缓存。指纹检查已能兜住，这是双保险。

    连带作废 embeddings 里那份 paper 级矩阵：本模块写的是同一张 vectors 表，
    不通知它就会出现「这里删干净了、那里还能搜到」的幽灵结果。
    这一步吞异常是安全的（不是静默降级）：embeddings 自己的 (条数, max(id)) 指纹
    已经能兜住同一份变更，这个调用只是把失效提前，失败最多是晚一次查询才生效。
    """
    with _CACHE_LOCK:
        _CACHE.clear()
    try:
        from . import embeddings
        embeddings.invalidate_cache()
    except Exception:
        pass


def _fingerprint(c, model: str) -> tuple:
    """缓存有效性指纹 (条数, max(id))，vectors 与 vector_meta 各一份。

    覆盖本模块所有真实变更路径：upsert 是先删后插（autoincrement 让 max(id) 增长）、
    delete_paper 改条数、只改 meta 也会让 vector_meta 的 max(id) 增长。
    两条聚合查询都走覆盖索引（idx_vectors_model_id / idx_vector_meta_key），真库实测
    0.066ms——**前提是索引在**，否则 MAX(id) 会回表，每行拖着 4KB BLOB 一路读过去。
    """
    a = c.execute("SELECT COUNT(1) n, COALESCE(MAX(id),0) m FROM vectors WHERE model=?",
                  (model,)).fetchone()
    b = c.execute("SELECT COUNT(1) n, COALESCE(MAX(id),0) m FROM vector_meta WHERE model=?",
                  (model,)).fetchone()
    return (model, a["n"], a["m"], b["n"], b["m"])


def _snapshot(c=None) -> dict:
    """常驻内存的归一化矩阵 + 按 kind 的行号索引 + meta。

    只缓存「选 top-k 之前用得到」的东西：矩阵要算分、meta 要做 where 过滤，
    两者都得在选出 top-k 之前扫全表；而 **text 只有选中的 k 行才需要**，
    按行号回查 SQLite 是 k 次索引命中，没必要把全库正文常驻内存（本项目 784 条
    chunk 的正文就有 MB 级）。

    传入 `c` 是为了让一次 search 只开一条连接：真库实测 `sqlite3.connect` 本身
    就要 0.55ms，比指纹查询（0.066ms）和正文回取（0.04ms）加起来还贵，
    开两条就等于把查询延迟翻倍。
    """
    if c is None:
        with db.conn() as own:
            return _snapshot(own)
    model = _model()
    ck = (str(config.DB_PATH), model)
    key = _fingerprint(c, model)
    with _CACHE_LOCK:
        hit = _CACHE.get(ck)
        if hit is not None and hit["key"] == key:
            return hit
    rows = _iter_truth(model, c)
    if not rows:
        snap = {"key": key, "M": None, "keys": [], "rowids": [], "metas": [],
                "by_kind": {}, "dim": None}
    else:
        sizes = {len(r["vec"]) for r in rows}
        if len(sizes) > 1:
            # 库里混了多种维度：reshape 会「成功」但每行都错位——必须报错。
            raise DimensionMismatch(
                f"库内 model={model!r} 的向量存在多种维度 "
                f"{sorted(s // 4 for s in sizes)}，请先重建索引")
        dim = sizes.pop() // 4
        raw = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32)
        M = raw.reshape(len(rows), dim).astype(np.float32, copy=True)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)   # 查询期只剩一次矩阵乘
        keys = [(int(r["paper_id"]), r["kind"], int(r["idx"])) for r in rows]
        by_kind: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            by_kind.setdefault(k[1], []).append(i)
        snap = {
            "key": key, "M": M, "keys": keys,
            "rowids": [int(r["id"]) for r in rows],
            "metas": [json.loads(r["meta_json"]) if r["meta_json"] else {} for r in rows],
            "by_kind": {k: np.asarray(v, dtype=np.int64) for k, v in by_kind.items()},
            "dim": dim,
        }
    with _CACHE_LOCK:
        _CACHE[ck] = snap
    return snap


def _prep_query(qvec, dim: int | None) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float32).ravel()
    if q.size == 0:
        raise VectorStoreError("查询向量为空")
    if dim is not None and q.size != dim:
        raise DimensionMismatch(
            f"向量维度不一致：库内 {dim} 维、查询 {q.size} 维；"
            f"请先重建索引，不做静默截断（截断出来的余弦是没有意义的数）")
    return q / (float(np.linalg.norm(q)) + 1e-9)


def _rank(cand: list[int], score_of, key_of, top_k: int) -> list[int]:
    """全序排序：分数降序，并列按 (paper_id, kind, idx) 决胜。

    排序键不全序，跨进程结果就会漂移——本项目踩过。
    """
    return sorted(cand, key=lambda i: (-score_of(i), key_of(i)))[:top_k]


# ── Protocol ──

@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, items: list[dict]) -> int: ...
    def search(self, qvec, top_k: int, kind: str | None = None,
               where: dict | None = None) -> list[dict]: ...
    def delete_paper(self, paper_id: int) -> int: ...
    def count(self, kind: str | None = None) -> int: ...
    def rebuild(self, progress=None) -> dict: ...


class NumpyStore:
    """默认后端：SQLite 是索引本身，内存里只多一份归一化矩阵。

    没有「派生索引」的概念——矩阵就是 vectors 表的一个纯函数，按指纹自动失效，
    所以 `rebuild()` 只是清缓存，天然不可能与真相不一致。
    """

    name = "numpy"
    degraded: str | None = None

    def __init__(self):
        _boot()

    def upsert(self, items: list[dict]) -> int:
        return len(_write_items(items))

    def search(self, qvec, top_k: int = 8, kind: str | None = None,
               where: dict | None = None) -> list[dict]:
        validate_where(where)
        with db.conn() as c:      # 一次查询只开一条连接（连接本身 0.55ms，比查询还贵）
            return self._search(c, qvec, top_k, kind, where)

    def _search(self, c, qvec, top_k, kind, where) -> list[dict]:
        snap = _snapshot(c)
        q = _prep_query(qvec, snap["dim"])
        if snap["M"] is None or top_k <= 0:
            return []
        sel = None
        if kind is not None:
            sel = snap["by_kind"].get(kind)
            if sel is None:
                return []
        if where:
            base = sel if sel is not None else np.arange(len(snap["keys"]), dtype=np.int64)
            metas, keys = snap["metas"], snap["keys"]
            picked = [int(i) for i in base
                      if match_where(_full_meta(keys[i], metas[i]), where)]
            if not picked:
                return []
            sel = np.asarray(picked, dtype=np.int64)
        M = snap["M"] if sel is None else snap["M"][sel]
        scores = M @ q
        k = min(int(top_k), int(scores.size))
        if k <= 0:
            return []
        if k < scores.size:
            part = np.argpartition(-scores, k - 1)[:k]
            # argpartition 在并列处的取舍是任意的：先拿到第 k 名的分数，再把**所有**
            # 并列的行捞回来一起全序排。否则「谁进 top-k」会随 numpy 版本漂。
            thresh = float(scores[part].min())
            cand = np.flatnonzero(scores >= thresh).tolist()
        else:
            cand = list(range(int(scores.size)))
        gid = (lambda i: int(sel[i])) if sel is not None else (lambda i: i)
        order = _rank(cand, lambda i: float(scores[i]), lambda i: snap["keys"][gid(i)], k)
        rowids = [snap["rowids"][gid(i)] for i in order]
        texts = _texts_by_rowid(c, rowids)
        return [{"paper_id": snap["keys"][gid(i)][0], "kind": snap["keys"][gid(i)][1],
                 "idx": snap["keys"][gid(i)][2], "score": float(scores[i]),
                 "text": texts.get(rid, "")}
                for i, rid in zip(order, rowids)]

    def delete_paper(self, paper_id: int) -> int:
        return _delete_paper_rows(int(paper_id))

    def count(self, kind: str | None = None) -> int:
        return _count_rows(kind)

    def rebuild(self, progress: Callable[[int, int], Any] | None = None) -> dict:
        """清缓存并重新装载。numpy 后端没有第二份数据，所以这一步不可能失败在一致性上。"""
        t0 = time.perf_counter()
        _invalidate()
        snap = _snapshot()
        n = len(snap["keys"])
        if progress:
            progress(n, n)
        return {"backend": self.name, "indexed": n,
                "elapsed_s": round(time.perf_counter() - t0, 4), "errors": []}


def _full_meta(key: tuple[int, str, int], meta: dict) -> dict:
    """对外可见的 metadata = 用户 meta + 存储层保留键。两个后端存的是同一份。"""
    return {**meta, "paper_id": key[0], "kind": key[1], "idx": key[2]}


class ChromaStore:
    """可选后端：Chroma PersistentClient + HNSW。**派生索引，不是真相来源。**

    写路径永远是「先落 SQLite，再同步 Chroma」；读路径的 text 也从 SQLite 回取。
    Chroma 里查到、SQLite 里已经没有的行（索引落后于真相）会被丢弃并计入
    `last_stale_hits` —— 丢弃是对的（那行已经删了），但必须留痕，不能静默。
    `index_count()` 与 `count()` 对不上就说明该 `rebuild()` 了。
    """

    name = "chroma"
    degraded: str | None = None

    def __init__(self):
        _boot()
        self.last_stale_hits = 0
        self._client = _chroma_client()
        self._col_model = config.EMBED_MODEL
        self._col = self._collection()

    def _collection(self):
        name = collection_name(self._col_model)
        try:
            return self._client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine"})  # 默认是 L2，必须显式指定
        except Exception as e:
            raise VectorStoreUnavailable(
                f"Chroma collection {name!r} 打开失败：{type(e).__name__}: {e}"
            ) from e

    @property
    def col(self):
        """按当前 EMBED_MODEL 解析 collection。

        模型可以在进程存续期间被改（测试会改、`cli.py embed` 迁移也会改），
        缓存住建集合时的模型名，一旦不同就换一个 collection——否则会拿着
        旧模型的索引去回答新模型的查询。
        """
        if config.EMBED_MODEL != self._col_model:
            self._col_model = config.EMBED_MODEL
            self._col = self._collection()
        return self._col

    @staticmethod
    def _id(paper_id: int, kind: str, idx: int) -> str:
        return f"{kind}:{paper_id}:{idx}"

    @staticmethod
    def _parse_id(vid: str) -> tuple[int, str, int] | None:
        parts = vid.split(":")
        if len(parts) != 3:
            return None
        try:
            return (int(parts[1]), parts[0], int(parts[2]))
        except ValueError:
            return None

    def upsert(self, items: list[dict]) -> int:
        prepared = _write_items(items)   # 真相先落地；Chroma 失败了也能 rebuild 回来
        if not prepared:
            return 0
        ids, embs, metas, docs = [], [], [], []
        for paper_id, kind, idx, text, vec, meta in prepared:
            ids.append(self._id(paper_id, kind, idx))
            embs.append(vec.tolist())    # 必须显式传 embeddings，否则 Chroma 会去下 all-MiniLM
            metas.append(_full_meta((paper_id, kind, idx), meta))
            docs.append(text or "")
        for i in range(0, len(ids), CHROMA_BATCH):
            sl = slice(i, i + CHROMA_BATCH)
            self.col.upsert(ids=ids[sl], embeddings=embs[sl],
                             metadatas=metas[sl], documents=docs[sl])
        return len(prepared)

    def search(self, qvec, top_k: int = 8, kind: str | None = None,
               where: dict | None = None) -> list[dict]:
        validate_where(where)
        model = _model()
        with db.conn() as c:             # 同样只开一条连接：维度校验与正文回取共用
            return self._search(c, model, qvec, top_k, kind, where)

    def _search(self, c, model, qvec, top_k, kind, where) -> list[dict]:
        q = _prep_query(qvec, _store_dim(c, model))   # 维度以 SQLite 为准，两后端同一句错
        self.last_stale_hits = 0
        if top_k <= 0:
            return []
        total = self.col.count()
        if not total:
            return []
        # 多取一些再全序排：HNSW 只保证近似 top-k，边界上并列的行谁进谁不进是不定的。
        # 多取几十条几乎不花钱（HNSW 次线性），却能让并列决胜的行为贴近精确检索。
        n_results = min(total, max(int(top_k) * 3 + 10, int(top_k)))
        res = self.col.query(query_embeddings=[q.tolist()], n_results=n_results,
                              where=_combine_where(kind, where),
                              include=["distances"])
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        pairs: list[tuple[tuple[int, str, int], float]] = []
        for vid, d in zip(ids, dists):
            key = self._parse_id(vid)
            if key is None:
                self.last_stale_hits += 1
                continue
            pairs.append((key, 1.0 - float(d)))   # cosine space 返回的是距离，不是相似度
        texts = _texts_for([p[0] for p in pairs], model, c)
        live = []
        for key, score in pairs:
            if key not in texts:
                self.last_stale_hits += 1       # 索引里还有、真相里已经没有 → 丢弃并留痕
                continue
            live.append((key, score))
        order = _rank(list(range(len(live))), lambda i: live[i][1],
                      lambda i: live[i][0], int(top_k))
        return [{"paper_id": live[i][0][0], "kind": live[i][0][1], "idx": live[i][0][2],
                 "score": float(live[i][1]), "text": texts[live[i][0]]} for i in order]

    def delete_paper(self, paper_id: int) -> int:
        paper_id = int(paper_id)
        n = _delete_paper_rows(paper_id)
        try:
            self.col.delete(where={"paper_id": paper_id})
        except Exception as e:
            raise VectorStoreUnavailable(
                f"SQLite 已删除 paper_id={paper_id}，但 Chroma 索引删除失败："
                f"{type(e).__name__}: {e}；请跑 rebuild() 使索引与真相重新一致") from e
        return n

    def count(self, kind: str | None = None) -> int:
        """按**真相来源**计数。派生索引自己的条数看 `index_count()`，两者不等即需 rebuild。"""
        return _count_rows(kind)

    def index_count(self) -> int:
        return int(self.col.count())

    def rebuild(self, progress: Callable[[int, int], Any] | None = None) -> dict:
        """丢掉整个 collection 再从 SQLite 全量重放。派生索引可以随时丢，这就是它的价值。"""
        t0 = time.perf_counter()
        errors: list[str] = []
        model = _model()
        rows = _iter_truth(model)
        try:
            self._client.delete_collection(collection_name(self._col_model))
        except Exception as e:
            errors.append(f"删除旧 collection 失败（按空集合继续）：{type(e).__name__}: {e}")
        self._col = self._collection()
        total, done = len(rows), 0
        for i in range(0, total, CHROMA_BATCH):
            batch = rows[i:i + CHROMA_BATCH]
            ids, embs, metas, docs = [], [], [], []
            for r in batch:
                key = (int(r["paper_id"]), r["kind"], int(r["idx"]))
                ids.append(self._id(key[0], key[1], key[2]))
                embs.append(np.frombuffer(r["vec"], dtype=np.float32).tolist())
                metas.append(_full_meta(
                    key, json.loads(r["meta_json"]) if r["meta_json"] else {}))
                docs.append(r["text"] or "")
            try:
                self.col.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
                done += len(ids)
            except Exception as e:
                errors.append(f"第 {i // CHROMA_BATCH + 1} 批（{len(ids)} 条）失败："
                              f"{type(e).__name__}: {e}")
            if progress:
                progress(done, total)
        return {"backend": self.name, "indexed": done,
                "elapsed_s": round(time.perf_counter() - t0, 4), "errors": errors}


# ── Chroma client：按目录缓存（PersistentClient 启动约 262ms，别每次 get_store 都付） ──

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[str, Any] = {}


def _chroma_client():
    path = str(config.DATA_DIR / CHROMA_DIR_NAME)
    with _CLIENT_LOCK:
        cli = _CLIENTS.get(path)
    if cli is not None:
        return cli
    try:
        import chromadb
    except Exception as e:      # ImportError，也可能是本机二进制不兼容
        raise VectorStoreUnavailable(
            f"chromadb 导入失败（{type(e).__name__}: {e}）；"
            f"`pip install chromadb` 后重试，或用 numpy 后端") from e
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cli = chromadb.PersistentClient(path=path)
    except Exception as e:
        raise VectorStoreUnavailable(
            f"Chroma PersistentClient 初始化失败（path={path}）："
            f"{type(e).__name__}: {e}") from e
    with _CLIENT_LOCK:
        _CLIENTS[path] = cli
    return cli


def reset_clients():
    """丢弃缓存的 Chroma client 与 numpy 矩阵。测试换临时目录后调用。

    **调用后此前拿到的 ChromaStore 实例即作废**（底层 System 已 stop，再用会炸在
    chromadb 内部），必须重新 `get_store()`。这是「reset」应有的语义：不真正放掉
    句柄的 reset 只是把泄漏藏起来。NumpyStore 不受影响。

    只从 `_CLIENTS` 里删引用是不够的：chromadb 在 `SharedSystemClient` 上还挂着
    一份**进程级**的 System 缓存，每个 System 拖着自己的 sqlite 连接和线程池。
    而它自带的 `clear_system_cache()` 只是把那个 dict 置空、**不 stop 任何 System**
    （chromadb 1.5.5 实现如此），句柄照样不放。所以这里先逐个 `system.stop()` 再清缓存。
    （试过额外加一次 `gc.collect()`：对释放句柄没有增益，反而让本模块测试从 15.0s
    涨到 21.2s，所以不加。）

    不这么做的后果是实打实的：(a) 反复换目录建 client 会一路堆积，
    (b) Windows 上临时目录删不掉——本机 %TEMP% 已攒下 3400+ 个测试残留目录、1.4GB。
    整段吞异常是安全的：这只是回收，不是正确性的一环，chromadb 换版本改了内部结构
    也不该让 reset 本身炸。
    """
    mod = sys.modules.get("papernest.milvusstore")
    if mod is not None:          # 只在已加载时重置，避免为了 reset 反而把它 import 进来
        try:
            mod.reset_clients()
        except Exception:        # 回收失败不该让 reset 本身炸
            pass
    with _CLIENT_LOCK:
        _CLIENTS.clear()
    with _CACHE_LOCK:
        _CACHE.clear()
    _schema_done.clear()
    try:
        from chromadb.api.client import SharedSystemClient
        for system in list(SharedSystemClient._identifier_to_system.values()):
            try:
                system.stop()
            except Exception:
                pass
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


# ── 工厂 ──

def _milvus_store_cls():
    """惰性取 MilvusStore。

    **不在模块顶层 import**：`milvusstore` 会 import pymilvus，而 pymilvus 未安装
    是常态（它只是可选后端）。顶层 import 会让整个 vectorstore 模块在没装 pymilvus
    的机器上直接崩掉——连默认的 numpy 后端都用不了。惰性到「真的要 milvus」那一刻。
    """
    from .milvusstore import MilvusStore
    return MilvusStore


_BACKENDS: dict[str, Any] = {"numpy": NumpyStore, "chroma": ChromaStore,
                             "milvus": _milvus_store_cls}


def get_store(backend: str | None = None) -> VectorStore:
    """按名字取后端。Chroma 不可用时**抛 `VectorStoreUnavailable`，不静默降级**。

    静默 fallback 的代价是：用户以为在跑 ANN 后端、A/B 报告里写的是 Chroma，
    实际一直是 numpy——这种「跑通了但结论是假的」比直接报错危险得多。
    要兜底请用 `get_store_or_degrade()`，它把降级原因交回给调用点去如实上报；
    或设 `PAPERNEST_VECTOR_FALLBACK=1` 让本函数自动退回 numpy（仍会在
    `store.degraded` 上留痕）。
    """
    name = (backend or BACKEND or "numpy").strip().lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        raise VectorStoreError(
            f"未知向量后端 {name!r}，可选：{sorted(_BACKENDS)}")
    try:
        if name == "milvus":
            # 惰性解析：pymilvus 没装时这里抛 ImportError，转成 Unavailable，
            # 让下面统一的降级/上报逻辑接住，而不是把裸 ImportError 抛给调用方。
            try:
                cls = cls()
            except ImportError as e:
                raise VectorStoreUnavailable(
                    f"milvus 后端不可用：pymilvus 未安装（{e}）。装：pip install pymilvus。"
                    f"注意 Milvus Lite 不支持 Windows，需连服务端："
                    f"docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d"
                ) from e
        store = cls()
    except VectorStoreUnavailable:
        if name != "numpy" and os.environ.get(FALLBACK_ENV, "") in ("1", "true", "True"):
            store = NumpyStore()
            store.degraded = (f"{name} 后端不可用，已按 {FALLBACK_ENV} 退回 numpy")
            print(f"[vectorstore] 降级：{store.degraded}", file=sys.stderr)
            return store
        raise
    store.degraded = None
    return store


def get_store_or_degrade(backend: str | None = None) -> tuple[VectorStore, str | None]:
    """要 Chroma、拿不到就退 numpy，并把**原因原样交回**给调用点去上报。

    返回 `(store, degrade)`；`degrade` 为 None 表示拿到的就是要的后端。
    这是本项目「降级如实标注而非静默吞掉」在存储层的落点。
    """
    name = (backend or BACKEND or "numpy").strip().lower()
    try:
        return get_store(name), None
    except VectorStoreUnavailable as e:
        store = NumpyStore()
        reason = f"{name} 后端不可用（{e}），已降级 numpy"
        store.degraded = reason
        return store, reason
