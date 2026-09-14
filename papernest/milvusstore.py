"""Milvus 向量后端：`vectorstore` 的第三个可插拔实现，服务端 ANN 索引。

## 为什么单开一个文件而不是塞进 vectorstore.py

numpy / Chroma 两个后端都是「进程内的一份数据」，而 Milvus 是**外部服务**：连接管理与
缓存、collection schema、索引类型与 metric、建完索引还要显式 load、一致性级别、
服务端表达式过滤、可用性探测与排查指引——这些都是服务端专属代码，与本地后端零共用。
vectorstore.py 已经 40KB，再塞进去只会让「本地后端」的逻辑被服务端细节淹没。

真相来源、写读路径、降级口径完全沿用 vectorstore 的约定（下面逐条说明），
本模块只提供 `MilvusStore` 这一个新实现。

## 三条与 Chroma 后端相同的硬约束（不是实现细节，是正确性的前提）

1. **SQLite 的 `vectors` 表是唯一真相来源，Milvus 只是派生索引。**
   写路径先 `vectorstore._write_items()` 落 SQLite 再同步 Milvus；读路径的 `text`
   也从 SQLite 回取。Milvus 里查到、SQLite 里已经删掉的行会被**丢弃并计入
   `last_stale_hits`**——丢弃是对的（那行确实没了），但必须留痕，不能静默。
   `rebuild()` 永远能从 SQLite 全量重放，所以整个 Milvus 实例随时可以扔掉。
2. **每个 embedding 模型一个 collection。** 这是 Chroma 后端刚修过的真实缺陷：
   id 是 `kind:paper_id:idx`、**不含 model**，两个模型的同一篇论文会撞成一条、后写的
   覆盖先写的；之后拿 A 模型的查询向量去比 B 模型的向量，**文本是对的、分数全错**
   （同向向量的 score 应为 1.0，实测返回 0.0），而且 `count()` 与 `index_count()`
   仍然相等，「两者不等即需 rebuild」这个自检判据一起失效。Milvus 这边还多一层
   硬约束：**维度是建 collection 时固定的**，不同维度的模型本来就塞不进同一个索引。
3. **分数是余弦相似度（越大越相关），排序必须全序。** 见下一节。

## metric 的方向：COSINE 返回的是相似度，不要再做 `1 - d`

Milvus 的搜索结果字段叫 `distance`，但**它的含义随 metric 变**：

| metric | `distance` 字段的含义 | 越大越相关？ |
|--------|----------------------|-------------|
| COSINE | 余弦相似度 [-1, 1]    | 是          |
| IP     | 内积                  | 是          |
| L2     | 欧氏距离的平方         | 否          |

依据不是猜的：pymilvus 3.0.1 自己的 `client/iterator/search_iterator.py`
里有 `metrics_positive_related()`，对 L2/JACCARD/HAMMING/TANIMOTO 返回 True、
对 **IP/COSINE/BM25 返回 False**——即后者的取值与「距离」正相关性相反，是相似度。

Chroma 后端正相反（cosine space 返回的是 `1 - 相似度`，必须转换），差点因此返回反的
排序。所以本模块**不写死一行 `1 - d`**，而是把方向做成 `_METRIC_KIND` 这张表，
`_to_score()` 查表转换，metric 不在表里直接报错——搞反的代价是排序整个倒过来，
而且分数看着还挺正常，测不出来。

## Milvus 最常见的坑：建了 collection 不等于能搜

`MilvusClient.create_collection(..., schema=...)` 在**不传 `index_params` 时既不建索引
也不 load**（pymilvus 3.0.1 `_create_collection_with_schema` 的实现就是
`if index_params: create_index(); load_collection()`）。此时 `insert` 会成功、
`search` 会报 `collection not loaded`。而且**已存在的 collection 在进程重启、
服务重启、或被别的进程 release 之后同样是未 load 状态**。

所以本模块 `_open()` 里：新建走 `index_params`（顺带建索引 + load），已存在的**每次
开集合都补一次 `load_collection()`**（幂等，已 load 时是廉价的空操作）。

## 一致性级别默认 Strong

Milvus 默认是 Bounded 有界陈旧：刚写进去的向量可能几秒内搜不到。本项目的
`rebuild()` 之后紧接着自检、导入完立刻检索都是常规流程，Bounded 会让人看到
「写进去了但搜不到」并以为是 bug。所以默认 `Strong`（用 `PAPERNEST_MILVUS_CONSISTENCY`
可改）。代价是查询要等同步点、延迟更高——这是拿延迟换「不出现无法解释的空结果」。

## 不可用时抛错，不静默降级

连不上就抛 `VectorStoreUnavailable`，消息里写清**四个排查方向**（服务没起来 /
地址不对 / 鉴权库名不对 / collection 没 load），由 `vectorstore.get_store_or_degrade()`
决定降不降级并把原因如实交回调用点。**模块 import 绝不 import pymilvus、更不连服务**：
开发机上 Milvus 没起来时，整个项目仍必须能 import。
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

from . import config, db
from .vectorstore import (
    RESERVED_META,
    DimensionMismatch,
    VectorStoreError,
    VectorStoreUnavailable,
    _boot,
    _count_rows,
    _delete_paper_rows,
    _full_meta,
    _iter_truth,
    _model,
    _prep_query,
    _rank,
    _store_dim,
    _texts_for,
    _write_items,
    validate_where,
)

# ── 环境变量：连接地址与索引类型都不写死 ──

URI_ENV = "PAPERNEST_MILVUS_URI"
TOKEN_ENV = "PAPERNEST_MILVUS_TOKEN"       # Zilliz Cloud / 开了鉴权的自建实例
DB_ENV = "PAPERNEST_MILVUS_DB"             # Milvus 的多 database，默认库留空即可
INDEX_ENV = "PAPERNEST_MILVUS_INDEX_TYPE"  # AUTOINDEX / HNSW / IVF_FLAT / FLAT ...
CONSISTENCY_ENV = "PAPERNEST_MILVUS_CONSISTENCY"

DEFAULT_URI = "http://localhost:19530"
DEFAULT_INDEX_TYPE = "AUTOINDEX"           # 让 Milvus 按数据规模自选，省得我们瞎调参
# 一致性级别。默认 Bounded 的**理由**是机制性的，不是某组具体数字：
# Strong 会等写入时间戳完全同步才开始搜，对「写一次、读很多次」的文献检索是错误的
# 默认值——一次入库之后要在此后每一次查询上都付这笔同步代价；而 Bounded 允许读落后
# 写几秒，对本场景无影响（新入库的论文晚几秒才能被检索到，用户感知不到）。
#
# ⚠️ 这里原来写着「实测 Strong p50 380.3ms / Bounded 8.0ms，差 47 倍」，
# **那组数字已删除**：README 旧版记的是 399.9 / 16.1（25 倍），两处互相矛盾，
# 而仓库里没有任何 bench 脚本能裁决——tests/ 下没有一行 Milvus 计时代码。
# 一个既无法复现、又与文档打架的「实测值」比没有数字更糟：它会被当成依据引用。
# 要重新给出数字，得先有一个能在真实 Milvus 上跑、并把环境（版本 / 数据量 / 索引类型 /
# 网络）一起记下来的 bench，再同步更新 README。在那之前不要引用任何一组。
#
# 需要写后立刻可读（测试、rebuild 后立即校验）时按调用点传 Strong，不要改全局默认。
DEFAULT_CONSISTENCY = "Bounded"

# metric 不给改。允许运维改 metric 等于允许把「相似度」和「距离」混着用，
# 而分数搞反了看不出来（排序倒过来但数值范围仍然正常）。要换请改代码并同步改测试。
METRIC = "COSINE"

# 字段名。id 用 VARCHAR 主键，与 Chroma 后端同一套编码，便于两个后端逐条对照。
PK_FIELD = "id"
VEC_FIELD = "vector"
PK_MAX_LEN = 512
KIND_MAX_LEN = 128

# 单次 insert/upsert 的行数。1024 维 float32 一行 4KB，1000 行约 4MB，
# 离 gRPC 默认 64MB 上限还很远；再大就要考虑消息体超限。
MILVUS_BATCH = 1000

# Milvus 服务端对 search 的 limit 上限
MILVUS_MAX_LIMIT = 16384

# collection 名规则（**与 Chroma 不同，不要借用 vectorstore.collection_name**）：
# Milvus 服务端 validateCollectionName —— 只能字母/数字/下划线、首字符必须是字母或
# 下划线、长度 ≤255。Chroma 允许的 `-` 和 `.` 在这里都是非法字符。
_COLLECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
COLLECTION_PREFIX = "pn_"
# 认领「本项目建的 collection」用：pn_<slug>_<8位十六进制>。
# delete_paper 要跨模型清理，只动符合这个形状的集合，不碰同一个 Milvus 上别人的表。
_OURS_RE = re.compile(r"^pn_[A-Za-z0-9_]*_[0-9a-f]{8}$")


def collection_name(model: str | None = None) -> str:
    """**每个 embedding 模型一个 collection**，名字按 Milvus 的规则生成。

    与 `vectorstore.collection_name()` 的做法一致（sanitize + 8 位哈希去重，
    因为模型名可能含中文，sanitize 后可能彼此相同），但字符集与首字符规则不同：
    Milvus 只认 `[A-Za-z0-9_]` 且首字符须字母或下划线，所以非法字符一律换成 `_`，
    并固定加 `pn_` 前缀保证首字符合法（模型名以数字开头时尤其需要）。

    长度：3(前缀) + ≤64(slug) + 1 + 8(哈希) = ≤76，远小于 255 上限。
    """
    model = model if model is not None else (config.EMBED_MODEL or "")
    slug = re.sub(r"[^A-Za-z0-9_]", "_", model).strip("_")[:64] or "default"
    h = hashlib.sha1(model.encode("utf-8")).hexdigest()[:8]
    name = f"{COLLECTION_PREFIX}{slug}_{h}"
    if not _COLLECTION_RE.match(name):   # 兜底自检：生成规则改坏了要立刻炸，别等服务端报
        raise VectorStoreError(
            f"生成的 collection 名 {name!r} 不符合 Milvus 规则"
            f"（首字符须字母或下划线、只含字母数字下划线、长度 ≤255）")
    return name


# ── metric 方向：查表，不写死 ──

_METRIC_KIND = {
    "COSINE": "similarity",   # distance 字段直接就是余弦相似度，越大越相关
    "IP": "similarity",       # 内积，越大越相关
    "L2": "distance",         # 平方欧氏距离，越小越相关
}


def _to_score(distance: float) -> float:
    """把 Milvus 返回的 `distance` 字段转成本项目约定的**余弦相似度（越大越相关）**。

    COSINE/IP 直接返回；L2 没有无损的相似度换算（要靠向量已归一化才有
    `sim = 1 - d/2`），与其给一个「看着像分数」的数不如直接拒绝——本模块只用 COSINE。
    """
    kind = _METRIC_KIND.get(METRIC)
    if kind == "similarity":
        return float(distance)
    raise VectorStoreError(
        f"metric={METRIC!r} 返回的不是相似度（{kind or '未知'}），"
        f"本后端约定 score 必须是余弦相似度（越大越相关）；请改回 COSINE")


# ── where → Milvus 布尔表达式 ──
#
# 为什么必须下推到服务端：`search(limit=k)` 是**先取 top-k 再返回**，
# 拿回来在客户端过滤等于「top-k 里恰好有几条命中就返回几条」，
# 结果条数和内容都会随数据分布乱跳。Milvus 的 filter 是在检索时生效的。

_OP_EXPR = {"$eq": "==", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _literal(v) -> str:
    """把 Python 值写成 Milvus 表达式字面量。字符串靠 json.dumps 做转义，别手工拼引号。"""
    if isinstance(v, bool):          # 必须在 int 之前判断：isinstance(True, int) 为真
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not np.isfinite(v):
            raise VectorStoreError(f"where 的值不能是 NaN/Inf，收到 {v!r}")
        return repr(v)
    if isinstance(v, str):
        # ensure_ascii=False：Milvus 的表达式解析器认 UTF-8 字面量，
        # 转成 \uXXXX 反而多一层不确定性（中文关键词在本项目里很常见）。
        return json.dumps(v, ensure_ascii=False)
    raise VectorStoreError(f"where 的值只能是 str/int/float/bool，收到 {type(v).__name__}")


def _field_ref(key: str) -> str:
    """字段引用。保留键是真·标量字段；用户 meta 走 dynamic field。

    dynamic field 显式写成 `$meta["k"]` 而不是裸 `k`：键名带点或空格时裸写会被
    表达式解析器当成别的语法，而 `$meta[...]` 里的键名是字符串字面量，怎么写都安全。
    """
    if key in RESERVED_META:
        return key
    return f"$meta[{json.dumps(key, ensure_ascii=False)}]"


def _expr(where: dict) -> str:
    """递归翻译一个已通过 `validate_where` 的条件。每层都加括号，免得算符优先级踩坑。"""
    key, val = next(iter(where.items()))
    if key in ("$and", "$or"):
        joiner = " and " if key == "$and" else " or "
        return "(" + joiner.join(_expr(sub) for sub in val) + ")"
    ref = _field_ref(key)
    if not isinstance(val, dict):
        return f"({ref} == {_literal(val)})"
    op, operand = next(iter(val.items()))
    if op in ("$in", "$nin"):
        items = ", ".join(_literal(x) for x in operand)
        return f"({ref} {'in' if op == '$in' else 'not in'} [{items}])"
    return f"({ref} {_OP_EXPR[op]} {_literal(operand)})"


def where_to_expr(kind: str | None, where: dict | None) -> str:
    """把 (kind, where) 合成一条 Milvus filter 表达式；无条件时返回空串（Milvus 的「不过滤」）。

    语法与校验完全沿用 `vectorstore.validate_where`，三个后端共用同一套 where 方言——
    否则同一份调用代码会在一个后端跑通、在另一个上炸，这层抽象就白做了。
    """
    validate_where(where)
    parts = []
    if kind is not None:
        parts.append(f"({_field_ref('kind')} == {_literal(kind)})")
    if where:
        parts.append(_expr(where))
    return " and ".join(parts)


# ── id 编码：与 ChromaStore 逐字相同，便于两个后端逐条对照排查 ──

def _make_id(paper_id: int, kind: str, idx: int) -> str:
    return f"{kind}:{paper_id}:{idx}"


def _check_meta_clash(meta: Any):
    """meta 撞上 Milvus 的 schema 字段名就会把主键/向量覆盖掉。

    `vectorstore._norm_item` 只拦了 paper_id/kind/idx 这三个存储层保留键，
    `id` 与 `vector` 是 Milvus 独有的字段名，只能在这里补拦。
    """
    if not isinstance(meta, dict):
        return
    clash = sorted(k for k in meta if k in (PK_FIELD, VEC_FIELD))
    if clash:
        raise VectorStoreError(
            f"meta 不能使用 Milvus 的 schema 字段名 {clash}"
            f"（{PK_FIELD!r} 是主键、{VEC_FIELD!r} 是向量字段）")


def _parse_id(vid: Any) -> tuple[int, str, int] | None:
    """反解主键。解不出来说明索引里混进了不是本模块写的行 → 计入 stale 丢弃。"""
    if not isinstance(vid, str):
        return None
    parts = vid.split(":")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[1]), parts[0], int(parts[2]))
    except ValueError:
        return None


# ── 惰性 import 与连接缓存 ──

_MOD_LOCK = threading.Lock()
_MOD: Any = None

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple, Any] = {}

# 「SQLite 有向量、Milvus 里连 collection 都没有」只在每个集合上提醒一次，
# 免得每次 search 都刷屏。
_WARNED: set[str] = set()


def _milvus():
    """惰性 import pymilvus。

    **模块 import 期绝不碰它**：本项目在开发机上要能 `import papernest.milvusstore`
    而机器上没有 Milvus 服务，甚至可以没装 pymilvus——import 就连服务的话，
    整个项目（api/cli 都会间接 import 到）在开发机上直接起不来。
    """
    global _MOD
    if _MOD is not None:
        return _MOD
    try:
        import pymilvus
    except Exception as e:      # ImportError，也可能是 grpc 二进制在本机不兼容
        raise VectorStoreUnavailable(
            f"pymilvus 导入失败（{type(e).__name__}: {e}）；"
            f"`pip install pymilvus` 后重试，或改用 numpy / chroma 后端") from e
    with _MOD_LOCK:
        _MOD = pymilvus
    return _MOD


def milvus_uri() -> str:
    """运行期读环境变量（不是 import 期）——测试改环境变量、运维改 .env 都要立刻生效。"""
    return (os.environ.get(URI_ENV) or DEFAULT_URI).strip()


def index_type() -> str:
    return (os.environ.get(INDEX_ENV) or DEFAULT_INDEX_TYPE).strip() or DEFAULT_INDEX_TYPE


def consistency_level() -> str:
    return (os.environ.get(CONSISTENCY_ENV) or DEFAULT_CONSISTENCY).strip() or DEFAULT_CONSISTENCY


def _conn_error(what: str, uri: str, e: Exception) -> VectorStoreUnavailable:
    """连接/RPC 失败时的统一错误。**必须指出排查方向，不能只说「失败了」。**

    四个方向覆盖了实际会遇到的全部情形，按从最常见到最少见排：服务没起来 >
    地址写错（尤其是写成本地路径被当成 Milvus Lite）> 鉴权/库名 > 索引没建/没 load。
    """
    return VectorStoreUnavailable(
        f"Milvus {what}（uri={uri}）：{type(e).__name__}: {e}\n"
        f"  1) 服务没起来？Milvus 必须是**服务端**（docker compose 起 milvus-standalone）。"
        f"注意 Milvus Lite（嵌入式）在 Windows 上不存在——全部历史版本零个 Windows wheel，"
        f"`pip install pymilvus[milvus_lite]` 在 Windows 上会静默跳过。\n"
        f"  2) 地址不对？当前 {uri}，用 {URI_ENV} 覆盖（默认 {DEFAULT_URI}）。"
        f"**必须带 http:// 前缀**；写成本地文件路径会被 pymilvus 当作 Milvus Lite，"
        f"报 'milvus-lite is required for local database connections'。\n"
        f"  3) 鉴权或库名不对？用 {TOKEN_ENV} 配 token、{DB_ENV} 配 database 名。\n"
        f"  4) collection 没建索引或没 load？本后端每次开集合都会补 load_collection()，"
        f"若仍报 'collection not loaded' / 'index not found'，跑一次 store.rebuild() "
        f"从 SQLite 全量重建。")


def _milvus_client(uri: str, token: str, db_name: str):
    """按 (uri, token, db) 缓存 MilvusClient——它内部拖着 gRPC channel 和线程，
    每次 get_store 都新建等于每次都握手一遍，还会攒连接。"""
    key = (uri, token, db_name)
    with _CLIENT_LOCK:
        cli = _CLIENTS.get(key)
    if cli is not None:
        return cli
    mv = _milvus()
    try:
        # MilvusClient 的构造函数**会真的建连接并握手**，连不上就在这里抛。
        # 这正是我们要的：`get_store("milvus")` 拿不到就抛 VectorStoreUnavailable，
        # 由 get_store_or_degrade 决定降级，而不是等到第一次 search 才炸。
        cli = mv.MilvusClient(uri=uri, token=token, db_name=db_name)
    except Exception as e:
        raise _conn_error("连接失败", uri, e) from e
    with _CLIENT_LOCK:
        _CLIENTS[key] = cli
    return cli


def reset_clients():
    """丢弃缓存的 Milvus 连接。测试换临时目录/换 uri 后调用。

    与 `vectorstore.reset_clients()` 同样的语义：**调用后此前拿到的 MilvusStore
    实例即作废**，必须重新 `get_store()`。不真正 close 的 reset 只是把连接泄漏藏起来
    （本项目已经因为「reset 不放句柄」在 Windows 上攒过几千个删不掉的临时目录）。

    整段吞异常是安全的：这只是回收，不是正确性的一环，pymilvus 换版本改了
    `close()` 的行为也不该让 reset 本身炸。
    """
    with _CLIENT_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for cli in clients:
        try:
            cli.close()
        except Exception:
            pass
    _WARNED.clear()


# ── 后端实现 ──

class MilvusStore:
    """Milvus 后端：服务端 ANN 索引。**派生索引，不是真相来源。**

    公开 API 与 `vectorstore.VectorStore` 协议一致（`upsert` / `search` /
    `delete_paper` / `count` / `rebuild`），另加两个自检用的：

    - `index_count()`：Milvus 侧的实际条数。与 `count()`（SQLite 真相）不等即需 `rebuild()`。
    - `last_stale_hits`：上次 `search()` 丢弃了几条「索引里还有、真相里已经没有」的行。
    """

    name = "milvus"
    degraded: str | None = None

    def __init__(self):
        _boot()
        self.last_stale_hits = 0
        self.uri = milvus_uri()
        self._client = _milvus_client(
            self.uri,
            os.environ.get(TOKEN_ENV, "") or "",
            os.environ.get(DB_ENV, "") or "",
        )
        # 已确认「存在 + 建了索引 + load 过」的集合名；维度也一并记住，
        # 免得每次 upsert/search 都去 describe_collection 走一趟 RPC。
        self._ready: set[str] = set()
        self._dims: dict[str, int] = {}

    # —— collection 生命周期 ——

    @property
    def collection(self) -> str:
        """当前 EMBED_MODEL 对应的 collection 名。

        每次现算而不缓存：模型可以在进程存续期间被改（测试会改、`cli.py embed`
        迁移也会改），缓存住就会拿旧模型的索引回答新模型的查询。
        `collection_name()` 只是一次 sha1，比一次错误的检索便宜得多。
        """
        return collection_name(config.EMBED_MODEL)

    def _create(self, name: str, dim: int):
        """建 collection：schema + 索引 + load 一次做完。

        **务必传 index_params**：pymilvus 3.0.1 的 `_create_collection_with_schema`
        只有在 `index_params` 非空时才会 `create_index()` + `load_collection()`。
        不传就是「建好了、能插入、一搜就报 collection not loaded」——Milvus 最常见的坑。
        """
        mv = _milvus()
        dt = mv.DataType
        schema = mv.MilvusClient.create_schema(
            auto_id=False,
            # dynamic field 用来放用户 meta：meta 的键是任意的，塞不进固定 schema。
            # 开了它，行 dict 里多出来的键会进隐藏的 $meta，且服务端可过滤。
            enable_dynamic_field=True,
        )
        schema.add_field(PK_FIELD, dt.VARCHAR, is_primary=True, max_length=PK_MAX_LEN)
        # paper_id / kind / idx 是**声明过的标量字段**，不是 dynamic field：
        # where 过滤和 delete_paper 全靠它们，走真字段才有服务端索引可用。
        schema.add_field("paper_id", dt.INT64)
        schema.add_field("kind", dt.VARCHAR, max_length=KIND_MAX_LEN)
        schema.add_field("idx", dt.INT64)
        schema.add_field(VEC_FIELD, dt.FLOAT_VECTOR, dim=int(dim))
        # 正文**不进 Milvus**：text 的权威在 SQLite（`_texts_for` 回取），
        # 而 VARCHAR 有 65535 上限，长 chunk 塞进去会直接写失败——存一份用不上的
        # 副本换一个真实的失败模式，不划算。

        index_params = mv.MilvusClient.prepare_index_params()
        index_params.add_index(field_name=VEC_FIELD, index_type=index_type(),
                               metric_type=METRIC, index_name=f"{VEC_FIELD}_idx")
        try:
            self._client.create_collection(
                collection_name=name, schema=schema, index_params=index_params,
                consistency_level=consistency_level())
        except Exception as e:
            # 并发下另一个进程可能刚好建好了，这不算失败。
            try:
                if self._client.has_collection(name):
                    self._load(name)
                    return
            except Exception:
                pass
            raise _conn_error(f"建 collection {name!r}（dim={dim}）失败", self.uri, e) from e

    def _load(self, name: str):
        """把 collection load 进内存。**已存在的集合每次开都要补这一下。**

        进程重启、Milvus 重启、别的进程调过 release_collection，都会让一个「明明
        存在、也建了索引」的集合处于未 load 状态，此时 search 报 collection not loaded。
        `load_collection` 幂等，已 load 时是廉价的空操作，所以无条件补。
        """
        try:
            self._client.load_collection(name)
        except Exception as e:
            raise _conn_error(f"load collection {name!r} 失败", self.uri, e) from e

    def _described_dim(self, name: str) -> int | None:
        """从服务端读回向量字段的维度。**维度是建 collection 时固定的，只能读不能改。**"""
        try:
            desc = self._client.describe_collection(name)
        except Exception as e:
            raise _conn_error(f"describe collection {name!r} 失败", self.uri, e) from e
        for f in (desc or {}).get("fields") or []:
            if f.get("name") == VEC_FIELD:
                d = (f.get("params") or {}).get("dim")
                return int(d) if d is not None else None
        return None

    def _open(self, dim: int | None, create: bool) -> str | None:
        """确保当前模型的 collection 可用，返回集合名；`create=False` 且不存在时返回 None。

        读路径不建集合（`create=False`）：一次查询在服务端建表是意料之外的副作用，
        而且读路径也未必知道维度。
        """
        name = self.collection
        if name not in self._ready:
            try:
                exists = bool(self._client.has_collection(name))
            except Exception as e:
                raise _conn_error(f"has_collection({name!r}) 失败", self.uri, e) from e
            if not exists:
                if not create:
                    return None
                if dim is None:
                    raise VectorStoreError(
                        f"collection {name!r} 不存在且维度未知，无法创建；"
                        f"Milvus 的维度在建 collection 时固定，必须先有一批向量")
                self._create(name, int(dim))
                self._dims[name] = int(dim)
            else:
                self._load(name)
                d = self._described_dim(name)
                if d is not None:
                    self._dims[name] = d
            self._ready.add(name)
        have = self._dims.get(name)
        if dim is not None and have is not None and int(have) != int(dim):
            # 绝不静默写入：维度不同的向量塞进同一个索引，算出来的分数没有意义。
            # 口径与 vectorstore.DimensionMismatch 一致。
            raise DimensionMismatch(
                f"向量维度不一致：Milvus collection {name!r} 建的是 {have} 维、"
                f"本次是 {dim} 维（model={config.EMBED_MODEL!r}）；"
                f"Milvus 的维度在建 collection 时固定、改不了，"
                f"换嵌入模型请换 model 名（会自动换一个 collection）或先 drop 后 rebuild()")
        return name

    # —— VectorStore 协议 ——

    def upsert(self, items: list[dict]) -> int:
        """先落 SQLite（真相），再同步 Milvus。Milvus 这步失败了也能 `rebuild()` 回来。"""
        # 字段名冲突要在**写真相之前**拦：写完 SQLite 再报错，等于留下一批
        # 「SQLite 有、Milvus 永远同步不上」的行，下次 upsert 还是同一个错。
        for it in items or []:
            _check_meta_clash(it.get("meta") if isinstance(it, dict) else None)
        prepared = _write_items(items)
        if not prepared:
            return 0
        dim = int(prepared[0][4].size)
        name = self._open(dim, create=True)
        rows = [self._row(p) for p in prepared]
        for i in range(0, len(rows), MILVUS_BATCH):
            batch = rows[i:i + MILVUS_BATCH]
            try:
                self._client.upsert(collection_name=name, data=batch)
            except Exception as e:
                raise _conn_error(
                    f"upsert 第 {i // MILVUS_BATCH + 1} 批（{len(batch)} 条）到 "
                    f"{name!r} 失败；SQLite 已写入，跑 store.rebuild() 可让索引追上",
                    self.uri, e) from e
        return len(prepared)

    @staticmethod
    def _row(prepared: tuple) -> dict:
        """规整行 → Milvus 行 dict。保留键进真字段，其余 meta 进 dynamic field。"""
        paper_id, kind, idx, _text, vec, meta = prepared
        # 再拦一次：rebuild 的数据是从 SQLite 读回来的，可能是别的后端当初写进去的，
        # 没走过 upsert 那道前置检查。
        _check_meta_clash(meta)
        row = {PK_FIELD: _make_id(paper_id, kind, idx), VEC_FIELD: vec.tolist()}
        # _full_meta 把 paper_id/kind/idx 补回来，与另两个后端存的是同一份 metadata。
        row.update(_full_meta((paper_id, kind, idx), meta))
        return row

    def search(self, qvec, top_k: int = 8, kind: str | None = None,
               where: dict | None = None) -> list[dict]:
        """向量检索。**score 是余弦相似度（越大越相关）**，排序全序。"""
        expr = where_to_expr(kind, where)   # 顺带校验 where 语法（三个后端同一套）
        model = _model()
        self.last_stale_hits = 0
        if top_k <= 0:
            return []
        with db.conn() as c:      # 一次查询只开一条连接：连接本身比查询还贵
            dim = _store_dim(c, model)
            q = _prep_query(qvec, dim)      # 维度以 SQLite 为准，三个后端同一句错
            if dim is None:                 # 真相来源里这个模型一条向量都没有
                return []
            name = self._open(dim, create=False)
            if name is None:
                self._warn_missing(name=self.collection, model=model)
                return []
            # 多取一些再全序排：ANN 只保证近似 top-k，边界上并列的行谁进谁不进是不定的；
            # 而且回取时还要丢掉 stale 行，取紧了会不够。
            limit = min(MILVUS_MAX_LIMIT, max(int(top_k) * 3 + 10, int(top_k)))
            try:
                res = self._client.search(
                    collection_name=name, data=[q.tolist()], limit=limit,
                    filter=expr, search_params={"metric_type": METRIC},
                    # 显式传：不传就继承 collection **建表时**的级别，
                    # 那样改默认值对已存在的 collection 完全不生效（实测踩到）。
                    consistency_level=consistency_level())
            except Exception as e:
                raise _conn_error(
                    f"search collection {name!r} 失败（filter={expr!r}）", self.uri, e) from e
            return self._collect(res, model, int(top_k), c)

    def _collect(self, res, model: str, top_k: int, c) -> list[dict]:
        """把 Milvus 的嵌套返回结构整理成本项目的结果格式。

        pymilvus 3.0.1 的 `search()` 返回 `List[List[dict]]`（每个查询向量一层），
        每条 hit 形如 `{"<主键字段名>": pk, "distance": float, "entity": {...}}`——
        主键字段名跟着 schema 走，本模块的主键就叫 `id`。
        """
        hits = list(res[0]) if res is not None and len(res) else []
        pairs: list[tuple[tuple[int, str, int], float]] = []
        for h in hits:
            key = _parse_id(_hit_pk(h))
            if key is None:
                self.last_stale_hits += 1     # 不是本模块写的行，丢弃并留痕
                continue
            pairs.append((key, _to_score(_hit_distance(h))))
        texts = _texts_for([p[0] for p in pairs], model, c)
        live = []
        for key, score in pairs:
            if key not in texts:
                self.last_stale_hits += 1     # 索引里还有、真相里已经没有 → 丢弃并留痕
                continue
            live.append((key, score))
        # 全序：分数降序，并列按 (paper_id, kind, idx) 决胜。
        # 排序键不全序，跨进程/跨次结果就会漂——本项目踩过。
        order = _rank(list(range(len(live))), lambda i: live[i][1],
                      lambda i: live[i][0], top_k)
        return [{"paper_id": live[i][0][0], "kind": live[i][0][1], "idx": live[i][0][2],
                 "score": float(live[i][1]), "text": texts[live[i][0]]} for i in order]

    @staticmethod
    def _warn_missing(name: str, model: str):
        """SQLite 有向量但 Milvus 连集合都没有 —— 返回空结果是对的，但不能一声不吭。

        这就是 `count() != index_count()` 的极端情形。每个集合只提醒一次，免得刷屏。
        """
        if name in _WARNED:
            return
        _WARNED.add(name)
        print(f"[milvusstore] collection {name!r}（model={model!r}）在 Milvus 上不存在，"
              f"检索返回空；SQLite 里是有向量的，跑 store.rebuild() 建索引", file=sys.stderr)

    def delete_paper(self, paper_id: int) -> int:
        """删掉该论文的全部向量。**真相先删**，再清所有模型的 collection。

        为什么要跨 collection：`vectorstore._delete_paper_rows` 删的是 SQLite 里
        **所有模型**的行（「这篇没了」不该只删当前模型那一份），Milvus 这边不跟着删，
        另一个模型的集合里就会永久留着一批 stale 行——search 时会被丢弃所以不影响
        正确性，但每次都白占 top-k 名额、也让 index_count 永远对不上。
        只动名字符合本项目形状（pn_<slug>_<8位hex>）的集合，不碰同一个 Milvus 上别人的表。
        """
        paper_id = int(paper_id)
        n = _delete_paper_rows(paper_id)
        expr = f"(paper_id == {paper_id})"
        failed = []
        for name in self._our_collections():
            try:
                self._client.delete(collection_name=name, filter=expr)
            except Exception as e:
                failed.append(f"{name}: {type(e).__name__}: {e}")
        if failed:
            raise VectorStoreUnavailable(
                f"SQLite 已删除 paper_id={paper_id}，但 Milvus 索引删除失败："
                f"{'; '.join(failed)}；请跑 store.rebuild() 使索引与真相重新一致")
        return n

    def _our_collections(self) -> list[str]:
        try:
            names = self._client.list_collections() or []
        except Exception as e:
            raise _conn_error("list_collections 失败", self.uri, e) from e
        return [n for n in names if isinstance(n, str) and _OURS_RE.match(n)]

    def count(self, kind: str | None = None) -> int:
        """按**真相来源**计数。派生索引自己的条数看 `index_count()`，两者不等即需 rebuild。"""
        return _count_rows(kind)

    def index_count(self) -> int:
        """Milvus 侧当前模型 collection 的实际条数。

        用 `query(output_fields=["count(*)"])` 而不是 `get_collection_stats()["row_count"]`：
        后者是 segment 级的估算，未 flush 的新写入和未 compact 的删除都不计入，
        拿它和 `count()` 比会得到假的「不一致」。
        """
        name = self._open(None, create=False)
        if name is None:
            return 0
        try:
            rows = self._client.query(collection_name=name, output_fields=["count(*)"])
        except Exception as e:
            raise _conn_error(f"query count(*) on {name!r} 失败", self.uri, e) from e
        for r in rows or []:
            for k in ("count(*)", "count"):
                if k in r:
                    return int(r[k])
        return 0

    def rebuild(self, progress: Callable[[int, int], Any] | None = None) -> dict:
        """丢掉整个 collection 再从 SQLite 全量重放。派生索引可以随时丢，这就是它的价值。"""
        t0 = time.perf_counter()
        errors: list[str] = []
        model = _model()
        rows = _iter_truth(model)
        name = collection_name(model)
        try:
            if self._client.has_collection(name):
                self._client.drop_collection(name)
        except Exception as e:
            errors.append(f"删除旧 collection {name!r} 失败（按空集合继续）："
                          f"{type(e).__name__}: {e}")
        self._ready.discard(name)
        self._dims.pop(name, None)
        _WARNED.discard(name)

        total, done = len(rows), 0
        if not rows:
            # Milvus 建 collection 必须给维度，一条向量都没有就建不出来——
            # 这不是失败，是「没什么可索引的」。如实返回 indexed=0。
            if progress:
                progress(0, 0)
            return {"backend": self.name, "indexed": 0,
                    "elapsed_s": round(time.perf_counter() - t0, 4), "errors": errors}

        dims = {len(r["vec"]) // 4 for r in rows}
        if len(dims) > 1:
            raise DimensionMismatch(
                f"库内 model={model!r} 的向量存在多种维度 {sorted(dims)}，"
                f"Milvus 的 collection 只能有一个维度；请先清理 SQLite 再 rebuild")
        self._open(dims.pop(), create=True)

        for i in range(0, total, MILVUS_BATCH):
            batch = rows[i:i + MILVUS_BATCH]
            data = []
            for r in batch:
                key = (int(r["paper_id"]), r["kind"], int(r["idx"]))
                meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
                vec = np.frombuffer(r["vec"], dtype=np.float32)
                data.append(self._row((key[0], key[1], key[2], r["text"], vec, meta)))
            try:
                # 刚 drop 过，集合是空的，insert 即可（与 ChromaStore.rebuild 用 add 同理）。
                self._client.insert(collection_name=name, data=data)
                done += len(data)
            except Exception as e:
                errors.append(f"第 {i // MILVUS_BATCH + 1} 批（{len(data)} 条）失败："
                              f"{type(e).__name__}: {e}")
            if progress:
                progress(done, total)

        # 落盘后强制可见。默认一致性是 Bounded（查询快 25 倍：16ms vs 400ms），
        # 代价是写入约 300ms 后才可读——**唯独 rebuild 不能接受这个滞后**：
        # 「重建完立刻查一下对不对」是最自然的用法，这时返回空会被当成 rebuild 坏了。
        # flush 一次把这个窗口关掉，成本只在重建时付一次，不摊到每次查询上。
        try:
            self._client.flush(collection_name=name)
        except Exception as e:      # flush 失败不该让已经写成功的重建报废
            errors.append(f"flush 失败（数据已写入，可能要等几百毫秒才可见）：{type(e).__name__}: {e}")
        return {"backend": self.name, "indexed": done,
                "elapsed_s": round(time.perf_counter() - t0, 4), "errors": errors}


def _hit_pk(hit: Any) -> Any:
    """取一条 hit 的主键。

    pymilvus 3.0.1 的 `Hit` 是 dict 子类，键名是**主键字段名**（`search_result.py`
    里 `Hit({pk_name: pks[i], "distance": ..., "entity": ...})`），本模块的主键
    就叫 `id`。`Hit.__getitem__` 取不到时还会回落到 `entity`，所以 entity 里带
    主键的情形也能取到；这里再兜一层普通 dict 的写法，免得依赖 Hit 的私有行为。
    """
    if isinstance(hit, dict):
        if PK_FIELD in hit:
            return hit[PK_FIELD]
        ent = hit.get("entity")
        if isinstance(ent, dict) and PK_FIELD in ent:
            return ent[PK_FIELD]
        return None
    return getattr(hit, "id", None)


def _hit_distance(hit: Any) -> float:
    """取一条 hit 的 `distance` 字段。**注意它在 COSINE 下是相似度**，见 `_to_score`。"""
    if isinstance(hit, dict):
        if "distance" in hit:
            return float(hit["distance"])
        ent = hit.get("entity")
        if isinstance(ent, dict) and "distance" in ent:
            return float(ent["distance"])
        raise VectorStoreError(f"Milvus 返回的 hit 里没有 distance 字段：{hit!r}")
    return float(getattr(hit, "distance"))
