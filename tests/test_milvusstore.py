"""Milvus 后端（papernest/milvusstore.py）：**完全离线**，用假服务端跑契约断言。

## 为什么全是 mock，而且一条 skip 都没有

开发机的 Docker Desktop 是停的，本机连不上任何 Milvus；而 Milvus Lite（嵌入式）
在 Windows 上根本不存在——全部历史版本零个 Windows wheel，
`pip install pymilvus[milvus_lite]` 在 Windows 上静默跳过，
`MilvusClient("./local.db")` 直接抛 `ConnectionConfigException`。
所以「有服务就测、没有就 skip」的写法在这台机器上等于**一条都不跑**，
CI 绿灯却什么也没验证。这里改成：假服务端把 Milvus 的真实行为建模出来，
核心用例一条不落地跑。

## 假服务端必须像真的，否则测了个寂寞

本项目已经三次踩过「合成夹具与真实数据形态不符」的亏，所以 `FakeMilvusClient`
不是随手 MagicMock，而是照着 pymilvus 3.0.1 的**源码**建的模型：

- `search()` 返回 `List[List[dict]]`，每条 hit 是
  `{"<主键字段名>": pk, "distance": float, "entity": {...}}`
  （依据：`pymilvus/client/search_result.py` 里
  `Hit({pk_name: pks[i], "distance": distances[i], "entity": entity})`）。
- **COSINE 的 `distance` 字段返回的是相似度**（越大越相关），不是距离。
  依据：`pymilvus/client/iterator/search_iterator.py` 的 `metrics_positive_related()`
  对 L2 返回 True、对 **IP/COSINE 返回 False**。
- `create_collection(schema=..., index_params=None)` **不建索引也不 load**，
  之后 `insert` 成功、`search` 报 `collection not loaded`。
  依据：`pymilvus/milvus_client/milvus_client.py` 的 `_create_collection_with_schema`
  里 `if index_params: self.create_index(...); self.load_collection(...)`。
  假服务端如实复刻这个坑——不复刻的话，「每次开集合都要补 load」这条就测不出来。
- schema 与 index_params 用的是**真的 pymilvus 类**（`MilvusClient.create_schema` /
  `prepare_index_params` 直接转发过去），所以字段类型、dim、metric 这些写错了
  会在构造期就炸，而不是被一个宽容的 mock 吞掉。
- 抛的是真的 `pymilvus.exceptions.MilvusException`。

## 临时目录

用 `tests/support.py` 的 `TempDbTestCase`（`tempfile.mkdtemp` +
`addCleanup(shutil.rmtree, ..., True)`）。本项目刚清理过 40,513 个测试残留目录、
9.29GB，不再手写 tearDown。
"""
import importlib
import io
import os
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

from papernest import config, db, milvusstore as ms, vectorstore as vs

# `-s tests -t tests` 时 support 是顶层模块，`-t .` 时是 tests 包的一部分——两种都兼容。
try:
    from .support import TempDbTestCase
except ImportError:                      # pragma: no cover - 取决于 discover 的 -t
    from support import TempDbTestCase

# 真 pymilvus：只借它的 schema/索引构造与异常类型，全程不连服务。
from pymilvus import DataType, MilvusClient as RealMilvusClient
from pymilvus.client.search_result import Hit as RealHit
from pymilvus.exceptions import MilvusException

DIM = 8
MODEL_A = "text-embedding-a"
MODEL_B = "text-embedding-b"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _basis(i: int, dim: int = DIM) -> np.ndarray:
    """单位基向量。互相正交（余弦 0）、与自身余弦 1——正好用来钉死分数方向。"""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _mix(a: int, b: int, w: float, dim: int = DIM) -> np.ndarray:
    """两个基向量的确定性混合，用来造「有区分度但不并列」的分数。"""
    return _unit(_basis(a, dim) * (1.0 - w) + _basis(b, dim) * w)


# ── 假 Milvus 服务端 ─────────────────────────────────────────────────────────

class _Missing:
    """dynamic field 缺失时的哨兵：任何比较都不匹配。

    对齐 `vectorstore.match_where` 的「缺失的键一律不匹配（不当 NULL 参与比较）」。
    """

    def __eq__(self, other): return False
    def __ne__(self, other): return False
    def __lt__(self, other): return False
    def __le__(self, other): return False
    def __gt__(self, other): return False
    def __ge__(self, other): return False

    __hash__ = None       # 不可哈希 → `x in [..]` 会走列表的逐个 == 比较，同样不匹配


MISSING = _Missing()


def _index_cfg(index_params) -> dict:
    """IndexParams 是 IndexParam 的列表，元素不是 dict——要 `.to_dict()` 才拿得到配置。

    照着 pymilvus/milvus_client/index.py 的真实结构来，不是猜的。
    """
    return list(index_params)[0].to_dict()


def _expr_to_python(expr: str) -> str:
    """把 Milvus 布尔表达式翻成等价的 Python 表达式，供假服务端求值。

    只处理本模块 `where_to_expr()` 会生成的那一小撮语法。逐字符扫描而不是正则替换，
    是因为**字符串字面量里可能出现 `$meta[` 或 `true` 这样的字样**，
    正则会把它们也换掉——那就是「夹具比被测代码还容易错」。
    """
    out, i, n = [], 0, len(expr)

    def _at_word(pos: int, word: str) -> bool:
        if not expr.startswith(word, pos):
            return False
        before = expr[pos - 1] if pos else " "
        after = expr[pos + len(word)] if pos + len(word) < n else " "
        return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")

    def _scan_string(pos: int) -> int:
        """返回闭合引号的下标。"""
        j = pos + 1
        while j < n:
            if expr[j] == "\\":
                j += 2
                continue
            if expr[j] == '"':
                return j
            j += 1
        raise AssertionError(f"未闭合的字符串字面量：{expr!r}")

    while i < n:
        ch = expr[i]
        if ch == '"':
            j = _scan_string(i)
            out.append(expr[i:j + 1])
            i = j + 1
            continue
        if expr.startswith('$meta["', i):
            j = _scan_string(i + 6)
            assert expr[j + 1] == "]", f"$meta 引用没闭合：{expr!r}"
            out.append(f"M({expr[i + 6:j + 1]})")
            i = j + 2
            continue
        if _at_word(i, "true"):
            out.append("True")
            i += 4
            continue
        if _at_word(i, "false"):
            out.append("False")
            i += 5
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _eval_filter(expr: str, row: dict) -> bool:
    if not expr:
        return True
    ns = {
        "M": lambda k: row.get(k, MISSING),
        "paper_id": row.get("paper_id"),
        "kind": row.get("kind"),
        "idx": row.get("idx"),
    }
    return bool(eval(_expr_to_python(expr), {"__builtins__": {}}, ns))  # noqa: S307


class FakeMilvusServer:
    """进程内的假 Milvus。**数据挂在 server 上而不是 client 上**——真实 Milvus 的数据
    在服务端，重建 client 不会丢数据，测「新进程要补 load_collection」时这点是关键。"""

    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.calls: list[tuple] = []          # (方法名, 主要参数) 流水，供断言调用顺序
        self.fail_connect: Exception | None = None
        self.fail_on: dict[str, Exception] = {}   # 方法名 → 抛什么，模拟单个 RPC 失败

    def rows(self, name: str) -> list[dict]:
        return list(self.collections[name]["rows"].values())


class FakeMilvusClient:
    """假 MilvusClient。schema / index_params 转发给真 pymilvus，行为照源码复刻。"""

    # 这两个是 MilvusClient 上的静态构造器，直接转发真实现：
    # 字段类型写错、dim 缺失、metric 名拼错都会在这里就炸，而不是被 mock 吞掉。
    create_schema = staticmethod(RealMilvusClient.create_schema)
    prepare_index_params = staticmethod(RealMilvusClient.prepare_index_params)

    def __init__(self, uri="", token="", db_name="", **kw):
        server = self.server
        server.calls.append(("__init__", uri, token, db_name))
        if server.fail_connect is not None:
            raise server.fail_connect
        self.uri = uri
        self.closed = False

    # 每个测试在 setUp 里把 server 绑到子类上（见 _MilvusTestBase）
    server: FakeMilvusServer = None

    # —— 内部 ——

    def _maybe_fail(self, method: str):
        exc = self.server.fail_on.get(method)
        if exc is not None:
            raise exc

    def _col(self, name: str) -> dict:
        col = self.server.collections.get(name)
        if col is None:
            raise MilvusException(message=f"collection not found[collection={name}]")
        return col

    # —— 集合生命周期 ——

    def has_collection(self, collection_name, **kw):
        self._maybe_fail("has_collection")
        self.server.calls.append(("has_collection", collection_name))
        return collection_name in self.server.collections

    def list_collections(self, **kw):
        self._maybe_fail("list_collections")
        self.server.calls.append(("list_collections",))
        return sorted(self.server.collections)

    def create_collection(self, collection_name, schema=None, index_params=None,
                          consistency_level=None, **kw):
        self._maybe_fail("create_collection")
        self.server.calls.append(("create_collection", collection_name,
                                  bool(index_params), consistency_level))
        if collection_name in self.server.collections:
            raise MilvusException(message=f"collection already exist[{collection_name}]")
        assert schema is not None, "本项目只走显式 schema 这条路"
        fields = {f.name: f for f in schema.fields}
        vec = fields.get("vector")
        assert vec is not None and vec.dtype == DataType.FLOAT_VECTOR
        self.server.collections[collection_name] = {
            "schema": schema,
            "fields": fields,
            "dim": int(vec.params["dim"]),
            "dynamic": bool(schema.enable_dynamic_field),
            "rows": {},
            # 复刻 pymilvus 3.0.1：不传 index_params 就既不建索引也不 load。
            "indexed": bool(index_params),
            "loaded": bool(index_params),
            "metric": (_index_cfg(index_params).get("metric_type") if index_params else None),
            "index_type": (_index_cfg(index_params).get("index_type") if index_params else None),
            "consistency": consistency_level,
        }

    def create_index(self, collection_name, index_params, **kw):
        self._maybe_fail("create_index")
        col = self._col(collection_name)
        col["indexed"] = True
        col["metric"] = _index_cfg(index_params).get("metric_type")
        self.server.calls.append(("create_index", collection_name))

    def load_collection(self, collection_name, **kw):
        self._maybe_fail("load_collection")
        col = self._col(collection_name)
        if not col["indexed"]:
            raise MilvusException(
                message=f"index not found[collection={collection_name}]")
        col["loaded"] = True
        self.server.calls.append(("load_collection", collection_name))

    def release_collection(self, collection_name, **kw):
        self._col(collection_name)["loaded"] = False

    def drop_collection(self, collection_name, **kw):
        self._maybe_fail("drop_collection")
        self.server.calls.append(("drop_collection", collection_name))
        self.server.collections.pop(collection_name, None)

    def describe_collection(self, collection_name, **kw):
        self._maybe_fail("describe_collection")
        col = self._col(collection_name)
        # 形状照 pymilvus/client/abstract.py 的 CollectionSchema.dict() / FieldSchema.dict()
        return {
            "collection_name": collection_name,
            "auto_id": False,
            "enable_dynamic_field": col["dynamic"],
            "fields": [{"field_id": 100 + i, "name": f.name, "description": f.description,
                        "type": f.dtype, "params": dict(f.params or {})}
                       for i, f in enumerate(col["schema"].fields)],
        }

    # —— 数据面 ——

    def _put(self, collection_name, data, op):
        col = self._col(collection_name)
        n = 0
        for row in data:
            unknown = [k for k in row if k not in col["fields"]]
            if unknown and not col["dynamic"]:
                raise MilvusException(message=f"field not exist: {sorted(unknown)}")
            for fname in col["fields"]:
                if fname not in row:
                    raise MilvusException(message=f"field {fname} missing in row")
            vec = row["vector"]
            if len(vec) != col["dim"]:
                raise MilvusException(
                    message=f"vector dimension mismatch, expected {col['dim']}, "
                            f"got {len(vec)}")
            pk = row["id"]
            if op == "insert" and pk in col["rows"]:
                raise MilvusException(message=f"duplicated primary key {pk}")
            col["rows"][pk] = dict(row)
            n += 1
        self.server.calls.append((op, collection_name, n))
        return {f"{op}_count": n}

    def insert(self, collection_name, data, **kw):
        self._maybe_fail("insert")
        return self._put(collection_name, data, "insert")

    def upsert(self, collection_name, data, **kw):
        self._maybe_fail("upsert")
        return self._put(collection_name, data, "upsert")

    def delete(self, collection_name, ids=None, filter=None, **kw):  # noqa: A002
        self._maybe_fail("delete")
        col = self._col(collection_name)
        hit = [pk for pk, row in col["rows"].items() if _eval_filter(filter or "", row)]
        for pk in hit:
            col["rows"].pop(pk)
        self.server.calls.append(("delete", collection_name, filter, len(hit)))
        return {"delete_count": len(hit)}

    def query(self, collection_name, filter="", output_fields=None, **kw):  # noqa: A002
        self._maybe_fail("query")
        col = self._col(collection_name)
        if not col["loaded"]:
            raise MilvusException(
                message=f"collection not loaded[collection={collection_name}]")
        rows = [r for r in col["rows"].values() if _eval_filter(filter, r)]
        self.server.calls.append(("query", collection_name, output_fields))
        if output_fields and "count(*)" in output_fields:
            return [{"count(*)": len(rows)}]
        return [{k: v for k, v in r.items() if k != "vector"} for r in rows]

    def flush(self, collection_name, **kw):
        """真实 client 有 flush；假的缺了它，被测代码的 flush 分支就永远走异常路径，
        「rebuild 后立刻可读」这条契约等于没测到。"""
        self._maybe_fail("flush")
        self._col(collection_name)          # 集合不存在要按真实行为报错
        self.server.calls.append(("flush", collection_name))

    def search(self, collection_name, data=None, limit=10, filter="",  # noqa: A002
               output_fields=None, search_params=None, **kw):
        self._maybe_fail("search")
        col = self._col(collection_name)
        # **这条是重点**：建了集合、插了数据，没 load 一样搜不了。
        if not col["loaded"]:
            raise MilvusException(
                message=f"collection not loaded[collection={collection_name}]")
        if not col["indexed"]:
            raise MilvusException(message=f"index not found[collection={collection_name}]")
        metric = (search_params or {}).get("metric_type")
        if metric and col["metric"] and metric != col["metric"]:
            raise MilvusException(
                message=f"metric type not match, expected {col['metric']}, got {metric}")
        q = np.asarray(data[0], dtype=np.float32)
        if q.size != col["dim"]:
            raise MilvusException(
                message=f"vector dimension mismatch, expected {col['dim']}, got {q.size}")
        qn = q / (float(np.linalg.norm(q)) + 1e-12)
        scored = []
        for row in col["rows"].values():
            if not _eval_filter(filter, row):
                continue
            v = np.asarray(row["vector"], dtype=np.float32)
            vn = v / (float(np.linalg.norm(v)) + 1e-12)
            # COSINE：Milvus 的 distance 字段返回的**就是余弦相似度**，不做 1-d。
            scored.append((row, float(np.dot(qn, vn))))
        # 稳定排序：并列时保持插入顺序。被测代码必须自己把并列排成全序，
        # 所以夹具**故意不**替它按 (paper_id,kind,idx) 决胜。
        scored.sort(key=lambda t: -t[1])
        hits = [{"id": r["id"], "distance": s,
                 "entity": {k: v for k, v in r.items()
                            if k in (output_fields or [])}}
                for r, s in scored[:limit]]
        self.server.calls.append(("search", collection_name, filter, limit))
        return [hits]

    def close(self):
        self.closed = True
        self.server.calls.append(("close",))


# ── 测试基类 ────────────────────────────────────────────────────────────────

class _MilvusTestBase(TempDbTestCase):
    """临时库 + 假 Milvus + 摁死的 EMBED_MODEL。"""

    prefix = "papernest_milvus"

    def setUp(self):
        super().setUp()                      # 临时库、embeddings/llm 已摁断
        self.server = FakeMilvusServer()

        # 每个用例一个独立的 client 类，免得 server 引用互相污染
        self.client_cls = type("BoundFakeClient", (FakeMilvusClient,),
                               {"server": self.server})
        self.fake_module = mock.Mock(MilvusClient=self.client_cls, DataType=DataType)
        p = mock.patch.object(ms, "_milvus", return_value=self.fake_module)
        p.start()
        self.addCleanup(p.stop)

        ms.reset_clients()
        self.addCleanup(ms.reset_clients)

        self.set_model(MODEL_A)
        os.environ.pop(ms.INDEX_ENV, None)
        self.addCleanup(os.environ.pop, ms.INDEX_ENV, None)

    def set_model(self, model: str):
        old = config.EMBED_MODEL
        config.EMBED_MODEL = model
        self.addCleanup(setattr, config, "EMBED_MODEL", old)

    def store(self) -> "ms.MilvusStore":
        return ms.MilvusStore()

    # —— 夹具 ——

    def seed(self, store, spec):
        """spec: [(paper_id, kind, idx, vec, text, meta?)]"""
        items = []
        for row in spec:
            paper_id, kind, idx, vec, text = row[:5]
            meta = row[5] if len(row) > 5 else None
            it = {"paper_id": paper_id, "kind": kind, "idx": idx,
                  "vec": np.asarray(vec, dtype=np.float32), "text": text}
            if meta:
                it["meta"] = meta
            items.append(it)
        return store.upsert(items)

    def sql_delete(self, paper_id: int, kind: str, idx: int):
        """绕过存储层直接从 SQLite 删一行——制造「Milvus 里还有、真相里没了」的状态。"""
        with db.conn() as c:
            c.execute("DELETE FROM vectors WHERE paper_id=? AND kind=? AND idx=? AND model=?",
                      (paper_id, kind, idx, config.EMBED_MODEL))
        vs._invalidate()


# ── 1. 模块 import 不能触发连接 ──────────────────────────────────────────────

class ImportSafetyTests(unittest.TestCase):
    """开发机上没有 Milvus 服务、甚至可以没装 pymilvus，`import` 必须照样成功。

    import 期就连服务的话，api.py / cli.py 一 import 到本模块，整个项目在开发机上
    就起不来了——这是「不可用时明确报错、但报错要发生在使用点」的底线。
    """

    def test_import_does_not_construct_client(self):
        import pymilvus
        with mock.patch.object(pymilvus, "MilvusClient") as fake_cls:
            importlib.reload(ms)
            self.assertEqual(fake_cls.call_count, 0,
                             "模块 import 期构造了 MilvusClient（=连了服务）")
        importlib.reload(ms)          # 还原成真模块，免得影响后续用例

    def test_import_does_not_even_import_pymilvus(self):
        """更强的一条：连 pymilvus 都不该在 import 期被拉进来。

        用子进程测，因为本测试文件自己 import 了 pymilvus，同进程里断不出来。
        """
        code = ("import sys; import papernest.milvusstore as m; "
                "print('pymilvus' in sys.modules, m._MOD is None)")
        root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ, PYTHONPATH=root, PYTHONIOENCODING="utf-8")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=root, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False True", out.stdout + out.stderr)


# ── 2. collection 命名必须过 Milvus 的规则（与 Chroma 不同） ──────────────────

class CollectionNameTests(unittest.TestCase):

    MILVUS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")

    MODELS = ["", "text-embedding-v3", "Qwen3.7-通用文本向量", "3-starts-with-digit",
              "BAAI/bge-large-zh-v1.5", "a" * 300, "...", "模型", "x.y-z_w"]

    def test_matches_milvus_rules(self):
        for m in self.MODELS:
            name = ms.collection_name(m)
            with self.subTest(model=m):
                self.assertRegex(name, self.MILVUS_RE)
                self.assertLessEqual(len(name), 255)
                self.assertTrue(name[0].isalpha() or name[0] == "_")

    def test_distinct_per_model(self):
        names = {ms.collection_name(m) for m in self.MODELS}
        self.assertEqual(len(names), len(self.MODELS), f"模型名撞车了：{sorted(names)}")

    def test_not_reusing_chroma_naming(self):
        """Chroma 的名字含 `-` 和 `.`，在 Milvus 上是非法字符——不能直接借用。"""
        chroma = vs.collection_name("text-embedding-v3")
        self.assertNotRegex(chroma, self.MILVUS_RE)
        self.assertRegex(ms.collection_name("text-embedding-v3"), self.MILVUS_RE)

    def test_stable(self):
        self.assertEqual(ms.collection_name(MODEL_A), ms.collection_name(MODEL_A))


# ── 3. where → Milvus 表达式 ────────────────────────────────────────────────

class WhereExprTests(unittest.TestCase):
    """过滤必须下推到服务端：`search(limit=k)` 是先取 top-k 再返回，
    拿回客户端再过滤会让结果条数随数据分布乱跳。"""

    def test_kind_only(self):
        self.assertEqual(ms.where_to_expr("chunk", None), '(kind == "chunk")')

    def test_no_condition(self):
        self.assertEqual(ms.where_to_expr(None, None), "")

    def test_reserved_key_is_real_field(self):
        self.assertEqual(ms.where_to_expr(None, {"paper_id": 7}), "(paper_id == 7)")

    def test_user_meta_goes_to_dynamic_field(self):
        self.assertEqual(ms.where_to_expr(None, {"section": "Method"}),
                         '($meta["section"] == "Method")')

    def test_operators(self):
        cases = {
            "$eq": "==", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=",
        }
        for op, sym in cases.items():
            self.assertEqual(ms.where_to_expr(None, {"paper_id": {op: 3}}),
                             f"(paper_id {sym} 3)")
        self.assertEqual(ms.where_to_expr(None, {"paper_id": {"$in": [1, 2]}}),
                         "(paper_id in [1, 2])")
        self.assertEqual(ms.where_to_expr(None, {"paper_id": {"$nin": [1]}}),
                         "(paper_id not in [1])")

    def test_boolean_composition_and_kind(self):
        expr = ms.where_to_expr("chunk", {"$or": [{"paper_id": 1}, {"tag": "x"}]})
        self.assertEqual(expr, '(kind == "chunk") and ((paper_id == 1) or ($meta["tag"] == "x"))')

    def test_string_escaping(self):
        expr = ms.where_to_expr(None, {"note": 'he said "hi" \\ ok'})
        self.assertEqual(expr, '($meta["note"] == "he said \\"hi\\" \\\\ ok")')
        # 翻回 Python 也必须还原成同一个字符串（假服务端就是这么求值的）
        self.assertTrue(_eval_filter(expr, {"note": 'he said "hi" \\ ok'}))
        self.assertFalse(_eval_filter(expr, {"note": "other"}))

    def test_bool_literal_is_lowercase(self):
        self.assertEqual(ms.where_to_expr(None, {"ok": True}), '($meta["ok"] == true)')

    def test_non_finite_rejected(self):
        with self.assertRaises(vs.VectorStoreError):
            ms.where_to_expr(None, {"x": {"$gt": float("inf")}})

    def test_reuses_shared_validation(self):
        """where 方言三个后端共用一套，否则同一份调用代码换后端就炸。"""
        with self.assertRaises(vs.VectorStoreError):
            ms.where_to_expr(None, {"a": 1, "b": 2})       # 顶层两个 key
        with self.assertRaises(vs.VectorStoreError):
            ms.where_to_expr(None, {"a": {"$like": "x"}})  # 不支持的操作符

    def test_missing_dynamic_key_does_not_match(self):
        expr = ms.where_to_expr(None, {"section": "Method"})
        self.assertFalse(_eval_filter(expr, {"paper_id": 1}))


# ── 4. 分数是相似度，不是距离 ────────────────────────────────────────────────

class ScoreDirectionTests(_MilvusTestBase):
    """Chroma 后端就因为「cosine space 返回的是距离」差点返回反的排序。
    Milvus 的 COSINE 相反——返回的**就是相似度**，再做一次 1-d 会把排序整个倒过来，
    而且分数看着还挺正常，肉眼测不出来。所以这条必须钉死。"""

    def test_same_direction_is_one_orthogonal_is_zero(self):
        st = self.store()
        self.seed(st, [
            (1, "chunk", 0, _basis(0), "同向"),
            (2, "chunk", 0, _basis(1), "正交"),
            (3, "chunk", 0, -_basis(0), "反向"),
        ])
        hits = st.search(_basis(0), top_k=3)
        by_pid = {h["paper_id"]: h for h in hits}
        self.assertAlmostEqual(by_pid[1]["score"], 1.0, places=5)
        self.assertAlmostEqual(by_pid[2]["score"], 0.0, places=5)
        self.assertAlmostEqual(by_pid[3]["score"], -1.0, places=5)
        self.assertEqual([h["paper_id"] for h in hits], [1, 2, 3],
                         "分数方向搞反了：越大应该越相关")

    def test_magnitude_does_not_change_score(self):
        """COSINE 与向量模长无关；分数若随模长变，说明用成了 IP。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0) * 5.0, "长向量")])
        hits = st.search(_basis(0) * 0.1, top_k=1)
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)

    def test_metric_passed_to_server_is_cosine(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "x")])
        st.search(_basis(0), top_k=1)
        name = ms.collection_name(MODEL_A)
        self.assertEqual(self.server.collections[name]["metric"], "COSINE")

    def test_to_score_rejects_distance_metric(self):
        """metric 换成返回距离的那种（L2），必须报错而不是把距离当分数返回。"""
        with mock.patch.object(ms, "METRIC", "L2"):
            with self.assertRaises(vs.VectorStoreError) as cm:
                ms._to_score(0.3)
        self.assertIn("相似度", str(cm.exception))

    def test_parses_real_pymilvus_hit_objects(self):
        """直接拿 pymilvus 真的 `Hit` 对象过一遍解析，证明不是只对着自造 dict 编的。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "真 Hit")])
        res = [[RealHit({"id": "chunk:1:0", "distance": 0.75, "entity": {}}, pk_name="id")]]
        with db.conn() as c:
            out = st._collect(res, MODEL_A, 3, c)
        self.assertEqual(out, [{"paper_id": 1, "kind": "chunk", "idx": 0,
                                "score": 0.75, "text": "真 Hit"}])


# ── 5. 排序全序 ──────────────────────────────────────────────────────────────

class OrderingTests(_MilvusTestBase):
    """排序键不全序，跨进程/跨次结果就会漂——本项目踩过。
    假服务端**故意**按插入顺序返回并列行，被测代码必须自己排成全序。"""

    def test_ties_broken_by_paper_kind_idx(self):
        st = self.store()
        v = _basis(0)
        # 故意按与目标全序相反的顺序写入
        self.seed(st, [
            (9, "sent", 1, v, "p9-sent-1"),
            (9, "chunk", 0, v, "p9-chunk-0"),
            (2, "chunk", 5, v, "p2-chunk-5"),
            (2, "chunk", 1, v, "p2-chunk-1"),
        ])
        hits = st.search(v, top_k=4)
        self.assertEqual([(h["paper_id"], h["kind"], h["idx"]) for h in hits],
                         [(2, "chunk", 1), (2, "chunk", 5), (9, "chunk", 0), (9, "sent", 1)])
        for h in hits:
            self.assertAlmostEqual(h["score"], 1.0, places=5)

    def test_stable_across_repeated_runs(self):
        st = self.store()
        v = _basis(0)
        self.seed(st, [(p, "chunk", i, v, f"{p}-{i}")
                       for p in (7, 3, 5) for i in (2, 0)])
        first = st.search(v, top_k=6)
        second = self.store().search(v, top_k=6)   # 新实例也必须一致
        self.assertEqual(first, second)

    def test_score_order_dominates_key_order(self):
        st = self.store()
        self.seed(st, [
            (1, "chunk", 0, _mix(0, 1, 0.9), "弱相关但 paper_id 小"),
            (9, "chunk", 0, _basis(0), "强相关但 paper_id 大"),
        ])
        hits = st.search(_basis(0), top_k=2)
        self.assertEqual([h["paper_id"] for h in hits], [9, 1])


# ── 6. 每个模型一个 collection，互不覆盖 ─────────────────────────────────────

class ModelIsolationTests(_MilvusTestBase):
    """这是 Chroma 后端刚修过的真实缺陷：id 不含 model，两个模型撞成一条，
    **文本对、分数全错**（同向向量 score 应为 1.0，实测返回 0.0），
    而且 count 与 index_count 仍然相等，自检判据一起失效。"""

    def test_two_models_do_not_overwrite(self):
        st_a = self.store()
        self.seed(st_a, [(1, "chunk", 0, _basis(0), "A 的文本")])

        self.set_model(MODEL_B)
        st_b = self.store()
        self.seed(st_b, [(1, "chunk", 0, _basis(1), "B 的文本")])   # 同一个 id，正交向量

        self.set_model(MODEL_A)
        hits = self.store().search(_basis(0), top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5,
                               msg="模型间串了：拿 A 的查询比到了 B 的向量")
        self.assertEqual(hits[0]["text"], "A 的文本")

        self.set_model(MODEL_B)
        hits_b = self.store().search(_basis(1), top_k=5)
        self.assertAlmostEqual(hits_b[0]["score"], 1.0, places=5)
        self.assertEqual(hits_b[0]["text"], "B 的文本")

    def test_separate_collections_created(self):
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "a")])
        self.set_model(MODEL_B)
        self.seed(self.store(), [(1, "chunk", 0, _basis(1), "b")])
        self.assertEqual(sorted(self.server.collections),
                         sorted([ms.collection_name(MODEL_A), ms.collection_name(MODEL_B)]))
        for name in self.server.collections:
            self.assertEqual(len(self.server.collections[name]["rows"]), 1)

    def test_index_count_self_check_would_catch_a_collision(self):
        """count()（真相）与 index_count()（索引）是自检判据，隔离后两者仍相等。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a"), (2, "chunk", 0, _basis(1), "b")])
        self.assertEqual(st.count(), 2)
        self.assertEqual(st.index_count(), 2)


# ── 7. 维度：建 collection 时固定，不一致必须明确报错 ────────────────────────

class DimensionTests(_MilvusTestBase):

    def test_query_dimension_mismatch(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        with self.assertRaises(vs.DimensionMismatch) as cm:
            st.search(np.ones(DIM * 2, dtype=np.float32), top_k=1)
        self.assertIn("维度", str(cm.exception))

    def test_write_dimension_mismatch_at_sqlite_layer(self):
        """同一个 model 换了维度，SQLite 这一层先拦下（三个后端同一句错）。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        with self.assertRaises(vs.DimensionMismatch):
            self.seed(st, [(2, "chunk", 0, np.ones(DIM * 2, dtype=np.float32), "b")])

    def test_collection_dim_mismatch_is_not_silently_written(self):
        """真实场景：有人清了 SQLite 但没清 Milvus，再写一批新维度的向量。

        SQLite 那层此时放行（库里没这个模型的向量了），必须由本模块比对
        collection 的既有维度并报错——**绝不能静默写进去**。
        """
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.assertEqual(self.server.collections[name]["dim"], DIM)

        with db.conn() as c:                       # 只清真相，不动 Milvus
            c.execute("DELETE FROM vectors WHERE model=?", (MODEL_A,))
        vs._invalidate()

        fresh = self.store()                       # 新实例，没有维度缓存
        with self.assertRaises(vs.DimensionMismatch) as cm:
            self.seed(fresh, [(2, "chunk", 0, np.ones(DIM * 2, dtype=np.float32), "b")])
        msg = str(cm.exception)
        self.assertIn(str(DIM), msg)
        self.assertIn(str(DIM * 2), msg)
        self.assertIn("固定", msg)                  # 要说清「维度改不了」这个 Milvus 事实
        self.assertEqual(len(self.server.collections[name]["rows"]), 1,
                         "维度不一致却往 Milvus 里写了")

    def test_server_side_dim_guard(self):
        """兜底：假服务端也复刻了 Milvus 的维度校验，绕过客户端检查照样写不进去。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        with self.assertRaises(MilvusException):
            self.client_cls().upsert(collection_name=name,
                                     data=[{"id": "chunk:9:0", "vector": [0.0] * (DIM * 2),
                                            "paper_id": 9, "kind": "chunk", "idx": 0}])


# ── 8. 建了 collection ≠ 能搜：索引与 load ──────────────────────────────────

class IndexAndLoadTests(_MilvusTestBase):
    """Milvus 最常见的坑。pymilvus 3.0.1 的 `_create_collection_with_schema` 只有在
    传了 index_params 时才 create_index + load_collection；不传就是
    「建好了、能插入、一搜报 collection not loaded」。"""

    def test_create_builds_index_and_loads(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        col = self.server.collections[name]
        self.assertTrue(col["indexed"], "建 collection 时没建索引")
        self.assertTrue(col["loaded"], "建 collection 时没 load")
        self.assertEqual(col["index_type"], ms.DEFAULT_INDEX_TYPE)
        create = [c for c in self.server.calls if c[0] == "create_collection"]
        self.assertEqual(len(create), 1)
        self.assertTrue(create[0][2], "create_collection 没传 index_params")

    def test_index_type_overridable_by_env(self):
        os.environ[ms.INDEX_ENV] = "HNSW"
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.assertEqual(self.server.collections[name]["index_type"], "HNSW")

    def test_consistency_level_is_bounded_by_default(self):
        """默认 Bounded 而不是 Strong——这是**真实 Milvus v2.5.10 上量出来的**：

        | 级别    | 查询 p50 | 写后可读延迟 |
        |---------|----------|--------------|
        | Strong  | 399.9 ms | 立即         |
        | Bounded |  16.1 ms | 300 ms       |

        Strong 让每次查询都贵 25 倍，换来的只是省掉 300ms 的写后可见延迟——
        对「写一次、读上千次」的文献检索是错误的取舍。
        写后可见那个窗口由 rebuild() 末尾的 flush 关掉（见下一条用例），
        而不是让每次查询都付代价。
        """
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.assertEqual(self.server.collections[name]["consistency"], "Bounded")

    def test_search_passes_consistency_explicitly(self):
        """search 必须显式传 consistency，不能靠继承 collection 建表时的级别。

        实测踩到：只改默认值而不在 search 里传，对**已经建好**的 collection
        完全不生效——线上那个集合仍按建表时的 Strong 走，改了等于没改。
        """
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        seen = {}
        real = self.client_cls.search

        def spy(inner, *a, **kw):
            seen.update(kw)
            return real(inner, *a, **kw)

        with mock.patch.object(self.client_cls, "search", spy):
            st.search(_basis(0), 3)
        self.assertEqual(seen.get("consistency_level"), "Bounded",
                         f"search 没显式传 consistency：{sorted(seen)}")

    def test_rebuild_flushes_so_results_are_immediately_visible(self):
        """Bounded 下写入约 300ms 后才可读，但「重建完立刻查一下对不对」是最自然的
        用法，那时返回空会被当成 rebuild 坏了。所以 rebuild 末尾 flush 一次，
        把这个窗口关掉——成本只在重建时付一次，不摊到每次查询上。
        """
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        self.server.calls.clear()
        st.rebuild()
        self.assertIn("flush", [c[0] for c in self.server.calls],
                      "rebuild 结束时必须 flush，否则紧接着的检索可能读到空")

    def test_existing_collection_is_reloaded(self):
        """进程重启/服务重启/别人 release 之后，集合是未 load 状态——每次开都要补 load。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.server.collections[name]["loaded"] = False      # 模拟重启

        fresh = self.store()                                  # 新实例，无 _ready 缓存
        hits = fresh.search(_basis(0), top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertTrue(self.server.collections[name]["loaded"])
        self.assertIn(("load_collection", name), self.server.calls)

    def test_fake_really_models_the_pitfall(self):
        """夹具自证：跳过 load 就是搜不了——否则上一条测的是个不存在的坑。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.server.collections[name]["loaded"] = False
        st._ready.add(name)                                   # 骗过「每次补 load」
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            st.search(_basis(0), top_k=1)
        self.assertIn("not loaded", str(cm.exception))


# ── 9. delete_paper ────────────────────────────────────────────────────────

class DeletePaperTests(_MilvusTestBase):

    def test_deletes_target_and_spares_others(self):
        st = self.store()
        self.seed(st, [
            (1, "chunk", 0, _basis(0), "p1c0"),
            (1, "chunk", 1, _basis(1), "p1c1"),
            (1, "sent", 0, _basis(2), "p1s0"),
            (2, "chunk", 0, _basis(0), "p2c0"),
        ])
        n = st.delete_paper(1)
        self.assertEqual(n, 3)
        self.assertEqual(st.count(), 1)
        self.assertEqual(st.index_count(), 1)
        name = ms.collection_name(MODEL_A)
        left = {r["id"] for r in self.server.rows(name)}
        self.assertEqual(left, {"chunk:2:0"})
        hits = st.search(_basis(0), top_k=5)
        self.assertEqual([h["paper_id"] for h in hits], [2])
        self.assertEqual(st.last_stale_hits, 0)

    def test_cleans_every_model_collection(self):
        """SQLite 删的是**所有模型**的行；Milvus 不跟着删，别的模型集合里就永久留脏行。"""
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "a"),
                                 (2, "chunk", 0, _basis(1), "a2")])
        self.set_model(MODEL_B)
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "b"),
                                 (2, "chunk", 0, _basis(1), "b2")])

        self.store().delete_paper(1)
        for model in (MODEL_A, MODEL_B):
            name = ms.collection_name(model)
            ids = {r["id"] for r in self.server.rows(name)}
            self.assertEqual(ids, {"chunk:2:0"}, f"{model} 的集合没清干净")

    def test_does_not_touch_foreign_collections(self):
        """同一个 Milvus 上可能跑着别的东西，只动名字符合本项目形状的集合。"""
        self.seed(self.store(), [(1, "chunk", 0, _basis(0), "a")])
        self.client_cls().create_collection(
            collection_name="someone_elses_table",
            schema=self._minimal_schema(),
            index_params=self._minimal_index())
        self.client_cls().insert(
            collection_name="someone_elses_table",
            data=[{"id": "chunk:1:0", "vector": [0.0] * DIM,
                   "paper_id": 1, "kind": "chunk", "idx": 0}])
        self.store().delete_paper(1)
        self.assertEqual(len(self.server.rows("someone_elses_table")), 1)

    def test_reports_index_failure_instead_of_swallowing(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        self.server.fail_on["delete"] = MilvusException(message="rpc boom")
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            st.delete_paper(1)
        msg = str(cm.exception)
        self.assertIn("SQLite 已删除", msg)
        self.assertIn("rebuild", msg)
        self.assertEqual(st.count(), 0, "真相应当已经删掉了")

    @staticmethod
    def _minimal_schema():
        s = RealMilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        s.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        s.add_field("paper_id", DataType.INT64)
        s.add_field("kind", DataType.VARCHAR, max_length=128)
        s.add_field("idx", DataType.INT64)
        s.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
        return s

    @staticmethod
    def _minimal_index():
        ip = RealMilvusClient.prepare_index_params()
        ip.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        return ip


# ── 10. 索引落后于真相：丢弃但留痕 ──────────────────────────────────────────

class StaleHitTests(_MilvusTestBase):
    """Milvus 里查到、SQLite 里已经没有的行必须丢弃——那行确实删了。
    但**必须留痕**，否则「检索结果莫名变少」就没有任何线索。"""

    def test_rows_deleted_from_sqlite_are_dropped_and_counted(self):
        st = self.store()
        self.seed(st, [
            (1, "chunk", 0, _basis(0), "留下"),
            (2, "chunk", 0, _mix(0, 1, 0.2), "删掉 A"),
            (3, "chunk", 0, _mix(0, 1, 0.3), "删掉 B"),
        ])
        self.sql_delete(2, "chunk", 0)
        self.sql_delete(3, "chunk", 0)

        hits = st.search(_basis(0), top_k=5)
        self.assertEqual([h["paper_id"] for h in hits], [1])
        self.assertEqual(st.last_stale_hits, 2)
        self.assertEqual(st.count(), 1)
        self.assertEqual(st.index_count(), 3, "索引里还留着，正是需要 rebuild 的信号")

    def test_stale_counter_resets_each_search(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a"), (2, "chunk", 0, _basis(1), "b")])
        self.sql_delete(2, "chunk", 0)
        st.search(_basis(0), top_k=5)
        self.assertEqual(st.last_stale_hits, 1)
        st.search(_basis(0), top_k=5, where={"paper_id": 1})
        self.assertEqual(st.last_stale_hits, 0, "计数没按次重置，会一路累加")

    def test_unparseable_id_is_counted_too(self):
        """索引里混进了不是本模块写的行，同样丢弃 + 留痕。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        res = [[{"id": "垃圾主键", "distance": 0.9, "entity": {}},
                {"id": "chunk:1:0", "distance": 0.8, "entity": {}}]]
        with db.conn() as c:
            out = st._collect(res, MODEL_A, 5, c)
        self.assertEqual([h["paper_id"] for h in out], [1])
        self.assertEqual(st.last_stale_hits, 1)

    def test_missing_collection_returns_empty_and_warns(self):
        """SQLite 有向量、Milvus 连集合都没有：返回空是对的，但不能一声不吭。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        name = ms.collection_name(MODEL_A)
        self.server.collections.pop(name)
        ms._WARNED.clear()

        buf = io.StringIO()
        with redirect_stderr(buf):
            hits = self.store().search(_basis(0), top_k=3)
        self.assertEqual(hits, [])
        self.assertIn("rebuild", buf.getvalue())
        self.assertIn(name, buf.getvalue())


# ── 11. rebuild：从 SQLite 全量重放 ─────────────────────────────────────────

class RebuildTests(_MilvusTestBase):

    SPEC = [
        (1, "chunk", 0, _basis(0), "p1c0", {"section": "Method"}),
        (1, "chunk", 1, _mix(0, 1, 0.25), "p1c1", {"section": "Intro"}),
        (2, "chunk", 0, _mix(0, 2, 0.4), "p2c0", None),
        (2, "sent", 0, _basis(3), "p2s0", None),
        (3, "chunk", 0, _mix(0, 1, 0.6), "p3c0", {"section": "Method"}),
    ]

    def _seed(self, st):
        self.seed(st, [(p, k, i, v, t, m) for p, k, i, v, t, m in self.SPEC])

    def test_results_identical_before_and_after(self):
        st = self.store()
        self._seed(st)
        before = st.search(_basis(0), top_k=5)
        before_filtered = st.search(_basis(0), top_k=5, where={"section": "Method"})

        self.server.collections.clear()             # 索引整个没了
        report = st.rebuild()

        self.assertEqual(report["backend"], "milvus")
        self.assertEqual(report["indexed"], len(self.SPEC))
        self.assertEqual(report["errors"], [])
        self.assertEqual(st.search(_basis(0), top_k=5), before)
        self.assertEqual(st.search(_basis(0), top_k=5, where={"section": "Method"}),
                         before_filtered)

    def test_rebuild_drops_stale_rows(self):
        st = self.store()
        self._seed(st)
        self.sql_delete(2, "chunk", 0)
        self.assertEqual(st.index_count(), len(self.SPEC))
        st.rebuild()
        self.assertEqual(st.index_count(), st.count())
        st.search(_basis(0), top_k=9)
        self.assertEqual(st.last_stale_hits, 0)

    def test_rebuild_reindexes_and_reloads(self):
        st = self.store()
        self._seed(st)
        self.server.collections.clear()
        st.rebuild()
        col = self.server.collections[ms.collection_name(MODEL_A)]
        self.assertTrue(col["indexed"])
        self.assertTrue(col["loaded"])

    def test_progress_callback(self):
        st = self.store()
        self._seed(st)
        seen = []
        st.rebuild(progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen[-1], (len(self.SPEC), len(self.SPEC)))

    def test_rebuild_on_empty_truth_is_not_an_error(self):
        """一条向量都没有时建不出 collection（Milvus 建表必须给维度），如实返回 0。"""
        st = self.store()
        report = st.rebuild()
        self.assertEqual(report["indexed"], 0)
        self.assertEqual(report["errors"], [])

    def test_batch_failure_is_reported_not_swallowed(self):
        st = self.store()
        self._seed(st)
        self.server.collections.clear()
        self.server.fail_on["insert"] = MilvusException(message="disk full")
        report = st.rebuild()
        self.assertEqual(report["indexed"], 0)
        self.assertTrue(report["errors"])
        self.assertIn("disk full", report["errors"][0])


# ── 12. 连接失败：报错要能指出排查方向 ──────────────────────────────────────

class ConnectionFailureTests(_MilvusTestBase):

    def test_error_message_points_at_diagnosis(self):
        self.server.fail_connect = MilvusException(
            message="failed to connect to localhost:19530: connection refused")
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            self.store()
        msg = str(cm.exception)
        for needle in ("connection refused",        # 原始错误不能丢
                       ms.DEFAULT_URI,              # 当前地址
                       ms.URI_ENV,                  # 怎么改地址
                       "服务",                       # 方向 1：服务没起来
                       "Milvus Lite",               # Windows 上没有嵌入式，别白折腾
                       "http://",                   # 方向 2：地址写成本地路径的经典错
                       ms.TOKEN_ENV,                # 方向 3：鉴权
                       "load"):                     # 方向 4：没建索引/没 load
            self.assertIn(needle, msg, f"排查指引里缺 {needle!r}")

    def test_uri_from_env(self):
        os.environ[ms.URI_ENV] = "http://milvus.internal:19530"
        self.addCleanup(os.environ.pop, ms.URI_ENV, None)
        self.server.fail_connect = MilvusException(message="nope")
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            self.store()
        self.assertIn("http://milvus.internal:19530", str(cm.exception))

    def test_default_uri(self):
        os.environ.pop(ms.URI_ENV, None)
        self.assertEqual(ms.milvus_uri(), "http://localhost:19530")

    def test_search_rpc_failure_is_not_silently_empty(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        self.server.fail_on["search"] = MilvusException(message="deadline exceeded")
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            st.search(_basis(0), top_k=1)
        self.assertIn("deadline exceeded", str(cm.exception))

    def test_upsert_failure_tells_you_truth_is_already_written(self):
        st = self.store()
        self.server.fail_on["upsert"] = MilvusException(message="quota exceeded")
        with self.assertRaises(vs.VectorStoreUnavailable) as cm:
            self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        msg = str(cm.exception)
        self.assertIn("SQLite 已写入", msg)
        self.assertIn("rebuild", msg)
        self.assertEqual(st.count(), 1, "真相应当已经落地，rebuild 才能救回来")


class PymilvusMissingTests(unittest.TestCase):
    """机器上没装 pymilvus（或本机 grpc 二进制不兼容）时的报错。

    不继承 `_MilvusTestBase`，因为那里把 `_milvus` patch 掉了，这里要调真函数。
    """

    def test_import_failure_is_actionable(self):
        with mock.patch.object(ms, "_MOD", None), \
                mock.patch.dict(sys.modules, {"pymilvus": None}):
            with self.assertRaises(vs.VectorStoreUnavailable) as cm:
                ms._milvus()
        msg = str(cm.exception)
        self.assertIn("pip install pymilvus", msg)
        self.assertIn("numpy", msg)          # 要告诉人还能换哪个后端


# ── 13. where 过滤 / count / 契约杂项 ───────────────────────────────────────

class ContractTests(_MilvusTestBase):

    def _seed_mixed(self, st):
        self.seed(st, [
            (1, "chunk", 0, _basis(0), "p1c0", {"section": "Method"}),
            (1, "sent", 0, _mix(0, 1, 0.1), "p1s0", {"section": "Method"}),
            (2, "chunk", 0, _mix(0, 1, 0.2), "p2c0", {"section": "Intro"}),
            (3, "chunk", 0, _mix(0, 1, 0.3), "p3c0", None),
        ])

    def test_kind_filter(self):
        st = self.store()
        self._seed_mixed(st)
        hits = st.search(_basis(0), top_k=9, kind="sent")
        self.assertEqual([(h["paper_id"], h["kind"]) for h in hits], [(1, "sent")])

    def test_where_on_reserved_key(self):
        st = self.store()
        self._seed_mixed(st)
        hits = st.search(_basis(0), top_k=9, where={"paper_id": {"$in": [2, 3]}})
        self.assertEqual([h["paper_id"] for h in hits], [2, 3])

    def test_where_on_user_meta(self):
        st = self.store()
        self._seed_mixed(st)
        hits = st.search(_basis(0), top_k=9, where={"section": "Method"})
        self.assertEqual({(h["paper_id"], h["kind"]) for h in hits},
                         {(1, "chunk"), (1, "sent")})

    def test_kind_and_where_combined(self):
        st = self.store()
        self._seed_mixed(st)
        hits = st.search(_basis(0), top_k=9, kind="chunk", where={"section": "Method"})
        self.assertEqual([(h["paper_id"], h["kind"]) for h in hits], [(1, "chunk")])

    def test_invalid_where_rejected(self):
        st = self.store()
        with self.assertRaises(vs.VectorStoreError):
            st.search(_basis(0), top_k=1, where={"a": 1, "b": 2})

    def test_count_by_kind(self):
        st = self.store()
        self._seed_mixed(st)
        self.assertEqual(st.count(), 4)
        self.assertEqual(st.count("chunk"), 3)
        self.assertEqual(st.count("sent"), 1)

    def test_top_k_zero_and_empty_store(self):
        st = self.store()
        self.assertEqual(st.search(_basis(0), top_k=5), [])   # 库里啥也没有
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        self.assertEqual(st.search(_basis(0), top_k=0), [])

    def test_upsert_overwrites_same_key(self):
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "旧文本")])
        self.seed(st, [(1, "chunk", 0, _basis(1), "新文本")])
        self.assertEqual(st.count(), 1)
        self.assertEqual(st.index_count(), 1)
        hits = st.search(_basis(1), top_k=1)
        self.assertEqual(hits[0]["text"], "新文本")
        self.assertAlmostEqual(hits[0]["score"], 1.0, places=5)

    def test_text_comes_from_sqlite_not_milvus(self):
        """正文的权威在 SQLite。Milvus 里根本不存 text（VARCHAR 有 65535 上限，
        长 chunk 塞进去会写失败），所以这里也顺便钉死「没往索引里存正文」。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "权威文本")])
        name = ms.collection_name(MODEL_A)
        row = self.server.rows(name)[0]
        self.assertNotIn("text", row)
        with db.conn() as c:
            c.execute("UPDATE vectors SET text=? WHERE paper_id=1", ("改过的文本",))
        vs._invalidate()
        self.assertEqual(st.search(_basis(0), top_k=1)[0]["text"], "改过的文本")

    def test_meta_cannot_clash_with_schema_fields(self):
        st = self.store()
        for bad in ("id", "vector"):
            with self.subTest(key=bad):
                with self.assertRaises(vs.VectorStoreError) as cm:
                    self.seed(st, [(1, "chunk", 0, _basis(0), "a", {bad: "x"})])
                self.assertIn(bad, str(cm.exception))
        self.assertEqual(st.count(), 0, "校验必须发生在写真相之前")

    def test_reserved_meta_still_rejected_by_shared_layer(self):
        st = self.store()
        with self.assertRaises(vs.VectorStoreError):
            self.seed(st, [(1, "chunk", 0, _basis(0), "a", {"paper_id": 9})])

    def test_scalar_fields_are_declared_not_dynamic(self):
        """paper_id / kind / idx 必须是真字段：where 过滤和 delete_paper 全靠它们。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        col = self.server.collections[ms.collection_name(MODEL_A)]
        self.assertEqual(sorted(col["fields"]), ["id", "idx", "kind", "paper_id", "vector"])
        self.assertTrue(col["dynamic"], "dynamic field 关了，用户 meta 就没处放")

    def test_primary_key_encoding_matches_chroma(self):
        st = self.store()
        self.seed(st, [(7, "sent", 3, _basis(0), "a")])
        ids = {r["id"] for r in self.server.rows(ms.collection_name(MODEL_A))}
        self.assertEqual(ids, {"sent:7:3"})
        self.assertEqual(ms._parse_id("sent:7:3"), (7, "sent", 3))
        self.assertIsNone(ms._parse_id("bad"))

    def test_over_fetches_before_truncating(self):
        """ANN 只保证近似 top-k，还要留出丢 stale 行的余量，所以多取一些再全序排。"""
        st = self.store()
        self.seed(st, [(p, "chunk", 0, _mix(0, 1, p / 100.0), f"p{p}") for p in range(1, 30)])
        st.search(_basis(0), top_k=3)
        call = [c for c in self.server.calls if c[0] == "search"][-1]
        self.assertGreater(call[3], 3, "limit 没放大，边界并列会随 ANN 抖")

    def test_implements_vectorstore_protocol(self):
        st = self.store()
        self.assertIsInstance(st, vs.VectorStore)
        self.assertEqual(st.name, "milvus")
        self.assertIsNone(st.degraded)


if __name__ == "__main__":       # pragma: no cover
    unittest.main()


class WhatTheseTestsCannotAnswerTests(_MilvusTestBase):
    """把这套测试**答不了**的那个问题，写成机器可检的断言。

    上面 1200 多行 Milvus 用例全部跑在 `FakeMilvusClient` 上，而这个假服务端对每一行
    做 numpy 精确余弦、按分数排序取 limit——**没有 ANN 近似、没有 segment、
    没有一致性滞后、写入即刻可见**。也就是说：

        「换成 Milvus 会不会改变召回结果」——这套测试从结构上就回答不了。

    这不是缺陷（本机连不上任何 Milvus，Milvus Lite 在 Windows 上不存在，见文件头），
    而是**边界**。把它写成断言是为了让边界随代码一起活着：哪天有人给假服务端加了
    ANN 近似或一致性滞后，下面这两条会红——那时就该回来把「召回等价性」重新问一遍。

    在那之前对外只能说「Milvus 后端的**契约**（建集合 / 索引 / load / metric /
    维度 / 错误码）已被覆盖」，**不能**说「换后端不改变检索结果」。
    """

    def test_the_fake_is_exact_not_approximate(self):
        """真 Milvus 的 AUTOINDEX / HNSW 是近似最近邻；假服务端是精确暴力 kNN。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a"),
                       (2, "chunk", 0, _basis(1), "b"),
                       (3, "chunk", 0, _basis(2), "c")])
        got = st.search(_basis(0), top_k=3)
        self.assertEqual(got[0]["paper_id"], 1,
                         "假服务端不再是精确 kNN 了——上面那些排序断言的含义随之改变，"
                         "该重新评估「换后端会不会改变召回」这个问题")

    def test_the_fake_makes_writes_visible_immediately(self):
        """真 Milvus 在 Bounded 一致性下允许读落后写几秒；假服务端写完就能搜到。"""
        st = self.store()
        self.seed(st, [(1, "chunk", 0, _basis(0), "a")])
        self.assertEqual(len(st.search(_basis(0), top_k=1)), 1,
                         "假服务端出现了写后不可见——说明它开始模拟一致性滞后，"
                         "此时应当补一批「写后立刻读」的真实场景用例")
