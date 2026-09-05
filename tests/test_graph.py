"""引文网络模块的离线测试。

两件事必须做到：
1. **算法用手算得出的期望值验证**，不是「跑通就行」。下面 `_seed_graph` 造的小图，
   每个共被引 / 耦合 / 缺口的期望数字都在注释里写了推导过程。
2. **绝不联网**。所有 S2 调用走 FakeS2；退避 sleep 也被摁住，测试跑完是秒级。
"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from papernest import config, db, graph
from papernest.normalize import norm_key


# ── 假的 Semantic Scholar ──

_NOT_JSON = object()   # 让 FakeResponse.json() 像真 httpx 一样抛 ValueError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if self._payload is _NOT_JSON:
            # httpx 的 Response.json() 对 HTML 错误页抛的就是这个（ValueError 子类）
            raise json.JSONDecodeError("Expecting value", self.text or "", 0)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


class FakeS2:
    """按 URL 末段（references / citations）路由的假数据源。

    同一路由给多个响应时按次序消费，最后一个会一直重复——429 → 200 的退避
    用例就靠这个。跨多个 http.client() 实例共享 calls，因为 _get_json 每次
    都新开一个 client。
    """

    def __init__(self, routes: dict):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple] = []

    def client(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        seq = self.routes.get(url.rsplit("/", 1)[-1])
        if not seq:
            return FakeResponse(404, text="未配置路由")
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def n(self, direction: str) -> int:
        return sum(1 for c in self.calls if c[0].endswith("/" + direction))


def _s2_paper(pid, title, year=2020, doi=None, arxiv=None, authors=(), cites=None):
    ext = {}
    if doi:
        ext["DOI"] = doi
    if arxiv:
        ext["ArXiv"] = arxiv
    return {"paperId": pid, "title": title, "year": year, "externalIds": ext,
            "authors": [{"name": a} for a in authors], "citationCount": cites}


class GraphTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_graph_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 本模块不用向量，但开发机 .env 配了 EMBED_MODEL；一并摁住防止任何
        # 间接路径把请求打出去（与 tests/test_write_online.py 同口径）。
        self._embed_patch = mock.patch("papernest.embeddings.available",
                                       return_value=False)
        self._embed_patch.start()
        # 退避 sleep 摁住：否则一次 429 用例要真睡 20 秒
        self._sleep_patch = mock.patch("papernest.graph._sleep")
        self.sleep = self._sleep_patch.start()
        db.init_db()
        graph.ensure_schema()

    def tearDown(self):
        self._sleep_patch.stop()
        self._embed_patch.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 夹具 ──

    def _paper(self, key, title, **kw):
        with db.conn() as c:
            return db.insert_l0(c, {"norm_key": key, "title": title,
                                    "abstract": kw.get("abstract"),
                                    "year": kw.get("year", 2023),
                                    "authors": kw.get("authors", []),
                                    "doi": kw.get("doi"),
                                    "arxiv_id": kw.get("arxiv_id"),
                                    "citation_count": kw.get("citation_count"),
                                    "source": "test", "s2_id": kw.get("s2_id")})

    def _edges(self, pairs):
        with db.conn() as c:
            for s, d in pairs:
                c.execute("INSERT OR IGNORE INTO citation_edges(src_key,dst_key) "
                          "VALUES(?,?)", (s, d))

    def _node(self, key, title, year=2019, cites=None, doi=None, authors=()):
        with db.conn() as c:
            graph._upsert_node(c, {"norm_key": key, "title": title, "year": year,
                                   "authors": list(authors), "doi": doi,
                                   "arxiv_id": None, "citation_count": cites,
                                   "s2_id": None})


# ── 手工小图上的算法验证（完全不联网）──

class AlgorithmTests(GraphTestBase):
    """图结构（src 引用 dst），库内只有 A、B 两篇：

        X → A, X → C          Y → A, Y → C          Z → A, Z → D
        A → R1, A → R2, A → R3
        E → R1, E → R2        F → R1
        B → R1, B → R9
    """

    def setUp(self):
        super().setUp()
        self.pa = self._paper("arxiv:a", "Paper A", arxiv_id="a")
        self.pb = self._paper("arxiv:b", "Paper B", arxiv_id="b")
        self._edges([
            ("arxiv:x", "arxiv:a"), ("arxiv:x", "arxiv:c"),
            ("arxiv:y", "arxiv:a"), ("arxiv:y", "arxiv:c"),
            ("arxiv:z", "arxiv:a"), ("arxiv:z", "arxiv:d"),
            ("arxiv:a", "arxiv:r1"), ("arxiv:a", "arxiv:r2"), ("arxiv:a", "arxiv:r3"),
            ("arxiv:e", "arxiv:r1"), ("arxiv:e", "arxiv:r2"),
            ("arxiv:f", "arxiv:r1"),
            ("arxiv:b", "arxiv:r1"), ("arxiv:b", "arxiv:r9"),
        ])
        for k, t, cc in [("arxiv:c", "Paper C", 40), ("arxiv:d", "Paper D", 30),
                         ("arxiv:e", "Paper E", 20), ("arxiv:f", "Paper F", 10),
                         ("arxiv:r1", "Ref One", 900), ("arxiv:r2", "Ref Two", 800),
                         ("arxiv:r9", "Ref Nine", 700)]:
            self._node(k, t, cites=cc)
        # arxiv:r3 与 arxiv:x/y/z 故意不建节点缓存：验证「拿不到标题就留 None」

    # ── neighbors ──

    def test_neighbors_depth1_is_exactly_the_direct_edges(self):
        g = graph.neighbors(self.pa, depth=1, limit=50)
        self.assertEqual(g["center"], "arxiv:a")
        self.assertEqual({n["norm_key"] for n in g["nodes"]},
                         {"arxiv:a", "arxiv:x", "arxiv:y", "arxiv:z",
                          "arxiv:r1", "arxiv:r2", "arxiv:r3"})
        # 只保留两端都在子图内的边：X→C、E→R1 等不能混进来
        self.assertEqual({(e["src"], e["dst"]) for e in g["edges"]},
                         {("arxiv:x", "arxiv:a"), ("arxiv:y", "arxiv:a"),
                          ("arxiv:z", "arxiv:a"), ("arxiv:a", "arxiv:r1"),
                          ("arxiv:a", "arxiv:r2"), ("arxiv:a", "arxiv:r3")})
        self.assertFalse(g["truncated"])
        center = next(n for n in g["nodes"] if n["is_center"])
        self.assertEqual((center["depth"], center["in_library"], center["paper_id"]),
                         (0, True, self.pa))

    def test_neighbors_marks_out_of_library_and_missing_titles_honestly(self):
        g = graph.neighbors(self.pa, depth=1)
        by = {n["norm_key"]: n for n in g["nodes"]}
        self.assertFalse(by["arxiv:r1"]["in_library"])
        self.assertIsNone(by["arxiv:r1"]["paper_id"])
        self.assertEqual(by["arxiv:r1"]["title"], "Ref One")
        # 连缓存都没有的端点：标题如实留 None，节点不能丢
        self.assertIsNone(by["arxiv:r3"]["title"])
        self.assertIsNone(by["arxiv:x"]["title"])

    def test_neighbors_depth2_expands_one_more_hop(self):
        g = graph.neighbors(self.pa, depth=2, limit=100)
        self.assertEqual({n["norm_key"] for n in g["nodes"]},
                         {"arxiv:a", "arxiv:x", "arxiv:y", "arxiv:z", "arxiv:r1",
                          "arxiv:r2", "arxiv:r3",           # 第 1 跳
                          "arxiv:c", "arxiv:d", "arxiv:e", "arxiv:f", "arxiv:b"})
        by = {n["norm_key"]: n for n in g["nodes"]}
        self.assertEqual(by["arxiv:c"]["depth"], 2)
        self.assertTrue(by["arxiv:b"]["in_library"])      # B 是库内论文，2 跳外
        self.assertNotIn("arxiv:r9", by)                  # R9 要 3 跳，不该出现

    def test_neighbors_limit_truncates_and_says_so(self):
        g = graph.neighbors(self.pa, depth=1, limit=3)
        self.assertEqual(g["node_count"], 3)
        self.assertTrue(g["truncated"])
        self.assertIn("arxiv:a", {n["norm_key"] for n in g["nodes"]})

    def test_neighbors_of_isolated_paper_is_just_itself(self):
        pid = self._paper("arxiv:lonely", "Lonely Paper")
        g = graph.neighbors(pid)
        self.assertEqual(g["node_count"], 1)
        self.assertEqual(g["edges"], [])

    def test_neighbors_truncated_is_honest_when_a_whole_hop_is_dropped(self):
        """回归：limit 被上一跳「刚好」填满时，下一跳整跳被丢掉却报 truncated=False。

        depth=2 不限量是 12 个节点；depth=1 正好 7 个。limit=7 时第 2 跳的
        5 个节点（C/D/E/F/B）一个都进不来——这就是截断，必须说出来，
        否则 UI 会把残缺子图当成完整的引文网络画出去。
        """
        full = graph.neighbors(self.pa, depth=2, limit=100)
        self.assertEqual(full["node_count"], 12)
        self.assertFalse(full["truncated"])
        g = graph.neighbors(self.pa, depth=2, limit=7)
        self.assertEqual(g["node_count"], 7)
        self.assertTrue(g["truncated"])
        # 边界另一侧：limit 刚好装得下整个 2 跳子图时不能误报截断
        self.assertFalse(graph.neighbors(self.pa, depth=2, limit=12)["truncated"])
        # 深度用尽（而不是名额用尽）不算截断——那是 depth 字段的职责
        self.assertFalse(graph.neighbors(self.pa, depth=1, limit=100)["truncated"])

    # ── related：共被引 + 文献耦合 ──

    def test_related_single_seed_matches_hand_computation(self):
        """种子 A：citers(A)={X,Y,Z}(3)，refs(A)={R1,R2,R3}(3)。

        共被引 Jaccard = |共同施引| / (|citers(A)| + |citers(cand)| - |共同施引|)
          C: 施引 {X,Y}，inter=2，indeg(C)=2 → 2/(3+2-2)=2/3
          D: 施引 {Z}  ，inter=1，indeg(D)=1 → 1/(3+1-1)=1/3
        耦合 Jaccard = |共同参考| / (|refs(A)| + |refs(cand)| - |共同参考|)
          E: 共 {R1,R2}，inter=2，outdeg(E)=2 → 2/(3+2-2)=2/3
          F: 共 {R1}   ，inter=1，outdeg(F)=1 → 1/(3+1-1)=1/3
          B: 共 {R1}   ，inter=1，outdeg(B)=2 → 1/(3+2-1)=1/4
        score = (0.5*co_j + 0.5*bc_j) / 1
        """
        got = graph.related([self.pa], top_k=20)
        self.assertEqual([r["norm_key"] for r in got],
                         ["arxiv:c", "arxiv:e", "arxiv:d", "arxiv:f", "arxiv:b"])
        exp = {"arxiv:c": 0.5 * (2 / 3), "arxiv:e": 0.5 * (2 / 3),
               "arxiv:d": 0.5 * (1 / 3), "arxiv:f": 0.5 * (1 / 3),
               "arxiv:b": 0.5 * (1 / 4)}
        for r in got:
            self.assertAlmostEqual(r["score"], exp[r["norm_key"]], places=6,
                                   msg=r["norm_key"])
        by = {r["norm_key"]: r for r in got}
        self.assertEqual((by["arxiv:c"]["cocitation"], by["arxiv:c"]["coupling"]), (2, 0))
        self.assertEqual((by["arxiv:e"]["cocitation"], by["arxiv:e"]["coupling"]), (0, 2))
        self.assertEqual((by["arxiv:b"]["cocitation"], by["arxiv:b"]["coupling"]), (0, 1))

    def test_related_never_returns_the_seed_itself(self):
        got = graph.related([self.pa, self.pb], top_k=20)
        self.assertNotIn("arxiv:a", {r["norm_key"] for r in got})
        self.assertNotIn("arxiv:b", {r["norm_key"] for r in got})

    def test_related_multi_seed_averages_over_seeds(self):
        """种子 [A,B]，n_seeds=2。B 无人引用，只贡献耦合：refs(B)={R1,R9}(2)。
          E: A 侧 2/3；B 侧 inter=1,outdeg=2 → 1/(2+2-1)=1/3 → 和=1 → 0.5*1/2=0.25
          F: A 侧 1/3；B 侧 inter=1,outdeg=1 → 1/(2+1-1)=1/2 → 和=5/6 → 0.5*(5/6)/2
          C: 只有 A 侧共被引 2/3 → 0.5*(2/3)/2 = 1/6
          D: 只有 A 侧共被引 1/3 → 0.5*(1/3)/2 = 1/12
        """
        got = graph.related([self.pa, self.pb], top_k=20)
        self.assertEqual([r["norm_key"] for r in got],
                         ["arxiv:e", "arxiv:f", "arxiv:c", "arxiv:d"])
        exp = {"arxiv:e": 0.25, "arxiv:f": 0.5 * (5 / 6) / 2,
               "arxiv:c": 0.5 * (2 / 3) / 2, "arxiv:d": 0.5 * (1 / 3) / 2}
        for r in got:
            self.assertAlmostEqual(r["score"], exp[r["norm_key"]], places=6,
                                   msg=r["norm_key"])
        by = {r["norm_key"]: r for r in got}
        self.assertEqual(by["arxiv:e"]["coupling"], 3)   # A 侧 2 条 + B 侧 1 条
        self.assertEqual(by["arxiv:f"]["coupling"], 2)

    def test_related_reason_states_which_signal_and_how_many_hits(self):
        by = {r["norm_key"]: r for r in graph.related([self.pa], top_k=20)}
        self.assertIn("共被引 2 次", by["arxiv:c"]["reason"])
        self.assertNotIn("文献耦合", by["arxiv:c"]["reason"])
        self.assertIn("文献耦合 2 次", by["arxiv:e"]["reason"])
        self.assertNotIn("共被引", by["arxiv:e"]["reason"])

    def test_related_reports_in_library_flag(self):
        by = {r["norm_key"]: r for r in graph.related([self.pa], top_k=20)}
        self.assertTrue(by["arxiv:b"]["in_library"])
        self.assertEqual(by["arxiv:b"]["paper_id"], self.pb)
        self.assertFalse(by["arxiv:c"]["in_library"])
        self.assertIsNone(by["arxiv:c"]["paper_id"])

    def test_related_top_k_truncates_by_score(self):
        got = graph.related([self.pa], top_k=2)
        self.assertEqual([r["norm_key"] for r in got], ["arxiv:c", "arxiv:e"])

    def test_related_empty_and_unknown_inputs(self):
        self.assertEqual(graph.related([], top_k=5), [])
        with self.assertRaises(graph.GraphError):
            graph.related([99999])

    def test_top_k_zero_or_negative_returns_nothing(self):
        """回归：`max(1, top_k)` 让「要 0 条」变成「给 1 条」。
        gap_papers 更要命——负数一路进 SQL 的话 `LIMIT -1` 在 SQLite 里是「不限」。"""
        self.assertEqual(graph.related([self.pa], top_k=0), [])
        self.assertEqual(graph.related([self.pa], top_k=-1), [])
        self.assertEqual(graph.gap_papers(top_k=0), [])
        self.assertEqual(graph.gap_papers(top_k=-1), [])
        self.assertEqual(len(graph.gap_papers(top_k=1)), 1)   # 正数仍然照常

    def test_chunking_over_400_keys_sums_correctly(self):
        """_CHUNK=400：500 个施引论文会被切成两片，跨片累加必须还原成整表结果。
        （高被引论文的 citers 轻松过 400，这条路径在原测试里一次都没走过。）"""
        pid = self._paper("arxiv:hot", "Hot Paper")
        with db.conn() as c:
            for i in range(500):
                c.execute("INSERT INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                          (f"arxiv:cit{i}", "arxiv:hot"))
                c.execute("INSERT INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                          (f"arxiv:cit{i}", "arxiv:twin"))
        got = graph.related([pid], top_k=3)
        twin = next(r for r in got if r["norm_key"] == "arxiv:twin")
        # citers(hot)=citers(twin)=同一批 500 篇 → Jaccard = 500/500 = 1.0
        self.assertEqual(twin["cocitation"], 500)
        self.assertAlmostEqual(twin["score"], 0.5, places=9)
        g = graph.neighbors(pid, depth=1, limit=2000)
        self.assertEqual((g["node_count"], g["edge_count"]), (501, 500))
        self.assertFalse(g["truncated"])

    # ── gap_papers ──

    def test_gap_papers_ranked_by_in_library_citers(self):
        """库内是 A、B。它们引到的库外文献：
             R1 被 A、B 各引一次 → 2；R2/R3（A）与 R9（B）各 → 1。
           X/Y/Z 引用了 A，但它们自己不是库内论文的被引对象，不算缺口。"""
        gaps = graph.gap_papers(top_k=10)
        self.assertEqual([(g["norm_key"], g["cited_by_count"]) for g in gaps],
                         [("arxiv:r1", 2), ("arxiv:r2", 1),
                          ("arxiv:r3", 1), ("arxiv:r9", 1)])
        r1 = gaps[0]
        self.assertEqual(sorted(r1["cited_by_titles"]), ["Paper A", "Paper B"])
        self.assertEqual(r1["title"], "Ref One")
        self.assertTrue(r1["has_metadata"])
        self.assertFalse(r1["in_library"])

    def test_gap_papers_admits_when_metadata_is_missing(self):
        r3 = next(g for g in graph.gap_papers(top_k=10) if g["norm_key"] == "arxiv:r3")
        self.assertIsNone(r3["title"])        # 不编标题
        self.assertFalse(r3["has_metadata"])

    def test_gap_papers_drops_a_paper_once_it_is_in_library(self):
        self.assertIn("arxiv:r1", {g["norm_key"] for g in graph.gap_papers()})
        graph.adopt("arxiv:r1")
        self.assertNotIn("arxiv:r1", {g["norm_key"] for g in graph.gap_papers()})

    def test_gap_papers_top_k(self):
        self.assertEqual(len(graph.gap_papers(top_k=2)), 2)

    # ── stats ──

    def test_stats_counts(self):
        s = graph.stats()
        self.assertEqual(s["edges"], 14)
        self.assertEqual(s["papers_total"], 2)
        self.assertEqual(s["papers_with_edges"], 2)
        self.assertEqual(s["external_nodes"], 7)   # 建了缓存的 7 个库外节点
        self.assertEqual(s["gap_papers"], 4)
        self.assertEqual(s["coverage"], 1.0)

    # ── adopt ──

    def test_adopt_creates_l0_and_backfills_edge_ids(self):
        out = graph.adopt("arxiv:r1")
        self.assertEqual(out["status"], "adopted")
        self.assertEqual(out["title"], "Ref One")
        self.assertEqual(out["level"], 0)
        with db.conn() as c:
            row = db.get_by_norm_key(c, "arxiv:r1")
            self.assertEqual(row["level"], 0)
            self.assertIsNone(row["abstract"])      # 引文端点没有摘要，不编
            self.assertEqual(row["source"], "graph")
            self.assertEqual(row["citation_count"], 900)
            ids = [r["dst_paper_id"] for r in c.execute(
                "SELECT dst_paper_id FROM citation_edges WHERE dst_key='arxiv:r1'")]
        # 指向 R1 的边共 4 条：A→R1、E→R1、F→R1、B→R1，全部回填
        self.assertEqual(ids, [out["paper_id"]] * 4)
        self.assertEqual(out["edges_backfilled"], 4)

    def test_adopt_then_in_library_flips_true_in_graph_views(self):
        pid = graph.adopt("arxiv:r1")["paper_id"]
        by = {n["norm_key"]: n for n in graph.neighbors(self.pa, depth=1)["nodes"]}
        self.assertTrue(by["arxiv:r1"]["in_library"])
        self.assertEqual(by["arxiv:r1"]["paper_id"], pid)

    def test_adopt_existing_paper_only_backfills(self):
        out = graph.adopt("arxiv:b")
        self.assertEqual(out["status"], "exists")
        self.assertEqual(out["paper_id"], self.pb)
        self.assertEqual(out["edges_backfilled"], 2)   # B→R1、B→R9 的 src_paper_id

    def test_adopt_unknown_key_raises(self):
        with self.assertRaises(graph.GraphError):
            graph.adopt("arxiv:r3")          # 只当过边端点，没有元数据缓存
        with self.assertRaises(graph.GraphError):
            graph.adopt("doi:nope")

    def test_adopt_refuses_node_without_title(self):
        self._node("arxiv:untitled", None)
        with self.assertRaises(graph.GraphError):
            graph.adopt("arxiv:untitled")

    # ── 库外论文后来正常入库：norm_key 自动对上号 ──

    def test_external_node_recognised_after_normal_ingest(self):
        """不走 adopt，走正常检索入库的路径（db.insert_l0），
        只要 norm_key 一致，图立刻就认它在库里——这是「边用 norm_key」的全部意义。"""
        key = norm_key(arxiv_id="c")
        self.assertEqual(key, "arxiv:c")
        before = {n["norm_key"]: n for n in graph.neighbors(self.pa, depth=2)["nodes"]}
        self.assertFalse(before["arxiv:c"]["in_library"])
        pid = self._paper(key, "Paper C from search", arxiv_id="c")
        after = {n["norm_key"]: n for n in graph.neighbors(self.pa, depth=2)["nodes"]}
        self.assertTrue(after["arxiv:c"]["in_library"])
        self.assertEqual(after["arxiv:c"]["paper_id"], pid)
        self.assertEqual(after["arxiv:c"]["title"], "Paper C from search")  # 库内优先
        # related 也跟着变
        by = {r["norm_key"]: r for r in graph.related([self.pa])}
        self.assertTrue(by["arxiv:c"]["in_library"])

    def test_corrupt_authors_json_degrades_to_empty_not_crash(self):
        """库里的 authors 列被写坏过（历史数据 / 手工改库）时，读图不能整个炸掉；
        如实降级成空作者列表，也绝不编一个出来。"""
        with db.conn() as c:
            c.execute("UPDATE papers SET authors='{坏JSON' WHERE id=?", (self.pa,))
            c.execute("UPDATE citation_nodes SET authors_json='nope' "
                      "WHERE norm_key='arxiv:r1'")
        by = {n["norm_key"]: n for n in graph.neighbors(self.pa, depth=1)["nodes"]}
        self.assertEqual(by["arxiv:a"]["authors"], [])
        self.assertEqual(by["arxiv:r1"]["authors"], [])
        self.assertEqual(by["arxiv:r1"]["title"], "Ref One")   # 其余字段不受影响

    def test_backfill_ids_is_optional_optimisation(self):
        """夹具的边是裸插的（两个 *_paper_id 全空），全量回填会补齐所有库内端点：
           src 侧 A→R1/R2/R3 与 B→R1/R9 共 5 条，dst 侧 X/Y/Z→A 与 →R1 的 4 条共 7 条。"""
        pid = self._paper("arxiv:r1", "Ref One ingested", arxiv_id="r1")
        self.assertEqual(graph.backfill_ids(), 12)
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM citation_edges "
                          "WHERE dst_key='arxiv:r1' AND dst_paper_id=?",
                          (pid,)).fetchone()["n"]
            nulls = c.execute("SELECT COUNT(*) n FROM citation_edges e "
                              "WHERE EXISTS(SELECT 1 FROM papers p "
                              "WHERE p.norm_key=e.src_key) AND e.src_paper_id IS NULL"
                              ).fetchone()["n"]
        self.assertEqual(n, 4)
        self.assertEqual(nulls, 0)
        self.assertEqual(graph.backfill_ids(), 0)   # 幂等：再跑一次没得可补


# ── schema ──

class SchemaTests(GraphTestBase):
    def test_ensure_schema_is_idempotent(self):
        graph.ensure_schema()
        graph.ensure_schema()
        with db.conn() as c:
            names = {r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertLessEqual({"citation_edges", "citation_nodes", "citation_fetch"},
                             names)

    def test_self_edge_rejected_at_db_level(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            with db.conn() as c:
                c.execute("INSERT INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                          ("arxiv:a", "arxiv:a"))

    def test_dst_key_index_exists(self):
        with db.conn() as c:
            idx = {r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertIn("idx_cedges_dst", idx)

    def test_ensure_schema_is_memoised_per_db_path(self):
        """每个公开函数入口都调 ensure_schema，原来每次都多开一条连接跑一遍
        executescript（db.py 的注释里把这个反模式当成踩过的坑记着）。
        守卫要按 DB 路径记，换库时必须重新建表。"""
        with mock.patch.object(graph.db, "conn", wraps=graph.db.conn) as spy:
            graph.ensure_schema()
            graph.ensure_schema()
        self.assertEqual(spy.call_count, 0)          # 已经建过就一条连接都不开

        old = config.DB_PATH
        try:                                          # 换一个库 → 必须真的建表
            config.DB_PATH = Path(self._tmp) / "other.db"
            graph.ensure_schema()
            with db.conn() as c:
                names = {r["name"] for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("citation_edges", names)
        finally:
            config.DB_PATH = old

    def test_ensure_schema_force_rebuilds(self):
        graph.ensure_schema()
        with db.conn() as c:
            c.execute("DROP TABLE citation_edges")
        graph.ensure_schema(force=True)
        with db.conn() as c:
            names = {r["name"] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("citation_edges", names)


# ── fetch_edges：全部走 FakeS2，绝不联网 ──

REF_PAYLOAD = {"data": [
    {"citedPaper": _s2_paper("s2-r1", "Attention Is All You Need", 2017,
                             doi="10.1/attn", authors=["Vaswani"], cites=100000)},
    {"citedPaper": _s2_paper("s2-self", "Center Paper", 2024, arxiv="2401.00001")},
    {"citedPaper": _s2_paper("s2-blank", "", 2019)},
    {"citedPaper": None},
]}
CIT_PAYLOAD = {"data": [
    {"citingPaper": _s2_paper("s2-c1", "A Later Survey", 2025,
                              arxiv="2505.00002", authors=["Zhang"], cites=3)},
]}


class FetchTests(GraphTestBase):
    def setUp(self):
        super().setUp()
        self.pid = self._paper("arxiv:2401.00001", "Center Paper",
                               arxiv_id="2401.00001", s2_id="s2-center")
        self.fake = FakeS2({"references": [FakeResponse(200, REF_PAYLOAD)],
                            "citations": [FakeResponse(200, CIT_PAYLOAD)]})

    def _run(self, fake=None, **kw):
        f = fake or self.fake
        with mock.patch("papernest.http.client", new=f.client):
            return graph.fetch_edges(self.pid, **kw)

    def test_calls_correct_urls_and_params(self):
        out = self._run(limit=50)
        urls = [c[0] for c in self.fake.calls]
        self.assertEqual(urls, [
            "https://api.semanticscholar.org/graph/v1/paper/s2-center/references",
            "https://api.semanticscholar.org/graph/v1/paper/s2-center/citations"])
        self.assertEqual(self.fake.calls[0][1],
                         {"fields": "title,year,authors,externalIds,citationCount",
                          "limit": 50})
        self.assertTrue(out["ok"])

    def test_edges_and_nodes_persisted(self):
        out = self._run()
        self.assertEqual(out["edges_added"], 2)
        with db.conn() as c:
            edges = {(r["src_key"], r["dst_key"]) for r in
                     c.execute("SELECT src_key, dst_key FROM citation_edges")}
            node = c.execute("SELECT * FROM citation_nodes WHERE norm_key='doi:10.1/attn'"
                             ).fetchone()
        self.assertEqual(edges, {("arxiv:2401.00001", "doi:10.1/attn"),
                                 ("arxiv:2505.00002", "arxiv:2401.00001")})
        self.assertEqual(node["title"], "Attention Is All You Need")
        self.assertEqual(node["year"], 2017)
        self.assertEqual(json.loads(node["authors_json"]), ["Vaswani"])
        self.assertEqual(node["citation_count"], 100000)
        self.assertEqual(node["s2_id"], "s2-r1")

    def test_self_citation_rejected_and_counted(self):
        out = self._run()
        ref = out["directions"]["references"]
        self.assertEqual(ref["self_loops"], 1)
        self.assertEqual(ref["no_key"], 2)      # 空标题 + null 条目
        self.assertEqual(ref["edges_added"], 1)
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM citation_edges "
                          "WHERE src_key=dst_key").fetchone()["n"]
        self.assertEqual(n, 0)

    def test_single_direction_only(self):
        out = self._run(direction="references")
        self.assertEqual(self.fake.n("citations"), 0)
        self.assertEqual(list(out["directions"]), ["references"])

    def test_bad_direction_raises(self):
        with self.assertRaises(graph.GraphError):
            graph.fetch_edges(self.pid, direction="sideways")

    def test_refetch_skipped_within_max_age(self):
        self._run()
        n_before = len(self.fake.calls)
        out = self._run(max_age_days=30)
        self.assertEqual(len(self.fake.calls), n_before)      # 一次网络请求都没多发
        for d in ("references", "citations"):
            self.assertEqual(out["directions"][d]["status"], "skipped")
            self.assertIsNotNone(out["directions"][d]["age_days"])
        self.assertEqual(out["edges_added"], 0)

    def test_max_age_zero_forces_refetch(self):
        self._run()
        out = self._run(max_age_days=0)
        self.assertEqual(self.fake.n("references"), 2)
        self.assertEqual(out["directions"]["references"]["status"], "fetched")
        self.assertEqual(out["edges_added"], 0)   # 边已存在，重拉不重复计数

    def test_empty_response_still_records_the_attempt(self):
        """零参考文献也要记「拉过了」，否则下次还会白跑一趟网络。"""
        fake = FakeS2({"references": [FakeResponse(200, {"data": []})],
                       "citations": [FakeResponse(200, {"data": []})]})
        self._run(fake=fake)
        out = self._run(fake=fake)
        self.assertEqual(out["directions"]["references"]["status"], "skipped")
        self.assertEqual(fake.n("references"), 1)

    def test_429_backs_off_then_succeeds(self):
        fake = FakeS2({"references": [FakeResponse(429, text="rate limited"),
                                      FakeResponse(200, REF_PAYLOAD)],
                       "citations": [FakeResponse(200, CIT_PAYLOAD)]})
        out = self._run(fake=fake)
        self.assertEqual(fake.n("references"), 2)
        self.assertEqual(out["directions"]["references"]["status"], "fetched")
        self.assertEqual(out["directions"]["references"]["edges_added"], 1)
        self.assertTrue(out["ok"])
        # 退避基数 20s（±20% 抖动）——秒级退避会全落回同一个 5 分钟封锁窗口
        self.assertEqual(self.sleep.call_count, 1)
        self.assertTrue(16 <= self.sleep.call_args[0][0] <= 24)

    def test_persistent_5xx_fails_honestly_without_killing_other_direction(self):
        fake = FakeS2({"references": [FakeResponse(503, text="down")],
                       "citations": [FakeResponse(200, CIT_PAYLOAD)]})
        out = self._run(fake=fake)
        self.assertFalse(out["ok"])
        ref = out["directions"]["references"]
        self.assertEqual(ref["status"], "failed")
        self.assertIn("503", ref["error"])
        self.assertEqual(fake.n("references"), 4)             # 共尝试 4 次
        # 睡 3 次不是 4 次：最后一次尝试之后没有下一次请求了，再睡 150s 纯属
        # 白锁调用方（both 方向就是 5 分钟变 10 分钟）。间隔仍是 20/45/90s。
        self.assertEqual(self.sleep.call_count, 3)
        lows, highs = (16, 36, 72), (24, 54, 108)             # 20/45/90 ±20%
        for got, lo, hi in zip((c[0][0] for c in self.sleep.call_args_list),
                               lows, highs):
            self.assertTrue(lo <= got <= hi, f"退避 {got}s 不在 [{lo},{hi}]")
        self.assertEqual(out["directions"]["citations"]["status"], "fetched")
        # 失败的方向不写抓取日志，下次还会重试
        with db.conn() as c:
            got = {r["direction"] for r in
                   c.execute("SELECT direction FROM citation_fetch")}
        self.assertEqual(got, {"citations"})

    def test_html_error_page_with_200_fails_only_that_direction(self):
        """回归：网关插页 / HTML 错误页带着 200 回来时 r.json() 抛 ValueError，
        它既不是 httpx.HTTPError 也不是 GraphError——原来会一路穿出 fetch_edges，
        citations 方向连请求都没发出去就被一起掀翻了。"""
        fake = FakeS2({"references": [FakeResponse(
                           200, _NOT_JSON, text="<html>502 Bad Gateway</html>")],
                       "citations": [FakeResponse(200, CIT_PAYLOAD)]})
        out = self._run(fake=fake)                    # 不抛异常
        self.assertFalse(out["ok"])
        ref = out["directions"]["references"]
        self.assertEqual(ref["status"], "failed")
        self.assertIn("不是 JSON", ref["error"])
        self.assertIn("502 Bad Gateway", ref["error"])          # 原文如实带出
        self.assertEqual(out["directions"]["citations"]["status"], "fetched")
        self.assertEqual(out["directions"]["citations"]["edges_added"], 1)
        with db.conn() as c:                          # 失败方向不写抓取日志
            got = {r["direction"] for r in
                   c.execute("SELECT direction FROM citation_fetch")}
        self.assertEqual(got, {"citations"})

    def test_unexpected_json_shape_fails_without_guessing(self):
        """payload 是数组、或 data 不是数组：不猜结构，如实记为失败。"""
        fake = FakeS2({"references": [FakeResponse(200, ["not", "a", "dict"])],
                       "citations": [FakeResponse(200, {"data": {"oops": 1}})]})
        out = self._run(fake=fake)
        self.assertFalse(out["ok"])
        self.assertIn("list", out["directions"]["references"]["error"])
        self.assertEqual(out["directions"]["citations"]["status"], "failed")
        self.assertIn("dict", out["directions"]["citations"]["error"])
        with db.conn() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) n FROM citation_edges"
                                       ).fetchone()["n"], 0)

    def test_bad_limit_raises_graph_error_before_any_request(self):
        """回归：int(limit) 原本埋在 _fetch_one 的 try 之外，limit='abc' 会以
        ValueError 穿出去，而且是在第一个方向已经打过网络之后才炸。"""
        for bad in ("abc", None, [10]):
            with self.subTest(limit=bad):
                with mock.patch("papernest.http.client", new=self.fake.client):
                    with self.assertRaises(graph.GraphError):
                        graph.fetch_edges(self.pid, limit=bad)
        self.assertEqual(self.fake.calls, [])         # 一个请求都不该发出去

    def test_400_fails_immediately_without_backoff(self):
        fake = FakeS2({"references": [FakeResponse(400, text="bad id")],
                       "citations": [FakeResponse(200, CIT_PAYLOAD)]})
        out = self._run(fake=fake)
        self.assertEqual(fake.n("references"), 1)
        self.assertEqual(self.sleep.call_count, 0)
        self.assertIn("400", out["directions"]["references"]["error"])

    def test_connection_error_retries_then_fails_that_direction_only(self):
        """传输层异常（DNS 挂了 / 代理没开）：退避重试后如实失败，不掀翻另一个方向。"""
        class Boom(FakeS2):
            def get(self, url, params=None, headers=None):
                self.calls.append((url, dict(params or {}), dict(headers or {})))
                if url.endswith("/references"):
                    raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")
                return FakeResponse(200, CIT_PAYLOAD)

        fake = Boom({})
        out = self._run(fake=fake)
        self.assertFalse(out["ok"])
        self.assertEqual(out["directions"]["references"]["status"], "failed")
        self.assertIn("getaddrinfo", out["directions"]["references"]["error"])
        self.assertEqual(fake.n("references"), 4)
        self.assertEqual(self.sleep.call_count, 3)
        self.assertEqual(out["directions"]["citations"]["status"], "fetched")

    def test_404_says_s2_does_not_know_this_paper(self):
        fake = FakeS2({"citations": [FakeResponse(200, CIT_PAYLOAD)]})  # 无 references 路由
        out = self._run(fake=fake)
        self.assertEqual(out["directions"]["references"]["status"], "failed")
        self.assertIn("404", out["directions"]["references"]["error"])
        self.assertEqual(self.sleep.call_count, 0)       # 404 不退避
        self.assertEqual(out["directions"]["citations"]["status"], "fetched")

    def test_refetch_does_not_wipe_previously_cached_fields(self):
        """第二次抓取时 S2 少给了年份/作者，缓存里已有的不能被抹成 NULL。"""
        self._run()
        thin = {"data": [{"citedPaper": {"paperId": None, "title": None,
                                         "externalIds": {"DOI": "10.1/attn"},
                                         "authors": [], "year": None,
                                         "citationCount": None}}]}
        fake = FakeS2({"references": [FakeResponse(200, thin)],
                       "citations": [FakeResponse(200, {"data": []})]})
        self._run(fake=fake, max_age_days=0)
        with db.conn() as c:
            n = c.execute("SELECT * FROM citation_nodes WHERE norm_key='doi:10.1/attn'"
                          ).fetchone()
        self.assertEqual(n["title"], "Attention Is All You Need")
        self.assertEqual(n["year"], 2017)
        self.assertEqual(json.loads(n["authors_json"]), ["Vaswani"])
        self.assertEqual(n["citation_count"], 100000)
        self.assertEqual(n["s2_id"], "s2-r1")

    def test_id_preference_s2_then_doi_then_arxiv(self):
        with db.conn() as c:
            row = lambda k: c.execute("SELECT * FROM papers WHERE norm_key=?",
                                      (k,)).fetchone()
            self._paper("doi:10.9/only", "DOI only", doi="10.9/only")
            self._paper("arxiv:9999.1", "arXiv only", arxiv_id="9999.1")
        with db.conn() as c:
            self.assertEqual(graph.s2_ref_id(
                c.execute("SELECT * FROM papers WHERE id=?", (self.pid,)).fetchone()),
                "s2-center")
            self.assertEqual(graph.s2_ref_id(
                c.execute("SELECT * FROM papers WHERE norm_key='doi:10.9/only'"
                          ).fetchone()), "DOI:10.9/only")
            self.assertEqual(graph.s2_ref_id(
                c.execute("SELECT * FROM papers WHERE norm_key='arxiv:9999.1'"
                          ).fetchone()), "arXiv:9999.1")

    def test_paper_without_any_identifier_raises_not_guesses(self):
        pid = self._paper("title:deadbeefdeadbeef", "Uploaded PDF")
        with mock.patch("papernest.http.client", new=self.fake.client):
            with self.assertRaises(graph.GraphError) as cm:
                graph.fetch_edges(pid)
        self.assertIn("拉不了引文", str(cm.exception))
        self.assertEqual(self.fake.calls, [])     # 没有外部标识就不该发请求

    def test_unknown_paper_id_raises(self):
        with self.assertRaises(graph.GraphError):
            graph.fetch_edges(123456)

    def test_api_key_header_sent_only_when_configured(self):
        with mock.patch.object(config, "S2_API_KEY", "secret-key"):
            self._run(direction="references")
        self.assertEqual(self.fake.calls[0][2], {"x-api-key": "secret-key"})
        fake2 = FakeS2({"references": [FakeResponse(200, REF_PAYLOAD)]})
        with mock.patch.object(config, "S2_API_KEY", ""):
            self._run(fake=fake2, direction="references", max_age_days=0)
        self.assertEqual(fake2.calls[0][2], {})

    def test_fetch_then_read_the_graph_end_to_end(self):
        """抓取 → neighbors → adopt 的完整闭环（依旧零联网）。"""
        self._run()
        g = graph.neighbors(self.pid, depth=1)
        self.assertEqual({n["norm_key"] for n in g["nodes"]},
                         {"arxiv:2401.00001", "doi:10.1/attn", "arxiv:2505.00002"})
        gaps = graph.gap_papers()
        self.assertEqual([g["norm_key"] for g in gaps], ["doi:10.1/attn"])
        self.assertEqual(gaps[0]["cited_by_titles"], ["Center Paper"])
        out = graph.adopt("doi:10.1/attn")
        self.assertEqual(out["status"], "adopted")
        self.assertEqual(graph.gap_papers(), [])
        s = graph.stats()
        self.assertEqual((s["edges"], s["papers_fetched"], s["gap_papers"]), (2, 1, 0))


if __name__ == "__main__":
    unittest.main()
