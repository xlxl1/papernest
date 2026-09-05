"""章节块向量化：待办清单、成本估算、分批嵌入、失败隔离、覆盖率。

夹具形态贴着**真实库**来（本项目吃过三次"合成夹具全绿、真实数据全坏"的亏）：
真实库 784 块 / 45 篇，长度 13~11924 字符、均值 1648，其中 4 条超过 6000 字，
section_path 有 "3 EVALUATION > 3.1 Benchmarks" 这样的层级，kind 分 text / table。
所以这里也放一条 13 字符的碎块、一条 8000 字的超长块、一条表格块和带层级的路径——
这些正是把 truncated 计数、空块过滤、排序全序打穿的形状。

**关于假向量**：`_fake_vec` 是 sha256 定种的高斯向量。这在本文件是成立的，
因为这里测的是**记账与写库对齐**（哪一块配哪条向量、写没写重、失败批次回没回滚），
不是 ANN 召回质量——召回类实验必须用真实分布的向量（随机高斯在 1024 维下彼此
近乎等距，会得出 recall@5≈0.56 的假结论），那类实验不在本文件。
"""
import shutil
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papernest import chunkembed, config, db, embeddings

DIM = 1024          # 与真实库一致（qwen3.7-text-embedding 是 1024 维）
MODEL = "test-embed-1024"

# 真实形态的学术正文种子：英文、有术语、有从句。中文提问打不中它的词面——
# 这正是"正文没有向量"这个缺口造成的问题，也是本轮要补的东西。
_SEED_A = ("Near-field channel estimation for extremely large-scale MIMO arrays "
           "must account for the spherical wavefront, since the planar approximation "
           "breaks down once the propagation distance falls below the Rayleigh "
           "distance of the aperture. We therefore parameterise the steering vector "
           "by both angle and range, and recover the sparse representation with an "
           "orthogonal matching pursuit variant over a polar-domain dictionary. ")
_SEED_B = ("Reconfigurable intelligent surfaces introduce a cascaded channel whose "
           "dimension grows with the number of reflecting elements, so the pilot "
           "overhead of least-squares estimation quickly becomes prohibitive. "
           "We exploit the common support across subcarriers to decouple the "
           "estimation problem into a sequence of low-dimensional subproblems. ")


def _body(n: int, seed: str) -> str:
    """把一段真实形态的正文铺到**恰好** n 个字符，让 estimate 的数字可以手算。"""
    t = (seed * (n // len(seed) + 1))[:n]
    # 末位落在空格上会被 db.replace_chunks 的 strip() 去掉，长度就对不上手算了
    return (t[:-1] + ".") if t[-1].isspace() else t


def _fake_vec(text: str, dim: int = DIM) -> list[float]:
    """按文本内容定种的确定性假向量：同文同向量、异文异向量。

    "同文同向量"才让"第 k 条向量确实是第 k 块的文本算出来的"成为可断言的事实——
    zip 错位这类 bug 只有这样才抓得住。
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    v = np.random.RandomState(seed).normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class FakeEmbedder:
    """记录每次调用的假 embedding 端点；可指定第几次调用抛异常。"""

    def __init__(self, fail_on: set[int] | None = None, short_by: int = 0):
        self.calls: list[list[str]] = []
        self.purposes: list[str] = []
        self.fail_on = fail_on or set()
        self.short_by = short_by      # 故意少返回几条，测"条数不匹配"

    def __call__(self, texts, purpose: str = "embed"):
        self.calls.append(list(texts))
        self.purposes.append(purpose)
        if len(self.calls) in self.fail_on:
            raise RuntimeError("embeddings 接口失败 HTTP 429：rate limited")
        vecs = [_fake_vec(t) for t in texts]
        return vecs[:len(vecs) - self.short_by] if self.short_by else vecs

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def batch_sizes(self) -> list[int]:
        return [len(c) for c in self.calls]


# 夹具：两篇论文 8 块。长度写死，estimate 的每个数都能手算复核。
PAPER_A_CHUNKS = [
    # (section_path, level, start_page, end_page, kind, n_chars)
    ("", 1, 1, 1, "text", 1744),                                   # 摘要之前的首块
    ("1 INTRODUCTION", 1, 1, 2, "text", 3985),
    ("2 SYSTEM MODEL", 1, 2, 2, "text", 13),                       # 真实库里就有 13 字符的碎块
    ("2 SYSTEM MODEL > 2.1 Polar-Domain Dictionary", 2, 2, 4, "text", 2733),
    ("4 EXPERIMENTS > Table 2", 2, 6, 6, "table", 812),            # 表格块
]
PAPER_B_CHUNKS = [
    ("3 METHOD > 3.2 Cascaded Estimation", 2, 3, 5, "text", 8000),  # 超过 MAX_CHARS
    ("5 CONCLUSION", 1, 8, 8, "text", 1200),
    ("References", 1, 9, 9, "text", 640),
]
TOTAL_CHARS = 19127        # 1744+3985+13+2733+812 + 8000+1200+640
EST_CHARS = 17127          # 上面减去超限那条被截掉的 2000
EST_TOKENS = 4282          # round(17127 / 4)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_chunkembed_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = MODEL
        embeddings.invalidate_cache()
        db.init_db()
        self.pa = self._add_paper(1, "Near-Field Channel Estimation for XL-MIMO",
                                  PAPER_A_CHUNKS, _SEED_A)
        self.pb = self._add_paper(2, "Cascaded Channel Estimation for RIS Systems",
                                  PAPER_B_CHUNKS, _SEED_B)

    def tearDown(self):
        embeddings.invalidate_cache()
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old

    def _add_paper(self, i: int, title: str, spec: list[tuple], seed: str) -> int:
        """走 db.insert_l0 / db.replace_chunks 真实写入路径，不手搓 INSERT。"""
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": f"arxiv:24{i:02d}.0000{i}", "title": title,
                "abstract": "We study channel estimation.", "year": 2024,
                "venue": "IEEE TWC", "authors": ["L. Wei"], "doi": None,
                "arxiv_id": f"24{i:02d}.0000{i}", "source": "s2"})
            db.replace_chunks(c, pid, [
                {"section_path": sp, "level": lv, "start_page": s, "end_page": e,
                 "kind": k, "text": _body(n, seed)}
                for sp, lv, s, e, k, n in spec])
        return pid

    def _vec_rows(self, kind: str = "chunk") -> list[sqlite3.Row]:
        with db.conn() as c:
            return c.execute("SELECT * FROM vectors WHERE kind=? "
                             "ORDER BY paper_id, idx", (kind,)).fetchall()

    def _run(self, fake: FakeEmbedder | None = None, available: bool = True, **kw):
        """在 mock 掉 available/embed_texts 的前提下跑 embed_chunks。绝不真调接口。"""
        fake = fake or FakeEmbedder()
        with mock.patch.object(embeddings, "available", return_value=available), \
             mock.patch.object(embeddings, "embed_texts", new=fake):
            return chunkembed.embed_chunks(**kw), fake


class FixtureShapeTests(_Base):
    """夹具自检：写进库的形态必须真的是我以为的形态，否则后面所有断言都是空转。"""

    def test_chunks_are_stored_with_the_expected_lengths(self):
        with db.conn() as c:
            got = [(r["paper_id"], r["chunk_no"], len(r["text"]), r["kind"])
                   for r in c.execute("SELECT paper_id,chunk_no,text,kind FROM chunks "
                                      "ORDER BY paper_id, chunk_no")]
        want = ([(self.pa, i + 1, s[5], s[4]) for i, s in enumerate(PAPER_A_CHUNKS)]
                + [(self.pb, i + 1, s[5], s[4]) for i, s in enumerate(PAPER_B_CHUNKS)])
        self.assertEqual(got, want)
        self.assertEqual(sum(g[2] for g in got), TOTAL_CHARS)

    def test_no_chunk_vectors_exist_yet(self):
        """这正是本轮要补的缺口：真实库里 784 块、0 条向量。"""
        self.assertEqual(self._vec_rows("chunk"), [])


class PendingTests(_Base):
    def test_lists_every_chunk_when_nothing_is_embedded(self):
        items = chunkembed.pending()
        self.assertEqual(len(items), 8)
        self.assertEqual({"paper_id", "chunk_no", "text", "section_path", "n_chars"},
                         set(items[0]))
        self.assertEqual([it["n_chars"] for it in items],
                         [s[5] for s in PAPER_A_CHUNKS] + [s[5] for s in PAPER_B_CHUNKS])
        self.assertEqual(items[3]["section_path"],
                         "2 SYSTEM MODEL > 2.1 Polar-Domain Dictionary")

    def test_chunk_with_a_vector_is_excluded(self):
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pa, "chunk", 2, "x", MODEL,
                       np.zeros(DIM, dtype=np.float32).tobytes()))
        left = {(it["paper_id"], it["chunk_no"]) for it in chunkembed.pending()}
        self.assertNotIn((self.pa, 2), left)
        self.assertEqual(len(left), 7)

    def test_vector_of_another_kind_does_not_mask_a_chunk(self):
        """真实库里 kind='sent' 的 idx 是 0~33，与 chunk_no 大量重叠。
        连接条件漏了 kind='chunk' 就会把这些块误判成"已完成"——一条正文向量都不会建。"""
        with db.conn() as c:
            for kind, idx in (("sent", 1), ("sent", 3), ("paper", 0)):
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,?,?,?,?,?)",
                          (self.pa, kind, idx, "x", MODEL,
                           np.zeros(DIM, dtype=np.float32).tobytes()))
        self.assertEqual(len(chunkembed.pending()), 8)

    def test_vector_of_another_model_does_not_mask_a_chunk(self):
        """换模型 = 换向量空间。拿旧模型的向量当"已完成"，检索出来的余弦没有意义。"""
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pa, "chunk", 1, "x", "some-old-embed",
                       np.zeros(DIM, dtype=np.float32).tobytes()))
        self.assertEqual(len(chunkembed.pending()), 8)
        self.assertEqual(len(chunkembed.pending(model="some-old-embed")), 7)

    def test_order_is_total_and_stable(self):
        """排序键是主键 (paper_id, chunk_no)，并列不存在——跨进程不会漂。"""
        keys = [(it["paper_id"], it["chunk_no"]) for it in chunkembed.pending()]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))
        for _ in range(3):
            self.assertEqual(
                [(it["paper_id"], it["chunk_no"]) for it in chunkembed.pending()], keys)

    def test_blank_text_chunks_are_skipped(self):
        """空块嵌不出有意义的向量，留在待办里只会让这个列表永远排不空。
        （db.replace_chunks 不写空块，所以这里绕过它直接插——防御性用例。）"""
        with db.conn() as c:
            c.execute("INSERT INTO chunks(paper_id,chunk_no,section_path,level,"
                      "start_page,end_page,kind,text) VALUES(?,?,?,?,?,?,?,?)",
                      (self.pa, 99, "", 1, 9, 9, "text", "\n \t\n"))
        self.assertNotIn((self.pa, 99),
                         [(it["paper_id"], it["chunk_no"]) for it in chunkembed.pending()])


class EstimateTests(_Base):
    def test_numbers_are_hand_checkable(self):
        est = chunkembed.estimate([{"n_chars": 100}, {"n_chars": 900}])
        self.assertEqual(est["n_chunks"], 2)
        self.assertEqual(est["total_chars"], 1000)
        self.assertEqual(est["est_chars"], 1000)
        self.assertEqual(est["n_over_limit"], 0)
        self.assertEqual(est["est_tokens"], 250)          # 1000 / 4
        self.assertIn("粗估", est["note"])

    def test_over_limit_chars_are_billed_as_truncated(self):
        """超限的部分不会被送出去，也就不该算进成本——报 8000 字的账是虚高。"""
        est = chunkembed.estimate([{"n_chars": 8000}, {"n_chars": 400}])
        self.assertEqual(est["total_chars"], 8400)
        self.assertEqual(est["est_chars"], 6400)          # 6000 + 400
        self.assertEqual(est["est_tokens"], 1600)
        self.assertEqual(est["n_over_limit"], 1)

    def test_estimate_over_the_real_pending_list(self):
        est = chunkembed.estimate(chunkembed.pending())
        self.assertEqual((est["n_chunks"], est["total_chars"], est["est_chars"],
                          est["n_over_limit"], est["est_tokens"]),
                         (8, TOTAL_CHARS, EST_CHARS, 1, EST_TOKENS))

    def test_empty_input(self):
        est = chunkembed.estimate([])
        self.assertEqual((est["n_chunks"], est["total_chars"], est["est_tokens"]),
                         (0, 0, 0))

    def test_items_without_n_chars_fall_back_to_text_length(self):
        est = chunkembed.estimate([{"text": "x" * 37}])
        self.assertEqual(est["total_chars"], 37)


class EmbedChunksTests(_Base):
    def test_batches_are_split_by_batch_size(self):
        res, fake = self._run(batch=3)
        self.assertEqual(fake.batch_sizes, [3, 3, 2])
        self.assertEqual(res["batches"], 3)
        self.assertEqual(res["embedded"], 8)
        self.assertEqual(res["skipped"], 0)
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["est_tokens"], EST_TOKENS)
        self.assertGreaterEqual(res["elapsed_s"], 0.0)

    def test_rows_are_written_with_the_right_kind_idx_and_model(self):
        self._run()
        rows = self._vec_rows("chunk")
        self.assertEqual(len(rows), 8)
        self.assertEqual([(r["paper_id"], r["idx"]) for r in rows],
                         [(self.pa, i) for i in range(1, 6)]
                         + [(self.pb, i) for i in range(1, 4)])
        self.assertTrue(all(r["kind"] == "chunk" and r["model"] == MODEL for r in rows))
        self.assertTrue(all(len(r["vec"]) // 4 == DIM for r in rows))

    def test_text_column_keeps_only_the_first_400_chars(self):
        """与现有 paper 级一致：这一列是给人看的锚点，正文回 chunks 表取，不留第二份。"""
        self._run()
        with db.conn() as c:
            row = c.execute("SELECT v.text vt, c.text ct FROM vectors v JOIN chunks c "
                            "ON c.paper_id=v.paper_id AND c.chunk_no=v.idx "
                            "WHERE v.kind='chunk' AND v.paper_id=? AND v.idx=2",
                            (self.pa,)).fetchone()
        self.assertEqual(len(row["vt"]), 400)
        self.assertEqual(row["vt"], row["ct"][:400])

    def test_each_vector_belongs_to_its_own_chunk(self):
        """抓 zip 错位：错位的向量比没有向量更糟——检索会给出张冠李戴的"证据"。"""
        self._run()
        with db.conn() as c:
            rows = c.execute("SELECT v.paper_id, v.idx, v.vec, c.text FROM vectors v "
                             "JOIN chunks c ON c.paper_id=v.paper_id AND c.chunk_no=v.idx "
                             "WHERE v.kind='chunk' ORDER BY v.paper_id, v.idx").fetchall()
        self.assertEqual(len(rows), 8)
        for r in rows:
            want = np.asarray(_fake_vec(r["text"][:chunkembed.MAX_CHARS]), dtype=np.float32)
            got = np.frombuffer(r["vec"], dtype=np.float32)
            np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_calls_are_tagged_with_a_purpose_for_cost_accounting(self):
        _, fake = self._run(batch=4)
        self.assertEqual(set(fake.purposes), {"embed_chunk"})

    def test_running_twice_does_not_embed_the_same_chunk_again(self):
        res1, fake1 = self._run()
        self.assertEqual(res1["embedded"], 8)
        res2, fake2 = self._run()
        self.assertEqual(fake2.n_calls, 0, "第二次不该再花一分钱")
        self.assertEqual((res2["embedded"], res2["batches"], res2["skipped"]), (0, 0, 0))
        self.assertEqual(len(self._vec_rows("chunk")), 8, "不能写出重复向量")

    def test_long_chunk_is_truncated_and_counted(self):
        """8000 字的块只送出 6000，且**如实计数**——不静默砍掉尾巴。"""
        res, fake = self._run(batch=16)
        sent = {len(t) for t in fake.calls[0]}
        self.assertIn(chunkembed.MAX_CHARS, sent)
        self.assertTrue(all(n <= chunkembed.MAX_CHARS for n in sent))
        self.assertEqual(res["truncated"], 1)
        self.assertEqual(res["embedded"], 8)

    def test_a_failing_batch_does_not_abort_the_rest(self):
        res, fake = self._run(FakeEmbedder(fail_on={2}), batch=3)
        self.assertEqual(fake.batch_sizes, [3, 3, 2], "失败后仍要继续跑第三批")
        self.assertEqual(res["embedded"], 5)
        self.assertEqual(res["skipped"], 3)
        self.assertEqual(res["batches"], 3)
        self.assertEqual(len(res["errors"]), 1)
        err = res["errors"][0]
        self.assertEqual(err["batch"], 2)
        self.assertEqual(err["n"], 3)
        self.assertEqual(err["chunks"], [(self.pa, 4), (self.pa, 5), (self.pb, 1)])
        self.assertIn("429", err["error"])
        # 失败批次里那条 8000 字的块没进库，所以 truncated 不该把它算上
        self.assertEqual(res["truncated"], 0)

    def test_successful_batches_survive_a_later_failure(self):
        self._run(FakeEmbedder(fail_on={2}), batch=3)
        got = {(r["paper_id"], r["idx"]) for r in self._vec_rows("chunk")}
        self.assertEqual(got, {(self.pa, 1), (self.pa, 2), (self.pa, 3),
                               (self.pb, 2), (self.pb, 3)}, "已成功的批次不该回滚")

    def test_rerun_after_failure_picks_up_exactly_what_is_missing(self):
        self._run(FakeEmbedder(fail_on={2}), batch=3)
        res, fake = self._run(batch=3)
        self.assertEqual(fake.batch_sizes, [3])
        self.assertEqual(res["embedded"], 3)
        self.assertEqual(res["truncated"], 1, "这次那条 8000 字的块进库了")
        self.assertEqual(len(self._vec_rows("chunk")), 8)

    def test_short_response_is_reported_not_stored(self):
        """条数对不上就无法保证对应关系，宁可整批丢弃也不能错位存。"""
        res, _ = self._run(FakeEmbedder(short_by=1), batch=8)
        self.assertEqual(res["embedded"], 0)
        self.assertEqual(res["skipped"], 8)
        self.assertIn("条数不匹配", res["errors"][0]["error"])
        self.assertEqual(self._vec_rows("chunk"), [])

    def test_limit_takes_the_first_n_in_total_order(self):
        res, fake = self._run(limit=4)
        self.assertEqual(res["embedded"], 4)
        self.assertEqual({(r["paper_id"], r["idx"]) for r in self._vec_rows("chunk")},
                         {(self.pa, 1), (self.pa, 2), (self.pa, 3), (self.pa, 4)})
        # 估算只覆盖被选中的 4 块：(1744+3985+13+2733)/4 = 2118.75 → 2119
        self.assertEqual(res["est_tokens"], 2119)

    def test_progress_is_called_once_per_batch_including_failures(self):
        seen: list[dict] = []
        self._run(FakeEmbedder(fail_on={2}), batch=3, progress=seen.append)
        self.assertEqual([s["batch"] for s in seen], [1, 2, 3])
        self.assertEqual([s["batches"] for s in seen], [3, 3, 3])
        self.assertEqual([s["embedded"] for s in seen], [3, 3, 5])
        self.assertEqual([s["errors"] for s in seen], [0, 1, 1])
        self.assertEqual([s["total"] for s in seen], [8, 8, 8])

    def test_a_broken_progress_callback_does_not_lose_paid_work(self):
        """钱已经花了，不能因为一个打印函数把结果丢掉——但要如实记进 errors。"""
        def boom(_info):
            raise ValueError("控制台编码炸了")

        res, _ = self._run(batch=3, progress=boom)
        self.assertEqual(res["embedded"], 8)
        self.assertEqual(len(self._vec_rows("chunk")), 8)
        self.assertEqual(len(res["errors"]), 1, "回调坏了之后不该再调，所以只记一条")
        self.assertIn("progress 回调抛异常", res["errors"][0]["error"])

    def test_dry_run_makes_zero_calls(self):
        res, fake = self._run(dry_run=True)
        self.assertEqual(fake.n_calls, 0)
        self.assertTrue(res["dry_run"])
        self.assertEqual(res["n_chunks"], 8)
        self.assertEqual(res["total_chars"], TOTAL_CHARS)
        self.assertEqual(res["est_tokens"], EST_TOKENS)
        self.assertEqual((res["embedded"], res["batches"]), (0, 0))
        self.assertEqual(res["planned_batches"], 1)
        self.assertEqual(self._vec_rows("chunk"), [])

    def test_dry_run_works_without_configuration(self):
        """估算不花钱，没配 key 也该能看——否则"要不要跑"这个决定没法做。"""
        res, fake = self._run(available=False, dry_run=True)
        self.assertEqual(fake.n_calls, 0)
        self.assertEqual(res["n_chunks"], 8)

    def test_unconfigured_raises_instead_of_silently_doing_nothing(self):
        fake = FakeEmbedder()
        with mock.patch.object(embeddings, "available", return_value=False), \
             mock.patch.object(embeddings, "embed_texts", new=fake):
            with self.assertRaises(embeddings.EmbedUnavailable) as cm:
                chunkembed.embed_chunks()
        self.assertIn("EMBED_MODEL", str(cm.exception))
        self.assertEqual(fake.n_calls, 0)
        self.assertEqual(self._vec_rows("chunk"), [])

    def test_nothing_pending_is_a_clean_no_op(self):
        self._run()
        res, fake = self._run()
        self.assertEqual((res["embedded"], res["batches"], res["est_tokens"]), (0, 0, 0))
        self.assertEqual(fake.n_calls, 0)


class ControlCharTests(_Base):
    """真实库 784 块里有 10 块含 C0 控制字符（1 块含 \\x00）——量出来的，不是想象的边界。

    合成夹具永远碰不到它：db.replace_chunks 的 strip() 去不掉 \\x00，PDF 抽取却会残留。
    JSON 里的 \\u0000 会被不少 embedding 端点判成非法入参，**整批一起失败**，
    而且每次重跑都在同一条上再失败——这一批会永远补不上。
    """

    def _add_dirty_chunk(self, text: str) -> tuple[int, int]:
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:2403.00003", "title": "Sparse Recovery Notes",
                "abstract": "a", "year": 2024, "venue": "V", "authors": [],
                "doi": None, "arxiv_id": "2403.00003", "source": "s2"})
            db.replace_chunks(c, pid, [{"section_path": "2 METHOD", "level": 1,
                                        "start_page": 1, "end_page": 1,
                                        "kind": "text", "text": text}])
        return pid, 1

    def test_control_chars_survive_the_chunk_writer(self):
        """先证明这个脏数据真的进得了库——否则后面的用例是在测一个不存在的场景。"""
        pid, no = self._add_dirty_chunk("Polar\x00domain\x0bdictionary recovery.")
        with db.conn() as c:
            got = c.execute("SELECT text FROM chunks WHERE paper_id=? AND chunk_no=?",
                            (pid, no)).fetchone()["text"]
        self.assertIn("\x00", got)

    def test_control_chars_are_stripped_before_the_request_and_counted(self):
        pid, no = self._add_dirty_chunk("Polar\x00domain\x0bdictionary recovery.")
        res, fake = self._run(batch=16)
        sent = [t for call in fake.calls for t in call]
        self.assertTrue(all("\x00" not in t and "\x0b" not in t for t in sent))
        self.assertIn("Polardomaindictionary recovery.", sent)
        self.assertEqual(res["sanitized"], 1, "清洗要计数上报，不能静默改数据")
        self.assertEqual(res["embedded"], 9)

    def test_stored_preview_is_the_cleaned_text(self):
        """存进 vectors.text 的必须是**真正送出去的那份**，否则审计对不上。"""
        pid, no = self._add_dirty_chunk("Polar\x00domain recovery.")
        self._run()
        with db.conn() as c:
            row = c.execute("SELECT text, vec FROM vectors WHERE kind='chunk' "
                            "AND paper_id=? AND idx=?", (pid, no)).fetchone()
        self.assertEqual(row["text"], "Polardomain recovery.")
        np.testing.assert_allclose(
            np.frombuffer(row["vec"], dtype=np.float32),
            np.asarray(_fake_vec("Polardomain recovery."), dtype=np.float32),
            rtol=0, atol=0)

    def test_tabs_and_newlines_are_kept(self):
        """\\t \\n \\r 是正常排版，去掉会把段落粘成一坨，反而伤语义。"""
        pid, no = self._add_dirty_chunk("Line one.\n\tLine two.\r\nLine three.")
        res, fake = self._run()
        self.assertIn("Line one.\n\tLine two.\r\nLine three.",
                      [t for call in fake.calls for t in call])
        self.assertEqual(res["sanitized"], 0)

    def test_clean_corpus_reports_zero_sanitized(self):
        res, _ = self._run()
        self.assertEqual(res["sanitized"], 0)


class CoverageTests(_Base):
    def test_zero_before_anything_is_embedded(self):
        cov = chunkembed.coverage()
        self.assertEqual(cov, {"chunks_total": 8, "chunks_embedded": 0, "coverage": 0.0,
                               "papers_with_chunks": 2, "papers_fully_embedded": 0,
                               "model": MODEL})

    def test_partial_state_is_reported_exactly(self):
        self._run(limit=5)          # 恰好是 A 篇的全部 5 块
        cov = chunkembed.coverage()
        self.assertEqual(cov["chunks_total"], 8)
        self.assertEqual(cov["chunks_embedded"], 5)
        self.assertEqual(cov["coverage"], 0.625)
        self.assertEqual(cov["papers_with_chunks"], 2)
        self.assertEqual(cov["papers_fully_embedded"], 1)

    def test_a_paper_missing_one_chunk_is_not_fully_embedded(self):
        self._run(limit=4)          # A 篇 5 块只嵌了 4 块
        self.assertEqual(chunkembed.coverage()["papers_fully_embedded"], 0)

    def test_full_coverage(self):
        self._run()
        cov = chunkembed.coverage()
        self.assertEqual((cov["chunks_embedded"], cov["coverage"],
                          cov["papers_fully_embedded"]), (8, 1.0, 2))

    def test_coverage_is_model_scoped(self):
        self._run()
        self.assertEqual(chunkembed.coverage(model="another-embed")["chunks_embedded"], 0)
        config.EMBED_MODEL = "another-embed"
        self.assertEqual(chunkembed.coverage()["coverage"], 0.0,
                         "换模型等于换向量空间，覆盖率必须归零")

    def test_duplicate_vectors_cannot_push_coverage_above_one(self):
        """并发万一写重了，统计口径也不该被带偏（EXISTS 而不是 JOIN 计数）。"""
        self._run()
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pa, "chunk", 1, "dup", MODEL,
                       np.zeros(DIM, dtype=np.float32).tobytes()))
        cov = chunkembed.coverage()
        self.assertEqual(cov["chunks_embedded"], 8)
        self.assertEqual(cov["coverage"], 1.0)


class BadVectorTests(_Base):
    """端点返回的向量本身有毛病时，宁可整批不写，也不能把坏数据放进 vectors。

    这几条都是**跨模块**的约束：`vectors` 表还被 `vectorstore` 和 `embeddings` 读，
    坏行的代价不局限在 chunk 这一路（见每条用例的说明）。
    """

    def _seed_vector(self, dim: int, model: str = MODEL):
        """在库里放一条该 model 的既有向量，制造"库内已有维度"。"""
        with db.conn() as c:
            c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                      "VALUES(?,?,?,?,?,?)",
                      (self.pa, "paper", 0, "T", model,
                       np.ones(dim, dtype=np.float32).tobytes()))

    def test_dimension_that_conflicts_with_the_store_aborts_the_run(self):
        """库内是 512 维、端点给 1024 维：混写会让 vectorstore 对**整个库**抛
        DimensionMismatch（paper/sent 一起不可用），所以一条都不能写；
        而且每批都会同样失败，必须第一批就收工——否则 49 批的钱全打水漂。"""
        self._seed_vector(512)
        res, fake = self._run(batch=3)
        self.assertEqual(res["embedded"], 0)
        self.assertEqual(self._vec_rows("chunk"), [], "维度不一致时一条都不该落库")
        self.assertEqual(fake.n_calls, 1, "撞上就停手，不该继续为同一个错误付钱")
        self.assertEqual(res["batches"], 1)
        self.assertEqual(res["planned_batches"], 3)
        self.assertIsNotNone(res["aborted"])
        self.assertIn("维度不一致", res["aborted"])
        self.assertEqual(res["skipped"], 8)

    def test_matching_dimension_is_not_blocked(self):
        """守卫不能过严：库内已有的就是 1024 维，本次也是 1024 维，必须照常跑完。"""
        self._seed_vector(DIM)
        res, _ = self._run()
        self.assertEqual(res["embedded"], 8)
        self.assertIsNone(res["aborted"])

    def test_another_models_dimension_is_not_the_baseline(self):
        """别的 model 的 512 维向量与本次无关，不该拿它当基准把人挡住。"""
        self._seed_vector(512, model="some-other-embed")
        res, _ = self._run()
        self.assertEqual(res["embedded"], 8)
        self.assertIsNone(res["aborted"])

    def test_nan_vectors_are_discarded_not_stored(self):
        """NaN 会**静默**毁掉排序：归一化后整行 NaN，检索分数直接变 nan
        （vectorstore 实测返回 [1.0, nan, 1.0, ...]），比没有向量更糟。"""
        class NaNEmbedder(FakeEmbedder):
            def __call__(self, texts, purpose="embed"):
                vecs = super().__call__(texts, purpose)
                if len(self.calls) == 2:
                    vecs[1] = [float("nan")] * DIM
                return vecs

        res, fake = self._run(NaNEmbedder(), batch=3)
        self.assertEqual(fake.batch_sizes, [3, 3, 2], "坏批之后仍要继续跑")
        self.assertEqual(res["embedded"], 5, "只丢坏的那一批")
        self.assertIn("NaN", res["errors"][0]["error"])
        with db.conn() as c:
            rows = c.execute("SELECT vec FROM vectors WHERE kind='chunk'").fetchall()
        self.assertTrue(all(np.isfinite(np.frombuffer(r["vec"], dtype=np.float32)).all()
                            for r in rows), "库里不该出现 NaN 向量")

    def test_inf_vectors_are_discarded_too(self):
        class InfEmbedder(FakeEmbedder):
            def __call__(self, texts, purpose="embed"):
                vecs = super().__call__(texts, purpose)
                vecs[0] = [1e40] * DIM        # float64 → float32 溢出成 inf
                return vecs

        res, _ = self._run(InfEmbedder(), batch=8)
        self.assertEqual(res["embedded"], 0)
        self.assertEqual(self._vec_rows("chunk"), [])


class BlankAfterSanitizeTests(_Base):
    """清洗后变成空串的块**一条都不能送**。

    OpenAI 兼容端点把空字符串判成非法入参，而入参非法是**整个请求**失败——
    一条全是控制字符的块能把同批 15 条正常块一起拖下水，且每次重跑都在同一批上再失败。
    这正是 `_sanitize` 想避免的「永远补不上的一批」，不能在清洗这一步自己造出来。
    """

    def _add_chunk(self, text: str) -> tuple[int, int]:
        with db.conn() as c:
            c.execute("INSERT INTO chunks(paper_id,chunk_no,section_path,level,"
                      "start_page,end_page,kind,text) VALUES(?,?,?,?,?,?,?,?)",
                      (self.pa, 90, "", 1, 9, 9, "text", text))
        return self.pa, 90

    def test_all_control_chars_chunk_is_never_sent(self):
        # \x00 不是 Python 的空白字符，strip() 去不掉，所以它能通过 pending 的空块过滤
        self._add_chunk("\x00\x00\x01\x02")
        self.assertIn((self.pa, 90),
                      [(it["paper_id"], it["chunk_no"]) for it in chunkembed.pending()])
        res, fake = self._run(batch=16)
        sent = [t for call in fake.calls for t in call]
        self.assertEqual([t for t in sent if not t.strip()], [], "空串一条都不该送出去")
        self.assertEqual(res["blank"], 1)
        self.assertEqual(res["embedded"], 8, "同批的 8 条正常块必须照常写进去")
        self.assertEqual(len(self._vec_rows("chunk")), 8)

    def test_a_blank_chunk_does_not_poison_its_batch(self):
        """整批一起失败才是真正的代价——这条盯的是"其余 15 条没被连坐"。"""
        self._add_chunk("\x0e" * 50)
        res, _ = self._run(batch=3)
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["embedded"], 8)

    def test_clean_corpus_reports_zero_blank(self):
        res, _ = self._run()
        self.assertEqual(res["blank"], 0)
        self.assertEqual(res["embedded"], 8)


class BatchCapTests(_Base):
    """`embeddings.embed_texts` 内部固定按 16 拆 HTTP，一段失败整个调用抛异常，
    前面已成功（已计费、已进 llm_calls）的段被丢弃——实测 batch=32、第 2 个 HTTP
    失败时付了 1600 prompt_tokens、库里 0 条向量。所以本模块的 batch 上限就是 16。"""

    def _pad_to(self, n_extra: int):
        """夹具只有 8 块，凑不出「一批超过 16 条」——不补块的话这条用例是空转的
        （断言 8 ≤ 16 恒真，去掉上限也照样绿）。"""
        with db.conn() as c:
            for k in range(n_extra):
                c.execute("INSERT INTO chunks(paper_id,chunk_no,section_path,level,"
                          "start_page,end_page,kind,text) VALUES(?,?,?,?,?,?,?,?)",
                          (self.pa, 100 + k, "6 APPENDIX", 1, 10, 10, "text",
                           _body(300, _SEED_A) + f" [{k}]"))

    def test_batch_larger_than_the_api_batch_is_capped(self):
        self._pad_to(25)                       # 共 33 块，batch=64 本来会切成一批 33 条
        self.assertEqual(len(chunkembed.pending()), 33)
        res, fake = self._run(batch=64)
        self.assertEqual(chunkembed.API_BATCH, 16)
        self.assertEqual(fake.batch_sizes, [16, 16, 1],
                         f"传 64 实际发出 {fake.batch_sizes}，失败连坐面比 API 分批还大")
        self.assertEqual(res["embedded"], 33)

    def test_small_batch_is_untouched(self):
        _, fake = self._run(batch=3)
        self.assertEqual(fake.batch_sizes, [3, 3, 2])



class EarlyAbortTests(_Base):
    """端点级永久错误 / 连续失败：必须尽早停手，不要拿几十批的时间和钱撞同一堵墙。

    实测踩到的真事：中转端点只有 chat 路由、`/embeddings` 恒 404，
    原实现把 49 个批次全部跑完（100.2 秒）才收工，每一批都在验证同一件已知的事。
    修复后第 1 批即停（1.8 秒）。
    """

    def _boom(self, msg):
        def f(texts, purpose="x"):
            raise RuntimeError(msg)
        return f

    def _run_failing(self, msg, batch=4):
        with mock.patch.object(embeddings, "available", return_value=True),              mock.patch.object(embeddings, "embed_texts",
                               side_effect=self._boom(msg)) as m:
            r = chunkembed.embed_chunks(batch=batch)
        return r, m

    def test_permanent_404_aborts_on_first_batch(self):
        r, m = self._run_failing("embeddings 接口失败 HTTP 404：nginx")
        self.assertEqual(r["batches"], 1, "永久错误必须第 1 批就停")
        self.assertGreater(r["planned_batches"], 1, "本来有多批，说明确实提前停了")
        self.assertIn("端点不可用", r["aborted"] or "")
        self.assertEqual(m.call_count, 1, "只该调一次接口")
        self.assertEqual(r["embedded"], 0)

    def test_permanent_401_also_aborts(self):
        r, _ = self._run_failing("接口失败 HTTP 401 unauthorized")
        self.assertEqual(r["batches"], 1)
        self.assertIn("HTTP 401", r["aborted"] or "")

    def test_transient_5xx_is_not_treated_as_permanent(self):
        """5xx 属于值得重试的瞬时故障，不能与 404 同等对待——但连续到上限仍要停。"""
        r, m = self._run_failing("接口失败 HTTP 503 upstream busy", batch=2)
        self.assertGreater(m.call_count, 1, "瞬时错误不该第 1 批就放弃")
        # 夹具批数可能少于上限，此时会自然跑完；只要没被当成永久错误即可
        self.assertLessEqual(m.call_count, chunkembed.CONSECUTIVE_FAIL_LIMIT)
        self.assertNotIn("端点不可用", r["aborted"] or "",
                         "5xx 不该被判成端点级永久错误")

    def test_intermittent_failure_does_not_abort(self):
        """成功-失败交替不该触发中止——连续计数要能被成功清零。"""
        real = FakeEmbedder()
        calls = {"n": 0}

        def flaky(texts, purpose="x"):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("接口失败 HTTP 500")
            return real(texts, purpose)

        with mock.patch.object(embeddings, "available", return_value=True),              mock.patch.object(embeddings, "embed_texts", side_effect=flaky):
            r = chunkembed.embed_chunks(batch=4)
        self.assertIsNone(r["aborted"], "间歇失败不该被当成连环失败")
        self.assertGreater(r["embedded"], 0)

    def test_abort_message_says_what_to_check(self):
        """错误信息要能直接指向排查方向，不能只说「失败了」。"""
        r, _ = self._run_failing("embeddings 接口失败 HTTP 404：nginx")
        msg = r["aborted"] or ""
        self.assertIn("LLM_API_BASE", msg)
        self.assertIn("EMBED_MODEL", msg)


if __name__ == "__main__":
    unittest.main()
