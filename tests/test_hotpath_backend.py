"""检索热路径与向量后端抽象层的接线。

背景：向量后端抽象层（`vectorstore`）一度只被 `cli.py vec` 子命令用到，问答的
检索热路径自己读 SQLite 的 BLOB 建矩阵——于是「换后端」这件事对线上检索毫无影响，
后端选型也就无从谈起。现在 `embeddings.search_papers` 走 `get_store()`，本文件钉住
三件不能回退的事：

1. 热路径确实经过 `get_store_or_degrade()`，而不是绕过它自己算；
2. 派生索引落后于真相（collection 没建、rebuild 没跑完）时**退回真相来源现算**，
   并留下降级记录——空结果和「确实没有相关论文」在上层长得一模一样，
   静默返回空是这一层最危险的失败模式；
3. 降级记录能一路传到 `search_hybrid` 的 `degraded` 里，不会被吞掉。

另外钉住一条老规矩：换了嵌入模型但库里还是旧维度时，报错而不是静默截断。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from papernest import config, db, degrade, embeddings, vectorstore

DIM = 8


def _blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


class FakeStore:
    """假后端：只记录被问了什么，返回预置结果。用来验证「热路径真的问了后端」。"""

    def __init__(self, rows, truth_count=0, name="fake"):
        self.name = name
        self.degraded = None
        self._rows = rows
        self._truth_count = truth_count
        self.calls = []

    def search(self, qvec, top_k, kind=None, where=None):
        self.calls.append({"top_k": top_k, "kind": kind, "where": where})
        return list(self._rows)

    def count(self, kind=None):
        return self._truth_count


class HotPathBackendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_hotpath_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = "test-embed"
        embeddings.invalidate_cache()
        embeddings.take_store_note()          # 清掉别的用例可能留下的记录
        self.addCleanup(embeddings.take_store_note)
        db.init_db()
        with db.conn() as c:
            self.ids = []
            for i, vec in enumerate([
                    [1, 0, 0, 0, 0, 0, 0, 0],       # 与查询完全同向
                    [0.8, 0.6, 0, 0, 0, 0, 0, 0],   # 夹角居中
            ], 1):
                pid = db.insert_l0(c, {
                    "norm_key": f"arxiv:{8100+i}", "title": f"P{i}", "abstract": "a",
                    "year": 2024, "venue": "V", "authors": [], "doi": None,
                    "arxiv_id": str(8100 + i), "source": "s2"})
                self.ids.append(pid)
                c.execute("INSERT INTO vectors(paper_id,kind,idx,text,model,vec) "
                          "VALUES(?,?,?,?,?,?)",
                          (pid, "paper", 0, f"P{i}", "test-embed", _blob(vec)))

    def tearDown(self):
        embeddings.invalidate_cache()
        config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL = self._old

    def _search(self, qv=(1, 0, 0, 0, 0, 0, 0, 0), top_k=3):
        with mock.patch.object(embeddings, "embed_texts", return_value=[list(qv)]):
            return embeddings.search_papers("q", top_k)

    # ── 1. 热路径经过抽象层 ──

    def test_search_papers_goes_through_the_store_abstraction(self):
        """结果必须来自 `get_store()` 给的后端，不能是热路径自己算的另一份。"""
        fake = FakeStore([{"paper_id": 42, "kind": "chunk", "idx": 7, "score": 0.9},
                          {"paper_id": 42, "kind": "paper", "idx": 0, "score": 0.5},
                          {"paper_id": 43, "kind": "paper", "idx": 0, "score": 0.3}])
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(fake, None)):
            hits = self._search()
        self.assertEqual([h["paper_id"] for h in hits], [42, 43],
                         "热路径没有用后端返回的结果，说明它绕过了抽象层")
        self.assertEqual(hits[0]["chunk_no"], 7,
                         "赢在章节块上时要把是哪一块带出来，否则上下文组装只能靠猜")
        self.assertIsNone(hits[1]["chunk_no"], "赢在标题摘要向量上时没有对应的正文块")
        self.assertEqual(len(fake.calls), 1, "一次检索只该问一次后端")

    def test_only_paper_and_chunk_vectors_take_part(self):
        """句级向量是引用推荐那条路用的，混进论文级检索会让同一篇论文刷屏。"""
        fake = FakeStore([])
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(fake, None)):
            self._search()
        self.assertEqual(fake.calls[0]["where"], {"kind": {"$in": ["paper", "chunk"]}})

    def test_asks_for_more_rows_than_papers_wanted(self):
        """后端返回的是向量行，这里要的是论文：不超取的话，章节多的论文会占满名额。"""
        fake = FakeStore([])
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(fake, None)):
            self._search(top_k=5)
        self.assertGreater(fake.calls[0]["top_k"], 5)

    # ── 2. 索引落后于真相：退回真相来源，并留痕 ──

    def test_falls_back_to_truth_when_derived_index_is_behind(self):
        """派生索引里一行都没有、真相里却有向量：必须现算，不能把空结果交上去。"""
        behind = FakeStore([], truth_count=2, name="milvus")
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(behind, None)):
            hits = self._search()
        self.assertEqual([h["paper_id"] for h in hits], self.ids,
                         "索引落后时没有退回真相来源，检索静默变成了空结果")
        note = embeddings.take_store_note()
        self.assertIsNotNone(note, "退回真相来源是降级，必须留痕")
        self.assertIn("rebuild", note, "降级记录要告诉人怎么修")

    def test_empty_result_is_not_a_degradation_when_truth_is_also_empty(self):
        """真相里本来就没有向量，返回空是正常结果，不该报成降级。"""
        empty = FakeStore([], truth_count=0, name="milvus")
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(empty, None)):
            self.assertEqual(self._search(), [])
        self.assertIsNone(embeddings.take_store_note())

    def test_backend_unavailable_reason_is_passed_through(self):
        """后端连不上时 `get_store_or_degrade` 给的原因要原样交给调用点。"""
        fallback = FakeStore([{"paper_id": 1, "kind": "paper", "idx": 0, "score": 0.4}])
        with mock.patch.object(vectorstore, "get_store_or_degrade",
                               return_value=(fallback, "milvus 后端不可用，已降级 numpy")):
            self._search()
        self.assertEqual(embeddings.take_store_note(), "milvus 后端不可用，已降级 numpy")

    # ── 3. 降级要一路传到 search_hybrid ──

    def test_search_hybrid_records_backend_degradation(self):
        """降级记录被吞掉，界面上就会显示「一切正常」而召回其实是残的。"""
        with mock.patch.object(embeddings, "available", return_value=True), \
             mock.patch.object(embeddings, "search_papers", return_value=[]), \
             mock.patch.object(embeddings, "take_store_note",
                               side_effect=[None, "milvus 索引落后于 SQLite 真相"]):
            res = embeddings.search_hybrid("q", 3)
        self.assertIn(degrade.VECTOR_BACKEND_DEGRADED, [d.code for d in res.degraded])

    def test_no_degradation_recorded_when_backend_is_healthy(self):
        """后端好好的却报降级，和吞掉降级一样会让人不再相信这个信号。"""
        with mock.patch.object(embeddings, "available", return_value=True), \
             mock.patch.object(embeddings, "search_papers",
                               return_value=[{"paper_id": 1, "score": 0.9,
                                              "chunk_no": None}]), \
             mock.patch.object(embeddings, "take_store_note", return_value=None):
            res = embeddings.search_hybrid("q", 3)
        self.assertEqual([d.code for d in res.degraded
                          if d.code == degrade.VECTOR_BACKEND_DEGRADED], [])

    # ── 4. 老规矩不能因为换了后端就丢 ──

    def test_dimension_mismatch_still_raises_instead_of_truncating(self):
        """截断出来的余弦是没有意义的数，比报错更危险——换后端之后这条仍然成立。"""
        with self.assertRaises(embeddings.EmbedUnavailable) as cm:
            self._search(qv=(1, 0, 0))          # 3 维查询 vs 8 维库
        self.assertIn("维度不一致", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
