"""检索精排：后端可插拔、失败降级、排序确定、不静默。

这层的实测结论是**负结果**（BM25 精排净伤害，默认 off，见 rerank.py 顶部注释），
但接口契约仍要守住——将来接上真正的 Cross-Encoder 时靠它保证行为不跑偏。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, rerank


class RerankTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_rr_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._embed = mock.patch("papernest.embeddings.available", return_value=False)
        self._embed.start()
        db.init_db()
        with db.conn() as c:
            self.p_hit = db.insert_l0(c, {
                "norm_key": "arxiv:9001",
                "title": "Channel Estimation for Massive MIMO Systems",
                "abstract": "We propose a channel estimation method for massive MIMO.",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "9001", "source": "s2"})
            self.p_mid = db.insert_l0(c, {
                "norm_key": "arxiv:9002", "title": "A Survey of Beamforming",
                "abstract": "Beamforming in MIMO systems, including some channel topics.",
                "year": 2023, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "9002", "source": "s2"})
            self.p_off = db.insert_l0(c, {
                "norm_key": "arxiv:9003", "title": "Neural Machine Translation",
                "abstract": "Translation with transformers.",
                "year": 2022, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "9003", "source": "s2"})
        self.pool = [self.p_off, self.p_mid, self.p_hit]     # 故意把最相关的放最后

    def tearDown(self):
        self._embed.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir


class BM25BackendTests(RerankTestBase):
    def test_promotes_the_most_relevant_paper(self):
        ids, used = rerank.rerank("channel estimation massive MIMO", self.pool, 3, "bm25")
        self.assertEqual(used, "bm25")
        self.assertEqual(ids[0], self.p_hit)

    def test_title_weighted_above_body(self):
        """标题命中权重 3x 正文——「方法」出现在标题里比出现在正文里说明问题。"""
        ids, _ = rerank.rerank("beamforming", self.pool, 3, "bm25")
        self.assertEqual(ids[0], self.p_mid)

    def test_ordering_is_deterministic(self):
        runs = {tuple(rerank.rerank("channel", self.pool, 3, "bm25")[0]) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_ties_fall_back_to_recall_order(self):
        """全都不命中时不能乱序——保持召回池原始名次。"""
        ids, _ = rerank.rerank("完全无关的查询词", self.pool, 3, "bm25")
        self.assertEqual(ids, self.pool)

    def test_chinese_query_is_tokenised(self):
        self.assertIn("信道", rerank._terms("近场信道估计"))
        self.assertEqual(rerank._terms("MIMO channel"), ["mimo", "channel"])


class BackendSelectionTests(RerankTestBase):
    def test_off_returns_pool_unchanged(self):
        ids, used = rerank.rerank("channel estimation", self.pool, 2, "off")
        self.assertEqual(used, "off")
        self.assertEqual(ids, self.pool[:2])

    def test_api_failure_degrades_to_bm25_not_silently_off(self):
        """端点 404 是常态（很多中转站没有 /rerank）——必须降级且如实上报。"""
        with mock.patch("papernest.rerank._api_scores", return_value=None):
            ids, used = rerank.rerank("channel estimation massive MIMO",
                                      self.pool, 3, "api")
        self.assertEqual(used, "bm25")
        self.assertEqual(ids[0], self.p_hit)

    def test_api_success_is_used_and_reported(self):
        fake = {self.p_off: 0.9, self.p_mid: 0.1, self.p_hit: 0.2}
        with mock.patch("papernest.rerank._api_scores", return_value=fake):
            ids, used = rerank.rerank("q", self.pool, 3, "api")
        self.assertEqual(used, "api")
        self.assertEqual(ids[0], self.p_off)       # 真按 API 分数排，不是嘴上说说

    def test_llm_failure_degrades_to_bm25(self):
        with mock.patch("papernest.rerank._llm_scores", return_value=None):
            _ids, used = rerank.rerank("channel", self.pool, 3, "llm")
        self.assertEqual(used, "bm25")

    def test_llm_backend_parses_scores(self):
        raw = ('{"scores": [{"id": 1, "score": 2}, {"id": 2, "score": 9}, '
               '{"id": 3, "score": 1}]}')
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", return_value=raw):
            ids, used = rerank.rerank("q", self.pool, 3, "llm")
        self.assertEqual(used, "llm")
        # ids 按 paper_id 升序编号 1..3 → 第 2 个（p_mid）得分最高
        self.assertEqual(ids[0], sorted(self.pool)[1])

    def test_llm_malformed_scores_do_not_crash(self):
        for raw in ('{"scores": [{"id": "x"}]}', '{"scores": [{"id": 99, "score": 5}]}',
                    '{"nope": 1}', 'not json at all'):
            with mock.patch("papernest.llm.available", return_value=True), \
                 mock.patch("papernest.llm.chat", return_value=raw):
                ids, _used = rerank.rerank("q", self.pool, 3, "llm")
            self.assertEqual(sorted(ids), sorted(self.pool))

    def test_empty_pool(self):
        ids, used = rerank.rerank("q", [], 5, "bm25")
        self.assertEqual((ids, used), ([], "off"))


class TraceTests(RerankTestBase):
    def test_trace_reports_pool_and_backend(self):
        ids, mode, trace = rerank.retrieve_and_rerank("channel estimation", 2, 4, "bm25")
        self.assertLessEqual(len(ids), 2)
        self.assertIn("rerank:bm25", mode)
        self.assertEqual(trace["backend"], "bm25")
        self.assertGreaterEqual(trace["pool"], len(ids))

    def test_off_backend_still_reports_honestly(self):
        _ids, mode, trace = rerank.retrieve_and_rerank("channel estimation", 2, 4, "off")
        self.assertIn("rerank:off", mode)
        self.assertEqual(trace["backend"], "off")



class CrossEncoderApiTests(RerankTestBase):
    """Cross-Encoder rerank 接口：两种返回格式、解析失败要降级而不是给一堆 0 分。

    真实验证过（DashScope qwen3.7-text-rerank）：把两篇信道估计论文放在召回池最后、
    两篇 LLM Agent 论文放最前，精排后信道估计被提到前两位——这正是 BM25 精排做不到、
    而「打分器信息量必须比召回器大」那条推断预测的结果。
    """

    def _resp(self, payload, status=200):
        """造一个 http.client 的替身。

        注意 `http.client(...)` 是**调用后**才返回上下文管理器，所以这里要返回
        一个可调用对象而不是类本身——返回类的话 `with http.client(timeout=60)`
        会把 timeout 当成 __init__ 参数，post 永远走不到，_api_scores 静默返回 None，
        测试看起来「降级正确」其实一行被测代码都没执行。
        """
        class R:
            status_code = status
            def json(self_inner):
                return payload

        class C:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
            def post(self_inner, *a, **kw):
                return R()

        return lambda *a, **kw: C()

    def test_dashscope_native_format_is_parsed(self):
        """DashScope 原生：结果埋在 output.results 下。"""
        payload = {"output": {"results": [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5}]}}
        with mock.patch.object(rerank, "api_available", return_value=True), \
             mock.patch.object(rerank.http, "client", self._resp(payload)):
            got = rerank._api_scores("q", self._docs())
        self.assertEqual(got, {self.p_hit: 0.1, self.p_mid: 0.9, self.p_off: 0.5})

    def test_cohere_style_format_is_parsed(self):
        """Cohere/Jina/SiliconFlow 风格：结果在顶层 results。换提供方不该改代码。"""
        payload = {"results": [
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.2},
            {"index": 2, "relevance_score": 0.4}]}
        with mock.patch.object(rerank, "api_available", return_value=True), \
             mock.patch.object(rerank.http, "client", self._resp(payload)):
            got = rerank._api_scores("q", self._docs())
        self.assertEqual(got, {self.p_hit: 0.7, self.p_mid: 0.2, self.p_off: 0.4})

    def test_unrecognised_format_degrades_instead_of_zeroing(self):
        """一条都解析不出来时必须返回 None 让调用方降级。

        返回全 0 分会**把召回顺序打乱**——比不精排更糟，而且不报错、查不出来。
        """
        for payload in ({"data": [{"idx": 0}]}, {"results": []}, {}, {"output": {}}):
            with mock.patch.object(rerank, "api_available", return_value=True), \
                 mock.patch.object(rerank.http, "client", self._resp(payload)):
                self.assertIsNone(rerank._api_scores("q", self._docs()),
                                  f"{payload!r} 应降级")

    def test_http_error_degrades(self):
        with mock.patch.object(rerank, "api_available", return_value=True), \
             mock.patch.object(rerank.http, "client", self._resp({}, status=404)):
            self.assertIsNone(rerank._api_scores("q", self._docs()))

    def test_out_of_range_index_is_ignored(self):
        payload = {"results": [{"index": 99, "relevance_score": 0.9},
                               {"index": 0, "relevance_score": 0.3}]}
        with mock.patch.object(rerank, "api_available", return_value=True), \
             mock.patch.object(rerank.http, "client", self._resp(payload)):
            got = rerank._api_scores("q", self._docs())
        self.assertEqual(got[self.p_hit], 0.3)

    def _docs(self):
        return {p: {"title": f"T{p}", "abstract": "a", "chunks": []}
                for p in (self.p_hit, self.p_mid, self.p_off)}

if __name__ == "__main__":
    unittest.main()
