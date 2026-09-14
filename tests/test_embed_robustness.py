"""嵌入调用的健壮性：重试、自适应截断、以及换模型的安全性。

这三条都是换 embedding 模型（`qwen3.7-text-embedding` →
`qwen3.7-text-embedding-flash`）时**真跑出来**的，静态检查一条都发现不了：

1. **`embed_texts` 没有任何重试**——一次瞬态 `ReadTimeout` 就让整条
   `cli.py embed` 挂掉，而这条命令要跑几百次调用。原审计标记过，一直没修。
2. **端点对超限输入是挂起、不是返回 400**。实测同一篇摘要截到 1870 字符正常
   返回 0.7s、1878 字符必定超时（连测 4 次），而另外两篇在 2100 字符处完全正常
   ——不是固定字符阈值（token/字符比随语言差好几倍），只能自适应缩短。
3. **换模型时新旧嵌入空间几乎正交（中位数余弦 0.0240）而维度相同（都 1024）**，
   维度检查不会报错。靠 `vectors.model` 列隔离才没让「拿新查询向量去和旧文档向量
   算余弦」这种无意义结果静默发生。
"""
from __future__ import annotations

import unittest
from unittest import mock

import httpx

from papernest import config, embeddings


class _Resp:
    def __init__(self, status=200, n=1, dim=4):
        self.status_code = status
        self._n, self._dim = n, dim
        self.text = "err"

    def json(self):
        return {"data": [{"index": i, "embedding": [0.1] * self._dim}
                         for i in range(self._n)],
                "usage": {"prompt_tokens": 1}}


class _Ctx:
    """替身 http.client()：按脚本返回响应或抛异常。"""

    def __init__(self, script, calls):
        self.script, self.calls = script, calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append(json["input"])
        step = self.script.pop(0) if self.script else _Resp(n=len(json["input"]))
        if isinstance(step, Exception):
            raise step
        if step is None:
            return _Resp(n=len(json["input"]))
        return step


def _patched(script, calls):
    """脚本必须**跨调用共享**——每次 client() 都复制一份的话，
    每次调用都会拿到脚本的第一步，重试逻辑就永远测不到。"""
    shared = list(script)
    return mock.patch.object(embeddings.http, "client",
                             lambda *a, **k: _Ctx(shared, calls))


class Base(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(config, "EMBED_MODEL", "m"),
                  mock.patch.object(config, "EMBED_API_KEY", "k"),
                  mock.patch.object(config, "EMBED_API_BASE", "https://x.invalid/v1"),
                  mock.patch.object(embeddings.db, "conn"),
                  mock.patch.object(embeddings, "_EMBED_DELAYS", (0, 0, 0))):
            p.start()
            self.addCleanup(p.stop)
        embeddings._LEARNED_CAP[0] = embeddings.INPUT_CHARS
        embeddings.LAST_TRUNCATION.clear()
        self.addCleanup(embeddings.LAST_TRUNCATION.clear)


class TransientFailuresAreRetried(Base):
    """回归：原来一次超时就让整条 embed 命令挂掉。"""

    def test_recovers_from_a_transient_error(self):
        calls = []
        with _patched([httpx.ConnectError("boom"), None], calls):
            out = embeddings.embed_texts(["hello"])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(calls), 2, "没有重试")

    def test_retries_5xx(self):
        calls = []
        with _patched([_Resp(status=503), None], calls):
            embeddings.embed_texts(["hello"])
        self.assertEqual(len(calls), 2)

    def test_permanent_errors_are_not_retried(self):
        """400/401/403 重试多少次都是同一个错——口径与 llm.chat 一致。"""
        for status in (400, 401, 403):
            calls = []
            with _patched([_Resp(status=status)] * 5, calls):
                with self.assertRaises(embeddings.EmbedUnavailable):
                    embeddings.embed_texts(["hello"])
            self.assertEqual(len(calls), 1, f"HTTP {status} 不该重试")

    def test_gives_up_after_the_budget(self):
        calls = []
        with _patched([httpx.ConnectError("x")] * 10, calls):
            with self.assertRaises(embeddings.EmbedUnavailable):
                embeddings.embed_texts(["hello"])
        self.assertLessEqual(len(calls), len(embeddings._EMBED_DELAYS) + 1)


class AdaptiveTruncationOnTimeout(Base):
    """端点对超限输入挂起而不是报错——退避多久都没用，只能缩短输入。"""

    def test_shrinks_input_after_timeout(self):
        calls = []
        long_text = "x" * 5000
        # 前几次超时（长输入），缩短后成功
        script = [httpx.ReadTimeout("t")] * 4 + [None]
        with _patched(script, calls):
            embeddings.embed_texts([long_text])
        self.assertGreater(len(calls), 1)
        self.assertLess(len(calls[-1][0]), len(long_text),
                        "超时之后没有缩短输入，只是干等")

    def test_truncation_is_recorded_not_silent(self):
        """向量代表的不再是完整文本，静默截断会让人以为「覆盖了整篇」。"""
        calls = []
        with _patched([httpx.ReadTimeout("t")] * 4 + [None], calls):
            embeddings.embed_texts(["y" * 5000])
        self.assertTrue(embeddings.LAST_TRUNCATION, "截断没有留痕")

    def test_no_truncation_means_no_record(self):
        calls = []
        with _patched([None], calls):
            embeddings.embed_texts(["short"])
        self.assertEqual(embeddings.LAST_TRUNCATION, [])

    def test_learned_cap_is_reused_across_batches(self):
        """回归：每批都从 6000 重新往下踩超时，804 条 chunk 要跑几小时。
        实测修好后从 16 条/十几分钟变成 788 条/207 秒。"""
        calls = []
        texts = ["z" * 5000] * (embeddings.BATCH_SIZE * 2)   # 两批
        script = [httpx.ReadTimeout("t")] * 4 + [None, None]
        with _patched(script, calls):
            embeddings.embed_texts(texts)
        first_len = len(calls[-2][0])
        second_len = len(calls[-1][0])
        self.assertEqual(first_len, second_len,
                         "第二批没有复用学到的上限，又从头踩了一遍超时")

    def test_short_inputs_are_never_shrunk_below_the_floor(self):
        calls = []
        with _patched([httpx.ReadTimeout("t")] * 12, calls):
            with self.assertRaises(embeddings.EmbedUnavailable):
                embeddings.embed_texts(["tiny"])
        # 输入本来就短于下限，不该无限缩短重试
        self.assertLessEqual(len(calls), len(embeddings._EMBED_DELAYS) + 1)


class ModelSwitchIsolatesVectors(unittest.TestCase):
    """换模型必须**失败得响**，不能拿新查询向量去和旧文档向量算余弦。

    实测两个模型维度相同（都 1024）但空间几乎正交（中位数余弦 0.0240）——
    维度检查完全挡不住。挡住它的是 `vectors.model` 列隔离。
    """

    def test_matrix_is_scoped_by_model(self):
        import inspect
        src = inspect.getsource(embeddings._paper_matrix)
        self.assertIn("model=?", src, "向量矩阵没有按 model 过滤")

    def test_matrix_key_includes_the_model(self):
        import inspect
        self.assertIn("EMBED_MODEL", inspect.getsource(embeddings._matrix_key),
                      "缓存指纹不含模型名，换模型后会读到上一个模型的矩阵")

    def test_empty_index_for_a_new_model_is_reported(self):
        """换模型后矩阵为空，必须报 VECTOR_INDEX_EMPTY 而不是静默走 FTS。"""
        from papernest import degrade
        with mock.patch.object(embeddings, "available", return_value=True), \
             mock.patch.object(embeddings, "search_papers", return_value=[]), \
             mock.patch.object(embeddings.db, "search_fts", return_value=[]), \
             mock.patch.object(embeddings.db, "search_chunks_hits", return_value={}), \
             mock.patch.object(embeddings.db, "conn"):
            res = embeddings.search_hybrid("q", 5)
        self.assertIn(degrade.VECTOR_INDEX_EMPTY, [n.code for n in res.degraded])
        self.assertIn("检查模型名", res.degraded[0].message,
                      "提示里没告诉人去核对模型名——那正是历史上踩过的坑")


if __name__ == "__main__":
    unittest.main()
