# -*- coding: utf-8 -*-
"""chunk 字符上限必须在「产 chunk」这一层收口，不能只由 structure 一条路守。

`MAX_CHUNK_CHARS` 原来不是一条被强制的不变量，只是 `structure.section_chunks` 的一个
参数默认值。产 chunk 有三条路，另外两条一刀不切：
  · `db.chunks_from_pages` —— 一页 / 一节原样当一块，零长度处理；
  · `docimport._chunks_of` —— 段落缓冲是「先 append 再判」，单段超限时切不开；
    table 分支整块 append。

真库形态（2026-09-07，804 块 / 45 篇）：长度 13~11924、中位 1227；**17 块（2.11%）
超过 4000、4 块（0.50%）超过嵌入侧的 6000，全部出自 `chunks_from_pages`**
（structure 那条路的 max 恰好是 4000，一条没越界）。
那 4 块的 34553 字里有 10553 字（30.5%）从没进过向量，最差一块丢 49.7%。

**危害的口径要说准**（这条修正了审计的说法）：丢掉的尾巴在 `chunks_fts` 里仍有完整
索引，英文词面查得到；但中文提问词面恒 0（真库实测 3 条中文问句均 0 行），
向量那一路又不含尾部——所以不是「检索不到」，是「**向量路**检索不到」。

正文取自真库 paper_id=490 第 4 页的真实段落，NEEDLE 是它真正的尾句
（位于第 6000 字之后，当前代码下完全不进向量）。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import chunkembed, config, db, docimport, embeddings

MODEL = "test-embed-1024"
DIM = 1024
REAL_MAX_CHUNK = 11924          # 真库最长一块

_REAL_PARAS = [
    "As already stated, our main proposal is using neural embedding models to "
    "learn a linear transformation matrix from the vector space of one language "
    "to the vector space of another, and then employing this matrix to map "
    "words from the source language into the target language space.",
    "We trained continuous bag-of-words and continuous skipgram models on the "
    "Ukrainian and the Russian corpora, with vector size 300, symmetric window "
    "of 2 words, 10 negative samples and 5 iterations over the corpus.",
    "The regularization term was used to tune the influence of the "
    "transformation matrix, and we divided the bilingual dictionary into 4500 "
    "noun pairs used as a training set and 500 noun pairs used as a test set.",
]
NEEDLE = ("For reference, we also report the accuracy of quazi-translation via "
          "Damerau-Levenshtein edit distance BIBREF9 , as a sort of a baseline.")


def _real_page_text(n_chars: int = REAL_MAX_CHUNK) -> str:
    """真实段落铺到 n_chars，尾句压在最后。"""
    body, i = "", 0
    while len(body) + len(NEEDLE) + 2 < n_chars:
        body += _REAL_PARAS[i % len(_REAL_PARAS)] + "\n"
        i += 1
    return body[:n_chars - len(NEEDLE) - 1] + "\n" + NEEDLE


class _Resp:
    """假的 /embeddings 200 响应。不发任何网络请求。"""

    status_code = 200
    text = ""

    def __init__(self, batch):
        self._n = len(batch)

    def json(self):
        return {"data": [{"index": i, "embedding": [0.01] * DIM}
                         for i in range(self._n)], "usage": {"prompt_tokens": 0}}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_chunkcap_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL,
                     config.EMBED_API_KEY, config.EMBED_API_BASE,
                     list(embeddings._LEARNED_CAP))
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        config.EMBED_MODEL = MODEL
        config.EMBED_API_KEY = "stub-key"
        config.EMBED_API_BASE = "http://stub.invalid/v1"
        embeddings._LEARNED_CAP[:] = [embeddings.INPUT_CHARS]
        embeddings.invalidate_cache()
        db.init_db()

    def tearDown(self):
        embeddings.invalidate_cache()
        (config.DB_PATH, config.DATA_DIR, config.EMBED_MODEL,
         config.EMBED_API_KEY, config.EMBED_API_BASE,
         embeddings._LEARNED_CAP[:]) = self._old

    def _seed_page(self, text: str) -> int:
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "qasper:490", "title": "Clustering Comparable Corpora",
                "abstract": "", "year": 2019, "venue": None, "authors": [],
                "doi": None, "arxiv_id": None, "source": "qasper"})
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (pid, 4, text))
        return pid

    def _seed_chunks(self) -> int:
        pid = self._seed_page(_real_page_text())
        with db.conn() as c:
            db.replace_chunks(c, pid, db.chunks_from_pages(c, pid))
        return pid


class ChunkLengthCapIsEnforcedAtProduction(_Base):
    def test_chunks_from_pages_never_exceeds_the_cap(self):
        """真库 17/804 块越界全出自这里：一整页原样当一块，没有任何长度处理。"""
        pid = self._seed_page(_real_page_text())
        with db.conn() as c:
            chunks = db.chunks_from_pages(c, pid)
        worst = max(len(ch["text"]) for ch in chunks)
        self.assertLessEqual(
            worst, db.MAX_CHUNK_CHARS,
            f"chunks_from_pages 产出 {worst} 字的块，超过上限 {db.MAX_CHUNK_CHARS}"
            f"（真库最长 11924）——上限只在 structure 那条路上生效，兜底这条路一刀不切")

    def test_docimport_chunks_of_never_exceeds_the_cap(self):
        """段落缓冲是「先 append 再判」，单个超长段落一刀都切不开。"""
        para = _REAL_PARAS[0]
        n = docimport.MAX_CHUNK_CHARS // len(para) + 1
        blocks = [{"kind": "paragraph", "text": para} for _ in range(n)]
        blocks.append({"kind": "paragraph", "text": _real_page_text(6800)})
        out = docimport._chunks_of({"blocks": blocks})
        worst = max(len(ch["text"]) for ch in out)
        self.assertLessEqual(
            worst, docimport.MAX_CHUNK_CHARS,
            f"_chunks_of 产出 {worst} 字的块，超过上限 {docimport.MAX_CHUNK_CHARS}")

    def test_cap_stays_strictly_below_the_embedding_budget(self):
        """口径必须只有一处在切：块上限严格小于嵌入侧上限。"""
        self.assertLess(db.MAX_CHUNK_CHARS, chunkembed.MAX_CHARS)
        self.assertLess(db.MAX_CHUNK_CHARS, embeddings.INPUT_CHARS)

    def test_table_chunks_are_left_alone(self):
        """表格块有自己的行级口径，按字符再切一次会把行劈开、续块丢表头。"""
        long_table = "| a | b |\n" * 800
        out = db.cap_chunks([{"text": long_table, "kind": "table"}])
        self.assertEqual(len(out), 1, "表格块被按字符切开了")
        self.assertEqual(out[0]["text"], long_table)


class PageTailReachesTheEmbeddingEndpoint(_Base):
    def test_tail_of_a_long_page_is_actually_sent_to_the_endpoint(self):
        self._seed_chunks()
        sent: list[str] = []

        def _fake_post(url, batch):
            sent.extend(batch)
            return _Resp(batch)

        with mock.patch.object(embeddings, "_post_once", _fake_post):
            res = chunkembed.embed_chunks()
        self.assertEqual(res["errors"], [])
        self.assertGreater(len(sent), 0, "一条文本都没送出去，测试没测到东西")
        self.assertTrue(
            any(NEEDLE in t for t in sent),
            "整页最后那句（真库里位于第 6000 字之后）没有一个字送进 embedding 端点："
            "对应向量不含这段内容，跨语言提问时词面命中为 0，这段内容检索不到。"
            f"实际送出 {len(sent)} 条、最长 {max(len(t) for t in sent)} 字，"
            f"整页 {REAL_MAX_CHUNK} 字")

    def test_no_chunk_is_truncated_at_embed_time(self):
        """收口之后 truncated 必须是 0——只在产块那一层切过一次。"""
        self._seed_chunks()
        with mock.patch.object(embeddings, "_post_once",
                               lambda url, batch: _Resp(batch)):
            res = chunkembed.embed_chunks()
        self.assertEqual(res["truncated"], 0,
                         f"仍有 {res['truncated']} 条块在嵌入时被截——"
                         f"上限没在产块那一层收住，两处各切了一次")


if __name__ == "__main__":
    unittest.main()
