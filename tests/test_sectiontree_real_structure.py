# -*- coding: utf-8 -*-
"""`build_all` 必须用 structure 落库的章节，而不是从 PDF 页首行猜标题。

`build_all` 循环里只有 `build_index(pid)`，从不传 `sections`，于是每一篇都走
「从 pages 首行猜标题」的降级路径。真实 PDF 的页首行是**页码**（'1'、'2'），
`_looks_like_heading` 对纯数字返回 False，于是整页成一节、title=''、
section_path='（第 N 页）'。而 sectiontree 的第一级打分由 TITLE_W=6.0 / PATH_W=1.5
主导——标题全空等于把第一级最强的信号整个抹掉。
同时 `build_index` 明明返回了 `degraded`，`build_all` 读都没读。

真库副本实测（45 篇 / 590 节点）：
  · 假节 28/587 = 4.77% → **2/590 = 0.34%**（只剩两篇真·无结构的）；
    paper 1 从 26 个「（第 N 页）」变成 29 个带真实层级的节点；
  · `build_all` 现在如实报「2/45 篇有页给不出节标题…paper 486、499」。

**三处必须一起做，少一处就是把响亮的错换成安静的错：**

① **守卫是「一页标题都没有」，不是「build_index 报了 degraded」**。
   `_nodes_from_pages` 只要有**一页**首行不像标题就置 degraded。真库 paper 486
   的 11 页里 10 页是完美的 QASPER 节名、只有第 9 页因以句号结尾被判 False——
   按「degraded 非空」当守卫会把整篇翻到 PDF 坐标系去。

② **第二级要用节点自己的正文定位，不能回 pages 表按页区间猜**。结构化章节的页区间
   会互相重叠（父节的引言段与第一个子节共占一页），按页回取会把片段贴上不属于它的
   `section_path`。实测：按页回取时标签正确率 46%，改用节点正文后 **100%**。

③ **同位置去重**。页区间重叠时两个节会选中同一 (页, 偏移)，实测最坏一条查询
   5 个名额全是同一段 400 字正文、只是挂了 5 个不同的章节标签。

用真库数据，不用合成夹具：「页首行是不是页码」「chunks 里有没有 section_path」
这两件事只有真实 PDF 论文能钉住——自己造页文本时首行总会写成像标题的样子，
缺陷就被夹具盖掉了。真库不在就 skip。
"""
import re
import sqlite3
import unittest

from papernest import config, db, degrade
from papernest import sectiontree as st

try:                       # `python -m unittest tests.x` vs `discover -s tests`
    from .support import TempDbTestCase
except ImportError:
    from support import TempDbTestCase

_FAKE_PATH = re.compile(r"^（第 \d+ 页）$")
_WS = re.compile(r"\s+")


def _n(s: str) -> str:
    return _WS.sub("", s or "").lower()


class BuildAllUsesStructuredSectionsTests(TempDbTestCase):
    prefix = "papernest_sectree_realdata"

    #: 真库路径必须在 super().setUp() 改写 config.DATA_DIR **之前**算好
    REAL_DB = config.ROOT / "data" / "papernest.db"

    def setUp(self):
        super().setUp()
        if not self.REAL_DB.exists():
            self.skipTest("没有真库 data/papernest.db")
        pages, chunks = self._real_pdf_paper()
        if pages is None:
            self.skipTest("真库里没有「pages 来自真实 PDF + chunks 带 section_path」的论文")
        db.init_db()
        st.ensure_schema()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "real:pdf-paper", "title": "Real PDF Paper",
                "abstract": "", "year": 2025, "venue": "real", "authors": [],
                "doi": None, "arxiv_id": None, "source": "pdf"})
            for r in pages:
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (self.pid, r["page_no"], r["text"]))
            db.reindex_pages(c, self.pid)
            db.replace_chunks(c, self.pid, [dict(r) for r in chunks])

    def _real_pdf_paper(self):
        """从真库只读取一篇「首行是页码、且 chunks 带结构」的论文的原始页与块。"""
        src = sqlite3.connect(f"file:{self.REAL_DB.as_posix()}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        try:
            for (pid,) in src.execute(
                    "SELECT DISTINCT paper_id FROM chunks "
                    "WHERE kind='text' AND COALESCE(section_path,'')<>'' "
                    "ORDER BY paper_id"):
                pages = src.execute(
                    "SELECT page_no,text FROM pages WHERE paper_id=? ORDER BY page_no",
                    (pid,)).fetchall()
                if len(pages) < 3:
                    continue
                # 「首行是页码」是这条缺陷的触发条件，挑真的满足它的那一篇
                heads = [(p["text"] or "").strip().split("\n")[0].strip()
                         for p in pages[:3]]
                if not all(h.isdigit() for h in heads):
                    continue
                chunks = src.execute(
                    "SELECT section_path,level,start_page,end_page,kind,text FROM chunks "
                    "WHERE paper_id=? ORDER BY chunk_no", (pid,)).fetchall()
                return pages, chunks
        finally:
            src.close()
        return None, None

    def _nodes(self):
        with db.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM section_nodes WHERE paper_id=? ORDER BY id", (self.pid,))]

    # ── ① build_all 必须用落库的结构化章节，而不是从页首行猜 ──

    def test_build_all_uses_structured_sections_not_page_number_stubs(self):
        st.build_all()
        nodes = self._nodes()
        self.assertTrue(nodes, "这篇论文一个节点都没建出来")
        fake = [n["section_path"] for n in nodes
                if _FAKE_PATH.match(n["section_path"] or "")]
        self.assertEqual(
            fake, [],
            f"build_all 建出了 {len(fake)}/{len(nodes)} 个「（第 N 页）」假节："
            f"它没有用 chunks 里已经切好的 structure 章节。样例：{fake[:3]}")
        self.assertTrue([n for n in nodes if (n["title"] or "").strip()],
                        "所有节点的 title 都是空的——标题没从结构里拿到")
        self.assertTrue([n for n in nodes if " > " in (n["section_path"] or "")],
                        "没有任何多级 section_path——章节层级整个丢了")

    # ── ② 守卫：只有「一页标题都没有」才换 chunks ──

    def test_a_single_headless_page_does_not_flip_the_whole_paper(self):
        """真库 paper 486：11 页里 10 页是完美的 QASPER 节名，只有 1 页不像标题。

        按「degraded 非空」当守卫会把整篇翻到 PDF 坐标系去，它的 pages 与 chunks
        是两套页号，第二级会照错误的页区间取正文。判据必须是 no_head == pages_seen。
        """
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "real:mostly-headed", "title": "Mostly Headed",
                "abstract": "", "year": 2025, "venue": "v", "authors": [],
                "doi": None, "arxiv_id": None, "source": "s2"})
            for no, text in enumerate(
                    ["Introduction\nWe study contextual representations.",
                     "Related Work\nGloVe and word2vec produce static vectors.",
                     "7\nThis page starts with a page number instead of a heading."], 1):
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (pid, no, text))
            # 这一篇**有**结构化 chunks，而且是另一套页号（模拟真库 QASPER 论文：
            # pages 是数据集正文、chunks 是后来按真实 PDF 重切的）。守卫要是只看
            # 「degraded 非空」，它就会被整篇翻到 chunks 的坐标系上去。
            db.replace_chunks(c, pid, [
                {"text": "Some unrelated PDF body text about neural networks.",
                 "section_path": "I. INTRODUCTION", "level": 1,
                 "start_page": 1, "end_page": 1, "kind": "text"},
                {"text": "More PDF body text about experiments and results.",
                 "section_path": "V. EXPERIMENTS", "level": 1,
                 "start_page": 2, "end_page": 2, "kind": "text"}])
            db.reindex_pages(c, pid)
        r = st.build_index(pid)
        self.assertTrue(r["degraded"], "夹具要有一页给不出标题，否则用例是空跑")
        self.assertLess(r["pages_without_heading"], r["pages_seen"],
                        "夹具里应当只有一页给不出标题")
        st.build_all()
        with db.conn() as c:
            paths = [x[0] for x in c.execute(
                "SELECT section_path FROM section_nodes WHERE paper_id=?", (pid,))]
        self.assertIn("Introduction", paths,
                      f"整篇被翻掉了：{paths}——守卫应当只在一页标题都没有时才换 chunks")

    # ── ③ 溯源标签必须是真的，同位置不许重复 ──

    def test_passage_really_comes_from_the_section_it_is_labelled_with(self):
        """按页区间回取时，重叠的页会让片段贴上不属于它的 section_path。"""
        st.build_all()
        bodies = {n["node_id"]: (n["title"] or "") + (n["body"] or "")
                  for n in self._nodes()}
        checked = bad = 0
        for q in ("evaluation benchmarks", "security and privacy", "collaboration",
                  "future directions", "applications"):
            hits = st.search_in_scope(q, st.select_scope(q, top_nodes=12), top_k=5)
            for h in hits:
                hay = bodies.get(h["node_id"])
                if not hay:
                    continue
                checked += 1
                if _n(h["text"])[:120] not in _n(hay):
                    bad += 1
        self.assertGreater(checked, 0, "一条可判定的片段都没有，用例是空跑")
        self.assertEqual(bad, 0,
                         f"{bad}/{checked} 个片段的正文并不出自它被标注的那一节——"
                         f"章节路径是按页区间猜的，不是真出处")

    def test_no_duplicate_passage_position_in_the_results(self):
        """页区间重叠时两个节会选中同一 (页, 偏移)，看上去像两条独立证据。"""
        st.build_all()
        for q in ("evaluation benchmarks", "security and privacy", "collaboration"):
            hits = st.search_in_scope(q, st.select_scope(q, top_nodes=12), top_k=5)
            pos = [(h["paper_id"], h["page_no"], h["offset"]) for h in hits]
            self.assertEqual(len(pos), len(set(pos)),
                             f"查询 {q!r} 的结果里有同位置的重复片段：{pos}")

    # ── ④ 拿不到结构时，降级必须结构化地报上来，不能被 build_all 吞掉 ──

    def test_build_all_reports_page_fallback_as_structured_degradation(self):
        with db.conn() as c:
            c.execute("DELETE FROM chunks WHERE paper_id=?", (self.pid,))
            c.execute("DELETE FROM chunks_fts WHERE paper_id=?", (self.pid,))
        r = st.build_all()

        # 这一篇确实退回了「整页成一节」——前提成立，下面的断言才有意义
        self.assertTrue([n for n in self._nodes()
                         if _FAKE_PATH.match(n["section_path"] or "")])
        self.assertIn("degraded_detail", r, "build_all 把 build_index 的 degraded 丢了")
        self.assertEqual([d["code"] for d in r["degraded_detail"]],
                         [degrade.SECTION_TREE_PAGE_FALLBACK])
        self.assertTrue(r.get("degraded"), "没有给人看的那句降级说明")


if __name__ == "__main__":
    unittest.main()
