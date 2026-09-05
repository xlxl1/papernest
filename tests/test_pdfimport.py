"""本地 PDF 上传入库的离线测试。

测试用的 PDF 全部由 pymupdf 现场生成——不依赖任何外部样本文件，
字号/坐标写死，标题启发式的判定因此是确定性的。
LLM 与 embeddings 一律摁死为不可用：开发机 .env 里真配了 key，
不摁住这两个开关，测试会真的发请求（慢、花钱、结果随网络漂移）。
"""
import shutil
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from papernest import config, db, pdfimport


def make_pdf(title_lines=("Deep Learning for Near-Field",
                          "Channel Estimation in XL-MIMO"),
             authors="Alice Smith, Bob Lee and Carol Wang",
             affiliation="Tsinghua University, Beijing, China",
             doi="DOI: 10.1234/tsp.2024.567890.",
             arxiv="arXiv:2401.12345v2 [eess.SP] 3 Jan 2024",
             abstract=("We propose a deep learning based estimator for near-field "
                       "channels. Experiments show 3.2 dB gain over baselines on the "
                       "DeepMIMO dataset, which is a meaningful improvement."),
             meta=None, pages=2, title_size=18.0, header="") -> bytes:
    """造一篇「长得像论文」的 PDF：标题最大字号在顶部，下面依次是作者/单位/标识/摘要。

    header 用来模拟期刊页眉——它在标题**上方**，字号更小；「取最大字号块」
    能过而「取第一行」会挂，两种启发式在这里被区分开。
    """
    doc = pymupdf.open()
    p = doc.new_page()
    if header:
        p.insert_text((72, 70), header, fontsize=8)
    y = 110
    for line in title_lines:
        p.insert_text((72, y), line, fontsize=title_size, fontname="hebo")
        y += title_size + 6
    y += 20
    for txt, size in ((authors, 11.0), (affiliation, 9.0), (doi, 9.0), (arxiv, 9.0)):
        if txt:
            p.insert_text((72, y), txt, fontsize=size)
            y += size + 9
    if abstract:
        y += 20
        p.insert_text((72, y), "Abstract", fontsize=12, fontname="hebo")
        p.insert_textbox(pymupdf.Rect(72, y + 8, 520, y + 120), abstract, fontsize=10)
        y += 130
        p.insert_text((72, y), "1 Introduction", fontsize=13, fontname="hebo")
        p.insert_text((72, y + 20), "Massive MIMO has been widely studied.", fontsize=10)
    for i in range(pages - 1):
        p2 = doc.new_page()
        p2.insert_text((72, 100), f"{i + 2} Method section body text.", fontsize=10)
    if meta:
        doc.set_metadata(meta)
    data = doc.tobytes()
    doc.close()
    return data


def hard_layout_pdf() -> bytes:
    """更接近真实 arXiv/IEEE 排版的版面：竖排 arXiv 戳、跨行断词标题、
    上标作者标记、行内 Abstract— 起头、Index Terms 收尾。"""
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((30, 700), "arXiv:2403.09876v1  [cs.CL]  14 Mar 2024",
                  fontsize=9, rotate=90)                      # 左侧竖排戳记
    p.insert_text((72, 60), "Proceedings of the 41st International Conference", fontsize=8)
    p.insert_text((100, 120), "Retrieval-Augmented Genera-", fontsize=17, fontname="hebo")
    p.insert_text((100, 144), "tion for Long-Context Models", fontsize=17, fontname="hebo")
    p.insert_text((100, 180), "Alice Smith1, Bob Q. Lee1,2, and Carol Wang3", fontsize=11)
    p.insert_text((100, 196), "1Tsinghua University  2MIT  3Google DeepMind", fontsize=8)
    p.insert_text((100, 212), "{alice,bob}@example.edu", fontsize=8)
    p.insert_text((72, 250), "Abstract—We study retrieval augmentation for long context",
                  fontsize=10)
    p.insert_text((72, 264), "models and report a 12.4 point gain on LongBench.", fontsize=10)
    p.insert_text((72, 285), "Index Terms—RAG, long context, evaluation", fontsize=10)
    p.insert_text((72, 320), "I. INTRODUCTION", fontsize=11, fontname="hebo")
    doc.set_metadata({"title": "", "author": ""})
    data = doc.tobytes()
    doc.close()
    return data


def title_only_pdf(title: str, meta=None) -> bytes:
    """只有一个大字号标题行的 PDF：专门用来验证「版面标题」判据本身。

    字号取 14 而不是 18：insert_text 不会自动折行，18pt 下 55 字以上的标题会被
    页边裁掉，测的就变成 pymupdf 的排版而不是被测的启发式了。
    """
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 110), title, fontsize=14, fontname="hebo")
    p.insert_text((72, 200), "Some body text that is not a title at all.", fontsize=9)
    if meta:
        doc.set_metadata(meta)
    data = doc.tobytes()
    doc.close()
    return data


def chinese_pdf() -> bytes:
    """中文论文：标题/作者/单位+城市/摘要/关键词，全部走 CJK 分支。"""
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 110), "基于深度学习的近场信道估计方法", fontsize=18, fontname="china-s")
    p.insert_text((72, 150), "张三, 李四, 王五", fontsize=11, fontname="china-s")
    p.insert_text((72, 175), "清华大学电子工程系, 北京", fontsize=9, fontname="china-s")
    p.insert_text((72, 210), "摘要", fontsize=12, fontname="china-s")
    p.insert_textbox(pymupdf.Rect(72, 220, 520, 300),
                     "本文提出了一种基于深度学习的近场信道估计方法，在 DeepMIMO 数据集上"
                     "相比基线提升 3.2 dB，具有显著的性能优势和实用价值。",
                     fontsize=10, fontname="china-s")
    p.insert_text((72, 320), "关键词: 近场, 信道估计", fontsize=10, fontname="china-s")
    data = doc.tobytes()
    doc.close()
    return data


def blank_pdf(meta=None) -> bytes:
    """没有任何可用元数据的 PDF：正文只有一个两位数，内嵌元数据是 Word 垃圾。"""
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "42", fontsize=8)
    if meta:
        doc.set_metadata(meta)
    data = doc.tobytes()
    doc.close()
    return data


class PdfImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_pdfimport_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            # 卡片必须走 mock-extractive：否则 .env 有 key 时会真调 LLM
            mock.patch("papernest.llm.available", return_value=False),
        ]
        for p in self._patches:
            p.start()
        db.init_db()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _count_papers(self) -> int:
        with db.conn() as c:
            return c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]

    # ── 正常入库 ──

    def test_ingest_extracts_title_doi_abstract_and_pages(self):
        r = pdfimport.ingest_pdf(make_pdf(), "my paper.pdf")
        self.assertFalse(r["duplicate"])
        self.assertEqual(r["title"], "Deep Learning for Near-Field Channel Estimation in XL-MIMO")
        self.assertEqual(r["extracted"]["title"]["from"], "largest-font-block")
        self.assertEqual(r["norm_key"], "doi:10.1234/tsp.2024.567890")
        self.assertEqual(r["extracted"]["doi"]["value"], "10.1234/tsp.2024.567890")
        self.assertEqual(r["extracted"]["arxiv_id"]["value"], "2401.12345")
        self.assertEqual(r["extracted"]["year"]["value"], 2024)
        self.assertEqual(r["extracted"]["authors"]["value"],
                         ["Alice Smith", "Bob Lee", "Carol Wang"])
        self.assertIn("deep learning based estimator",
                      r["extracted"]["abstract"]["value"].lower())
        self.assertNotIn("Introduction", r["extracted"]["abstract"]["value"])
        self.assertEqual(r["pages_stored"], 2)
        self.assertEqual(r["card_model"], "mock-extractive")
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (r["paper_id"],)).fetchone()
            n = c.execute("SELECT COUNT(*) n FROM pages WHERE paper_id=?",
                          (r["paper_id"],)).fetchone()["n"]
        self.assertEqual(n, 2)
        self.assertEqual(row["source"], "upload")
        self.assertEqual(row["file_sha256"], r["sha256"])
        self.assertEqual(row["pdf_path"], r["pdf_path"])
        self.assertEqual(json.loads(row["authors"]), ["Alice Smith", "Bob Lee", "Carol Wang"])
        self.assertTrue(Path(r["pdf_path"]).is_file())

    def test_make_card_false_skips_card(self):
        r = pdfimport.ingest_pdf(make_pdf(), "a.pdf", make_card=False)
        self.assertIsNone(r["card_model"])
        with db.conn() as c:
            self.assertIsNone(
                c.execute("SELECT card_json FROM papers WHERE id=?",
                          (r["paper_id"],)).fetchone()["card_json"])

    # ── 去重 ──

    def test_same_bytes_different_filename_is_duplicate(self):
        data = make_pdf()
        first = pdfimport.ingest_pdf(data, "原始文件名.pdf")
        second = pdfimport.ingest_pdf(data, "完全不同的名字 (1).pdf")
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["duplicate_by"], "sha256")
        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(second["extracted"], {})  # 不重复解析
        self.assertEqual(self._count_papers(), 1)
        self.assertTrue(any("sha256" in w for w in second["warnings"]))
        # 落盘文件按内容哈希命名，第二次不该多出一个文件
        self.assertEqual(len(list((config.DATA_DIR / "uploads").glob("*.pdf"))), 1)

    def test_norm_key_dedupe_attaches_pdf_to_searched_paper(self):
        """同一篇论文先被检索入库、后被本地上传：挂 PDF，不新建记录。"""
        with db.conn() as c:
            pid = db.insert_l0(c, {"norm_key": "doi:10.1234/tsp.2024.567890",
                                   "title": "Deep Learning for Near-Field Channel Estimation",
                                   "abstract": "from s2", "year": 2024,
                                   "doi": "10.1234/tsp.2024.567890", "source": "s2"})
        r = pdfimport.ingest_pdf(make_pdf(), "local copy.pdf")
        self.assertTrue(r["duplicate"])
        self.assertEqual(r["duplicate_by"], "norm_key")
        self.assertEqual(r["paper_id"], pid)
        self.assertEqual(self._count_papers(), 1)
        self.assertEqual(r["pages_stored"], 2)
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        self.assertEqual(row["file_sha256"], r["sha256"])
        self.assertTrue(row["pdf_path"].endswith(".pdf"))

    def test_second_file_for_same_paper_does_not_silently_replace(self):
        first = pdfimport.ingest_pdf(make_pdf(), "v1.pdf")
        # 同 DOI、不同排版（标题字号不同 → 字节不同 → sha 不同）
        other = make_pdf(title_size=20.0)
        second = pdfimport.ingest_pdf(other, "v2.pdf")
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["duplicate_by"], "norm_key")
        self.assertEqual(self._count_papers(), 1)
        self.assertTrue(any("未替换关联" in w for w in second["warnings"]))
        with db.conn() as c:
            row = c.execute("SELECT file_sha256 FROM papers WHERE id=?",
                            (first["paper_id"],)).fetchone()
        self.assertEqual(row["file_sha256"], first["sha256"])
        self.assertEqual(len(list((config.DATA_DIR / "uploads").glob("*.pdf"))), 1)

    # ── 安全约束 ──

    def test_reject_non_pdf(self):
        with self.assertRaises(pdfimport.PdfImportError) as ctx:
            pdfimport.ingest_pdf(b"<html>not a pdf</html>", "evil.pdf")
        self.assertIn("%PDF-", str(ctx.exception))
        self.assertEqual(self._count_papers(), 0)

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            pdfimport.ingest_pdf(b"", "empty.pdf")

    def test_reject_oversize(self):
        big = b"%PDF-" + b"0" * pdfimport.MAX_PDF_BYTES
        with self.assertRaises(pdfimport.PdfImportError) as ctx:
            pdfimport.ingest_pdf(big, "big.pdf")
        self.assertIn("50MB", str(ctx.exception))
        self.assertEqual(self._count_papers(), 0)

    def test_corrupt_pdf_is_rejected_not_half_ingested(self):
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.ingest_pdf(b"%PDF-1.4 garbage not really a pdf", "broken.pdf")
        self.assertEqual(self._count_papers(), 0)

    def test_sanitize_filename_kills_traversal(self):
        for raw, want in [("../../evil.pdf", "evil.pdf"),
                          (r"C:\windows\system32\x.pdf", "x.pdf"),
                          ("....//....//etc/passwd", "passwd"),
                          ("..", "upload.pdf"),
                          ("", "upload.pdf"),
                          ("a/b\\c.pdf", "c.pdf")]:
            got = pdfimport.sanitize_filename(raw)
            self.assertEqual(got, want)
            self.assertNotIn("/", got)
            self.assertNotIn("\\", got)

    def test_malicious_filename_never_escapes_uploads_dir(self):
        r = pdfimport.ingest_pdf(make_pdf(), "../../../../evil.pdf")
        path = Path(r["pdf_path"]).resolve()
        uploads = (config.DATA_DIR / "uploads").resolve()
        self.assertEqual(path.parent, uploads)
        self.assertEqual(path.name, f"{r['sha256'][:16]}.pdf")
        self.assertEqual([p.name for p in uploads.glob("*")], [path.name])
        # 上级目录里不该凭空多出文件
        self.assertFalse((Path(self._tmp) / "evil.pdf").exists())
        self.assertFalse((Path(self._tmp).parent / "evil.pdf").exists())

    # ── 抽不到就留空，不编造 ──

    def test_unextractable_metadata_warns_and_fabricates_nothing(self):
        data = blank_pdf(meta={"title": "Microsoft Word - paper.docx",
                               "author": "Administrator"})
        r = pdfimport.ingest_pdf(data, "扫描件 2019.pdf")
        ex = r["extracted"]
        self.assertIsNone(ex["authors"]["value"])
        self.assertIsNone(ex["abstract"]["value"])
        self.assertIsNone(ex["doi"]["value"])
        self.assertEqual(ex["title"]["from"], "filename-fallback")
        self.assertEqual(r["title"], "扫描件 2019")
        self.assertTrue(r["norm_key"].startswith("title:"))
        joined = " ".join(r["warnings"])
        self.assertIn("请手工修正", joined)
        self.assertIn("作者抽取失败", joined)
        self.assertIn("Microsoft Word - paper.docx", joined)  # 垃圾标题如实报告
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (r["paper_id"],)).fetchone()
        self.assertEqual(json.loads(row["authors"]), [])
        self.assertIsNone(row["abstract"])

    def test_junk_metadata_title_is_ignored_in_favor_of_layout(self):
        data = make_pdf(meta={"title": "Microsoft Word - draft_v3_final.docx"})
        r = pdfimport.ingest_pdf(data, "x.pdf")
        self.assertTrue(r["title"].startswith("Deep Learning for Near-Field"))
        self.assertTrue(any("排版软件垃圾" in w for w in r["warnings"]))

    def test_metadata_title_agreeing_with_layout_is_marked_as_such(self):
        title = "Deep Learning for Near-Field Channel Estimation in XL-MIMO"
        data = make_pdf(meta={"title": title})
        r = pdfimport.ingest_pdf(data, "x.pdf")
        self.assertEqual(r["extracted"]["title"]["from"],
                         "largest-font-block+pdf-metadata")

    def test_no_doi_falls_back_to_title_hash_with_warning(self):
        data = make_pdf(doi="", arxiv="")
        r = pdfimport.ingest_pdf(data, "x.pdf")
        self.assertTrue(r["norm_key"].startswith("title:"))
        self.assertTrue(any("退化为标题哈希" in w for w in r["warnings"]))
        # 正文里没有任何年份 → 留空 + 如实告警，不用 PDF 创建日期冒充发表年
        self.assertIsNone(r["extracted"]["year"]["value"])
        self.assertEqual(r["extracted"]["year"]["from"], "none")
        self.assertTrue(any("年份抽取失败" in w for w in r["warnings"]))

    def test_journal_header_above_title_does_not_win(self):
        """页眉在标题上方且字号更小：「最大字号块」必须胜过「第一行文本」。"""
        data = make_pdf(header="IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 72, 2024",
                        doi="", arxiv="")
        r = pdfimport.ingest_pdf(data, "x.pdf")
        self.assertEqual(r["title"],
                         "Deep Learning for Near-Field Channel Estimation in XL-MIMO")
        self.assertEqual(r["extracted"]["title"]["from"], "largest-font-block")
        self.assertEqual(r["extracted"]["year"]["value"], 2024)  # 年份从页眉捡到

    def test_hard_layout_arxiv_ieee_style(self):
        r = pdfimport.ingest_pdf(hard_layout_pdf(), "1234.pdf")
        ex = r["extracted"]
        # 跨行断词的标题要拼回去，竖排 arXiv 戳不能被当成标题
        self.assertEqual(ex["title"]["value"],
                         "Retrieval-Augmented Generation for Long-Context Models")
        # 上标标记剥掉、单位与邮箱不许混进作者
        self.assertEqual(ex["authors"]["value"],
                         ["Alice Smith", "Bob Q. Lee", "Carol Wang"])
        self.assertEqual(ex["arxiv_id"]["value"], "2403.09876")
        self.assertEqual(r["norm_key"], "arxiv:2403.09876")
        self.assertEqual(ex["year"]["value"], 2024)
        self.assertEqual(ex["year"]["from"], "arxiv-id-prefix")
        ab = ex["abstract"]["value"]
        self.assertTrue(ab.startswith("We study retrieval augmentation"), ab)
        self.assertNotIn("Index Terms", ab)

    def test_sha_hit_restores_missing_file_on_disk(self):
        data = make_pdf()
        first = pdfimport.ingest_pdf(data, "x.pdf")
        Path(first["pdf_path"]).unlink()
        second = pdfimport.ingest_pdf(data, "x.pdf")
        self.assertTrue(second["duplicate"])
        self.assertTrue(Path(second["pdf_path"]).is_file())
        self.assertTrue(any("补回" in w for w in second["warnings"]))
        self.assertEqual(self._count_papers(), 1)

    # ── 人工修正 ──

    def test_update_metadata_changes_norm_key_and_fts(self):
        r = pdfimport.ingest_pdf(make_pdf(doi="", arxiv=""), "x.pdf")
        old_key = r["norm_key"]
        out = pdfimport.update_metadata(r["paper_id"],
                                        title="Quantum Channel Estimation Revisited",
                                        year=2023, authors=["Dana Ho"])
        self.assertTrue(out["norm_key_changed"])
        self.assertNotEqual(out["norm_key"], old_key)
        self.assertTrue(out["norm_key"].startswith("title:"))
        self.assertIn("title", out["changed"])
        with db.conn() as c:
            hits = db.search_fts(c, "Revisited")
            row = c.execute("SELECT * FROM papers WHERE id=?", (r["paper_id"],)).fetchone()
        self.assertEqual([h["id"] for h in hits], [r["paper_id"]])
        self.assertEqual(row["year"], 2023)
        self.assertEqual(json.loads(row["authors"]), ["Dana Ho"])

    def test_update_metadata_doi_recomputes_key(self):
        r = pdfimport.ingest_pdf(make_pdf(doi="", arxiv=""), "x.pdf")
        out = pdfimport.update_metadata(r["paper_id"], doi="https://doi.org/10.9999/abc.1")
        self.assertEqual(out["norm_key"], "doi:10.9999/abc.1")

    def test_update_metadata_conflict_raises_and_changes_nothing(self):
        a = pdfimport.ingest_pdf(make_pdf(doi="", arxiv=""), "a.pdf")
        b = pdfimport.ingest_pdf(make_pdf(title_lines=("Another Totally Different Title",),
                                          doi="", arxiv=""), "b.pdf")
        self.assertNotEqual(a["paper_id"], b["paper_id"])
        with self.assertRaises(pdfimport.PdfImportError) as ctx:
            pdfimport.update_metadata(b["paper_id"], title=a["title"])
        self.assertIn("冲突", str(ctx.exception))
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (b["paper_id"],)).fetchone()
        self.assertEqual(row["norm_key"], b["norm_key"])
        self.assertEqual(row["title"], b["title"])

    def test_update_metadata_rejects_unknown_field_and_bad_year(self):
        r = pdfimport.ingest_pdf(make_pdf(), "x.pdf")
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.update_metadata(r["paper_id"], level=2)
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.update_metadata(r["paper_id"], year="去年")
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.update_metadata(r["paper_id"], title="   ")
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.update_metadata(99999, title="不存在的论文")

    # ── 回归：版面标题不该被「内嵌元数据垃圾词表」误杀 ──

    def test_real_titles_starting_with_junk_words_survive(self):
        """"Template Matching…"/"Layout Analysis…" 是成片存在的真实论文题目。

        排版软件垃圾词表（^template/^layout/^document/^paper/^output/^ms…）只针对
        PDF **内嵌**元数据标题；一旦套到版面最大字号块上，这些题目会被整片毙掉，
        标题退化成文件名、norm_key 退化成文件名哈希，同一篇论文与检索入库的
        那条记录就永远对不上了。
        """
        for title in ["Template Matching for Object Detection in Aerial Images",
                      "Layout Analysis of Historical Newspaper Documents",
                      "Document Understanding with Vision Transformers",
                      "Output-Sensitive Algorithms for Convex Hulls",
                      "MS MARCO: A Human Generated Reading Comprehension Dataset",
                      "TCP/IP Congestion Control for Modern Data Centers"]:
            with self.subTest(title=title):
                ex = pdfimport.extract_metadata(title_only_pdf(title))["extracted"]
                self.assertEqual(ex["title"]["value"], title)
                self.assertEqual(ex["title"]["from"], "largest-font-block")

    def test_junk_wordlist_still_guards_embedded_metadata_title(self):
        """放宽版面判据不等于放行内嵌垃圾：'Template' 这种仍必须被毙掉。"""
        r = pdfimport.ingest_pdf(blank_pdf(meta={"title": "Template", "author": ""}),
                                 "扫描件.pdf")
        self.assertEqual(r["extracted"]["title"]["from"], "filename-fallback")
        self.assertTrue(any("排版软件垃圾" in w for w in r["warnings"]))

    def test_path_like_embedded_title_is_still_rejected(self):
        r = pdfimport.ingest_pdf(
            blank_pdf(meta={"title": r"C:\Users\me\Desktop\draft.docx", "author": ""}),
            "扫描件.pdf")
        self.assertEqual(r["extracted"]["title"]["from"], "filename-fallback")

    # ── 回归：标题里含 Abstract 一词不该污染摘要 ──

    def test_abstract_word_in_title_does_not_poison_abstract(self):
        """"Abstract Meaning Representation …" 这类题目里的 Abstract 不是小标题。

        取第一个匹配会把「标题后半句 + 作者行」当成摘要写进库，而 from 还标着
        keyword:Abstract→Introduction —— 那就是在编造。
        """
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((72, 110), "Abstract Meaning Representation Parsing at Scale",
                      fontsize=18, fontname="hebo")
        p.insert_text((72, 150), "Alice Smith, Bob Lee", fontsize=11)
        p.insert_text((72, 200), "Abstract", fontsize=12, fontname="hebo")
        p.insert_text((72, 215),
                      "We present a fast AMR parser that improves smatch by 3 points.",
                      fontsize=10)
        p.insert_text((72, 250), "1 Introduction", fontsize=12, fontname="hebo")
        data = doc.tobytes()
        doc.close()
        ex = pdfimport.extract_metadata(data)["extracted"]
        self.assertEqual(ex["title"]["value"],
                         "Abstract Meaning Representation Parsing at Scale")
        self.assertTrue(ex["abstract"]["value"].startswith("We present a fast AMR"),
                        ex["abstract"]["value"])
        self.assertNotIn("Alice Smith", ex["abstract"]["value"])
        self.assertNotIn("Meaning Representation", ex["abstract"]["value"])

    # ── 回归：不编造作者 ──

    def test_chinese_paper_end_to_end_and_city_is_not_an_author(self):
        """中文单位行「清华大学电子工程系, 北京」不许贡献出一位叫「北京」的作者。"""
        r = pdfimport.ingest_pdf(chinese_pdf(), "中文论文.pdf")
        ex = r["extracted"]
        self.assertEqual(ex["title"]["value"], "基于深度学习的近场信道估计方法")
        self.assertEqual(ex["authors"]["value"], ["张三", "李四", "王五"])
        self.assertIn("近场信道估计", ex["abstract"]["value"])
        self.assertNotIn("关键词", ex["abstract"]["value"])
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?",
                            (r["paper_id"],)).fetchone()
        self.assertEqual(json.loads(row["authors"]), ["张三", "李四", "王五"])

    def test_author_line_heuristics_fail_closed(self):
        f = pdfimport._authors_from_line
        # 姓在前的写法逐段判只剩半个「Bob Q」——那不是任何一位真实作者
        self.assertEqual(f("Smith, Alice; Lee, Bob Q."), [])
        # 整行含机构/地名线索 → 整行作废
        self.assertEqual(f("清华大学电子工程系, 北京"), [])
        self.assertEqual(f("Tsinghua University, Beijing, China"), [])
        self.assertEqual(f("{alice,bob}@example.edu"), [])
        self.assertEqual(f("Madonna"), [])
        # 正常作者行照常工作，And/AND/and 都要能切
        self.assertEqual(f("Alice Smith And Bob Lee"), ["Alice Smith", "Bob Lee"])
        self.assertEqual(f("A. Smith, B. Q. Lee, C. Wang"),
                         ["A. Smith", "B. Q. Lee", "C. Wang"])

    # ── 回归：挂到既有论文时不能吞掉卡片与新信息 ──

    def test_attach_fills_only_blank_fields_and_makes_card(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {"norm_key": "doi:10.1234/tsp.2024.567890",
                                   "title": "标题以既有记录为准", "abstract": "既有摘要不许被覆盖",
                                   "year": None, "authors": [], "source": "s2",
                                   "doi": "10.1234/tsp.2024.567890"})
        r = pdfimport.ingest_pdf(make_pdf(), "local.pdf", make_card=True)
        self.assertEqual(r["paper_id"], pid)
        self.assertEqual(r["card_model"], "mock-extractive")  # 不能静默无卡片
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        self.assertEqual(row["abstract"], "既有摘要不许被覆盖")   # 非空字段不覆盖
        self.assertEqual(row["title"], "标题以既有记录为准")
        self.assertEqual(row["year"], 2024)                      # 空字段补齐
        self.assertEqual(json.loads(row["authors"]),
                         ["Alice Smith", "Bob Lee", "Carol Wang"])
        self.assertIsNotNone(row["card_json"])
        self.assertTrue(any("补齐既有记录的空字段" in w for w in r["warnings"]))

    # ── 此前没有任何用例走到的分支 ──

    def test_encrypted_pdf_is_rejected_not_half_ingested(self):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "secret", fontsize=12)
        enc = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="o", user_pw="u")
        doc.close()
        with self.assertRaises(pdfimport.PdfImportError) as ctx:
            pdfimport.ingest_pdf(enc, "locked.pdf")
        self.assertIn("加密", str(ctx.exception))
        self.assertEqual(self._count_papers(), 0)
        self.assertFalse(list((config.DATA_DIR / "uploads").glob("*.pdf"))
                         if (config.DATA_DIR / "uploads").exists() else [])

    def test_creation_date_year_is_labelled_as_not_publication_year(self):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 100), "Nothing useful here", fontsize=8)
        doc.set_metadata({"title": "", "author": "", "creationDate": "D:20210607120000Z"})
        data = doc.tobytes()
        doc.close()
        out = pdfimport.extract_metadata(data)
        ex = out["extracted"]
        self.assertEqual(ex["year"]["value"], 2021)
        self.assertIn("非发表年", ex["year"]["from"])   # 来源必须自曝可疑
        # papers.year 这一列存不下「仅供参考」，所以 warnings 里也必须点名
        self.assertTrue(any("不是发表年" in w for w in out["warnings"]),
                        out["warnings"])

    def test_page_cap_is_reported_not_silently_truncated(self):
        doc = pymupdf.open()
        for i in range(pdfimport.MAX_PAGES + 5):
            doc.new_page().insert_text((72, 100), f"Page {i + 1} body text.", fontsize=10)
        doc.set_metadata({"title": "A Very Long Report On Something", "author": ""})
        data = doc.tobytes()
        doc.close()
        r = pdfimport.ingest_pdf(data, "long.pdf", make_card=False)
        self.assertEqual(r["extracted"]["pages"]["value"], pdfimport.MAX_PAGES + 5)
        self.assertEqual(r["pages_stored"], pdfimport.MAX_PAGES)
        self.assertTrue(any("不会被检索到" in w for w in r["warnings"]), r["warnings"])

    def test_ensure_schema_backfills_column_on_old_db(self):
        old = Path(self._tmp) / "old.db"
        c = sqlite3.connect(old)
        c.executescript("CREATE TABLE papers(id INTEGER PRIMARY KEY, norm_key TEXT "
                        "UNIQUE, title TEXT, pdf_path TEXT);")
        c.commit()
        c.close()
        prev, config.DB_PATH = config.DB_PATH, old
        try:
            pdfimport.ensure_schema()
            pdfimport.ensure_schema()          # 幂等
            c = sqlite3.connect(old)
            cols = {r[1] for r in c.execute("PRAGMA table_info(papers)")}
            c.close()
        finally:
            config.DB_PATH = prev
        self.assertIn("file_sha256", cols)

    def test_concurrent_upload_of_same_new_pdf_creates_one_paper(self):
        """UNIQUE(norm_key) 撞车后要退回「挂到既有论文」，不能把 IntegrityError 抛给用户。"""
        data = make_pdf(title_lines=("Concurrent Upload Race Condition Title",),
                        doi="", arxiv="")
        out, errs = [], []

        def worker():
            try:
                out.append(pdfimport.ingest_pdf(data, "race.pdf", make_card=False))
            except Exception as e:      # noqa: BLE001 - 测试要看到任何异常
                errs.append(e)

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(errs, [])
        self.assertEqual(self._count_papers(), 1)
        self.assertEqual(len({r["paper_id"] for r in out}), 1)

    def test_sanitize_filename_keeps_text_before_colon(self):
        # 冒号不是路径分隔符：按它切会把标题兜底值砍掉半句
        self.assertEqual(pdfimport.sanitize_filename("Attention: All You Need.pdf"),
                         "Attention_ All You Need.pdf")
        self.assertEqual(pdfimport.sanitize_filename("x.pdf:evil"), "x.pdf_evil")

    def test_list_uploads(self):
        r = pdfimport.ingest_pdf(make_pdf(), "x.pdf")
        rows = pdfimport.list_uploads()
        self.assertEqual([x["id"] for x in rows], [r["paper_id"]])

    def test_ingest_file_reads_from_disk(self):
        p = Path(self._tmp) / "on disk.pdf"
        p.write_bytes(make_pdf())
        r = pdfimport.ingest_file(p)
        self.assertFalse(r["duplicate"])
        self.assertEqual(r["filename"], "on disk.pdf")
        with self.assertRaises(pdfimport.PdfImportError):
            pdfimport.ingest_file(Path(self._tmp) / "nope.pdf")


if __name__ == "__main__":
    unittest.main()
