"""BibTeX / RIS / Zotero CSV 导入。全程离线确定性：向量按住、S2 一律 mock。

重点不是「能解析」，而是三件容易悄悄坏掉的事：
  ① 一条坏 entry 不能吞掉整份文件；
  ② 与 cite.py 导出的往返一致（导出→导入回来字段不变形）；
  ③ 重复导入不增行、且「补空不覆盖」——库内人工修订永远优先。
"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from papernest import bibimport, cite, config, db, http

# ── 样本 ──

# 混合 entry type + 注释行 + @string/@comment/@preamble + 嵌套花括号 + 重音 + 两种作者写法
BIB_GOOD = r"""
% 这一行是文件级注释，应当被跳过
@preamble{ "\newcommand{\noop}[1]{}" }
@comment{jabref-meta: databaseType:bibtex;}
@string{tmlr = "Transactions on Machine Learning Research"}

@article{devlin2019bert,
  title     = {{BERT}: Pre-training of Deep {Bidirectional} Transformers},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton},
  journal   = tmlr,
  year      = {2019},
  doi       = {10.18653/v1/N19-1423},
  abstract  = {We introduce a new language representation model
               called BERT.}
}

@inproceedings{muller2021survey,
  Title     = "The {\"O}sterreich Study of Caf{\'e}s \& Bars",
  AUTHOR    = {Bj{\"o}rn M{\"u}ller and Jos{\'e} Garc{\'i}a},
  booktitle = {Proceedings of the 42nd Conference},
  YEAR      = 2021,
  url       = {https://example.org/papers/muller.pdf}
}

@misc{vaswani2017,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam and others},
  eprint = {1706.03762},
  archivePrefix = {arXiv},
  year = {2017}
}

@book{knuth1984,
  title = {The {\TeX}book},
  author = {Knuth, Donald E.},
  publisher = {Addison-Wesley},
  year = {1984}
}
"""

# 第 2 条缺右花括号（下一条 @ 顶上来），第 3 条字段没等号且没 title
BIB_BROKEN = r"""
@article{ok1,
  title = {First Good Paper},
  author = {Alice Smith},
  year = {2020},
  doi = {10.1000/ok1}
}

@article{broken1,
  title = {Missing Closing Brace Paper},
  author = {Bob Lee},
  year = {2021},
  doi = {10.1000/broken1}

@article{junk1,
  this field has no equals sign,
  year = {2022}
}

@article{ok2,
  title = {Second Good Paper},
  author = {Carol Ng},
  year = {2023},
  doi = {10.1000/ok2}
}
"""

RIS_SAMPLE = """TY  - JOUR
TI  - A Very Long Title That
  Continues On The Next Line
AU  - Smith, Alice
AU  - Lee, Bob
PY  - 2023/06/01
JO  - Journal of Tests
DO  - 10.5555/ris-one
AB  - First line of the abstract.
  Second line of the abstract.
ER  -

TY  - CONF
TI  - Conference RIS Entry
AU  - Carol Ng
PY  - 2022
T2  - Proceedings of Nowhere
UR  - https://arxiv.org/abs/2201.01234
ER  -
"""

CSV_SAMPLE = (
    "Key,Item Type,Publication Year,Author,Title,Publication,DOI,Abstract Note,Url\n"
    "ABCD1234,journalArticle,2020,\"Smith, Alice; Lee, Bob\",CSV Imported Paper,"
    "Journal of CSV,10.7777/csv-one,\"An abstract, with a comma.\","
    "https://doi.org/10.7777/csv-one\n"
    "EFGH5678,preprint,2024,\"García, José\",CSV ArXiv Paper,,,,"
    "https://arxiv.org/abs/2401.09999\n"
)


class BibImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_bibimport_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 开发机 .env 真配了 EMBED_MODEL，不按住会真发请求
        self._embed_patch = mock.patch("papernest.embeddings.available", return_value=False)
        self._embed_patch.start()
        # 任何走到网络的路径都应当先被显式 mock；漏了就让它当场炸，而不是偷偷联网。
        # 但 _s2_get 本身（404 / 重试 / 退避 / URL 拼接）也是要测的代码，
        # 整个换掉就没人跑它了——先留一份真身，那几个用例用假 transport 驱动它。
        self._real_s2_get = bibimport._s2_get
        self._net_patch = mock.patch.object(
            bibimport, "_s2_get",
            side_effect=AssertionError("测试中不允许真的访问 Semantic Scholar"))
        self._net_patch.start()
        db.init_db()
        bibimport.ensure_schema()

    def tearDown(self):
        self._net_patch.stop()
        self._embed_patch.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 工具 ──

    @staticmethod
    def _by_title(papers, needle):
        return next(p for p in papers if needle.lower() in p["title"].lower())

    @staticmethod
    def _count():
        with db.conn() as c:
            return c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]

    @staticmethod
    def _mock_http(handler):
        """把 papernest.http.client 换成挂了 MockTransport 的 httpx.Client。

        比 mock 掉 _s2_get 低一层：这样 _s2_get 里的 404 分支、重试分支、
        退避时长、URL 拼接才真的被执行到，同时一个字节都不出网。
        """
        return mock.patch.object(
            http, "client",
            lambda *a, **k: httpx.Client(transport=httpx.MockTransport(handler)))

    @staticmethod
    def _insert(**kw):
        p = {"norm_key": None, "title": "", "abstract": None, "year": None, "venue": None,
             "authors": [], "doi": None, "arxiv_id": None, "source": "s2"}
        p.update(kw)
        with db.conn() as c:
            return db.insert_l0(c, p)

    # ── BibTeX 解析 ──

    def test_nested_braces_and_case_protection(self):
        errs = []
        papers = bibimport.parse_bibtex(BIB_GOOD, errs)
        bert = self._by_title(papers, "pre-training")
        self.assertEqual(bert["title"],
                         "BERT: Pre-training of Deep Bidirectional Transformers")
        self.assertEqual(bert["doi"], "10.18653/v1/N19-1423")
        self.assertEqual(bert["abstract"],
                         "We introduce a new language representation model called BERT.")
        # @string 宏展开
        self.assertEqual(bert["venue"], "Transactions on Machine Learning Research")

    def test_latex_accents_and_escapes(self):
        papers = bibimport.parse_bibtex(BIB_GOOD)
        p = self._by_title(papers, "sterreich")
        self.assertEqual(p["title"], "The Österreich Study of Cafés & Bars")
        self.assertEqual(p["authors"], ["Björn Müller", "José García"])

    def test_author_both_styles(self):
        papers = bibimport.parse_bibtex(BIB_GOOD)
        # "Last, First and Last, First"
        self.assertEqual(self._by_title(papers, "pre-training")["authors"],
                         ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"])
        # "First Last and First Last"
        self.assertEqual(self._by_title(papers, "sterreich")["authors"],
                         ["Björn Müller", "José García"])
        # `others` 是 et-al 标记不是人名，不能当作者写进去
        self.assertEqual(self._by_title(papers, "attention")["authors"],
                         ["Ashish Vaswani", "Noam Shazeer"])

    def test_mixed_entry_types_and_skipped_blocks(self):
        errs = []
        papers = bibimport.parse_bibtex(BIB_GOOD, errs)
        self.assertEqual(len(papers), 4, f"应解析出 4 条（跳过 string/comment/preamble），errs={errs}")
        self.assertEqual(errs, [])
        # venue 取 journal → booktitle → publisher 第一个非空
        self.assertEqual(self._by_title(papers, "sterreich")["venue"],
                         "Proceedings of the 42nd Conference")
        self.assertEqual(self._by_title(papers, "book")["venue"], "Addison-Wesley")
        # 裸词年份（YEAR = 2021，无花括号无引号）
        self.assertEqual(self._by_title(papers, "sterreich")["year"], 2021)
        # 不认识的命令 \TeX 原样保留，不炸也不吞内容
        self.assertIn("TeX", self._by_title(papers, "book")["title"])

    def test_arxiv_and_pdf_url_extraction(self):
        papers = bibimport.parse_bibtex(BIB_GOOD)
        att = self._by_title(papers, "attention")
        self.assertEqual(att["arxiv_id"], "1706.03762")
        self.assertEqual(att["norm_key"], "arxiv:1706.03762")
        # url 指向 .pdf → 认成 oa_pdf_url；落地页则留空（不编造）
        self.assertEqual(self._by_title(papers, "sterreich")["oa_pdf_url"],
                         "https://example.org/papers/muller.pdf")
        self.assertIsNone(self._by_title(papers, "book")["oa_pdf_url"])

    def test_broken_entry_does_not_kill_the_file(self):
        errs = []
        papers = bibimport.parse_bibtex(BIB_BROKEN, errs)
        titles = [p["title"] for p in papers]
        self.assertIn("First Good Paper", titles)
        self.assertIn("Second Good Paper", titles)
        self.assertNotIn(None, titles)
        # 缺右花括号：记错误 + 按已读到的字段尽力解析（不是整份文件报废）
        self.assertTrue(any("缺少右花括号" in e for e in errs), errs)
        # 字段没等号 + 无 title：该条整体跳过并记录
        self.assertTrue(any("没有等号" in e for e in errs), errs)
        self.assertTrue(any("没有 title" in e for e in errs), errs)
        self.assertNotIn("Missing Closing Brace Paper", [t for t in titles if t is None])

        res = bibimport.import_text(BIB_BROKEN, fmt="bibtex")
        self.assertGreaterEqual(res["imported"], 2)
        with db.conn() as c:
            got = {r["title"] for r in c.execute("SELECT title FROM papers")}
        self.assertIn("First Good Paper", got)
        self.assertIn("Second Good Paper", got)

    # ── 与 cite.py 的往返一致性 ──

    def test_roundtrip_with_cite_export_bibtex(self):
        src = {
            "norm_key": "doi:10.1234/ab_cd",
            "title": "Round {Trip} & 50% Coverage",   # 花括号 + & + % 三种转义
            "abstract": "An abstract.", "year": 2024,
            "venue": "Journal of Tests & Trials",
            "authors": ["Björn Müller", "Alice Smith"],
            "doi": "10.1234/ab_cd", "arxiv_id": "2401.00001", "source": "s2",
        }
        with db.conn() as c:
            pid = db.insert_l0(c, src)
        text = cite.export([pid], "bibtex")

        back = bibimport.parse_bibtex(text)
        self.assertEqual(len(back), 1, text)
        got = back[0]
        for field in ("title", "year", "doi", "arxiv_id", "venue"):
            self.assertEqual(got[field], src[field], f"{field} 往返不一致\n{text}")
        self.assertEqual(got["authors"], src["authors"], text)
        self.assertEqual(got["norm_key"], src["norm_key"])

        # 再导入一次：同一篇只占一个 norm_key，库内不增行
        res = bibimport.import_entries(back)
        self.assertEqual((res["imported"], res["skipped"]), (0, 1), res)
        self.assertEqual(self._count(), 1)

    def test_roundtrip_with_cite_export_ris(self):
        src = {"norm_key": "doi:10.9999/ris-rt", "title": "RIS Round Trip Paper",
               "abstract": None, "year": 2022, "venue": "Journal of RIS",
               "authors": ["Alice Smith", "Bob Lee"], "doi": "10.9999/ris-rt",
               "arxiv_id": "2202.00002", "source": "s2"}
        with db.conn() as c:
            pid = db.insert_l0(c, src)
        text = cite.export([pid], "ris")
        got = bibimport.parse_ris(text)
        self.assertEqual(len(got), 1, text)
        self.assertEqual(got[0]["title"], src["title"])
        self.assertEqual(got[0]["authors"], src["authors"])
        self.assertEqual(got[0]["year"], 2022)
        self.assertEqual(got[0]["doi"], src["doi"])
        self.assertEqual(got[0]["arxiv_id"], src["arxiv_id"])
        self.assertEqual(got[0]["norm_key"], src["norm_key"])

    # ── RIS ──

    def test_ris_multiline_fields(self):
        errs = []
        papers = bibimport.parse_ris(RIS_SAMPLE, errs)
        self.assertEqual(errs, [])
        self.assertEqual(len(papers), 2)
        a = papers[0]
        self.assertEqual(a["title"], "A Very Long Title That Continues On The Next Line")
        self.assertEqual(a["abstract"],
                         "First line of the abstract. Second line of the abstract.")
        self.assertEqual(a["authors"], ["Alice Smith", "Bob Lee"])
        self.assertEqual(a["year"], 2023)
        self.assertEqual(a["venue"], "Journal of Tests")
        self.assertEqual(a["doi"], "10.5555/ris-one")
        self.assertEqual(a["source"], "ris")
        b = papers[1]
        self.assertEqual(b["venue"], "Proceedings of Nowhere")
        self.assertEqual(b["arxiv_id"], "2201.01234")

    def test_ris_missing_er_is_recovered_with_error(self):
        errs = []
        papers = bibimport.parse_ris("TY  - JOUR\nTI  - No End Marker\nPY  - 2020\n", errs)
        self.assertEqual([p["title"] for p in papers], ["No End Marker"])
        self.assertTrue(any("缺少 ER" in e for e in errs), errs)

    # ── Zotero CSV ──

    def test_zotero_csv(self):
        errs = []
        papers = bibimport.parse_zotero_csv(CSV_SAMPLE, errs)
        self.assertEqual(errs, [])
        self.assertEqual(len(papers), 2)
        a, b = papers
        self.assertEqual(a["title"], "CSV Imported Paper")
        self.assertEqual(a["authors"], ["Alice Smith", "Bob Lee"])
        self.assertEqual(a["year"], 2020)
        self.assertEqual(a["venue"], "Journal of CSV")
        self.assertEqual(a["doi"], "10.7777/csv-one")
        self.assertEqual(a["abstract"], "An abstract, with a comma.")
        self.assertEqual(a["source"], "zotero-csv")
        self.assertEqual(b["authors"], ["José García"])
        self.assertEqual(b["arxiv_id"], "2401.09999")
        self.assertIsNone(b["venue"])  # 空列如实留空，不猜

    def test_zotero_csv_column_name_tolerance(self):
        text = ("item_type,PUBLICATION year, Title ,DOI\n"
                "journalArticle,2019,Case Insensitive Columns,10.1/x\n")
        papers = bibimport.parse_zotero_csv(text)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Case Insensitive Columns")
        self.assertEqual(papers[0]["year"], 2019)
        self.assertEqual(papers[0]["doi"], "10.1/x")

    # ── 嗅探与顶层入口 ──

    def test_sniff_format(self):
        self.assertEqual(bibimport.sniff_format(BIB_GOOD), "bibtex")
        self.assertEqual(bibimport.sniff_format(RIS_SAMPLE), "ris")
        self.assertEqual(bibimport.sniff_format(CSV_SAMPLE), "csv")
        self.assertEqual(bibimport.sniff_format("just some prose"), "unknown")
        with self.assertRaises(ValueError):
            bibimport.import_text("just some prose")

    def test_import_text_auto(self):
        res = bibimport.import_text(BIB_GOOD, source_name="zotero.bib")
        self.assertEqual(res["format"], "bibtex")
        self.assertEqual(res["imported"], 4, res)
        self.assertEqual(res["failed"], 0, res)
        self.assertEqual(len(res["paper_ids"]), 4)
        runs = bibimport.list_imports()
        self.assertEqual(runs[0]["source_name"], "zotero.bib")
        self.assertEqual(runs[0]["imported"], 4)

    # ── 去重与「补空不覆盖」 ──

    def test_reimport_is_all_skipped(self):
        first = bibimport.import_text(BIB_GOOD)
        n = self._count()
        second = bibimport.import_text(BIB_GOOD)
        self.assertEqual(first["imported"], 4)
        self.assertEqual(second["imported"], 0, second)
        self.assertEqual(second["skipped"], 4, second)
        self.assertEqual(self._count(), n, "重复导入不得增加论文行数")

    def test_fill_empty_but_never_overwrite(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "doi:10.1000/merge", "title": "Canonical Title Kept By User",
                "abstract": None, "year": None, "venue": None, "authors": [],
                "doi": "10.1000/merge", "arxiv_id": None, "source": "s2"})
        text = r"""@article{m1,
  title = {A Different Title Written By The Bib File},
  author = {Alice Smith and Bob Lee},
  journal = {Journal of Merge},
  year = {2018},
  doi = {10.1000/merge},
  abstract = {The abstract that was missing.}
}"""
        res = bibimport.import_text(text)
        self.assertEqual((res["imported"], res["skipped"], res["updated"]), (0, 1, 1), res)
        self.assertEqual(res["updated_ids"], [pid])
        self.assertEqual(set(res["filled"][0]["fields"]),
                         {"abstract", "year", "venue", "authors"})
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        self.assertEqual(row["title"], "Canonical Title Kept By User", "已有标题不得被覆盖")
        self.assertEqual(row["abstract"], "The abstract that was missing.")
        self.assertEqual(row["year"], 2018)
        self.assertEqual(row["venue"], "Journal of Merge")
        self.assertEqual(json.loads(row["authors"]), ["Alice Smith", "Bob Lee"])
        self.assertEqual(row["level"], 1, "补上摘要后应与 insert_l0 同语义升到 L1")
        # 补上的摘要要能被 FTS 检索到（补空后索引必须重建）
        with db.conn() as c:
            hits = db.search_fts(c, "abstract that was missing")
        self.assertIn(pid, [r["id"] for r in hits])

    def test_fill_empty_does_not_clobber_existing_values(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "doi:10.1000/keep", "title": "Kept", "abstract": "库内原摘要",
                "year": 2010, "venue": "Original Venue", "authors": ["Zoe Original"],
                "doi": "10.1000/keep", "arxiv_id": None, "source": "s2"})
        res = bibimport.import_text(r"""@article{k1,
  title = {Kept}, author = {New Person}, journal = {New Venue},
  year = {2019}, doi = {10.1000/keep}, abstract = {new abstract}
}""")
        self.assertEqual((res["skipped"], res["updated"]), (1, 0), res)
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        self.assertEqual(row["abstract"], "库内原摘要")
        self.assertEqual(row["venue"], "Original Venue")
        self.assertEqual(row["year"], 2010)
        self.assertEqual(json.loads(row["authors"]), ["Zoe Original"])

    def test_entry_without_norm_key_counts_as_failed(self):
        res = bibimport.import_entries([{"title": "", "norm_key": None}])
        self.assertEqual(res["failed"], 1)
        self.assertTrue(res["errors"])

    # ── enrich：全程 mock，不联网 ──

    def test_enrich_fills_missing_abstract(self):
        payload = {"paperId": "S2ID123", "title": "Attention Is All You Need",
                   "abstract": "S2 补回来的摘要。", "year": 2017,
                   "venue": "NeurIPS", "citationCount": 12345,
                   "authors": [{"name": "Ashish Vaswani"}],
                   "openAccessPdf": {"url": "https://example.org/a.pdf"},
                   "externalIds": {"ArXiv": "1706.03762"}}
        with mock.patch.object(bibimport, "_s2_get", return_value=payload) as m:
            res = bibimport.import_text(BIB_GOOD, enrich=True)
        self.assertEqual(res["imported"], 4, res)
        # 只有「有 doi/arxiv 且缺 abstract」的条目才查：BERT 自带摘要不查，
        # Österreich / book 没有 doi 也没有 arXiv ID 查不了 —— 只剩 Attention 一条
        called = sorted(c.args[0] for c in m.call_args_list)
        self.assertEqual(called, ["arXiv:1706.03762"])
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE arxiv_id='1706.03762'").fetchone()
        self.assertEqual(row["abstract"], "S2 补回来的摘要。")
        self.assertEqual(row["s2_id"], "S2ID123")
        self.assertEqual(row["citation_count"], 12345)
        self.assertEqual(row["venue"], "NeurIPS")
        self.assertEqual(row["title"], "Attention Is All You Need",
                         "补全不得改写文件里的标题")

    def test_enrich_uses_doi_endpoint_form(self):
        """按 DOI 查单篇必须走 `DOI:{doi}` 这个标识符前缀，不是拼查询串。"""
        text = ("@article{d1, title = {No Abstract Here}, author = {A B}, "
                "year = {2020}, doi = {10.1000/needs-abstract}}")
        with mock.patch.object(bibimport, "_s2_get",
                               return_value={"abstract": "补回来了"}) as m:
            res = bibimport.import_text(text, enrich=True)
        m.assert_called_once_with("DOI:10.1000/needs-abstract")
        self.assertEqual(res["imported"], 1, res)
        with db.conn() as c:
            row = c.execute("SELECT abstract FROM papers").fetchone()
        self.assertEqual(row["abstract"], "补回来了")

    def test_enrich_failure_does_not_break_import(self):
        with mock.patch.object(bibimport, "_s2_get",
                               side_effect=RuntimeError("网络不通")):
            res = bibimport.import_text(BIB_GOOD, enrich=True)
        self.assertEqual(res["imported"], 4, res)
        self.assertEqual(res["failed"], 0, res)
        self.assertTrue(any("网络不通" in e for e in res["errors"]), res["errors"])
        # 失败就如实留空，不许拿任何东西凑数
        with db.conn() as c:
            row = c.execute("SELECT * FROM papers WHERE arxiv_id='1706.03762'").fetchone()
        self.assertIsNone(row["abstract"])

    def test_enrich_not_found_is_recorded_not_faked(self):
        with mock.patch.object(bibimport, "_s2_get", return_value=None):
            res = bibimport.import_text(BIB_GOOD, enrich=True)
        self.assertTrue(any("未收录" in e for e in res["errors"]), res["errors"])
        self.assertEqual(res["imported"], 4)

    def test_enrich_off_by_default_makes_no_network_call(self):
        # setUp 里 _s2_get 被换成会抛 AssertionError 的桩：真调了就当场失败
        res = bibimport.import_text(BIB_GOOD)
        self.assertEqual(res["imported"], 4, res)

    # ── 以下为复核补的回归用例：每条都对应一个实测复现过的缺陷 ──

    def test_folded_doi_and_url_do_not_poison_norm_key(self):
        """.bib 普遍在 80 列折行。折行留下的换行+缩进如果跟着 DOI 进了 norm_key，
        同一篇论文的折行版与不折行版会各占一行，去重（本模块的核心承诺）当场失效。"""
        folded = bibimport.parse_bibtex(r"""@article{w1,
  title = {Wrapped Identifier Paper},
  doi   = {10.1234/very-long-doi-that-is
           -wrapped},
  url   = {https://example.org/a/very/long/
           path/to/file.pdf}
}""")[0]
        flat = bibimport.parse_bibtex(
            "@article{w2, title = {Wrapped Identifier Paper}, "
            "doi = {10.1234/very-long-doi-that-is-wrapped}}")[0]
        self.assertEqual(folded["doi"], "10.1234/very-long-doi-that-is-wrapped")
        self.assertEqual(folded["norm_key"], flat["norm_key"], "折行与不折行必须同键")
        self.assertEqual(folded["oa_pdf_url"],
                         "https://example.org/a/very/long/path/to/file.pdf")

    def test_commented_out_entries_and_fields_are_not_imported(self):
        """用户在 .bib 里 `%` 掉的东西是明确划掉的，导进来比漏导更糟。"""
        papers = bibimport.parse_bibtex("""@article{live, title = {Live Paper}, year = {2020}}

% @article{deadone,
%   title = {Commented Out Paper},
%   author = {Ghost Writer},
% }
""")
        self.assertEqual([p["title"] for p in papers], ["Live Paper"])
        one = bibimport.parse_bibtex(
            "@article{pc, title = {Real Title},\n  % author = {Should Be Ignored},\n"
            "  year = {2020}\n}")[0]
        self.assertEqual(one["authors"], [])
        # 只吃「整行注释」——值里的 50\% 一根汗毛都不能动（cite.py 导出就会产生它）
        pct = bibimport.parse_bibtex(
            r"@article{q, title = {Coverage of 50\% and more}, year = {2020}}")[0]
        self.assertEqual(pct["title"], "Coverage of 50% and more")

    def test_failed_counts_lost_entries_not_warnings(self):
        """failed 是「几篇没进来」，不是「几条错误信息」。
        BIB_BROKEN 里 3 条进库、只有 junk1 真丢了，failed 必须是 1 而不是 3。"""
        res = bibimport.import_text(BIB_BROKEN, fmt="bibtex")
        self.assertEqual((res["parsed"], res["imported"], res["failed"]), (3, 3, 1), res)
        # 警告仍要如实出现在 errors 里，只是不计进 failed
        self.assertTrue(any("缺少右花括号" in e for e in res["errors"]), res["errors"])
        self.assertTrue(any("没有等号" in e for e in res["errors"]), res["errors"])

    def test_ris_consecutive_records_without_er_are_not_merged(self):
        """两条都缺 ER 时，早先会被合并成一条：第二篇论文静默消失且 errors 里毫无痕迹。"""
        errs = []
        papers = bibimport.parse_ris(
            "TY  - JOUR\nTI  - Record One\nPY  - 2020\n\n"
            "TY  - JOUR\nTI  - Record Two\nPY  - 2021\n", errs)
        self.assertEqual([p["title"] for p in papers], ["Record One", "Record Two"])
        self.assertEqual(len([e for e in errs if "缺少 ER" in e]), 2, errs)
        # 正常带 ER 的文件不能被这条补救误伤
        errs2 = []
        ok = bibimport.parse_ris("TY  - JOUR\nTI  - A\nER  - \n\nTY  - JOUR\nTI  - B\nER  - \n",
                                 errs2)
        self.assertEqual(([p["title"] for p in ok], errs2), (["A", "B"], []))

    def test_merge_fill_keeps_keywords_and_survives_broken_card_json(self):
        """补空要重建 FTS，重建时 keywords 必须从 card 里取回来；
        card_json 万一不是对象也不能把整条「补空」搞成入库失败。"""
        kw = self._insert(norm_key="doi:10.1/kw", title="KW Row", doi="10.1/kw")
        bad = self._insert(norm_key="doi:10.1/bad", title="Bad Card", doi="10.1/bad")
        with db.conn() as c:
            c.execute("UPDATE papers SET card_json=? WHERE id=?",
                      ('{"keywords": ["近场信道估计"]}', kw))
            c.execute("UPDATE papers SET card_json=? WHERE id=?", ('["not","a","dict"]', bad))
        res = bibimport.import_text(
            "@article{a, title = {KW Row}, doi = {10.1/kw}, abstract = {new abs}}\n"
            "@article{b, title = {Bad Card}, doi = {10.1/bad}, abstract = {new abs 2}}")
        self.assertEqual((res["updated"], res["failed"]), (2, 0), res)
        with db.conn() as c:
            got = c.execute("SELECT keywords FROM papers_fts WHERE rowid=?", (kw,)).fetchone()
        self.assertIn("近场信道估计", got["keywords"], "重建 FTS 不得把卡片关键词洗掉")

    def test_author_separator_is_case_insensitive(self):
        """BibTeX 的 ` and ` 分隔符大小写不敏感；带花括号那条路径本来就 lower() 了，
        不带花括号的那条漏了 re.I，两个人会被当成一个人。"""
        self.assertEqual(bibimport._split_authors("Alice Smith AND Bob Lee"),
                         ["Alice Smith", "Bob Lee"])
        # 花括号包住的机构名里的 and 仍然不能切
        self.assertEqual(bibimport._split_authors(r"{Smith and Sons Lab} and Alice Smith"),
                         ["Smith and Sons Lab", "Alice Smith"])

    def test_paren_style_entry_and_doi_recovered_from_url(self):
        """`@article(...)` 圆括号定界是合法 BibTeX；没有 doi 字段时从 url 里捞。"""
        p = bibimport.parse_bibtex(
            "@ARTICLE(paren1,\n  title = {Paren Delimited Entry},\n  year = {2015}\n)")
        self.assertEqual([x["title"] for x in p], ["Paren Delimited Entry"])
        self.assertEqual(p[0]["year"], 2015)
        u = bibimport.parse_bibtex(
            "@misc{u1, title = {No DOI Field}, url = {https://doi.org/10.4444/from-url}}")[0]
        self.assertEqual(u["doi"], "10.4444/from-url")
        self.assertEqual(u["norm_key"], "doi:10.4444/from-url")

    def test_wrong_format_is_reported_not_silently_zero(self):
        """把 .bib 按 csv 解析会得到 0 条且一条错误都没有——UI 上就是「导入成功，0 篇」。"""
        res = bibimport.import_text("@article{q, title = {X}, year = {2020}}", fmt="csv")
        self.assertEqual((res["parsed"], res["imported"]), (0, 0))
        self.assertTrue(any("一条文献都没得到" in e for e in res["errors"]), res["errors"])

    def test_enrich_skips_papers_whose_abstract_is_already_in_db(self):
        """补全跑在去重之前：库里已有摘要的条目再问一次 S2，拿回来也会被「补空不覆盖」丢掉。
        S2 共享池按 5 分钟窗口限流，几百条白问一遍能把一次导入拖到以小时计。"""
        self._insert(norm_key="doi:10.7/has-abs", title="Has Abstract",
                     abstract="库内已有摘要", year=2020, doi="10.7/has-abs")
        calls = []
        with mock.patch.object(bibimport, "_s2_get",
                               side_effect=lambda i: calls.append(i) or {"abstract": "S2"}):
            bibimport.import_text(
                "@article{h, title = {Has Abstract}, doi = {10.7/has-abs}}\n"
                "@article{n, title = {Brand New}, doi = {10.7/new-one}}", enrich=True)
        self.assertEqual(calls, ["DOI:10.7/new-one"], "缺摘要的要查，库里已有的不该再查")
        with db.conn() as c:
            row = c.execute("SELECT abstract FROM papers WHERE doi='10.7/has-abs'").fetchone()
        self.assertEqual(row["abstract"], "库内已有摘要")

    def test_parser_drop_paths_lose_only_the_bad_record(self):
        """三种格式的「丢弃」分支此前一条都没被跑到过——只测了 happy path。"""
        errs = []
        got = bibimport.parse_ris(
            "garbage line before anything\n"
            "TY  - JOUR\nAU  - Nobody\nPY  - 2020\nER  - \n"
            "TY  - JOUR\nTI  - Good One\nER  - \n", errs)
        self.assertEqual([p["title"] for p in got], ["Good One"])
        self.assertTrue(any("不是合法标签行" in e for e in errs), errs)
        self.assertTrue(any(e.startswith(bibimport.DROP_TAG) and "没有 TI/T1" in e
                            for e in errs), errs)
        errs2 = []
        rows = bibimport.parse_zotero_csv("Title,DOI\n,10.1/no-title\nHas Title,10.1/ok\n", errs2)
        self.assertEqual([r["title"] for r in rows], ["Has Title"])
        self.assertTrue(any(e.startswith(bibimport.DROP_TAG) for e in errs2), errs2)

    def test_three_part_author_and_nameless_field(self):
        """`von Last, Jr, First` 三段式与空字段名——都是自述报告点名、测试却没碰的分支。"""
        self.assertEqual(bibimport._norm_author("von Beethoven, Jr, Ludwig"),
                         "Ludwig von Beethoven Jr")
        errs = []
        p = bibimport.parse_bibtex(
            "@article{e1, title = {Nameless Field}, = {orphan}, year = {2020}}", errs)
        self.assertEqual([x["title"] for x in p], ["Nameless Field"])
        self.assertTrue(any("字段名为空" in e for e in errs), errs)

    # ── _s2_get 真身：别的用例把它整个 mock 掉了，这几条用假 transport 驱动它 ──

    def test_s2_get_404_returns_none_without_retrying(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(404, json={})

        with self._mock_http(handler):
            self.assertIsNone(self._real_s2_get("DOI:10.1/not-indexed"))
        self.assertEqual(len(seen), 1, "404 是「未收录」不是故障，不该重试")
        self.assertTrue(seen[0].startswith(bibimport.S2_PAPER_URL + "DOI:10.1/not-indexed"))
        self.assertIn("fields=", seen[0])

    def test_s2_get_retries_rate_limit_then_raises(self):
        seen, slept = [], []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(429, json={})

        with self._mock_http(handler), mock.patch.object(bibimport, "_sleep", slept.append):
            with self.assertRaises(RuntimeError):
                self._real_s2_get("DOI:10.1/rate-limited")
        self.assertEqual(len(seen), len(bibimport.ENRICH_DELAYS) + 1)
        # 退避基数要和 sources/semantic_scholar.py 同调（20/45/90s，±20% 抖动）
        self.assertEqual(len(slept), len(bibimport.ENRICH_DELAYS))
        for base, got in zip(bibimport.ENRICH_DELAYS, slept):
            self.assertTrue(base * 0.8 <= got <= base * 1.2, (base, slept))

    def test_s2_get_retries_network_errors_too(self):
        tries, slept = [], []

        def handler(request):
            tries.append(1)
            raise httpx.ConnectError("网络不通")

        with self._mock_http(handler), mock.patch.object(bibimport, "_sleep", slept.append):
            with self.assertRaises(RuntimeError):
                self._real_s2_get("DOI:10.1/offline")
        self.assertEqual(len(tries), len(bibimport.ENRICH_DELAYS) + 1)

    def test_s2_get_does_not_let_a_bib_file_rewrite_the_request_url(self):
        """DOI 直接来自用户上传的 .bib。裸拼进 URL 路径时，`?`/`#` 会截断标识符、
        `..` 会被 httpx 归一化掉——拿回来的是别的端点的数据，却被当成这篇论文的元数据
        写进库里，正是项目红线里的「编造」。"""
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, json={"abstract": "x"})

        with self._mock_http(handler):
            self._real_s2_get("DOI:10.1234/a?b=c")
            self._real_s2_get("DOI:10.1234/a#frag")
            self._real_s2_get("DOI:10.1234/ab_cd")   # 正常 DOI 不能被改写
        self.assertIn("%3F", seen[0], seen)
        self.assertIn("%23", seen[1], seen)
        self.assertTrue(seen[2].startswith(bibimport.S2_PAPER_URL + "DOI:10.1234/ab_cd?"), seen)
        for bad in ("DOI:../../../etc/passwd", "arXiv:2401.1/../../admin"):
            with self.assertRaises(ValueError, msg=bad):
                self._real_s2_get(bad)

    def test_s2_get_identifier_errors_are_recorded_not_fatal(self):
        """消毒抛的 ValueError 必须被补全阶段吃掉，不能让一条脏 DOI 毁掉整次导入。"""
        res = bibimport.import_text(
            r"@article{t, title = {Traversal DOI}, doi = {../../../etc/passwd}}",
            enrich=True)
        self.assertEqual(res["imported"], 1, res)
        self.assertTrue(any("补全" in e for e in res["errors"]), res["errors"])


if __name__ == "__main__":
    unittest.main()
