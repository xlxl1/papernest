# -*- coding: utf-8 -*-
"""数据完整性五条：DOI 收口、arXiv error-feed、NULL 主键、查询缓存不过期、补全无熔断。

共同点是**都不报错，只悄悄产出错的东西**：
  · DOI 前缀不剥 → BibTeX 写成 `doi = {https://doi.org/10.x}`（真库 100/242 篇）；
    这条正则在仓库里被独立抄了 4 份，而漏掉的那个（cite.py）就是导出坏掉的地方；
  · arXiv 查询非法时返回的是**形状完全合法的 Atom feed**，里面一条 title=Error 的条目；
    subscribe.py 早就挡了，平行实现的采集路径一直没挡；
  · 复合主键里的可空列让 `ON CONFLICT` 永不触发，同一条引用挂几次存几行；
  · 查询缓存只写不过期，同一关键词第二次检索永远返回旧结果、`new_papers` 恒为 0；
  · 补全没有熔断，S2 挂掉时 500 条导出要逐条陪跑约 21 小时。
"""
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from papernest import bibimport, cite, config, db, normalize, workspace
from papernest.sources import arxiv


class DoiIsNormalisedInOnePlaceTests(unittest.TestCase):
    #: 真库 242 篇有 DOI 的论文里 100 篇（41.3%）是这个形态，全部来自 openalex 源
    REAL = "https://doi.org/10.1109/lwc.2019.2948632"
    BARE = "10.1109/lwc.2019.2948632"

    def test_bare_doi_handles_every_prefix_form(self):
        for raw in (self.REAL, "http://dx.doi.org/" + self.BARE,
                    "doi.org/" + self.BARE, self.BARE, "  " + self.REAL + " "):
            with self.subTest(raw=raw):
                self.assertEqual(normalize.bare_doi(raw), self.BARE)
        self.assertIsNone(normalize.bare_doi(""))
        self.assertIsNone(normalize.bare_doi(None))

    def _paper(self):
        return {"paper_id": 1, "title": "T", "authors": ["A B"], "venue": "IEEE",
                "year": 2020, "doi": self.REAL, "arxiv_id": None}

    def test_bibtex_writes_a_bare_doi(self):
        """BibTeX 的 doi 字段约定是裸 DOI——多数样式会自己再拼一次 https 前缀。"""
        out = cite.to_bibtex(self._paper())
        self.assertIn("doi = {" + self.BARE + "},", out)
        self.assertNotIn("https://doi.org", out)

    def test_ris_writes_a_bare_doi(self):
        out = cite.to_ris(self._paper())
        self.assertIn("DO  - " + self.BARE, out)
        self.assertNotIn("https://doi.org", out)

    def test_norm_key_is_unchanged_by_the_refactor(self):
        """收口不能顺手改动去重键——那会让同一篇论文换一个键、直接绕过去重。"""
        self.assertEqual(normalize.norm_key(doi=self.REAL),
                         normalize.norm_key(doi=self.BARE))
        self.assertEqual(normalize.norm_key(doi=self.REAL), "doi:" + self.BARE)


class ArxivErrorFeedIsNotAPaperTests(unittest.TestCase):
    FEED = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/api/errors#incorrect_id_format</id>
        <title>Error</title><summary>incorrect id format</summary></entry>
      <entry><id>http://arxiv.org/abs/2401.12345v1</id><title>A Real Paper</title>
        <published>2024-01-01</published><summary>abs</summary></entry>
      <entry><id>http://arxiv.org/abs/math.GT/0309136</id><title>Old Style</title>
        <published>2003-09-01</published><summary>abs</summary></entry>
    </feed>"""

    def test_error_entry_is_dropped_and_real_ones_survive(self):
        """不挡的话它会入库：标题「Error」、arxiv_id 是一整条 URL、
        oa_pdf_url 拼成 https://arxiv.org/pdf/http://arxiv.org/api/errors#…"""
        got = arxiv._parse(ET.fromstring(self.FEED))
        self.assertEqual([p["title"] for p in got], ["A Real Paper", "Old Style"])
        self.assertEqual([p["arxiv_id"] for p in got],
                         ["2401.12345v1", "math.GT/0309136"])


class _TempDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_integrity_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)
        db.init_db()

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def _paper(self, key):
        with db.conn() as c:
            return db.insert_l0(c, {
                "norm_key": key, "title": "T", "abstract": "a", "year": 2024,
                "venue": "v", "authors": [], "doi": None, "arxiv_id": None,
                "source": "s2"})


class NullablePrimaryKeyDoesNotDefeatUpsertTests(_TempDb):
    def test_attaching_the_same_source_twice_updates_instead_of_duplicating(self):
        """`PRIMARY KEY(document_id, paper_id, page_no)` 里 page_no 可空，而 UNIQUE
        索引里 NULL 互不相等——按主键做 upsert 永远不触发。而 `attach_source` 的
        page_no 默认就是 None，所以这是默认路径，不是边界情况。"""
        pid = self._paper("x:1")
        with db.conn() as c:
            c.execute("INSERT INTO documents(id,title) VALUES(1,'D')")
        for i in range(3):
            workspace.attach_source(1, pid, page_no=None, quote="q%d" % i)
        with db.conn() as c:
            rows = c.execute("SELECT quote FROM document_sources").fetchall()
        self.assertEqual(len(rows), 1, "同一条引用存了 %d 行" % len(rows))
        self.assertEqual(rows[0]["quote"], "q2", "upsert 没有覆盖 quote")

    def test_different_pages_are_still_distinct(self):
        """收口不能把「同一篇的不同页」也合并掉。"""
        pid = self._paper("x:2")
        with db.conn() as c:
            c.execute("INSERT INTO documents(id,title) VALUES(2,'D')")
        workspace.attach_source(2, pid, page_no=None, quote="none")
        workspace.attach_source(2, pid, page_no=3, quote="p3")
        workspace.attach_source(2, pid, page_no=4, quote="p4")
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM document_sources "
                          "WHERE document_id=2").fetchone()["n"]
        self.assertEqual(n, 3)


class QueryCacheExpiresTests(_TempDb):
    def test_stale_entry_is_not_served(self):
        """只写不过期时，同一关键词第二次检索永远拿不到新论文——而 ingest 的模块
        docstring 恰恰把「同一课题第二次检索只增量入库」当成去重的验收证据。"""
        with db.conn() as c:
            db.put_query_cache(c, "q", "s2", ["k1"])
            self.assertIsNotNone(db.get_query_cache(c, "q"))
            c.execute("UPDATE query_cache SET ts=datetime('now','localtime',?)",
                      ("-%d hours" % (db.QUERY_CACHE_TTL_HOURS + 1),))
            self.assertIsNone(db.get_query_cache(c, "q"), "过期的缓存仍然被端了出来")

    def test_fresh_entry_is_still_served(self):
        with db.conn() as c:
            db.put_query_cache(c, "q2", "s2", ["k1"])
            c.execute("UPDATE query_cache SET ts=datetime('now','localtime',?)",
                      ("-%d hours" % max(db.QUERY_CACHE_TTL_HOURS - 1, 0),))
            self.assertIsNotNone(db.get_query_cache(c, "q2"))

    def test_explicit_bypass(self):
        """「我就是要看有没有新文献」这条路径要有显式开关。"""
        with db.conn() as c:
            db.put_query_cache(c, "q3", "s2", ["k1"])
            self.assertIsNone(db.get_query_cache(c, "q3", ttl_hours=0))


class EnrichHasACircuitBreakerTests(unittest.TestCase):
    def test_a_dead_source_stops_after_a_short_streak(self):
        """每条 `_s2_get` 是 4 次尝试 + 155s 退避，而外层把每条失败都吞掉继续下一条。
        S2 整体挂掉时，一份 500 条的 Zotero 导出要跑 ≈21 小时并独占一个任务槽。"""
        calls = []

        def boom(ident):
            calls.append(ident)
            raise RuntimeError("S2 down")

        entries = [{"doi": "10.1/%d" % i, "title": "T%d" % i} for i in range(500)]
        with mock.patch.object(bibimport, "_s2_get", boom):
            errs = bibimport.enrich_entries(entries)
        self.assertLessEqual(len(calls), bibimport.ENRICH_FAILURE_STREAK,
                             "S2 全挂时仍发起了 %d 次补全" % len(calls))
        self.assertTrue(any("已停止补全" in e for e in errs), "停手了却没说")

    def test_intermittent_failures_do_not_trip_it(self):
        """单条偶发失败（限流抖动）不该把整批补全掐掉。"""
        calls, n = [], [0]

        def flaky(ident):
            calls.append(ident)
            n[0] += 1
            if n[0] % 7 == 0:
                raise RuntimeError("blip")
            return {"abstract": "a"}

        entries = [{"doi": "10.2/%d" % i, "title": "T%d" % i} for i in range(50)]
        with mock.patch.object(bibimport, "_s2_get", flaky):
            bibimport.enrich_entries(entries)
        self.assertEqual(len(calls), 50, "偶发失败误触发了熔断")

    def test_not_indexed_is_a_normal_answer_not_a_failure(self):
        """「未收录」说明网络是通的，不该计进熔断。"""
        calls = []

        def missing(ident):
            calls.append(ident)
            return None

        with mock.patch.object(bibimport, "_s2_get", missing):
            bibimport.enrich_entries(
                [{"doi": "10.3/%d" % i, "title": "T"} for i in range(20)])
        self.assertEqual(len(calls), 20)


if __name__ == "__main__":
    unittest.main()


class AddingAColumnNeedsAMigrationTests(unittest.TestCase):
    """新表放进 SCHEMA 就够了；**给已有表加列必须走迁移**。

    `table_summaries` 是加进 SCHEMA 的新表，靠 `CREATE TABLE IF NOT EXISTS`
    自动出现——这没错，不用改版本。但后来往这张表**加列**（`structure_ok`）时，
    我又只改了 SCHEMA：`IF NOT EXISTS` 对已经建好的表是空跑，列压根不会加上去。

    实测就是这么炸的——而且是在**已经花了钱、跑到一半**的时候：

        sqlite3.OperationalError: table table_summaries has no column named structure_ok

    所以有了 `_migrate_9`。这个用例造一个「老版本库」（v8 + 无 structure_ok 的
    table_summaries + 一行历史数据）走一遍 init_db，钉住三件事：列补上了、
    版本推进了、**历史数据没丢**。
    """

    def setUp(self):
        import shutil
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.mkdtemp(prefix="papernest_mig9_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        self.addCleanup(lambda: setattr(config, "DB_PATH", self._old[0]))
        self.addCleanup(lambda: setattr(config, "DATA_DIR", self._old[1]))
        config.DATA_DIR = Path(self._tmp)
        config.DB_PATH = config.DATA_DIR / "old.db"

    def _make_v8_db(self):
        import sqlite3 as s3
        c = s3.connect(config.DB_PATH)
        c.execute("""CREATE TABLE table_summaries(
            table_hash TEXT PRIMARY KEY, paper_id INTEGER NOT NULL, page_no INTEGER,
            n_rows INTEGER, n_cols INTEGER, summary TEXT NOT NULL, model TEXT,
            is_model_written INTEGER NOT NULL DEFAULT 1, created_at TEXT)""")
        c.execute("INSERT INTO table_summaries VALUES"
                  "('h1',1,2,3,4,'迁移前就存在的摘要','glm-5.2',1,'2026-09-09')")
        c.execute("PRAGMA user_version=8")
        c.commit()
        c.close()

    def test_the_column_is_added_and_data_survives(self):
        self._make_v8_db()
        db.init_db(force=True)
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(table_summaries)")}
            row = c.execute("SELECT summary, structure_ok FROM table_summaries "
                            "WHERE table_hash='h1'").fetchone()
            ver = c.execute("PRAGMA user_version").fetchone()[0]
        self.assertIn("structure_ok", cols, "迁移没把列加上——批量任务会跑到一半炸")
        self.assertEqual(row["summary"], "迁移前就存在的摘要", "历史数据被冲掉了")
        self.assertEqual(row["structure_ok"], 1,
                         "历史行默认按「结构完好」算——它们生成时还没有这个区分")
        self.assertGreaterEqual(ver, 9)

    def test_running_it_twice_is_a_no_op(self):
        """迁移必须幂等：`init_db` 在每个新进程/新库路径上都会跑。"""
        self._make_v8_db()
        db.init_db(force=True)
        db.init_db(force=True)          # 不抛就是通过
        with db.conn() as c:
            n = sum(1 for r in c.execute("PRAGMA table_info(table_summaries)")
                    if r[1] == "structure_ok")
        self.assertEqual(n, 1, "列被加了两次")

    def test_a_fresh_db_gets_the_column_from_schema(self):
        """新库走 SCHEMA 那条路，也必须有这一列——两条路不能分叉。"""
        db.init_db(force=True)
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(table_summaries)")}
        self.assertIn("structure_ok", cols)

    def test_every_migration_step_is_registered(self):
        """漏注册一步，后面的库就永远停在旧版本上而且没人报错。"""
        for step in range(1, db.SCHEMA_VERSION + 1):
            self.assertIn(step, db.MIGRATIONS, f"迁移 {step} 没登记进 MIGRATIONS")
