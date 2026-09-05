"""papernest/subscribe.py 的离线确定性测试。

隔离口径与 tests/test_write_online.py 一致：DB_PATH/DATA_DIR 指到 tempfile，
embeddings 与 llm 一律摁死（开发机 .env 真配了 EMBED_MODEL / LLM_*，
不摁住会真发请求——又慢又花钱，结果还随网络漂移）。
所有 HTTP 走假客户端；arXiv 的 3 秒礼貌间隔也 mock 掉，否则一个用例要跑好几秒。
"""
import shutil
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from papernest import config, db, subscribe

UTC = timezone.utc
NOW = datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)   # 固定时钟：日期过滤可手算


# ── arXiv Atom 夹具（形状照着官方响应写，含一条故意畸形的 entry）──

def _entry(aid: str, title: str, published: str, summary: str,
           authors=("Alice Smith", "Bob Lee")) -> str:
    people = "".join(f"<author><name>{a}</name></author>" for a in authors)
    return f"""  <entry>
    <id>http://arxiv.org/abs/{aid}</id>
    <updated>{published}</updated>
    <published>{published}</published>
    <title>{title}</title>
    <summary>  {summary}
</summary>
    {people}
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
"""


def _feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<feed xmlns="http://www.w3.org/2005/Atom">\n'
            '  <title>ArXiv Query</title>\n'
            + "".join(entries) + "</feed>\n")


FEED_OK = _feed(
    _entry("2608.00011v1", "Agent Evaluation with Tool Use Reliability",
           "2026-08-31T09:00:00Z",
           "We study agent evaluation and tool use reliability for language model agents."),
    _entry("2608.00012v2", "Retrieval Augmented Generation for Long Documents",
           "2026-08-30T18:00:00Z",
           "A retrieval pipeline for long document question answering."),
    _entry("2508.09999v1", "An Older Paper About Channel Estimation",
           "2026-08-25T10:00:00Z",
           "Channel estimation for massive MIMO systems."),
)

# 畸形：① 缺 id ② 缺 title ③ published 不是时间。中间夹着一条正常的。
FEED_MALFORMED = _feed(
    """  <entry>
    <updated>2026-08-31T09:00:00Z</updated>
    <published>2026-08-31T09:00:00Z</published>
    <title>Entry Without Any Id</title>
    <summary>no id here</summary>
  </entry>
""",
    """  <entry>
    <id>http://arxiv.org/abs/2608.00021v1</id>
    <published>2026-08-31T08:00:00Z</published>
    <summary>no title here</summary>
  </entry>
""",
    _entry("2608.00022v1", "A Perfectly Fine Agent Paper", "2026-08-31T07:00:00Z",
           "Agent planning and reasoning."),
    """  <entry>
    <id>http://arxiv.org/abs/2608.00023v1</id>
    <published>not-a-timestamp</published>
    <title>Broken Published Field</title>
    <summary>bad date</summary>
  </entry>
""",
)

# arXiv 查询非法时返回的**不是** HTTP 错误，而是一个形状完全合法的 Atom feed，
# 里面一条 id=.../api/errors#... 、title=Error 的 entry。照抄官方响应的形状。
FEED_ARXIV_ERROR = _feed(
    """  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
    <title>Error</title>
    <summary>incorrect id format for x</summary>
    <updated>2026-08-31T00:00:00-04:00</updated>
    <published>2026-08-31T00:00:00-04:00</published>
    <author><name>arXiv api core</name></author>
  </entry>
""")


def _full_page(prefix: str, published: str, n: int = 100) -> str:
    """满页（100 条）响应：只有满页才会触发翻第二页。"""
    return _feed(*[_entry(f"{prefix}.{i:05d}v1", f"Filler paper {i}", published,
                          "filler summary") for i in range(n)])


class _FakeResp:
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        return None


class _FakeClient:
    """按调用次序吐出预置页面；最后一页之后一直重复最后一页。"""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": dict(params or {}),
                           "headers": dict(headers or {})})
        return _FakeResp(self.pages[min(len(self.calls) - 1, len(self.pages) - 1)])


class SubscribeTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_subscribe_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            mock.patch("papernest.llm.available", return_value=False),
            mock.patch("papernest.sources.arxiv.polite_sleep", return_value=None),
        ]
        for p in self._patches:
            p.start()
        db.init_db()
        subscribe.ensure_schema()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 夹具工具 ──

    def _add_paper(self, key, title, keywords=(), created_at=None,
                   venue="arXiv", abstract=None):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": key, "title": title, "abstract": abstract,
                "year": 2026, "venue": venue, "authors": ["Someone"],
                "doi": None, "arxiv_id": key.replace("arxiv:", ""), "source": "s2"})
            if keywords:
                c.execute("UPDATE papers SET card_json=? WHERE id=?",
                          (json.dumps({"keywords": list(keywords)},
                                      ensure_ascii=False), pid))
            if created_at:
                c.execute("UPDATE papers SET created_at=? WHERE id=?", (created_at, pid))
        return pid

    @staticmethod
    def _weight(profile, term):
        for t in profile["terms"]:
            if t["term"] == term:
                return t["weight"]
        return None

    @staticmethod
    def _fake_http(*pages):
        fake = _FakeClient(pages)
        return fake, mock.patch("papernest.http.client", return_value=fake)


# ── 1. build_profile ──

class BuildProfileTests(SubscribeTestBase):
    def test_empty_library_does_not_explode(self):
        p = subscribe.build_profile()
        self.assertEqual(p, {"terms": [], "arxiv_categories": [], "n_papers": 0})

    def test_frequent_keyword_outweighs_rare_one(self):
        """三篇都带 'agent'、一篇带 'mimo'：df 越高权重越高（同一天入库，近期系数相同）。"""
        for i in range(3):
            self._add_paper(f"arxiv:100{i}", f"Paper number {i}",
                            keywords=["agent"], created_at="2026-08-31 10:00:00")
        self._add_paper("arxiv:1009", "Another one", keywords=["mimo"],
                        created_at="2026-08-31 10:00:00")
        p = subscribe.build_profile()
        self.assertEqual(p["n_papers"], 4)
        w_agent, w_mimo = self._weight(p, "agent"), self._weight(p, "mimo")
        self.assertEqual(w_agent, 1.0)                  # 最高权重被归一化成 1.0
        self.assertAlmostEqual(w_mimo, 1 / 3, places=4)  # 3 篇 vs 1 篇
        self.assertGreater(w_agent, w_mimo)

    def test_recent_paper_weighs_more_than_old_one(self):
        """同为 df=1 的关键词，近期那篇权重更高（一年半衰：两年前 ≈ 0.25）。"""
        self._add_paper("arxiv:2001", "Recent work", keywords=["agent"],
                        created_at="2026-08-31 10:00:00")
        self._add_paper("arxiv:2002", "Ancient work", keywords=["mimo"],
                        created_at="2024-08-31 10:00:00")
        p = subscribe.build_profile()
        self.assertEqual(self._weight(p, "agent"), 1.0)
        # 2 年 = 730 天，0.5**(730/365) = 0.25
        self.assertAlmostEqual(self._weight(p, "mimo"), 0.25, places=3)

    def test_keyword_outranks_bare_title_word(self):
        """来源档位：keywords(3.0) > 标题(1.0)。同为 df=1 时关键词该更重。"""
        self._add_paper("arxiv:3001", "Something about widgets", keywords=["agent"],
                        created_at="2026-08-31 10:00:00")
        p = subscribe.build_profile()
        self.assertEqual(self._weight(p, "agent"), 1.0)
        self.assertAlmostEqual(self._weight(p, "widgets"), 1 / 3, places=4)

    def test_profile_is_totally_ordered_and_stable(self):
        for i in range(6):
            self._add_paper(f"arxiv:400{i}", f"Retrieval augmented generation {i}",
                            keywords=["agent", "retrieval"],
                            created_at="2026-08-31 10:00:00")
        a = subscribe.build_profile()
        b = subscribe.build_profile()
        self.assertEqual(a, b)
        pairs = [(-t["weight"], t["term"]) for t in a["terms"]]
        self.assertEqual(pairs, sorted(pairs))

    def test_categories_guessed_from_terms(self):
        self._add_paper("arxiv:5001", "Language model agents",
                        keywords=["large language model", "agent"],
                        created_at="2026-08-31 10:00:00")
        p = subscribe.build_profile()
        self.assertIn("cs.CL", p["arxiv_categories"])
        self.assertLessEqual(len(p["arxiv_categories"]), 3)

    def test_unmappable_terms_yield_no_category(self):
        """猜不出就返回空列表——绝不硬塞一个可能错的分类。"""
        self._add_paper("arxiv:5002", "Zzzz qqqq wwww",
                        keywords=["zzzzqq", "wwwwqq"], created_at="2026-08-31 10:00:00")
        self.assertEqual(subscribe.build_profile()["arxiv_categories"], [])

    def test_chinese_titles_produce_terms_without_boilerplate(self):
        """中文没有空格分词，走二元词面；「研究/方法」这类套话不该进画像。"""
        self._add_paper("arxiv:6001", "近场信道估计方法研究",
                        created_at="2026-08-31 10:00:00")
        p = subscribe.build_profile()
        terms = {t["term"] for t in p["terms"]}
        self.assertIn("信道", terms)
        self.assertIn("近场", terms)
        self.assertNotIn("研究", terms)
        self.assertNotIn("方法", terms)

    def test_category_terms_match_on_word_boundaries(self):
        """回归：分类映射曾用裸子串匹配，cs.IR 的 'rag' 会命中 average / coverage /
        storage，cs.CV 的 'vision' 会命中 supervision。后果比打分层那个 bug 更重：
        一个做无线通信的用户（画像里全是 'average rate'、'coverage probability'）
        会被塞一个 cs.IR，整条订阅源变成噪音——正是模块开头说「绝不硬塞」的那件事。
        """
        for bogus in ("average throughput", "coverage probability",
                      "storage systems", "fragmentation", "supervision"):
            self.assertEqual(
                subscribe.guess_categories([{"term": bogus, "weight": 1.0}]), [],
                f"{bogus!r} 不该投出任何分类")

    def test_category_terms_still_match_plurals_and_hyphens(self):
        """收紧匹配不能把真信号一起误杀：复数与连字符变体仍要认。"""
        cases = {"agents": "cs.AI", "images": "cs.CV", "antennas": "eess.SP",
                 "multi-agent": "cs.AI", "large language model": "cs.CL",
                 "retrieval-augmented generation": "cs.IR"}
        for term, cat in cases.items():
            self.assertIn(cat, subscribe.guess_categories([{"term": term, "weight": 1.0}]),
                          f"{term!r} 应当投给 {cat}")

    def test_generic_venue_is_not_a_term(self):
        self._add_paper("arxiv:5003", "Some agent paper", keywords=["agent"],
                        venue="arXiv", created_at="2026-08-31 10:00:00")
        self.assertIsNone(self._weight(subscribe.build_profile(), "arxiv"))


# ── 2. fetch_recent ──

class FetchRecentTests(SubscribeTestBase):
    def test_parses_atom_fields(self):
        fake, patch = self._fake_http(FEED_OK)
        with patch:
            got = subscribe.fetch_recent(["cs.CL", "cs.LG"], days=7, now=NOW)
        self.assertEqual(len(got), 3)
        first = got[0]
        self.assertEqual(first["arxiv_id"], "2608.00011v1")
        self.assertEqual(first["norm_key"], "arxiv:2608.00011")   # 版本号被合并
        self.assertEqual(first["title"], "Agent Evaluation with Tool Use Reliability")
        self.assertIn("tool use reliability", first["abstract"])
        self.assertEqual(first["published"], "2026-08-31T09:00:00Z")
        self.assertEqual(first["authors"], ["Alice Smith", "Bob Lee"])
        self.assertEqual(first["year"], 2026)
        self.assertEqual(first["venue"], "arXiv")
        self.assertEqual(first["source"], "arxiv")
        self.assertEqual(first["oa_pdf_url"], "https://arxiv.org/pdf/2608.00011v1")
        # 查询串与排序参数（arXiv 要求降序按提交时间才拿得到「最近的」）
        params = fake.calls[0]["params"]
        self.assertEqual(params["search_query"], "cat:cs.CL OR cat:cs.LG")
        self.assertEqual(params["sortBy"], "submittedDate")
        self.assertEqual(params["sortOrder"], "descending")

    def test_days_filter(self):
        """days=1 时 cutoff = 2026-08-30T12:00Z：08-31 与 08-30T18:00 留下，08-25 掉。"""
        fake, patch = self._fake_http(FEED_OK)
        with patch:
            one_day = subscribe.fetch_recent(["cs.CL"], days=1, now=NOW)
        self.assertEqual([p["arxiv_id"] for p in one_day],
                         ["2608.00011v1", "2608.00012v2"])
        fake2, patch2 = self._fake_http(FEED_OK)
        with patch2:
            week = subscribe.fetch_recent(["cs.CL"], days=7, now=NOW)
        self.assertEqual(len(week), 3)

    def test_malformed_entries_are_skipped_not_fatal(self):
        fake, patch = self._fake_http(FEED_MALFORMED)
        with patch:
            got = subscribe.fetch_recent(["cs.CL"], days=1, now=NOW)
        self.assertEqual([p["arxiv_id"] for p in got], ["2608.00022v1"])
        self.assertEqual(got[0]["title"], "A Perfectly Fine Agent Paper")

    def test_empty_categories_rejected(self):
        with self.assertRaises(subscribe.SubscribeError):
            subscribe.fetch_recent([], days=1, now=NOW)

    def test_illegal_category_rejected(self):
        with self.assertRaises(subscribe.SubscribeError):
            subscribe.fetch_recent(["cs.CL; DROP TABLE"], days=1, now=NOW)

    def test_limit_is_respected(self):
        fake, patch = self._fake_http(FEED_OK)
        with patch:
            got = subscribe.fetch_recent(["cs.CL"], days=7, limit=2, now=NOW)
        self.assertEqual(len(got), 2)

    def test_short_page_stops_paging(self):
        """一页不满就收工：不该为了凑 limit 再发一次请求（arXiv 有礼貌间隔）。"""
        fake, patch = self._fake_http(FEED_OK)
        with patch:
            subscribe.fetch_recent(["cs.CL"], days=7, limit=200, now=NOW)
        self.assertEqual(len(fake.calls), 1)

    def test_full_page_pages_forward_with_correct_offset(self):
        """满页 100 条必须继续翻第二页，且 start 偏移正确、不重不漏。

        真实 arXiv 一天的 cs.LG 远超 100 条，这条路径是生产上的常态。
        """
        page1 = _full_page("2608", "2026-08-31T09:00:00Z", 100)
        page2 = _full_page("2607", "2026-08-31T08:00:00Z", 40)
        fake, patch = self._fake_http(page1, page2)
        with patch:
            got = subscribe.fetch_recent(["cs.LG"], days=1, limit=200, now=NOW)
        self.assertEqual(len(got), 140)
        self.assertEqual(len({p["norm_key"] for p in got}), 140)   # 无重复
        self.assertEqual([c["params"]["start"] for c in fake.calls], [0, 100])
        self.assertEqual([c["params"]["max_results"] for c in fake.calls], [100, 100])

    def test_limit_stops_paging_midway(self):
        """limit 落在第一页之内时不该再发第二次请求。"""
        fake, patch = self._fake_http(_full_page("2608", "2026-08-31T09:00:00Z", 100),
                                      _full_page("2607", "2026-08-31T08:00:00Z", 100))
        with patch:
            got = subscribe.fetch_recent(["cs.LG"], days=1, limit=60, now=NOW)
        self.assertEqual(len(got), 60)
        self.assertEqual(len(fake.calls), 1)

    def test_arxiv_error_feed_is_not_mistaken_for_a_paper(self):
        """arXiv 的错误响应是一个合法 Atom feed（title=Error，id 是一整条 URL）。

        不识别的话，它会被当成论文打分、落进 digest_items、再被 adopt 写进 papers，
        库里就多一篇标题「Error」、arxiv_id 是 URL、pdf 链接拼成
        https://arxiv.org/pdf/http://... 的垃圾记录。
        """
        fake, patch = self._fake_http(FEED_ARXIV_ERROR)
        with patch:
            got = subscribe.fetch_recent(["cs.CL"], days=1, now=NOW)
        self.assertEqual(got, [])

    def test_non_arxiv_shaped_id_is_rejected(self):
        weird = _feed(_entry("not-an-arxiv-id", "Looks Fine But Id Is Junk",
                             "2026-08-31T09:00:00Z", "x"),
                      _entry("2608.00077v1", "Genuine Paper",
                             "2026-08-31T09:00:00Z", "x"))
        fake, patch = self._fake_http(weird)
        with patch:
            got = subscribe.fetch_recent(["cs.CL"], days=1, now=NOW)
        self.assertEqual([p["arxiv_id"] for p in got], ["2608.00077v1"])

    def test_old_style_arxiv_ids_still_accepted(self):
        """旧式 id（math.GT/0309136、hep-th/9901001）是真实存在的，别一并误杀。"""
        old = _feed(_entry("math.GT/0309136v1", "An Old Paper",
                           "2026-08-31T09:00:00Z", "x"),
                    _entry("hep-th/9901001", "An Even Older Paper",
                           "2026-08-31T09:00:00Z", "x"))
        fake, patch = self._fake_http(old)
        with patch:
            got = subscribe.fetch_recent(["math.GT"], days=1, now=NOW)
        self.assertEqual([p["arxiv_id"] for p in got],
                         ["math.GT/0309136v1", "hep-th/9901001"])

    def test_multiline_title_and_summary_are_flattened(self):
        """真实 arXiv 的 title/summary 一定带换行与缩进空白。"""
        feed = _feed(_entry("2608.00088v1",
                            "Deep Learning for\n  Massive MIMO Channel\n  Estimation",
                            "2026-08-31T09:00:00Z",
                            "Summary\n  spanning\n  several lines."))
        fake, patch = self._fake_http(feed)
        with patch:
            got = subscribe.fetch_recent(["eess.SP"], days=1, now=NOW)
        self.assertEqual(got[0]["title"],
                         "Deep Learning for Massive MIMO Channel Estimation")
        self.assertEqual(got[0]["abstract"], "Summary spanning several lines.")


# ── 3. score_papers ──

class ScorePapersTests(SubscribeTestBase):
    # 画像手工构造：top-5 = 三个词全部 → denom = 1.0 + 0.6 + 0.3 = 1.9
    PROFILE = {"terms": [{"term": "agent", "weight": 1.0},
                         {"term": "retrieval", "weight": 0.6},
                         {"term": "mimo", "weight": 0.3}],
               "arxiv_categories": ["cs.AI"], "n_papers": 3}

    def test_scores_are_hand_computable(self):
        papers = [
            {"arxiv_id": "a1", "title": "Agent retrieval mimo everything",
             "abstract": ""},                                    # 三词全在标题
            {"arxiv_id": "a2", "title": "An agent story", "abstract": "about retrieval"},
            {"arxiv_id": "a3", "title": "Nothing relevant here", "abstract": "empty"},
        ]
        got = subscribe.score_papers(papers, self.PROFILE)
        by = {p["arxiv_id"]: p for p in got}
        # a1: (1.0+0.6+0.3)*1.0 / 1.9 = 1.0
        self.assertAlmostEqual(by["a1"]["score"], 1.0, places=6)
        # a2: 标题命中 agent(1.0) + 摘要命中 retrieval(0.6*0.4=0.24) = 1.24 / 1.9
        self.assertAlmostEqual(by["a2"]["score"], round(1.24 / 1.9, 6), places=6)
        self.assertEqual(by["a3"]["score"], 0.0)
        self.assertEqual([p["arxiv_id"] for p in got], ["a1", "a2", "a3"])

    def test_reasons_report_every_hit_honestly(self):
        got = subscribe.score_papers(
            [{"arxiv_id": "b1", "title": "An agent story",
              "abstract": "about retrieval and mimo"}], self.PROFILE)
        reasons = got[0]["reasons"]
        self.assertEqual([r["term"] for r in reasons], ["agent", "retrieval", "mimo"])
        # idf=1.0：批太小（3 篇 < IDF_MIN_BATCH），不做区分度惩罚
        self.assertEqual(reasons[0], {"term": "agent", "where": "title",
                                      "weight": 1.0, "idf": 1.0, "contribution": 1.0})
        self.assertEqual(reasons[1], {"term": "retrieval", "where": "abstract",
                                      "weight": 0.6, "idf": 1.0, "contribution": 0.24})
        self.assertEqual(reasons[2], {"term": "mimo", "where": "abstract",
                                      "weight": 0.3, "idf": 1.0, "contribution": 0.12})
        self.assertAlmostEqual(sum(r["contribution"] for r in reasons),
                               got[0]["raw_score"], places=6)

    def test_title_hit_wins_over_abstract_hit_for_same_term(self):
        got = subscribe.score_papers(
            [{"arxiv_id": "c1", "title": "agent", "abstract": "agent agent agent"}],
            self.PROFILE)
        self.assertEqual(len(got[0]["reasons"]), 1)     # 同一个词只算一次
        self.assertEqual(got[0]["reasons"][0]["where"], "title")

    def test_word_boundary_prevents_bogus_hits(self):
        """'rag' 不该命中 'storage'——这种理由会直接摧毁用户对分数的信任。"""
        prof = {"terms": [{"term": "rag", "weight": 1.0}]}
        got = subscribe.score_papers(
            [{"arxiv_id": "d1", "title": "Cheap storage for images", "abstract": ""},
             {"arxiv_id": "d2", "title": "RAG pipelines", "abstract": ""}], prof)
        by = {p["arxiv_id"]: p for p in got}
        self.assertEqual(by["d1"]["score"], 0.0)
        self.assertEqual(by["d1"]["reasons"], [])
        # d2 只命中一个词项，score 会吃「广度折扣」；这里要钉的是词边界，
        # 所以看未打折的 raw_score 与理由本身
        self.assertEqual([r["term"] for r in by["d2"]["reasons"]], ["rag"])
        self.assertAlmostEqual(by["d2"]["raw_score"], 1.0, places=6)
        self.assertGreater(by["d2"]["score"], 0.0)

    def test_phrase_hit_suppresses_its_component_words(self):
        """双词概念不该因为拆出成分词而拿三份分（也不该给出三条同义理由）。"""
        prof = {"terms": [{"term": "channel estimation", "weight": 1.0},
                          {"term": "channel", "weight": 1.0},
                          {"term": "estimation", "weight": 1.0}]}
        got = subscribe.score_papers(
            [{"arxiv_id": "e1", "title": "Channel estimation for MIMO", "abstract": ""}],
            prof)
        self.assertEqual([r["term"] for r in got[0]["reasons"]], ["channel estimation"])
        self.assertAlmostEqual(got[0]["raw_score"], 1.0, places=6)   # 1 份，不是 3 份

    def test_component_word_still_counts_when_phrase_absent(self):
        prof = {"terms": [{"term": "channel estimation", "weight": 1.0},
                          {"term": "channel", "weight": 1.0}]}
        got = subscribe.score_papers(
            [{"arxiv_id": "e2", "title": "Channel capacity bounds", "abstract": ""}], prof)
        self.assertEqual([r["term"] for r in got[0]["reasons"]], ["channel"])

    def test_tie_break_is_deterministic_across_runs(self):
        """三篇同分：必须按 arxiv_id 决胜，且两次调用结果一致（跨进程不漂移）。"""
        papers = [{"arxiv_id": x, "title": "agent", "abstract": ""}
                  for x in ("2608.00030", "2608.00010", "2608.00020")]
        first = [p["arxiv_id"] for p in subscribe.score_papers(papers, self.PROFILE)]
        second = [p["arxiv_id"] for p in
                  subscribe.score_papers(list(reversed(papers)), self.PROFILE)]
        self.assertEqual(first, ["2608.00010", "2608.00020", "2608.00030"])
        self.assertEqual(first, second)

    # 下面这份画像照着真实库（498 篇无线通信论文）的形状写：top 词项互相包含
    # （massive / mimo / massive mimo、channel / estimation / channel estimation），
    # 这正是原来那份手工画像（agent/retrieval/mimo，三个词互不相干）测不到的形态。
    REAL_SHAPED = {"terms": [
        {"term": "massive", "weight": 1.0},
        {"term": "mimo", "weight": 1.0},
        {"term": "deep", "weight": 0.95},
        {"term": "massive mimo", "weight": 0.93},
        {"term": "channel", "weight": 0.92},
        {"term": "estimation", "weight": 0.91},
        {"term": "channel estimation", "weight": 0.81},
        {"term": "deep learning", "weight": 0.60},
        {"term": "learning", "weight": 0.88},
    ]}

    def test_more_matching_text_never_lowers_the_score(self):
        """补上更多相关内容，分数只能升不能降。

        回归：短语在摘要命中曾把成分词在标题的强命中整条压掉——同一个标题，
        摘要多写一句 'channel estimation'，总分反而掉 18%，没有人看得懂。
        """
        bare = {"arxiv_id": "m1", "title": "Channel Prediction for Massive Arrays",
                "abstract": ""}
        richer = dict(bare, arxiv_id="m2",
                      abstract="We perform channel estimation with deep learning.")
        got = subscribe.score_papers([bare, richer], self.REAL_SHAPED)
        by = {p["arxiv_id"]: p for p in got}
        self.assertGreater(by["m2"]["raw_score"], by["m1"]["raw_score"])
        self.assertGreaterEqual(by["m2"]["score"], by["m1"]["score"])
        self.assertEqual([p["arxiv_id"] for p in got], ["m2", "m1"])
        # 标题里的 channel（强证据）不能因为摘要里出现短语而被吞掉
        self.assertIn(("channel", "title"),
                      [(r["term"], r["where"]) for r in by["m2"]["reasons"]])

    def test_one_concept_yields_one_reason(self):
        """channel / estimation / channel estimation 是同一个概念，只该出现一条理由。"""
        got = subscribe.score_papers(
            [{"arxiv_id": "n1", "title": "Channel estimation methods", "abstract": ""}],
            self.REAL_SHAPED)[0]
        terms = [r["term"] for r in got["reasons"]]
        self.assertEqual(len(terms), len(set(terms)))
        self.assertEqual(sum(1 for t in terms
                             if t in ("channel", "estimation", "channel estimation")), 1)
        self.assertAlmostEqual(got["raw_score"],
                               sum(r["contribution"] for r in got["reasons"]), places=6)

    def test_chinese_phrase_absorbs_its_own_bigrams(self):
        """中文只能切二元词面：「信道估计」会连带产生 信道/道估/估计。

        不合并的话一个概念拿四份分，理由里还会出现「道估」这种谁也看不懂的碎片。
        """
        prof = {"terms": [{"term": "信道估计", "weight": 0.9},
                          {"term": "信道", "weight": 1.0},
                          {"term": "道估", "weight": 0.8},
                          {"term": "估计", "weight": 1.0},
                          {"term": "评测", "weight": 0.7}]}
        got = subscribe.score_papers(
            [{"arxiv_id": "cn1", "title": "近场信道估计评测", "abstract": ""}], prof)[0]
        terms = [r["term"] for r in got["reasons"]]
        self.assertNotIn("道估", terms)                      # 碎片不该露给用户
        self.assertEqual(len(terms), 2)                      # 信道估计 + 评测，各一条
        self.assertIn("评测", terms)
        self.assertAlmostEqual(got["raw_score"], 1.0 + 0.7, places=6)

    def test_dedupe_is_deterministic_across_hash_seeds(self):
        """概念合并用了并查集与字典，必须给出与输入顺序无关的稳定结果。"""
        prof = dict(self.REAL_SHAPED)
        paper = {"arxiv_id": "s1", "title": "Deep learning for massive MIMO channel "
                                            "estimation", "abstract": ""}
        a = subscribe.score_papers([paper], prof)[0]
        prof2 = {"terms": list(reversed(prof["terms"]))}
        b = subscribe.score_papers([dict(paper)], prof2)[0]
        self.assertEqual(a["reasons"], b["reasons"])
        self.assertEqual(a["raw_score"], b["raw_score"])

    def test_empty_profile_scores_zero_but_keeps_order(self):
        got = subscribe.score_papers(
            [{"arxiv_id": "z2", "title": "agent"}, {"arxiv_id": "z1", "title": "agent"}],
            {"terms": []})
        self.assertEqual([p["score"] for p in got], [0.0, 0.0])
        self.assertEqual([p["arxiv_id"] for p in got], ["z1", "z2"])


# ── 4. llm_rerank ──

class LLMRerankTests(SubscribeTestBase):
    @staticmethod
    def _scored():
        return [{"arxiv_id": "r1", "title": "A", "abstract": "", "score": 0.9},
                {"arxiv_id": "r2", "title": "B", "abstract": "", "score": 0.5}]

    def test_offline_returns_unchanged_and_flags_degraded(self):
        got = subscribe.llm_rerank(self._scored(), "agent 评测")
        self.assertEqual([p["arxiv_id"] for p in got], ["r1", "r2"])
        self.assertNotIn("llm_note", got[0])
        _, note = subscribe._rerank(self._scored(), "agent 评测")
        self.assertIn("未配置 LLM", note)

    def test_does_not_mutate_caller_dicts(self):
        """重排是「给我一份新排序」，不是「就地改我的数据」。"""
        original = self._scored()
        reply = json.dumps({"items": [{"i": 0, "score": 0.1, "note": "跑题"}]})
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", return_value=reply):
            subscribe.llm_rerank(original, "agent 评测")
        self.assertEqual(original, self._scored())

    def test_llm_failure_does_not_break_the_pipeline(self):
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", side_effect=RuntimeError("boom")):
            got, note = subscribe._rerank(self._scored(), "agent 评测")
        self.assertEqual([p["arxiv_id"] for p in got], ["r1", "r2"])   # 原样返回
        self.assertIn("重排失败", note)

    def test_llm_reranks_and_attaches_notes(self):
        reply = json.dumps({"items": [{"i": 0, "score": 0.1, "note": "跑题"},
                                      {"i": 1, "score": 1.0, "note": "正中方向"}]})
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", return_value=reply):
            got, note = subscribe._rerank(self._scored(), "agent 评测")
        self.assertIsNone(note)
        # r1: 0.5*0.9+0.5*0.1 = 0.5；r2: 0.5*0.5+0.5*1.0 = 0.75 → r2 升到第一
        self.assertEqual([p["arxiv_id"] for p in got], ["r2", "r1"])
        self.assertEqual(got[0]["llm_note"], "正中方向")
        self.assertEqual(got[0]["rank_score"], 0.75)
        self.assertEqual(got[0]["score"], 0.5)          # 确定性分数没被模型覆盖

    def test_garbage_json_falls_back(self):
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", return_value='{"items": []}'):
            got, note = subscribe._rerank(self._scored(), "topic")
        self.assertEqual([p["arxiv_id"] for p in got], ["r1", "r2"])
        self.assertIn("没有可用条目", note)


# ── 5. digest ──

class _DigestFixture:
    """digest 系列共用的夹具。故意不继承 TestCase：否则被子类一继承，
    整套 digest 用例会在 discover 时跑两遍（看起来「用例更多」，其实只是重复）。"""

    def _seed_library(self):
        for i in range(3):
            self._add_paper(f"arxiv:900{i}", f"Agent evaluation study {i}",
                            keywords=["agent", "evaluation"],
                            created_at="2026-08-31 10:00:00")

    @staticmethod
    def _fetched():
        """两篇候选，第一篇明显更贴画像（标题里就有 agent）。"""
        return [
            {"norm_key": "arxiv:2608.00011", "arxiv_id": "2608.00011v1",
             "title": "Agent Evaluation with Tool Use", "abstract": "agent evaluation",
             "published": "2026-08-31T09:00:00Z", "authors": ["Alice Smith"],
             "venue": "arXiv", "source": "arxiv", "year": 2026,
             "oa_pdf_url": "https://arxiv.org/pdf/2608.00011v1"},
            {"norm_key": "arxiv:2608.00012", "arxiv_id": "2608.00012v1",
             "title": "Quantum Cheese Manufacturing", "abstract": "nothing related",
             "published": "2026-08-30T09:00:00Z", "authors": ["Bob Lee"],
             "venue": "arXiv", "source": "arxiv", "year": 2026,
             "oa_pdf_url": "https://arxiv.org/pdf/2608.00012v1"},
        ]

    def _run(self, fetched=None, **kw):
        with mock.patch.object(subscribe, "fetch_recent",
                               return_value=list(fetched if fetched is not None
                                                 else self._fetched())):
            return subscribe.digest(**kw)


class DigestTests(_DigestFixture, SubscribeTestBase):
    def test_offline_digest_runs_and_flags_degraded(self):
        self._seed_library()
        out = self._run(use_llm=True, top_k=5)
        self.assertEqual(out["n_fetched"], 2)
        self.assertEqual(out["n_new"], 2)
        self.assertEqual(len(out["items"]), 2)
        self.assertTrue(any("未配置 LLM" in d for d in out["degraded"]))
        self.assertEqual(out["items"][0]["arxiv_id"], "2608.00011v1")
        self.assertGreater(out["items"][0]["score"], out["items"][1]["score"])
        self.assertTrue(out["items"][0]["reasons"])     # 必须说得清为什么推
        self.assertEqual(out["date"], out["date"][:10])

    def test_papers_already_in_library_are_dropped(self):
        self._seed_library()
        self._add_paper("arxiv:2608.00011", "Agent Evaluation with Tool Use",
                        created_at="2026-08-31 10:00:00")
        out = self._run(top_k=5)
        self.assertEqual(out["n_fetched"], 2)
        self.assertEqual(out["n_new"], 1)
        self.assertEqual([i["norm_key"] for i in out["items"]], ["arxiv:2608.00012"])

    def test_no_repeat_within_window(self):
        self._seed_library()
        first = self._run(top_k=5)
        self.assertEqual(len(first["items"]), 2)
        second = self._run(top_k=5)
        self.assertEqual(second["n_new"], 0)
        self.assertEqual(second["items"], [])

    def test_repeat_allowed_again_after_window_expires(self):
        self._seed_library()
        first = self._run(top_k=5)
        with db.conn() as c:      # 把上一份 digest 挪到 60 天前
            c.execute("UPDATE digests SET ts=datetime('now','localtime','-60 days') "
                      "WHERE id=?", (first["digest_id"],))
        again = self._run(top_k=5, repeat_after_days=30)
        self.assertEqual(again["n_new"], 2)

    def test_repeat_window_zero_disables_the_filter(self):
        self._seed_library()
        self._run(top_k=5)
        again = self._run(top_k=5, repeat_after_days=0)
        self.assertEqual(again["n_new"], 2)

    def test_top_k_truncates_but_n_new_stays_honest(self):
        self._seed_library()
        out = self._run(top_k=1)
        self.assertEqual(out["n_new"], 2)               # 候选数如实报
        self.assertEqual(len(out["items"]), 1)

    def test_no_category_no_fabrication(self):
        """空库 → 猜不出分类 → 不拉取、如实降级，而不是硬塞一个 cs.LG。"""
        with mock.patch.object(subscribe, "fetch_recent") as fetch:
            out = subscribe.digest(top_k=5)
        fetch.assert_not_called()
        self.assertEqual(out["n_fetched"], 0)
        self.assertEqual(out["items"], [])
        self.assertTrue(any("arXiv 分类" in d for d in out["degraded"]))

    def test_fetch_failure_is_degraded_not_crash(self):
        self._seed_library()
        with mock.patch.object(subscribe, "fetch_recent",
                               side_effect=RuntimeError("network down")):
            out = subscribe.digest(top_k=5)
        self.assertEqual(out["items"], [])
        self.assertTrue(any("拉取失败" in d for d in out["degraded"]))

    def test_progress_callback_is_driven(self):
        self._seed_library()
        seen = []
        with mock.patch.object(subscribe, "fetch_recent", return_value=self._fetched()):
            subscribe.digest(top_k=5, progress=lambda f, s, m: seen.append((f, s)))
        self.assertEqual(seen[-1][1], "done")
        self.assertEqual(seen[-1][0], 1.0)
        self.assertEqual([f for f, _ in seen], sorted(f for f, _ in seen))

    def test_end_to_end_through_real_fetch_recent(self):
        """不打桩 fetch_recent，只打桩 HTTP：确认两段真的接得上（字段名对不上会在这里露馅）。"""
        self._seed_library()
        fake, patch = self._fake_http(FEED_OK)
        with patch:
            # now 固定：否则这个用例会在 2026-09-01 之后自己过期（时间炸弹）
            out = subscribe.digest(days=7, top_k=5, categories=["cs.CL"], now=NOW)
        self.assertEqual(out["n_fetched"], 3)
        self.assertEqual(out["n_new"], 3)
        top = out["items"][0]
        self.assertEqual(top["arxiv_id"], "2608.00011v1")     # 标题里就有 agent
        self.assertEqual(top["norm_key"], "arxiv:2608.00011")
        self.assertTrue(top["reasons"])
        self.assertEqual(top["authors"], ["Alice Smith", "Bob Lee"])
        # 落库的内容能原样取回，adopt 也能基于它入库
        full = subscribe.get_digest(out["digest_id"])
        self.assertEqual(len(full["items"]), 3)
        r = subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        self.assertEqual(r["status"], "adopted")

    def test_digest_is_reproducible(self):
        """同样的输入连跑两次（关掉时间窗），推荐顺序与分数必须一模一样。"""
        self._seed_library()
        a = self._run(top_k=5, repeat_after_days=0)
        b = self._run(top_k=5, repeat_after_days=0)
        self.assertEqual([(i["norm_key"], i["score"]) for i in a["items"]],
                         [(i["norm_key"], i["score"]) for i in b["items"]])


# ── 6. adopt / dismiss / 历史 ──

class AdoptDismissTests(_DigestFixture, SubscribeTestBase):
    def test_adopt_puts_paper_in_library(self):
        self._seed_library()
        out = self._run(top_k=5)
        r = subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        self.assertEqual(r["status"], "adopted")
        with db.conn() as c:
            row = db.get_by_norm_key(c, "arxiv:2608.00011")
        self.assertIsNotNone(row)
        self.assertEqual(row["id"], r["paper_id"])
        self.assertEqual(row["norm_key"], "arxiv:2608.00011")
        self.assertEqual(row["source"], "arxiv-digest")
        self.assertEqual(row["arxiv_id"], "2608.00011v1")
        self.assertEqual(row["title"], "Agent Evaluation with Tool Use")
        self.assertEqual(json.loads(row["authors"]), ["Alice Smith"])
        self.assertEqual(row["year"], 2026)

    def test_adopt_twice_is_idempotent(self):
        self._seed_library()
        out = self._run(top_k=5)
        first = subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        second = subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        self.assertEqual(second["status"], "exists")
        self.assertEqual(second["paper_id"], first["paper_id"])
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM papers WHERE norm_key=?",
                          ("arxiv:2608.00011",)).fetchone()["n"]
        self.assertEqual(n, 1)

    def test_adopt_survives_a_concurrent_insert(self):
        """「查一下不在 → 插进去」之间被别人抢先插入（前端连点两次、后台检索任务
        同时把这篇捞进来）不该炸成 IntegrityError/500，语义上就是「已在库」。"""
        self._seed_library()
        out = self._run(top_k=5)
        real = db.get_by_norm_key
        fired = []

        def racy(c, key):
            row = real(c, key)
            # 抢先者必须是**另一条连接**并且已提交，否则 adopt 里的 rollback
            # 会把它一起回滚掉，就模拟不出真实的并发了。
            if row is None and key == "arxiv:2608.00011" and not fired:
                fired.append(1)
                with db.conn() as other:
                    db.insert_l0(other, {"norm_key": key, "title": "抢先插入的同一篇",
                                         "source": "s2"})
            return row

        with mock.patch.object(db, "get_by_norm_key", racy):
            r = subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        self.assertEqual(r["status"], "exists")
        self.assertIsNotNone(r["paper_id"])
        with db.conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM papers WHERE norm_key=?",
                          ("arxiv:2608.00011",)).fetchone()["n"]
        self.assertEqual(n, 1)

    def test_adopt_unknown_item_raises(self):
        self._seed_library()
        out = self._run(top_k=5)
        with self.assertRaises(subscribe.SubscribeError):
            subscribe.adopt(out["digest_id"], "arxiv:does-not-exist")

    def test_dismissed_paper_never_comes_back(self):
        self._seed_library()
        first = self._run(top_k=5)
        subscribe.dismiss(first["digest_id"], "arxiv:2608.00011")
        # 连时间窗都关掉，被忽略的那篇仍然不该回来
        again = self._run(top_k=5, repeat_after_days=0)
        keys = [i["norm_key"] for i in again["items"]]
        self.assertNotIn("arxiv:2608.00011", keys)
        self.assertEqual(keys, ["arxiv:2608.00012"])

    def test_dismiss_unknown_item_raises(self):
        self._seed_library()
        out = self._run(top_k=5)
        with self.assertRaises(subscribe.SubscribeError):
            subscribe.dismiss(out["digest_id"], "arxiv:nope")

    def test_list_and_get_digest(self):
        self._seed_library()
        out = self._run(top_k=5)
        subscribe.adopt(out["digest_id"], "arxiv:2608.00011")
        subscribe.dismiss(out["digest_id"], "arxiv:2608.00012")

        listed = subscribe.list_digests(10)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], out["digest_id"])
        self.assertEqual(listed[0]["n_items"], 2)
        self.assertEqual(listed[0]["n_dismissed"], 1)

        full = subscribe.get_digest(out["digest_id"])
        self.assertEqual(len(full["items"]), 2)
        adopted = next(i for i in full["items"] if i["norm_key"] == "arxiv:2608.00011")
        self.assertTrue(adopted["in_library"])          # 实时 JOIN papers 判定
        self.assertIsNotNone(adopted["paper_id"])
        self.assertTrue(adopted["reasons"])
        self.assertEqual(adopted["authors"], ["Alice Smith"])
        other = next(i for i in full["items"] if i["norm_key"] == "arxiv:2608.00012")
        self.assertFalse(other["in_library"])
        self.assertTrue(other["dismissed"])

    def test_get_missing_digest_returns_none(self):
        self.assertIsNone(subscribe.get_digest(4242))


if __name__ == "__main__":
    unittest.main()
