"""对**本批修复自身**引入的缺陷的回归。

这些不是历史遗留问题，是那六项修复自己带进来的——由一次独立对抗审计
（7 维度、每条再核验）发现。留在这里是因为它们大多是「修 A 的时候踩出 B」，
而 B 往往正是 A 声称要消灭的那个形状。
"""
from __future__ import annotations

import hmac
import io
import re
import unittest
from pathlib import Path
from unittest import mock

from papernest import api, chat, db, degrade
from papernest.stats import describe, sign_flip_test

try:                                    # discover -s tests 时 tests/ 不当包用
    from .support import make_tempdir
except ImportError:
    from support import make_tempdir

try:
    from .support import TempDbTestCase
except ImportError:                          # noqa: F401
    from support import TempDbTestCase

ROOT = Path(__file__).resolve().parent.parent


def _read(p) -> str:
    with io.open(p, encoding="utf-8") as f:
        return f.read()


class NonAsciiApiKey(unittest.TestCase):
    """`hmac.compare_digest` 对两个 str 只支持 ASCII；含中文的口令会 TypeError。

    而 fix #5 让 docker-compose 用**中文提示**强制要求口令——填中文口令是完全正常的
    选择，然后每个 /api/* 恒定 500（不是 401），前端只在 401 才弹输入框，
    用户连「口令错了」都看不到。这是修复自己制造的整站不可用路径。
    """

    def test_raw_str_compare_would_have_crashed(self):
        with self.assertRaises(TypeError):
            hmac.compare_digest("口令", "口令")

    def test_middleware_encodes_before_comparing(self):
        src = _read(ROOT / "papernest" / "api.py")
        i = src.index("def _auth")
        body = src[i:i + 900]
        self.assertIn("encode", body, "compare_digest 两侧没有 encode，中文口令会 500")

    def test_non_ascii_key_is_rejected_at_startup(self):
        """比「能比较」更进一步：HTTP 头本身就装不下非 ASCII，客户端发不出去。
        与其留一个「SSE 能过、普通请求全废」的半坏状态，不如启动时拒绝。"""
        with mock.patch.object(api, "API_KEY", "中文口令"):
            with self.assertRaises(RuntimeError) as cm:
                api._check_key_transportable()
            self.assertIn("ASCII", str(cm.exception))

    def test_ascii_key_passes_and_authenticates(self):
        """**必须先把库指到临时目录**：带对口令的那次请求会真的进 `/api/stats`，
        而它开头就是 `db.init_db()`——那是写路径，会对用户的生产库执行
        `executescript(SCHEMA)` + `_migrate()`。实测就是这条用例把新加的
        `table_summaries` 建到了真库里（2026-09-09）。
        `test_eval.py` 早就为同一件事栽过一次（user_version 5→6），
        那次是直接调 init_db，这次是隔着 TestClient 的线程池，更难看出来。
        """
        from fastapi.testclient import TestClient

        from papernest import config
        tmp = make_tempdir(self, "papernest_apikey")
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        self.addCleanup(lambda: (setattr(config, "DB_PATH", old_db),
                                 setattr(config, "DATA_DIR", old_dir)))
        config.DATA_DIR = tmp / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"

        with mock.patch.object(api, "API_KEY", "s3cret-ascii"):
            api._check_key_transportable()          # 不该抛
            client = TestClient(api.app)
            self.assertEqual(client.get("/api/stats").status_code, 401)
            self.assertEqual(
                client.get("/api/stats", headers={"x-api-key": "wrong"}).status_code, 401)
            self.assertEqual(
                client.get("/api/stats",
                           headers={"x-api-key": "s3cret-ascii"}).status_code, 200)

    def test_empty_key_does_not_trip_the_check(self):
        with mock.patch.object(api, "API_KEY", ""):
            api._check_key_transportable()


class FrontendResourceLoadsCarryTheKey(unittest.TestCase):
    """`<img src>` / `<iframe src>` / `<a href>` 是**浏览器自己**发起的，
    不经过 window.fetch，包装器碰不到，也没法给它们设请求头。

    fix #5 只包了 fetch/EventSource，于是设了口令后 PDF 阅读视图与「跳到第 N 页」
    整条 401；而 <img> 的 onerror 会显示成「这一页取不到」，看起来像渲染 bug。
    """

    @classmethod
    def setUpClass(cls):
        cls.html = _read(ROOT / "web" / "index.html")

    def test_helper_exists(self):
        self.assertIn("function pnUrl(", self.html,
                      "缺少给子资源拼 api_key 的助手")

    def test_pdf_iframe_uses_it(self):
        self.assertRegex(self.html, r"<iframe src=\"\$\{pnUrl\(")

    def test_page_image_uses_it(self):
        self.assertRegex(self.html, r"<img src=\"\$\{pnUrl\(")

    def test_big_image_link_uses_it(self):
        i = self.html.index("大图")
        self.assertIn("pnUrl(", self.html[max(0, i - 400):i])

    def test_no_bare_api_paper_resource_urls_remain(self):
        """把口令拼进去之后，不该还有裸的 /api/paper/.../pdf 或 .png 出现在
        src= / href= 里。"""
        import re
        bare = re.findall(r'(?:src|href)="/api/paper/[^"]*"', self.html)
        self.assertEqual(bare, [], f"仍有绕过口令的资源加载：{bare}")


class DegradedIsPropagatedNotDiscarded(TempDbTestCase):
    """fix #2 声称「降级信号接通」，但 rcs 与 agent 两条链路根本没接。"""

    prefix = "papernest_regr_deg"

    def test_rcs_does_not_discard_the_retrieval_trace(self):
        src = _read(ROOT / "papernest" / "rcs.py")
        self.assertNotIn("_rt = deepsearch.deep_retrieve", src,
                         "检索 trace 被丢进下划线变量——正是本项目要消灭的形状")
        self.assertIn("retrieval_notes", src)

    def test_rcs_success_path_does_not_hardcode_none(self):
        src = _read(ROOT / "papernest" / "rcs.py")
        self.assertNotIn('"degraded": None', src,
                         "成功路径硬编码 degraded=None 是主动的假阴性")

    def test_agent_accumulates_instead_of_overwriting(self):
        """第 1 步的 CRITICAL 降级不能被第 2 步的低级降级顶掉。"""
        src = _read(ROOT / "papernest" / "agent.py")
        self.assertNotIn("degraded = data[\"degraded\"]", src,
                         "后写覆盖：前一步的严重降级会被后一步顶掉")
        self.assertIn("degrade.merge", src)

    def test_merge_keeps_the_critical_one(self):
        a = [degrade.Degradation(degrade.VECTOR_INDEX_EMPTY, "向量整路失败")]
        b = [degrade.Degradation(degrade.PAGE_PICK_FALLBACK, "按关键词选页")]
        merged = degrade.merge(a, b)
        self.assertTrue(degrade.has_critical(merged),
                        "合并之后 CRITICAL 记录不见了")
        self.assertEqual(len(merged), 2)


class ArxivHttpUrlsStillWork(unittest.TestCase):
    """fix #5 的 netguard 只放行 https，而 S2 的 openAccessPdf 会给出
    `http://arxiv.org/pdf/...`（真库 5 篇）——这些论文被我们自己的防线永久挡死。"""

    def test_fallback_triggers_on_non_https_url(self):
        src = _read(ROOT / "papernest" / "fulltext.py")
        i = src.index("def fetch_pdf")
        body = src[i:i + 1200]
        self.assertIn("startswith(\"https://\")", body,
                      "arXiv 兜底只判 `not url`，http:// 链接会被 SSRF 防线挡死")

    def test_http_arxiv_is_upgraded_not_blocked(self):
        from papernest import fulltext
        row = {"id": 1, "arxiv_id": "2403.11809",
               "oa_pdf_url": "http://arxiv.org/pdf/2403.11809"}
        url = row["oa_pdf_url"]
        if row["arxiv_id"] and (not url or not url.lower().startswith("https://")):
            url = f"https://arxiv.org/pdf/{row['arxiv_id']}"
        self.assertTrue(url.startswith("https://"))
        # 且升级后的地址能过 netguard 的 scheme/端口校验
        from papernest import netguard
        with mock.patch.object(netguard, "resolve", return_value=["151.101.3.42"]):
            netguard.check_url(url)


class AsciiRunsAreStripped(unittest.TestCase):
    """fix #3 的 docstring 以 `XL-MIMO?` 为例声称修掉了「标点粘在词面上」，
    但中英混合词这个入口原样复现：`_ASCII_RUN` 的字符类含 . - /。"""

    def test_mixed_token_with_trailing_hyphen(self):
        terms = db._expand_terms("XL-MIMO-的近场估计")
        self.assertIn("XL-MIMO", terms)
        self.assertNotIn("XL-MIMO-", terms, "ASCII 段带着尾连字符出来了")

    def test_mixed_token_with_trailing_period(self):
        terms = db._expand_terms("用BERT.做的实验")
        self.assertIn("BERT", terms)
        self.assertNotIn("BERT.", terms)

    def test_technical_tokens_still_survive_in_mixed_words(self):
        self.assertIn("GPT-4", db._expand_terms("GPT-4的表现如何"))


class DescribeRespectsSign(unittest.TestCase):
    """`stats.describe` 在不显著分支硬编码「方向为正」，不看 mean_diff 符号——
    而这个模块存在的唯一理由就是如实报告负结果。"""

    def test_negative_delta_is_not_called_positive(self):
        r = sign_flip_test([1.0] * 32, [0.0] + [1.0] * 31)
        text = describe(r)
        self.assertLess(r["mean_diff"], 0)
        self.assertIn("方向为负", text)
        self.assertNotIn("方向为正", text, "变差的结果被描述成了方向为正")

    def test_positive_delta_still_says_positive(self):
        r = sign_flip_test([0.0] + [1.0] * 31, [1.0] * 32)
        self.assertIn("方向为正", describe(r))

    def test_arrow_and_wording_agree(self):
        for before, after in (([1.0] * 32, [0.0] + [1.0] * 31),
                              ([0.0] + [1.0] * 31, [1.0] * 32)):
            t = describe(sign_flip_test(before, after))
            if "↓" in t:
                self.assertIn("方向为负", t)
            if "↑" in t:
                self.assertIn("方向为正", t)


class AssignIndicesLockScope(TempDbTestCase):
    """fix #6 修 lost update 是对的，但闸门开得太宽：什么都不用写的调用
    也在入口抢 SQLite 写锁，而它在 rag.prepare 的必经路径上。"""

    prefix = "papernest_lockscope"

    def setUp(self):
        super().setUp()
        db.init_db()
        self.sid = chat.ensure_session(None, "s")

    def test_read_only_call_takes_no_write_lock(self):
        chat.assign_indices(self.sid, [1, 2, 3])          # 先建好编号
        seen = {"immediate": 0}
        real = db.conn

        def spy(immediate=False):
            if immediate:
                seen["immediate"] += 1
            return real(immediate=immediate)

        with mock.patch.object(db, "conn", spy):
            got = chat.assign_indices(self.sid, [1, 2, 3])   # 全都已有编号
        self.assertEqual(got, {1: 1, 2: 2, 3: 3})
        self.assertEqual(seen["immediate"], 0,
                         "没有新论文也去抢了写锁")

    def test_new_paper_still_takes_the_write_lock(self):
        seen = {"immediate": 0}
        real = db.conn

        def spy(immediate=False):
            if immediate:
                seen["immediate"] += 1
            return real(immediate=immediate)

        with mock.patch.object(db, "conn", spy):
            chat.assign_indices(self.sid, [9])
        self.assertEqual(seen["immediate"], 1, "有新论文却没走写事务，竞态会回来")

    def test_still_atomic_under_concurrency(self):
        """收窄闸门不能把 lost update 放回来。"""
        import threading
        from collections import Counter
        N = 12
        got = {}
        barrier = threading.Barrier(N)

        def w(pid):
            barrier.wait(timeout=10)
            got[pid] = chat.assign_indices(self.sid, [pid])[pid]

        ths = [threading.Thread(target=w, args=(500 + i,)) for i in range(N)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(timeout=60)
        dupes = {i: n for i, n in Counter(got.values()).items() if n > 1}
        self.assertFalse(dupes, f"收窄闸门后竞态回来了：{dupes}")


if __name__ == "__main__":
    unittest.main()


class FrontendEscapingTests(unittest.TestCase):
    """前端三条：作者字段 XSS、内联事件属性里的引号、流式出错不渲染。

    **这三条的行为已在真浏览器里验过**（起 `cli.py serve`，在真页面上跑）：
      · `onclick="openPaperByName(${jsArg(name)})"` 对
        "Assessing BERT's Syntactic Abilities"（真库里就有这一篇）编译成功、逐字还原；
        而仓库原来的写法 `esc(x).replace(/'/g,'&#39;')` **是坏的**——HTML 解析器会在
        把属性值交给 JS 编译之前把 `&#39;` 解回单引号，实测 `el.onclick === null`，
        点了没任何反应。所以那不是「漏打补丁」，是补丁本身不对。
      · 作者串里的 `<img src=x onerror=...>` 现在渲染成**文本**（0 个 img 元素、
        XSS 未触发）。
      · 用假 SSE 流（sources → delta → error，之后不发 done）驱动真的 `doChat()`：
        UI 不再停在「正在组织回答…」，半截答案 / 错误 / 来源都显示出来。

    下面是静态守卫，只负责防止有人把这三处改回去；**它们不能替代上面那次真跑**
    （本仓已经三次栽在「grep 过了就以为验证了」上）。
    """

    @classmethod
    def setUpClass(cls):
        cls.html = _read(ROOT / "web" / "index.html")

    def test_authors_go_through_esc(self):
        self.assertNotIn("${(p.authors||[]).join(', ')}", self.html,
                         "作者字段又裸进 innerHTML 了——来源是外部元数据与用户 prompt() 输入")
        self.assertIn("${esc((p.authors||[]).join(', '))}", self.html)

    def test_no_inline_attribute_builds_a_js_string_by_hand(self):
        """内联事件属性里不许再出现 `f('${...}')` 这种手拼 JS 字符串。"""
        bad = re.findall(r"on\w+=\"[^\"]*\('\$\{", self.html)
        self.assertEqual(bad, [], f"这些内联属性还在手拼 JS 字符串：{bad[:3]}")
        self.assertIn("const jsArg=", self.html, "jsArg 辅助函数没了")

    def test_the_broken_quote_patch_is_gone_from_code(self):
        """`replace(/'/g,'&#39;')` 这个写法必须绝迹（注释里解释它为什么错的不算）。"""
        lines = [ln for ln in self.html.splitlines()
                 if "replace(/'/g" in ln and not ln.strip().startswith("//")]
        self.assertEqual(lines, [], f"还有 {len(lines)} 处在用那个坏写法")

    def test_stream_error_branch_finishes_the_message(self):
        """服务端发完 error 就 return、**不再发 done**，前端必须自己收尾。"""
        i = self.html.index("else if(ev==='error')")
        j = self.html.index("else if(ev==='done')", i)
        branch = self.html[i:j]
        self.assertIn("done(", branch,
                      "error 分支没有调用 done()：UI 会永远停在「正在组织回答…」")
        self.assertIn("hist.push(", branch, "半截答案没有进历史，下一轮就丢了上下文")
