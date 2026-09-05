"""部署安全：默认形态不该是「谁都能删数据 + 烧 LLM 额度」。

审计确认的形态：
- 鉴权是可选项（不设 `PAPERNEST_API_KEY` 就整段放行），而唯一的部署方式
  docker-compose 把 8765 绑到 0.0.0.0、也没设这个变量；
- **就算设了口令，自带前端一处都不带**（30 处 fetch + 1 处 EventSource 全裸），
  界面会全线 401 ——于是这道锁事实上不可启用，运维只能撤掉口令退回裸奔；
- 服务端鉴权本身是有测试的，但没有任何用例检查**前端能不能带上口令**，
  于是「鉴权已覆盖」是一种假信心。

所以这里既测服务端，也测前端契约与部署配置——三者任缺其一，这把锁就开不起来。
"""
from __future__ import annotations

import io
import re
import unittest
from pathlib import Path
from unittest import mock

from papernest import api, health, netguard

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "index.html"


def _read(p) -> str:
    """读文本并**关闭文件**——裸 io.open().read() 会漏 fd，跑全套时刷一屏 ResourceWarning。"""
    with io.open(p, encoding="utf-8") as f:
        return f.read()


class StartupExposureGuard(unittest.TestCase):
    """绑了非回环却没口令 —— 必须拒绝启动，而不是打一行日志继续跑。"""

    def _guard(self, host, key, allow=False):
        with mock.patch.object(api, "BIND_HOST", host), \
             mock.patch.object(api, "API_KEY", key), \
             mock.patch.object(api, "ALLOW_NO_AUTH", allow):
            api._check_exposure()

    def test_exposed_without_key_refuses_to_start(self):
        with self.assertRaises(RuntimeError) as cm:
            self._guard("0.0.0.0", "")
        msg = str(cm.exception)
        self.assertIn("PAPERNEST_API_KEY", msg, "报错要说清怎么修")
        self.assertIn("PAPERNEST_ALLOW_NO_AUTH", msg, "报错要给出逃生口")

    def test_loopback_without_key_is_fine(self):
        """本机自用是主要场景，不能因为加了闸门就把它挡掉。"""
        for host in ("127.0.0.1", "::1", "localhost"):
            self._guard(host, "")

    def test_exposed_with_key_is_fine(self):
        self._guard("0.0.0.0", "s3cret")

    def test_explicit_opt_out_is_honoured(self):
        self._guard("0.0.0.0", "", allow=True)


class FrontendSendsTheKey(unittest.TestCase):
    """前端契约：这些断言此前一条都没有，而缺的正是它们。"""

    @classmethod
    def setUpClass(cls):
        cls.html = _read(WEB)

    def test_fetch_is_wrapped_to_inject_the_header(self):
        self.assertIn("x-api-key", self.html,
                      "前端从不发送 x-api-key —— 设了口令界面就全线 401")

    def test_eventsource_uses_query_param(self):
        """EventSource 不能设请求头，SSE 只能走 ?api_key=。"""
        self.assertRegex(self.html, r"api_key=",
                         "SSE 没有带口令的途径，任务进度流会 401")

    def test_key_is_read_from_local_storage_not_hardcoded(self):
        self.assertIn("localStorage", self.html)
        # 硬编码的口令会随代码分发出去
        for pat in (r"x-api-key['\"]\s*[,:]\s*['\"][A-Za-z0-9]{6,}",
                    r"api_key=[A-Za-z0-9]{6,}"):
            self.assertNotRegex(self.html, pat, "前端里疑似硬编码了口令")

    def test_every_api_call_goes_through_the_wrapper(self):
        """包的是 window.fetch 本身，所以 30 处调用点无需逐个改——
        但必须确认没有绕过包装的原生调用残留。"""
        self.assertIn("window.fetch=", self.html.replace(" ", ""),
                      "没有包装 window.fetch，个别调用点会绕过口令注入")


class DeploymentConfig(unittest.TestCase):

    def test_compose_binds_loopback_only(self):
        text = _read(ROOT / "docker-compose.yml")
        ports = re.findall(r'-\s*"([^"]+)"', text)
        self.assertTrue(ports, "compose 里找不到端口映射")
        for p in ports:
            self.assertTrue(p.startswith("127.0.0.1:"),
                            f"端口 {p} 绑在所有网卡上——同网段可无凭据访问")

    def test_compose_requires_the_key(self):
        text = _read(ROOT / "docker-compose.yml")
        self.assertIn("PAPERNEST_API_KEY:", text)
        self.assertIn(":?", text, "缺口令时 compose 应当直接报错，而不是静默启动")

    def test_dockerfile_declares_its_bind_host(self):
        text = _read(ROOT / "Dockerfile")
        self.assertIn("PAPERNEST_BIND_HOST", text,
                      "容器内绑 0.0.0.0 却没声明，启动自检就形同虚设")


class HealthDoesNotLeakEndpoints(unittest.TestCase):

    def test_config_check_reports_host_not_full_url(self):
        """/api/health/retrieval 在未设口令时可匿名读，别把私有中转站地址交出去。"""
        self.assertEqual(health._host_only("https://gw.example.com/v1/x?k=1"),
                         "gw.example.com")

    def test_host_only_survives_garbage(self):
        for bad in ("", None, "not a url"):
            self.assertIsInstance(health._host_only(bad), str)


class SsrfGuard(unittest.TestCase):
    """`oa_pdf_url` 可由用户上传的 .bib 写入，是不可信输入。"""

    def _allow(self, url, ips):
        with mock.patch.object(netguard, "resolve", return_value=ips):
            return netguard.check_url(url)

    def _blocked(self, url, ips=("93.184.216.34",)):
        with mock.patch.object(netguard, "resolve", return_value=list(ips)):
            with self.assertRaises(netguard.BlockedURL):
                netguard.check_url(url)

    def test_public_https_is_allowed(self):
        self._allow("https://arxiv.org/pdf/2401.00001", ["151.101.3.42"])

    def test_cloud_metadata_service_is_blocked(self):
        self._blocked("https://169.254.169.254/latest/meta-data/pdf/",
                      ["169.254.169.254"])

    def test_private_ranges_are_blocked(self):
        for ip in ("10.0.0.5", "192.168.1.1", "172.16.0.1", "127.0.0.1", "::1"):
            self._blocked(f"https://internal.example.com/x.pdf", [ip])

    def test_a_single_bad_record_blocks_the_whole_host(self):
        """一条 A 记录指向内网就拒绝——不给部分命中留缝。"""
        self._blocked("https://mixed.example.com/x.pdf",
                      ["93.184.216.34", "10.0.0.5"])

    def test_http_scheme_is_blocked(self):
        self._blocked("http://arxiv.org/pdf/x", ["151.101.3.42"])

    def test_nonstandard_port_is_blocked(self):
        """放行任意端口等于把内网端口扫描能力交出去。"""
        self._blocked("https://example.com:8080/x.pdf", ["93.184.216.34"])

    def test_unresolvable_host_is_blocked(self):
        with mock.patch.object(netguard, "resolve", return_value=[]):
            with self.assertRaises(netguard.BlockedURL):
                netguard.check_url("https://nope.invalid/x.pdf")

    def test_error_message_does_not_reach_the_caller(self):
        """拒绝原因是内网探测的反馈，不能原样回吐。"""
        src = _read(ROOT / "papernest" / "fulltext.py")
        i = src.index("except netguard.BlockedURL")
        block = src[i:src.index("except Exception", i)]   # 只看这一个处理块
        self.assertNotIn("{e}", block, "把 BlockedURL 的细节回吐给了调用方")
        self.assertIn("不允许访问", block)

    def test_fetch_pdf_does_not_follow_redirects_automatically(self):
        """自动跟随重定向会绕过入口校验：合法外域 302 到内网同样打得进来。"""
        src = _read(ROOT / "papernest" / "fulltext.py")
        i = src.index("def fetch_pdf")
        body = src[i:src.index("def extract_pages")]
        self.assertIn("follow_redirects=False", body)
        self.assertIn("netguard.check_url", body)
        self.assertGreaterEqual(body.count("netguard.check_url"), 2,
                                "重定向的每一跳都要重新校验，不能只校验入口")


class HttpClientDefaults(unittest.TestCase):
    """`http.client()` 的默认值必须**可覆盖**。

    回归：`follow_redirects=True` 原来是硬写死的关键字，调用方再传一次就
    `TypeError: got multiple values`，而该异常被调用点的 `except Exception` 吞成
    一句「下载失败」。结果是 SSRF 那条路径上**校验根本没跑到**，表面却像是生效了
    ——静态断言全绿，端到端一打才露出来。
    """

    def test_follow_redirects_can_be_overridden(self):
        from papernest import http as pn_http
        with pn_http.client(follow_redirects=False) as c:
            self.assertFalse(c.follow_redirects)

    def test_default_is_still_follow_redirects(self):
        from papernest import http as pn_http
        with pn_http.client() as c:
            self.assertTrue(c.follow_redirects)

    def test_other_defaults_are_overridable_too(self):
        from papernest import http as pn_http
        with pn_http.client(trust_env=True) as c:
            self.assertTrue(c.trust_env)


class SsrfEndToEnd(unittest.TestCase):
    """真起一个本地 HTTP 靶机，用真实的 fetch_pdf 去打——靶机必须一次都收不到请求。

    这一层不能只靠静态断言：上面那个 TypeError 就是「源码里写了校验、
    但那条路径压根没执行到」的活例子。
    """

    def test_internal_target_is_never_reached(self):
        import shutil
        import tempfile
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        hits = []

        class Target(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                body = b"%PDF-1.4 internal"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Target)
        port = srv.server_port
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)

        from papernest import config, db, fulltext
        tmp = Path(tempfile.mkdtemp(prefix="papernest_ssrf_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        old = (config.DATA_DIR, config.DB_PATH, fulltext.PDF_DIR)
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", old[0]),
                                 setattr(config, "DB_PATH", old[1]),
                                 setattr(fulltext, "PDF_DIR", old[2])))
        config.DATA_DIR, config.DB_PATH = tmp, tmp / "t.db"
        fulltext.PDF_DIR = tmp / "pdf"
        db.init_db()

        urls = [f"http://127.0.0.1:{port}/x.pdf",
                f"https://127.0.0.1:{port}/x.pdf",
                f"https://localhost:{port}/x.pdf",
                "https://169.254.169.254/latest/meta-data/pdf/",
                "file:///etc/passwd"]
        with db.conn() as c:
            for i, url in enumerate(urls, 1):
                pid = db.insert_l0(c, {
                    "norm_key": f"ssrf:{i}", "title": f"T{i}", "abstract": "",
                    "year": 2024, "venue": "", "authors": [], "doi": None,
                    "arxiv_id": None, "source": "bibtex"})
                c.execute("UPDATE papers SET oa_pdf_url=? WHERE id=?", (url, pid))
                c.commit()
                r = fulltext.fetch_pdf(pid)
                self.assertIn("error", r, f"{url} 被放行了")
                self.assertNotIn("path", r)
                # 拒绝原因不能泄露目标地址
                self.assertNotIn("127.0.0.1", r["error"])
                self.assertNotIn("169.254", r["error"])

        self.assertEqual(hits, [], f"内网靶机被真实访问了：{hits}")


if __name__ == "__main__":
    unittest.main()
