"""API 层（FastAPI TestClient）：错误码、任务生命周期、去重、鉴权、会话。

改之前这一层 40+ 个端点一个用例都没有——错误码是 200+error 还是 HTTPException
全靠人肉记忆，任务并发提交也没人守着。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from papernest import api, config, db, jobs  # noqa: F401  (config 用于路径归属校验用例)


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_api_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            mock.patch("papernest.llm.available", return_value=False),
            # 任务不真跑：这里测的是 HTTP 层，跑起来会打外网
            mock.patch("papernest.jobs.submit"),
        ]
        for p in self._patches:
            p.start()
        db.init_db()
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "arxiv:6001", "title": "Near Field Channel Estimation",
                "abstract": "Near field channel estimation for XL-MIMO systems.",
                "year": 2024, "venue": "TWC", "authors": ["Li Wei"],
                "doi": "10.1000/abc", "arxiv_id": "6001", "source": "s2"})
            c.execute("UPDATE papers SET pdf_path=? WHERE id=?",
                      (r"C:\secret\local\path.pdf", self.p1))
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir


class LibraryEndpointTests(ApiTestBase):
    def test_list_papers(self):
        r = self.client.get("/api/papers")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["papers"][0]["title"], "Near Field Channel Estimation")

    def test_list_papers_never_leaks_local_path(self):
        body = self.client.get("/api/papers").json()
        self.assertNotIn("pdf_path", body["papers"][0])

    def test_paper_detail_hides_local_path_but_reports_presence(self):
        body = self.client.get(f"/api/paper/{self.p1}").json()
        self.assertNotIn("pdf_path", body)      # 本机绝对路径不出网
        self.assertTrue(body["has_pdf"])
        self.assertEqual(body["authors"], ["Li Wei"])

    def test_paper_detail_missing_is_404_not_200(self):
        r = self.client.get("/api/paper/999999")
        self.assertEqual(r.status_code, 404)

    def test_year_filter(self):
        self.assertEqual(self.client.get("/api/papers?year_from=2030").json()["count"], 0)
        self.assertEqual(self.client.get("/api/papers?year_to=2024").json()["count"], 1)

    def test_search_by_query(self):
        self.assertEqual(self.client.get("/api/papers?q=channel").json()["count"], 1)

    def test_stats(self):
        body = self.client.get("/api/stats").json()
        self.assertEqual(body["papers_total"], 1)
        self.assertIn("vectors", body)

    def test_export_bibtex(self):
        r = self.client.post("/api/export",
                             json={"paper_ids": [self.p1], "fmt": "bibtex"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("@article{", r.text)
        self.assertIn("Near Field Channel Estimation", r.text)

    def test_export_unknown_format_is_reported(self):
        r = self.client.post("/api/export", json={"paper_ids": [self.p1], "fmt": "xyz"})
        self.assertIn("不支持的格式", r.text)

    def test_ingest_empty_query_is_400(self):
        r = self.client.post("/api/ingest", json={"query": "  "})
        self.assertEqual(r.status_code, 400)


class JobEndpointTests(ApiTestBase):
    def test_unknown_kind_is_400(self):
        r = self.client.post("/api/jobs", json={"kind": "nope", "params": {}})
        self.assertEqual(r.status_code, 400)

    def test_create_and_poll(self):
        r = self.client.post("/api/jobs",
                             json={"kind": "survey", "params": {"topic": "XL-MIMO"}})
        self.assertEqual(r.status_code, 200)
        job_id = r.json()["job_id"]
        got = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["status"], "queued")
        self.assertEqual(got.json()["params"]["topic"], "XL-MIMO")

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/api/jobs/deadbeef").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/deadbeef/events").status_code, 404)

    def test_write_job_is_deduped_per_run(self):
        """同一个写作 run 不能同时挂两个阶段任务——否则两个线程交错写 sections。"""
        first = self.client.post("/api/jobs",
                                 json={"kind": "write",
                                       "params": {"run_id": "r1", "stage": "sections"}})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/api/jobs",
                                  json={"kind": "write",
                                        "params": {"run_id": "r1", "stage": "polish"}})
        self.assertEqual(second.status_code, 409)

    def test_other_kinds_are_not_deduped(self):
        a = self.client.post("/api/jobs", json={"kind": "survey", "params": {"topic": "x"}})
        b = self.client.post("/api/jobs", json={"kind": "survey", "params": {"topic": "x"}})
        self.assertEqual((a.status_code, b.status_code), (200, 200))

    def test_recover_stale_jobs_marks_orphans_failed(self):
        job_id = jobs.create_job("survey", {"topic": "x"})
        jobs.claim_job(job_id)
        self.assertEqual(jobs.get_job(job_id)["status"], "running")
        n = jobs.recover_stale_jobs()           # 模拟服务重启
        self.assertGreaterEqual(n, 1)
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("服务重启", job["error"])

    def test_claim_is_idempotent(self):
        job_id = jobs.create_job("survey", {"topic": "x"})
        self.assertTrue(jobs.claim_job(job_id))
        self.assertFalse(jobs.claim_job(job_id))   # 第二次认领不成功 → 不会跑两遍


class WriteRunEndpointTests(ApiTestBase):
    def test_missing_run_is_404_everywhere(self):
        for path in ("/api/write/runs/nope", "/api/write/runs/nope/report"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        self.assertEqual(
            self.client.post("/api/write/runs/nope/polish").status_code, 404)

    def test_create_run_rejects_short_topic(self):
        """schemas.WriteRunCreate 有 min_length=4，pydantic 先拦下来（422）。"""
        r = self.client.post("/api/write/runs", json={"topic": "ab"})
        self.assertEqual(r.status_code, 422)

    def test_create_run_starts_topic_job(self):
        r = self.client.post("/api/write/runs", json={"topic": "大模型 Agent 评测方法"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["job_id"])
        state = self.client.get(f"/api/write/runs/{body['run']['id']}")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["run"]["job_id"], body["job_id"])


class WorkspaceEndpointTests(ApiTestBase):
    """课题工作区：后端 CRUD 一直都在，但直到现在才有界面用它——顺手把契约钉住。"""

    def _project(self, name="XL-MIMO 近场"):
        return self.client.post("/api/projects",
                                json={"name": name, "topic": "近场信道估计"}).json()

    def test_project_crud(self):
        p = self._project()
        self.assertEqual(p["name"], "XL-MIMO 近场")
        listed = self.client.get("/api/projects").json()["projects"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["note_count"], 0)
        self.assertEqual(self.client.delete(f"/api/projects/{p['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/projects/{p['id']}").status_code, 404)

    def test_empty_project_name_is_400(self):
        self.assertEqual(
            self.client.post("/api/projects", json={"name": "   "}).status_code, 400)

    def test_note_crud_and_get(self):
        p = self._project()
        n = self.client.post(f"/api/projects/{p['id']}/notes",
                             json={"title": "一个洞见", "content": "正文",
                                   "note_type": "insight", "tags": ["a"],
                                   "paper_id": self.p1}).json()
        got = self.client.get(f"/api/notes/{n['id']}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["content"], "正文")
        self.assertEqual(got.json()["paper_id"], self.p1)
        patched = self.client.patch(f"/api/notes/{n['id']}", json={"content": "改过"})
        self.assertEqual(patched.json()["content"], "改过")
        self.assertEqual(self.client.delete(f"/api/notes/{n['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/notes/{n['id']}").status_code, 404)

    def test_notes_of_missing_project_is_404(self):
        self.assertEqual(self.client.get("/api/projects/999/notes").status_code, 404)

    def test_document_crud_and_versioning(self):
        p = self._project()
        d = self.client.post(f"/api/projects/{p['id']}/documents",
                             json={"title": "初稿", "content": "", "status": "draft"}).json()
        self.assertEqual(d["version"], 1)
        updated = self.client.patch(f"/api/documents/{d['id']}",
                                    json={"content": "写了点东西"}).json()
        self.assertEqual(updated["version"], 2)      # 每次保存版本 +1
        self.assertEqual(updated["content"], "写了点东西")
        self.assertEqual(self.client.get(f"/api/documents/{d['id']}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/documents/{d['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/documents/{d['id']}").status_code, 404)

    def test_attach_source_to_document(self):
        p = self._project()
        d = self.client.post(f"/api/projects/{p['id']}/documents",
                             json={"title": "初稿"}).json()
        r = self.client.post(f"/api/documents/{d['id']}/sources",
                             json={"paper_id": self.p1, "page_no": 3,
                                   "quote": "近场信道估计", "relation": "support"})
        self.assertEqual(r.status_code, 200)
        doc = self.client.get(f"/api/documents/{d['id']}").json()
        self.assertEqual(doc["sources"][0]["paper_id"], self.p1)
        self.assertEqual(doc["sources"][0]["page_no"], 3)

    def test_write_offline_returns_template_not_error(self):
        p = self._project()
        d = self.client.post(f"/api/projects/{p['id']}/documents",
                             json={"title": "初稿"}).json()
        r = self.client.post(f"/api/documents/{d['id']}/write",
                             json={"instruction": "列个大纲", "mode": "outline",
                                   "paper_ids": [], "save": True})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["model"], "offline-template")
        self.assertIn("未配置 LLM", body["degraded"])   # 如实说明降级，而不是假装生成了


class ChatSessionEndpointTests(ApiTestBase):
    def test_session_listing_and_404(self):
        self.assertEqual(self.client.get("/api/chat/sessions").json()["sessions"], [])
        self.assertEqual(
            self.client.get("/api/chat/sessions/nope").status_code, 404)
        self.assertEqual(
            self.client.delete("/api/chat/sessions/nope").status_code, 404)

    def test_session_created_by_chat_is_listed(self):
        from papernest import chat
        sid = chat.ensure_session(None, "第一个问题")
        chat.append(sid, "user", "第一个问题")
        listed = self.client.get("/api/chat/sessions").json()["sessions"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], sid)
        self.assertEqual(listed[0]["n"], 1)
        detail = self.client.get(f"/api/chat/sessions/{sid}").json()
        self.assertEqual(detail["messages"][0]["content"], "第一个问题")
        self.assertEqual(self.client.delete(f"/api/chat/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client.get("/api/chat/sessions").json()["sessions"], [])


def _tiny_pdf(title: str = "A Deterministic Test Paper", doi: str = "10.1234/testdoi") -> bytes:
    """程序化造一份带标题/DOI/摘要的 PDF，不依赖任何外部样本文件。"""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 96), title, fontsize=20)
    page.insert_text((72, 130), "Alice Smith, Bob Lee", fontsize=10)
    page.insert_text((72, 160), f"doi:{doi}", fontsize=9)
    page.insert_text((72, 200), "Abstract", fontsize=11)
    page.insert_text((72, 220), "We evaluate deterministic pipelines end to end.", fontsize=10)
    page.insert_text((72, 260), "1 Introduction", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class LocalImportEndpointTests(ApiTestBase):
    """本地 PDF 上传与 BibTeX 导入——「个人科研文献 Agent」原来最缺的两条入口。"""

    def test_upload_pdf_creates_paper(self):
        r = self.client.post("/api/papers/upload",
                             files={"files": ("paper.pdf", _tiny_pdf(), "application/pdf")},
                             data={"make_card": "false"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["imported"], body["failed"]), (1, 0))
        p = body["papers"][0]
        self.assertIn("Deterministic Test Paper", p["title"])
        self.assertEqual(p["norm_key"], "doi:10.1234/testdoi")
        self.assertGreaterEqual(p["pages_stored"], 1)
        self.assertFalse(p["duplicate"])

    def test_uploading_same_bytes_twice_is_deduped(self):
        data = _tiny_pdf()
        first = self.client.post("/api/papers/upload",
                                 files={"files": ("a.pdf", data, "application/pdf")}).json()
        second = self.client.post("/api/papers/upload",
                                  files={"files": ("renamed.pdf", data, "application/pdf")}).json()
        self.assertTrue(second["papers"][0]["duplicate"])
        self.assertEqual(second["papers"][0]["paper_id"], first["papers"][0]["paper_id"])
        self.assertEqual(self.client.get("/api/papers").json()["count"], 2)  # 原有 1 篇 + 新 1 篇

    def test_non_pdf_is_rejected_without_500(self):
        r = self.client.post("/api/papers/upload",
                             files={"files": ("evil.pdf", b"not a pdf at all",
                                              "application/pdf")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["imported"], 0)
        self.assertEqual(r.json()["failed"], 1)

    def test_one_bad_file_does_not_sink_the_batch(self):
        r = self.client.post("/api/papers/upload", files=[
            ("files", ("good.pdf", _tiny_pdf("Good Paper Title", "10.1/good"),
                       "application/pdf")),
            ("files", ("bad.pdf", b"garbage", "application/pdf")),
        ]).json()
        self.assertEqual((r["imported"], r["failed"]), (1, 1))

    def test_traversal_filename_does_not_escape_uploads_dir(self):
        from papernest import pdfimport
        r = self.client.post("/api/papers/upload", files={
            "files": ("../../evil.pdf", _tiny_pdf("Traversal Probe", "10.1/trav"),
                      "application/pdf")}).json()
        stored = Path(r["papers"][0]["pdf_path"]).resolve()
        self.assertTrue(stored.is_relative_to(pdfimport.uploads_dir().resolve()))

    def test_fix_metadata_recomputes_norm_key(self):
        up = self.client.post("/api/papers/upload", files={
            "files": ("x.pdf", _tiny_pdf("Wrong Title Here", "10.1/fixme"),
                      "application/pdf")}).json()
        pid = up["papers"][0]["paper_id"]
        r = self.client.patch(f"/api/paper/{pid}/metadata",
                              json={"title": "Corrected Title", "year": 2021})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["title"], "Corrected Title")
        # 改后的标题必须立刻能被库内检索命中（papers_fts 同步刷新）
        found = self.client.get("/api/papers?q=Corrected").json()
        self.assertIn("Corrected Title", [p["title"] for p in found["papers"]])

    def test_fix_metadata_missing_paper_is_404(self):
        self.assertEqual(
            self.client.patch("/api/paper/999999/metadata",
                              json={"title": "x"}).status_code, 404)

    def test_fix_metadata_norm_key_clash_is_409(self):
        """撞主键要报冲突让人来决定，不能静默合并两条记录。"""
        r = self.client.patch(f"/api/paper/{self.p1}/metadata",
                              json={"doi": "10.1234/testdoi"})
        self.assertEqual(r.status_code, 200)      # 库里还没有这个 key，可以改
        up = self.client.post("/api/papers/upload", files={
            "files": ("y.pdf", _tiny_pdf("Another", "10.1234/testdoi"),
                      "application/pdf")}).json()
        # 上传的那篇 norm_key 与 p1 撞了 → 走 norm_key 去重，不是新建
        self.assertTrue(up["papers"][0]["duplicate"])

    def test_import_bibtex_text(self):
        bib = """@article{smith2024agent,
  title = {A Study of {LLM} Agents},
  author = {Smith, Alice and Lee, Bob},
  journal = {Journal of Tests},
  year = {2024},
  doi = {10.5555/agent2024}
}"""
        r = self.client.post("/api/import/bibliography",
                             json={"text": bib, "fmt": "auto"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["format"], "bibtex")
        self.assertEqual(body["imported"], 1)
        found = self.client.get("/api/papers?q=Agents").json()["papers"]
        self.assertIn("A Study of LLM Agents", [p["title"] for p in found])

    def test_reimporting_same_file_skips_everything(self):
        bib = "@misc{k, title={Repeat Import Probe}, author={Ann Ray}, year={2020}}"
        first = self.client.post("/api/import/bibliography", json={"text": bib}).json()
        second = self.client.post("/api/import/bibliography", json={"text": bib}).json()
        self.assertEqual(first["imported"], 1)
        self.assertEqual((second["imported"], second["skipped"]), (0, 1))

    def test_import_file_upload(self):
        ris = "TY  - JOUR\nTI  - RIS Upload Probe\nAU  - Ray, Ann\nPY  - 2019\nER  - \n"
        r = self.client.post("/api/import/bibliography/file",
                             files={"file": ("lib.ris", ris.encode(), "text/plain")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["format"], "ris")
        self.assertEqual(r.json()["imported"], 1)

    def test_unknown_format_is_400(self):
        r = self.client.post("/api/import/bibliography",
                             json={"text": "whatever", "fmt": "nope"})
        self.assertEqual(r.status_code, 400)


class PdfServingTests(ApiTestBase):
    """内嵌阅读器的取文件端点——它读的是 DB 里一列普通 TEXT，必须做归属校验。"""

    def test_serves_uploaded_pdf(self):
        up = self.client.post("/api/papers/upload", files={
            "files": ("v.pdf", _tiny_pdf("Viewer Probe", "10.1/view"),
                      "application/pdf")}).json()
        pid = up["papers"][0]["paper_id"]
        r = self.client.get(f"/api/paper/{pid}/pdf")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/pdf")
        self.assertTrue(r.content.startswith(b"%PDF-"))

    def test_paper_without_pdf_is_404(self):
        with db.conn() as c:
            c.execute("UPDATE papers SET pdf_path=NULL WHERE id=?", (self.p1,))
        self.assertEqual(self.client.get(f"/api/paper/{self.p1}/pdf").status_code, 404)

    def test_path_outside_data_dir_is_refused(self):
        """setUp 里 p1 的 pdf_path 就是库外的绝对路径——不能因为它写在库里就照发。"""
        r = self.client.get(f"/api/paper/{self.p1}/pdf")
        self.assertEqual(r.status_code, 404)

    def test_traversal_in_stored_path_is_refused(self):
        with db.conn() as c:
            c.execute("UPDATE papers SET pdf_path=? WHERE id=?",
                      (str(Path(config.DATA_DIR) / ".." / ".." / "windows" / "win.ini"),
                       self.p1))
        self.assertEqual(self.client.get(f"/api/paper/{self.p1}/pdf").status_code, 404)

    def test_missing_paper_is_404(self):
        self.assertEqual(self.client.get("/api/paper/999999/pdf").status_code, 404)

    # ── 按页渲染（证据跳页真正依赖的那条路）──

    def _uploaded(self):
        up = self.client.post("/api/papers/upload", files={
            "files": ("r.pdf", _tiny_pdf("Render Probe", "10.1/render"),
                      "application/pdf")}).json()
        return up["papers"][0]["paper_id"]

    def test_page_image_renders_png(self):
        pid = self._uploaded()
        r = self.client.get(f"/api/paper/{pid}/page/1.png")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")
        self.assertTrue(r.content.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_page_out_of_range_is_404_not_500(self):
        pid = self._uploaded()
        self.assertEqual(self.client.get(f"/api/paper/{pid}/page/99.png").status_code, 404)
        self.assertEqual(self.client.get(f"/api/paper/{pid}/page/0.png").status_code, 404)

    def test_dpi_is_clamped(self):
        """dpi 是外部可控参数，不能让它把渲染撑成几百 MB。"""
        pid = self._uploaded()
        big = self.client.get(f"/api/paper/{pid}/page/1.png?dpi=100000")
        self.assertEqual(big.status_code, 200)
        self.assertLess(len(big.content), 8 * 1024 * 1024)

    def test_page_image_respects_path_ownership(self):
        with db.conn() as c:
            c.execute("UPDATE papers SET pdf_path=? WHERE id=?",
                      (r"C:\windows\win.ini", self.p1))
        self.assertEqual(self.client.get(f"/api/paper/{self.p1}/page/1.png").status_code, 404)


class GraphEndpointTests(ApiTestBase):
    def _edge(self, src_key, dst_key):
        from papernest import graph
        graph.ensure_schema()
        with db.conn() as c:
            c.execute("INSERT OR IGNORE INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                      (src_key, dst_key))
            c.execute("""INSERT OR IGNORE INTO citation_nodes(norm_key,title,year)
                         VALUES(?,?,?)""", (dst_key, f"外部论文 {dst_key}", 2020))

    def test_stats_and_gaps_are_empty_before_any_edges(self):
        self.assertEqual(self.client.get("/api/graph/stats").json()["edges"], 0)
        self.assertEqual(self.client.get("/api/graph/gaps").json()["gaps"], [])

    def test_gap_papers_surface_uncollected_references(self):
        with db.conn() as c:
            p2 = db.insert_l0(c, {"norm_key": "doi:10.9/two", "title": "第二篇",
                                  "abstract": "x", "year": 2023, "venue": "V",
                                  "authors": [], "doi": "10.9/two", "arxiv_id": None,
                                  "source": "s2"})
        self._edge("arxiv:6001", "doi:10.9/gap")      # self.p1 的 norm_key
        self._edge("doi:10.9/two", "doi:10.9/gap")    # 两篇库内论文都引了它
        self._edge("doi:10.9/outsider", "doi:10.9/gap")   # 库外论文引的不算数
        gaps = self.client.get("/api/graph/gaps").json()["gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["norm_key"], "doi:10.9/gap")
        self.assertEqual(gaps[0]["cited_by_count"], 2)   # 只数库内施引者
        self.assertIsNotNone(p2)

    def test_adopt_puts_external_node_into_library(self):
        self._edge("doi:10.1000/abc", "doi:10.9/adoptme")
        r = self.client.post("/api/graph/adopt?norm_key=doi:10.9/adoptme")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "adopted")
        detail = self.client.get(f"/api/paper/{r.json()['paper_id']}").json()
        self.assertEqual(detail["level"], 0)          # 引文端点没有摘要，如实存 L0
        self.assertEqual(self.client.get("/api/graph/gaps").json()["gaps"], [])

    def test_adopt_unknown_key_is_404(self):
        self.assertEqual(
            self.client.post("/api/graph/adopt?norm_key=doi:10.9/nope").status_code, 404)

    def test_edges_for_paper_without_external_id_is_400(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {"norm_key": "title:abc", "title": "只有标题的论文",
                                   "abstract": None, "year": None, "venue": None,
                                   "authors": [], "doi": None, "arxiv_id": None,
                                   "source": "upload"})
        r = self.client.post(f"/api/paper/{pid}/edges")
        self.assertEqual(r.status_code, 400)          # 不拿标题去瞎搜一个可能不是它的 id


class DeepSearchEndpointTests(ApiTestBase):
    def test_deep_search_never_returns_fewer_than_single_round(self):
        r = self.client.post("/api/search/deep",
                             json={"question": "channel estimation", "top_k": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("prf", body["mode"])
        self.assertIn(self.p1, body["paper_ids"])
        self.assertEqual(body["trace"]["fusion"], "append")

    def test_deep_answer_requires_llm(self):
        """离线时如实报 400，而不是排一个注定失败的任务。"""
        r = self.client.post("/api/answer/deep",
                             json={"question": "近场信道估计的方法有哪些"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("未配置 LLM", r.json()["detail"])

    def test_deep_answer_queues_job_when_llm_available(self):
        with mock.patch("papernest.llm.available", return_value=True):
            r = self.client.post("/api/answer/deep",
                                 json={"question": "近场信道估计", "max_rounds": 2})
        self.assertEqual(r.status_code, 200)
        job = self.client.get(f"/api/jobs/{r.json()['job_id']}").json()
        self.assertEqual(job["kind"], "deep_answer")
        self.assertEqual(job["params"]["max_rounds"], 2)


class AuthMiddlewareTests(ApiTestBase):
    def test_no_key_configured_means_open(self):
        self.assertEqual(self.client.get("/api/stats").status_code, 200)

    def test_key_configured_blocks_unauthenticated_api_calls(self):
        with mock.patch.object(api, "API_KEY", "s3cret"):
            self.assertEqual(self.client.get("/api/stats").status_code, 401)
            self.assertEqual(
                self.client.get("/api/stats", headers={"x-api-key": "wrong"}).status_code,
                401)
            self.assertEqual(
                self.client.get("/api/stats", headers={"x-api-key": "s3cret"}).status_code,
                200)

    def test_key_does_not_block_the_web_page(self):
        with mock.patch.object(api, "API_KEY", "s3cret"):
            self.assertEqual(self.client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
