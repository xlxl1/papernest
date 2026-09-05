import gc
import shutil
import tempfile
import unittest
from pathlib import Path

from papernest import config, db, jobs


class JobsLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_test_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp) / "test.db"
        db.init_db()

    def tearDown(self):
        # sqlite3 的连接由 `with conn()` 提交但不关闭，Windows 下文件句柄未放
        # 需先触发 GC 回收连接，再删临时目录（忽略残留删除失败）
        config.DB_PATH = self._old_db
        gc.collect()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_create_progress_finish_roundtrip(self):
        job_id = jobs.create_job("ingest", {"query": "agent eval"})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["params"], {"query": "agent eval"})

        jobs.set_progress(job_id, 0.4, "ingest", "处理 2/5")
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["progress"], 0.4)
        self.assertEqual(job["stage"], "ingest")

        jobs.finish_job(job_id, {"new_papers": 5})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"], {"new_papers": 5})
        self.assertEqual(job["progress"], 1.0)

    def test_fail_records_error(self):
        job_id = jobs.create_job("read", {"paper_id": 1})
        jobs.fail_job(job_id, "RuntimeError: 爆了")
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("爆了", job["error"])

    def test_progress_is_clamped(self):
        job_id = jobs.create_job("survey", {"topic": "t"})
        jobs.set_progress(job_id, 7.5, "s", "m")
        self.assertEqual(jobs.get_job(job_id)["progress"], 1.0)

    def test_unknown_job_returns_none(self):
        self.assertIsNone(jobs.get_job("nope"))


if __name__ == "__main__":
    unittest.main()
