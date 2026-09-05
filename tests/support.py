"""测试公共支撑：临时目录、库隔离、外部依赖摁断。

**为什么有这个文件**：26 个测试文件里 21 个用 `tempfile.mkdtemp()` 建临时库却从不删。
单看每个用例只泄漏几百 KB，但 885 条测试跑上几十遍之后，实测宿主机 `%TEMP%` 下
攒了 **40,513 个目录、9.29 GB**。这类问题不会让任何一条用例变红，只会悄悄吃硬盘——
正是最该由基类兜住、而不是指望每个人记得写 tearDown 的东西。

用法：

    from .support import TempDbTestCase

    class MyTests(TempDbTestCase):
        prefix = "papernest_mything"      # 可选，默认 papernest_test

        def setUp(self):
            super().setUp()              # 库已建好、向量与 LLM 已摁断
            ...

`super().setUp()` 之后即可用：
- `self.tmp`      —— 临时目录 Path，退出时自动删
- `config.DB_PATH`/`DATA_DIR` 已指向临时目录，退出时还原
- `embeddings.available()` 与 `llm.available()` 已 patch 成 False
  （开发机 `.env` 是真 key，不摁断测试会真发请求——又慢又花钱）
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config


def make_tempdir(case: unittest.TestCase, prefix: str = "papernest_test") -> Path:
    """建一个用例结束时自动删除的临时目录。

    Windows 上 SQLite/Chroma 的句柄有时还没释放，rmtree 会失败——
    这里吞掉异常（`ignore_errors=True`）：清理失败不该让一条本来通过的用例变红，
    但绝大多数目录能被回收，泄漏量从「无上限增长」降到「个位数残留」。
    """
    d = Path(tempfile.mkdtemp(prefix=prefix + "_"))
    case.addCleanup(shutil.rmtree, d, True)
    return d


class TempDbTestCase(unittest.TestCase):
    """临时库 + 外部依赖摁断的基类。子类的 setUp 记得先调 `super().setUp()`。"""

    prefix = "papernest_test"
    #: 子类置 False 可放行真实 LLM/向量（目前没有用例需要，留作显式逃生口）
    isolate_externals = True

    def setUp(self):
        super().setUp()
        self.tmp = make_tempdir(self, self.prefix)

        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        self.addCleanup(self._restore_config, old_db, old_dir)
        config.DATA_DIR = self.tmp / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"

        if self.isolate_externals:
            for target in ("papernest.embeddings.available",
                           "papernest.llm.available"):
                p = mock.patch(target, return_value=False)
                p.start()
                self.addCleanup(p.stop)

    @staticmethod
    def _restore_config(db_path, data_dir):
        config.DB_PATH, config.DATA_DIR = db_path, data_dir
