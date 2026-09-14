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

from papernest import config, db

# ── 生产库写保护 ────────────────────────────────────────────────────────────
#
# `db.init_db()` 是**写路径**：它跑 `executescript(SCHEMA)` + `_migrate()`。
# `config.DB_PATH` 默认指向用户真实的 `data/papernest.db`（本机 80MB、503 篇），
# 所以任何一条忘记重定向库的用例，跑一次测试就顺带给生产库做了一次迁移。
#
# 这已经发生过两次，形态完全不同：
#   ① `test_eval.py` 直接调 init_db 校验 gold 键——把真库 user_version 5→6 迁了。
#   ② 2026-09-09：`test_fix_regressions` 带对口令请求 `/api/stats`，而那个端点
#      第一行就是 `db.init_db()`，于是新加的 `table_summaries` 被建到了真库里。
#
# ② 光读代码发现不了：用例里没有 `init_db` 三个字，写入发生在 TestClient 的
# 线程池 worker 里，栈上连一帧 `tests/` 都没有——查它花了三轮全量扫描。
# 指望「写 TestClient 用例时记得重定向库」靠不住，所以在这里一次性堵死。
#
# **装在 support.py 而不是 tests/__init__.py**：README 的跑法是
# `discover -s tests -t tests`，那种模式下 `tests/` 不当包用，`__init__.py`
# 根本不会被导入（实测过）。而 support.py 被二十来个测试文件在**模块导入期**
# 引入，discover 又是先导入全部模块再跑用例，所以装在这里对整轮都生效。
#
# 只拦写，不拦读：有几条用例**有意**读真库（`test_eval.py` 的 gold 键校验、
# `test_agent.py` 的全量路由校验、`test_eval_set_composition.py` 的成分统计），
# 它们都是 `sqlite3.connect(..., mode=ro)` 直接开的，不走 init_db，不受影响。
# 读真库是这个仓库的一条纪律（合成夹具掩盖过真实数据缺陷），要禁的只有写。

#: 生产库的绝对路径，在任何用例改动 `config.DB_PATH` 之前记下来。
PRODUCTION_DB = config.DB_PATH.resolve()


def _install_production_db_guard():
    """幂等安装。多个测试模块都 import 本文件，不能装两层。"""
    if getattr(db.init_db, "_papernest_prod_guard", False):
        return
    real = db.init_db

    def guarded(force: bool = False):
        if config.DB_PATH.resolve() == PRODUCTION_DB:
            raise RuntimeError("\n".join([
                f"测试试图在**生产库**上执行 init_db（{PRODUCTION_DB}）。",
                "init_db 是写路径（executescript(SCHEMA) + _migrate()），"
                "跑一次测试就等于给用户的库做了一次迁移。",
                "改法：让用例继承 tests.support.TempDbTestCase，或在建 TestClient / "
                "调任何库函数之前把 config.DATA_DIR / config.DB_PATH 指到临时目录"
                "（make_tempdir 就是干这个的）。",
                "注意：用 TestClient 打任何一个端点都可能间接触发它"
                "（`/api/stats` 第一行就是 db.init_db()），而那发生在线程池里，"
                "栈上看不到你的用例。"]))
        return real(force)

    guarded._papernest_prod_guard = True
    db.init_db = guarded


_install_production_db_guard()


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
