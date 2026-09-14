# -*- coding: utf-8 -*-
"""跑一次测试，不许动用户的生产库。

`db.init_db()` 是写路径：`executescript(SCHEMA)` + `_migrate()`。而
`config.DB_PATH` 默认就指着用户真实的 `data/papernest.db`（本机 80MB、503 篇论文、
4816 条向量）。所以**任何一条忘记重定向库的用例，跑一次测试就顺带给生产库做了
一次迁移**——不报错、不留痕、跑完才发现库变了。

这已经发生过两次，形态完全不同：

① `tests/test_eval.py` 直接调 `db.init_db()` 校验 gold 键。
   实测把真库 `user_version` 从 5 迁到了 6。改成只读打开修掉了。

② 2026-09-09：`test_fix_regressions::test_ascii_key_passes_and_authenticates`
   带对口令请求 `/api/stats`，而那个端点第一行就是 `db.init_db()`。
   于是当天新加的 `table_summaries` 表被建到了真库里。

**②比①危险得多，因为它看不见**：用例源码里根本没有 `init_db` 三个字，
写入发生在 TestClient 的线程池 worker 里，异常栈上连一帧 `tests/` 都没有。
第一轮排查我按「栈里有没有 tests/ 帧」过滤，把它整个漏掉了，
得出「没有任何用例在真库上调 init_db」的错误结论；跑了三轮全量扫描才定位到。

所以不能靠「写 TestClient 用例时记得重定向库」这种约定——`tests/support.py`
在导入期装了一道闸：`config.DB_PATH` 指向生产库时 `init_db` 直接抛。

**为什么装在 support.py 而不是 tests/__init__.py**：README 的跑法是
`discover -s tests -t tests`，那种模式下 `tests/` 不当包用，`__init__.py` 压根不会
被导入（实测过，先写在那里、发现是死代码才搬的）。support.py 被二十来个测试文件
在模块导入期引入，而 discover 是先导入全部模块再跑用例，所以装在这里对整轮生效。

**只拦写，不拦读**：有几条用例**有意**读真库（`test_eval` 的 gold 键校验、
`test_agent` 的全量路由校验、`test_eval_set_composition` 的成分统计），
它们都是 `sqlite3.connect(..., mode=ro)` 直接开的，不受影响。拿真库跑是这个仓库的
一条纪律（合成夹具掩盖过真实数据缺陷），要禁的只有写。
"""
import re
import unittest
from pathlib import Path

from papernest import config, db

try:
    from .support import PRODUCTION_DB, make_tempdir
except ImportError:                     # discover -s tests 时 tests/ 不当包用
    from support import PRODUCTION_DB, make_tempdir

TESTS_DIR = Path(__file__).resolve().parent


class TheGuardIsInstalledTests(unittest.TestCase):
    def test_init_db_is_wrapped(self):
        """导入 support 就该装上——它不是需要各用例主动调用的东西。"""
        self.assertTrue(getattr(db.init_db, "_papernest_prod_guard", False),
                        "生产库写保护没装上；tests/support.py 被改动过？")

    def test_it_refuses_the_production_path(self):
        old = config.DB_PATH
        try:
            config.DB_PATH = PRODUCTION_DB
            with self.assertRaises(RuntimeError) as cm:
                db.init_db()
        finally:
            config.DB_PATH = old
        msg = str(cm.exception)
        self.assertIn("生产库", msg)
        self.assertIn("TempDbTestCase", msg, "报错没告诉人该怎么改，等于只是把人卡住")

    def test_it_lets_a_temp_db_through(self):
        """闸门只拦生产库那一个路径——临时库照常建，否则全套都跑不起来。"""
        tmp = make_tempdir(self, "papernest_guardcheck")
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        self.addCleanup(lambda: (setattr(config, "DB_PATH", old_db),
                                 setattr(config, "DATA_DIR", old_dir)))
        config.DATA_DIR = tmp / "data"
        config.DB_PATH = config.DATA_DIR / "t.db"
        db.init_db()                                    # 不抛就是通过
        self.assertTrue(config.DB_PATH.exists())

    def test_installing_twice_does_not_stack(self):
        """二十来个模块都 import support，装两层的话报错信息会套娃。"""
        try:
            from .support import _install_production_db_guard as install
        except ImportError:
            from support import _install_production_db_guard as install
        before = db.init_db
        install()
        self.assertIs(db.init_db, before)


class NoTestClientEscapesTheTempDbTests(unittest.TestCase):
    """结构闸：建 TestClient 的文件必须自己把库指走。

    这条和上面那道运行时闸是互补的——运行时闸只在端点**真的**调了 init_db 时才响，
    而有些端点今天不调、明天加一行就调了。这条在源码层先拦一道。
    """

    def test_every_testclient_file_redirects_the_db(self):
        offenders = []
        for f in sorted(TESTS_DIR.glob("test_*.py")):
            src = f.read_text(encoding="utf-8", errors="replace")
            if "TestClient(" not in src:
                continue
            redirects = ("config.DB_PATH =" in src
                         or "config, \"DB_PATH\"" in src
                         or "TempDbTestCase" in src
                         or "make_tempdir" in src)
            if not redirects:
                offenders.append(f.name)
        self.assertEqual(
            offenders, [],
            f"这些文件建了 TestClient 却没把 config.DB_PATH 指到临时库：{offenders}。"
            f"打任何一个端点都可能间接触发 db.init_db()（`/api/stats` 第一行就是），"
            f"那会对用户的生产库执行 executescript(SCHEMA) + _migrate()。")


if __name__ == "__main__":
    unittest.main()
