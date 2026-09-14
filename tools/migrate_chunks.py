#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 2026-09-07 那批切块修复补到**存量库**上。默认只报账，不动任何东西。

代码修完只对**新导入**的 PDF 生效；库里已有的 chunk 是旧代码切的，要享受修复必须
重切。这个脚本把三条修复的存量迁移合成一次，并且**只重切真的需要重切的论文**：

  ① 参考文献块（`kind='reference'`）—— 它们本不该进 chunks_fts / 向量 / 回答上下文。
     真库 13.9% 的可检索文本是他人的文献条目，64 条真实查询里 16 条的块级 top-5
     含参考文献块（最坏一条 5/5）。
  ② 漏检子标题造成的错误 `section_path` —— 引用带的章节号是错的出处。
  ③ 超过 `db.MAX_CHUNK_CHARS` 的块 —— 块尾在嵌入时被静默截掉，不进向量。

**为什么不能直接跑 `cli.py rechunk`**：它无差别重切全部有 PDF 的论文，
`db.replace_chunks` 会把这些论文的 chunk 向量**全部删掉**（重切后 chunk_no 的含义变了，
留着会「用旧向量匹配、回取新文本」），于是即使切出来一模一样也要全部重嵌。
真库实测那是约 4.9 倍的无谓花销。这个脚本逐篇比对新旧切分，**逐字相同的一律不动**。

用法（仓库根目录）：

    python tools/migrate_chunks.py                  # 只报账：会动哪些论文、要重嵌多少
    python tools/migrate_chunks.py --db <副本路径>   # 先在库副本上跑一遍（**建议**）
    python tools/migrate_chunks.py --apply          # 真改（会二次确认）

`--apply` 之后还要跑 `python cli.py vectors embed-chunks` 才会把新块嵌进去；
在那之前这些论文的 chunk 检索会退化（向量少了那几块），**不是**静默——
`cli.py health` 的向量覆盖率闸门会报出来。

⚠️ 记两条本仓踩过的：迁移**先在库副本上跑**（「消融实验要在库副本上做」，
之前差点丢 784 条向量）；`embeddings._paper_matrix()` 是进程内缓存，
改完要重启 API 进程，否则已删的向量还会继续参与检索。
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from papernest import config, db  # noqa: E402

_WS = re.compile(r"\s+")


def _n(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def _new_chunks(c: sqlite3.Connection, pid: int, pdf_path: str | None) -> list[dict]:
    """按**当前代码**重新切这一篇。有 PDF 走 structure，没有走 pages 兜底。"""
    if pdf_path and Path(pdf_path).exists():
        from papernest import fulltext
        return fulltext.build_chunks(pdf_path, c, pid)
    return db.chunks_from_pages(c, pid)


def plan(conn: sqlite3.Connection, only=None) -> list[dict]:
    """逐篇比对新旧切分，返回需要动的论文。`only` 给定时只看这几篇。"""
    rows = conn.execute(
        "SELECT DISTINCT p.id, p.title, p.pdf_path FROM papers p "
        "JOIN chunks ch ON ch.paper_id = p.id ORDER BY p.id").fetchall()
    todo = []
    for r in rows:
        pid = r["id"]
        if only is not None and pid not in only:
            continue
        old = conn.execute(
            "SELECT chunk_no, kind, text FROM chunks WHERE paper_id=? ORDER BY chunk_no",
            (pid,)).fetchall()
        try:
            new = _new_chunks(conn, pid, r["pdf_path"])
        except Exception as e:                                    # noqa: BLE001
            todo.append({"pid": pid, "title": r["title"], "why": f"切块失败：{e}",
                         "mode": "error", "old_n": len(old), "new_n": 0,
                         "reembed_chars": 0})
            continue

        old_txt = [_n(x["text"]) for x in old]
        new_txt = [_n(x.get("text") or "") for x in new]
        old_kind = [x["kind"] or "text" for x in old]
        new_kind = [(x.get("kind") or "text") for x in new]

        if old_txt == new_txt and old_kind == new_kind:
            continue                                             # 一模一样，不动

        # 文本序列逐位相同、只有 kind 变了 → 原地打标签，**零重嵌**
        if old_txt == new_txt:
            flipped = [old[i]["chunk_no"] for i in range(len(old))
                       if old_kind[i] != new_kind[i]]
            todo.append({
                "pid": pid, "title": r["title"], "mode": "relabel",
                "why": f"{len(flipped)} 块改 kind（多为参考文献）", "nos": flipped,
                "kinds": {old[i]["chunk_no"]: new_kind[i] for i in range(len(old))
                          if old_kind[i] != new_kind[i]},
                "old_n": len(old), "new_n": len(new), "reembed_chars": 0})
            continue

        # 文本真的变了 → 必须重切，该篇 chunk 向量全部作废、需重嵌非参考文献块
        reembed = sum(len(x.get("text") or "") for x in new
                      if (x.get("kind") or "text") != "reference")
        todo.append({
            "pid": pid, "title": r["title"], "mode": "rechunk",
            "why": f"切分变了 {len(old)} → {len(new)} 块",
            "old_n": len(old), "new_n": len(new), "reembed_chars": reembed})
    return todo


def report(todo: list[dict], conn: sqlite3.Connection) -> None:
    relabel = [t for t in todo if t["mode"] == "relabel"]
    rechunk = [t for t in todo if t["mode"] == "rechunk"]
    errors = [t for t in todo if t["mode"] == "error"]

    print(f"需要动的论文：{len(todo)} 篇"
          f"（原地打标签 {len(relabel)} · 重切 {len(rechunk)} · 切块失败 {len(errors)}）")
    if relabel:
        n = sum(len(t["nos"]) for t in relabel)
        print(f"\n【原地打标签】{len(relabel)} 篇 / {n} 块 —— **零重嵌**，"
              f"只改 kind、摘出 chunks_fts、删掉这些块的向量")
        for t in relabel[:8]:
            print(f"    paper {t['pid']:<4} {t['why']:<28} {(t['title'] or '')[:44]}")
        if len(relabel) > 8:
            print(f"    …另外 {len(relabel) - 8} 篇")
    if rechunk:
        chars = sum(t["reembed_chars"] for t in rechunk)
        print(f"\n【必须重切】{len(rechunk)} 篇 —— chunk 向量会被作废，需要重嵌")
        for t in rechunk:
            print(f"    paper {t['pid']:<4} {t['why']:<28} {(t['title'] or '')[:44]}")
        # 与 chunkembed.estimate 同口径（chars/4）
        print(f"\n  重嵌规模：{chars} 字符 ≈ {chars // 4} token")
        drop = conn.execute(
            "SELECT COUNT(*) n FROM vectors WHERE kind='chunk' AND paper_id IN ({})"
            .format(",".join(str(t["pid"]) for t in rechunk))).fetchone()["n"]
        print(f"  会被作废的 chunk 向量：{drop} 条")
    if errors:
        print(f"\n【切块失败】{len(errors)} 篇（多半是 pdf_path 指向的文件不在了）")
        for t in errors:
            print(f"    paper {t['pid']}: {t['why']}")
    print("\n对照：`cli.py rechunk` 会无差别重切全部有 PDF 的论文并清掉它们的全部 "
          "chunk 向量——上面这份计划就是为了避开那笔无谓花销。")


def apply(conn: sqlite3.Connection, todo: list[dict]) -> None:
    for t in todo:
        pid = t["pid"]
        if t["mode"] == "relabel":
            for no, kind in t["kinds"].items():
                conn.execute("UPDATE chunks SET kind=? WHERE paper_id=? AND chunk_no=?",
                             (kind, pid, no))
                if kind == "reference":
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE paper_id=? AND chunk_no=?", (pid, no))
                    conn.execute(
                        "DELETE FROM vectors WHERE paper_id=? AND kind='chunk' AND idx=?",
                        (pid, no))
            print(f"  paper {pid}: 打标签 {len(t['kinds'])} 块")
        elif t["mode"] == "rechunk":
            r = conn.execute("SELECT pdf_path FROM papers WHERE id=?", (pid,)).fetchone()
            db.replace_chunks(conn, pid, _new_chunks(conn, pid, r["pdf_path"]))
            print(f"  paper {pid}: 重切 {t['old_n']} → {t['new_n']} 块（向量已作废）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="改用这个库（**建议先拿副本跑一遍**）")
    ap.add_argument("--apply", action="store_true", help="真改（默认只报账）")
    ap.add_argument("--papers", default="",
                    help="只处理这几篇（逗号或空格分隔的 id）。表格摘要生效只需要"
                         "重切有表的那 20 来篇，全库重切是白花钱。")
    args = ap.parse_args()
    only = {int(x) for x in args.papers.replace(",", " ").split()} or None

    if args.db:
        config.DB_PATH = Path(args.db)
    print(f"库：{config.DB_PATH}")
    if not config.DB_PATH.exists():
        print("库不存在"); return 1

    with db.conn() as conn:
        todo = plan(conn, only)
        if not todo:
            print("没有需要迁移的论文——当前切分与代码一致。")
            return 0
        report(todo, conn)
        if not args.apply:
            print("\n（这是 dry-run。确认账单后加 --apply 才会真改。）")
            return 0
        if input("\n确认对上面这些论文执行迁移？输入 yes 继续：").strip() != "yes":
            print("已取消"); return 1
        apply(conn, todo)

    print("\n迁移完成。还要做两件事：")
    print("  1) python cli.py vectors embed-chunks   # 把新块嵌进去（在那之前检索会退化）")
    print("  2) 重启 API 进程                        # _paper_matrix 是进程内缓存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
