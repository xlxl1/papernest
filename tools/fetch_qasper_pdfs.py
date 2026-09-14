# -*- coding: utf-8 -*-
"""把 QASPER dev 的论文 PDF 抓下来，**给评测集扩样本**。

## 为什么这件事排在「改检索」前面

2026-09-09 一连试了几个检索改动，全都是「方向为正、但不显著」：

    候选集 IDF 加权     问题召回 0.2754 → 0.3333   p=0.219（6 题变化）
    per_paper 2 → 3     问题召回 0.2754 → 0.3333   p=0.343（10 题变化）

不是这些改动没用，是**判不出来**：评测集只有 69 道题，任何改动都只动 5~10 道，
过不了显著性。再调下去就是在噪声里挑参数——本仓已经因为这个撤回过结论
（32 道题里一道题就是 p=0.031）。

瓶颈在样本量，而样本量的瓶颈在 PDF：`chunk_sweep` 的任务 =
「本地有真 PDF」∩「QASPER 有 gold 证据」，本地只有 26 份 PDF，交出来 25 篇 / 69 题。

而 **QASPER dev 的键就是 arXiv ID**（`1912.01214` 这种），共 281 篇、
**927 道带 gold 证据的题**。把缺的那些 PDF 抓下来，样本量就是 13 倍——
上面那两个改动届时要么被证实，要么被证伪，不用再猜。

## 为什么必须用真 PDF，不能直接用 QASPER 自带的 full_text

因为要评的正是**我们自己的 PDF 抽取与切块**。用 QASPER 的结构化正文，
等于把待测环节换成了完美输入——那测的是别人的抽取质量。
（同样的理由见 `qasper.chunk_sweep` 的注释：「切法的差异只在真实版面上才显出来」。）

## 抓哪个版本：**v1 通常更对**

`arxiv.org/pdf/<id>` 给的是**最新版**，而 QASPER 是 2021 年前后从当时的快照建的。
论文改版之后，同一句话的措辞就变了——实测 1909.07575：

    gold : ...and a decoder. It is trained from scratch on the ST-TED corpus.
    最新版: ...and a decoder, which is trained from scratch with only the speech-translation data.

意思一样、字不一样，逐字匹配全灭。改抓 v1 之后：

    1909.11687   0.0% → 77.8%      2004.01853   0.0% → 66.7%
    1909.07575   9.5% → 52.4%      1912.13337  10.2% → 34.7%

（另有四篇没变化——只有一个版本，或差异在别处。）

所以默认 `--version v1`，并且**两版都留、按与 QASPER 正文的吻合度挑**
（`--pick-best`）：v1 也不保证就是 QASPER 用的那版，让数据说话比拍一个规则稳。

## 礼貌抓取

arXiv 允许批量下载，但要求限速。默认每次请求间隔 3 秒（`--delay`），
断点续传（已存在的文件跳过），失败不重试整批。默认 `--limit 5` 只抓 5 篇试水，
要全量抓得显式给数字。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papernest import config, qasper  # noqa: E402

#: arXiv 的批量下载礼仪：请求之间至少隔几秒。别改小。
DEFAULT_DELAY = 3.0

#: 存哪。跟已有的 PDF 分开放，免得和用户自己的文献混在一起。
DEST = config.ROOT / "data" / "qasper_pdf"

_UA = "PaperNest-eval/1.0 (offline retrieval evaluation; contact: local user)"

#: 已经定版的清单：`{arxiv_id: "v1" | "latest"}`。
#: **断点续传靠它，不靠「文件存不存在」**——换版本时必须 `--refetch`，
#: 而那会让「已存在就跳过」失效，中断后重跑就是从头再抓一遍
#: （279 篇 × 两版 ≈ 28 分钟全白费，还给 arXiv 白添一遍负担）。
MANIFEST = DEST / "_versions.json"


def load_manifest(dest: Path) -> dict:
    f = dest / MANIFEST.name
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def save_manifest(dest: Path, m: dict) -> None:
    try:
        (dest / MANIFEST.name).write_text(
            json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
    except OSError:
        pass                                # 记不下来只是下次多抓一遍，不该中断下载


def wanted() -> list[tuple[str, str, int]]:
    """QASPER dev 里**带 gold 证据**的论文：`[(arxiv_id, title, n_questions)]`。"""
    raw = json.loads(qasper.dev_path().read_text(encoding="utf-8"))
    out = []
    for aid, paper in raw.items():
        n = sum(1 for q in paper.get("qas", [])
                if any((a.get("answer") or {}).get("evidence")
                       for a in q.get("answers", [])))
        if n:
            out.append((aid, paper.get("title") or "", n))
    out.sort(key=lambda x: -x[2])          # 先抓 gold 题最多的，早停也划算
    return out


def already_have() -> set[str]:
    """已经下过的（本工具目录）+ 库里已有的同 arXiv id。"""
    have = {p.stem for p in DEST.glob("*.pdf")} if DEST.exists() else set()
    try:
        from papernest import db
        with db.conn() as c:
            for r in c.execute("SELECT arxiv_id FROM papers WHERE arxiv_id IS NOT NULL"):
                a = re.sub(r"v\d+$", "", (r["arxiv_id"] or "").strip())
                if a:
                    have.add(a)
    except Exception:                                   # noqa: BLE001
        pass                                # 库读不了不影响下载，只是可能多抓几份
    return have


def fetch_one(arxiv_id: str, dest: Path, timeout: int = 90,
              version: str = "") -> tuple[bool, str]:
    url = f"https://arxiv.org/pdf/{arxiv_id}{version}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception as exc:                            # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if not data.startswith(b"%PDF"):
        return False, f"不是 PDF（{len(data)} 字节，开头 {data[:16]!r}）"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, f"{len(data) / 1024 / 1024:.1f} MB"


def _better(arxiv_id: str, a: Path, b: Path) -> Path:
    """两个 PDF 里，哪个与 QASPER 自己的正文更吻合就用哪个。

    判据是 gold 探针的命中数——那正是评测要量的东西，
    用别的相似度只是绕远。取不到就退回第一个（不折腾）。
    """
    raw = json.loads(qasper.dev_path().read_text(encoding="utf-8"))
    entry = raw.get(arxiv_id)
    if not entry:
        return a
    probes = [x for _q, (_, golds) in qasper._evidence_map(entry).items()
              for g in golds for x in qasper.gold_probes(g)]
    if not probes:
        return a

    def score(path: Path) -> int:
        try:
            from papernest import structure
            chs = structure.section_chunks(str(path), 4000)
        except Exception:                               # noqa: BLE001
            return -1
        txt = qasper._norm("".join(c.get("text") or "" for c in chs))
        return sum(1 for p in probes if p in txt)

    return a if score(a) >= score(b) else b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=5,
                    help="抓几篇（默认 5 篇试水）。要全量就给一个大数。")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help="请求间隔秒数，别调小（arXiv 的批量下载礼仪）")
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--version", default="v1",
                    help="抓哪个版本（默认 v1；给空串就是最新版）。见模块注释。")
    ap.add_argument("--pick-best", action="store_true",
                    help="两版都抓，保留与 QASPER 正文吻合度更高的那版（慢一倍）")
    ap.add_argument("--refetch", action="store_true",
                    help="换版本时用：忽略磁盘上的旧文件。**定过版的仍然跳过**"
                         "（断点续传看 _versions.json，不看文件在不在）")
    ap.add_argument("--seed-manifest-after", default="",
                    help="把 mtime 晚于这个时刻的 PDF 记成已定版（"
                         "格式 'YYYY-MM-DD HH:MM'）。给中断在半路、"
                         "而当时还没有清单的那次跑收尾用。")
    ap.add_argument("--apply", action="store_true",
                    help="真下载（默认只报计划）")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    dest = Path(args.dest)
    if args.seed_manifest_after:
        import datetime as _dt
        try:
            cut = _dt.datetime.strptime(args.seed_manifest_after, "%Y-%m-%d %H:%M").timestamp()
        except ValueError:
            print("时刻格式应为 'YYYY-MM-DD HH:MM'"); return 1
        m = load_manifest(dest)
        n0 = len(m)
        for f in dest.glob("*.pdf"):
            if f.stat().st_mtime >= cut:
                # 记 "seeded" 而不是某个具体版本：**这批里哪些选了 v1、
                # 哪些选了最新版，事后从文件是看不出来的**，写一个具体版本号
                # 就是往数据文件里塞一句不确定的话。清单只用来决定跳不跳。
                m.setdefault(f.stem, "seeded")
        save_manifest(dest, m)
        print(f"清单补记 {len(m) - n0} 篇（共 {len(m)} 篇已定版）")
        return 0
    todo = wanted()
    manifest = load_manifest(dest)
    # 定过版的一律跳过——**这就是断点续传**，`--refetch` 也不例外。
    have = set(manifest) if args.refetch else (already_have() | set(manifest))
    missing = [(a, t, n) for a, t, n in todo if a not in have]
    tot_q = sum(n for _, _, n in todo)
    got_q = sum(n for a, _, n in todo if a in have)

    print(f"QASPER dev 带 gold 证据的论文 {len(todo)} 篇 / {tot_q} 道题")
    print(f"  已有 PDF 的: {len(todo) - len(missing)} 篇（{got_q} 道题）")
    print(f"  还缺的     : {len(missing)} 篇（{tot_q - got_q} 道题）")
    take = missing[:max(0, args.limit)]
    print(f"\n本次计划抓 {len(take)} 篇，约 {sum(n for _, _, n in take)} 道题，"
          f"间隔 {args.delay}s → 约 {len(take) * args.delay / 60:.1f} 分钟")
    for a, t, n in take[:8]:
        print(f"    {a}  ({n:>2} 题)  {t[:56]}")
    if len(take) > 8:
        print(f"    …… 还有 {len(take) - 8} 篇")
    if not args.apply:
        print("\n（这是计划。确认后加 --apply 才会真下载。）")
        return 0

    ok = fail = 0
    for i, (a, t, n) in enumerate(take, 1):
        p = dest / f"{a}.pdf"
        if a in manifest:
            continue
        if p.exists() and not args.refetch:
            manifest[a] = "existing"; save_manifest(dest, manifest)
            continue
        good, msg = fetch_one(a, p, version=args.version)
        chosen = args.version or "latest"
        if good and args.pick_best:
            alt = dest / f"{a}__alt.pdf"
            time.sleep(args.delay)
            if fetch_one(a, alt, version="")[0]:
                keep = _better(a, p, alt)
                if keep is alt:
                    p.unlink(missing_ok=True); alt.rename(p)
                    chosen = "latest"; msg += "（用最新版）"
                else:
                    alt.unlink(missing_ok=True); msg += f"（用 {args.version}）"
        if good:
            # **每篇写一次**，不攒到最后——攒着的话中断就等于没记。
            manifest[a] = chosen
            save_manifest(dest, manifest)
        ok, fail = ok + int(good), fail + int(not good)
        print(f"  [{i}/{len(take)}] {a} {'OK' if good else '失败'} {msg}")
        if i < len(take):
            time.sleep(args.delay)
    print(f"\n下载完成：成功 {ok}，失败 {fail}，存在 {dest}")
    print("评测会**直接从这个目录读**（`qasper.eval_tasks()` 按 arXiv id 对应），"
          "不用导进文献库——它们是评测夹具，混进库里会污染检索。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
