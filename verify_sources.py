"""第 0 步：数据源验证（立项方案 02 节）。

对 Semantic Scholar / OpenAlex / arXiv 三源各拉一批真实论文，量三个数：
  元数据完整率（标题+年份+作者）、摘要覆盖率、OA PDF 获取率（抽样实测下载）。
三率及格线：>=90% / >=85% / >=40%（按合并去重后的口径）。结果落盘 verify_report.json。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from papernest import config, http  # noqa: E402
from papernest.normalize import norm_key  # noqa: E402
from papernest.sources import arxiv, openalex, semantic_scholar  # noqa: E402

TH = {"meta": 0.90, "abstract": 0.85, "oa": 0.40}


def metrics(papers: list[dict]) -> dict:
    total = len(papers)
    oa_urls = [p["oa_pdf_url"] for p in papers if p.get("oa_pdf_url")]
    if total == 0:
        return {"total": 0, "meta_ok": 0, "abstract_ok": 0, "oa_listed": 0,
                "oa_urls": [], "meta_rate": 0.0, "abstract_rate": 0.0,
                "oa_rate_listed": 0.0}
    meta_ok = sum(1 for p in papers
                  if p.get("title") and p.get("year")
                  and (p.get("authors") or p.get("venue")))
    abs_ok = sum(1 for p in papers if p.get("abstract"))
    return {
        "total": total, "meta_ok": meta_ok, "abstract_ok": abs_ok,
        "oa_listed": len(oa_urls), "oa_urls": oa_urls,
        "meta_rate": round(meta_ok / total, 3),
        "abstract_rate": round(abs_ok / total, 3),
        "oa_rate_listed": round(len(oa_urls) / total, 3),
    }


def probe_pdfs(urls: list[str], sample: int = 5) -> dict:
    """抽样实测：真下载开头字节，验证 %PDF 魔数（不是只看字段存在）。"""
    sample_urls = urls[:sample]
    ok = 0
    details = []
    with http.client(timeout=40, headers={"User-Agent": "PaperNest/0.1 verify"}) as client:
        for u in sample_urls:
            try:
                r = client.get(u)
                is_pdf = r.content[:5] == b"%PDF-"
                ok += is_pdf
                details.append({"url": u[:80], "status": r.status_code,
                                "pdf": is_pdf, "bytes": len(r.content)})
            except Exception as e:
                details.append({"url": u[:80], "error": str(e)[:80]})
    rate = round(ok / len(sample_urls), 3) if sample_urls else None
    return {"sampled": len(sample_urls), "pdf_ok": ok, "download_rate": rate,
            "details": details}


def run(query: str, limit: int) -> dict:
    report = {"query": query, "limit": limit, "sources": {}, "errors": {}}
    fetchers = {
        "semantic_scholar": semantic_scholar.search,
        "openalex": openalex.search,
        "arxiv": arxiv.search,
    }
    merged: dict[str, dict] = {}
    for name, fn in fetchers.items():
        try:
            papers = fn(query, limit)
            m = metrics(papers)
            if m.get("total") and m.get("oa_urls"):
                m["pdf_probe"] = probe_pdfs(m["oa_urls"])
            report["sources"][name] = m
            for p in papers:
                k = p.get("norm_key")
                if k and k not in merged:
                    merged[k] = p
        except Exception as e:
            report["errors"][name] = f"{type(e).__name__}: {e}"[:200]

    union = list(merged.values())
    um = metrics(union)
    # 合并口径：任一源列出 OA 链接即计入（arXiv 恒为 OA）
    um["pdf_probe"] = probe_pdfs(um["oa_urls"])
    um["meta_pass"] = um.get("meta_rate", 0) >= TH["meta"]
    um["abstract_pass"] = um.get("abstract_rate", 0) >= TH["abstract"]
    um["oa_pass"] = (um["pdf_probe"]["download_rate"] or 0) >= TH["oa"]
    report["merged"] = um
    return report


def print_report(rep: dict):
    print(f"\n===== PaperNest 第 0 步 · 数据源验证（query={rep['query']!r}, limit={rep['limit']}）=====")
    for name, m in rep["sources"].items():
        if m.get("total", 0) == 0:
            print(f"[{name:<16}] 返回 0 条")
            continue
        print(f"[{name:<16}] 共{m['total']:>3} 篇 | 元数据完整 {m['meta_ok']:>3} ({m['meta_rate']:.0%})"
              f" | 有摘要 {m['abstract_ok']:>3} ({m['abstract_rate']:.0%})"
              f" | 列出OA PDF {m['oa_listed']:>3} ({m['oa_rate_listed']:.0%})")
        pr = m.get("pdf_probe")
        if pr and pr["sampled"]:
            print(f"[{name:<16}] PDF 抽样实测 {pr['pdf_ok']}/{pr['sampled']} 可下载"
                  f"（{pr['download_rate']:.0%}）")
    if rep["errors"]:
        print("--- 错误（如实记录）---")
        for name, e in rep["errors"].items():
            print(f"[{name:<16}] {e}")
    u = rep["merged"]
    dl = (u.get("pdf_probe") or {}).get("download_rate") or 0
    print("--- 合并去重口径（验收标准）---")
    print(f"合并 {u['total']} 篇 | 元数据完整率 {u['meta_rate']:.0%}（线 {TH['meta']:.0%}，{'PASS' if u['meta_pass'] else 'FAIL'}）"
          f" | 摘要覆盖率 {u['abstract_rate']:.0%}（线 {TH['abstract']:.0%}，{'PASS' if u['abstract_pass'] else 'FAIL'}）"
          f" | PDF 实测下载率 {dl:.0%}（线 {TH['oa']:.0%}，{'PASS' if u['oa_pass'] else 'FAIL'}）")
    overall = u["meta_pass"] and u["abstract_pass"] and u["oa_pass"]
    print(f"\n结论：{'第 0 步通过，可动工' if overall else '第 0 步未过，先解决数据源再动工'}")
    return overall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="large language model agent evaluation")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--save", default=str(config.DATA_DIR / "verify_report.json"))
    args = ap.parse_args()
    rep = run(args.query, args.limit)
    ok = print_report(rep)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"报告已存：{args.save}")
    sys.exit(0 if ok else 1)
