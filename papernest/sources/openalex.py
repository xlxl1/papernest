"""OpenAlex —— 备用数据源（官方 API、完全免费、聚合池只需带 mailto）。

摘要以倒排索引存储，需重建；S2 不可用或返回为空时兜底。
"""
from .. import config, http
from ..normalize import norm_key

SEARCH_URL = "https://api.openalex.org/works"


def _abstract_from_inverted(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search(query: str, limit: int = 20) -> list[dict]:
    params = {"search": query, "per-page": min(limit, 100)}
    if config.OPENALEX_MAILTO:      # 有邮箱才进礼貌池；空值不要发上去
        params["mailto"] = config.OPENALEX_MAILTO
    with http.client() as client:
        r = client.get(SEARCH_URL, params=params)
        r.raise_for_status()
        works = r.json().get("results") or []
    papers = []
    for w in works:
        if not w.get("title"):
            continue
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        oa = w.get("open_access") or {}
        best = w.get("best_oa_location") or {}
        doi = w.get("doi")
        papers.append({
            "norm_key": norm_key(doi=doi, title=w.get("title")),
            "title": w["title"].strip(),
            "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
            "year": w.get("publication_year"),
            "venue": src.get("display_name"),
            "authors": [a.get("author", {}).get("display_name")
                        for a in w.get("authorships") or []
                        if a.get("author", {}).get("display_name")],
            "doi": doi,
            "arxiv_id": None,
            "citation_count": w.get("cited_by_count"),
            "oa_pdf_url": best.get("pdf_url") or oa.get("oa_url"),
            "source": "openalex",
            "s2_id": None,
        })
    return papers
