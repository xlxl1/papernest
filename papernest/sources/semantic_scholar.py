"""Semantic Scholar Graph API —— 主数据源（官方 API，可申请免费 key）。

429/5xx 退避基数取 10s 级：S2 无 key 共享池按 5 分钟窗口限流，
秒级退避会全部落回同一个封锁窗口（QC-Bench F1 的教训）。
"""
import time

import httpx

from .. import config, http
from ..normalize import norm_key

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = ("paperId,title,abstract,year,venue,externalIds,authors,"
          "citationCount,openAccessPdf,isOpenAccess")


class SourceError(Exception):
    pass


def _backoff_statuses():
    return (429, 500, 502, 503, 504)


def search(query: str, limit: int = 20) -> list[dict]:
    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    headers = {}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY
    # 无 key 共享池按突发拥塞 429；退避跨过 5 分钟窗口边界（±20% 抖动防惊群）
    delays = (20, 45, 90, 150)
    last = None

    def _backoff(attempt: int):
        """第 attempt 次失败后的退避。**最后一次尝试之后不睡**——后面已经没有下一次
        请求了，睡满那 150 秒只是把调用方白白多锁 150 秒（单次查询最坏从 155s 变 305s）。
        口径与 graph.py 的 `_backoff` 一致；这个文件原来的注释声称「与 graph.py 同一套
        序列」，但恰恰漏了 graph.py 明确写出来要避免的这一条。
        """
        if attempt < len(delays) - 1:
            _sleep(delays[attempt] * _jitter())

    with http.client() as client:
        for attempt in range(4):
            try:
                r = client.get(SEARCH_URL, params=params, headers=headers)
            except httpx.HTTPError as e:
                last = e
                _backoff(attempt)
                continue
            if r.status_code in _backoff_statuses():
                last = f"HTTP {r.status_code}"
                _backoff(attempt)
                continue
            if r.status_code == 400:
                raise SourceError(f"Semantic Scholar 请求被拒（400）：{r.text[:200]}")
            r.raise_for_status()
            return [_to_paper(p) for p in r.json().get("data") or [] if p.get("title")]
        raise SourceError(f"Semantic Scholar 连续 4 次失败：{last}")


def _jitter() -> float:
    import random
    return 0.8 + random.random() * 0.4


def _sleep(sec: float):
    time.sleep(sec)


def _to_paper(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return {
        "norm_key": norm_key(doi=ext.get("DOI"), arxiv_id=ext.get("ArXiv"),
                             title=p.get("title")),
        "title": (p.get("title") or "").strip(),
        "abstract": p.get("abstract"),
        "year": p.get("year"),
        "venue": p.get("venue") or None,
        "authors": [a.get("name") for a in p.get("authors") or [] if a.get("name")],
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "citation_count": p.get("citationCount"),
        "oa_pdf_url": oa.get("url"),
        "source": "s2",
        "s2_id": p.get("paperId"),
    }
