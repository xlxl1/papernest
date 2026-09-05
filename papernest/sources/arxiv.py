"""arXiv API —— 预印本与免费全文 PDF 的来源（官方 API，Atom XML）。"""
import time
import xml.etree.ElementTree as ET

from .. import config, http
from ..normalize import norm_key

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def search(query: str, limit: int = 20) -> list[dict]:
    params = {
        "start": 0,
        "max_results": min(limit, 100),
        "sortBy": "relevance",
    }
    with http.client() as client:
        # 先试精确短语，0 结果时降级为 AND 词组（arXiv 短语匹配较严）
        for q in (f'all:"{query}"',
                  " AND ".join(f"all:{w}" for w in query.split()[:6])):
            params["search_query"] = q
            r = client.get(API_URL, params=params,
                           headers={"User-Agent": "PaperNest/0.1 (personal research tool)"})
            r.raise_for_status()
            root = ET.fromstring(r.text)
            papers = _parse(root)
            if papers:
                break
    return papers


def _parse(root: ET.Element) -> list[dict]:
    papers = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", "", NS) or "")
        arxiv_id = raw_id.split("/abs/")[-1].strip()
        title = (e.findtext("a:title", "", NS) or "").strip().replace("\n", " ")
        if not arxiv_id or not title:
            continue
        published = (e.findtext("a:published", "", NS) or "")[:10]
        summary = (e.findtext("a:summary", "", NS) or "").strip()
        papers.append({
            "norm_key": norm_key(title=title, arxiv_id=arxiv_id),
            "title": title,
            "abstract": summary or None,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
            "authors": [a.findtext("a:name", "", NS)
                        for a in e.findall("a:author", NS)],
            "doi": None,
            "arxiv_id": arxiv_id,
            "citation_count": None,
            # arXiv 全文 PDF 恒为 OA；abs 页如果带版本号，PDF 同样可用
            "oa_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "source": "arxiv",
            "s2_id": None,
        })
    return papers


def polite_sleep():
    """arXiv 官方建议连续请求间隔 3 秒。"""
    time.sleep(3)
