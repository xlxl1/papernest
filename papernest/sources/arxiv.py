"""arXiv API —— 预印本与免费全文 PDF 的来源（官方 API，Atom XML）。"""
import re
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


# 合法 arXiv id：新式 2608.00011[v3]，旧式 math.GT/0309136[v1] / hep-th/9901001。
# **必须校验**：arXiv 在查询非法时返回的不是 HTTP 错误，而是一个**形状完全合法的
# Atom feed**，里面一条 id=http://arxiv.org/api/errors#... 、title=Error 的 entry。
# 不校验的话这条会被当成一篇论文入库——标题「Error」、arxiv_id 是一整条 URL、
# oa_pdf_url 拼成 https://arxiv.org/pdf/http://arxiv.org/api/errors#...。
# subscribe.py 早就挡了这个（那里有 6 行注释解释原因），而平行实现的**采集路径**
# 一直没挡——同一个数据源两套判据，其中一套是空的。
_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$")


def _parse(root: ET.Element) -> list[dict]:
    papers = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", "", NS) or "")
        if "/abs/" not in raw_id:
            continue          # 不是论文条目（arXiv 的 error feed 走的就是这条）
        arxiv_id = raw_id.split("/abs/")[-1].strip()
        title = (e.findtext("a:title", "", NS) or "").strip().replace(chr(10), " ")
        if not title or not _ARXIV_ID_RE.match(arxiv_id):
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
