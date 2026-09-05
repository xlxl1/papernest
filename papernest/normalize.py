"""归一化主键：同一篇论文在任何来源、任何版本下只占一个缓存位。"""
import hashlib
import re


def norm_key(doi: str | None = None, arxiv_id: str | None = None,
             title: str | None = None) -> str | None:
    if doi:
        d = doi.lower().strip()
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
        if d:
            return "doi:" + d
    if arxiv_id:
        a = arxiv_id.strip().lower().replace("arxiv:", "")
        a = re.sub(r"v\d+$", "", a)  # 版本合并：2401.12345v2 -> arxiv:2401.12345
        if a:
            return "arxiv:" + a
    if title:
        t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title.lower())
        if t:
            return "title:" + hashlib.sha1(t.encode()).hexdigest()[:16]
    return None
