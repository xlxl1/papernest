"""归一化主键：同一篇论文在任何来源、任何版本下只占一个缓存位。"""
import hashlib
import re


#: `https://doi.org/10.x` / `dx.doi.org` / 裸 `doi.org/10.x` 三种前缀。
#: 这条正则在仓库里曾被独立抄了 4 份（normalize / bibimport / graph / obsidian），
#: 每一份都是某个模块自己踩坑之后局部打的补丁——于是「哪些下游是安全的」取决于
#: 谁碰巧记得加。而漏掉的那个（cite.py）就是真库里 100/242 篇导出即坏的原因。
_DOI_PREFIX_RE = re.compile(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/", re.I)


def bare_doi(doi: str | None) -> str | None:
    """剥掉 DOI 的 URL 前缀，拿到裸 DOI（`10.x/...`）。空值返回 None。

    **所有要用 DOI 的地方都该过这里**：OpenAlex 返回的就是完整 URL 形式，
    真库 242 篇有 DOI 的论文里 100 篇（41.3%）是这么存的。不剥前缀的后果各不相同
    但都很难看：BibTeX 里 `doi = {https://doi.org/10.x}`、Obsidian 链接拼成
    `https://doi.org/https://doi.org/…`、S2 查询拼成 `DOI:https://doi.org/10.x` 一律 404。
    """
    d = _DOI_PREFIX_RE.sub("", str(doi or "").strip())
    return d or None


def norm_key(doi: str | None = None, arxiv_id: str | None = None,
             title: str | None = None) -> str | None:
    if doi:
        d = (bare_doi(doi) or "").lower()
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
