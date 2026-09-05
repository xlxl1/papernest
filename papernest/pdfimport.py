"""本地 PDF 上传入库：字节流 → 校验 → 落盘 → 元数据启发式抽取 → papers/pages 入库。

设计取舍（为什么这么写）：
- **落盘文件名只用内容 sha256**。用户传来的 filename 一律不参与任何路径拼接——
  上传接口最经典的漏洞就是 filename 里塞 `../../`；filename 只在「标题什么都抽不到」
  时做显示兜底，且先 sanitize 成纯基名。
- **两级去重**：先按内容 sha256（同一份文件换个名字重传不该产生第二条论文），
  再按 norm_key（同一篇论文常常先被检索入库、后又被本地上传，此时应该把 PDF
  挂到既有论文上，而不是造一条重复记录）。
- **元数据靠启发式，必然有抽错抽不到的时候**：所以每个字段都带来源标注
  （extracted[字段]["from"]），调用方/前端可以据此决定要不要人工复核；
  抽不到就留空 + 进 warnings，绝不用文件名、排版软件写的垃圾标题、
  或从摘要里瞎猜的人名来填充。这是项目「证据可审计、不编造」红线在本模块的落点。
"""
import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import date
from pathlib import Path

from . import cards, config, db, fulltext
from .normalize import norm_key as _norm_key

# 上传硬约束：魔数 + 体积上限。50MB 之上多半是扫描件合订本，解析代价与收益不成比例。
PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 50 * 1024 * 1024

# 全文抽页上限，与 fulltext.extract_pages 的默认值保持一致
MAX_PAGES = 40


class PdfImportError(ValueError):
    """上传被拒或解析失败。继承 ValueError，调用方 except ValueError 也能兜住。"""


# ── schema ──

def ensure_schema():
    """幂等：本模块不新建表，只确保 papers.file_sha256 与其索引在位。

    file_sha256 由 db.MIGRATIONS 的 v2 提供；这里再兜一手是为了让本模块在
    「库文件由更老版本创建、迁移因故没跑到」时也不会 no such column 崩掉。
    不动 db.py 的迁移表。
    """
    with db.conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(papers)")}
        if "file_sha256" not in cols:
            c.execute("ALTER TABLE papers ADD COLUMN file_sha256 TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS idx_papers_sha ON papers(file_sha256)")


def _boot():
    db.init_db()
    ensure_schema()


# ── 文本清洗与文件名净化 ──

_WS_RE = re.compile(r"[\s\u00a0\u2000-\u200b]+")


def _clean(s: str | None) -> str:
    """NFKC 归一（连字 ﬁ→fi、全角→半角）+ 去软连字符 + 压缩空白。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00ad", "")
    return _WS_RE.sub(" ", s).strip()


def sanitize_filename(name: str | None) -> str:
    """把用户传来的 filename 压成一个安全的纯基名（仅用于显示兜底，不用于拼路径）。

    `../../evil.pdf`、`C:\\win\\x.pdf`、`a\\b/c.pdf` 一律只剩最后一段；
    盘符、控制字符、Windows 非法字符与首尾点全部清掉。
    """
    n = _clean(name).replace("\\", "/")
    n = re.sub(r"^[A-Za-z]:", "", n)   # 只削 Windows 盘符前缀
    n = n.split("/")[-1]
    # 冒号（含 NTFS 数据流后缀）替换成下划线而不是按它切断：
    # 按冒号切会把 "Attention: All You Need.pdf" 砍成 "All You Need.pdf"，
    # 而这个名字是标题抽不到时的兜底显示值，砍掉前半句纯属自残。
    n = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", n)
    n = n.strip(" .")             # ".." / "..." / 尾点（Windows 上尾点会被忽略）
    return n[:200] or "upload.pdf"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def uploads_dir() -> Path:
    """每次调用时读 config.DATA_DIR：测试把 DATA_DIR 指到临时目录才能生效。"""
    d = config.DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stored_path(sha: str) -> Path:
    return uploads_dir() / f"{sha[:16]}.pdf"


# ── 元数据启发式抽取 ──

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
# 新格式 2401.12345(v2)；旧格式 math.GT/0309136、hep-th/9901001
ARXIV_NEW_RE = re.compile(r"arxiv[:\s]*\s*(\d{4}\.\d{4,5})(v\d+)?", re.I)
ARXIV_OLD_RE = re.compile(r"arxiv[:\s]*\s*([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
ARXIV_STAMP_RE = re.compile(r"arxiv[:\s]", re.I)
YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")

_ABS_RE = re.compile(r"(?:^|\n)[ \t]*(?:abstract|a\s?b\s?s\s?t\s?r\s?a\s?c\s?t|摘\s*要)",
                     re.I)
# 「Abstract」与正文之间的分隔符五花八门（IEEE 用 —，有的 PDF 抽出来是 · 或 |），
# 与其穷举不如统一把正文开头的标点空白剥掉
_ABS_LEAD_RE = re.compile(r"^[\s\-—–‒·•∙:：.。|、,]+")
_ABS_END_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:[IVX]+|\d)\s*[.、]?\s*)?"
    r"(?:introduction|keywords?|key\s+words|index\s+terms|引\s*言|绪\s*论|关\s*键\s*词|关\s*键\s*字)",
    re.I)

# 排版软件塞进 PDF 元数据的垃圾标题
_META_TITLE_JUNK = re.compile(
    r"(?i)^(microsoft\s+word|untitled|document\s*\d*|no\s+title|doc\d*|manuscript|"
    r"main|ms|template|output|layout|print|paper)\b"
    r"|\.(docx?|pdf|tex|dvi|ps|rtf|indd|odt)\s*$|^pii\b", )
# 常见的「作者位」垃圾（Word 默认作者名）
_META_AUTHOR_JUNK = {"", "administrator", "admin", "user", "windows user", "unknown",
                     "author", "owner", "pc", "dell", "lenovo", "hp", "asus", "acer",
                     "microsoft", "microsoft office user", "microsoft word", "用户",
                     "管理员", "计算机", "本机用户"}
_AFFIL_WORDS = ("universi", "institut", "department", "laborator", "college", "school",
                "academy", "corporation", "company", "center", "centre", "hospital",
                "email", "e-mail", "@", ".com", ".edu", ".org", ".cn", "china", "usa",
                "germany", "france", "japan", "korea", "beijing", "shanghai", "abstract",
                "大学", "学院", "研究所", "实验室", "摘要", "中国", "北京", "上海",
                "南京", "杭州", "深圳", "广州", "武汉", "西安", "成都", "天津",
                "重庆", "香港", "台湾", "通信作者", "邮编")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# 一眼是文件路径而非标题（排版软件把源文件路径写进元数据是常事）
_PATHY_RE = re.compile(r"\\|://|^[A-Za-z]:[/\\]|^/[^ ]*/")


def _title_shape_ok(t: str) -> bool:
    """标题的**形状**判据：长度、字母占比、不是 arXiv 戳、不像文件路径。

    版面抽出来的标题只走这一条。见 `_looks_like_title`：那份「垃圾词表」是给
    PDF 内嵌元数据用的，套到版面标题上会误杀真实论文题目。
    """
    t = _clean(t)
    if not (6 <= len(t) <= 300):
        return False
    if _PATHY_RE.search(t):
        return False
    if ARXIV_STAMP_RE.search(t):
        return False
    letters = sum(1 for ch in t if ch.isalpha() or _CJK_RE.match(ch))
    return letters >= max(4, len(t) * 0.4)


def _looks_like_title(t: str) -> bool:
    """**仅用于 PDF 内嵌元数据标题**：形状判据 + 排版软件垃圾词表。

    词表（^microsoft word / ^document / ^template / ^layout / ^paper …）只在这里生效：
    内嵌标题写成 "Document1"、"template" 的概率极高，而版面上最大字号那一块叫
    "Template Matching for Object Detection"、"Layout Analysis of ..." 的概率同样极高
    ——同一套判据套两个地方，就会把后者一起毙掉：标题退化成文件名，norm_key
    跟着退化成「文件名哈希」，同一篇论文与检索入库的那条就再也对不上了。
    """
    t = _clean(t)
    if not _title_shape_ok(t):
        return False
    if "/" in t:                       # 内嵌标题里的 "/" 基本只出现在路径里
        return False
    return not _META_TITLE_JUNK.search(t)


def _page_lines(page) -> list[dict]:
    """把 page.get_text("dict") 摊平成行：文本 + 最大字号 + bbox。

    只保留水平方向的行——arXiv 的左侧竖排戳记 dir 不是 (1,0)，混进来会污染
    「最大字号块」的判定。
    """
    out = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            d = ln.get("dir") or (1.0, 0.0)
            if abs(d[0]) < 0.98:
                continue
            spans = [s for s in ln.get("spans", []) if (s.get("text") or "").strip()]
            if not spans:
                continue
            text = _clean("".join(s["text"] for s in spans))
            if not text:
                continue
            bbox = ln.get("bbox") or (0, 0, 0, 0)
            out.append({"text": text, "size": max(round(s["size"], 1) for s in spans),
                        "bold": any("bold" in (s.get("font") or "").lower() for s in spans),
                        "x0": bbox[0], "y0": bbox[1], "y1": bbox[3]})
    out.sort(key=lambda l: (round(l["y0"], 1), l["x0"]))
    return out


def _title_from_largest_font(lines: list[dict], page_height: float) -> tuple[str, int]:
    """页面顶部最大字号的**连续**文本块 = 标题。返回 (标题, 该块最后一行的下标)。

    只在页面上 60% 找候选：底部大字号多半是章节标题或图注；找不到再退回全页。
    纵向连续性用「行间距 <= 1.8 倍字号」判断，防止把标题和下方同字号的
    独立大字（如刊名）粘在一起。
    """
    if not lines:
        return "", -1
    top = [l for l in lines if l["y0"] <= page_height * 0.6] or lines
    top = [l for l in top if not ARXIV_STAMP_RE.search(l["text"])]
    if not top:
        return "", -1
    max_size = max(l["size"] for l in top)
    picked = [l for l in top if l["size"] >= max_size - 0.6]
    if not picked:
        return "", -1
    chosen = [picked[0]]
    for l in picked[1:]:
        if l["y0"] - chosen[-1]["y1"] > 1.8 * max_size:
            break
        chosen.append(l)
    parts = []
    for l in chosen:
        t = l["text"]
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1] + t      # 断词连字符：跨行拼回去
        else:
            parts.append(t)
    title = _clean(" ".join(parts))
    last_idx = lines.index(chosen[-1])
    # 版面标题只过「形状」判据：内嵌元数据那套垃圾词表在这里会误杀真实题目
    return (title, last_idx) if _title_shape_ok(title) else ("", last_idx)


_NAME_SPLIT_RE = re.compile(r"\s*(?:,|;|、|&|\band\b)\s*", re.I)
_MARKER_RE = re.compile(r"[0-9*†‡§¶#^∗⋆·•]+")


def _is_person_name(tok: str) -> bool:
    t = _clean(tok).strip(" ,.;·-")
    if not (2 <= len(t) <= 60) or "@" in t:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    low = t.lower()
    if any(w in low for w in _AFFIL_WORDS):
        return False
    if _CJK_RE.search(t):
        return 2 <= len(t) <= 10          # 中文姓名
    words = [w for w in t.split() if w]
    # 只有一个词的「作者」多半是机构缩写或误切，宁可漏也不编
    if not (2 <= len(words) <= 5):
        return False
    return all(w[0].isupper() for w in words)


def _authors_from_line(text: str) -> list[str]:
    """一行文本 → 人名列表。两道「宁可漏也不编」的闸门：

    ① 整行含机构/地名/邮箱线索的，**整行**作废——不是只丢掉那一段。
       「清华大学电子工程系, 北京」里逐段判会把「北京」当成中文姓名收下
       （2 个汉字、无数字、不在词表里），凭空造出一位叫「北京」的作者。
    ② 一行里「不像人名的段」不少于「像人名的段」时整行作废——
       「Smith, Alice; Lee, Bob Q.」这种姓在前的写法逐段判只剩半个「Bob Q」，
       那既不是真作者也不是真姓名，宁可留空让人工补。
    """
    cleaned = _clean(text)
    low = cleaned.lower()
    if any(w in low for w in _AFFIL_WORDS):
        return []
    toks = [t for t in _NAME_SPLIT_RE.split(_MARKER_RE.sub(" ", cleaned)) if _clean(t)]
    names = [t for t in toks if _is_person_name(t)]
    if len(names) <= len(toks) - len(names):
        return []
    return [_clean(n).strip(" ,.;·-") for n in names]


def _authors_below_title(lines: list[dict], title_idx: int) -> list[str]:
    """标题下方那一块：连续采集人名，遇到第一条「已采到人名后又没人名」的行就停。

    作者块是连续的；不连续地往下捞会把摘要里的大写词误判成人名。
    """
    out: list[str] = []
    for l in lines[title_idx + 1: title_idx + 8]:
        low = l["text"].lower()
        if re.match(r"^\s*(abstract|摘\s*要|keywords|index terms|1\s|i\.\s)", low):
            break
        if DOI_RE.search(l["text"]) or ARXIV_STAMP_RE.search(l["text"]):
            if out:
                break
            continue
        names = _authors_from_line(l["text"])
        if names:
            out.extend(names)
        elif out:
            break
    seen, uniq = set(), []
    for n in out:
        if n.lower() not in seen:
            seen.add(n.lower())
            uniq.append(n)
    return uniq


def _clean_doi(raw: str) -> str:
    d = raw.strip()
    d = re.sub(r"[.,;:'\"]+$", "", d)
    while d.endswith(")") and d.count(")") > d.count("("):
        d = d[:-1]
    while d.endswith("]") and d.count("]") > d.count("["):
        d = d[:-1]
    return d


def _find_doi(text: str) -> str:
    m = DOI_RE.search(text)
    return _clean_doi(m.group(0)) if m else ""


def _find_arxiv(text: str) -> str:
    m = ARXIV_NEW_RE.search(text) or ARXIV_OLD_RE.search(text)
    return m.group(1) if m else ""


def _year_from_arxiv(aid: str) -> int | None:
    m = re.match(r"^(\d{2})(\d{2})\.", aid or "")
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return None
    y = 2000 + yy
    return y if y <= date.today().year + 1 else 1900 + yy


def _find_year(text: str) -> int | None:
    cur = date.today().year + 1
    years = [int(m.group(0)) for m in YEAR_RE.finditer(text)]
    years = [y for y in years if 1900 <= y <= cur]
    return max(years) if years else None


_ABS_HEADING_RE = re.compile(r"^[ \t]*(?:\n|[\-—–‒·•∙:：.。|、])")


def _find_abstract(text: str) -> str:
    """Abstract/摘要 → Introduction/关键词 之间的正文。

    坑：`Abstract` 也可能出现在**标题**里（"Abstract Meaning Representation Parsing"
    是一整类真实论文）。取第一个匹配会把「标题剩下半句 + 作者行 + 单位」当成摘要
    写进库，而 from 还标着 keyword:Abstract→Introduction——这是在编造。
    所以先挑「像小标题的那个」：其后紧跟换行或分隔符（`Abstract\\n` / `Abstract—` /
    `Abstract:`）；一个都没有才退回第一个匹配（行为与原来一致）。
    """
    cands = list(_ABS_RE.finditer(text))
    if not cands:
        return ""
    heads = [m for m in cands if _ABS_HEADING_RE.match(text[m.end():])]
    for m in (heads or cands[:1]):
        rest = text[m.end():]
        end = _ABS_END_RE.search(rest)
        body = rest[:end.start()] if end else rest[:3000]
        body = re.sub(r"-\n", "", body)          # 行末断词
        body = _ABS_LEAD_RE.sub("", _clean(body))
        if len(body) >= 40:
            return body
    return ""


def extract_metadata(data: bytes, filename: str = "") -> dict:
    """从 PDF 字节流抽元数据。返回 {"extracted": {...带来源...}, "warnings": [...]}。

    extracted 的每个字段都是 {"value": ..., "from": "..."}；抽不到的字段
    value=None、from="none"，并在 warnings 里如实说明——调用方据此提示人工修正。
    """
    import pymupdf
    warnings: list[str] = []
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PdfImportError(f"PDF 解析失败（文件可能损坏）：{e}") from e
    try:
        if doc.needs_pass:
            raise PdfImportError("PDF 已加密（需要打开密码），无法抽取元数据与全文")
        meta = doc.metadata or {}
        n_pages = doc.page_count
        lines = _page_lines(doc[0]) if n_pages else []
        page_h = doc[0].rect.height if n_pages else 842.0
        head_text = "\n".join(doc[i].get_text("text") for i in range(min(2, n_pages)))
    finally:
        doc.close()

    ex: dict[str, dict] = {}

    # ── 标题：内嵌元数据 vs 顶部最大字号块 ──
    meta_title = _clean(meta.get("title"))
    meta_ok = _looks_like_title(meta_title)
    if meta_title and not meta_ok:
        warnings.append(f"PDF 内嵌标题疑似排版软件垃圾，已忽略：{meta_title[:60]!r}")
    font_title, title_idx = _title_from_largest_font(lines, page_h)
    if font_title and meta_ok:
        a = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", font_title.lower())
        b = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", meta_title.lower())
        if a == b or a in b or b in a:
            ex["title"] = {"value": font_title, "from": "largest-font-block+pdf-metadata"}
        else:
            ex["title"] = {"value": font_title, "from": "largest-font-block"}
            warnings.append(
                f"内嵌标题与版面标题不一致，取版面标题；内嵌为 {meta_title[:60]!r}，请人工确认")
    elif font_title:
        ex["title"] = {"value": font_title, "from": "largest-font-block"}
    elif meta_ok:
        ex["title"] = {"value": meta_title, "from": "pdf-metadata"}
    else:
        ex["title"] = {"value": None, "from": "none"}
        warnings.append("标题抽取失败（内嵌元数据不可用且版面无明显最大字号块）")

    # ── 作者 ──
    meta_author = _clean(meta.get("author"))
    meta_names = ([] if meta_author.lower() in _META_AUTHOR_JUNK
                  else _authors_from_line(meta_author))
    if meta_names:
        ex["authors"] = {"value": meta_names, "from": "pdf-metadata-author"}
    else:
        if meta_author and not meta_names:
            warnings.append(f"PDF 内嵌作者不可用，已忽略：{meta_author[:60]!r}")
        below = _authors_below_title(lines, title_idx) if title_idx >= 0 else []
        if below:
            ex["authors"] = {"value": below, "from": "below-title-block"}
        else:
            ex["authors"] = {"value": None, "from": "none"}
            warnings.append("作者抽取失败，已留空（按不编造原则不猜人名，请手工补全）")

    # ── DOI / arXiv ──
    doi = _find_doi(head_text)
    ex["doi"] = ({"value": doi, "from": "regex:first-2-pages"} if doi
                 else {"value": None, "from": "none"})
    aid = _find_arxiv(head_text)
    ex["arxiv_id"] = ({"value": aid, "from": "regex:first-2-pages"} if aid
                      else {"value": None, "from": "none"})
    if not doi and not aid:
        warnings.append("未抽到 DOI / arXiv ID，去重主键退化为标题哈希"
                        "（同一篇的不同版本可能被判成两篇）")

    # ── 年份 ──
    y = _year_from_arxiv(aid)
    if y:
        ex["year"] = {"value": y, "from": "arxiv-id-prefix"}
    else:
        y = _find_year(head_text)
        if y:
            ex["year"] = {"value": y, "from": "regex:first-2-pages(max)"}
        else:
            y = _year_from_pdf_date(meta)
            if y:
                ex["year"] = {"value": y, "from": "pdf-metadata-creationDate（非发表年，仅供参考）"}
                # papers.year 这一列本身带不了「仅供参考」的标签，下游（引文格式化、
                # 综述表格）只会当成发表年直接印出来，所以必须在 warnings 里点名。
                warnings.append(f"正文中未找到年份，暂用 PDF 创建日期的 {y} 兜底——"
                                "那是文件生成年、不是发表年（扫描件尤其不可信），请人工核对")
            else:
                ex["year"] = {"value": None, "from": "none"}
                warnings.append("年份抽取失败，已留空")

    # ── 摘要 ──
    ab = _find_abstract(head_text)
    if ab:
        ex["abstract"] = {"value": ab[:6000], "from": "keyword:Abstract→Introduction"}
    else:
        ex["abstract"] = {"value": None, "from": "none"}
        warnings.append("摘要抽取失败（未找到 Abstract/摘要 段落），已留空")

    ex["pages"] = {"value": n_pages, "from": "pymupdf"}
    if not n_pages:
        warnings.append("PDF 页数为 0，可能是空文件")
    elif n_pages > MAX_PAGES:
        # 不说这句，用户会以为全文都进库了，然后在对话里怎么也检索不到第 41 页往后的内容
        warnings.append(f"PDF 共 {n_pages} 页，全文抽取只入库前 {MAX_PAGES} 页；"
                        f"第 {MAX_PAGES + 1} 页之后的内容不会被检索到")
    return {"extracted": ex, "warnings": warnings}


def _year_from_pdf_date(meta: dict) -> int | None:
    m = re.search(r"D:(\d{4})", meta.get("creationDate") or "")
    if not m:
        return None
    y = int(m.group(1))
    return y if 1900 <= y <= date.today().year + 1 else None


def _val(ex: dict, field: str):
    return (ex.get(field) or {}).get("value")


# ── 入库 ──

def _validate(data: bytes):
    if not data:
        raise PdfImportError("上传内容为空")
    if len(data) > MAX_PDF_BYTES:
        raise PdfImportError(
            f"文件 {len(data) / 1048576:.1f}MB 超过 {MAX_PDF_BYTES // 1048576}MB 上限，已拒绝")
    if data[:len(PDF_MAGIC)] != PDF_MAGIC:
        raise PdfImportError("不是 PDF 文件（缺少 %PDF- 魔数），已拒绝")


def _pages_count(paper_id: int) -> int:
    with db.conn() as c:
        return c.execute("SELECT COUNT(*) n FROM pages WHERE paper_id=?",
                         (paper_id,)).fetchone()["n"]


def _refresh_fts(c: sqlite3.Connection, paper_id: int):
    """FTS 是外部内容表，papers 改了必须手工 DELETE+INSERT（同 db.save_card）。"""
    row = c.execute("SELECT title, abstract, card_json FROM papers WHERE id=?",
                    (paper_id,)).fetchone()
    if not row:
        return
    try:
        card = json.loads(row["card_json"] or "{}")
        kws = (json.dumps(card.get("keywords") or [], ensure_ascii=False)
               if isinstance(card, dict) else "")
    except ValueError:
        kws = ""
    c.execute("DELETE FROM papers_fts WHERE rowid=?", (paper_id,))
    c.execute("INSERT INTO papers_fts(rowid,title,abstract,keywords) VALUES(?,?,?,?)",
              (paper_id, row["title"], row["abstract"] or "", kws))


def _make_card(paper_id: int, title: str, abstract: str | None, topic: str | None) -> str:
    card, model = cards.make_card(
        {"_db_id": paper_id, "title": title, "abstract": abstract},
        topic or config.RESEARCH_TOPIC)
    with db.conn() as c:
        db.save_card(c, paper_id, card, model)
    return model


def ingest_pdf(data: bytes, filename: str, topic: str | None = None,
               make_card: bool = True) -> dict:
    """本地 PDF 入库主入口。

    返回 dict：paper_id / title / norm_key / pages_stored / duplicate /
    duplicate_by / extracted / warnings / sha256 / pdf_path / card_model / filename。
    被拒或解析失败一律抛 PdfImportError（ValueError 子类），不返回半成品。
    """
    _boot()
    _validate(data)
    safe_name = sanitize_filename(filename)
    sha = sha256_hex(data)
    warnings: list[str] = []

    # ① 内容去重：同一份文件换个名字重传，只认哈希不认文件名
    with db.conn() as c:
        row = c.execute("SELECT * FROM papers WHERE file_sha256=?", (sha,)).fetchone()
    if row:
        warnings.append(f"命中内容 sha256 去重（上传名 {safe_name!r} 不影响判定），"
                        "未重复解析元数据")
        path = stored_path(sha)
        if not path.exists():
            path.write_bytes(data)      # 库里有记录但文件丢了：顺手补回
            warnings.append("既有记录的 PDF 文件缺失，已用本次上传的同哈希内容补回")
        n = _pages_count(row["id"])
        if n == 0:
            n = _extract_pages_safe(str(path), row["id"], warnings)
        with db.conn() as c:
            c.execute("UPDATE papers SET pdf_path=?, updated_at=datetime('now','localtime') "
                      "WHERE id=?", (str(path), row["id"]))
        return _result(row["id"], row["title"], row["norm_key"], n, True, "sha256",
                       {}, warnings, sha, str(path), row["card_model"], safe_name)

    # ② 解析元数据
    parsed = extract_metadata(data, safe_name)
    ex, warnings = parsed["extracted"], warnings + parsed["warnings"]
    title = _val(ex, "title")
    doi, aid = _val(ex, "doi"), _val(ex, "arxiv_id")
    if not title:
        stem = re.sub(r"\.pdf$", "", safe_name, flags=re.I).strip() or f"未命名上传-{sha[:8]}"
        title = stem
        ex["title"] = {"value": stem, "from": "filename-fallback"}
        warnings.append("元数据抽取失败，标题暂用文件名兜底，请手工修正"
                        "（update_metadata 会重算 norm_key）")
    key = _norm_key(doi=doi, arxiv_id=aid, title=title)
    if not key:
        raise PdfImportError("无法为该 PDF 生成归一化主键（标题、DOI、arXiv ID 全部为空）")

    # ③ norm_key 去重：这篇可能早就被检索入库过，此时把 PDF 挂上去而不是造重复行
    with db.conn() as c:
        exist = db.get_by_norm_key(c, key)
    if exist:
        return _attach_to_existing(exist, data, sha, key, ex, warnings, safe_name,
                                   topic, make_card)

    # ④ 新论文入库
    paper = {"norm_key": key, "title": title, "abstract": _val(ex, "abstract"),
             "year": _val(ex, "year"), "venue": None,
             "authors": _val(ex, "authors") or [], "doi": doi, "arxiv_id": aid,
             "citation_count": None, "oa_pdf_url": None, "source": "upload",
             "s2_id": None}
    path = stored_path(sha)
    path.write_bytes(data)
    try:
        with db.conn() as c:
            pid = db.insert_l0(c, paper)
            c.execute("UPDATE papers SET file_sha256=?, pdf_path=? WHERE id=?",
                      (sha, str(path), pid))
    except sqlite3.IntegrityError:
        # 并发上传同一篇：UNIQUE(norm_key) 兜住，退回「挂到既有论文」分支
        with db.conn() as c:
            exist = db.get_by_norm_key(c, key)
        if not exist:
            raise
        return _attach_to_existing(exist, data, sha, key, ex, warnings, safe_name,
                                   topic, make_card)

    n = _extract_pages_safe(str(path), pid, warnings)
    card_model = None
    if make_card:
        card_model = _make_card(pid, title, _val(ex, "abstract"), topic)
    return _result(pid, title, key, n, False, None, ex, warnings, sha, str(path),
                   card_model, safe_name)


def _fill_blanks(pid: int, ex: dict, warnings: list[str]):
    """只补既有记录里**空着**的字段，绝不覆盖已有值，并如实记 warning。

    检索入库的 L0 记录常常缺摘要/作者/年份，而本地 PDF 恰好抽到了；
    直接丢弃这些信息是浪费。覆盖已有值则是有损操作，留给 update_metadata 由人来做。
    """
    with db.conn() as c:
        row = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
    sets, params, filled = [], [], []
    ab = _val(ex, "abstract")
    if ab and not (row["abstract"] or "").strip():
        sets.append("abstract=?")
        params.append(ab)
        filled.append("abstract")
    yr = _val(ex, "year")
    if yr and not row["year"]:
        sets.append("year=?")
        params.append(yr)
        filled.append("year")
    au = _val(ex, "authors")
    try:
        had = json.loads(row["authors"] or "[]")
    except ValueError:
        had = []
    if au and not had:
        sets.append("authors=?")
        params.append(json.dumps(au, ensure_ascii=False))
        filled.append("authors")
    if not sets:
        return row
    warnings.append(f"已用本次 PDF 抽取结果补齐既有记录的空字段：{filled}"
                    "（原有非空字段一律未改动）")
    with db.conn() as c:
        c.execute(f"UPDATE papers SET {', '.join(sets)}, "
                  "updated_at=datetime('now','localtime') WHERE id=?", params + [pid])
        _refresh_fts(c, pid)
        return c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()


def _attach_to_existing(exist, data: bytes, sha: str, key: str, ex: dict,
                        warnings: list[str], safe_name: str,
                        topic: str | None = None, make_card: bool = True) -> dict:
    """norm_key 命中既有论文：把本地 PDF 挂上去，绝不新建第二条记录。"""
    pid = exist["id"]
    warnings.append(f"norm_key 命中既有论文 #{pid}（{exist['norm_key']}），"
                    "本次上传不新建记录")
    old_sha = exist["file_sha256"] if "file_sha256" in exist.keys() else None
    if old_sha and old_sha != sha:
        # 同一篇论文的另一份文件（不同版本/不同排版）：不静默替换既有关联
        warnings.append(f"该论文已关联另一个 PDF（sha256 {old_sha[:16]}…），"
                        "本次文件未落盘、未替换关联；如需替换请先删除原关联")
        path = exist["pdf_path"]
    else:
        path = str(stored_path(sha))
        if not Path(path).exists():
            Path(path).write_bytes(data)
        with db.conn() as c:
            c.execute("UPDATE papers SET file_sha256=?, pdf_path=?, "
                      "updated_at=datetime('now','localtime') WHERE id=?",
                      (sha, path, pid))
    n = _pages_count(pid)
    if n == 0 and path and Path(path).exists():
        n = _extract_pages_safe(path, pid, warnings)
    row = _fill_blanks(pid, ex, warnings)
    # 既有记录还没卡片而调用方要卡片：这里补，否则 make_card=True 会静默无卡片
    card_model = row["card_model"]
    if make_card and not row["card_json"]:
        card_model = _make_card(pid, row["title"], row["abstract"], topic)
    return _result(pid, row["title"], row["norm_key"], n, True, "norm_key",
                   ex, warnings, sha, path, card_model, safe_name)


def _extract_pages_safe(path: str, paper_id: int, warnings: list[str]) -> int:
    """抽页失败不该让整次上传回滚——论文行已经有价值，如实记 warning 即可。"""
    try:
        fulltext.extract_pages(path, paper_id, max_pages=MAX_PAGES)
    except Exception as e:
        warnings.append(f"全文抽页失败（论文元数据已入库）：{e}")
        return 0
    return _pages_count(paper_id)


def _result(paper_id, title, key, pages, dup, dup_by, ex, warnings, sha, path,
            card_model, filename) -> dict:
    return {"paper_id": paper_id, "title": title, "norm_key": key,
            "pages_stored": pages, "duplicate": dup, "duplicate_by": dup_by,
            "extracted": ex, "warnings": warnings, "sha256": sha,
            "pdf_path": path, "card_model": card_model, "filename": filename}


def ingest_file(path: str | Path, topic: str | None = None,
                make_card: bool = True) -> dict:
    """CLI 便捷入口：从本地路径读盘再走 ingest_pdf（体积上限先按 stat 挡一道）。"""
    p = Path(path)
    if not p.is_file():
        raise PdfImportError(f"文件不存在：{p}")
    if p.stat().st_size > MAX_PDF_BYTES:
        raise PdfImportError(
            f"文件 {p.stat().st_size / 1048576:.1f}MB 超过 "
            f"{MAX_PDF_BYTES // 1048576}MB 上限，已拒绝")
    return ingest_pdf(p.read_bytes(), p.name, topic=topic, make_card=make_card)


# ── 人工修正 ──

EDITABLE_FIELDS = ("title", "authors", "year", "doi", "arxiv_id", "abstract", "venue")


def _coerce(field: str, v):
    if field == "title":
        t = _clean(v)
        if not t:
            raise PdfImportError("title 不能为空")
        return t
    if field == "authors":
        if v is None:
            return []
        if isinstance(v, str):
            return [x for x in (_clean(s) for s in _NAME_SPLIT_RE.split(v)) if x]
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v if _clean(x)]
        raise PdfImportError("authors 需要 list 或逗号分隔的字符串")
    if field == "year":
        if v in (None, ""):
            return None
        try:
            y = int(v)
        except (TypeError, ValueError):
            raise PdfImportError(f"year 不是整数：{v!r}") from None
        if not 1500 <= y <= date.today().year + 2:
            raise PdfImportError(f"year 超出合理范围：{y}")
        return y
    if field == "doi":
        d = _clean(v)
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
        return d or None
    if field == "arxiv_id":
        a = re.sub(r"^arxiv:", "", _clean(v), flags=re.I)
        return a or None
    return _clean(v) or None


def update_metadata(paper_id: int, **fields) -> dict:
    """人工修正抽错的元数据。改动 title/doi/arxiv_id 时重算 norm_key。

    norm_key 与其它论文冲突时**报错**而不是静默覆盖——两条记录合并是有损操作，
    必须由人来决定留哪条。同时刷新 papers_fts，让改后的标题立刻能被检索到。
    """
    _boot()
    unknown = [k for k in fields if k not in EDITABLE_FIELDS]
    if unknown:
        raise PdfImportError(f"不支持修改的字段：{unknown}；可改：{list(EDITABLE_FIELDS)}")
    if not fields:
        raise PdfImportError("没有要修改的字段")

    with db.conn() as c:
        row = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        raise PdfImportError(f"论文 {paper_id} 不存在")

    new = {k: _coerce(k, v) for k, v in fields.items()}
    changed: dict[str, list] = {}
    for k, v in new.items():
        old = row[k]
        if k == "authors":
            try:
                old = json.loads(old or "[]")
            except ValueError:
                old = []
        if old != v:
            changed[k] = [old, v]
    warnings: list[str] = []
    key = row["norm_key"]
    if {"title", "doi", "arxiv_id"} & set(changed):
        merged_title = new.get("title", row["title"])
        merged_doi = new["doi"] if "doi" in new else row["doi"]
        merged_aid = new["arxiv_id"] if "arxiv_id" in new else row["arxiv_id"]
        cand = _norm_key(doi=merged_doi, arxiv_id=merged_aid, title=merged_title)
        if not cand:
            raise PdfImportError("修改后无法生成归一化主键（title/doi/arxiv_id 全空）")
        if cand != key:
            with db.conn() as c:
                clash = db.get_by_norm_key(c, cand)
            if clash and clash["id"] != paper_id:
                raise PdfImportError(
                    f"norm_key 冲突：{cand} 已被论文 #{clash['id']}"
                    f"《{clash['title'][:60]}》占用；不做静默覆盖，"
                    "请先确认两条是否为同一篇再决定合并或改用别的标识")
            warnings.append(f"norm_key 已从 {key} 重算为 {cand}")
            key = cand

    sets, params = [], []
    for k, v in new.items():
        sets.append(f"{k}=?")
        params.append(json.dumps(v, ensure_ascii=False) if k == "authors" else v)
    sets.append("norm_key=?")
    params.append(key)
    sets.append("updated_at=datetime('now','localtime')")
    params.append(paper_id)
    with db.conn() as c:
        c.execute(f"UPDATE papers SET {', '.join(sets)} WHERE id=?", params)
        _refresh_fts(c, paper_id)
        out = c.execute("SELECT id,title,norm_key,year,doi,arxiv_id,authors "
                        "FROM papers WHERE id=?", (paper_id,)).fetchone()
    return {"paper_id": paper_id, "title": out["title"], "norm_key": out["norm_key"],
            "norm_key_changed": out["norm_key"] != row["norm_key"],
            "changed": changed, "warnings": warnings}


def list_uploads(limit: int = 50) -> list[dict]:
    """本地上传进来的论文（source='upload' 或带 file_sha256 的）。"""
    _boot()
    with db.conn() as c:
        rows = c.execute(
            "SELECT id,title,year,norm_key,pdf_path,file_sha256,level,card_model,"
            "created_at FROM papers WHERE source='upload' OR file_sha256 IS NOT NULL "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
