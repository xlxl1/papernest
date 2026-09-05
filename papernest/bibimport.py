"""BibTeX / RIS / Zotero CSV 导入 —— 存量文献的迁移入口。

项目此前只有导出（cite.py），用户在 Zotero/EndNote/Mendeley 里攒的几百篇一篇都搬不进来。
这里补上反向通道，与 cite.to_bibtex / to_ris 的输出保持往返一致。

三条设计红线（对齐项目「证据可审计、不编造」）：
  1) 抽不到的字段一律留空，绝不用启发式猜测填充（venue 不猜、oa_pdf_url 不由 arXiv ID 拼）；
  2) 单条 entry 解析失败只记进 errors 并继续下一条，不让一个坏括号毁掉整份 .bib；
  3) 入库遇到已存在的 norm_key 只「补空」不覆盖——用户库里的人工修订永远优先于导入文件。

不引入 bibtexparser（未安装），BibTeX 全部手写扫描。
"""
import csv
import io
import json
import re
import time
import unicodedata
from urllib.parse import quote

import httpx

from . import config, db, http
from .normalize import norm_key

# 错误信息前缀：带这个标记的表示「有一条文献真的没进来」，不带的只是警告
# （例如某个字段被跳过、缺右花括号但条目仍尽力解析出来了）。
# import_text 用它来算 failed —— 否则警告会被当成失败条目计数，
# 3 条成功导入的文件会显示成「failed: 3」。
DROP_TAG = "【未导入】"

# ── 本模块自带的幂等 schema：导入批次的审计流水 ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS import_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now','localtime')),
  fmt TEXT NOT NULL DEFAULT '',
  source_name TEXT DEFAULT '',      -- 文件名等来源标注，由调用方传入
  total INTEGER DEFAULT 0,
  imported INTEGER DEFAULT 0,
  skipped INTEGER DEFAULT 0,
  updated INTEGER DEFAULT 0,        -- 「补空不覆盖」命中的老记录数
  failed INTEGER DEFAULT 0,
  errors_json TEXT NOT NULL DEFAULT '[]'
);
"""

_schema_done: set[str] = set()


def ensure_schema():
    """幂等建表。按 DB 路径记忆，避免每次公开调用都 executescript（测试会切换 DB_PATH）。"""
    key = str(config.DB_PATH)
    if key in _schema_done and config.DB_PATH.exists():
        return
    with db.conn() as c:
        c.executescript(_SCHEMA)
    _schema_done.add(key)


def _ensure():
    db.init_db()
    ensure_schema()


# ── LaTeX 反转义 ──
# 说明：只做常见几类；遇到不认识的命令原样保留而不是抛错或吞掉内容。

# 私有区哨兵：转义出来的字符不能再参与后续的「剥花括号 / ~ 转空格」两步
_S_LBRACE, _S_RBRACE, _S_TILDE, _S_BSLASH, _S_CARET = "\x01\x02\x03\x04\x05"

_ACCENTS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "c": "\u0327", "v": "\u030c", "u": "\u0306",
    "H": "\u030b", "r": "\u030a", "d": "\u0323", "b": "\u0331",
}
# 重音基字母允许无点 i/j（\'{\i} 这种写法先被下面的 _DOTLESS 换成 ı 再吃重音）
_BASE = "A-Za-z\u0131\u0237"
_ACCENT_SYM_RE = re.compile(r'\\(["\'`^~=.])\s*\{?\s*([' + _BASE + r'])\s*\}?')
_ACCENT_CMD_RE = re.compile(r"\\([cvuHrdb])\s*\{\s*([" + _BASE + r"])\s*\}")

_SPECIAL = {"ss": "ß", "aa": "å", "AA": "Å", "ae": "æ", "AE": "Æ",
            "oe": "œ", "OE": "Œ", "o": "ø", "O": "Ø", "l": "ł", "L": "Ł"}
_SPECIAL_RE = re.compile(r"\\(ss|aa|AA|ae|AE|oe|OE|o|O|l|L)(?![A-Za-z])")
_DOTLESS_RE = re.compile(r"\\([ij])(?![A-Za-z])")
# 纯排版命令：丢命令留内容（花括号在后面统一剥掉）
_TEXTCMD_RE = re.compile(
    r"\\(?:textit|textbf|texttt|textsc|textrm|textsl|emph|mbox|text|uppercase|lowercase)\s*(?=\{)")


def _accent(m: "re.Match") -> str:
    return unicodedata.normalize("NFC", m.group(2) + _ACCENTS[m.group(1)])


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _latex_unescape(s: str, keep_tilde: bool = False) -> str:
    """BibTeX 值 → 普通文本：重音、特殊字母、转义标点、大小写保护花括号。

    keep_tilde=True 用于 URL/DOI——裸 `~` 在 LaTeX 里是不断行空格，但在 URL 里是路径字符，
    按文本规则转成空格会把链接改坏。
    """
    if not s:
        return ""
    out = s.replace(r"\textbackslash{}", _S_BSLASH).replace(r"\textbackslash", _S_BSLASH)
    out = out.replace(r"\textasciitilde{}", _S_TILDE).replace(r"\textasciitilde", _S_TILDE)
    out = out.replace(r"\textasciicircum{}", _S_CARET).replace(r"\textasciicircum", _S_CARET)
    out = _TEXTCMD_RE.sub("", out)
    out = _DOTLESS_RE.sub(lambda m: "\u0131" if m.group(1) == "i" else "\u0237", out)
    out = _ACCENT_CMD_RE.sub(_accent, out)
    out = _ACCENT_SYM_RE.sub(_accent, out)
    out = _SPECIAL_RE.sub(lambda m: _SPECIAL[m.group(1)], out)
    out = out.replace(r"\{", _S_LBRACE).replace(r"\}", _S_RBRACE)
    out = re.sub(r"\\([&%$#_])", r"\1", out)
    out = out.replace("{", "").replace("}", "")
    if not keep_tilde:
        out = out.replace("~", " ")
    out = (out.replace(_S_LBRACE, "{").replace(_S_RBRACE, "}")
              .replace(_S_TILDE, "~").replace(_S_BSLASH, "\\").replace(_S_CARET, "^"))
    return _collapse(out) if not keep_tilde else out.strip()


# ── BibTeX：entry 切分 ──

_ENTRY_HEAD_RE = re.compile(r"@\s*([A-Za-z]+)\s*([{(])")
_SKIP_TYPES = {"comment", "preamble"}
# 整行注释：首个非空白字符是 %。只吃整行——值里的 `50%` 绝不能碰。
_COMMENT_LINE_RE = re.compile(r"^[ \t]*%[^\n]*", re.M)


def _blank_comment_lines(text: str) -> str:
    """按 BibTeX/biber 惯例丢弃整行 `%` 注释，但**保留换行**，让错误信息里的行号
    仍是用户文件里的真行号。

    不做这一步的话，用户在 .bib 里注释掉的整条 entry（`% @article{...}` 这种，
    Zotero/JabRef 用户常见）会被当成正常 entry 导进库里——导入了用户明确划掉的东西，
    比漏导更糟。
    """
    return _COMMENT_LINE_RE.sub("", text)


def _at_line_start(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] in " \t\r":
        j -= 1
    return j < 0 or text[j] == "\n"


def _line_no(text: str, i: int) -> int:
    return text.count("\n", 0, i) + 1


def _split_entries(text: str, errors: list[str]) -> list[tuple[str, str, int]]:
    """切成 (entry_type, body, 行号)。entry 之间的注释行、裸文本自然被跳过。

    容错的要害在这：扫描配对花括号时若在未闭合状态下撞见「行首且形如 @type{ 的 @」，
    就判定上一条缺右花括号——就地截断、记一条错误、从这个 @ 接着解析。
    否则一条坏 entry 会把文件剩余部分整个吞进自己的花括号里，后面全丢。
    """
    text = _blank_comment_lines(text)
    out: list[tuple[str, str, int]] = []
    i, n = 0, len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        m = _ENTRY_HEAD_RE.match(text, at)
        if not m:
            i = at + 1
            continue
        etype, open_ch = m.group(1).lower(), m.group(2)
        start = m.end()
        j, brace, in_quote, closed = start, 0, False, False
        while j < n:
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            if in_quote:
                if ch == '"':
                    in_quote = False
                j += 1
                continue
            if ch == '"' and brace == 0:
                in_quote = True
            elif ch == "{":
                brace += 1
            elif ch == "}":
                if open_ch == "{" and brace == 0:
                    closed = True
                    break
                brace = max(brace - 1, 0)
            elif ch == ")" and open_ch == "(" and brace == 0:
                closed = True
                break
            elif ch == "@" and _at_line_start(text, j) and _ENTRY_HEAD_RE.match(text, j):
                break
            j += 1
        if not closed:
            errors.append(f"第 {_line_no(text, at)} 行 @{etype} 缺少右花括号，"
                          f"已按下一条 entry 的起点截断（该条按已读到的字段尽力解析）")
        out.append((etype, text[start:j], _line_no(text, at)))
        i = j + 1 if closed else j
    return out


def _split_top(s: str, sep: str) -> list[str]:
    """按分隔符切分，但花括号组与双引号串内部的分隔符不算数。"""
    parts, buf, brace, in_quote = [], [], 0, False
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i:i + 2])
            i += 2
            continue
        if in_quote:
            buf.append(ch)
            if ch == '"':
                in_quote = False
            i += 1
            continue
        if ch == '"' and brace == 0:
            in_quote = True
            buf.append(ch)
        elif ch == "{":
            brace += 1
            buf.append(ch)
        elif ch == "}":
            brace = max(brace - 1, 0)
            buf.append(ch)
        elif ch == sep and brace == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _parse_value(raw: str, macros: dict[str, str]) -> str:
    """值的三种包裹 + `#` 串接 + @string 宏展开。裸词认不出宏就原样保留。"""
    out = []
    for piece in _split_top(raw, "#"):
        p = piece.strip()
        if len(p) >= 2 and p[0] == "{" and p[-1] == "}":
            out.append(p[1:-1])
        elif len(p) >= 2 and p[0] == '"' and p[-1] == '"':
            out.append(p[1:-1])
        elif p:
            out.append(macros.get(p.lower(), p))
    return "".join(out)


def _parse_body(body: str, macros: dict[str, str],
                errors: list[str], where: str) -> tuple[str, dict[str, str]]:
    """body 形如 `citekey, field = value, field = value,`。字段名一律小写。"""
    chunks = _split_top(body, ",")
    citekey = chunks[0].strip() if chunks else ""
    fields: dict[str, str] = {}
    for chunk in chunks[1:]:
        if not chunk.strip():
            continue
        if "=" not in chunk:
            errors.append(f"{where}：字段片段 {_collapse(chunk)[:40]!r} 没有等号，已跳过该字段")
            continue
        name, _, raw = chunk.partition("=")
        name = re.sub(r"[^a-z0-9]", "", name.strip().lower())
        if not name:
            errors.append(f"{where}：字段名为空，已跳过该字段")
            continue
        fields[name] = _parse_value(raw, macros)
    return citekey, fields


# ── 字段 → 统一 paper dict ──

_ARXIV_NEW = r"\d{4}\.\d{4,5}(?:v\d+)?"
_ARXIV_OLD = r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(" + _ARXIV_NEW + "|" + _ARXIV_OLD + ")",
                           re.I)
_ARXIV_TAG_RE = re.compile(r"arxiv\s*:\s*(" + _ARXIV_NEW + "|" + _ARXIV_OLD + ")", re.I)
_ARXIV_BARE_RE = re.compile(r"^\s*(?:arxiv:)?(" + _ARXIV_NEW + "|" + _ARXIV_OLD + r")\s*$", re.I)
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_DOI_PREFIX_RE = re.compile(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/", re.I)
_DOI_IN_TEXT_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"'{}<>]+)")

_VENUE_KEYS = ("journal", "journaltitle", "booktitle", "publisher",
               "school", "institution", "organization")
_WS_RE = re.compile(r"\s+")


def _unfold(s: str | None) -> str | None:
    """标识符（DOI / arXiv ID / URL）里不允许有空白。

    .bib 导出普遍在 80 列处折行，长 DOI/URL 会被拆成两行加缩进。留着那截
    「换行 + 空格」会让 norm_key 变成 `doi:10.1234/very-long\n   -id`——
    同一篇论文折行版与不折行版算成两条，去重（本模块的核心承诺）当场失效；
    oa_pdf_url 里带换行则下游 fulltext 一下就挂。
    """
    if not s:
        return None
    return _WS_RE.sub("", s) or None


def _text(v) -> str | None:
    s = _latex_unescape(v or "")
    return s or None


def _plain(v) -> str | None:
    """URL/DOI 用：剥花括号与转义标点，但不动 `~`、不压缩内部空白后再返回。"""
    s = _latex_unescape(v or "", keep_tilde=True)
    s = s.strip().rstrip(".,;")
    return s or None


def _year(v) -> int | None:
    m = _YEAR_RE.search(str(v or ""))
    return int(m.group(1)) if m else None


def _norm_author(raw: str) -> str | None:
    """统一成 "First Last"。`others`（BibTeX 的 et-al 标记）不是人名，丢弃。"""
    a = _collapse(raw)
    if not a or a.lower() in ("others", "et al", "et al.", "and others"):
        return None
    parts = [p.strip() for p in a.split(",")]
    if len(parts) == 1:
        return a
    if len(parts) == 2:
        last, first = parts
        return _collapse(f"{first} {last}") if first else (last or None)
    # "von Last, Jr, First" 三段式
    return _collapse(f"{' '.join(parts[2:])} {parts[0]} {parts[1]}") or None


def _split_authors(raw: str) -> list[str]:
    """BibTeX 的 ` and ` 分隔（花括号组内的 and 不算），逐个规范化。"""
    if not raw:
        return []
    flat = re.sub(r"\s+", " ", raw)  # 作者串常跨行折行，先拉平再切
    # BibTeX 的 ` and ` 分隔符大小写不敏感（带花括号那条路径本来就 .lower() 了，
    # 这条不加 re.I 会让 "Alice Smith AND Bob Lee" 变成一个人）
    chunks = (_split_and_braced(flat) if "{" in flat
              else re.split(r"\s+and\s+", flat, flags=re.I))
    return [n for n in (_norm_author(_latex_unescape(ch)) for ch in chunks) if n]


def _split_and_braced(raw: str) -> list[str]:
    """带花括号时按深度切 ` and `——`{Smith and Sons Lab}` 是一个机构名，不能切开。"""
    parts, buf, brace = [], [], 0
    i, n = 0, len(raw)
    while i < n:
        if brace == 0 and raw[i:i + 5].lower() == " and ":
            parts.append("".join(buf))
            buf = []
            i += 5
            continue
        ch = raw[i]
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(brace - 1, 0)
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _find_arxiv(*texts: str | None) -> str | None:
    for t in texts:
        if not t:
            continue
        m = _ARXIV_URL_RE.search(t) or _ARXIV_TAG_RE.search(t) or _ARXIV_BARE_RE.match(t)
        if m:
            return m.group(1)
    return None


def _pdf_url(url: str | None) -> str | None:
    """只认「看起来真是 PDF」的链接。Zotero 的 Url 多是落地页，塞进 oa_pdf_url
    会让下游 fulltext 下到一堆 HTML——宁可留空。"""
    url = _unfold(url)
    if not url:
        return None
    low = url.lower()
    if low.endswith(".pdf") or "/pdf/" in low or low.endswith("/pdf"):
        return url
    return None


def _make_paper(*, title, abstract=None, year=None, venue=None, authors=None,
                doi=None, arxiv_id=None, oa_pdf_url=None, source="bibtex") -> dict:
    # 三种格式的 doi/arxiv_id 都在这里收口：先去前缀再去折行留下的空白，
    # 保证 norm_key 只由「干净的标识符」算出来
    if doi:
        doi = _unfold(_DOI_PREFIX_RE.sub("", doi))
    if arxiv_id:
        arxiv_id = _unfold(re.sub(r"^arxiv:", "", arxiv_id.strip(), flags=re.I))
    return {
        "norm_key": norm_key(doi=doi, arxiv_id=arxiv_id, title=title),
        "title": title, "abstract": abstract, "year": year, "venue": venue,
        "authors": authors or [], "doi": doi, "arxiv_id": arxiv_id,
        "citation_count": None, "oa_pdf_url": oa_pdf_url,
        "source": source, "s2_id": None,
    }


def _fields_to_paper(f: dict[str, str]) -> dict | None:
    title = _text(f.get("title"))
    if not title:
        return None
    url = _plain(f.get("url") or f.get("howpublished"))
    doi = _plain(f.get("doi"))
    if not doi:
        for cand in (url, f.get("note"), f.get("eprint")):
            m = _DOI_IN_TEXT_RE.search(_plain(cand) or "")
            if m:
                doi = m.group(1)
                break
    eprint = _plain(f.get("eprint"))
    prefix = (f.get("archiveprefix") or f.get("eprinttype") or "").lower()
    arxiv = None
    if eprint and ("arxiv" in prefix or _ARXIV_BARE_RE.match(eprint)):
        arxiv = re.sub(r"^arxiv:", "", eprint, flags=re.I)
    if not arxiv:
        arxiv = _find_arxiv(url, f.get("note"), f.get("journal"), f.get("howpublished"))
    venue = next((_text(f[k]) for k in _VENUE_KEYS if f.get(k) and _text(f[k])), None)
    return _make_paper(
        title=title, abstract=_text(f.get("abstract")),
        year=_year(f.get("year") or f.get("date")), venue=venue,
        authors=_split_authors(f.get("author") or f.get("editor") or ""),
        doi=doi, arxiv_id=arxiv, oa_pdf_url=_pdf_url(url), source="bibtex")


def parse_bibtex(text: str, errors: list[str] | None = None) -> list[dict]:
    """手写 BibTeX 解析。返回统一 paper dict 列表；坏 entry 记进 errors 并继续。

    errors 是可选出参（传一个 list 进来收集）；不传则解析问题静默丢弃——
    上层 import_text 一定会传，保证错误可见。
    """
    errs = errors if errors is not None else []
    macros: dict[str, str] = {}
    out: list[dict] = []
    for etype, body, line in _split_entries(text or "", errs):
        where = f"第 {line} 行 @{etype}"
        if etype in _SKIP_TYPES:
            continue
        if etype == "string":
            # @string{key = "value"}：宏体没有 citekey，整段就是一个赋值
            name, _, raw = body.partition("=")
            name = name.strip().lower()
            if name and raw:
                macros[name] = _parse_value(raw, macros)
            continue
        try:
            _key, fields = _parse_body(body, macros, errs, where)
            paper = _fields_to_paper(fields)
        except Exception as e:  # 单条崩溃不许波及整份文件
            errs.append(f"{DROP_TAG}{where} 解析异常：{type(e).__name__}: {e}")
            continue
        if not paper:
            errs.append(f"{DROP_TAG}{where}：没有 title 字段，无法作为文献入库，已跳过")
            continue
        if not paper["norm_key"]:
            errs.append(f"{DROP_TAG}{where}：标题无法归一化出主键，已跳过")
            continue
        out.append(paper)
    return out


# ── RIS ──

# 标签行必须是「两字符标签 + 至少一个空格 + 短横」，否则 "de-novo ..." 这种
# 未缩进的续行会被误认成 DE 标签，把正文吃进关键词字段。
_RIS_TAG_RE = re.compile(r"^([A-Za-z][A-Za-z0-9])\s{1,4}-\s?(.*)$")
_RIS_TITLE = ("TI", "T1", "BT", "CT")
_RIS_AUTHOR = ("AU", "A1", "A2", "A3", "A4")
_RIS_VENUE = ("JO", "JF", "JA", "J1", "J2", "T2", "T3", "BT", "PB")
_RIS_ABSTRACT = ("AB", "N2")


def _ris_record_to_paper(rec: dict[str, list[str]]) -> dict | None:
    def first(keys):
        for k in keys:
            for v in rec.get(k, []):
                if v.strip():
                    return _collapse(v)
        return None

    title = first(_RIS_TITLE)
    if not title:
        return None
    urls = rec.get("UR", []) + rec.get("L1", []) + rec.get("L2", []) + rec.get("LK", [])
    doi = first(("DO", "DOI"))
    if not doi:
        for u in urls:
            m = _DOI_IN_TEXT_RE.search(u)
            if m:
                doi = m.group(1)
                break
    authors = [a for a in (_norm_author(v) for k in _RIS_AUTHOR for v in rec.get(k, [])) if a]
    pdf = next((_pdf_url(u.strip()) for u in urls if _pdf_url(u.strip())), None)
    venue = first(_RIS_VENUE)
    if venue == title:
        venue = None  # BT 既可能是书名（=title）也可能是丛书名，同值时不重复填
    return _make_paper(
        title=title, abstract=first(_RIS_ABSTRACT),
        year=_year(first(("PY", "Y1", "DA", "Y2"))), venue=venue,
        authors=authors, doi=doi,
        arxiv_id=_find_arxiv(*urls, first(("N1",)), first(("C1",))),
        oa_pdf_url=pdf, source="ris")


def parse_ris(text: str, errors: list[str] | None = None) -> list[dict]:
    """RIS 解析。一个字段可跨多行（续行缩进），ER 结束一条记录。

    没有 ER 的收尾记录照样收下（很多导出工具最后一条漏 ER），但会记一条 errors。
    """
    errs = errors if errors is not None else []
    out: list[dict] = []
    rec: dict[str, list[str]] = {}
    last_tag: str | None = None
    start_line = 1
    saw_er = True

    def flush(line_no: int, complete: bool):
        nonlocal rec, last_tag
        if not rec:
            rec, last_tag = {}, None
            return
        where = f"RIS 第 {start_line}-{line_no} 行记录"
        if not complete:
            errs.append(f"{where}：缺少 ER 结束标记，已按文件结尾收尾")
        try:
            paper = _ris_record_to_paper(rec)
        except Exception as e:
            errs.append(f"{DROP_TAG}{where} 解析异常：{type(e).__name__}: {e}")
            rec, last_tag = {}, None
            return
        if not paper:
            errs.append(f"{DROP_TAG}{where}：没有 TI/T1 标题字段，已跳过")
        elif not paper["norm_key"]:
            errs.append(f"{DROP_TAG}{where}：标题无法归一化出主键，已跳过")
        else:
            out.append(paper)
        rec, last_tag = {}, None

    for i, line in enumerate((text or "").splitlines(), start=1):
        if not line.strip():
            continue
        m = _RIS_TAG_RE.match(line)
        if m:
            tag, val = m.group(1).upper(), m.group(2).strip()
            if tag == "ER":
                flush(i, True)
                saw_er = True
                continue
            if tag == "TY" and rec:
                # 上一条还没等到 ER 就又来了一个 TY：先把它收尾。不这么做的话，
                # 两条记录会被 setdefault 合并成一条（TI 有两个值、first() 只取第一个），
                # 第二篇论文**静默消失**且 errors 里一个字都没有——比报错糟得多。
                flush(i - 1, False)
            if not rec:
                start_line, saw_er = i, False
            rec.setdefault(tag, []).append(val)
            last_tag = tag
        elif last_tag and rec:
            # 续行：接到上一个字段末尾（RIS 的续行以空格开头，但也容忍不缩进的）
            rec[last_tag][-1] = _collapse(rec[last_tag][-1] + " " + line)
        else:
            errs.append(f"RIS 第 {i} 行不是合法标签行且没有可续接的字段，已忽略：{line.strip()[:40]!r}")
    if rec:
        flush(len((text or "").splitlines()), saw_er)
    return out


# ── Zotero CSV ──

_CSV_TITLE = ("title",)
_CSV_YEAR = ("publicationyear", "year", "date")
_CSV_AUTHOR = ("author", "authors", "creator", "creators")
_CSV_VENUE = ("publicationtitle", "publication", "journal", "journalabbreviation",
              "proceedingstitle", "booktitle", "conferencename", "publisher")
_CSV_ABSTRACT = ("abstractnote", "abstract", "note")
_CSV_URL = ("url", "link", "attachmenturl")


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _split_csv_authors(raw: str) -> list[str]:
    """Zotero 用 `; ` 分隔，每个是 "Last, First"；没有分号时退回 ` and `。"""
    if not raw:
        return []
    chunks = raw.split(";") if ";" in raw else re.split(r"\s+and\s+", raw, flags=re.I)
    return [a for a in (_norm_author(ch) for ch in chunks) if a]


def parse_zotero_csv(text: str, errors: list[str] | None = None) -> list[dict]:
    """Zotero CSV 导出。列名大小写/空格/下划线一律容错（归一化成纯小写字母数字）。"""
    errs = errors if errors is not None else []
    out: list[dict] = []
    if not (text or "").strip():
        return out
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    for i, raw_row in enumerate(reader, start=2):  # 2 = 表头之后的第一行
        try:
            row = {_norm_col(k): (v or "") for k, v in raw_row.items() if k}

            def pick(keys, _row=row):
                return next((_row[k].strip() for k in keys if _row.get(k, "").strip()), None)

            title = pick(_CSV_TITLE)
            if not title:
                errs.append(f"{DROP_TAG}CSV 第 {i} 行没有 Title 列或标题为空，已跳过")
                continue
            url = pick(_CSV_URL)
            extra = row.get("extra", "")
            doi = pick(("doi",))
            if not doi:
                m = _DOI_IN_TEXT_RE.search(f"{url or ''} {extra}")
                doi = m.group(1) if m else None
            paper = _make_paper(
                title=_collapse(title), abstract=pick(_CSV_ABSTRACT),
                year=_year(pick(_CSV_YEAR)), venue=pick(_CSV_VENUE),
                authors=_split_csv_authors(pick(_CSV_AUTHOR) or ""),
                doi=doi, arxiv_id=_find_arxiv(url, extra, row.get("itemtype")),
                oa_pdf_url=_pdf_url(url), source="zotero-csv")
        except Exception as e:
            errs.append(f"{DROP_TAG}CSV 第 {i} 行解析异常：{type(e).__name__}: {e}")
            continue
        if not paper["norm_key"]:
            errs.append(f"{DROP_TAG}CSV 第 {i} 行标题无法归一化出主键，已跳过")
            continue
        out.append(paper)
    return out


# ── Semantic Scholar 补全（可选，失败不影响导入主流程）──

S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/"
S2_FIELDS = ("paperId,title,abstract,year,venue,authors,citationCount,"
             "openAccessPdf,externalIds")
# 与 sources/semantic_scholar.py 同调：无 key 共享池按 5 分钟窗口限流，秒级退避没用
ENRICH_DELAYS = (20, 45, 90)
_RETRY_STATUS = (429, 500, 502, 503, 504)


def _sleep(sec: float):
    time.sleep(sec)


def _jitter() -> float:
    import random
    return 0.8 + random.random() * 0.4


def _safe_ident(ident: str) -> str:
    """标识符要拼进 URL 路径，而它来自用户上传的 .bib —— 必须先消毒。

    裸拼的三种坏法（都实测过）：
      `10.x/a?b=c`  → `?` 之后被当成 query，查的其实是别的东西；
      `10.x/a#f`    → `#` 之后被当成 fragment，同上；
      `../../admin` → httpx 会归一化路径段，请求打到 `graph/v1/admin`。
    三种都会把「另一个端点的返回」当成这篇论文的元数据写进用户的库——
    那就是项目红线里的「编造」。宿主固定在 S2 域名下，所以不是任意 SSRF，
    但张冠李戴的元数据同样不能容忍。
    """
    if any(seg in ("..", ".") for seg in ident.split("/")):
        raise ValueError(f"标识符含 '..' 路径段，拒绝拼进请求 URL：{ident!r}")
    return quote(ident, safe=":/")  # `:` `/` 是 S2 标识符的正常写法，其余一律转义


def _s2_get(ident: str) -> dict | None:
    """按 `DOI:xxx` / `arXiv:xxx` 查单篇。404 返回 None（S2 没收录，不算错误）。"""
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    url = S2_PAPER_URL + _safe_ident(ident)
    last = None
    with http.client() as client:
        for attempt in range(len(ENRICH_DELAYS) + 1):
            try:
                r = client.get(url, params={"fields": S2_FIELDS}, headers=headers)
            except httpx.HTTPError as e:
                last = e
            else:
                if r.status_code == 404:
                    return None
                if r.status_code not in _RETRY_STATUS:
                    r.raise_for_status()
                    return r.json()
                last = f"HTTP {r.status_code}"
            if attempt < len(ENRICH_DELAYS):
                _sleep(ENRICH_DELAYS[attempt] * _jitter())
    raise RuntimeError(f"Semantic Scholar 单篇查询连续 {len(ENRICH_DELAYS) + 1} 次失败：{last}")


def enrich_entries(entries: list[dict]) -> list[str]:
    """给「有 doi/arxiv_id 但缺 abstract」的条目补摘要，返回错误说明列表。

    只填条目里本来就空的字段——用户 .bib 里的人工修订优先于外部源。
    补到 doi/arxiv_id 后会重算 norm_key，让同一篇在库里仍只占一个键。
    """
    errs: list[str] = []
    for e in entries:
        if e.get("abstract"):
            continue
        ident = (f"DOI:{e['doi']}" if e.get("doi")
                 else f"arXiv:{e['arxiv_id']}" if e.get("arxiv_id") else None)
        if not ident:
            continue
        try:
            data = _s2_get(ident)
        except Exception as ex:
            errs.append(f"补全 {ident} 失败（已跳过，不影响导入）：{type(ex).__name__}: {ex}")
            continue
        if not data:
            errs.append(f"补全 {ident}：Semantic Scholar 未收录，摘要保持为空")
            continue
        ext = data.get("externalIds") or {}
        oa = data.get("openAccessPdf") or {}
        for key, val in (("abstract", data.get("abstract")),
                         ("year", data.get("year")),
                         ("venue", data.get("venue") or None),
                         ("citation_count", data.get("citationCount")),
                         ("oa_pdf_url", oa.get("url")),
                         ("s2_id", data.get("paperId")),
                         ("doi", ext.get("DOI")),
                         ("arxiv_id", ext.get("ArXiv"))):
            if val not in (None, "") and not e.get(key):
                e[key] = val
        if not e.get("authors"):
            e["authors"] = [a.get("name") for a in data.get("authors") or [] if a.get("name")]
        e["norm_key"] = norm_key(doi=e.get("doi"), arxiv_id=e.get("arxiv_id"),
                                 title=e.get("title")) or e.get("norm_key")
    return errs


# ── 入库 ──

# 「补空不覆盖」允许补的列。title 不在内：它是 NOT NULL 且必然有值，
# 用导入文件里的另一种写法覆盖库内标题只会制造无谓的漂移。
_MERGE_FIELDS = ("abstract", "year", "venue", "doi", "arxiv_id",
                 "citation_count", "oa_pdf_url", "s2_id")


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _refresh_fts(c, pid: int):
    """补空后重建该行的 FTS 索引；keywords 从已有卡片里取回，别把它洗没了。"""
    row = c.execute("SELECT title, abstract, card_json FROM papers WHERE id=?",
                    (pid,)).fetchone()
    if not row:
        return
    keywords = ""
    if row["card_json"]:
        try:
            card = json.loads(row["card_json"])
            keywords = json.dumps((card.get("keywords") if isinstance(card, dict) else None)
                                  or [], ensure_ascii=False)
        except Exception:
            # 只 catch ValueError 不够：card_json 若是 JSON 数组，
            # `.get` 会抛 AttributeError，把整条「补空」变成入库失败
            keywords = ""
    c.execute("DELETE FROM papers_fts WHERE rowid=?", (pid,))
    c.execute("INSERT INTO papers_fts(rowid,title,abstract,keywords) VALUES(?,?,?,?)",
              (pid, row["title"], row["abstract"] or "", keywords))


def _merge_fill(c, row, e: dict) -> list[str]:
    """补空不覆盖：只把库里为空的列用新条目填上，返回实际补了哪些列。"""
    sets, params, filled = [], [], []
    for f in _MERGE_FIELDS:
        if _is_empty(row[f]) and not _is_empty(e.get(f)):
            sets.append(f"{f}=?")
            params.append(e[f])
            filled.append(f)
    try:
        old_authors = json.loads(row["authors"] or "[]")
    except Exception:
        old_authors = []
    if not old_authors and e.get("authors"):
        sets.append("authors=?")
        params.append(json.dumps(e["authors"], ensure_ascii=False))
        filled.append("authors")
    if not sets:
        return []
    if "abstract" in filled:
        # 与 db.insert_l0 同一套语义：有摘要即算 L0→L1 的下限，不要因为入口不同而分叉
        sets.append("level=MAX(level,1)")
    sets.append("updated_at=datetime('now','localtime')")
    c.execute(f"UPDATE papers SET {', '.join(sets)} WHERE id=?", params + [row["id"]])
    if "abstract" in filled:
        _refresh_fts(c, row["id"])
    return filled


def _pending_enrich(entries: list[dict]) -> list[dict]:
    """筛掉「库里已经有摘要」的条目，剩下的才值得为它去问 S2。

    补全跑在去重之前，所以重复导入同一份 .bib 时，每条都会再问一次 S2 —— 而拿回来的
    摘要马上又被「补空不覆盖」丢掉（库里那列不空）。S2 共享池按 5 分钟窗口限流、
    退避一档就是 20/45/90s，几百条白问一遍能把一次导入拖到以小时计。
    """
    out: list[dict] = []
    try:
        with db.conn() as c:
            for e in entries:
                key = e.get("norm_key")
                row = db.get_by_norm_key(c, key) if key else None
                if row is None or _is_empty(row["abstract"]):
                    out.append(e)
    except Exception:
        return list(entries)  # 查不动就退回原行为，别因为一次优化把补全整个关掉
    return out


def import_entries(entries: list[dict], enrich: bool = False,
                   fmt: str = "", source_name: str = "") -> dict:
    """把统一 paper dict 列表写库。按 norm_key 去重，已存在的走「补空不覆盖」。

    返回 {"imported","skipped","updated","failed","errors","paper_ids","updated_ids","filled"}。
    paper_ids 只含**新入库**的 id；被补空的老记录 id 在 updated_ids 里。
    """
    _ensure()
    errors: list[str] = []
    imported, skipped, failed = 0, 0, 0
    paper_ids: list[int] = []
    updated_ids: list[int] = []
    filled_log: list[dict] = []

    if enrich:
        try:
            errors.extend(enrich_entries(_pending_enrich(entries)))
        except Exception as e:  # 补全整体崩了也不许拖垮导入
            errors.append(f"补全阶段整体失败（已跳过，不影响导入）：{type(e).__name__}: {e}")

    for e in entries:
        title = (e.get("title") or "").strip()
        key = e.get("norm_key")
        if not title or not key:
            failed += 1
            errors.append(f"条目缺少 title 或 norm_key，无法入库：{str(e)[:120]}")
            continue
        try:
            with db.conn() as c:
                row = db.get_by_norm_key(c, key)
                if row:
                    got = _merge_fill(c, row, e)
                    skipped += 1
                    if got:
                        updated_ids.append(row["id"])
                        filled_log.append({"paper_id": row["id"], "fields": got,
                                           "title": row["title"]})
                else:
                    pid = db.insert_l0(c, e)
                    imported += 1
                    paper_ids.append(pid)
        except Exception as ex:
            failed += 1
            errors.append(f"入库失败 {title[:60]!r}：{type(ex).__name__}: {ex}")

    updated = len(updated_ids)
    try:
        with db.conn() as c:
            c.execute("""INSERT INTO import_runs(fmt,source_name,total,imported,skipped,
                         updated,failed,errors_json) VALUES(?,?,?,?,?,?,?,?)""",
                      (fmt, source_name, len(entries), imported, skipped, updated, failed,
                       json.dumps(errors, ensure_ascii=False)))
    except Exception as ex:  # 审计流水写失败不该让已成功的导入显示为失败
        errors.append(f"导入流水记录写入失败（导入结果本身有效）：{type(ex).__name__}: {ex}")

    return {"imported": imported, "skipped": skipped, "updated": updated,
            "failed": failed, "errors": errors, "paper_ids": paper_ids,
            "updated_ids": updated_ids, "filled": filled_log}


# ── 顶层入口 ──

PARSERS = {"bibtex": parse_bibtex, "ris": parse_ris, "csv": parse_zotero_csv}

_RIS_SNIFF_RE = re.compile(r"^TY\s{1,4}-\s", re.M)  # 与 _RIS_TAG_RE 同宽，嗅探到就一定解析得动
_BIB_SNIFF_RE = re.compile(r"^\s*@[A-Za-z]+\s*[{(]", re.M)


def sniff_format(text: str) -> str:
    """按内容嗅探格式，认不出返回 "unknown"（不猜、不默认成 bibtex）。"""
    t = (text or "").lstrip("\ufeff")
    if _RIS_SNIFF_RE.search(t):
        return "ris"
    if _BIB_SNIFF_RE.search(t):
        return "bibtex"
    head = t.splitlines()[0] if t.strip() else ""
    cols = {_norm_col(c) for c in head.split(",")}
    if "title" in cols and len(cols) >= 2:
        return "csv"
    return "unknown"


def import_text(text: str, fmt: str = "auto", enrich: bool = False,
                source_name: str = "") -> dict:
    """顶层入口：嗅探/指定格式 → 解析 → 入库。解析失败的条目计入 failed 与 errors。"""
    _ensure()
    resolved = sniff_format(text) if fmt == "auto" else fmt.lower()
    parser = PARSERS.get(resolved)
    if not parser:
        raise ValueError(
            f"无法识别的导入格式：{resolved!r}（可选 {list(PARSERS)}；"
            "auto 嗅探依据：'TY  - ' → ris，行首 '@type{' → bibtex，首行含 Title 列 → csv）")
    parse_errors: list[str] = []
    entries = parser(text, parse_errors)
    if not entries and (text or "").strip():
        # 静默的 0 结果会被 UI 显示成「导入成功，0 篇」。格式选错（把 .bib 按 csv 解析）
        # 正是走到这里，必须说出来。
        parse_errors.append(
            f"按 {resolved} 格式解析后一条文献都没得到——多半是格式选错了，"
            "可改用 fmt='auto' 让它按内容嗅探")
    res = import_entries(entries, enrich=enrich, fmt=resolved, source_name=source_name)
    res["format"] = resolved
    res["parsed"] = len(entries)
    # 解析阶段**真的丢掉了条目**才计进 failed。带 DROP_TAG 的才算，
    # 「缺右花括号但仍解析出来了」「跳过一个没等号的字段」只是警告——
    # 早先按 len(parse_errors) 计数，3 条全部成功导入的文件会报成 failed=3。
    res["failed"] += sum(1 for e in parse_errors if e.startswith(DROP_TAG))
    res["errors"] = parse_errors + res["errors"]
    return res


def list_imports(limit: int = 20) -> list[dict]:
    """导入流水（审计用）。"""
    _ensure()
    with db.conn() as c:
        rows = c.execute("SELECT * FROM import_runs ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["errors"] = json.loads(d.pop("errors_json") or "[]")
        except Exception:
            d["errors"] = []
        out.append(d)
    return out
