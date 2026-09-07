"""PDF 版面结构解析：字号/字体 + 编号启发式识别章节，按章节切 chunk，抽参考文献。

为什么不上 GROBID：GROBID 要额外跑一个 Java 服务（JVM + 几百 MB 模型），对个人
自用工具太重。本模块只用 PyMuPDF 已有的 span 信息（字号 / 字体 / 坐标）加上编号与
关键词启发式，**零新依赖、0 token、离线可跑**，拿到 GROBID 的两块主要收益：
① 按章节切 chunk（替代 fulltext.extract_pages 的按页切，检索与引用定位更准）；
② 从 References 段回收 DOI / arXiv id（本地上传的 PDF 因此能接进 graph.py 的引文图）。

**明确不等价于 GROBID**（能力边界写在这里，不在别处粉饰）：
- 双栏排版按 MuPDF 的自然块序输出，列交错时章节归属可能错乱；
- 公式区、表格、图注不做版面分类，会混进正文 chunk（只对最常见的 Figure/Table
  题注做了标题误判排除）；
- 作者—机构对应、引用位置（in-text citation）与参考文献的对齐一概不做；
- 扫描件（无文本层）拿不到任何 span，一律降级为「按页切」并如实标 degraded。

失败策略：除「PDF 根本打不开」抛 StructureError 外，任何一步失败都降级返回，
并把原因写进 degraded / warnings，绝不静默吞掉，也绝不用编造的结构填充。
"""
import math
import re
import unicodedata

from .normalize import norm_key as _norm_key

# ── 常量与阈值（集中在这里，便于按语料调参）──

#: 标题候选的字符数上限。标题通常独占一行且短，超过就当正文。
MAX_HEADING_CHARS = 80
#: 标题候选允许跨的行数（长标题会折行，但不会折成一整段）。
MAX_HEADING_LINES = 2
#: 判定「字号显著大于正文」的双阈值：既要按比例大，也要按绝对值大。
#: 只用比例，正文 8pt 时 9pt 也能过；只用绝对值，正文 18pt 的大字排版会全军覆没。
HEADING_SIZE_RATIO = 1.12
HEADING_SIZE_DELTA = 0.6
#: 判定为标题所需的最低分（评分见 _score_heading）。
#: 取 3 而不是 2，是为了让「单个信号」永远不够：版面信号各值 2 分，文本信号各值 1 分，
#: 所以 3 分意味着至少「一个版面信号 + 一个文本信号」或「两个版面信号」。
#: 实测（26 页真实综述）：门槛 2 时识别出 125 个「章节」，其中二十余个是
#: 分类图里又粗又短的图元标签（Self-Learning / Centralized / Memory Mechanism…），
#: 它们只命中 bold 一个信号就达标了——而 bold 在图表标签、表头、行内强调里遍地都是。
HEADING_MIN_SCORE = 3
#: 同一段文字在 >= 该页数上出现 → 判为页眉/页脚，排除出标题候选
REPEAT_PAGE_THRESHOLD = 3
#: 页眉/页脚还必须出现在「足够高比例的页」上。只看绝对页数会把
#: "Chapter 1"/"Chapter 2"/… 这类**同模板不同编号的真章节标题**整批当页眉删掉
#: （归一化页码后它们的 key 完全相同）。真页眉几乎每页都有，真章节标题不会。
REPEAT_PAGE_RATIO = 0.4

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WS_RE = re.compile(r"[\s\u00a0\u2000-\u200b]+")

# 章节编号：阿拉伯多级 / 罗马 / 中文「第X章|节」/ 附录字母
_NUM_ARABIC_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.、]?\s+\S")
_NUM_ROMAN_RE = re.compile(r"^([IVXL]{1,5})[.、]\s+\S")
_NUM_CJK_RE = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]{1,4})\s*([章节節部分篇])")
_NUM_APPENDIX_RE = re.compile(r"^(?:appendix|附录)\s*([A-Z\d一二三四五六七八九十]{0,3})\b", re.I)

# 关键词表：命中后正文分句仍要过「句子样」否决，见 _looks_like_sentence
_KEYWORDS_EN = (
    "abstract", "introduction", "background", "related work", "related works",
    "related literature", "preliminaries", "problem formulation", "problem statement",
    "system model", "method", "methods", "methodology", "approach", "proposed method",
    "materials and methods", "experiment", "experiments", "experimental setup",
    "experimental results", "evaluation", "results", "results and discussion",
    "discussion", "analysis", "ablation", "ablation study", "limitations",
    "conclusion", "conclusions", "future work", "references", "bibliography",
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "appendix", "supplementary material",
)
_KEYWORDS_ZH = (
    "摘要", "引言", "绪论", "前言", "研究背景", "相关工作", "研究现状", "预备知识",
    "问题描述", "系统模型", "方法", "本文方法", "所提方法", "实验", "实验设置",
    "实验结果", "结果", "讨论", "分析", "消融", "局限", "结论", "总结与展望",
    "总结", "参考文献", "致谢", "附录",
)

# 图表题注 / 定理环境：常常又粗又短，不排除会被当成章节标题。
# 定理环境（Theorem 1 / Lemma 2 / 定理 3）在数学味重的论文里逐页都有，
# 漏掉它们会把一节正文切成十几个假章节——这条比图表题注更常见。
_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?|algorithm|alg\.?|listing|eq\.?|equation"
    r"|theorem|lemma|corollary|proposition|definition|remark|example|assumption"
    r"|claim|conjecture|observation|property|proof|axiom"
    r"|图|表|算法|公式|定理|引理|推论|命题|定义|假设|性质|证明|例)"
    r"\s*\.?\s*[\dIVX]+\s*[.:：、]?", re.I)

# 参考文献段起始（整行只有标题）与「标题和第一条挤在同一行」两种情形
_REF_HEAD_RE = re.compile(
    r"^\s*(?:\d{1,2}[.、]?\s*|[IVXL]{1,5}[.、]\s*)?"
    r"(references?|reference list|bibliography|literature cited|参\s*考\s*文\s*献|引用文献)"
    r"\s*[:：]?\s*$", re.I)
_REF_HEAD_INLINE_RE = re.compile(
    r"^\s*(?:\d{1,2}[.、]?\s*)?"
    r"(?:references?|bibliography|参\s*考\s*文\s*献)\s*[:：]?\s+(?=\S)", re.I)
_REF_END_RE = re.compile(r"^\s*(appendix|supplementary\s+material|附\s*录|补充材料)\b", re.I)

# 条目切分标记
_MARK_BRACKET_RE = re.compile(r"^\s*\[(\d{1,3})\]\s*")
_MARK_NUMDOT_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+")
_MARK_BRACKET_ANY_RE = re.compile(r"\[(\d{1,3})\]\s*")

# 标识符正则（与 pdfimport 保持一致的写法，避免两处行为分叉）
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
# 参考文献里的 arXiv 除了 "arXiv:2401.12345"，还有大量条目只写 URL
# （https://arxiv.org/abs/2401.12345、arxiv.org/pdf/1706.03762v2）。
# 只认冒号写法会白白丢掉一大批本可以接进引文图的标识符。
_ARXIV_LEAD = r"arxiv(?:[:\s]*|\.org/(?:abs|pdf|format)/)\s*"
ARXIV_NEW_RE = re.compile(_ARXIV_LEAD + r"(\d{4}\.\d{4,5})(v\d+)?", re.I)
ARXIV_OLD_RE = re.compile(_ARXIV_LEAD + r"([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# 折行修复：只在「行尾看起来是被截断的标识符」时才无缝粘回去。
# 不能无条件粘：DOI 已经写完、下一行是新作者名时粘回去会造出 10.1234/abc.567Smith。
#
# 截断分两档，因为「行尾是 /」和「行尾是 .」的确定性完全不同：
#   硬截断：停在 "doi:"、"doi.org/"、DOI 内部的 / - _ 上 —— 后半截必然在下一行，
#           续行只要以数字或小写字母开头就粘。
#   软截断：DOI 以 "." 收尾 —— 既可能是折行，也可能只是条目的句号。这一档只在
#           续行「长得像标识符后半截」（≥4 位连续数字，且不是 1234-1245 这种页码
#           范围）时才粘。否则会造出 10.1109/tsp.2021.123456.pp（接了 "pp. 1234-"）、
#           10.1000/xyz123.4（接了下一条 "4. Wang..."）这类假 DOI —— 它们会当作
#           norm_key 写进引文图，比抽不出来有害得多。
_DOI_TRUNC_HARD_RE = re.compile(
    r"(?:(?:https?://)?(?:dx\.)?doi\.org/|doi\s*[:：]?\s*)$"
    r"|10\.\d{4,9}/(?:[-._;()/:A-Za-z0-9]*[/\-_])?$", re.I)
_DOI_TRUNC_SOFT_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]*\.$", re.I)
_ARXIV_TRUNC_RE = re.compile(r"arxiv\s*(?:[:：]\s*|\s+)(?:\d{4}\.?\d{0,3})?$", re.I)
# 续行要以数字或小写字母开头才可能是标识符的后半截；下一行是 "Smith, J." 这种
# 大写开头的新句子时坚决不粘——粘了就是造一个 10.1109/tsp.2023.123456.Smith 出来。
_CONT_HARD_RE = re.compile(r"^[0-9a-z]")
_CONT_SOFT_RE = re.compile(r"^\d{4,}(?!\s*[-–—])")


class StructureError(ValueError):
    """PDF 打不开 / 加密到读不出任何内容。其余情况一律降级，不抛异常。"""


# ── 基础工具 ──

def _clean(s: str | None) -> str:
    """NFKC 归一（连字 ﬁ→fi、全角→半角）+ 去软连字符 + 压缩空白。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00ad", "")
    return _WS_RE.sub(" ", s).strip()


def _token_count(t: str) -> int:
    """粗略词数：中文按「两字一词」折算，用于「像句子还是像标题」的判断。"""
    cjk = len(_CJK_RE.findall(t))
    latin = len([w for w in re.sub(r"[\u4e00-\u9fff]", " ", t).split() if w])
    return latin + cjk // 2


#: 嵌入子集字体名前缀（"ABCDEF+NimbusRomNo9L-Medi"）。前缀是随机六字母，
#: 不剥掉就可能在里面撞出 "bd"/"bx" 之类的假粗体证据。
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
#: LaTeX Computer Modern 系的粗体字体名（cmbx12 / sfbx1000 / cmssbx10）里
#: 没有 "bold" 字样。arXiv 上的 pdfTeX 论文大量使用，漏掉它=IEEE 式标题全失效。
_CM_BOLD_RE = re.compile(r"bx\d")


def _is_bold(span: dict) -> bool:
    """PyMuPDF 的 flags bit4(=16) 是粗体位；嵌入子集字体常常只能从名字看出来。"""
    if int(span.get("flags") or 0) & 16:
        return True
    font = _SUBSET_PREFIX_RE.sub("", span.get("font") or "").lower()
    if any(k in font for k in ("bold", "black", "heavy", "semib", "-bd", "medi")):
        return True
    return bool(_CM_BOLD_RE.search(font))


def _open(pdf_path):
    import pymupdf
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:  # 文件不存在 / 不是 PDF / 结构损坏
        raise StructureError(f"PDF 打不开：{e}") from e
    if getattr(doc, "needs_pass", False):
        doc.close()
        raise StructureError("PDF 已加密（需要打开密码），无法解析版面结构")
    return doc


# ── ① 块抽取 ──

#: 判定「这里本来有个空格」的水平间距阈值，按字号取比例。
#: 字间距（kerning）接近 0，词间空格通常 0.25~0.33 em，取 0.12 em 有足够裕量。
SPACE_GAP_RATIO = 0.12


def join_spans(spans) -> str:
    """按几何间距把一行的 span 拼成文本，**该有空格的地方补回空格**。

    不能直接 `"".join`：PyMuPDF 在字体切换处会把一行切成多个 span，词间空格
    要么是独立的空白 span（会被 `.strip()` 过滤掉），要么根本不在文本里、
    只体现为 bbox 之间的水平间距。两种情况下 `"".join` 都会把相邻的词粘死。

    真库实测：走这条路的 `chunks` 里 **435/784（55.5%）** 含 25 字母以上的粘连词
    （`pioNER:DatasetsandBaselinesforArmenian`），而同一批 PDF 走
    `page.get_text("text")` 的 `pages` 表是 **0%**。

    粘连文本对三处都是毒：trigram FTS 匹配不到正常词、embedding 语义变差、
    以及**打断机械回取校验**——本项目要求证据句能在原文里逐字找到。
    """
    out: list[str] = []
    prev = None
    for s in spans or []:
        t = s.get("text") or ""
        if not t:
            continue
        if prev is not None and out:
            joined_tail = out[-1]
            pb = prev.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            cb = s.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            try:
                gap = float(cb[0]) - float(pb[2])
            except (TypeError, IndexError, ValueError):
                gap = 0.0
            size = 0.0
            for cand in (s.get("size"), prev.get("size")):
                try:
                    size = float(cand or 0.0)
                except (TypeError, ValueError):
                    size = 0.0
                if size:
                    break
            size = size or 10.0
            # 已经有空白、或断词连字符结尾，就不再补
            if (not joined_tail.endswith((" ", "\t", "-", "­"))
                    and not t[:1].isspace()
                    and gap > SPACE_GAP_RATIO * size):
                out.append(" ")
        out.append(t)
        prev = s
    return "".join(out)


def _line_of(ln: dict, page_no: int) -> dict | None:
    """一行 = 若干 span 合并。字号取行内最大（上标/角标会把均值拉低）。"""
    raw_spans = ln.get("spans", []) or []
    # 统计（字号/粗体）只看有实字的 span；**拼文本要看全部 span**——
    # 纯空白的 span 恰恰是词边界的证据，先过滤再拼就是把空格丢掉。
    spans = [s for s in raw_spans if (s.get("text") or "").strip()]
    if not spans:
        return None
    text = _clean(join_spans(raw_spans))
    if not text:
        return None
    total = sum(len((s.get("text") or "").strip()) for s in spans) or 1
    bold_chars = sum(len((s.get("text") or "").strip()) for s in spans if _is_bold(s))
    main = max(spans, key=lambda s: len((s.get("text") or "").strip()))
    bbox = ln.get("bbox") or (0.0, 0.0, 0.0, 0.0)
    return {
        "text": text,
        "size": max(round(float(s.get("size") or 0.0), 1) for s in spans),
        "font": main.get("font") or "",
        "bold": bold_chars / total >= 0.6,
        "bbox": tuple(round(float(v), 1) for v in bbox),
        "x0": round(float(bbox[0]), 1),
        "y0": round(float(bbox[1]), 1),
        "y1": round(float(bbox[3]), 1),
        "page_no": page_no,
    }


def _blocks_of_page(page, page_no: int) -> list[list[dict]]:
    """页 → 行分组。MuPDF 自己的 block 分组再按「字号/粗体/行距」二次切分。

    为什么要二次切：标题和紧随其后的正文常常被 MuPDF 收进同一个 block，
    不切开的话标题的字号信息会被正文稀释，字号判据直接失效。
    """
    groups: list[list[dict]] = []
    try:
        raw = page.get_text("dict")
    except Exception:
        return groups  # 单页解析失败不该毁掉整篇：跳过该页
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        cur: list[dict] = []
        for ln in blk.get("lines", []):
            d = ln.get("dir") or (1.0, 0.0)
            if abs(float(d[0])) < 0.98:
                continue  # 竖排（arXiv 侧边戳记之类）不参与版面判断
            line = _line_of(ln, page_no)
            if not line:
                continue
            if cur:
                prev = cur[-1]
                gap = line["y0"] - prev["y1"]
                if (abs(line["size"] - prev["size"]) > 0.5
                        or line["bold"] != prev["bold"]
                        or gap > 1.8 * max(prev["size"], 1.0)):
                    groups.append(cur)
                    cur = []
            cur.append(line)
        if cur:
            groups.append(cur)
    return groups


def _merge_lines(lines: list[dict]) -> str:
    """块内行合并：行尾断词连字符接回，其余用空格；中文行间不补空格。"""
    parts: list[str] = []
    for ln in lines:
        t = ln["text"]
        if not parts:
            parts.append(t)
            continue
        prev = parts[-1]
        if prev.endswith("-") and t[:1].islower():
            parts[-1] = prev[:-1] + t
        elif _CJK_RE.search(prev[-1:]) and _CJK_RE.search(t[:1]):
            parts[-1] = prev + t
        else:
            parts.append(t)
    return " ".join(parts)


def _read(pdf_path) -> tuple[list[dict], int]:
    """打开 PDF 抽块。返回 (blocks, n_pages)。document 一定在 finally 里关。"""
    doc = _open(pdf_path)
    blocks: list[dict] = []
    n_pages = 0
    try:
        n_pages = doc.page_count
        for pno in range(n_pages):
            try:
                page = doc[pno]
            except Exception:
                continue
            for lines in _blocks_of_page(page, pno + 1):
                text = _merge_lines(lines)
                if not text:
                    continue
                blocks.append({
                    "block_index": len(blocks),
                    "page_no": pno + 1,
                    "text": text,
                    "size": max(l["size"] for l in lines),
                    "min_size": min(l["size"] for l in lines),
                    "bold": all(l["bold"] for l in lines),
                    "font": lines[0]["font"],
                    "x0": min(l["x0"] for l in lines),
                    "y0": min(l["y0"] for l in lines),
                    "y1": max(l["y1"] for l in lines),
                    "n_lines": len(lines),
                    "n_chars": len(text),
                    "lines": lines,
                })
    finally:
        doc.close()
    return blocks, n_pages


def extract_blocks(pdf_path) -> list[dict]:
    """PDF → 文本块列表（后面所有启发式的基础）。

    每块：block_index / page_no / text / size / min_size / bold / font /
    x0,y0,y1 / n_lines / n_chars / lines（每行含 text,size,font,bold,bbox,page_no）。
    无文本层（扫描件）返回空列表，不抛异常。
    """
    return _read(pdf_path)[0]


# ── ② 章节识别 ──

def body_font_size(blocks: list[dict]) -> float:
    """正文字号 = 全文行字号的「众数」（按字符数加权）。没有文本返回 0.0。"""
    tally: dict[float, int] = {}
    for b in blocks:
        for ln in b["lines"]:
            tally[ln["size"]] = tally.get(ln["size"], 0) + len(ln["text"])
    if not tally:
        return 0.0
    # 全序：先比字符数，再比字号（并列时取小的那个——正文一般小于标题）
    return min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _numbering(text: str) -> tuple[str, int] | None:
    """编号判据。返回 (编号串, 层级)；层级从编号深度推断，推不出返回 None。"""
    m = _NUM_ARABIC_RE.match(text)
    if m:
        return m.group(1), len(m.group(1).split("."))
    m = _NUM_ROMAN_RE.match(text)
    if m:
        return m.group(1), 1
    m = _NUM_CJK_RE.match(text)
    if m:
        return m.group(0).strip(), 1 if m.group(2) in "章篇部" else 2
    m = _NUM_APPENDIX_RE.match(text)
    if m:
        return m.group(0).strip(), 1
    return None


def _strip_numbering(text: str) -> str:
    num = _numbering(text)
    if not num:
        return text
    return text[len(num[0]):].lstrip(" .、:：)").strip() or text


def _keyword_hit(text: str) -> str | None:
    """关键词判据：去掉编号前缀后，以某个章节名开头即算命中。"""
    body = _strip_numbering(text).strip(" .:：-—–")
    low = re.sub(r"\s+", " ", body.lower())
    for kw in _KEYWORDS_EN:
        if low == kw or low.startswith(kw + " ") or low.startswith(kw + ":"):
            return kw
    for kw in _KEYWORDS_ZH:
        if body.startswith(kw):
            return kw
    return None


def _looks_like_sentence(text: str) -> bool:
    """像正文句子：以句末标点收尾且不止三两个词。

    这是防「正文里的编号行被当成标题」的关键否决项——
    "1. First we note that the loss decreases fast." 命中编号判据，靠这条毙掉。
    """
    t = text.rstrip()
    if not t:
        return False
    if t.endswith((".", "。", "!", "?", "！", "？", ";", "；", ",", "，", "、")):
        return _token_count(t) >= 4
    return False


def _score_heading(b: dict, body_size: float) -> tuple[int, list[str], int]:
    """标题评分。返回 (得分, 命中的判据列表, 层级)。

    判据权重的理由：字号与粗体是**版面信号**，正文里几乎不会出现；编号与关键词是
    **文本信号**，正文里随处可见（"1. First we ..."、"Results show ..."）。
    所以单靠文本信号不足以判定标题，必须两类信号叠加或两个文本信号同时命中。
    """
    text = b["text"]
    matched: list[str] = []
    score = 0

    big = (body_size > 0 and b["size"] >= body_size * HEADING_SIZE_RATIO
           and b["size"] >= body_size + HEADING_SIZE_DELTA)
    if big:
        matched.append("font_size")
        score += 2
    if b["bold"]:
        matched.append("bold")
        score += 2
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) >= 4 and all(ch.isupper() for ch in letters):
        matched.append("all_caps")
        score += 1
    num = _numbering(text)
    if num:
        matched.append("numbering")
        score += 1
    if _keyword_hit(text):
        matched.append("keyword")
        score += 1
    if len(text) <= MAX_HEADING_CHARS:
        matched.append("short_line")
    return score, matched, (num[1] if num else 1)


#: 页眉去重时只把「独立成词的数字」（页码）归一化。
#: 不能把所有数字都换掉——"1.0 Numbered Section" 那样的编号标题会被归成同一个
#: 模板，于是整篇的章节标题被当作重复页眉全部误删。
_PAGENO_RE = re.compile(r"(?<!\S)\d+(?!\S)")


def _edge_blocks(blocks: list[dict]) -> set[int]:
    """每页最上/最下的文本块。页眉页脚只可能长在这两处——中间的块再怎么重复
    也是正文或章节标题，不该被当页眉删掉。"""
    by_page: dict[int, list[dict]] = {}
    for b in blocks:
        by_page.setdefault(b["page_no"], []).append(b)
    edge: set[int] = set()
    for bs in by_page.values():
        top = min(b["y0"] for b in bs)
        bot = max(b["y1"] for b in bs)
        for b in bs:
            if b["y0"] <= top + 2.0 or b["y1"] >= bot - 2.0:
                edge.add(b["block_index"])
    return edge


def _repeated_texts(blocks: list[dict], n_pages: int) -> set[str]:
    """跨多页重复出现的短文本 = 页眉/页脚，排除出标题候选（正文仍保留）。

    三个条件缺一不可，每一条都对应一个真实误删：
    ① 至少 REPEAT_PAGE_THRESHOLD 页上出现；
    ② 必须是所在页的首块或末块（`_edge_blocks`）——否则 "Topic Number 1..5"
       这种版心内的同模板标题会被整批删掉；
    ③ 出现比例还要够高（REPEAT_PAGE_RATIO）——真页眉几乎每页都有，而
       "Chapter 1".."Chapter 5" 在整本论文里只占几页；归一化页码后它们的 key
       相同，只有②③挡得住。

    仍然挡不住的一种：每页都以「同模板标题」开头的文档（每页一章），此时它与
    真页眉在统计上无法区分，会被当页眉滤掉——章名那一层（第二块）仍保留。
    反过来，页眉只出现在少数页（< REPEAT_PAGE_RATIO）时不再被滤掉，会多出几个
    假章节；这比「把全篇章节结构静默删光」轻，且 matched_by 里看得见。
    """
    if n_pages < REPEAT_PAGE_THRESHOLD:
        return set()
    edge = _edge_blocks(blocks)
    pages: dict[str, set[int]] = {}
    for b in blocks:
        if len(b["text"]) > MAX_HEADING_CHARS:
            continue
        if b["block_index"] not in edge:
            continue
        key = _PAGENO_RE.sub("#", b["text"])
        pages.setdefault(key, set()).add(b["page_no"])
    need = max(REPEAT_PAGE_THRESHOLD, math.ceil(n_pages * REPEAT_PAGE_RATIO))
    return {k for k, v in pages.items() if len(v) >= need}


def _detect_sections(blocks: list[dict], body_size: float, n_pages: int) -> list[dict]:
    repeated = _repeated_texts(blocks, n_pages)
    out: list[dict] = []
    for b in blocks:
        text = b["text"]
        if not text or b["n_lines"] > MAX_HEADING_LINES:
            continue
        if len(text) > MAX_HEADING_CHARS:
            continue
        if len(text) < 2 or not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
            continue
        if _CAPTION_RE.match(text):
            continue  # 图表题注：又粗又短，最容易被误判成章节
        if _PAGENO_RE.sub("#", text) in repeated:
            continue
        if _looks_like_sentence(text):
            continue
        score, matched, level = _score_heading(b, body_size)
        if score < HEADING_MIN_SCORE:
            continue
        out.append({
            "title": text,
            "level": level,
            "page_no": b["page_no"],
            "block_index": b["block_index"],
            "matched_by": matched,
            "font_size": b["size"],
            "score": score,
        })
    return _drop_paper_title(blocks, out)


def _drop_paper_title(blocks: list[dict], sections: list[dict]) -> list[dict]:
    """首页那块全文最大字号、既没编号也没关键词的标题 = 论文题名，不是章节。

    留着它会让题名、作者、单位被当成「第一章的正文」，section_path 从此全篇
    挂在题名下面。判据卡得很紧（首页 + 全文最大字号 + **严格大于其余所有标题的
    字号** + 无编号 + 无关键词 + 后面还有别的章节），"Introduction" 这类真章节
    因为命中关键词不会被误删。

    「严格大于其余标题」这一条是必需的：文档里所有标题同字号时（内部报告、
    没有独立题名页的文稿），第一节是货真价实的章节，删掉就是静默丢一节。
    """
    if len(sections) < 2:
        return sections
    first = sections[0]
    if first["page_no"] != 1:
        return sections
    if {"numbering", "keyword"} & set(first["matched_by"]):
        return sections
    if first["font_size"] < max(b["size"] for b in blocks) - 0.01:
        return sections
    if first["font_size"] <= max(s["font_size"] for s in sections[1:]) + 0.01:
        return sections
    return sections[1:]


def detect_sections(pdf_path) -> list[dict]:
    """识别章节标题。

    返回 [{title, level, page_no, block_index, matched_by, font_size, score}]，
    按出现顺序排列。matched_by 是命中的判据名（font_size / bold / all_caps /
    numbering / keyword / short_line），让「凭什么认定它是标题」可审计。
    一个都识别不出来时返回空列表（调用方据此走降级）。
    """
    blocks, n_pages = _read(pdf_path)
    return _detect_sections(blocks, body_font_size(blocks), n_pages)


# ── ③ 按章节切 chunk ──

def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """超长单段的二次切分：句末标点 > 空白 > 硬切。"""
    out: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        # 句末标点连同其后的空格一起归左边，切完两半都读得通
        ends = [window.rfind(p) + len(p) for p in
                ("。", "！", "？", "；", ". ", "! ", "? ", "; ") if window.rfind(p) >= 0]
        cut = max(ends) if ends else -1
        if cut < max_chars * 0.4:      # 靠太前就等于没切，退化到空白/硬切
            sp = window.rfind(" ")
            cut = sp + 1 if sp > max_chars * 0.3 else max_chars
        cut = max(1, min(cut, len(rest)))
        piece = rest[:cut].strip()
        if piece:
            out.append(piece)
        rest = rest[cut:].lstrip()
    if rest.strip():
        out.append(rest.strip())
    return out


def _pack(items: list[tuple[str, int]], max_chars: int) -> list[list[tuple[str, int]]]:
    """把 (段落文本, 页码) 贪心装箱：切分点优先落在段落边界，段落本身超长才内部切。"""
    groups: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_len = 0
    for text, page in items:
        pieces = _split_paragraph(text, max_chars) if len(text) > max_chars else [text]
        for pc in pieces:
            add = len(pc) + (2 if cur else 0)
            if cur and cur_len + add > max_chars:
                groups.append(cur)
                cur, cur_len = [], 0
                add = len(pc)
            cur.append((pc, page))
            cur_len += add
    if cur:
        groups.append(cur)
    return groups


def _emit(groups, title: str, path: str, level: int, start_at: int,
          degraded: str | None) -> list[dict]:
    out: list[dict] = []
    for part, grp in enumerate(groups, 1):
        text = "\n\n".join(t for t, _ in grp)
        pages = [p for _, p in grp]
        chunk = {
            "section_title": title,
            "section_path": path,
            "level": level,
            "start_page": min(pages) if pages else None,
            "end_page": max(pages) if pages else None,
            "text": text,
            "chunk_index": start_at + part - 1,
            "part": part,
            "n_parts": len(groups),
            "n_chars": len(text),
        }
        if degraded:
            chunk["degraded"] = degraded
        out.append(chunk)
    return out


PAGE_FALLBACK_NOTE = "未识别到章节结构，已退化为按页切分"


def _page_chunks(blocks: list[dict], max_chars: int) -> list[dict]:
    """降级路径：按页切。扫描件 / 纯图片 / 无标题排版是常态，必须优雅降级。"""
    out: list[dict] = []
    by_page: dict[int, list[tuple[str, int]]] = {}
    order: list[int] = []
    for b in blocks:
        if b["page_no"] not in by_page:
            by_page[b["page_no"]] = []
            order.append(b["page_no"])
        by_page[b["page_no"]].append((b["text"], b["page_no"]))
    for page in order:
        out.extend(_emit(_pack(by_page[page], max_chars), "", "", 0,
                         len(out), PAGE_FALLBACK_NOTE))
    return out


#: max_chars 的下限。再小的切分只会把句子剁碎，对检索没有意义；
#: 传入更小的值会被抬到这里（调用方据此不能假设 n_chars <= 任意小的 max_chars）。
MIN_CHUNK_CHARS = 200


def _section_chunks(blocks: list[dict], sections: list[dict],
                    max_chars: int) -> list[dict]:
    max_chars = max(MIN_CHUNK_CHARS, int(max_chars))
    if not blocks:
        return []
    if not sections:
        return _page_chunks(blocks, max_chars)

    out: list[dict] = []
    bounds = [s["block_index"] for s in sections] + [len(blocks)]

    head = [(b["text"], b["page_no"]) for b in blocks[:bounds[0]]]
    if head:
        # 首个标题之前的内容（题名/作者/摘要）不能丢，单独成块并如实标注它没有章节归属
        out.extend(_emit(_pack(head, max_chars), "", "（文首，章节之前）", 0, 0, None))

    stack: list[tuple[int, str]] = []
    for i, sec in enumerate(sections):
        level = max(1, int(sec.get("level") or 1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, sec["title"]))
        path = " > ".join(t for _, t in stack)
        body = blocks[sec["block_index"] + 1: bounds[i + 1]]
        items = [(b["text"], b["page_no"]) for b in body]
        if not items:
            continue  # 父标题下紧跟子标题：没有正文就不产 chunk，但仍留在 path 上
        out.extend(_emit(_pack(items, max_chars), sec["title"], path, level,
                         len(out), None))
    return out


def section_chunks(pdf_path, max_chars: int = 4000) -> list[dict]:
    """按章节切 chunk。

    每个 chunk：section_title / section_path（"3 Method > 3.2 Training"）/ level /
    start_page / end_page / text / chunk_index / part / n_parts / n_chars。
    超长章节按 max_chars 二次切分，切分点优先落段落边界，同章节多个 chunk 共享
    section_title 与 section_path。一个章节都没识别出来时退化为按页切，
    每个 chunk 带 degraded="未识别到章节结构，已退化为按页切分"。
    无文本层时返回空列表。

    注意：max_chars 有下限 MIN_CHUNK_CHARS(=200)，传更小的值会被静默抬到下限；
    章节标题本身不进 chunk 的 text（在 section_title / section_path 里），
    因此把所有 chunk 的 text 拼起来 ≠ 全文（少了标题行）。
    """
    blocks, n_pages = _read(pdf_path)
    sections = _detect_sections(blocks, body_font_size(blocks), n_pages)
    return _section_chunks(blocks, sections, max_chars)


# ── ④ 参考文献抽取（0 token，纯正则）──

def _clean_doi(raw: str) -> str:
    d = raw.strip().rstrip("-")
    d = re.sub(r"[.,;:'\"]+$", "", d)
    while d.endswith(")") and d.count(")") > d.count("("):
        d = d[:-1]
    while d.endswith("]") and d.count("]") > d.count("["):
        d = d[:-1]
    return d


def _find_doi(text: str) -> str:
    m = DOI_RE.search(text or "")
    return _clean_doi(m.group(0)) if m else ""


def _find_arxiv(text: str) -> str:
    m = ARXIV_NEW_RE.search(text or "") or ARXIV_OLD_RE.search(text or "")
    return m.group(1) if m else ""


def _join_entry(lines: list[str]) -> str:
    """条目内换行拼接：折行的 DOI/arXiv 无缝粘回，断词连字符去掉，其余补空格。

    只在行尾**看起来被截断**时才无缝粘，且按确定性分硬/软两档（见上面
    _DOI_TRUNC_HARD_RE / _DOI_TRUNC_SOFT_RE 的注释）：DOI 已写完而下一行是别的
    内容时无脑粘，会造出 10.1234/abc.567Smith、10.1109/x.123456.pp 这种假 DOI
    ——宁可修不回来，也不能造一个错的出来。
    """
    parts: list[str] = []
    for ln in lines:
        cur = ln.strip()
        if not cur:
            continue
        if not parts:
            parts.append(cur)
            continue
        prev = parts[-1]
        if _DOI_TRUNC_HARD_RE.search(prev) and _CONT_HARD_RE.match(cur):
            parts[-1] = prev + cur
        elif ((_DOI_TRUNC_SOFT_RE.search(prev) or _ARXIV_TRUNC_RE.search(prev))
                and _CONT_SOFT_RE.match(cur)):
            parts[-1] = prev + cur
        elif prev.endswith("-") and cur[:1].islower():
            parts[-1] = prev[:-1] + cur
        elif _CJK_RE.search(prev[-1:]) and _CJK_RE.search(cur[:1]):
            parts[-1] = prev + cur
        else:
            parts.append(cur)
    return _clean(" ".join(parts))


def _authorish(seg: str) -> bool:
    """这一段是不是作者名列表。宁可多判成作者让标题留空，也不要把
    「Zhang San, Li Si」当标题印出去——错的标题比空标题更有害。"""
    if re.findall(r"\b[A-Z]\.", seg) and len(seg.split()) <= 14:
        return True
    if re.search(r"\bet\s+al\b", seg, re.I):
        return True
    if re.match(r"^[\u4e00-\u9fff]{2,4}(\s*[,，、;；]\s*[\u4e00-\u9fff]{2,4})+\s*$", seg):
        return True
    # 「Zhang San, Li Si」：分成多段，每段 1~3 个首字母大写的词、没有小写虚词
    parts = [p.strip() for p in re.split(r"\s*(?:[,;&]|\band\b)\s*", seg) if p.strip()]
    if len(parts) >= 2 and all(
            1 <= len(p.split()) <= 3 and all(w[:1].isupper() for w in p.split())
            for p in parts):
        return True
    return False


def _title_guess(raw: str) -> str:
    """条目标题猜测：引号内优先，否则取第一个「不像作者名」的分段。抽不准就留空。"""
    t = re.sub(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s*", "", raw)
    m = re.search(r"[“\"]([^”\"]{8,300})[”\"]", t)
    if m:
        return _clean(m.group(1)).strip(" ,.;")
    # 句点切段，但不切人名缩写（"J. Smith." 里的 "J." 不是段落边界）
    segs = [s.strip() for s in re.split(r"(?<![A-Z])[.。]\s+", t) if s.strip()]
    for seg in segs:
        if _authorish(seg):
            continue
        if re.search(r"(doi|arxiv|http|www\.)", seg, re.I):
            continue
        if _token_count(seg) < 3:
            continue
        if re.match(r"^(in\s+|proc\b|proceedings\b|journal\b|ieee\b|acm\b)", seg, re.I):
            continue
        return _clean(seg).strip(" ,.;")
    return ""


def _year_guess(raw: str, doi: str, arxiv_id: str) -> int | None:
    """年份：括号年优先；否则剔掉 DOI/arXiv 串（里面有会误判的数字）后取最后一个。"""
    m = re.search(r"[(（]((?:19|20)\d{2})[a-z]?[)）]", raw)
    if m:
        return int(m.group(1))
    text = raw
    for s in (doi, arxiv_id):
        if s:
            text = text.replace(s, " ")
    years = [int(x.group(0)) for x in _YEAR_RE.finditer(text)]
    return years[-1] if years else None


def _ref_lines(blocks: list[dict]) -> tuple[list[dict], int | None, str | None]:
    """定位 References 段，返回 (该段的行, 起始页, 降级说明)。"""
    lines: list[dict] = [dict(ln, block_index=b["block_index"])
                         for b in blocks for ln in b["lines"]]
    if not lines:
        return [], None, "PDF 没有可提取的文本层，未抽取参考文献"
    start = -1
    inline_cut = 0
    for i, ln in enumerate(lines):
        if _REF_HEAD_RE.match(ln["text"]):
            start, inline_cut = i + 1, 0
        else:
            m = _REF_HEAD_INLINE_RE.match(ln["text"])
            # 「References [1] Smith...」标题与首条挤在一行：从标题后面接着切。
            # 必须要求后面紧跟条目编号，否则正文里的 "References to prior work ..."
            # 也会被当成参考文献段起点（而且取最后一个匹配，会盖掉真的那个）。
            rest = ln["text"][m.end():] if m else ""
            if rest and (_MARK_BRACKET_RE.match(rest) or _MARK_NUMDOT_RE.match(rest)):
                start, inline_cut = i, m.end()
    if start < 0 or start >= len(lines):
        return [], None, "未定位到 References / Bibliography / 参考文献 标题，未抽取任何条目"
    seg = [dict(l) for l in lines[start:]]
    if inline_cut:
        seg[0]["text"] = seg[0]["text"][inline_cut:].strip()
    for j, ln in enumerate(seg):
        if _REF_END_RE.match(ln["text"]) and len(ln["text"]) < 40:
            seg = seg[:j]  # 附录：参考文献段到此为止
            break
    seg = [l for l in seg if l["text"].strip() and not re.fullmatch(r"[\d\s\-–—]+", l["text"])]
    if not seg:
        return [], lines[start - 1]["page_no"] if start else None, \
            "找到 References 标题但其后没有可解析的文本行"
    return seg, seg[0]["page_no"], None


def _split_entries(seg: list[dict]) -> tuple[list[list[dict]], str]:
    """条目切分：方括号编号 > 数字点号 > 悬挂缩进 > 行距 > 一行一条。"""
    n_bracket = sum(1 for l in seg if _MARK_BRACKET_RE.match(l["text"]))
    n_numdot = sum(1 for l in seg if _MARK_NUMDOT_RE.match(l["text"]))
    if n_bracket >= 2:
        return _split_at(seg, [i for i, l in enumerate(seg)
                              if _MARK_BRACKET_RE.match(l["text"])]), "bracket"
    if n_numdot >= 2:
        return _split_at(seg, [i for i, l in enumerate(seg)
                              if _MARK_NUMDOT_RE.match(l["text"])]), "numdot"
    # 编号挤在同一行：整段文本按 [n] 再切一次。
    # 用 finditer 而不是 re.split——split 的正则带捕获组时会把组内容也塞进结果，
    # 每个编号数字会变成一个只有 "1"、"2" 的假条目。
    flat = " ".join(l["text"] for l in seg)
    hits = list(_MARK_BRACKET_ANY_RE.finditer(flat))
    if len(hits) >= 2:
        out = []
        for k, h in enumerate(hits):
            end = hits[k + 1].start() if k + 1 < len(hits) else len(flat)
            out.append([{"text": flat[h.start():end].strip(),
                         "page_no": seg[0]["page_no"], "x0": 0.0}])
        return out, "bracket-inline"
    xs = sorted({l["x0"] for l in seg})
    if len(xs) >= 2 and xs[-1] - xs[0] >= 2.0:
        base = xs[0]
        starts = [i for i, l in enumerate(seg) if l["x0"] <= base + 1.0]
        if 2 <= len(starts) < len(seg):
            return _split_at(seg, starts), "hanging-indent"
    gaps = [seg[i]["y0"] - seg[i - 1]["y1"] for i in range(1, len(seg))
            if seg[i]["page_no"] == seg[i - 1]["page_no"]]
    if gaps:
        ordered = sorted(gaps)
        med = ordered[len(ordered) // 2]
        starts = [0] + [i for i in range(1, len(seg))
                        if seg[i]["page_no"] != seg[i - 1]["page_no"]
                        or seg[i]["y0"] - seg[i - 1]["y1"] > med + 2.0]
        if 2 <= len(starts) < len(seg):
            return _split_at(seg, starts), "line-gap"
    return [[l] for l in seg], "one-line-per-entry"


def _split_at(seg: list[dict], starts: list[int]) -> list[list[dict]]:
    if not starts:
        return [seg]
    if starts[0] != 0:
        starts = [0] + starts  # 第一条之前的零头（常是被切碎的标题行）单独成条，后续会被过滤
    out: list[list[dict]] = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(seg)
        out.append(seg[s:e])
    return out


def _extract_references(blocks: list[dict]) -> dict:
    seg, start_page, degraded = _ref_lines(blocks)
    if not seg:
        return {"n_entries": 0, "entries": [], "with_doi": 0, "with_arxiv": 0,
                "with_norm_key": 0, "start_page": start_page, "split_by": None,
                "degraded": degraded}
    groups, how = _split_entries(seg)
    entries: list[dict] = []
    dropped = 0
    for grp in groups:
        raw = _join_entry([l["text"] for l in grp])
        if len(raw) < 10:
            dropped += 1
            continue
        doi = _find_doi(raw)
        aid = _find_arxiv(raw)
        m = _MARK_BRACKET_RE.match(raw) or _MARK_NUMDOT_RE.match(raw)
        entries.append({
            "raw": raw,
            "index": int(m.group(1)) if m else None,
            "doi": doi or "",
            "arxiv_id": aid or "",
            "title_guess": _title_guess(raw),
            "year": _year_guess(raw, doi, aid),
            # 只有拿到真外部标识才给 norm_key：猜出来的标题不可靠，
            # 拿它当键会往引文图里塞错节点（graph.py 的边端点就是 norm_key）
            "norm_key": _norm_key(doi=doi or None, arxiv_id=aid or None) if (doi or aid) else None,
            "page_no": grp[0]["page_no"],
        })
    notes = []
    if how == "one-line-per-entry":
        notes.append("未找到 [n] / n. / 悬挂缩进等条目标记，按「一行一条」粗切，条目边界可能不准")
    if dropped:
        notes.append(f"{dropped} 个过短片段（<10 字符）被丢弃，未计入条目")
    return {
        "n_entries": len(entries),
        "entries": entries,
        "with_doi": sum(1 for e in entries if e["doi"]),
        "with_arxiv": sum(1 for e in entries if e["arxiv_id"]),
        "with_norm_key": sum(1 for e in entries if e["norm_key"]),
        "start_page": start_page,
        "split_by": how,
        "degraded": "；".join(notes) or None,
    }


def extract_references(pdf_path) -> dict:
    """抽参考文献条目（纯正则，0 token，离线可跑）。

    返回 {n_entries, entries, with_doi, with_arxiv, with_norm_key, start_page,
    split_by, degraded}；每条 entry 为 {raw, index, doi, arxiv_id, title_guess,
    year, norm_key, page_no}，抽不出的字段留空（doi/arxiv_id 为 ""，
    year/norm_key 为 None），绝不用猜测值填充。
    norm_key 只在拿到 DOI 或 arXiv id 时才给——它是 graph.py 引文边的端点，
    用猜来的标题当键会往引文图里塞错节点。
    """
    return _extract_references(extract_blocks(pdf_path))


# ── ⑤ 一次拿全 ──

def summarize(pdf_path, max_chars: int = 4000) -> dict:
    """一次打开 PDF，返回章节 / chunk 数 / 参考文献 / 正文字号 / 降级说明。"""
    blocks, n_pages = _read(pdf_path)
    body = body_font_size(blocks)
    sections = _detect_sections(blocks, body, n_pages)
    chunks = _section_chunks(blocks, sections, max_chars)
    refs = _extract_references(blocks)

    warnings: list[str] = []
    degraded = None
    if not blocks:
        degraded = "PDF 没有可提取的文本层（扫描件/纯图片 PDF），版面结构与参考文献均无法解析"
    elif not sections:
        degraded = PAGE_FALLBACK_NOTE
    if refs.get("degraded"):
        warnings.append(f"参考文献：{refs['degraded']}")
    if body <= 0 and blocks:
        warnings.append("正文字号无法统计，字号判据本次未生效")
    return {
        "sections": sections,
        "n_sections": len(sections),
        "n_chunks": len(chunks),
        "references": refs,
        "body_font_size": body,
        "degraded": degraded,
        "n_pages": n_pages,
        "n_blocks": len(blocks),
        "warnings": warnings,
    }
