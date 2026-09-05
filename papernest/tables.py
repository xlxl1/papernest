"""PDF 表格区域识别 + 「整行不截断」的表格切分。

**要解决的问题**：`fulltext.extract_pages` 按页切文本，一张跨页的表在页边界被拦腰砍断，
下半张失去表头后完全不可读——检索到它等于检索到一堆没有列名的数字。本模块提供
「表格整行聚合（行数 / 字符数双限）+ 后续块重复表头」的切分，让第 2 块被单独召回时仍可读。

**识别路径有两条，且在结果里如实标注是哪条**（`detected_by`，这是可审计性的要求）：
① `page.find_tables()`（PyMuPDF 1.23+ 自带，靠框线）——有框线的表最可靠，优先用；
② 几何启发式兜底——无框线（学术论文里的三线表就是无框线）时 find_tables 常常返回 0，
   于是退回 `page.get_text("dict")` 的 span 坐标：先把 span 按「明显的水平间隙」切成单元格，
   再把多行按 x 区间聚类成列，要求列在足够多的行上重复出现（列对齐）才认成表。

**明确的能力边界**（写在这里，不在报告里粉饰）：
- 几何兜底要求每行 >= 3 个单元格（即 >= 2 个明显水平间隙）。这不是随手定的：2 列门槛会把
  参考文献（"[1]" + 悬挂缩进正文）、编号列表整段判成表格——它们的 x 区间对齐得比真表还整齐。
  代价是**真正的两列表格识别不到**，这是本模块已知且故意接受的漏检。
- 合并单元格（rowspan/colspan）不还原：find_tables 给 None，几何路径按列归位，都会留空格。
- 单元格内文本折行会被拆成两「行」（行聚类按 y 重叠做），跨行单元格的表行数会偏多。
- 跨页表格不做跨页拼接（本模块按页识别）。要拼接需要主进程在页间做接续判断。
- 扫描件（无文本层）两条路径都拿不到东西，返回 0 张表，不报错。

失败策略：除「PDF 打不开 / 加密」抛 TableError 外，任何一步失败都降级并把原因写进
warnings，绝不静默吞掉，也绝不用编造的行列填充。全程离线、0 token、只依赖 pymupdf。
"""
import math
import re
import unicodedata

# ── 阈值常量（集中在这里，便于按语料调参）──

#: 几何兜底：一张表至少要有这么多行（含表头）。3 行以下的「对齐」几乎全是巧合。
GEOM_MIN_ROWS = 3
#: 几何兜底：整张表最终至少要有这么多列（= 行内至少 2 个明显水平间隙）。
#: 取 3 而不是 2 的理由见模块 docstring——2 会让参考文献段整段变成「表格」。
GEOM_MIN_COLS = 3
#: 一行要有这么多单元格才够格加入候选区。定 2 而不是 3 是拿真论文测出来的：
#: 学术表的「类别」列常常几行才写一次（跨行合并单元格的排版效果），
#: 卡在 3 会让整张表碎成一行一段、一张都认不出来。列数下限由 GEOM_MIN_COLS 在最后把关。
RUN_MIN_CELLS = 2
#: 网格至少要有这么多「强列」（在多数行上都出现的列）。
#: 只要求 >=2：稀疏的类别列本来就不该被要求每行都有。
GEOM_MIN_STRONG_COLS = 2
#: 单元格切分的水平间隙阈值 = max(GAP_MIN_PT, GAP_SIZE_RATIO * 字号)。
#: 10pt 正文里一个空格宽约 2.8pt，取 10pt 阈值可以放过词间空格、切开列间空白。
GAP_MIN_PT = 6.0
GAP_SIZE_RATIO = 1.0
#: 相邻两行的垂直间隙超过「上一行高度 * 该系数」就断开候选区（隔了一段正文的两块不是一张表）。
ROW_GAP_RATIO = 2.5
#: 候选区内允许连续出现多少个「只有 1 个单元格」的稀疏行（跨行合并的类别标签）。
#: 定 1：连着两行都只有一格，那就是正文，必须打断。
MAX_SPARSE_ROWS = 1
#: 两段列区间相距在该值以内就并成一列（吸收 1pt 的分箱抖动）。
COL_MERGE_PAD = 1.0
#: 切列的分隔阈值：某个 x 位置被不超过「行数 * 该比例」个单元格穿过，就算列间空白。
#: 定 0.05 而不是更大：类别列常常十几行才写一次，阈值一大它就被当成空白抹掉，
#: 三列表退化成两列后会被 GEOM_MIN_COLS 直接毙掉。
COL_SEP_RATIO = 0.05
#: 一列要在这么高比例的行上出现，才算「强列」。
COL_STRONG_RATIO = 0.6
#: 强列占全部列的比例下限——不到就说明网格很破碎，判为不是表格。
COL_STRONG_FRACTION = 0.5

# ── 双栏版面的中缝（gutter）检测参数 ──
# 为什么必须有这一步：双栏论文里，左栏正文和右栏表格处在同一个 y 上，
# 按整页做行聚类会把「左栏的一句话」和「右栏表格的三个格」拼成同一行，
# 于是整页正文被判成一张四列表。拿真论文（26 页双栏综述）实测：不分栏时
# 识别出 17 张「表」且大半掺着正文，分栏后才对得上。合成夹具完全测不出这个。
#: 页面行数少于这个数就不做分栏（版面证据不足，宁可不分）。
GUTTER_MIN_LINES = 24
#: 中缝两侧各自至少要有这么多行 / 这么高比例的行，才认这是真的双栏。
#: 这两条是用来挡「表格列间空白被当成中缝、把一张表劈成两半」的。
GUTTER_MIN_SIDE_LINES = 12
GUTTER_MIN_SIDE_RATIO = 0.25
#: 中缝最小宽度。真论文实测只有 4~9pt，所以不能定大。
GUTTER_MIN_WIDTH = 3.0
#: 中缝中点必须落在「文本水平中心 ± 该比例 * 文本宽度」内。
GUTTER_CENTER_TOL = 0.10
#: 允许这么小比例的行横跨中缝（跨栏图注之类），它们单独成一组处理。
GUTTER_CROSS_RATIO = 0.02
#: 宽度超过文本宽度这个比例的行不参与中缝判定：标题、摘要、跨栏图注本来就横穿版心，
#: 让它们参与判定会让「上半页跨栏 + 下半页双栏」的首页永远找不到中缝
#: （真论文首页实测：16 行横跨，全是标题/作者/摘要）。表格单元格永远没这么宽。
WIDE_LINE_RATIO = 0.6
#: 中缝两侧至少有一侧的「行宽中位数 / 该侧跨度」要达到这个值，才认这是双栏正文。
#: 表格的列间空白也长得像中缝，靠这一条把它挡回去（见 _column_bands 的注释）。
GUTTER_PROSE_FILL = 0.5
#: 几何候选与 find_tables 结果的 bbox 交叠达到该比例即视为同一张表（保留 find_tables 的）。
DEDUPE_OVERLAP = 0.5
#: split_page_text 判「这行属于表格」时，命中的单元格文本要覆盖该行这么高比例的字符。
LINE_COVER_RATIO = 0.5

_WS_RE = re.compile(r"[\s\u00a0\u2000-\u200b]+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
#: 「像数字」：纯数值 / 百分比 / 带正负号与误差的实验数字。用于「首行是不是表头」的判断。
_NUMERICISH_RE = re.compile(r"^[-+\u00b1]?\d[\d,.\s%\u00b1\u00d7/\u2212-]*$")
#: 段落切分与句子切分（正文块用）。
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?。！？；;])\s+")

#: markdown 表头分隔行的单元格。
_SEP_CELL = "---"
NOTE_BORROWED_HEADER = "> 注：未识别到表头，下方第一行为原表首行（代用表头）。"
NOTE_CONT_WITH_HEADER = "> 注：续表（表头已重复，便于本块被单独检索时仍可读）。"
NOTE_CONT_NO_HEADER = "> 注：续表（未重复表头，表头见第 1 块）。"


class TableError(ValueError):
    """PDF 打不开 / 加密到读不出内容。其余情况一律降级，不抛异常。"""


# ── 基础工具 ──

def _clean(s) -> str:
    """NFKC 归一 + 去软连字符 + 压缩空白（不含换行的场合用）。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00ad", "")
    return _WS_RE.sub(" ", s).strip()


def _squash(s) -> str:
    """去掉全部空白 + 小写：用于「这行正文属不属于某张表」的匹配。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00ad", "")
    return re.sub(r"\s+", "", s).lower()


_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
_CM_BOLD_RE = re.compile(r"bx\d")


def _is_bold(span: dict) -> bool:
    """flags bit4(=16) 是粗体位；嵌入子集字体常常只能从字体名看出来。

    自己实现而不是从 structure.py 复用同名私有函数：那是别的模块的私有细节，
    跨模块依赖私有名会在对方重构时无声炸掉。
    """
    if int(span.get("flags") or 0) & 16:
        return True
    font = _SUBSET_PREFIX_RE.sub("", span.get("font") or "").lower()
    if any(k in font for k in ("bold", "black", "heavy", "semib", "-bd", "medi")):
        return True
    return bool(_CM_BOLD_RE.search(font))


def _numericish(s: str) -> bool:
    t = _clean(s)
    return bool(t) and bool(_NUMERICISH_RE.match(t))


def _open(pdf_path):
    import pymupdf
    # PyMuPDF 1.28 会往 stdout 打一条推销 pymupdf_layout 的话，污染 CLI/JSON 输出。
    # 有这个开关就关掉；老版本没有就算了（getattr 探测，不能硬调）。
    quiet = getattr(pymupdf, "no_recommend_layout", None)
    if callable(quiet):
        try:
            quiet()
        except Exception:
            pass
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:  # 文件不存在 / 不是 PDF / 结构损坏
        raise TableError(f"PDF 打不开：{e}") from e
    if getattr(doc, "needs_pass", False):
        doc.close()
        raise TableError("PDF 已加密（需要打开密码），无法识别表格")
    return doc


def _bbox(b) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (float(v) for v in tuple(b)[:4])
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    return (round(min(x0, x1), 2), round(min(y0, y1), 2),
            round(max(x0, x1), 2), round(max(y0, y1), 2))


def _overlap_frac(a, b) -> float:
    """两个 bbox 交叠面积 / 较小者面积。0 表示不相交。"""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    small = min(area_a, area_b)
    return (ix * iy) / small if small > 0 else 0.0


# ── ① 页面几何：span → 单元格 → 行 → 列 ──

def _span_atoms(span: dict) -> list[dict]:
    """一个 span 切成若干「原子」：span 内部 >=2 个空格也算列分隔。

    为什么要在 span 内部再切：PDF 里一整行表格常常被抽成一个 span，
    列之间只剩空格，不切就永远只有 1 个单元格。x 坐标按字符数等宽线性插值估算——
    真实字宽当然不等宽，但这个估算只用于「间隙够不够大」的比较，误差可以接受。
    """
    text = span.get("text") or ""
    if not text.strip():
        return []
    bbox = span.get("bbox") or (0.0, 0.0, 0.0, 0.0)
    x0, x1 = float(bbox[0]), float(bbox[2])
    width = max(x1 - x0, 0.0)
    n = len(text)
    size = float(span.get("size") or 0.0)
    bold = _is_bold(span)

    segs: list[tuple[int, int]] = []
    pos = 0
    for m in _MULTI_SPACE_RE.finditer(text):
        segs.append((pos, m.start()))
        pos = m.end()
    segs.append((pos, n))

    out: list[dict] = []
    for a, b in segs:
        raw = text[a:b]
        if not raw.strip():
            continue
        a += len(raw) - len(raw.lstrip())
        b -= len(raw) - len(raw.rstrip())
        sx0 = x0 + width * (a / n) if n else x0
        sx1 = x0 + width * (b / n) if n else x1
        out.append({"x0": sx0, "x1": max(sx1, sx0), "text": raw.strip(),
                    "size": size, "bold": bold})
    return out


def _page_lines(page) -> tuple[list[dict], str | None]:
    """页 → 行列表（每行带 span 原子）。单页解析失败返回 ([], 原因)，不炸整篇。"""
    try:
        raw = page.get_text("dict")
    except Exception as e:
        return [], f"get_text('dict') 失败：{e}"
    out: list[dict] = []
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            d = ln.get("dir") or (1.0, 0.0)
            if abs(float(d[0])) < 0.98:
                continue  # 竖排文字（侧边戳记之类）不参与版面判断
            atoms: list[dict] = []
            for sp in ln.get("spans", []):
                atoms.extend(_span_atoms(sp))
            if not atoms:
                continue
            bbox = ln.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            out.append({"y0": float(bbox[1]), "y1": float(bbox[3]),
                        "x0": min(a["x0"] for a in atoms),
                        "x1": max(a["x1"] for a in atoms), "atoms": atoms})
    return out, None


def _column_bands(lines: list[dict]) -> list[list[dict]]:
    """按页面中缝把行分成左右两栏；找不到中缝就整页一组。

    中缝 = 页面水平中部一条「几乎没有文字穿过」的竖带。判定卡得很紧
    （宽度、居中、两侧行数都要够），因为一旦把表格的列间空白误当成中缝，
    整张表会被劈成两半——那比不分栏还糟。横跨中缝的行（跨栏图注）单独成组。
    """
    n = len(lines)
    if n < GUTTER_MIN_LINES:
        return [lines]
    x_min = min(l["x0"] for l in lines)
    x_max = max(l["x1"] for l in lines)
    span = x_max - x_min
    if span <= 0:
        return [lines]

    start = int(math.floor(x_min + 0.25 * span))
    end = int(math.ceil(x_min + 0.75 * span))
    if end <= start:
        return [lines]
    cov = [0] * (end - start + 1)
    narrow = [l for l in lines if l["x1"] - l["x0"] < WIDE_LINE_RATIO * span]
    for l in narrow:
        a = max(int(math.floor(l["x0"])), start)
        b = min(int(math.ceil(l["x1"])), end)
        for x in range(a, b + 1):
            cov[x - start] += 1

    tol = int(GUTTER_CROSS_RATIO * n)
    best: tuple[int, int] | None = None
    run: tuple[int, int] | None = None
    for i, c in enumerate(cov):
        if c <= tol:
            run = (i, i) if run is None else (run[0], i)
            continue
        if run and (best is None or run[1] - run[0] > best[1] - best[0]):
            best = run
        run = None
    if run and (best is None or run[1] - run[0] > best[1] - best[0]):
        best = run
    if best is None:
        return [lines]

    a, b = float(start + best[0]), float(start + best[1])
    if b - a < GUTTER_MIN_WIDTH:
        return [lines]
    if abs((a + b) / 2 - (x_min + x_max) / 2) > GUTTER_CENTER_TOL * span:
        return [lines]

    left = [l for l in lines if l["x1"] <= b]
    right = [l for l in lines if l["x0"] >= a]
    cross = [l for l in lines if l["x0"] < a and l["x1"] > b]
    need = max(GUTTER_MIN_SIDE_LINES, GUTTER_MIN_SIDE_RATIO * n)
    if len(left) < need or len(right) < need:
        return [lines]
    # 最后一道闸：至少一侧要「像正文栏」——行宽中位数占该侧宽度的大半。
    # 没有这一条，一张占满整页的多列表会被它自己的列间空白当成中缝劈成两半，
    # 每半只剩两列、双双被 GEOM_MIN_COLS 毙掉，整张表凭空消失。
    # 正文栏的行是撑满栏宽的，表格单元格永远不是——这是两者最稳的差别。
    if max(_fill_ratio(left), _fill_ratio(right)) < GUTTER_PROSE_FILL:
        return [lines]
    return [left, right] + ([cross] if cross else [])


def _fill_ratio(side: list[dict]) -> float:
    """该组行的「宽度中位数 / 该组横向跨度」。正文栏接近 1，表格列只有零点几。"""
    if not side:
        return 0.0
    span = max(l["x1"] for l in side) - min(l["x0"] for l in side)
    if span <= 0:
        return 0.0
    widths = sorted(l["x1"] - l["x0"] for l in side)
    return widths[len(widths) // 2] / span


def _merge_atoms(atoms: list[dict]) -> list[dict]:
    """原子按 x 排序，间隙小于阈值的并成一个单元格。"""
    ordered = sorted(atoms, key=lambda a: (round(a["x0"], 2), round(a["x1"], 2), a["text"]))
    cells: list[dict] = []
    for a in ordered:
        if cells:
            prev = cells[-1]
            thr = max(GAP_MIN_PT, GAP_SIZE_RATIO * max(prev["size"], a["size"], 1.0))
            if a["x0"] - prev["x1"] < thr:
                prev["text"] = (prev["text"] + " " + a["text"]).strip()
                prev["x1"] = max(prev["x1"], a["x1"])
                prev["size"] = max(prev["size"], a["size"])
                prev["bold"] = prev["bold"] or a["bold"]
                continue
        cells.append(dict(a))
    return cells


def _visual_rows(lines: list[dict]) -> list[dict]:
    """行聚类：y 区间重叠过半的行合成一「视觉行」（同一行的单元格常是各自独立的 line）。"""
    ordered = sorted(lines, key=lambda l: (round(l["y0"], 1), round(l["x0"], 1)))
    rows: list[dict] = []
    for ln in ordered:
        merged = False
        if rows:
            r = rows[-1]
            h = min(r["y1"] - r["y0"], ln["y1"] - ln["y0"])
            h = h if h > 0 else 1.0
            ov = min(r["y1"], ln["y1"]) - max(r["y0"], ln["y0"])
            if ov >= 0.5 * h:
                r["y0"] = min(r["y0"], ln["y0"])
                r["y1"] = max(r["y1"], ln["y1"])
                r["atoms"].extend(ln["atoms"])
                merged = True
        if not merged:
            rows.append({"y0": ln["y0"], "y1": ln["y1"], "atoms": list(ln["atoms"])})
    for r in rows:
        r["cells"] = _merge_atoms(r["atoms"])
    return rows


def _candidate_runs(rows: list[dict]) -> list[list[dict]]:
    """把「单元格数达标且垂直连续」的行聚成候选区。

    单元格数不够的行不是简单地打断候选区：真论文里「类别」列的标签常常竖直居中在
    若干行之间，自成一个只有 1 格的视觉行（跨行合并单元格的排版效果）。直接打断
    会把一张 40 行的表碎成三四段（真论文实测就是这样）。所以允许**最多连续
    MAX_SPARSE_ROWS 个稀疏行**夹在中间，但它们既不能开头也不能结尾——
    连续多个稀疏行就是正文，必须打断。
    """
    runs: list[list[dict]] = []
    cur: list[dict] = []
    pending: list[dict] = []
    for r in rows:
        if len(r["cells"]) >= RUN_MIN_CELLS:
            if cur:
                prev = (pending or cur)[-1]
                hh = max(prev["y1"] - prev["y0"], 1.0)
                if r["y0"] - prev["y1"] > ROW_GAP_RATIO * hh:
                    runs.append(cur)
                    cur, pending = [], []
            if cur:
                cur.extend(pending)
            pending = []
            cur.append(r)
        elif cur and len(pending) < MAX_SPARSE_ROWS:
            pending.append(r)
        else:
            if cur:
                runs.append(cur)
            cur, pending = [], []
    if cur:
        runs.append(cur)
    return runs


def _columns_of(run: list[dict]) -> list[list[float]]:
    """按 x 方向的单元格覆盖直方图切列：被极少数行穿过的 x 位置就是列间空白。

    不用「x 区间重叠就传递合并」：那样**一个偏宽的单元格就能把两列永久焊死**。
    真论文实测（一张 45 行的表）就栽在这里——三列被并成两列，整张表随后被判不合格。
    直方图法下，一个宽格只让分隔处的覆盖 +1，远够不上阈值。
    阈值随行数放宽（int(COL_SEP_RATIO * 行数)），小表退化成「有覆盖即成列」的老行为。
    """
    cells = [c for r in run for c in r["cells"]]
    if not cells:
        return []
    # 只用「正常行」建直方图：跨行合并的类别标签 / 居中的分节标题本来就横跨两列，
    # 让它们参与切列会把两列焊死（真论文实测：45 行的表因此只剩 2 列而被毙掉）。
    dense = [r for r in run if len(r["cells"]) >= RUN_MIN_CELLS]
    dense_cells = [c for r in dense for c in r["cells"]] or cells
    x_min = min(c["x0"] for c in cells)
    x_max = max(c["x1"] for c in cells)
    start, end = int(math.floor(x_min)), int(math.ceil(x_max))
    cov = [0] * (end - start + 1)
    for c in dense_cells:
        a = max(int(math.floor(c["x0"])), start)
        b = min(int(math.ceil(c["x1"])), end)
        for x in range(a, b + 1):
            cov[x - start] += 1

    thr = int(COL_SEP_RATIO * len(dense or run))
    cols: list[list[float]] = []
    cur: list[float] | None = None
    for i, v in enumerate(cov):
        if v > thr:
            x = float(start + i)
            cur = [x, x] if cur is None else [cur[0], x]
        elif cur is not None:
            cols.append(cur)
            cur = None
    if cur is not None:
        cols.append(cur)
    merged: list[list[float]] = []
    for col in cols:
        if merged and col[0] - merged[-1][1] <= COL_MERGE_PAD:
            merged[-1][1] = col[1]
        else:
            merged.append(col)

    # 稀疏行的单元格若压根不落在任何一列上（类别列常常只有它自己），补一列，
    # 否则那一列的内容会被塞进相邻列，网格与内容都错。
    for r in run:
        if len(r["cells"]) >= RUN_MIN_CELLS:
            continue
        for c in r["cells"]:
            if any(min(c["x1"], m[1]) - max(c["x0"], m[0]) > 0 for m in merged):
                continue
            merged.append([c["x0"], c["x1"]])
            merged.sort(key=lambda m: (m[0], m[1]))
    return merged


def _grid_of(run: list[dict], cols: list[list[float]]) -> tuple[list[list[str]], list[list[bool]]]:
    """单元格按「与列区间的交叠最大」归位；同一格落两个单元格时用空格接起来（不丢内容）。"""
    texts: list[list[str]] = []
    bolds: list[list[bool]] = []
    for r in run:
        row_t = [""] * len(cols)
        row_b = [False] * len(cols)
        for c in r["cells"]:
            best, best_ov = 0, -1.0
            for j, (cx0, cx1) in enumerate(cols):
                ov = min(c["x1"], cx1) - max(c["x0"], cx0)
                if ov > best_ov:
                    best, best_ov = j, ov
            row_t[best] = (row_t[best] + " " + c["text"]).strip() if row_t[best] else c["text"]
            row_b[best] = row_b[best] or c["bold"]
        texts.append(row_t)
        bolds.append(row_b)
    return texts, bolds


def _guess_header(rows: list[list[str]], bolds: list[list[bool]] | None) -> bool:
    """首行是不是表头。两条确定性规则，任一命中即算表头：

    A 粗体规则：首行非空单元格里粗体占比 >= 0.6，且其余行的粗体占比 < 0.4；
    B 数值规则：首行没有任何「像数字」的单元格，而半数以上的数据行至少有一个数字。
    两条都不命中就如实返回 False（宁可标「无表头」，也不假装有）。
    """
    if len(rows) < 2:
        return False
    head = [c for c in rows[0] if _clean(c)]
    if not head:
        return False

    if bolds:
        hb = [b for b, c in zip(bolds[0], rows[0]) if _clean(c)]
        rest = [b for i in range(1, len(rows))
                for b, c in zip(bolds[i], rows[i]) if _clean(c)]
        if hb and sum(hb) / len(hb) >= 0.6 and (not rest or sum(rest) / len(rest) < 0.4):
            return True

    if any(_numericish(c) for c in head):
        return False
    body = rows[1:]
    with_num = sum(1 for r in body if any(_numericish(c) for c in r))
    return bool(body) and with_num * 2 > len(body)


# ── ② 两条识别路径 ──

def _rows_from_extract(raw_rows) -> list[list[str]]:
    """find_tables().extract() 的行清洗：None → ""，统一换行符，保留单元格内换行。"""
    out: list[list[str]] = []
    for r in raw_rows or []:
        cells = []
        for v in r or []:
            s = "" if v is None else str(v)
            cells.append(s.replace("\r\n", "\n").replace("\r", "\n").strip())
        out.append(cells)
    return out


def _find_tables_path(page, page_no: int, bold_texts) -> tuple[list[dict], list[str]]:
    """路径①：pymupdf 自带 find_tables()。不存在或抛异常都优雅降级。"""
    warnings: list[str] = []
    fn = getattr(page, "find_tables", None)
    if not callable(fn):
        return [], ["pymupdf 版本没有 find_tables()，全部走几何启发式"]
    try:
        finder = fn()
        found = list(getattr(finder, "tables", None) or [])
    except Exception as e:
        return [], [f"第 {page_no} 页 find_tables() 失败（{e}），该页降级为几何启发式"]

    out: list[dict] = []
    for t in found:
        try:
            rows = _rows_from_extract(t.extract())
        except Exception as e:
            warnings.append(f"第 {page_no} 页某表 extract() 失败（{e}），已跳过")
            continue
        external = False
        try:
            hdr = getattr(t, "header", None)
            names = [str(x) for x in (getattr(hdr, "names", None) or [])]
            external = bool(getattr(hdr, "external", False))
            # header.external=True 表示表头在 bbox 之外、不在 extract() 里 —— 补回来，
            # 否则这张表的所有块都没有列名，正是本模块要消灭的情形。
            if external and any(n.strip() for n in names):
                rows = [names] + rows
        except Exception:
            external = False
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            continue
        n_cols = max(len(r) for r in rows)
        has_header = external or _guess_header(rows, _bold_grid(rows, bold_texts))
        out.append({
            "page_no": page_no,
            "bbox": _bbox(getattr(t, "bbox", (0, 0, 0, 0))),
            "n_rows": len(rows),
            "n_cols": n_cols,
            "rows": rows,
            "detected_by": "find_tables",
            "has_header": has_header,
        })
    return out, warnings


def _bold_grid(rows: list[list[str]], bold_texts) -> list[list[bool]] | None:
    """用「本页所有粗体文本」的集合给 find_tables 的行补粗体证据。

    find_tables 只给文本、不给字体，而「表头是粗的」是最强的表头信号。
    这里按归一化文本匹配——够用且确定性，代价是同一串文本在页面别处是粗体时会误判。
    """
    if not bold_texts:
        return None
    return [[_squash(c) in bold_texts for c in r] for r in rows]


def _geometry_tables(rows: list[dict], page_no: int) -> tuple[list[dict], list[str]]:
    """路径②：几何启发式。列对齐 + 行内多个明显水平间隙。"""
    warnings: list[str] = []
    out: list[dict] = []
    for run in _candidate_runs(rows):
        if len(run) < GEOM_MIN_ROWS:
            continue
        cols = _columns_of(run)
        if len(cols) < GEOM_MIN_COLS:
            continue
        texts, bolds = _grid_of(run, cols)
        used = [sum(1 for r in texts if r[j].strip()) for j in range(len(cols))]
        strong = sum(1 for u in used if u >= COL_STRONG_RATIO * len(texts))
        if strong < GEOM_MIN_STRONG_COLS or strong < COL_STRONG_FRACTION * len(cols):
            continue  # 网格太破碎，多半是参差的列表或对齐巧合，不认成表
        x0 = min(c["x0"] for r in run for c in r["cells"])
        x1 = max(c["x1"] for r in run for c in r["cells"])
        out.append({
            "page_no": page_no,
            "bbox": _bbox((x0, run[0]["y0"], x1, run[-1]["y1"])),
            "n_rows": len(texts),
            "n_cols": len(cols),
            "rows": texts,
            "detected_by": "geometry",
            "has_header": _guess_header(texts, bolds),
        })
    return out, warnings


def _detect_page(page, page_no: int) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    lines, err = _page_lines(page)
    if err:
        warnings.append(f"第 {page_no} 页版面读取失败：{err}")
    bold_texts = frozenset(
        _squash(a["text"]) for ln in lines for a in ln["atoms"]
        if a["bold"] and _squash(a["text"])
    )
    found, w1 = _find_tables_path(page, page_no, bold_texts)
    warnings.extend(w1)
    # 关键：先分栏再做行聚类。双栏页整页聚类会把左栏正文和右栏表格拼成同一行。
    geom: list[dict] = []
    for band in _column_bands(lines):
        t, w2 = _geometry_tables(_visual_rows(band), page_no)
        geom.extend(t)
        warnings.extend(w2)

    # 两条路径都命中同一张表时保留 find_tables 的（框线证据比几何猜测硬）。
    # 这是常态而非降级，所以不进 warnings——warnings 里只放「真的退化了」的事，
    # 否则每张有框线的表都刷一条，warnings 很快就没人看了。
    kept = [g for g in geom
            if not any(_overlap_frac(g["bbox"], f["bbox"]) >= DEDUPE_OVERLAP for f in found)]
    return found + kept, warnings


# ── ③ 公开函数 ──

def detect_tables(pdf_path, page_no: int | None = None) -> list[dict]:
    """识别 PDF 里的表格区域。page_no 传 1-based 页码则只看那一页。

    返回按 (页码, y, x) 排序的列表，每项：
    `{"page_no", "bbox", "n_rows", "n_cols", "rows": [[cell,...]],
      "detected_by": "find_tables"|"geometry", "has_header": bool}`
    PDF 打不开 / 加密抛 TableError；其余任何失败都降级为「这页没识别到表」。
    """
    tables, _ = _detect(pdf_path, page_no)
    return tables


def _detect(pdf_path, page_no: int | None = None) -> tuple[list[dict], list[str]]:
    doc = _open(pdf_path)
    warnings: list[str] = []
    out: list[dict] = []
    try:
        n_pages = int(getattr(doc, "page_count", 0) or 0)
        if n_pages <= 0:
            return [], ["PDF 页数为 0，没有可解析的页面"]
        if page_no is None:
            indices = list(range(n_pages))
        elif 1 <= int(page_no) <= n_pages:
            indices = [int(page_no) - 1]
        else:
            return [], [f"页码 {page_no} 越界（PDF 共 {n_pages} 页）"]
        for i in indices:
            try:
                page = doc[i]
            except Exception as e:
                warnings.append(f"第 {i + 1} 页打不开：{e}")
                continue
            try:
                t, w = _detect_page(page, i + 1)
            except Exception as e:  # 单页任何意外都不该毁掉整篇
                warnings.append(f"第 {i + 1} 页表格识别失败：{e}")
                continue
            out.extend(t)
            warnings.extend(w)
    finally:
        doc.close()
    # 全序排序键：同一 bbox 时用 detected_by 做 tiebreaker，避免跨进程漂移
    out.sort(key=lambda d: (d["page_no"], d["bbox"][1], d["bbox"][0],
                            d["bbox"][3], d["bbox"][2], d["detected_by"]))
    return out, warnings


# ── ④ Markdown 渲染 ──

def _md_cell(v) -> str:
    """单元格转义：反斜杠、竖线要转义，换行变 <br>，否则整张表在 markdown 里散架。"""
    if v is None:
        return ""
    s = unicodedata.normalize("NFKC", str(v)).replace("\u00ad", "")
    s = s.replace("\\", "\\\\").replace("|", "\\|")
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return _WS_RE.sub(" ", s).strip()


def _grid(rows, n_cols: int) -> list[list[str]]:
    """行规整成等宽网格并逐格转义：短行补空，超长行把多出来的并进最后一格（不丢内容）。"""
    out: list[list[str]] = []
    for r in rows or []:
        cells = [_md_cell(v) for v in (r or [])]
        if n_cols > 0 and len(cells) > n_cols:
            tail = " ".join(x for x in cells[n_cols - 1:] if x)
            cells = cells[:n_cols - 1] + [tail]
        cells += [""] * (n_cols - len(cells))
        out.append(cells)
    return out


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _n_cols_of(table: dict) -> int:
    """列数以 n_cols 声明的为准（那是识别阶段定下的网格），没声明才按最宽的行推。

    不取 max(声明, 最宽)：那样任何一个多出格子的脏行都会把整张表撑宽一列，
    其余行全被补上一列空格——网格该由识别阶段说了算，脏行按 _grid 并进最后一格。
    """
    declared = int(table.get("n_cols") or 0)
    if declared > 0:
        return declared
    return max((len(r or []) for r in (table.get("rows") or [])), default=0)


def table_to_markdown(table: dict) -> str:
    """表格 dict → Markdown 表格文本。

    表头缺失（has_header=False）时用第一行当表头，并在表格上方加一行说明——
    「借用的表头」必须让读到这段文本的人（和模型）知道，不能装作是真表头。
    """
    rows = table.get("rows") or []
    n_cols = _n_cols_of(table)
    if not rows or n_cols <= 0:
        return ""
    grid = _grid(rows, n_cols)
    has_header = bool(table.get("has_header"))
    lines: list[str] = []
    if not has_header:
        lines.append(NOTE_BORROWED_HEADER)
    lines.append(_md_row(grid[0]))
    lines.append(_md_row([_SEP_CELL] * n_cols))
    lines.extend(_md_row(r) for r in grid[1:])
    return "\n".join(lines)


# ── ⑤ 表格切分（本模块的核心）──

def chunk_table(table: dict, max_rows: int = 50, max_chars: int = 8000,
                repeat_header: bool = True) -> list[dict]:
    """把一张表切成若干 Markdown 块，**行级切分，绝不从行中间截断**。

    - 双限：数据行数达到 max_rows、或字符数达到 max_chars，任一触顶即切；
      max_rows 只数**数据行**，表头不计入（表头是每块都要带的固定开销）。
    - 一行本身就超过 max_chars 时，该行整行单独成一块并在块里记 `warning`，
      **宁可超限也不截断**——半行表格数据比没有更糟，它会被当成完整数据引用。
    - repeat_header=True 时，第 2 块起重复表头行并标 `header_repeated=True`；
      这正是本功能的全部意义：第 2 块被单独检索到时仍然读得懂列名。
      repeat_header=False 时续块用空表头行占位，保证列数与可解析性不变。

    返回 `[{"text"(markdown), "n_rows"(该块数据行数), "part", "of", "header_repeated"}]`，
    超长行的块额外带 `"warning"`。空表返回 []。
    """
    rows = table.get("rows") or []
    n_cols = _n_cols_of(table)
    if not rows or n_cols <= 0:
        return []
    max_rows = max(1, int(max_rows))
    max_chars = max(1, int(max_chars))

    grid = _grid(rows, n_cols)
    has_header = bool(table.get("has_header"))
    header_md = _md_row(grid[0])
    sep_md = _md_row([_SEP_CELL] * n_cols)
    empty_md = _md_row([""] * n_cols)
    body_md = [_md_row(r) for r in grid[1:]]

    def head_block(first: bool) -> str:
        lines: list[str] = []
        if first:
            if not has_header:
                lines.append(NOTE_BORROWED_HEADER)
            lines.append(header_md)
        else:
            lines.append(NOTE_CONT_WITH_HEADER if repeat_header else NOTE_CONT_NO_HEADER)
            lines.append(header_md if repeat_header else empty_md)
        lines.append(sep_md)
        return "\n".join(lines)

    def block_len(body: list[str], first: bool) -> int:
        return len(head_block(first)) + sum(1 + len(r) for r in body)

    if not body_md:  # 只有表头一行的表：仍然出一块，如实记 n_rows=0
        return [{"text": head_block(True), "n_rows": 0, "part": 1, "of": 1,
                 "header_repeated": False}]

    packed: list[dict] = []
    cur: list[str] = []
    for i, rmd in enumerate(body_md, 1):
        first = not packed
        if cur and (len(cur) + 1 > max_rows or block_len(cur + [rmd], first) > max_chars):
            packed.append({"rows": cur, "warning": None})
            cur = []
            first = False
        if not cur and block_len([rmd], first) > max_chars:
            packed.append({"rows": [rmd], "warning": (
                f"第 {i} 个数据行本身长 {len(rmd)} 字符，超过 max_chars={max_chars}："
                f"整行单独成块，不截断（截断会产出看起来完整的半行数据）")})
            continue
        cur.append(rmd)
    if cur:
        packed.append({"rows": cur, "warning": None})

    total = len(packed)
    out: list[dict] = []
    for k, item in enumerate(packed, 1):
        first = k == 1
        chunk = {
            "text": head_block(first) + "\n" + "\n".join(item["rows"]),
            "n_rows": len(item["rows"]),
            "part": k,
            "of": total,
            "header_repeated": (not first) and repeat_header,
        }
        if item["warning"]:
            chunk["warning"] = item["warning"]
        out.append(chunk)
    return out


# ── ⑥ 一页拆成「表格块」+「正文块」 ──

def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """超长段落：先按句号切，再不行才硬切（硬切是最后手段，会切断句子）。"""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    cur = ""
    for sent in _SENT_RE.split(text):
        if not sent:
            continue
        cand = (cur + " " + sent).strip() if cur else sent
        if cur and len(cand) > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = cand
    if cur:
        out.append(cur)
    final: list[str] = []
    for piece in out:
        while len(piece) > max_chars:
            final.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            final.append(piece)
    return final


def _text_chunks(text: str, max_chars: int) -> list[str]:
    """正文按段落边界打包，尽量不在段中间切。"""
    paras = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paras:
        stripped = text.strip()
        paras = [stripped] if stripped else []
    out: list[str] = []
    cur = ""
    for p in paras:
        for piece in _split_paragraph(p, max_chars):
            cand = (cur + "\n\n" + piece) if cur else piece
            if cur and len(cand) > max_chars:
                out.append(cur)
                cur = piece
            else:
                cur = cand
    if cur:
        out.append(cur)
    return out


def _line_owner(line: str, tokens: list[tuple[frozenset, frozenset]]) -> int | None:
    """这行正文属于第几张表：整行等于某个单元格，或命中该表 >=2 个单元格且覆盖过半。

    两条规则各有分工：抽取器把每个单元格吐成独立一行时走前者（一行 = 一格），
    把整行表格吐成一行时走后者（一行 = 一整排格）。都不命中就是正文。

    覆盖率这一条是必须的：只数命中个数，一句正文里偶然含 "recall"、"full" 两个词
    就会被判进表格区、从正文里被摘走。要求命中的单元格文本占该行一半以上字符，
    才排除得掉这种偶然。
    """
    s = _squash(line)
    if not s:
        return None
    best, best_score = None, 0
    for i, (exact, multi) in enumerate(tokens):
        score = 0
        if s in exact:
            score = len(s) + 1  # 整行等于一个单元格：最强证据
        hits = [t for t in multi if t in s]
        if len(hits) >= 2 and sum(len(t) for t in hits) >= LINE_COVER_RATIO * len(s):
            score = max(score, sum(len(t) for t in hits))
        if score > best_score:
            best, best_score = i, score
    return best


def split_page_text(page_text: str, tables: list[dict], max_chars: int = 4000) -> list[dict]:
    """把一页文本拆成「表格块」与「正文块」，两者绝不混在同一个块里。

    tables 传该页的表（detect_tables 的结果按 page_no 过滤）。表格区域的行被从正文里摘出来，
    走 chunk_table 单独成块；剩下的正文按段落边界切。表格在正文里的**出现位置**决定块序，
    完全匹配不上的表统一追加在末尾（宁可位置不准，也不丢表）。

    返回 `[{"kind": "table"|"text", "text", "index", ...}]`：
    text 块带 `n_chars`；table 块带 `table_index`/`part`/`of`/`n_rows`/`header_repeated`
    （超长行的块还带 `warning`）。
    """
    tables = list(tables or [])
    tokens: list[tuple[frozenset, frozenset]] = []
    for t in tables:
        cells = [_squash(c) for r in (t.get("rows") or []) for c in (r or [])]
        cells = [c for c in cells if c]
        tokens.append((frozenset(cells), frozenset(c for c in cells if len(c) >= 2)))

    lines = (page_text or "").splitlines()
    owners: list[int | None] = [_line_owner(ln, tokens) if ln.strip() else None
                                for ln in lines]
    # 空行夹在同一张表的两段之间时并进表区，否则一张表会被空行劈成两段
    for i, ln in enumerate(lines):
        if ln.strip() or owners[i] is not None:
            continue
        prev = next((owners[j] for j in range(i - 1, -1, -1) if lines[j].strip()), None)
        nxt = next((owners[j] for j in range(i + 1, len(lines)) if lines[j].strip()), None)
        if prev is not None and prev == nxt:
            owners[i] = prev

    segments: list[tuple[int | None, list[str]]] = []
    for owner, ln in zip(owners, lines):
        if segments and segments[-1][0] == owner:
            segments[-1][1].append(ln)
        else:
            segments.append((owner, [ln]))

    blocks: list[dict] = []
    emitted = [False] * len(tables)

    def add_table(i: int):
        if emitted[i]:
            return
        emitted[i] = True
        for ch in chunk_table(tables[i], max_chars=max(max_chars, 1)):
            blk = {"kind": "table", "text": ch["text"], "table_index": i,
                   "part": ch["part"], "of": ch["of"], "n_rows": ch["n_rows"],
                   "header_repeated": ch["header_repeated"]}
            if "warning" in ch:
                blk["warning"] = ch["warning"]
            blocks.append(blk)

    for owner, seg in segments:
        if owner is None:
            for piece in _text_chunks("\n".join(seg), max_chars):
                blocks.append({"kind": "text", "text": piece, "n_chars": len(piece)})
        else:
            add_table(owner)
    for i in range(len(tables)):
        add_table(i)  # 一行都没匹配上的表：追加在末尾，不丢

    for k, b in enumerate(blocks):
        b["index"] = k
    return blocks


# ── ⑦ 汇总 ──

def summarize(pdf_path) -> dict:
    """全文表格识别的可审计汇总。

    返回 `{"n_tables", "pages_with_tables", "by_detector", "warnings", "n_pages", "degraded"}`。
    degraded=True 表示过程中有降级（有 warnings），调用方要把它照实展示出去。
    """
    tables, warnings = _detect(pdf_path)
    by_detector = {"find_tables": 0, "geometry": 0}
    pages: list[int] = []
    for t in tables:
        key = t.get("detected_by") or "geometry"
        by_detector[key] = by_detector.get(key, 0) + 1
        if t["page_no"] not in pages:
            pages.append(t["page_no"])
    n_pages = 0
    try:
        doc = _open(pdf_path)
        try:
            n_pages = int(getattr(doc, "page_count", 0) or 0)
        finally:
            doc.close()
    except TableError:
        pass
    return {
        "n_tables": len(tables),
        "pages_with_tables": sorted(pages),
        "by_detector": by_detector,
        "warnings": warnings,
        "n_pages": n_pages,
        "degraded": bool(warnings),
    }
