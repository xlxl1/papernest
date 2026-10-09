"""Small dependency-free LaTeX to Word OMML bridge.

It intentionally handles the notation most often emitted by the writing agent
and keeps the original source available when a construct is unfamiliar.
The resulting ``m:oMath`` is a native Word equation and can be edited/imported
by MathType.
"""
from __future__ import annotations

from docx.oxml import OxmlElement


GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ",
    "pi": "π", "sigma": "σ", "phi": "φ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω",
}
OPS = {"times": "×", "cdot": "·", "leq": "≤", "geq": "≥",
       "neq": "≠", "approx": "≈", "infty": "∞", "pm": "±",
       "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬",
       "cup": "∪", "cap": "∩"}
SKIP_COMMANDS = {"left", "right", "middle", "displaystyle", "limits"}


def _r(text: str):
    r = OxmlElement("m:r")
    t = OxmlElement("m:t")
    t.text = text
    r.append(t)
    return r


def _e(tag: str, *children):
    x = OxmlElement("m:" + tag)
    for child in children:
        x.append(child)
    return x


def _group(s: str, i: int) -> tuple[str, int]:
    if i >= len(s) or s[i] != "{":
        return (s[i:i + 1], i + 1)
    depth, j = 1, i + 1
    while j < len(s) and depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return s[i + 1:j - 1], j


def _atoms(s: str):
    out, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == "{":
            val, i = _group(s, i)
            out.append(("group", val)); continue
        if s[i] == "\\":
            j = i + 1
            while j < len(s) and s[j].isalpha(): j += 1
            name = s[i + 1:j]
            if name in SKIP_COMMANDS:
                i = j
                continue
            out.append(("text", GREEK.get(name, OPS.get(name, name))))
            i = j; continue
        if s[i] in "^_":
            marker = s[i]
            val, i = _group(s, i + 1) if i + 1 < len(s) and s[i + 1] == "{" else (s[i + 1:i + 2], i + 2)
            out.append(("sup" if marker == "^" else "sub", val))
            continue
        out.append(("text", s[i])); i += 1
    return out


def _content(s: str):
    # OMML matrices are a native m:m container.  Parse the common LaTeX
    # matrix environments before tokenising so '&' and '\\' remain cell/row
    # separators rather than literal equation text.
    import re
    matrix = re.fullmatch(r"\\begin\{(?:matrix|pmatrix|bmatrix|cases)\}(.*?)\\end\{(?:matrix|pmatrix|bmatrix|cases)\}",
                          s.strip(), flags=re.S)
    if matrix:
        body = matrix.group(1).strip()
        rows = re.split(r"\\\\", body)
        mm = _e("m")
        for row in rows:
            if not row.strip():
                continue
            mr = _e("mr")
            for cell in row.split("&"):
                mr.append(_e("e", *_content(cell.strip())))
            mm.append(mr)
        return [mm]
    atoms = _atoms(s)
    out, i = [], 0
    while i < len(atoms):
        kind, val = atoms[i]
        if kind == "text" and val == "frac" and i + 2 < len(atoms):
            # \frac is represented as the command text followed by two groups.
            if atoms[i + 1][0] == "group" and atoms[i + 2][0] == "group":
                out.append(_e("f", _e("num", *_content(atoms[i + 1][1])),
                               _e("den", *_content(atoms[i + 2][1]))))
                i += 3; continue
        if kind == "text" and val in {"∑", "∏", "∫", "∬", "∪", "∩"}:
            # Native n-ary operator with optional lower/upper limits.  The
            # preceding tokenizer represents x_i and x^j as sub/sup atoms,
            # which can be attached directly to m:nary.
            props = _e("naryPr", _e("chr",))
            props[0].set("{http://schemas.openxmlformats.org/officeDocument/2006/math}val", val)
            children = [props]
            j = i + 1
            lower = upper = None
            while j < len(atoms) and atoms[j][0] in {"sub", "sup"}:
                if atoms[j][0] == "sub": lower = atoms[j][1]
                else: upper = atoms[j][1]
                j += 1
            if lower is not None: children.append(_e("sub", *_content(lower)))
            if upper is not None: children.append(_e("sup", *_content(upper)))
            children.append(_e("e"))
            out.append(_e("nary", *children))
            i = j
            continue
        if kind == "sup" and out:
            base = out.pop()
            out.append(_e("sSup", _e("e", base), _e("sup", *_content(val))))
        elif kind == "sub" and out:
            base = out.pop()
            out.append(_e("sSub", _e("e", base), _e("sub", *_content(val))))
        elif kind == "group":
            out.extend(_content(val))
        else:
            out.append(_r(val))
        i += 1
    return out


def omml_paragraph(latex: str):
    """Return a Word paragraph containing a native OMML equation."""
    latex = (latex or "").strip()
    if latex.startswith("$$") and latex.endswith("$$"):
        latex = latex[2:-2].strip()
    elif latex.startswith("\\(") and latex.endswith("\\)"):
        latex = latex[2:-2].strip()
    p = OxmlElement("w:p")
    omath = _e("oMath", *_content(latex))
    p.append(omath)
    return p


__all__ = ["omml_paragraph"]
