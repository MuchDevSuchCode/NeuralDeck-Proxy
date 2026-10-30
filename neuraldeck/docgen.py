"""Files the chat can hand back: documents rendered from Markdown (PDF,
Word, HTML, Markdown, plain text), spreadsheets, and plain files of any
kind. Each function returns the bytes; the tools layer names and stores
them under the account's generated folder.

PDF comes from fpdf2 with DejaVu fonts when the system has them (so any
script renders), Word from python-docx with a small Markdown walker, and
spreadsheets from openpyxl. Every library is optional: a missing one
fails only the format that needs it.
"""

import html as _html
import io
import os
import re

__all__ = ["slug", "render_document", "render_spreadsheet", "FORMATS", "DocError"]

FORMATS = ("pdf", "docx", "html", "md", "txt")
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF")
MAX_CELLS = 200_000


class DocError(ValueError):
    pass


def slug(name: str, fallback: str = "file", limit: int = 48) -> str:
    """A safe file stem from whatever the model called the file."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    s = re.sub(r"-{2,}", "-", s)[:limit].rstrip("-.")
    return s or fallback


# ── Markdown → HTML ─────────────────────────────────────────────────────────

def _md_html(text: str) -> str:
    try:
        import markdown
    except ImportError:
        raise DocError("the markdown package is not installed (pip install markdown)")
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html")


def render_html(title: str, md: str) -> bytes:
    body = _md_html(md)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
body{{font:16px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#1c2129;background:#fff;max-width:820px;margin:40px auto;padding:0 20px}}
h1,h2,h3{{line-height:1.25}} h1{{font-size:2em;border-bottom:1px solid #e3e6ea;padding-bottom:.3em}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}}
pre{{background:#f4f6f8;padding:12px 14px;border-radius:6px;overflow:auto}} code{{background:#f4f6f8;padding:1px 4px;border-radius:3px}} pre code{{background:none;padding:0}}
table{{border-collapse:collapse;margin:1em 0}} th,td{{border:1px solid #d7dbe0;padding:6px 10px;text-align:left}} th{{background:#f4f6f8}}
blockquote{{border-left:4px solid #d7dbe0;margin:1em 0;padding:.2em 1em;color:#555}}
img{{max-width:100%}}
</style></head><body>
{body}
</body></html>
"""
    return doc.encode("utf-8")


# ── PDF ─────────────────────────────────────────────────────────────────────

def _font_file(name: str):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def render_pdf(title: str, md: str) -> bytes:
    try:
        from fpdf import FPDF, FontFace
        try:
            from fpdf.fonts import TextStyle
        except ImportError:   # older fpdf2
            TextStyle = FontFace
    except ImportError:
        raise DocError("the fpdf2 package is not installed (pip install fpdf2)")
    body = _md_html(md)

    class PDF(FPDF):
        def footer(self):
            self.set_y(-14)
            self.set_font(self._face, size=8)
            self.set_text_color(120)
            self.cell(0, 8, f"{title}  ·  {self.page_no()}/{{nb}}", align="C")

    pdf = PDF(format="A4")
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(True, margin=20)
    pdf.alias_nb_pages()
    pdf.set_title(title)
    pdf.set_creator("NeuralDeck")
    styles = {}
    reg = _font_file("DejaVuSans.ttf")
    if reg:
        pdf.add_font("DejaVu", "", reg)
        pdf.add_font("DejaVu", "B", _font_file("DejaVuSans-Bold.ttf") or reg)
        pdf.add_font("DejaVu", "I", _font_file("DejaVuSans-Oblique.ttf") or reg)
        pdf.add_font("DejaVu", "BI", _font_file("DejaVuSans-BoldOblique.ttf") or reg)
        mono = _font_file("DejaVuSansMono.ttf")
        if mono:
            pdf.add_font("DejaVuMono", "", mono)
            pdf.add_font("DejaVuMono", "B", _font_file("DejaVuSansMono-Bold.ttf") or mono)
            styles = {"code": FontFace(family="DejaVuMono", size_pt=9), "pre": FontFace(family="DejaVuMono", size_pt=9)}
        face = "DejaVu"
    else:
        face = "Helvetica"
        body = body.encode("latin-1", "replace").decode("latin-1")
    pdf._face = face
    ink, dim = (28, 33, 41), (90, 96, 104)
    mono_face = "DejaVuMono" if styles else "Courier"

    def ts(family=face, emphasis=None, size=None, color=ink, **kw):
        # fpdf2 names these differently across versions
        try:
            return TextStyle(font_family=family, font_style=emphasis, font_size_pt=size, color=color, **kw)
        except TypeError:
            return TextStyle(family=family, emphasis=emphasis, size_pt=size, color=color, **kw)

    def hs(size, top, bottom=1.5):
        return ts(emphasis="B", size=size, t_margin=top, b_margin=bottom)
    styles.update({
        "h1": hs(22, 6, 2.5), "h2": hs(17, 6, 2), "h3": hs(14, 5), "h4": hs(12, 4), "h5": hs(11, 3), "h6": hs(11, 3),
        "p": ts(size=11, t_margin=0, b_margin=2.2),
        "li": ts(size=11, b_margin=0.6),
        "blockquote": ts(emphasis="I", size=11, color=dim, l_margin=6, t_margin=2, b_margin=2),
        "pre": ts(family=mono_face, size=9, fill_color=(243, 245, 247), l_margin=3, t_margin=3, b_margin=4),
        "code": ts(family=mono_face, size=9.5),
        "a": ts(color=(20, 90, 180)),
    })
    pdf.set_font(face, size=11)
    pdf.set_text_color(*ink)
    pdf.add_page()
    pdf.write_html(body, font_family=face, tag_styles=styles, table_line_separators=True)
    return bytes(pdf.output())


# ── Word ────────────────────────────────────────────────────────────────────

_INLINE = re.compile(r"(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]\n]+\]\([^)\s]+\)|~~[^~\n]+~~)")


def _docx_inline(par, text, mono_name="Consolas"):
    from docx.shared import Pt
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") or part.startswith("__") and part.endswith("__"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("~~") and part.endswith("~~"):
            par.add_run(part[2:-2]).font.strike = True
        elif (part.startswith("*") and part.endswith("*") or part.startswith("_") and part.endswith("_")) and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = mono_name
            r.font.size = Pt(9.5)
        elif part.startswith("[") and "](" in part:
            label, url = part[1:-1].split("](", 1)
            r = par.add_run(label)
            r.underline = True
            par.add_run(f" ({url})")
        else:
            par.add_run(part)


def render_docx(title: str, md: str) -> bytes:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        raise DocError("the python-docx package is not installed (pip install python-docx)")
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    if title and not re.match(r"^\s*#\s", md):
        doc.add_heading(title, 0)
    lines = md.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)
    para_buf = []

    def flush():
        if para_buf:
            _docx_inline(doc.add_paragraph(), " ".join(x.strip() for x in para_buf))
            para_buf.clear()

    while i < n:
        line = lines[i]
        s = line.strip()
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            flush()
            mark = fence.group(1)[0]
            i += 1
            code = []
            while i < n and not re.match(rf"^\s*{re.escape(mark)}{{3,}}\s*$", lines[i]):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(code))
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            p.paragraph_format.left_indent = Pt(12)
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", s)
        if m:
            flush()
            _docx_inline(doc.add_heading("", min(len(m.group(1)), 4)), m.group(2))
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", s):
            flush()
            doc.add_paragraph()
            i += 1
            continue
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^\s*:?-{2,}", cells[0]):
                    rows.append(cells)
                i += 1
            if rows:
                width = max(len(r) for r in rows)
                t = doc.add_table(rows=len(rows), cols=width)
                t.style = "Table Grid"
                for ri, r in enumerate(rows):
                    for ci in range(width):
                        cell = t.cell(ri, ci)
                        cell.text = ""
                        _docx_inline(cell.paragraphs[0], r[ci] if ci < len(r) else "")
                        if ri == 0:
                            for run in cell.paragraphs[0].runs:
                                run.bold = True
                doc.add_paragraph()
            continue
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            flush()
            depth = min(len(m.group(1).replace("\t", "    ")) // 2, 2)
            kind = "List Number" if m.group(2)[0].isdigit() else "List Bullet"
            style = kind + (f" {depth + 1}" if depth else "")
            try:
                p = doc.add_paragraph(style=style)
            except KeyError:
                p = doc.add_paragraph(style=kind)
            _docx_inline(p, m.group(3))
            i += 1
            continue
        if s.startswith(">"):
            flush()
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            try:
                p = doc.add_paragraph(style="Quote")
            except KeyError:
                p = doc.add_paragraph()
            _docx_inline(p, " ".join(quote))
            continue
        para_buf.append(line)
        i += 1
    flush()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── the document entry point ────────────────────────────────────────────────

def render_document(title: str, md: str, fmt: str) -> tuple:
    """(bytes, extension) for a Markdown document in the given format."""
    fmt = (fmt or "pdf").lower().lstrip(".")
    if fmt == "markdown":
        fmt = "md"
    if fmt == "text":
        fmt = "txt"
    if fmt not in FORMATS:
        raise DocError("format must be one of " + ", ".join(FORMATS))
    if fmt == "pdf":
        return render_pdf(title, md), ".pdf"
    if fmt == "docx":
        return render_docx(title, md), ".docx"
    if fmt == "html":
        return render_html(title, md), ".html"
    if fmt == "md":
        head = f"# {title}\n\n" if title and not md.lstrip().startswith("#") else ""
        return (head + md).encode("utf-8"), ".md"
    return _plain(title, md).encode("utf-8"), ".txt"


def _plain(title: str, md: str) -> str:
    """Markdown with the markup taken off, for a .txt."""
    t = re.sub(r"^\s*(`{3,}|~{3,}).*$", "", md, flags=re.M)
    t = re.sub(r"^(#{1,6})\s+(.*?)\s*#*$", lambda m: m.group(2).upper() if len(m.group(1)) == 1 else m.group(2), t, flags=re.M)
    t = re.sub(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)|(?<!\w)_([^_\n]+)_(?!\w)", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"`([^`\n]+)`", r"\1", t)
    t = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", r"\1 (\2)", t)
    t = re.sub(r"^\s*\|?\s*:?-{2,}[-|: ]*$\n?", "", t, flags=re.M)
    t = re.sub(r"^\s*\|(.*)\|\s*$", lambda m: "  ".join(c.strip() for c in m.group(1).split("|")), t, flags=re.M)
    return ((title.upper() + "\n" + "=" * len(title) + "\n\n") if title else "") + t.strip() + "\n"


# ── spreadsheets ────────────────────────────────────────────────────────────

_NUM = re.compile(r"^-?\d{1,15}(\.\d+)?$")


def render_spreadsheet(sheets: list, header: bool = True) -> bytes:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise DocError("the openpyxl package is not installed (pip install openpyxl)")
    if not sheets:
        raise DocError("give at least one sheet with rows")
    wb = Workbook()
    wb.remove(wb.active)
    cells = 0
    used = set()
    for si, sh in enumerate(sheets[:50]):
        rows = sh.get("rows") if isinstance(sh, dict) else sh
        if not isinstance(rows, list):
            raise DocError("each sheet needs a rows list (a list of lists of cells)")
        name = re.sub(r"[\[\]:*?/\\]", " ", str((sh.get("name") if isinstance(sh, dict) else "") or f"Sheet{si + 1}")).strip()[:31] or f"Sheet{si + 1}"
        base, k = name, 2
        while name.lower() in used:
            name, k = f"{base[:28]} {k}", k + 1
        used.add(name.lower())
        ws = wb.create_sheet(name)
        widths = {}
        for ri, row in enumerate(rows, 1):
            if not isinstance(row, list):
                row = [row]
            for ci, v in enumerate(row, 1):
                cells += 1
                if cells > MAX_CELLS:
                    raise DocError(f"a workbook is limited to {MAX_CELLS} cells")
                if isinstance(v, str) and _NUM.match(v.strip()):
                    v = float(v) if "." in v else int(v)
                elif isinstance(v, (dict, list)):
                    v = str(v)
                c = ws.cell(row=ri, column=ci, value=v)
                if header and ri == 1:
                    c.font = Font(bold=True)
                widths[ci] = max(widths.get(ci, 0), min(60, len(str(v if v is not None else ""))))
        for ci, w in widths.items():
            ws.column_dimensions[get_column_letter(ci)].width = max(8, w + 2)
        if header and rows:
            ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
