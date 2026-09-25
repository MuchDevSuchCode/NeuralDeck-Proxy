#!/usr/bin/env python3
"""Build docs/index.html from README.md, for GitHub Pages (Settings → Pages →
Deploy from branch → main, /docs).

The README stays the single source: this script renders it into a responsive
single-page site, so the two never drift. Run it after editing the README:

    python3 docs/build.py

Needs markdown-it-py (`pip install markdown-it-py`; it ships with `rich`, so
it is often installed already). The output is one self-contained HTML file
with no external requests.
"""

import html
import re
import sys
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:
    sys.exit("docs/build.py needs markdown-it-py:  pip install markdown-it-py")

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/MuchDevSuchCode/NeuralDeck-Proxy"


def slug(text: str) -> str:
    """GitHub's heading anchors, so the README's own #links keep working."""
    s = re.sub(r"<[^>]+>", "", text)
    s = html.unescape(s).strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def split_readme(md: str):
    """(title, intro markdown, body markdown). The intro is everything before
    the first horizontal rule; the README's own Contents section is dropped,
    since the site builds its navigation from the headings."""
    title = re.search(r"^# (.+)$", md, re.M).group(1).strip()
    after_title = md.split("\n", 1)[1]
    intro, _, body = after_title.partition("\n---\n")
    body = re.sub(r"^## Contents\n.*?(?=^## )", "", body, flags=re.S | re.M)
    body = body.replace("\n---\n", "\n")
    return title, intro.strip(), body.strip()


def render(md: str) -> str:
    mdi = MarkdownIt("commonmark", {"html": True, "typographer": False}) \
        .enable("table").enable("strikethrough")
    out = mdi.render(md)

    # anchors on h2-h4, GitHub-style, de-duplicated like GitHub does
    seen = {}

    def anchor(m):
        level, inner = m.group(1), m.group(2)
        base = slug(inner)
        n = seen.get(base, 0)
        seen[base] = n + 1
        sid = base if n == 0 else f"{base}-{n}"
        return (f'<h{level} id="{sid}"><a class="hash" href="#{sid}" '
                f'aria-label="Link to this section">#</a>{inner}</h{level}>')
    out = re.sub(r"<h([234])>(.*?)</h\1>", anchor, out, flags=re.S)

    # wide tables scroll inside their own box, never the page; each cell also
    # carries its column name, so phones can show a row as a labelled card
    def label_cells(m):
        table = m.group(0)
        heads = [re.sub(r"<[^>]+>", "", h).strip()
                 for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]

        def row(r):
            cells = iter(heads)
            # one wrapper per cell: a phone lays a cell out as a two-column
            # grid (label, value), and loose text nodes and <code> runs would
            # otherwise each become a grid item of their own
            return re.sub(r"<td([^>]*)>(.*?)</td>",
                          lambda c: f'<td{c.group(1)} data-label="'
                                    f'{html.escape(next(cells, ""), quote=True)}">'
                                    f'<span class="cell">{c.group(2)}</span></td>',
                          r.group(0), flags=re.S)
        table = re.sub(r"<tr>.*?</tr>", row, table, flags=re.S)
        return f'<div class="table-wrap">{table}</div>'
    out = re.sub(r"<table>.*?</table>", label_cells, out, flags=re.S)

    # relative links (LICENSE, files) point at the repository
    def fix_link(m):
        href = m.group(1)
        if href.startswith(("#", "http://", "https://", "mailto:")):
            return m.group(0)
        return f'href="{REPO}/blob/main/{href}"'
    out = re.sub(r'href="([^"]+)"', fix_link, out)
    return out


def features(intro_md: str) -> str:
    """The intro's bold-led bullets, as cards."""
    cards = []
    for m in re.finditer(r"^\* \*\*(.+?)\*\*:\s*(.+?)(?=^\* |\Z|^\s*$)",
                         intro_md, re.S | re.M):
        name = m.group(1).strip()
        text = " ".join(m.group(2).split())
        cards.append(f'<div class="card"><h3>{html.escape(name)}</h3>'
                     f'<p>{html.escape(text)}</p></div>')
    return "\n".join(cards)


def lead(intro_md: str) -> str:
    first = intro_md.split("\n\n", 1)[0]
    mdi = MarkdownIt("commonmark")
    return mdi.renderInline(" ".join(first.split()))


def notice(intro_md: str) -> str:
    m = re.search(r"^> (.+?)(?=\n[^>]|\Z)", intro_md, re.S | re.M)
    if not m:
        return ""
    text = re.sub(r"\n>\s?", " ", m.group(1))
    return MarkdownIt("commonmark").renderInline(" ".join(text.split()))


def nav(body_html: str) -> str:
    """Sidebar: every h2, with its h3s nested beneath it."""
    items, current = [], None
    for m in re.finditer(r'<h([23]) id="([^"]+)"><a[^>]*>#</a>(.*?)</h\1>',
                         body_html, re.S):
        level, sid, inner = m.group(1), m.group(2), m.group(3)
        label = re.sub(r"<[^>]+>", "", inner)
        if level == "2":
            current = {"id": sid, "label": label, "subs": []}
            items.append(current)
        elif current is not None:
            current["subs"].append({"id": sid, "label": label})
    parts = ['<ul class="toc">']
    for it in items:
        parts.append(f'<li><a href="#{it["id"]}">{it["label"]}</a>')
        if it["subs"]:
            parts.append('<ul>' + "".join(
                f'<li><a href="#{s["id"]}">{s["label"]}</a></li>'
                for s in it["subs"]) + '</ul>')
        parts.append("</li>")
    parts.append("</ul>")
    return "".join(parts)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · documentation</title>
<meta name="description" content="__DESC__">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1420'/%3E%3Cpath d='M7 22V10l9 7 9-7v12' fill='none' stroke='%2300d4c4' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E">
<script>
  // apply a saved theme before first paint
  try { const t = localStorage.getItem("nd-docs-theme");
        if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; } catch (e) {}
</script>
<style>
  :root {
    --bg: #ffffff; --bg-2: #f6f8fa; --panel: #ffffff; --ink: #1f2328; --ink-2: #3b4148;
    --muted: #59636e; --line: #d8dee4; --accent: #0a7d76; --accent-ink: #ffffff;
    --accent-soft: rgba(10,125,118,.10); --pink: #b3306f; --code-bg: #f3f5f7;
    --note-bg: #fff8e6; --note-line: #d9a400; --shadow: 0 1px 3px rgba(31,35,40,.08);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0b1017; --bg-2: #0f1620; --panel: #111a26; --ink: #e8f1f8; --ink-2: #c3d2de;
      --muted: #8ea2b4; --line: #223245; --accent: #2fd6c9; --accent-ink: #062521;
      --accent-soft: rgba(47,214,201,.10); --pink: #ff5aa8; --code-bg: #0d151f;
      --note-bg: #1f1a0c; --note-line: #c99a1a; --shadow: none;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0b1017; --bg-2: #0f1620; --panel: #111a26; --ink: #e8f1f8; --ink-2: #c3d2de;
    --muted: #8ea2b4; --line: #223245; --accent: #2fd6c9; --accent-ink: #062521;
    --accent-soft: rgba(47,214,201,.10); --pink: #ff5aa8; --code-bg: #0d151f;
    --note-bg: #1f1a0c; --note-line: #c99a1a; --shadow: none;
  }
  * { box-sizing: border-box; }
  html { scroll-padding-top: 72px; -webkit-text-size-adjust: 100%; }
  body { margin: 0; background: var(--bg); color: var(--ink-2);
         font: 16px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, sans-serif; }
  a { color: var(--accent); text-underline-offset: 2px; }
  code, pre, kbd { font-family: ui-monospace, "Cascadia Mono", "JetBrains Mono", Menlo, Consolas, monospace; }

  /* top bar */
  .topbar { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 12px;
            height: 56px; padding: 0 16px; background: color-mix(in srgb, var(--bg) 88%, transparent);
            backdrop-filter: saturate(1.4) blur(8px); border-bottom: 1px solid var(--line); }
  .brand { display: flex; align-items: center; gap: 10px; color: var(--ink); font-weight: 700;
           text-decoration: none; letter-spacing: .5px; }
  .brand svg { width: 26px; height: 26px; flex: none; }
  .topbar .spacer { flex: 1; }
  .iconbtn { display: inline-flex; align-items: center; justify-content: center; gap: 6px;
             height: 36px; min-width: 36px; padding: 0 10px; border: 1px solid var(--line);
             border-radius: 8px; background: var(--panel); color: var(--ink-2); font: inherit;
             font-size: 14px; cursor: pointer; text-decoration: none; }
  .iconbtn:hover { border-color: var(--accent); color: var(--ink); }
  .iconbtn svg { width: 18px; height: 18px; }
  #menu-btn { display: none; }

  /* hero */
  .hero { border-bottom: 1px solid var(--line);
          background: radial-gradient(1100px 380px at 12% -10%, var(--accent-soft), transparent 70%), var(--bg-2); }
  .hero-inner { max-width: 1180px; margin: 0 auto; padding: 56px 16px 44px; }
  .hero h1 { margin: 0 0 12px; color: var(--ink); font-size: clamp(32px, 6vw, 52px); line-height: 1.1;
             letter-spacing: -.5px; }
  .hero h1 span { color: var(--accent); }
  .hero .lead { max-width: 760px; margin: 0 0 22px; font-size: clamp(17px, 2.2vw, 19px); color: var(--ink-2); }
  .hero .cta { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 30px; }
  .btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px; border-radius: 9px;
         font-weight: 600; text-decoration: none; border: 1px solid var(--line); color: var(--ink);
         background: var(--panel); }
  .btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  .btn:hover { filter: brightness(1.06); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
  .card { padding: 16px 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          box-shadow: var(--shadow); }
  .card h3 { margin: 0 0 6px; font-size: 16px; color: var(--ink); }
  .card p { margin: 0; font-size: 14.5px; line-height: 1.55; color: var(--muted); }
  .notice { margin-top: 18px; font-size: 14.5px; color: var(--muted); }

  /* layout */
  .layout { max-width: 1180px; margin: 0 auto; display: grid; grid-template-columns: 250px minmax(0, 1fr);
            gap: 40px; padding: 0 16px; }
  .sidebar { position: sticky; top: 56px; align-self: start; max-height: calc(100vh - 56px); overflow-y: auto;
             padding: 28px 4px 40px 0; }
  .toc, .toc ul { list-style: none; margin: 0; padding: 0; }
  .toc > li { margin: 1px 0; }
  .toc a { display: block; padding: 5px 10px; border-radius: 6px; color: var(--muted); font-size: 14px;
           line-height: 1.4; text-decoration: none; border-left: 2px solid transparent; }
  .toc a:hover { color: var(--ink); background: var(--accent-soft); }
  .toc a.active { color: var(--accent); border-left-color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  .toc ul { display: none; margin: 2px 0 6px 10px; }
  .toc li.open > ul { display: block; }
  .toc ul a { font-size: 13px; padding: 3px 10px; }

  main { min-width: 0; padding: 28px 0 80px; }
  main h2 { margin: 56px 0 14px; padding-top: 8px; color: var(--ink); font-size: clamp(24px, 3.4vw, 30px);
            line-height: 1.25; border-top: 1px solid var(--line); padding-top: 28px; }
  main h2:first-child { margin-top: 0; border-top: 0; padding-top: 0; }
  main h3 { margin: 34px 0 10px; color: var(--ink); font-size: 20px; line-height: 1.3; }
  main h4 { margin: 26px 0 8px; color: var(--ink); font-size: 17px; }
  .hash { float: left; margin-left: -1.1em; padding-right: .3em; color: var(--muted); opacity: 0;
          text-decoration: none; font-weight: 400; }
  h2:hover .hash, h3:hover .hash, .hash:focus { opacity: 1; }
  main p, main li { max-width: 78ch; }
  main ul, main ol { padding-left: 1.4em; }
  main li { margin: 4px 0; }
  main strong { color: var(--ink); }
  main hr { border: 0; border-top: 1px solid var(--line); margin: 32px 0; }
  blockquote { margin: 18px 0; padding: 12px 16px; background: var(--note-bg); border-left: 4px solid var(--note-line);
               border-radius: 0 8px 8px 0; color: var(--ink-2); }
  blockquote p { margin: 0; }

  :not(pre) > code { padding: .12em .38em; border-radius: 5px; background: var(--code-bg);
                     border: 1px solid var(--line); font-size: .88em; color: var(--ink); word-break: break-word; }
  pre { position: relative; margin: 16px 0; padding: 14px 16px; overflow-x: auto; background: var(--code-bg);
        border: 1px solid var(--line); border-radius: 10px; font-size: 13.5px; line-height: 1.55; color: var(--ink); }
  pre code { background: none; border: 0; padding: 0; white-space: pre; }
  .copy { position: absolute; top: 8px; right: 8px; padding: 3px 9px; font: 12px/1.6 system-ui, sans-serif;
          color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
          cursor: pointer; opacity: 0; transition: opacity .15s; }
  pre:hover .copy, .copy:focus-visible { opacity: 1; }
  @media (hover: none) { .copy { opacity: 1; } }

  .table-wrap { margin: 16px 0; overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  th, td { padding: 9px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
  th { background: var(--bg-2); color: var(--ink); font-weight: 600; white-space: nowrap; }
  tr:last-child td { border-bottom: 0; }
  td code { white-space: nowrap; }
  details { margin: 16px 0; padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
  summary { cursor: pointer; color: var(--ink); font-weight: 600; }

  footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; }
  footer .inner { max-width: 1180px; margin: 0 auto; padding: 24px 16px; display: flex; flex-wrap: wrap; gap: 8px 24px; }

  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  /* the mobile menu's backdrop; outside the phone layout it must not exist,
     or it takes a grid column and pushes the content under the sidebar */
  .scrim { display: none; }

  /* phones and narrow tablets: the sidebar becomes a slide-in menu */
  @media (max-width: 900px) {
    #menu-btn { display: inline-flex; }
    .layout { display: block; }
    .sidebar { position: fixed; z-index: 30; top: 56px; left: 0; bottom: 0; width: min(300px, 86vw);
               max-height: none; padding: 16px 12px 40px; background: var(--bg); border-right: 1px solid var(--line);
               transform: translateX(-102%); transition: transform .2s ease; }
    body.nav-open .sidebar { transform: none; box-shadow: 0 0 40px rgba(0,0,0,.35); }
    .scrim { position: fixed; inset: 56px 0 0 0; z-index: 25; background: rgba(0,0,0,.35); display: none; }
    body.nav-open .scrim { display: block; }
    .hash { display: none; }
    .brand .full { display: none; }
  }
  /* phones: each table row becomes a card, each cell labelled by its column */
  @media (max-width: 640px) {
    .table-wrap { overflow: visible; border: 0; border-radius: 0; }
    .table-wrap table, .table-wrap tbody, .table-wrap tr, .table-wrap td { display: block; width: 100%; }
    .table-wrap thead { display: none; }
    .table-wrap tr { margin: 0 0 10px; padding: 6px 0; background: var(--panel);
                     border: 1px solid var(--line); border-radius: 10px; }
    .table-wrap td { display: grid; grid-template-columns: minmax(5.5em, 32%) 1fr; gap: 10px;
                     padding: 5px 12px; border: 0; }
    .table-wrap td::before { content: attr(data-label); color: var(--muted); font-size: 12.5px;
                             font-weight: 600; line-height: 1.9; }
    .table-wrap td[data-label=""] { grid-template-columns: 1fr; }
    .table-wrap td[data-label=""]::before { display: none; }
    td code { white-space: normal; }
  }
  @media (max-width: 480px) {
    .hero-inner { padding: 36px 16px 30px; }
    .iconbtn .label { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { .sidebar { transition: none; } html { scroll-behavior: auto; } }
  @media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
  @media print { .topbar, .sidebar, .copy, .hero .cta { display: none !important; } .layout { display: block; } }
</style>
</head>
<body>
<header class="topbar">
  <button class="iconbtn" id="menu-btn" aria-label="Open navigation" aria-expanded="false" aria-controls="sidebar">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
  </button>
  <a class="brand" href="#top">
    <svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#0d1420"/><path d="M7 22V10l9 7 9-7v12" fill="none" stroke="#00d4c4" stroke-width="3" stroke-linejoin="round"/></svg>
    <span>NeuralDeck<span class="full">-Proxy</span></span>
  </a>
  <span class="spacer"></span>
  <button class="iconbtn" id="theme-btn" aria-label="Toggle light or dark theme" title="Toggle theme">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
  </button>
  <a class="iconbtn" href="__REPO__" aria-label="Source on GitHub">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.7.08-.7 1.17.08 1.78 1.2 1.78 1.2 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a10.9 10.9 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/></svg>
    <span class="label">GitHub</span>
  </a>
</header>

<section class="hero" id="top">
  <div class="hero-inner">
    <h1>Neural<span>Deck</span>-Proxy</h1>
    <p class="lead">__LEAD__</p>
    <div class="cta">
      <a class="btn primary" href="#quick-start">Quick start</a>
      <a class="btn" href="#install">Install</a>
      <a class="btn" href="#troubleshooting">Troubleshooting</a>
      <a class="btn" href="__REPO__">View on GitHub</a>
    </div>
    <div class="cards">
__CARDS__
    </div>
    <p class="notice">__NOTICE__</p>
  </div>
</section>

<div class="layout">
  <nav class="sidebar" id="sidebar" aria-label="Documentation sections">
__NAV__
  </nav>
  <div class="scrim" id="scrim"></div>
  <main id="content">
__BODY__
  </main>
</div>

<footer>
  <div class="inner">
    <span>NeuralDeck-Proxy · MIT licence</span>
    <a href="__REPO__">Source on GitHub</a>
    <a href="__REPO__/issues">Issues</a>
    <span>Generated from README.md by docs/build.py</span>
  </div>
</footer>

<script>
(() => {
  const root = document.documentElement;
  // theme toggle: flips between light and dark, remembered per browser
  document.getElementById("theme-btn").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("nd-docs-theme", root.dataset.theme); } catch (e) {}
  });

  // mobile navigation
  const body = document.body, btn = document.getElementById("menu-btn");
  const setNav = open => { body.classList.toggle("nav-open", open); btn.setAttribute("aria-expanded", open); };
  btn.addEventListener("click", () => setNav(!body.classList.contains("nav-open")));
  document.getElementById("scrim").addEventListener("click", () => setNav(false));
  document.addEventListener("keydown", e => { if (e.key === "Escape") setNav(false); });
  document.querySelectorAll(".sidebar a").forEach(a => a.addEventListener("click", () => setNav(false)));

  // copy buttons on code blocks
  document.querySelectorAll("main pre").forEach(pre => {
    const b = document.createElement("button");
    b.className = "copy"; b.type = "button"; b.textContent = "copy";
    b.setAttribute("aria-label", "Copy code");
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(pre.querySelector("code")?.innerText ?? pre.innerText); b.textContent = "copied"; }
      catch (e) { b.textContent = "copy failed"; }
      setTimeout(() => b.textContent = "copy", 1500);
    });
    pre.appendChild(b);
  });

  // highlight the section in view and open its sub-list
  const links = new Map([...document.querySelectorAll(".sidebar a")].map(a => [a.hash.slice(1), a]));
  const heads = [...document.querySelectorAll("main h2[id], main h3[id]")];
  let ticking = false;
  const spy = () => {
    ticking = false;
    let cur = heads[0];
    for (const h of heads) { if (h.getBoundingClientRect().top < 110) cur = h; else break; }
    if (!cur) return;
    document.querySelectorAll(".sidebar a.active").forEach(a => a.classList.remove("active"));
    document.querySelectorAll(".toc li.open").forEach(li => li.classList.remove("open"));
    const a = links.get(cur.id);
    if (!a) return;
    a.classList.add("active");
    let li = a.closest("li");
    while (li) { li.classList.add("open"); li = li.parentElement.closest("li"); }
    const top = a.closest(".toc > li");
    top && top.classList.add("open");
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(spy); } }, { passive: true });
  spy();
})();
</script>
</body>
</html>
"""


def main():
    md = README.read_text(encoding="utf-8")
    title, intro, body = split_readme(md)
    body_html = render(body)
    lead_html = lead(intro)
    desc = re.sub(r"<[^>]+>", "", lead_html)
    page = (TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__DESC__", html.escape(" ".join(desc.split()), quote=True))
            .replace("__LEAD__", lead_html)
            .replace("__CARDS__", features(intro))
            .replace("__NOTICE__", notice(intro))
            .replace("__NAV__", nav(body_html))
            .replace("__BODY__", body_html)
            .replace("__REPO__", REPO))
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page) // 1024} KB) from {README.name}")


if __name__ == "__main__":
    main()
