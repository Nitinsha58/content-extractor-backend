"""
html_exporter.py
================
Converts a PageNode document tree into a self-contained HTML document
with inline KaTeX CDN rendering and two-column CSS grid layout when needed.
"""

from __future__ import annotations

from schema import PageNode, ContentGroupNode, Block


def _render_block_html(block: Block, base_url: str) -> str:
    t = block.type
    if t == "text":
        return f"<span>{_esc(block.value)}</span> "
    if t == "latex":
        if block.display:
            return f'<div class="b-latex-display">$${_esc(block.value)}$$</div>'
        return f'<span class="b-latex-inline">${_esc(block.value)}$</span>'
    if t == "image":
        url = block.url if block.url.startswith("http") else base_url.rstrip("/") + block.url
        return f'<img src="{url}" alt="{_esc(block.alt)}" style="max-width:100%;display:block;margin:6px 0"/>'
    if t == "table":
        rows_html = ""
        for ri, row in enumerate(block.cells or []):
            tag = "th" if ri == 0 else "td"
            cells_html = "".join(
                f"<{tag}>" + "".join(_render_block_html(cb, base_url) for cb in cell) + f"</{tag}>"
                for cell in row
            )
            rows_html += f"<tr>{cells_html}</tr>"
        return f'<table style="border-collapse:collapse;width:100%">{rows_html}</table>'
    return ""


def _render_group_html(grp: ContentGroupNode, base_url: str) -> str:
    inner = "".join(_render_block_html(b, base_url) for b in grp.blocks)
    label = grp.label
    color_map = {
        "plain_text": "#4f8ef7", "title": "#a78bfa",
        "isolate_formula": "#f59e0b", "table": "#22c55e", "figure": "#f43f5e",
    }
    color = color_map.get(label, "#888")
    tag = "h2" if label == "title" else "div"
    return (
        f'<{tag} class="content-group" data-label="{label}" '
        f'style="border-left:3px solid {color};padding:8px 12px;margin:6px 0">'
        f'{inner}</{tag}>'
    )


def export_html(page: PageNode, base_url: str = "/") -> str:
    two_col = len(page.columns) >= 2
    grid_style = "display:grid;grid-template-columns:1fr 1fr;gap:16px;" if two_col else ""

    columns_html = ""
    for col in page.columns:
        groups_html = "".join(_render_group_html(g, base_url) for g in col.groups)
        columns_html += f'<div class="column">{groups_html}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Extracted Document</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[
    {{left:'$$',right:'$$',display:true}},
    {{left:'$',right:'$',display:false}}
  ]}})"></script>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:960px;margin:0 auto;padding:24px;line-height:1.6;}}
  .column{{min-width:0;}}
  h2.content-group{{font-size:1.1em;}}
  .b-latex-display{{overflow-x:auto;padding:8px 0;text-align:center;}}
  table td,table th{{border:1px solid #ddd;padding:5px 8px;}}
  table tr:first-child th{{background:#f5f5f5;}}
</style>
</head>
<body>
<div style="{grid_style}">{columns_html}</div>
</body>
</html>"""


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
