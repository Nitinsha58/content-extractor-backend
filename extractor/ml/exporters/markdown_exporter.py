"""
markdown_exporter.py
====================
Converts a PageNode document tree into GitHub-Flavoured Markdown.
LaTeX is wrapped in $...$ (inline) or $$...$$ (display).
"""

from __future__ import annotations

from schema import PageNode, ContentGroupNode, Block


def _render_block_md(block: Block) -> str:
    t = block.type
    if t == "text":
        return block.value
    if t == "latex":
        if block.display:
            return f"\n$$\n{block.value}\n$$\n"
        return f"${block.value}$"
    if t == "image":
        return f"![{block.alt or 'image'}]({block.url})"
    if t == "table":
        if not block.cells:
            return ""
        lines = []
        for ri, row in enumerate(block.cells):
            cells = [" ".join(_render_block_md(cb) for cb in cell) for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
            if ri == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(lines)
    return ""


def _render_group_md(grp: ContentGroupNode) -> str:
    parts = []
    for b in grp.blocks:
        parts.append(_render_block_md(b))
    text = " ".join(p for p in parts if p)
    if grp.label == "title":
        return f"\n## {text.strip()}\n"
    return f"\n{text.strip()}\n"


def export_markdown(page: PageNode) -> str:
    sections: list[str] = []
    for col in page.columns:
        if len(page.columns) > 1:
            sections.append(f"\n<!-- column {col.column_idx} -->\n")
        for grp in col.groups:
            sections.append(_render_group_md(grp))
    return "\n".join(sections).strip() + "\n"
