"""
Table Formatter — post-process markdown tables from TableOCR.recognize().
Normalizes markdown pipe-table format, cleans cell text, and handles edge cases.
"""

import re

def _r(pattern, replacement, desc=""):
    """Compile pattern once."""
    return (re.compile(pattern), replacement, desc)


# ═══════════════════════════════════════════════════════════════
# CELL TEXT RULES — for individual cell contents
# ═══════════════════════════════════════════════════════════════
CELL_TEXT_RULES = [
    _r(r'^\s*\|\s*', '', "strip leading pipe and whitespace"),
    _r(r'\s*\|\s*$', '', "strip trailing pipe and whitespace"),
    _r(r'\\\|', '|', "unescape pipes"),
    _r(r'[^\S\r\n]+', ' ', "collapse horizontal whitespace"),
]

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _normalize_cell(cell: str) -> str:
    """Apply CELL_TEXT_RULES to one cell string."""
    for pattern, replacement, _ in CELL_TEXT_RULES:
        cell = pattern.sub(replacement, cell)
    return cell.strip()


def _split_row(row: str) -> list[str]:
    """Split a markdown table row on pipes and strip empty leading/trailing."""
    cells = row.split('|')
    # Remove leading/trailing empty cells (created by leading/trailing pipes)
    if cells and cells[0].strip() == '':
        cells = cells[1:]
    if cells and cells[-1].strip() == '':
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def _align_columns(rows: list[list[str]]) -> list[list[str]]:
    """Pad rows to the same column count (max columns seen)."""
    if not rows:
        return rows
    max_cols = max(len(row) for row in rows)
    return [row + [''] * (max_cols - len(row)) for row in rows]


def _render_row(cells: list[str]) -> str:
    """Render a list of cells as a markdown table row."""
    return '| ' + ' | '.join(cells) + ' |'


def _inject_separator(num_cols: int) -> str:
    """Render a markdown table separator row."""
    return '| ' + ' | '.join(['---'] * num_cols) + ' |'


def _is_header_like(cells: list[str]) -> bool:
    """Heuristic: does this row look like a table header?"""
    if not cells:
        return False
    # If most cells are short or all uppercase, likely header
    avg_len = sum(len(c) for c in cells) / len(cells) if cells else 0
    all_caps = all(c.isupper() or c == '' for c in cells if len(c) > 1)
    all_short = all(len(c) <= 15 for c in cells)
    return all_short or all_caps


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def format_table(ocr_result: dict) -> str:
    """
    Format a table from TableOCR.recognize() output.

    Input: dict with keys 'markdown' (list of strings), etc.
    Output: cleaned markdown table string.
    """
    markdown_lines = ocr_result.get('markdown', [])

    if not markdown_lines:
        return ''

    # Parse rows
    rows = []
    for line in markdown_lines:
        line = line.strip()
        if not line:
            continue
        cells = _split_row(line)
        if cells:  # skip empty rows
            rows.append([_normalize_cell(cell) for cell in cells])

    if not rows:
        return ''

    # Align columns
    rows = _align_columns(rows)
    num_cols = len(rows[0]) if rows else 0

    if num_cols == 0:
        return ''

    # Build output: detect header and inject separator if needed
    output = []
    separator_inserted = False

    for i, row in enumerate(rows):
        row_str = _render_row(row)
        output.append(row_str)

        # If this is the first row and looks like a header, inject separator after it
        if i == 0 and _is_header_like(row) and not separator_inserted:
            output.append(_inject_separator(num_cols))
            separator_inserted = True

        # If a separator row (all dashes), skip it (will be injected programmatically)
        elif all(cell == '---' or cell == '-' * len(cell) for cell in row):
            # Already inserted our own, remove the original
            output.pop()
            separator_inserted = True

    # If no separator was inserted, inject one after first row anyway
    if not separator_inserted and len(output) > 0:
        output.insert(1, _inject_separator(num_cols))

    return '\n'.join(output)


def extract_table_text(ocr_result: dict) -> str:
    """
    Fallback: extract table as plain tab-separated text (one row per line).
    Used if markdown is malformed or for simple text export.
    """
    markdown_lines = ocr_result.get('markdown', [])

    text_rows = []
    for line in markdown_lines:
        line = line.strip()
        if not line or all(c in '-| ' for c in line):
            # Skip separator rows
            continue
        cells = _split_row(line)
        if cells:
            text_rows.append('\t'.join(_normalize_cell(c) for c in cells))

    return '\n'.join(text_rows)
