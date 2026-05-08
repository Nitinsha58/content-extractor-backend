"""
Block and Question schema for the Exam Question Extractor.

Block types (user-defined):
  text      — plain string
  latex     — KaTeX/MathJax-ready LaTeX; display flag for block vs inline
  image     — S3/CDN/static URL + alt text + dimensions
  table     — 2-D array of cells (each cell is itself a list of SimpleBlocks)
  chemical  — SMILES string or image URL for chemical structures (stub)

Question types: mcq_single, mcq_multi, integer, assertion_reason, subjective, unknown
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Primitive block types
# ─────────────────────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    value: str


class LatexBlock(BaseModel):
    type: Literal["latex"] = "latex"
    value: str          # cleaned LaTeX string, KaTeX/MathJax compatible
    display: bool = False  # True = display/block math; False = inline


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    url: str            # e.g. /static/images/<uuid>.png
    alt: str = ""
    width: int
    height: int


class ChemicalBlock(BaseModel):
    """
    Stub for chemical structures.  SMILES extraction needs a separate model
    (DECIMER / chemocr).  Falls back to ImageBlock URL until that is wired up.
    """
    type: Literal["chemical"] = "chemical"
    smiles: Optional[str] = None
    url: Optional[str] = None   # image URL fallback


# SimpleBlock — allowed inside table cells (no nested tables)
SimpleBlock = Union[TextBlock, LatexBlock, ImageBlock, ChemicalBlock]


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    rows: int
    cols: int
    # cells[row_idx][col_idx] = list of SimpleBlock
    cells: List[List[List[SimpleBlock]]]


# Full discriminated union usable everywhere else
Block = Union[TextBlock, LatexBlock, ImageBlock, TableBlock, ChemicalBlock]


# ─────────────────────────────────────────────────────────────────────────────
# Question
# ─────────────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    MCQ_SINGLE       = "mcq_single"       # exactly one correct option
    MCQ_MULTI        = "mcq_multi"        # one or more correct options
    INTEGER          = "integer"          # numerical/integer answer
    ASSERTION_REASON = "assertion_reason" # assertion + reason structure
    SUBJECTIVE       = "subjective"       # short / long answer
    UNKNOWN          = "unknown"


class Question(BaseModel):
    number: Optional[int] = None          # detected question number (1, 2, …)
    type: QuestionType = QuestionType.UNKNOWN
    marks: Optional[float] = None
    content: List[Block] = Field(default_factory=list)
    # options keys: "a" / "b" / "c" / "d"  (normalised lowercase)
    options: Optional[Dict[str, List[Block]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Page / Document level
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedPage(BaseModel):
    page_number: int
    questions: List[Question] = Field(default_factory=list)
    # blocks that appear before the first detected question (instructions etc.)
    ungrouped_blocks: List[Block] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    source_file: str
    total_pages: int
    pages: List[ExtractedPage] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Inspector debug models
# ─────────────────────────────────────────────────────────────────────────────

class DebugLayoutBlock(BaseModel):
    """One detected layout region — matches LayoutBlock but JSON-serialisable."""
    id: str                       # uuid4 assigned server-side, stable across /debug/ocr
    label: str                    # plain_text | title | isolate_formula | table | figure
    bbox: List[int]               # [x1, y1, x2, y2] in image-pixel coordinates
    confidence: float
    column_idx: int               # 0 = left / single, 1 = right (two-column pages)
    reading_order: int            # 0-based sort index

    @field_validator('bbox', mode='before')
    @classmethod
    def round_bbox(cls, v):
        return [round(x) for x in v]


class DebugLayoutResponse(BaseModel):
    session_id: str
    image_url: str                # /static/debug/<session_id>.jpg
    image_width: int
    image_height: int
    layout_blocks: List[DebugLayoutBlock] = Field(default_factory=list)


class DebugOCRRequest(BaseModel):
    """Sent by the frontend after user edits layout boxes."""
    session_id: str
    layout_blocks: List[DebugLayoutBlock]


class DebugOCRBlock(BaseModel):
    """OCR output for one layout region."""
    block_id: str                 # matches DebugLayoutBlock.id
    label: str
    bbox: List[int]
    reading_order: int
    column_idx: int
    blocks: List[Block] = Field(default_factory=list)
    duration_ms: float = 0.0     # wall-clock time spent in process_region()

    @field_validator('bbox', mode='before')
    @classmethod
    def round_bbox(cls, v):
        return [round(x) for x in v]


class DebugOCRResponse(BaseModel):
    session_id: str
    image_url: str
    image_width: int
    image_height: int
    ocr_blocks: List[DebugOCRBlock] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Document container tree  (Phase B)
# ─────────────────────────────────────────────────────────────────────────────

class ContentGroupNode(BaseModel):
    """One extracted content region — wraps a single DebugOCRBlock."""
    block_id: str
    label: str
    bbox: List[int]
    reading_order: int
    blocks: List[Block] = Field(default_factory=list)


class ColumnNode(BaseModel):
    """One column of content on the page."""
    column_idx: int
    bbox: List[int]              # bounding box that covers all groups in column
    groups: List[ContentGroupNode] = Field(default_factory=list)


class PageNode(BaseModel):
    """Root of the document tree for one page."""
    width: int
    height: int
    session_id: str
    image_url: str
    columns: List[ColumnNode] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """Request body for POST /export."""
    session_id: str
    format: Literal["html", "markdown", "docx"]
    ocr_blocks: List[DebugOCRBlock]
