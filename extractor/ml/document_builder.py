"""
document_builder.py
====================
Converts a flat list of DebugOCRBlock objects into a hierarchical PageNode
by grouping blocks into columns and preserving reading order.
"""

from __future__ import annotations

from typing import List

from schema import (
    ColumnNode,
    ContentGroupNode,
    DebugOCRBlock,
    PageNode,
)


def build_document_tree(
    ocr_blocks: List[DebugOCRBlock],
    img_w: int,
    img_h: int,
    session_id: str,
    image_url: str,
) -> PageNode:
    """
    Group OCR blocks by column_idx, sort each column by reading_order,
    and build a PageNode tree.
    """
    # Bucket blocks by column
    columns_map: dict[int, List[DebugOCRBlock]] = {}
    for b in ocr_blocks:
        columns_map.setdefault(b.column_idx, []).append(b)

    column_nodes: List[ColumnNode] = []
    for col_idx in sorted(columns_map.keys()):
        col_blocks = sorted(columns_map[col_idx], key=lambda b: b.reading_order)

        # Compute bounding box that covers all blocks in this column
        all_x1 = [b.bbox[0] for b in col_blocks]
        all_y1 = [b.bbox[1] for b in col_blocks]
        all_x2 = [b.bbox[2] for b in col_blocks]
        all_y2 = [b.bbox[3] for b in col_blocks]
        col_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]

        groups: List[ContentGroupNode] = [
            ContentGroupNode(
                block_id=b.block_id,
                label=b.label,
                bbox=b.bbox,
                reading_order=b.reading_order,
                blocks=b.blocks,
            )
            for b in col_blocks
        ]

        column_nodes.append(
            ColumnNode(column_idx=col_idx, bbox=col_bbox, groups=groups)
        )

    return PageNode(
        width=img_w,
        height=img_h,
        session_id=session_id,
        image_url=image_url,
        columns=column_nodes,
    )
