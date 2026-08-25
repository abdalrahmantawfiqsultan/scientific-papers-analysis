"""
Smart PDF Router — triages documents to the optimal parser.

Triage logic:
  1. Sample first, middle, and last pages via PyMuPDF (cheap)
  2. Compute cheap features: text coverage, image count, font count, column hints
  3. Route to PyMuPDF (instant) or Docling (threaded, with/without OCR)
  4. After PyMuPDF, run a quality gate — fall back to Docling if extraction is poor
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import fitz  # PyMuPDF


@dataclass
class ParseResult:
    """Common output contract for all parsers. Decouples downstream code from any specific parser."""
    parser: str                        # "pymupdf", "docling", "docling_ocr"
    text: str                          # extracted markdown/text
    metadata: dict = field(default_factory=dict)   # title, abstract, etc.
    provenance: dict = field(default_factory=dict)  # document, method, timestamps
    quality_score: float = 0.0         # 0.0–1.0 from quality gate
    fallback_used: bool = False        # True if triage picked pymupdf but fell back to docling
    timings: dict = field(default_factory=dict)     # triage_ms, parser_ms, quality_gate_ms


@dataclass
class TriageResult:
    """Cheap features extracted during triage — used to pick a parser."""
    page_count: int = 0
    total_text_chars_sample: int = 0
    text_chars_per_page: float = 0.0
    text_coverage_ratio: float = 0.0   # sampled chars / expected chars
    number_of_images: int = 0
    number_of_fonts: int = 0
    scanned_page_ratio: float = 0.0    # fraction of sampled pages with <100 chars
    two_column_hint: bool = False

    # Derived booleans
    is_scanned: bool = False
    is_text_rich: bool = False
    is_complex_layout: bool = False

    recommended_parser: Literal["pymupdf", "docling", "docling_ocr"] = "docling"


def _sample_page_indices(page_count: int) -> set:
    """Sample first, middle, and last pages."""
    if page_count <= 0:
        return set()
    indices = {0}
    if page_count > 1:
        indices.add(page_count // 2)
        indices.add(page_count - 1)
    return indices


def triage_pdf(file_path: str) -> TriageResult:
    """Run cheap PyMuPDF-based feature extraction on sampled pages to classify the PDF."""
    result = TriageResult()

    try:
        doc = fitz.open(file_path)
    except Exception:
        # Can't even open it — let Docling try with OCR
        result.recommended_parser = "docling_ocr"
        result.is_scanned = True
        return result

    result.page_count = len(doc)
    if result.page_count == 0:
        doc.close()
        result.recommended_parser = "docling_ocr"
        return result

    sample_indices = _sample_page_indices(result.page_count)
    
    sampled_chars = []
    scanned_pages = 0
    total_images = 0
    all_fonts = set()
    column_hints = 0

    for idx in sample_indices:
        if idx >= result.page_count:
            continue
        page = doc[idx]

        # Text extraction
        text = page.get_text("text")
        char_count = len(text.strip())
        sampled_chars.append(char_count)

        if char_count < 100:
            scanned_pages += 1

        # Image count
        total_images += len(page.get_images(full=False))

        # Font diversity
        for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        all_fonts.add(span.get("font", ""))

        # Two-column hint: check if text blocks span less than 60% of page width
        page_width = page.rect.width
        blocks = page.get_text("blocks")
        narrow_blocks = 0
        for b in blocks:
            block_width = b[2] - b[0]  # x1 - x0
            if block_width < page_width * 0.55 and len(b[4]) > 50:
                narrow_blocks += 1
        if narrow_blocks > len(blocks) * 0.4 and len(blocks) > 3:
            column_hints += 1

    doc.close()

    # Compute aggregates
    num_sampled = len(sampled_chars)
    result.total_text_chars_sample = sum(sampled_chars)
    result.text_chars_per_page = result.total_text_chars_sample / max(num_sampled, 1)
    result.number_of_images = total_images
    result.number_of_fonts = len(all_fonts)
    result.scanned_page_ratio = scanned_pages / max(num_sampled, 1)
    result.two_column_hint = column_hints > 0

    # Expected: ~2000 chars per page for a text-rich document
    expected_chars = num_sampled * 2000
    result.text_coverage_ratio = result.total_text_chars_sample / max(expected_chars, 1)

    # Classify
    result.is_scanned = result.scanned_page_ratio > 0.5
    result.is_text_rich = result.text_coverage_ratio > 0.3 and not result.is_scanned
    result.is_complex_layout = (
        result.two_column_hint
        or result.number_of_fonts > 6
        or result.number_of_images > num_sampled * 2
    )

    # Route
    if result.is_scanned:
        result.recommended_parser = "docling_ocr"
    elif result.is_text_rich and not result.is_complex_layout:
        result.recommended_parser = "pymupdf"
    else:
        result.recommended_parser = "docling"

    return result


def quality_gate(extracted_text: str, triage: TriageResult) -> bool:
    """Post-PyMuPDF quality check. Returns True if extraction is acceptable."""
    if not extracted_text or len(extracted_text.strip()) < 200:
        return False

    # Check that we got a reasonable amount of text relative to page count
    expected_min = triage.page_count * 500
    if len(extracted_text) < expected_min * 0.3:
        return False

    # Check for garbled text (high ratio of non-printable / non-ASCII)
    printable_ratio = sum(1 for c in extracted_text[:5000] if c.isprintable() or c in '\n\t') / min(len(extracted_text), 5000)
    if printable_ratio < 0.85:
        return False

    # Check that we have at least some sentence-like structure
    sentence_endings = len(re.findall(r'[.!?]\s', extracted_text[:5000]))
    if sentence_endings < 3:
        return False

    return True


def route_and_parse(file_path: str, filename: str = "") -> ParseResult:
    """Full orchestrator: triage → parse → quality gate → fallback. Returns ParseResult."""
    timings = {}
    
    # 1. Triage
    t0 = time.perf_counter()
    triage = triage_pdf(file_path)
    timings["triage_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    
    chosen_parser = triage.recommended_parser
    fallback_used = False
    text = ""
    quality_score = 0.0
    
    # 2. PyMuPDF fast path
    if chosen_parser == "pymupdf":
        t1 = time.perf_counter()
        from src.ingestion.pymupdf_parser import PyMuPDFParser
        fast_parser = PyMuPDFParser()
        text = fast_parser.parse_pdf(file_path)
        timings["parser_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        
        # Quality gate
        t2 = time.perf_counter()
        passed = quality_gate(text, triage)
        timings["quality_gate_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        
        if passed:
            quality_score = 1.0
        else:
            # Fall back to Docling
            chosen_parser = "docling"
            fallback_used = True
            text = ""
    
    # 3. Docling path (or fallback)
    if chosen_parser in ("docling", "docling_ocr"):
        t1 = time.perf_counter()
        from src.ingestion.docling_parser import DoclingIngestor
        enable_ocr = chosen_parser == "docling_ocr"
        ingestor = DoclingIngestor(enable_ocr=enable_ocr)
        docling_doc = ingestor.parse_pdf(file_path)
        text = docling_doc.export_to_markdown()
        timings["parser_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        quality_score = 0.9  # Docling output is generally high quality
    
    return ParseResult(
        parser=chosen_parser,
        text=text,
        metadata={
            "triage": {
                "page_count": triage.page_count,
                "text_coverage": round(triage.text_coverage_ratio, 2),
                "is_scanned": triage.is_scanned,
                "is_complex": triage.is_complex_layout,
                "recommended": triage.recommended_parser,
            }
        },
        provenance={"document": filename or file_path, "method": chosen_parser},
        quality_score=quality_score,
        fallback_used=fallback_used,
        timings=timings,
    )

