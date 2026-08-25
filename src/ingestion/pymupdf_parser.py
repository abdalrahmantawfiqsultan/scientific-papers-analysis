"""
PyMuPDF Fast Structured Parser.

Extracts text, metadata, authors, sections, and references directly from PDF
structure and formatting without any ML models or external network calls.
"""

import re
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
from src.ingestion.schema import ScientificPaper, Researcher, Author


class PyMuPDFParser:
    """Ultra-fast structured document and metadata extraction."""

    def parse_pdf(self, file_path: str) -> str:
        """Extract all text from a PDF. Returns concatenated page text."""
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages)

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract embedded PDF metadata dictionary."""
        try:
            doc = fitz.open(file_path)
            meta = doc.metadata or {}
            doc.close()
            return meta
        except Exception:
            return {}

    def extract_title(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Heuristic title extraction from metadata or the first non-empty lines."""
        if meta and meta.get("title") and len(meta["title"].strip()) > 5:
            # Check that it's not a generic file name
            title_candidate = meta["title"].strip()
            if not title_candidate.lower().endswith(".pdf") and len(title_candidate) > 10:
                return title_candidate[:300]

        for line in text.split("\n"):
            line = line.strip()
            line = re.sub(r'^#+\s*', '', line).strip()
            if not line:
                continue
            if len(line) < 8:
                continue
            # Title is usually the first substantial line
            return line[:300]
        return "Untitled Paper"

    def extract_abstract(self, text: str) -> str:
        """Heuristic abstract extraction — look for 'Abstract' section or first long paragraph."""
        abstract_match = re.search(
            r'(?i)\babstract\b[:\s]*\n?(.*?)(?:\n\s*\n\s*(?:[1I]\.?\s*)?(?:introduction|keywords|index terms)\b|\n\s*\n\s*#+)',
            text[:8000],
            re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            if len(abstract) > 50:
                return abstract[:2000]

        # Fallback: first paragraph longer than 200 chars
        paragraphs = re.split(r'\n\s*\n', text[:8000])
        for para in paragraphs[1:]:  # skip title
            cleaned = para.strip()
            if len(cleaned) > 200:
                return cleaned[:2000]

        return text[200:2200] if len(text) > 200 else text[:2000]

    def extract_authors(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Researcher]:
        """Extract authors from PDF metadata or the header region above the Abstract."""
        authors = []
        # 1. Try PDF metadata first
        if meta and meta.get("author"):
            raw_author = meta["author"].strip()
            # Split multiple authors by comma, semicolon, or 'and'
            parts = re.split(r'[,;]|\band\b', raw_author)
            for part in parts:
                clean = re.sub(r'[^a-zA-Z\s\.\-]', '', part).strip()
                if 2 < len(clean) < 50:
                    authors.append(Researcher(name=clean))
            if authors:
                return authors

        # 2. Extract from header text (between Title and Abstract)
        header_match = re.search(r'(?i)^(.*?)\babstract\b', text[:4000], re.DOTALL)
        if header_match:
            header_lines = [l.strip() for l in header_match.group(1).split("\n") if l.strip()]
            # Skip first line (title), check subsequent lines for author names
            if len(header_lines) > 1:
                candidate_line = " ".join(header_lines[1:3])
                # Filter out emails and affiliations
                candidate_line = re.sub(r'[\w\.-]+@[\w\.-]+', '', candidate_line)
                candidate_line = re.sub(r'\b(?:University|Department|Institute|College|Laboratory|School|Email)\b.*', '', candidate_line, flags=re.IGNORECASE)
                parts = re.split(r'[,;*†‡§]|\band\b', candidate_line)
                for part in parts:
                    clean = re.sub(r'[^a-zA-Z\s\.\-]', '', part).strip()
                    if 2 < len(clean) < 40 and not any(w in clean.lower() for w in ["abstract", "ieee", "arxiv", "volume", "issue", "pages"]):
                        authors.append(Researcher(name=clean))

        return authors[:10]

    def extract_year(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """Extract publication year from metadata or document text."""
        # 1. Check metadata creation/mod date (e.g. D:20230512...)
        if meta and meta.get("creationDate"):
            date_str = str(meta["creationDate"])
            year_match = re.search(r'(?:19|20)\d{2}', date_str)
            if year_match:
                return int(year_match.group(0))

        # 2. Check first 3000 chars of document for 4-digit year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text[:3000])
        if year_match:
            return int(year_match.group(1))

        return 2025

    def extract_doi(self, text: str) -> str:
        """Extract DOI from text using standard DOI regex pattern."""
        doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b', text[:6000])
        if doi_match:
            return doi_match.group(1).rstrip('.')
        return ""

    def extract_sections(self, text: str) -> List[str]:
        """Extract major document sections."""
        section_headers = re.findall(r'(?m)^(?:#+\s*|(?:[I|V|X]+|\d+)\.\s+)([A-Z][A-Za-z\s]{3,40})$', text)
        sections = [s.strip() for s in section_headers if len(s.strip()) > 3]
        return sections[:15]

    def extract_references(self, text: str) -> List[str]:
        """Extract bibliographic citations from References or Bibliography section."""
        citations = []
        ref_match = re.search(r'(?i)(?:##+\s*(?:References|Bibliography)|(?:References|Bibliography)\s*\n)([\s\S]+)', text)
        if ref_match:
            ref_section = ref_match.group(1)[:12000]
            ref_lines = re.findall(r'(?:\[\d+\]|\d+\.)\s*([^\n\r]+)', ref_section)
            for r_line in ref_lines[:25]:
                cleaned_ref = r_line.strip()
                if len(cleaned_ref) > 15:
                    title_quote = re.search(r'["“]([^"”]+)["”]', cleaned_ref)
                    if title_quote and len(title_quote.group(1)) > 10:
                        citations.append(title_quote.group(1).strip())
                    else:
                        citations.append(cleaned_ref[:100].strip())
        return citations

    def extract_paper(self, file_path: str) -> ScientificPaper:
        """Complete structured extraction from PDF."""
        text = self.parse_pdf(file_path)
        meta = self.extract_metadata(file_path)
        
        title = self.extract_title(text, meta)
        abstract = self.extract_abstract(text)
        authors = self.extract_authors(text, meta)
        year = self.extract_year(text, meta)
        doi = self.extract_doi(text)
        cites = self.extract_references(text)

        return ScientificPaper(
            title=title,
            abstract=abstract,
            year=year,
            doi=doi,
            authors=authors,
            uses_methods=[],
            uses_datasets=[],
            addresses_problems=[],
            evaluated_by=[],
            reports_results=[],
            cites=cites,
            builds_on=[],
            extends=[],
            compares_to=[],
            contradicts=[]
        )

