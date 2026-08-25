import os
from pathlib import Path
from typing import Dict, Any

from docling.document_converter import DocumentConverter
from docling_graph.protocols import LLMClientProtocol

from src.ingestion.schema import ScientificPaper

# HuggingFaceEndpointClient deprecated in favor of central get_llm() in nodes.py
# which supports multiple providers and native structured output.

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

class DoclingIngestor:
    def __init__(self, enable_ocr: bool = False):
        # Threading: configurable via env, default 4. Benchmark 1/2/4/6/8 for your CPU.
        num_threads = int(os.getenv("DOCLING_NUM_THREADS", "4"))
        
        pipeline_options = ThreadedPdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = False
        pipeline_options.generate_picture_images = False
        
        # Profiling: off by default, enable via DOCLING_PROFILE=1 for benchmarking
        if os.getenv("DOCLING_PROFILE", "0") == "1":
            from docling.datamodel.settings import settings
            settings.debug.profile_pipeline_timings = True
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # HuggingFaceEndpointClient is deprecated. 
        # We now dynamically pull the correct model (Groq/OpenAI/Gemini/HF) at runtime.
        pass

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Convert PDF to a Docling document."""
        print(f"Docling: Converting {file_path}...")
        result = self.converter.convert(file_path)
        return result.document

    def _extract_title_from_markdown(self, text: str) -> str:
        """Pull the real title from the first markdown heading or first non-empty line."""
        import re
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match markdown headings: # Title or ## Title
            heading = re.match(r'^#+\s+(.+)', line)
            if heading:
                return heading.group(1).strip()
            # Otherwise the first non-empty line is likely the title
            return line[:200]
        return "Untitled Paper"

    def _extract_abstract_from_markdown(self, text: str) -> str:
        """Pull the first substantial paragraph as the abstract."""
        import re
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs[1:]:  # Skip the title paragraph
            cleaned = para.strip()
            if len(cleaned) > 100:  # Must be a real paragraph, not a short heading
                return cleaned[:1000]
        return text[200:1200] if len(text) > 200 else text[:1000]

    def extract_graph(self, docling_doc) -> ScientificPaper:
        """Extract structured scientific paper metadata directly from Docling document structure."""
        import re
        from src.ingestion.schema import ScientificPaper, Researcher
        text = docling_doc.export_to_markdown()
        
        title = self._extract_title_from_markdown(text)
        abstract = self._extract_abstract_from_markdown(text)
        
        # Authors from text header
        authors = []
        header_match = re.search(r'(?i)^(.*?)\babstract\b', text[:4000], re.DOTALL)
        if header_match:
            header_lines = [l.strip() for l in header_match.group(1).split("\n") if l.strip()]
            if len(header_lines) > 1:
                candidate_line = " ".join(header_lines[1:3])
                candidate_line = re.sub(r'[\w\.-]+@[\w\.-]+', '', candidate_line)
                parts = re.split(r'[,;*†‡§]|\band\b', candidate_line)
                for part in parts:
                    clean = re.sub(r'[^a-zA-Z\s\.\-]', '', part).strip()
                    if 2 < len(clean) < 40 and not any(w in clean.lower() for w in ["abstract", "ieee", "arxiv", "volume"]):
                        authors.append(Researcher(name=clean))
        
        # DOI
        doi = ""
        doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b', text[:6000])
        if doi_match:
            doi = doi_match.group(1).rstrip('.')
            
        # Year
        year = 2025
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text[:3000])
        if year_match:
            year = int(year_match.group(1))
            
        # References / Citations
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
            cites=citations,
            builds_on=[],
            extends=[],
            compares_to=[],
            contradicts=[]
        )


