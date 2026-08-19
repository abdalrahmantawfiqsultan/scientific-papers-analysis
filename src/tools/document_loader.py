import json
import os
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from src.tools.text_processing import extract_dense_sentences

@tool
def read_local_paper(file_path: str) -> str:
    """Extracts text content from a local PDF scientific paper. Pass the absolute or relative path to the PDF."""
    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found at path: {file_path}"})
    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        # Concatenate text from all pages
        full_text = "\n".join([page.page_content for page in pages])
        # Extract dense sentences to avoid LLM context blowout
        dense_chunk = extract_dense_sentences(full_text, max_chars=4000)
        return json.dumps({"success": True, "num_pages": len(pages), "content": dense_chunk})
    except Exception as e:
        return json.dumps({"error": str(e)})
