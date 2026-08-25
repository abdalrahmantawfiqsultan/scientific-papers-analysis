import os
import urllib.request
import json
import time

corpus_dir = "benchmarks/papers"
os.makedirs(corpus_dir, exist_ok=True)

papers = [
    # 1. Clean born-digital CS paper (Attention Is All You Need)
    {"id": "1706.03762", "filename": "1706.03762_Attention.pdf", "category": "Clean born-digital", "expected_route": "pymupdf"},
    # 2. Modern NLP paper (BERT)
    {"id": "1810.04805", "filename": "1810.04805_BERT.pdf", "category": "Clean born-digital", "expected_route": "pymupdf"},
    # 3. Long paper > 20 pages (GPT-3, 75 pages)
    {"id": "2005.14165", "filename": "2005.14165_GPT3.pdf", "category": "Long paper (>20 pages)", "expected_route": "pymupdf"},
    # 4. Math-heavy multi-column paper (Maldacena AdS/CFT - 1997)
    {"id": "hep-th/9711200", "filename": "9711200_AdSCFT.pdf", "category": "Multi-column / older format", "expected_route": "docling"},
    # 5. Biology / figure-heavy paper (AlphaFold)
    {"id": "2107.03220", "filename": "2107.03220_AlphaFold.pdf", "category": "Table/figure-heavy", "expected_route": "docling"},
    # 6. Complex math/layout (Perelman Poincare conjecture)
    {"id": "math/0211159", "filename": "0211159_Perelman.pdf", "category": "Complex math", "expected_route": "docling"}
]

manifest = {"corpus": []}

for p in papers:
    pdf_url = f"https://arxiv.org/pdf/{p['id']}.pdf"
    filepath = os.path.join(corpus_dir, p['filename'])
    if not os.path.exists(filepath):
        print(f"Downloading {p['filename']}...")
        try:
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            time.sleep(2) # Be nice to arXiv
        except Exception as e:
            print(f"Failed to download {p['id']}: {e}")
            continue
    
    manifest["corpus"].append({
        "filename": p["filename"],
        "category": p["category"],
        "expected_route": p["expected_route"]
    })

with open("benchmarks/corpus_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Corpus populated successfully.")
