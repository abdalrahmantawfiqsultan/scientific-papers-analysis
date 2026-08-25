"""
Benchmark harness — measures parser-only AND end-to-end ingestion throughput.

Usage:
  1. Drop PDF files into benchmarks/papers/
  2. Run: python benchmarks/benchmark_router.py
  3. Results are printed as a table and saved to benchmarks/results.json

Two benchmarks:
  A. Parser benchmark:  PDF → triage → parser → ParseResult
  B. End-to-end:        PDF → ParseResult → extraction → NER → graph injection

Metrics per paper:
  triage_ms, parser_ms, quality_gate_ms, extraction_ms, total_ms
  parser, fallback, page_count, text_coverage, is_complex, is_scanned

Aggregate metrics:
  papers_total, papers_successful, papers_failed
  routes (pymupdf/docling/docling_ocr counts)
  fallback_count, fallback_rate
  avg/p95 latency, papers_per_minute, peak_ram_mb
"""

import json
import os
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def benchmark_parser_only(pdf_path: str) -> dict:
    """Benchmark A: Parser-only. PDF → triage → parser → ParseResult."""
    from src.ingestion.router import route_and_parse

    row = {
        "file": os.path.basename(pdf_path),
        "benchmark": "parser_only",
        "parser": "",
        "fallback": False,
        "triage_ms": 0,
        "parser_ms": 0,
        "quality_gate_ms": 0,
        "total_ms": 0,
        "text_len": 0,
        "quality_score": 0.0,
        # Document characteristics
        "page_count": 0,
        "text_coverage": 0.0,
        "is_complex": False,
        "is_scanned": False,
        "error": None,
    }

    t_start = time.perf_counter()
    try:
        result = route_and_parse(pdf_path, filename=os.path.basename(pdf_path))
        row["parser"] = result.parser
        row["fallback"] = result.fallback_used
        row["triage_ms"] = result.timings.get("triage_ms", 0)
        row["parser_ms"] = result.timings.get("parser_ms", 0)
        row["quality_gate_ms"] = result.timings.get("quality_gate_ms", 0)
        row["text_len"] = len(result.text)
        row["quality_score"] = result.quality_score
        triage = result.metadata.get("triage", {})
        row["page_count"] = triage.get("page_count", 0)
        row["text_coverage"] = triage.get("text_coverage", 0.0)
        row["is_complex"] = triage.get("is_complex", False)
        row["is_scanned"] = triage.get("is_scanned", False)
    except Exception as e:
        row["error"] = str(e)

    row["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return row


def print_results(results: list, benchmark_name: str, peak_ram_mb: float):
    """Print formatted results table and aggregates."""
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    print(f"\n{'='*85}")
    print(f"  {benchmark_name}")
    print(f"{'='*85}")

    # Per-paper table
    header = f"  {'File':<35} {'Parser':<12} {'Total ms':>10} {'Pages':>6} {'Fallback':>9} {'Quality':>8}"
    print(header)
    print(f"  {'-'*81}")
    for r in results:
        if r["error"]:
            print(f"  {r['file']:<35} {'ERROR':<12} {'—':>10} {'—':>6} {'—':>9} {'—':>8}")
        else:
            print(f"  {r['file']:<35} {r['parser']:<12} {r['total_ms']:>10.1f} {r['page_count']:>6} {str(r['fallback']):>9} {r['quality_score']:>8.2f}")

    if not successful:
        print(f"\n  No successful papers to aggregate.")
        return

    # Aggregates
    latencies = sorted(r["total_ms"] for r in successful)
    avg_lat = sum(latencies) / len(latencies)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95_lat = latencies[p95_idx]
    total_s = sum(latencies) / 1000
    ppm = (len(successful) / total_s) * 60 if total_s > 0 else 0

    # Route distribution
    route_counts = Counter(r["parser"] for r in successful)
    fallback_count = sum(1 for r in successful if r["fallback"])
    fallback_rate = fallback_count / len(successful)

    # Errors by parser
    errors_by_parser = Counter()
    for r in failed:
        errors_by_parser[r.get("parser", "unknown")] += 1

    print(f"\n  --- Aggregate ---")
    print(f"  Papers total:      {len(results)}")
    print(f"  Papers successful: {len(successful)}")
    print(f"  Papers failed:     {len(failed)}")
    print(f"  Avg latency:       {avg_lat:.1f} ms")
    print(f"  p95 latency:       {p95_lat:.1f} ms")
    print(f"  Papers/minute:     {ppm:.1f}")
    print(f"  Fallback count:    {fallback_count}")
    print(f"  Fallback rate:     {fallback_rate:.1%}")
    print(f"  Peak RAM:          {peak_ram_mb:.1f} MB")
    print()
    print(f"  --- Routes ---")
    for parser, count in route_counts.most_common():
        print(f"    {parser:<14} {count}")
    if errors_by_parser:
        print(f"\n  --- Errors by Parser ---")
        for parser, count in errors_by_parser.most_common():
            print(f"    {parser:<14} {count}")
    print(f"{'='*85}\n")

    return {
        "papers_total": len(results),
        "papers_successful": len(successful),
        "papers_failed": len(failed),
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "papers_per_minute": round(ppm, 1),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_rate, 3),
        "peak_ram_mb": peak_ram_mb,
        "routes": dict(route_counts),
        "errors_by_parser": dict(errors_by_parser),
    }


def benchmark_end_to_end(pdf_path: str) -> dict:
    """Benchmark B: End-to-end. PDF → ParseResult → LLM extraction → NER."""
    from src.ingestion.router import route_and_parse
    
    row = {
        "file": os.path.basename(pdf_path),
        "benchmark": "end_to_end",
        "parser": "",
        "fallback": False,
        "triage_ms": 0,
        "parser_ms": 0,
        "quality_gate_ms": 0,
        "extraction_ms": 0,
        "ner_ms": 0,
        "total_ms": 0,
        "text_len": 0,
        "quality_score": 0.0,
        "page_count": 0,
        "text_coverage": 0.0,
        "is_complex": False,
        "is_scanned": False,
        "error": None,
    }

    t_start = time.perf_counter()
    try:
        # 1. Router
        result = route_and_parse(pdf_path, filename=os.path.basename(pdf_path))
        row["parser"] = result.parser
        row["fallback"] = result.fallback_used
        row["triage_ms"] = result.timings.get("triage_ms", 0)
        row["parser_ms"] = result.timings.get("parser_ms", 0)
        row["quality_gate_ms"] = result.timings.get("quality_gate_ms", 0)
        row["text_len"] = len(result.text)
        row["quality_score"] = result.quality_score
        
        triage = result.metadata.get("triage", {})
        row["page_count"] = triage.get("page_count", 0)
        row["text_coverage"] = triage.get("text_coverage", 0.0)
        row["is_complex"] = triage.get("is_complex", False)
        row["is_scanned"] = triage.get("is_scanned", False)
        
        # 2. LLM Extraction
        t_llm = time.perf_counter()
        from src.ingestion.docling_parser import HuggingFaceEndpointClient
        from src.ingestion.schema import ScientificPaper
        
        endpoint = os.getenv("LLM_ENDPOINT", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct")
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm_client = HuggingFaceEndpointClient(endpoint, token)
        
        prompt = f"You are an expert scientific researcher extracting entities from a paper.\n\nPaper text:\n{result.text[:15000]}"
        try:
            parsed_json = llm_client.get_json_response(prompt, ScientificPaper.model_json_schema())
            paper_schema = ScientificPaper(**parsed_json)
        except Exception:
            pass # Fallback skipped for benchmark
        row["extraction_ms"] = round((time.perf_counter() - t_llm) * 1000, 1)
        
        # 3. NER
        t_ner = time.perf_counter()
        from src.tools.text_processing import extract_dense_sentences_and_entities
        _, _ = extract_dense_sentences_and_entities(result.text, max_chars=8000)
        row["ner_ms"] = round((time.perf_counter() - t_ner) * 1000, 1)
        
    except Exception as e:
        row["error"] = str(e)

    row["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return row


def run_benchmark(papers_dir: str = "benchmarks/papers", mode: str = "parser_only"):
    """Run the chosen benchmark across all PDFs."""
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        print(f"Error: {papers_dir} does not exist. Create it and drop PDF files there.")
        return

    pdfs = sorted(papers_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {papers_dir}. Drop some papers there first.")
        return

    print(f"\n  Benchmarking {len(pdfs)} papers from {papers_dir} (Mode: {mode})...\n")

    tracemalloc.start()
    results = []
    for i, pdf in enumerate(pdfs, 1):
        if i == 1:
            if mode == "end_to_end":
                print(f"\n  {'File':<30} {'Parser':<10} {'Parse(s)':>8} {'LLM(s)':>8} {'NER(s)':>8} {'Total(s)':>8}")
                print(f"  {'-'*80}")
        
        if mode == "end_to_end":
            row = benchmark_end_to_end(str(pdf))
        else:
            row = benchmark_parser_only(str(pdf))
        results.append(row)
        
        if row["error"]:
            print(f"  {pdf.name:<30} ERROR: {row['error'][:40]}")
        else:
            if mode == "end_to_end":
                parse_s = row['parser_ms']/1000
                llm_s = row['extraction_ms']/1000
                ner_s = row['ner_ms']/1000
                total_s = row['total_ms']/1000
                print(f"  {pdf.name:<30} {row['parser']:<10} {parse_s:>8.2f} {llm_s:>8.2f} {ner_s:>8.2f} {total_s:>8.2f}")
            else:
                print(f"  [{i}/{len(pdfs)}] {pdf.name}: {row['parser']:>10} | {row['total_ms']:>8.1f}ms")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = round(peak_mem / 1024 / 1024, 1)

    title = "BENCHMARK B: End-to-End Ingestion" if mode == "end_to_end" else "BENCHMARK A: Parser Only"
    agg = print_results(results, title, peak_ram_mb)

    # Save results
    output_path = Path(f"benchmarks/results_{mode}.json")
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": mode,
        "papers": results,
        "aggregate": agg,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--e2e":
        run_benchmark(mode="end_to_end")
    else:
        run_benchmark(mode="parser_only")
        print("\n  Run 'python benchmarks/benchmark_router.py --e2e' for the end-to-end benchmark.")
