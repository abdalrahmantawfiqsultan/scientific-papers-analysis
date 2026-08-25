import json
import os
import re
from typing import Dict, List, Any, Tuple
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from rapidfuzz import fuzz
from src.ingestion.schema import ScientificPaper, Method, Researcher, Dataset, ResearchProblem, Metric, Result

_nlp_model = None

def get_nlp():
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm", disable=["lemmatizer", "ner"])
        except OSError:
            try:
                _nlp_model = spacy.load("en_core_sci_sm", disable=["lemmatizer", "ner"])
            except OSError:
                spacy.cli.download("en_core_web_sm")
                _nlp_model = spacy.load("en_core_web_sm", disable=["lemmatizer", "ner"])
    return _nlp_model

def load_vocab(filename: str) -> List[str]:
    vocab_path = os.path.join(os.path.dirname(__file__), "vocab", filename)
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def extract_entities_rule_based(text: str, title: str = "", abstract: str = "") -> Tuple[ScientificPaper, Dict]:
    """Extract entities deterministically using spaCy."""
    nlp = get_nlp()

    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    methods_vocab = load_vocab("methods.json")
    datasets_vocab = load_vocab("datasets.json")
    metrics_vocab = load_vocab("metrics.json")
    
    if methods_vocab:
        phrase_matcher.add("METHOD", [nlp.make_doc(m) for m in methods_vocab])
    if datasets_vocab:
        phrase_matcher.add("DATASET", [nlp.make_doc(d) for d in datasets_vocab])
    if metrics_vocab:
        phrase_matcher.add("METRIC", [nlp.make_doc(m) for m in metrics_vocab])
    
    doc = nlp(text[:20000]) # Bound extraction window to 20k chars
    
    methods_found = {}
    datasets_found = {}
    metrics_found = {}
    
    # Tier 1: PhraseMatcher (Gazetteer)
    matches = phrase_matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        entity_text = span.text.strip()
        
        # Simple deduplication via RapidFuzz within the document
        if label == "METHOD":
            if not any(fuzz.ratio(entity_text.lower(), k.lower()) > 90 for k in methods_found):
                methods_found[entity_text] = "spacy_phrasematcher"
        elif label == "DATASET":
            if not any(fuzz.ratio(entity_text.lower(), k.lower()) > 90 for k in datasets_found):
                datasets_found[entity_text] = "spacy_phrasematcher"
        elif label == "METRIC":
            if not any(fuzz.ratio(entity_text.lower(), k.lower()) > 90 for k in metrics_found):
                metrics_found[entity_text] = "spacy_phrasematcher"

    # Tier 2: Dependency/Pattern Matcher (Syntactic)
    matcher = Matcher(nlp.vocab)
    
    # "we use X", "we propose X", "we apply X" -> Method
    method_pattern = [
        {"LOWER": "we"},
        {"LOWER": {"IN": ["use", "propose", "employ", "apply", "present", "introduce", "utilize"]}},
        {"OP": "*", "IS_PUNCT": False, "LENGTH": {"<": 5}}, # capture up to 5 tokens of the object
        {"POS": "NOUN"} 
    ]
    matcher.add("PATTERN_METHOD", [method_pattern])
    
    # "tested on X", "evaluated on X" -> Dataset
    dataset_pattern = [
        {"LOWER": {"IN": ["tested", "evaluated", "trained"]}},
        {"LOWER": "on"},
        {"OP": "*", "IS_PUNCT": False, "LENGTH": {"<": 4}},
        {"POS": {"IN": ["PROPN", "NOUN"]}}
    ]
    matcher.add("PATTERN_DATASET", [dataset_pattern])
    
    # "measured by X", "using X" (where X is often metric) -> Metric
    metric_pattern = [
        {"LOWER": {"IN": ["measured", "evaluated"]}},
        {"LOWER": "by"},
        {"OP": "*", "IS_PUNCT": False, "LENGTH": {"<": 4}},
        {"POS": {"IN": ["NOUN", "PROPN"]}}
    ]
    matcher.add("PATTERN_METRIC", [metric_pattern])
    
    # Run Tier 2 matcher
    # Only tag if not already found
    pattern_matches = matcher(doc)
    for match_id, start, end in pattern_matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        
        # The actual entity is usually the last few tokens of the match (excluding 'we use')
        if label == "PATTERN_METHOD" and len(span) > 2:
            entity_text = span[2:].text.strip()
            if entity_text and not any(fuzz.ratio(entity_text.lower(), k.lower()) > 85 for k in methods_found):
                methods_found[entity_text] = "spacy_dependency_pattern"
        elif label == "PATTERN_DATASET" and len(span) > 2:
            entity_text = span[2:].text.strip()
            if entity_text and not any(fuzz.ratio(entity_text.lower(), k.lower()) > 85 for k in datasets_found):
                datasets_found[entity_text] = "spacy_dependency_pattern"
        elif label == "PATTERN_METRIC" and len(span) > 2:
            entity_text = span[2:].text.strip()
            if entity_text and not any(fuzz.ratio(entity_text.lower(), k.lower()) > 85 for k in metrics_found):
                metrics_found[entity_text] = "spacy_dependency_pattern"

    # 3. Section Heuristics for Problems and Results
    problems = []
    results = []
    
    # Search Abstract and first 5000 chars for Problem
    search_text = (abstract + "\n" + text[:5000]).replace("\n", " ")
    prob_match = re.search(r'(?i)(we\s+(?:address|investigate|tackle|study)\s+(?:the\s+problem\s+of|the\s+issue\s+of)?\s*[^.]{10,100}\.)', search_text)
    if prob_match:
        problems.append({"name": prob_match.group(1).strip(), "method": "section_heuristic"})
        
    # Search Abstract and last 5000 chars for Result
    res_text = (abstract + "\n" + text[-5000:]).replace("\n", " ")
    res_match = re.search(r'(?i)(we\s+(?:show|demonstrate|achieve|outperform)\s*[^.]{10,100}\.)', res_text)
    if res_match:
        results.append({"description": res_match.group(1).strip(), "method": "section_heuristic"})
    
    # 4. Citation Extraction from References / Bibliography section
    citations = []
    ref_match = re.search(r'(?i)(?:##+\s*(?:References|Bibliography)|(?:References|Bibliography)\s*\n)([\s\S]+)', text)
    if ref_match:
        ref_section = ref_match.group(1)[:10000]
        ref_lines = re.findall(r'(?:\[\d+\]|\d+\.)\s*([^\n\r]+)', ref_section)
        for r_line in ref_lines[:20]:  # Cap at 20 citations
            cleaned_ref = r_line.strip()
            if len(cleaned_ref) > 15:
                title_quote = re.search(r'["“]([^"”]+)["”]', cleaned_ref)
                if title_quote and len(title_quote.group(1)) > 10:
                    citations.append(title_quote.group(1).strip())
                else:
                    citations.append(cleaned_ref[:100].strip())

    # Build the final output objects
    out_methods = [Method(name=m, category="Algorithm") for m in methods_found.keys()]
    out_datasets = [Dataset(name=d) for d in datasets_found.keys()]
    out_metrics = [Metric(name=m, value="") for m in metrics_found.keys()]
    out_problems = [ResearchProblem(name=p["name"]) for p in problems]
    out_results = [Result(description=r["description"], improvement="") for r in results]
    
    paper = ScientificPaper(
        title=title,
        abstract=abstract,
        year=2025, # fallback, NER overwrites this
        doi="",
        authors=[], # NER overwrites this
        uses_methods=out_methods,
        uses_datasets=out_datasets,
        addresses_problems=out_problems,
        evaluated_by=out_metrics,
        reports_results=out_results,
        cites=citations,
        builds_on=[],
        extends=[],
        compares_to=[],
        contradicts=[]
    )
    
    provenance_map = {
        "methods": methods_found,
        "datasets": datasets_found,
        "metrics": metrics_found,
        "problems": {p["name"]: p["method"] for p in problems},
        "results": {r["description"]: r["method"] for r in results},
        "citations": {c: "citation_regex" for c in citations}
    }
    
    return paper, provenance_map
