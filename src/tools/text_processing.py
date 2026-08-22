import streamlit as st
import spacy
import spacy.cli

# Cap on how much raw text we run the spaCy pipeline over. We only ever keep
# max_chars of *output*, so there's no reason to run NER/parsing across an
# entire 40k-character PDF when the first ~20k chars (title, abstract, intro,
# methods) almost always contain enough entity-dense sentences to fill 4000
# output chars. This is the actual CPU bottleneck in this file — not model
# loading, which is already cached.
NLP_WINDOW_CHARS = 20_000

# Entity types relevant to paper metadata (authors, orgs, dates) vs. noise
# types (CARDINAL, ORDINAL, PERCENT, MONEY, QUANTITY) that inflate the
# "has_ents" density filter without being useful downstream.
METADATA_ENT_LABELS = {"PERSON", "ORG", "DATE", "GPE"}


@st.cache_resource
def get_spacy_model():
    """Lazily load and cache the scispaCy/spaCy NER model, or return None if unavailable.
    Disables pipeline components we never use (lemmatizer, attribute_ruler,
    tagger) to cut per-call latency — we only need sentence segmentation (parser)
    and NER."""
    disabled = ["lemmatizer", "attribute_ruler", "tagger"]
    try:
        return spacy.load("en_core_sci_sm", disable=disabled)
    except OSError:
        try:
            return spacy.load("en_core_web_sm", disable=disabled)
        except OSError:
            try:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm", disable=disabled)
            except Exception:
                return None


def extract_dense_sentences(full_text: str, max_chars: int = 4000) -> str:
    """Filter out non-informative sentences and return a compressed, entity-dense chunk.
    If spaCy is unavailable, falls back to a simple prefix slice."""
    text, ents = extract_dense_sentences_and_entities(full_text, max_chars=max_chars)
    return text


def extract_dense_sentences_and_entities(
    full_text: str, max_chars: int = 4000
) -> tuple[str, list[dict]]:
    """Same compression as extract_dense_sentences, but also returns the
    PERSON/ORG/DATE/GPE entities spaCy already found while scanning the text —
    previously discarded. Feed these into paper ingestion to populate
    authors/organizations/publication_date instead of running a second pass."""
    nlp = get_spacy_model()

    if nlp is None:
        # Fallback if no NLP models are installed/downloadable
        return full_text[:max_chars], []

    # Bound the NLP pass — see NLP_WINDOW_CHARS comment above.
    windowed_text = full_text[:NLP_WINDOW_CHARS]
    doc = nlp(windowed_text)

    dense_sentences = []
    seen_entities = {}  # dedupe by (text, label)
    for sent in doc.sents:
        if len(sent.ents) > 0:
            dense_sentences.append(sent.text)
        for ent in sent.ents:
            if ent.label_ in METADATA_ENT_LABELS:
                key = (ent.text.strip(), ent.label_)
                seen_entities[key] = {"text": ent.text.strip(), "label": ent.label_}

    entities = list(seen_entities.values())

    if not dense_sentences:
        # If no entities were found at all, just return the raw text slice
        return full_text[:max_chars], entities

    compressed_text = " ".join(dense_sentences)
    return compressed_text[:max_chars], entities


def normalize_date_to_year(date_text: str) -> int | None:
    """Attempt to parse a raw date string (from spaCy NER) into an integer year.
    Uses dateutil.parser for flexible parsing of formats like:
      'March 2023', '2023-03-15', 'in 2019', '15th January 2020'
    Falls back to regex extraction of a 4-digit year if dateutil fails.
    Returns None if no year can be extracted."""
    import re

    # Quick regex pass first — catches "2023", "in 2019", etc.
    year_match = re.search(r'\b(19|20)\d{2}\b', date_text)

    try:
        from dateutil import parser as dateutil_parser
        parsed = dateutil_parser.parse(date_text, fuzzy=True)
        return parsed.year
    except Exception:
        pass

    if year_match:
        return int(year_match.group(0))

    return None
