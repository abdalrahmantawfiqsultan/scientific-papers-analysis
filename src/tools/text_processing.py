import streamlit as st
import spacy
import spacy.cli

@st.cache_resource
def get_spacy_model():
    """Lazily load and cache the scispaCy/spaCy NER model, or return None if unavailable."""
    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            try:
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except:
                return None

def extract_dense_sentences(full_text: str, max_chars: int = 4000) -> str:
    """Filter out non-informative sentences and return a compressed, entity-dense chunk.
    If spaCy is unavailable, falls back to a simple prefix slice."""
    nlp = get_spacy_model()
    
    if nlp is None:
        # Fallback if no NLP models are installed/downloadable
        return full_text[:max_chars]
        
    doc = nlp(full_text)
    
    dense_sentences = []
    for sent in doc.sents:
        if len(sent.ents) > 0:
            dense_sentences.append(sent.text)
            
    if not dense_sentences:
        # If no entities were found at all, just return the raw text slice
        return full_text[:max_chars]
        
    compressed_text = " ".join(dense_sentences)
    return compressed_text[:max_chars]
