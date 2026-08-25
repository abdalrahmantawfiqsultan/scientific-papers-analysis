import numpy as np
import json

# Cache the models globally
_embedder = None
_cross_encoder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return _embedder

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
    return _cross_encoder

def search_concepts_in_memory(kg, query: str, top_k: int = 3):
    """Calculates Hybrid Search (BM25 Lexical + Vector Semantic) using Reciprocal Rank Fusion."""
    from rank_bm25 import BM25Okapi
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 1. Prepare Corpus
    corpus_ids = []
    corpus_texts = []
    for node_id, data in kg.graph.nodes(data=True):
        if "name" in data:
            corpus_ids.append(node_id)
            corpus_texts.append(data["name"])
            
    if not corpus_ids:
        return json.dumps([])
        
    # 2. Semantic Search (Dense Vector)
    query_emb = get_embedder().encode([query])
    
    # FETCH FROM CACHE (O(1) after first computation)
    corpus_embs = kg.vectors.get_embeddings(corpus_ids)
    
    dense_scores = cosine_similarity(query_emb, corpus_embs)[0]
    
    # Rank Dense
    dense_ranks = {corpus_ids[i]: rank for rank, i in enumerate(np.argsort(dense_scores)[::-1])}
    
    # 3. Lexical Search (BM25)
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Rank Sparse
    sparse_ranks = {corpus_ids[i]: rank for rank, i in enumerate(np.argsort(sparse_scores)[::-1])}
    
    # 4. Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (k + rank)  where k is usually 60
    rrf_scores = {}
    k = 60
    for node_id in corpus_ids:
        dense_rank = dense_ranks[node_id]
        sparse_rank = sparse_ranks[node_id]
        rrf_scores[node_id] = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
        
    # Get top_k by RRF score
    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    
    # Return names for the LLM
    top_names = [kg.graph.nodes[node_id]["name"] for node_id in top_indices]
    return json.dumps(top_names)
