import numpy as np
from storage import get_all_notes
from embeddings import deserialize_embedding, compute_embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=5):
    query_emb = compute_embedding(query)
    results = []
    for text, source, rating, emb_blob in get_all_notes():
        if emb_blob is None:
            continue
        emb = deserialize_embedding(emb_blob)
        score = cosine_similarity(query_emb, emb)
        results.append((text, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
