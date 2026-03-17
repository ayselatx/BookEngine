from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_embedding(text: str):
    return model.encode(text)

def serialize_embedding(embedding):
    return embedding.tobytes()

def deserialize_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32)
