import umap
import pandas as pd
from sklearn.cluster import KMeans
from storage import get_all_notes
from embeddings import deserialize_embedding
import numpy as np

def prepare_cluster_umap(n_clusters=5):
    notes = get_all_notes()
    embeddings, texts, sources, ratings = [], [], [], []

    for text, source, rating, emb_blob in notes:
        if emb_blob:
            embeddings.append(deserialize_embedding(emb_blob))
            texts.append(text)
            sources.append(source)
            ratings.append(rating)

    embeddings = np.array(embeddings)
    if len(embeddings) == 0:
        return None

    # Clustering
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # UMAP
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "text": texts,
        "source": sources,
        "rating": ratings,
        "cluster": labels,
        "x": embedding_2d[:,0],
        "y": embedding_2d[:,1]
    })

    return df
