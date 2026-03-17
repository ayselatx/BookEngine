import streamlit as st
from storage import init_db, add_note_to_db
from embeddings import compute_embedding, serialize_embedding
from retrieval import search
from analysis import prepare_cluster_umap
import plotly.express as px

# Init DB
init_db()
st.set_page_config(layout="wide")
st.title("Cognitive Engine Interactive Dashboard")

tab1, tab2, tab3 = st.tabs(["Add Note / Book", "Search Notes", "Interactive Dashboard"])

# ---------------- ADD NOTE ----------------
with tab1:
    text = st.text_area("Enter a note, idea, or book info")
    source = st.text_input("Source (e.g., Goodreads, Babelio, personal note)", "note")
    rating = st.slider("Your rating (optional)", 1, 5, 3)

    if st.button("Save"):
        emb = compute_embedding(text)
        emb_blob = serialize_embedding(emb)
        add_note_to_db(text, source, rating, emb_blob)
        st.success("Note saved successfully!")

# ---------------- SEARCH ----------------
with tab2:
    query = st.text_input("Search your notes")
    if query:
        results = search(query)
        st.subheader("Top results")
        for text, score in results:
            st.write(f"{text}  \nScore: {score:.3f}")

# ---------------- DASHBOARD ----------------
with tab3:
    df = prepare_cluster_umap(n_clusters=5)
    if df is not None:
        st.subheader("Clusters / Themes (Interactive)")

        fig = px.scatter(
            df, x="x", y="y",
            color="cluster",
            hover_data=["text", "source", "rating"],
            title="2D UMAP + KMeans Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("Notes par cluster:")
        for c in sorted(df['cluster'].unique()):
            st.write(f"**Cluster {c}:**")
            for note in df[df['cluster']==c]['text']:
                st.write(f" - {note}")
    else:
        st.write("Aucune note pour visualiser les clusters.")
