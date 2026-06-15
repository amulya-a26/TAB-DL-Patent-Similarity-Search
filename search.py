# ============================================================
# Member 3 - Step 2: Core Search Engine
# Import this in app.py and backend.py
# ============================================================

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Paths — relative to repo root ───────────────────────────
INDEX_PATH   = "patent_faiss.index"
DATASET_PATH = "dl_set.csv"          # main dataset in repo
MODEL_NAME   = "all-MiniLM-L6-v2"
# ────────────────────────────────────────────────────────────

print("Loading model and index... (first time takes ~10 seconds)")

# Load SBERT model
model = SentenceTransformer(MODEL_NAME)

# Load FAISS index (built by build_index.py)
index = faiss.read_index(INDEX_PATH)

# Load patent dataset
df = pd.read_csv(DATASET_PATH, encoding="utf-8")

# Make sure we have the right column names
# dl_set.csv uses: patent_id, title, abstract, title_clean, abstract_clean
print(f"Ready — {index.ntotal} patents loaded.")


# ── Main search function ─────────────────────────────────────
def search_patents(user_idea: str, top_k: int = 3) -> tuple:
    """
    Search for patents similar to user_idea.

    Parameters:
        user_idea : str  — the user's invention description
        top_k     : int  — number of results (default 3)

    Returns:
        results   : list of dicts with title, patent_id, similarity %
        novelty   : float — novelty score 0–100
    """
    if not user_idea.strip():
        return [], 0.0

    # Step 1 — Encode user idea into a vector
    query_vec = model.encode(
        [user_idea],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    # Step 2 — Normalize (same as patent embeddings)
    faiss.normalize_L2(query_vec)

    # Step 3 — Search FAISS
    scores, indices = index.search(query_vec, top_k)

    # Step 4 — Build results list
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        results.append({
            "rank"       : len(results) + 1,
            "title"      : row["title"],
            "patent_id"  : row["patent_id"],
            "abstract"   : str(row["abstract"])[:300] + "...",
            "similarity" : round(float(score) * 100, 1)
        })

    # Step 5 — Novelty score: 100 - highest similarity
    novelty = round((1 - float(scores[0][0])) * 100, 1)

    return results, novelty


# ── Novelty label ────────────────────────────────────────────
def get_novelty_label(novelty_score: float) -> dict:
    if novelty_score >= 70:
        return {
            "label"  : "Highly Novel",
            "color"  : "green",
            "message": "Very few similar patents exist. Strong novelty!"
        }
    elif novelty_score >= 40:
        return {
            "label"  : "Moderately Novel",
            "color"  : "orange",
            "message": "Some similar patents exist. Consider refining your idea."
        }
    else:
        return {
            "label"  : "Low Novelty",
            "color"  : "red",
            "message": "Very similar patents already exist. High overlap detected."
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    test_idea = "A system that uses AI sensors to control traffic signal timings automatically"
    print(f"\nTest query: '{test_idea}'\n")

    results, novelty = search_patents(test_idea)

    print("Top Similar Patents:")
    for r in results:
        print(f"  {r['rank']}. {r['title']} — {r['similarity']}%")

    label = get_novelty_label(novelty)
    print(f"\nNovelty Score : {novelty}%")
    print(f"Assessment    : {label['label']} — {label['message']}")