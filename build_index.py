
import faiss
import numpy as np
import os

EMBEDDINGS_PATH = "patent_embeddings.npy"
INDEX_SAVE_PATH = "patent_faiss.index"

def build_faiss_index():
    print("=" * 50)
    print("  FAISS Index Builder")
    print("=" * 50)

    print("\n[1/4] Loading patent_embeddings.npy...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"ERROR: {EMBEDDINGS_PATH} not found.")
        print("Make sure you are running this from the repo root folder.")
        return

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"      Loaded: {embeddings.shape[0]} patents, {embeddings.shape[1]} dimensions")
    print("\n[2/4] Converting to float32...")
    embeddings = embeddings.astype("float32")
    print("\n[3/4] Normalizing vectors...")
    faiss.normalize_L2(embeddings)
    print("\n[4/4] Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_SAVE_PATH)
    print("\n" + "=" * 50)
    print(f"  Done! Saved: {INDEX_SAVE_PATH}")
    print(f"  Total patents indexed: {index.ntotal}")
    print("=" * 50)


if __name__ == "__main__":
    build_faiss_index()