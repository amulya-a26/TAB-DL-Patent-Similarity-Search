# Patent Similarity and Novelty Detection using Transformer-Based Semantic Search

A Deep Learning system that identifies patents similar to a user's invention idea and estimates its novelty using **Sentence-BERT embeddings**, **FAISS vector search**, and **Cosine Similarity**.

----
##  Problem Statement

Researchers and innovators often find it difficult to determine whether their ideas are similar to existing patents. Manual patent searches are time-consuming and may miss semantically related inventions.

This system automatically identifies similar patents and estimates the novelty of a proposed idea — helping users quickly assess the uniqueness of their innovations.

---

##  Solution Overview

The system converts patent abstracts and user-submitted ideas into vector embeddings using a Transformer-based model (Sentence-BERT). Cosine similarity is then used to find the most relevant existing patents. Based on the similarity scores, the system retrieves the top matching patents and generates a novelty score.

```
User Idea
    ↓
SBERT Encoding (MiniLM-L6-v2)
    ↓
384-Dimensional Vector
    ↓
FAISS Similarity Search
    ↓
Top-3 Similar Patents + Novelty Score
```

---

##  Project Structure

```
 Patent-Similarity-and-Novelty-Detection/
│
├──  dl_set.csv                  
├──  project.ipynb               
├──  patent_embeddings.npy       
├──  patent_embeddings.csv       
│
├──  build_index.py              
├──  search.py                   
├──  backend.py                  
├──  app.py                      
│
└──  README.md                  
```

---

##  System Architecture

### Step 1 — Data & Preprocessing

- Downloaded patent abstracts via **SerpAPI** (Google Patents scraping)
- Merged with Kaggle patent dataset
- Performed EDA: analyzed abstract lengths, topic distribution, duplicates
- Cleaned text: lowercased, removed special characters, removed nulls and duplicates
- Generated `title_clean` and `abstract_clean` columns
- Prepared final dataset: `dl_set.csv`

### Step 2 — Transformer & Embeddings

- Loaded **Sentence-BERT** (`all-MiniLM-L6-v2`) — a 6-layer Transformer with 22M parameters
- Tokenized all  patent abstracts
- Generated 384-dimensional embeddings for every abstract
- Saved embeddings as `patent_embeddings.npy` 
- Evaluated embedding quality using `EmbeddingSimilarityEvaluator`
- Split dataset into train / dev / holdout sets (70 / 15 / 15)

### Member 3 — Similarity Search & Deployment

- Built a **FAISS index** (`IndexFlatIP`) from patent embeddings for fast vector search
- Implemented cosine similarity search to find top-K matching patents
- Calculated **novelty score** = `100% − highest similarity score`
- Built **Flask REST API** (`/search` endpoint) for programmatic access
- Built **Gradio web app** for interactive user-facing demo

---

##  How It Works

### Embedding Generation
Every patent abstract is converted into a 384-dimensional vector by SBERT. Semantically similar texts produce vectors that are close together in vector space.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(abstracts)  # shape: (3488, 384)
```

### Similarity Search
When a user submits an idea, it is encoded the same way and compared against all patent vectors using cosine similarity via FAISS.

```python
scores, indices = index.search(query_vector, top_k=3)
```

### Novelty Score
```
Novelty Score = 100% − Highest Similarity Score

Example:
  Most similar patent = 84% similar
  Novelty Score = 100 − 84 = 16%  (low novelty)

  Most similar patent = 25% similar
  Novelty Score = 100 − 25 = 75%  (highly novel)
```

### Novelty Interpretation
| Score | Label | Meaning |
|-------|-------|---------|
| 70% – 100% |  Highly Novel | Very few similar patents exist |
| 40% – 69% |  Moderately Novel | Some similar patents exist |
| 0% – 39% |  Low Novelty | Very similar patents already exist |

---

##  Installation & Setup

### Prerequisites
- Python 3.11 or 3.13 (recommended)
- Git

### Step 1 — Clone the repository
```bash
git clone "link"
cd Patent-Similarity-and-Novelty-Detection-using-Transformer-Based-Semantic-Search
```

### Step 2 — Install dependencies
```bash
pip install faiss-cpu sentence-transformers torch numpy pandas flask flask-cors gradio
```

### Step 3 — Build FAISS index (run once)
```bash
python build_index.py
```

### Step 4 — Test the search engine
```bash
python search.py
```

### Step 5 — Launch the web app
```bash
python app.py
```
Opens at: `http://localhost:7860`

### Step 6 — (Optional) Run Flask API
```bash
python backend.py
```
API available at: `http://localhost:5000`

---

##  API Usage

**Endpoint:** `POST /search`

**Request:**
```json
{
  "idea": "A system that uses AI to control traffic signal timings",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "A system that uses AI to control traffic signal timings",
  "results": [
    {"rank": 1, "title": "Traffic management device and system", "patent_id": "US10460601B2", "similarity": 84.2},
    {"rank": 2, "title": "Intelligent road facility system", "patent_id": "CN108447291B", "similarity": 79.1},
    {"rank": 3, "title": "Autonomous vehicle control system", "patent_id": "US12424101B2", "similarity": 74.3}
  ],
  "novelty_score": 15.8,
  "assessment": {
    "label": "Low Novelty",
    "color": "red",
    "message": "Very similar patents already exist. High overlap detected."
  }
}
```

---

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | ≥2.0 | SBERT model for text embeddings |
| `torch` | ≥2.0 | PyTorch backend for transformer model |
| `faiss-cpu` | ≥1.7 | Fast vector similarity search |
| `pandas` | ≥1.5 | Dataset handling |
| `numpy` | ≥1.23 | Numerical operations |
| `flask` | ≥2.0 | REST API backend |
| `flask-cors` | ≥3.0 | Cross-origin requests |
| `gradio` | ≥4.0 | Web UI framework |

---

##  Model Details

| Property | Details |
|----------|---------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | 6-layer Transformer (BERT-based) |
| Parameters | 22 million |
| Embedding dimension | 384 |
| Max input length | 128 tokens |
| Similarity metric | Cosine similarity via FAISS IndexFlatIP |
| Pre-trained on | 1B+ sentence pairs |

---

##  Example Output

```
Input: "A drone that delivers medicine to remote areas using GPS"

Top Similar Patents:
  1. Unmanned aerial vehicle drug delivery system   — 81.3%
  2. Remote medication transport via autonomous UAV — 76.8%
  3. GPS-guided delivery robot for rural healthcare — 71.2%

Novelty Score: 18.7%  Low Novelty
Assessment: Very similar patents already exist. High overlap detected.
```
