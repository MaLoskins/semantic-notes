# Semantic Embedding Graph Engine — Python Backend

## Quick Start

```bash
# 1) Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the server
python main.py
```

New terminal:

```
cd semantic-notes-app
# 1) Install deps
npm install
# 2) Start server
npm run dev

```
Server runs on `http://localhost:3000` by default.

---

## Configuration

Environment variables (optional):

- `EMBEDDING_MODEL_NAME` — default: `sentence-transformers/all-MiniLM-L6-v2`
- `EMBEDDING_DEVICE` — e.g. `cpu`, `cuda`
- `EMBEDDING_BATCH_SIZE` — default: `64`

To use **OpenAI embeddings** instead of local models:
- Set `OPENAI_API_KEY` and choose:
  - `EMBEDDING_MODEL_NAME=text-embedding-3-small` (or other)
  - In code, initialize `EmbeddingService(backend="openai")` or set it when wiring dependencies.
> Note: `openai` package is not included by default; install and wire if needed.

For faster kNN on large corpora, install FAISS (optional):
- CPU: `pip install faiss-cpu` (availability varies by platform)
- GPU: `pip install faiss-gpu`

---

## API

### Health
`GET /api/health` → `{ "status": "ok" }`

### Embed
`POST /api/embed`

**Request**
```json
{
  "documents": ["lorem ipsum", "dolor sit amet"]
}
```

**Response**
```json
{
  "embeddings": [[0.01, 0.02, ...], [0.03, 0.04, ...]]
}
```

### Graph
`POST /api/graph`

**Request**
```json
{
  "documents": ["AI in Finance", "Neural Networks", "Blockchain Overview"],
  "threshold": 0.7,
  "include_embeddings": false,
  "dr_method": "pca",
  "n_components": 2,
  "cluster": "kmeans",
  "n_clusters": 3,
  "labels": ["Doc A", "Doc B", "Doc C"]
}
```

- If `threshold` is provided, `top_k` is ignored.
- If `threshold` is omitted, `top_k` (default 5) is used to build kNN edges.
- `dr_method`: `none` | `pca` | `umap` | `tsne`
- `cluster`: `none` | `kmeans` | `agglomerative`

**Response**
```json
{
  "nodes": [
    {"id": "0", "label": "Doc A", "x": -0.13, "y": 1.42, "cluster": 1},
    {"id": "1", "label": "Doc B", "x": 0.65, "y": 0.08, "cluster": 0},
    {"id": "2", "label": "Doc C", "x": -0.52, "y": -1.10, "cluster": 1}
  ],
  "edges": [
    {"source": "0", "target": "1", "weight": 0.842193},
    {"source": "1", "target": "2", "weight": 0.793251}
  ]
}
```

---

## Files
- `main.py` — FastAPI app & routes
- `embedding_service.py` — lazy model loader, batch encode, L2‑norm, LRU cache
- `graph_service.py` — kNN / threshold graph, optional DR & clustering
- `requirements.txt` — dependencies
- `README.md` — this file

---

## Notes & Tips
- Embeddings are **unit‑normalized** so cosine similarity reduces to a dot product.
- For hundreds to a few thousand docs, NumPy/Sklearn paths are plenty fast.
- For large corpora (10k+), use FAISS kNN and consider persistently caching embeddings.
- Front‑ends (D3/ReactFlow/Cytoscape) can use returned `x, y` for layout, or compute their own.
- Prefer passing human‑readable `labels` in `/api/graph` for nicer node titles.
