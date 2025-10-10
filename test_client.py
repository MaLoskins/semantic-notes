#!/usr/bin/env python3
"""Integration test script for the Semantic Embedding Graph Engine backend.

Usage:
  python test_client.py --base-url http://localhost:8000
  python test_client.py --spawn --base-url http://127.0.0.1:8000
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import requests
except Exception:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

SAMPLE_DOCS = [
    "AI in Finance: using machine learning for fraud detection, risk scoring, and portfolio optimization.",
    "Neural Network Research: transformers, attention mechanisms, and representation learning.",
    "Blockchain Ledger Overview: decentralized consensus, cryptographic hashes, and transaction validation.",
    "Jazz music theory and improvisation techniques from the bebop era.",
    "Big data processing pipelines with Spark and distributed systems design.",
    "Auditing digital asset custodians: controls, risk matrices, and proof-of-reserves verification."
]

def wait_for_health(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/health", timeout=3)
            if r.ok and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def pretty(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def assert_close(a: float, b: float, tol: float = 1e-3, msg: str = ""):
    if abs(a-b) > tol:
        raise AssertionError(msg or f"Values not close: {a} vs {b} (tol={tol})")

def cosine_sim(a: List[float], b: List[float]) -> float:
    # inputs are expected unit vectors already
    return sum(x*y for x, y in zip(a, b))

def top_pairs(embeddings: List[List[float]], k: int = 5) -> List[Tuple[int,int,float]]:
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            w = cosine_sim(embeddings[i], embeddings[j])
            pairs.append((i, j, w))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]

def run_tests(base_url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Stats (optional)
    try:
        pretty("0) /api/stats")
        s = requests.get(f"{base_url}/api/stats", timeout=5)
        if s.ok:
            print(json.dumps(s.json(), indent=2))
    except Exception:
        pass

    # 1) Health
    pretty("1) /api/health")
    r = requests.get(f"{base_url}/api/health", timeout=5)
    r.raise_for_status()
    print("Health:", r.json())

    # 2) Embed
    pretty("2) /api/embed")
    t0 = time.time()
    r = requests.post(f"{base_url}/api/embed", json={"documents": SAMPLE_DOCS}, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    embeds = data["embeddings"]
    print(f"Received {len(embeds)} embeddings in {elapsed:.3f}s")
    assert len(embeds) == len(SAMPLE_DOCS), "Number of embeddings != number of docs"
    dim = len(embeds[0])
    print("Embedding dim:", dim)

    # Check L2 normalization
    for i, e in enumerate(embeds):
        norm = math.sqrt(sum(v*v for v in e))
        assert_close(norm, 1.0, tol=2e-2, msg=f"Embedding {i} not unit-normalized (||e||={norm:.4f})")
    print("All embeddings are ~unit length ✔")

    # Print top similar pairs
    pairs = top_pairs(embeds, k=6)
    print("\nTop similar pairs (cosine):")
    for i, j, w in pairs:
        print(f"  ({i}, {j}) -> {w:.3f}")

    # 3) Graph (threshold mode)
    pretty("3) /api/graph (threshold=0.5, PCA+KMeans)")
    payload = {
        "documents": SAMPLE_DOCS,
        "threshold": 0.5,
        "include_embeddings": False,
        "dr_method": "pca",
        "n_components": 2,
        "cluster": "kmeans",
        "n_clusters": 3,
        "labels": [f"Doc {i}" for i in range(len(SAMPLE_DOCS))]
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/api/graph", json=payload, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    g1 = r.json()
    print(f"Graph built in {elapsed:.3f}s with {len(g1['nodes'])} nodes and {len(g1['edges'])} edges")
    # Validate edges respect threshold
    for e in g1["edges"]:
        assert e["weight"] >= 0.5, "Edge weight below threshold"
        assert 0.0 <= e["weight"] <= 1.0, "Invalid cosine weight range"
    # Validate nodes have x/y and optional cluster
    for n in g1["nodes"]:
        assert "x" in n and "y" in n, "Missing x/y layout"
        assert "label" in n, "Missing label"
    (out_dir / "graph_threshold.json").write_text(json.dumps(g1, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "graph_threshold.json")

    # 3b) Threshold sweep to help pick a value
    pretty("3b) Threshold sweep")
    sweep = []
    for th in [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
        r = requests.post(f"{base_url}/api/graph", json={"documents": SAMPLE_DOCS, "threshold": th}, timeout=60)
        r.raise_for_status()
        g = r.json()
        sweep.append({"threshold": th, "edges": len(g["edges"])})
    print(json.dumps(sweep, indent=2))
    (out_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    # 4) Graph (kNN mode)
    pretty("4) /api/graph (top_k=2, UMAP if available else PCA)")
    payload = {
        "documents": SAMPLE_DOCS,
        "top_k": 2,
        "include_embeddings": True,
        "dr_method": "pca",
        "n_components": 2,
        "cluster": "none"
    }
    r = requests.post(f"{base_url}/api/graph", json=payload, timeout=60)
    r.raise_for_status()
    g2 = r.json()
    print(f"kNN Graph: {len(g2['nodes'])} nodes, {len(g2['edges'])} edges")
    # Basic validations
    for e in g2["edges"]:
        assert 0.0 <= e["weight"] <= 1.0, "Invalid cosine weight"
    # Verify embeddings included
    assert "embedding" in g2["nodes"][0], "Embeddings not included as requested"
    (out_dir / "graph_knn.json").write_text(json.dumps(g2, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "graph_knn.json")

    print("\nAll tests passed ✔")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--spawn", action="store_true", help="Spawn uvicorn server locally for the test run")
    parser.add_argument("--wait", type=float, default=45.0, help="Seconds to wait for server health")
    parser.add_argument("--out", default="tests/output", help="Where to write test artifacts")
    args = parser.parse_args()

    proc = None
    try:
        if args.spawn:
            # Spawn uvicorn from local project
            env = os.environ.copy()
            cmd = [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000", "--host", "127.0.0.1"]
            print("Spawning server:", " ".join(cmd))
            proc = subprocess.Popen(cmd, cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Wait for health
            if not wait_for_health(args.base_url, timeout=args.wait):
                # Dump a few lines of logs to help debugging
                if proc and proc.stdout:
                    print("\n--- Server output (last 50 lines) ---")
                    lines = proc.stdout.readlines()[-50:]
                    for line in lines:
                        print(line.rstrip())
                raise SystemExit("Server did not become healthy in time.")
        else:
            if not wait_for_health(args.base_url, timeout=args.wait):
                raise SystemExit("Server not reachable. Start it with: uvicorn main:app --reload")

        out_dir = Path(args.out)
        run_tests(args.base_url, out_dir)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

if __name__ == "__main__":
    main()
