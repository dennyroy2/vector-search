"""Step 42 — index the arXiv embeddings and verify semantic search works
locally, before any of this is wrapped in a FastAPI endpoint (step 43).

This file is yours to write. Query embedding, calling the index, and
mapping returned row ids back to documents are exactly the three things
CLAUDE.md marks as "I write these" — and not by accident: the `search()`
function below IS next week's /search endpoint body, minus the HTTP
wrapper around it. Get it right here, where you can rerun it in a second,
not later while also debugging FastAPI.

Two checks this script should produce (see functions below):
  1. Quantitative — does HNSWIndex agree with brute-force search on THIS
     corpus? There's no hand-labeled "correct answer" file for arXiv like
     SIFT's ground truth — so exact (brute-force) search over the same
     embeddings plays the role ground truth played in benchmarks/. Same
     idea as evaluate.report(), reused.
  2. Qualitative — for a few queries you write yourself, do the returned
     paper titles actually look related? Recall@k can read 1.0 against a
     broken embedding pipeline and still be nonsense — only reading the
     titles catches that class of bug.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index import BruteForceIndex, HNSWIndex   # your ctypes wrapper
from embed import MODEL_NAME                    # reuse — must match embed.py exactly
from evaluate import report
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUN_DIR = DATA_DIR / "embeddings" / "arxiv_38985"


def load_run(run_dir: Path = RUN_DIR):

    vectors = np.load(run_dir / "vectors.npy")
    docs = []
    with open(RUN_DIR / "docs.jsonl") as f:
        for line in f:
            docs.append(json.loads(line))
    assert len(docs) == vectors.shape[0]
    assert docs[0]['id'] == 0
    assert docs[-1]['id'] == len(docs)-1

    return vectors, docs
            



_model = None  # lazy singleton — see embed_query() for why this matters


def embed_query(text: str) -> np.ndarray:

    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    vec = _model.encode([text], normalize_embeddings=True)[0]

    return np.ascontiguousarray(vec, dtype=np.float32)


def search(index, query_text: str, docs: list[dict], k: int = 10, **search_kwargs):
 
    ...
    q = embed_query(query_text)
    if isinstance(index, HNSWIndex):
      ids, dists, n_distance_computations =  index.search(q, k=k, **search_kwargs)
    else:
      ids, dists =  index.search(q, k=k, **search_kwargs)

    results = []

    for i in range(len(ids)):
        result = {"title": docs[ids[i]]["title"], "summary": docs[ids[i]]["summary"], "distance": dists[i]}
        results.append(result)

    return results


def check_recall_against_bruteforce(vectors, docs, k=10, n_queries=200, ef=50, seed=0):

    rng = np.random.default_rng(seed)
    queries = rng.choice(vectors, size=n_queries, replace=False)
    with BruteForceIndex(vectors) as bf:
      bf_ids, _ = bf.search_batch(queries, k=k)
    with HNSWIndex(vectors, M= 16, ef_construction=100) as hnsw:
      hnsw_results = []
      for q in queries:
        hnsw_ids, hnsw_dists, _ = hnsw.search(q, ef=ef, k=k)
        hnsw_results.append(hnsw_ids)

    report(np.stack(hnsw_results), bf_ids, k=k, label="hnsw vs bruteforce")



if __name__ == "__main__":
    vectors, docs = load_run()
    print(f"loaded {len(docs)} docs, vectors {vectors.shape}")

    check_recall_against_bruteforce(vectors, docs)

    # TODO: write 3-5 queries you actually care about, e.g.
    #   "reinforcement learning for robotic manipulation"
    #   "transformer models for protein structure prediction"
    # Build one HNSWIndex (reuse it across queries — building is the slow
    # part), then for each query print the top 5 titles + distances and
    # judge with your own eyes whether they're on-topic.
    queries = [
       "RISC-V",
       "GPU Programming",
       "SQL",
       "Markov Chain",
       "Dynamic Programming"
    ]

    h = HNSWIndex(vectors, M = 16, ef_construction= 100, seed=42, max_ef=2000)
    for q in queries:
      results = search(h, q, docs)
      print(f"Query is {q}. Results: \n")
      for i in range(5):
         print(f"Title : {results[i]["title"]}, Distance: {results[i]["distance"]}\n ")