"""Embed the arXiv abstracts corpus into unit-length float32 vectors.

Reads data/corpus/archive/arxiv_data.csv (columns: titles, summaries, terms)
and writes one directory per named run under data/embeddings/<name>/:

  vectors.npy   (n, dim) float32, C-contiguous, unit-norm rows
  docs.jsonl    one JSON object per line: {"id": i, "title": ..., "summary": ...}
  meta.json     model name, dimension, count, source file, timestamp

docs.jsonl line i corresponds to vectors.npy row i. The search service
(step 43) depends on that: the index hands back a row number, and
docs.jsonl[row number] is the document that row number means.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_CSV = DATA_DIR / "corpus" / "archive" / "arxiv_data.csv"
EMBED_DIR = DATA_DIR / "embeddings"

# A few abstracts + their term lists exceed the default 128KB field size.
csv.field_size_limit(sys.maxsize)


def load_texts(source: Path = CORPUS_CSV, limit: int | None = None):
    """Read the arXiv CSV. Returns (texts, docs).

    texts: list[str]  — "title. summary", the string that actually gets embedded.
    docs:  list[dict] — {"title": ..., "summary": ...}, saved next to the
           vectors so a search result can be displayed as more than a row number.

    24.7% of rows in the raw CSV are exact (title, summary) duplicates —
    verified by counting them. Identical text embeds to a bit-identical
    vector, so keeping duplicates wastes compute and makes two unrelated
    row ids return the same result at the same distance, which looks like
    a search bug and isn't one. Skipped here via a `seen` set.
    """
    seen = set()
    texts, docs = [], []
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["titles"].strip()
            summary = row["summaries"].strip().replace("\n", " ")
            if len(summary) < 20:  # near-empty abstract embeds to noise
                continue
            key = (title, summary)
            if key in seen:
                continue
            seen.add(key)
            texts.append(f"{title}. {summary}")
            docs.append({"title": title, "summary": summary})
            if limit is not None and len(texts) >= limit:
                break
    return texts, docs


def embed(texts, batch_size=256):
    """Turn a list of strings into an (n, dim) float32 array of unit vectors."""
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "encode() did not return unit vectors"

    return vectors


def save(vectors: np.ndarray, docs: list[dict], name: str) -> Path:
    """Save vectors as .npy, docs as JSONL, and a metadata sidecar."""
    out_dir = EMBED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "vectors.npy", vectors)

    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            f.write(json.dumps({"id": i, **doc}) + "\n")

    meta = {
        "model": MODEL_NAME,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "source": str(CORPUS_CSV.relative_to(DATA_DIR.parent)),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return out_dir


if __name__ == "__main__":
    LIMIT = None  # None = embed the full deduplicated corpus (~39k docs, ~2 min on CPU)

    texts, docs = load_texts(limit=LIMIT)
    print(f"loaded {len(texts)} documents")

    vectors = embed(texts)
    print(f"embedded: shape={vectors.shape} dtype={vectors.dtype}")

    out_dir = save(vectors, docs, f"arxiv_{len(texts)}")
    print(f"saved to {out_dir}")
