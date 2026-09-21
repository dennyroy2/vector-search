# Step 47 — one image containing the C build, the Python wrapper, and the
# FastAPI service. Two stages: build the C library where gcc is available,
# then copy only the compiled result into a clean Python image that never
# needs a C compiler at all.

# ---- Stage 1: build the C shared library ----------------------------------
FROM debian:12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY src/ ./src/
COPY include/ ./include/
RUN make


# ---- Stage 2: the service ---------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first, deliberately. sentence-transformers depends on
# torch, and pip's default Linux wheel for torch bundles several GB of
# CUDA libraries that are useless on a CPU-only EC2 instance (ROADMAP's
# own target). Installing the CPU build first satisfies that dependency
# before anything else gets a chance to pull in the GPU one.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image at build time, not runtime.
# Without this, a fresh container's first request would reach out to
# Hugging Face Hub to download it — an external dependency and a multi-
# second delay this service shouldn't have on every cold start. Costs
# build time and ~90MB of image size; buys a container that starts
# without needing the internet, and matches the "fully reproducible"
# reason this model was picked in the first place (see ROADMAP).
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Only the compiled .so crosses from the builder stage — not gcc, not
# the C source, not intermediate .o files. This image never has a C
# compiler in it.
COPY --from=builder /app/build/libvindex.so ./build/libvindex.so

COPY python/ ./python/

# The embeddings under data/embeddings/ are deliberately NOT copied in.
# They're generated data, not code — this project's own rule is "don't
# bake data/ into anything," data/ also holds the multi-gigabyte SIFT
# sets, and a blanket copy would drag those in by accident too. Mount
# the embeddings directory at `docker run` time instead (see below);
# the image stays small and isn't tied to one specific embedded corpus.
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--app-dir", "python", "--host", "0.0.0.0", "--port", "8000"]
