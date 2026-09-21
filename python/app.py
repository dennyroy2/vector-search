"""Step 43 — a FastAPI service wrapping the search logic from local_search.py.

The app setup below (startup/shutdown, CORS, request/response shapes) is
plumbing — written for you, per CLAUDE.md. The /search route's actual body
is yours: it's the same "query embedding, index call, ID-to-document
mapping" you already wrote and proved correct in local_search.py's
search() function. This route should call that function, not re-derive it.

Run it with:  uvicorn app:app --reload   (from inside python/)
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from index import HNSWIndex
from local_search import load_run, search, embed_query  # your step 42 code
from rate_limit import limiter
from latency import tracker

# --- API-key auth (step 44) ------------------------------------------------
# API_KEYS maps key string -> a label for whoever holds it (a name, a
# client id — anything you'd want to see in a log). Step 45's per-key
# rate limiting needs to tell callers apart, which is why this is a dict
# of many possible keys, not a single shared password.
#
# Keys come from the API_KEYS environment variable as a JSON object, e.g.
#   export API_KEYS='{"abc123": "denny-local", "xyz789": "load-test"}'
# Environment variables, not a constant in this file, because a key
# baked into source code ends up in git history forever, even after
# you "remove" it later. A fallback dev key is provided so the server
# still runs with zero setup while you're building locally.
API_KEYS: dict[str, str] = json.loads(
    os.environ.get("API_KEYS", '{"dev-key-123": "local-dev"}')
)

# APIKeyHeader tells FastAPI to look for the key in a request header
# named X-API-Key (not a query string param — a query string gets
# logged in browser history, proxy logs, and server access logs by
# default; a header does not).
_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(_api_key_header)) -> str:
    """A FastAPI dependency: runs before the route body, given a chance
    to reject the request first. Returns the caller's label on success,
    which step 45 can use to key its per-caller rate limit.
    """
    if key not in API_KEYS:
        # 401 Unauthorized, not 403 Forbidden: 401 means "you haven't
        # proven who you are"; 403 means "I know who you are and you
        # still can't do this." A missing/wrong key is the former.
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return API_KEYS[key]

def enforce_rate_limit(caller: str = Depends(require_api_key)) -> str:
    if not limiter.allow(caller):
        raise HTTPException(status_code=429, detail="Exceeded rate limit. Try again shortly")
    return caller


# --- Request/response shapes ---------------------------------------------
# Pydantic models describe the shape of JSON going in and out. FastAPI
# uses SearchRequest to validate an incoming request automatically — if
# the JSON doesn't have a string "query", the caller gets back a 422
# error before your endpoint function ever runs. You don't hand-write
# that checking. SearchResponse does the same job in reverse: return one
# of these and FastAPI serializes it to JSON for you.

class SearchRequest(BaseModel):
    query: str
    k: int = 10


class SearchResult(BaseModel):
    title: str
    summary: str
    distance: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# --- Startup / shutdown ----------------------------------------------------
# Building the index takes real time (see build_seconds on HNSWIndex) and
# it never changes while the server runs — it must happen exactly once,
# not on every request. `lifespan` is FastAPI's hook for "run this before
# the server accepts traffic, and run this when it's shutting down."
# Everything before `yield` is startup; everything after is shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    vectors, docs = load_run()
    app.state.docs = docs
    app.state.index = HNSWIndex(vectors, M=16, ef_construction=100)
    embed_query("warmup")
    print(f"startup: loaded {len(docs)} docs, index ready")

    yield  # the server handles requests here, until it's told to stop

    app.state.index.close()


app = FastAPI(title="vector-search", lifespan=lifespan)

# CORS: browsers block JavaScript on one origin (e.g. a frontend on
# localhost:3000) from calling an API on a different origin (this server)
# unless the server explicitly allows it — a browser security default,
# not a bug you're fixing. Wide open here for local development; narrow
# `allow_origins` to your actual frontend's URL before step 48 puts this
# on a real server reachable by anyone.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(app.state.docs)}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest, caller: str = Depends(enforce_rate_limit)
) -> SearchResponse:

    
    """The actual search endpoint body. Yours to write.

    `caller` is the label for whoever's API key was used (see
    require_api_key above) — FastAPI resolves it before this function
    body even starts, and rejects the request first if the key's bad.
    You don't have to do anything with `caller` yet; step 45 will.

    """
    if request.k > app.state.index.max_ef:
        request.k = app.state.index.max_ef
    elif request.k < 1:
        raise HTTPException(status_code=400, detail="K must be greater than or equal to 1\n")
    if request.query.strip() == "":
        raise HTTPException(status_code=400, detail = "No string inputted\n")
    
    results = search(app.state.index, request.query, app.state.docs, k=request.k)
    search_results = [SearchResult(**result) for result in results]

    return SearchResponse(query = request.query, results = search_results)

@app.middleware("http")
async def time_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path == "/search" and response.status_code == 200:
        duration = time.perf_counter() - start
        tracker.record(duration)
    return response

@app.get("/stats")
def stats():
    return tracker.summary()


# --- Wiring this into app.py (yours) --------------------------------------
#
# FastAPI's HTTP middleware hook — a function decorated with
# @app.middleware("http") — wraps EVERY request through the app, not
# just one route. Its shape:
#
#     @app.middleware("http")
#     async def time_requests(request: Request, call_next):
#         start = time.perf_counter()
#         response = await call_next(request)   # runs the actual request
#         duration = time.perf_counter() - start
#         ...
#         return response
#
# `call_next(request)` is what actually lets the request continue into
# your auth check, rate limiter, and route handler — the timing you take
# around that call measures everything a real caller experiences, not
# just your endpoint function's own code.
#
# DECISION FOR YOU: should this record every request (including /health
# checks, and requests that got rejected with 401/429), or only real,
# successful /search calls? A near-zero-duration auth rejection mixed
# into the same percentiles as actual searches will drag p50 down and
# make the number mean something different than "how fast is search."
# request.url.path is available inside the middleware if you want to
# filter. Pick one, and be ready to say why — there's a real argument
# for "server-wide latency" too, just make it on purpose, not by default.
#
# Then a small route, same shape as /health:
#     @app.get("/stats")
#     def stats():
#         return tracker.summary()