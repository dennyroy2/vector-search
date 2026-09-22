"""Step 51 — load test the live, deployed service. Finds where it breaks,
and why: the rate limiter doing its job by design, or genuine server
capacity giving out (not by design) — those are different findings and
this script is built to tell them apart.

Yours to have me write — "load test scripts" is explicitly on the
"write it for me" list in CLAUDE.md.

Uses 10 dedicated load-test API keys (separate from your real key and
the frontend's public-demo key) specifically so this test doesn't
contend with, or get mistaken for, real traffic on those.
"""

import asyncio
import statistics
import time

import httpx

BASE_URL = "http://18.119.185.198:8000"

LOAD_TEST_KEYS = [
    "JP4INFI3YU_wDEGjky9SY9rS7DxBeKFt",
    "ImyJhfopBdQn0bw7EK7nATpvOSb1KsB4",
    "yTF4kDVc_sueZ4ZQW0sLdqJdtabGm4BM",
    "WX_Enr9chwfczj98CFLaCYmMLGyO9k3_",
    "pSkf3CkQyKjmzBO_M7hnS2mLkrBzN7Y7",
    "dtjrQvWvvIzbFKXweXPVXSSBNCn8gDLJ",
    "cyDDb-4mng2oO2tZqNOGJ1HrdO6Pl_am",
    "rgbb53Wror3lavH2RV_FfYeItGZB8Q9a",
    "YbLIuTeZokZgG7DRYjNK7zgVUkTpqnsd",
    "sRyLSO7Q81d8qxYOOc6_4EQaaLlSJ36j",
]

QUERIES = [
    "Markov Chain", "GPU Programming", "Dynamic Programming",
    "Reinforcement Learning", "Graph Neural Networks",
]


async def one_request(client: httpx.AsyncClient, key: str, query: str):
    """Fire one search request, timed. Returns (status_code, seconds).
    status_code is None specifically for a connection-level failure
    (timeout, refused, DNS) — distinct from a real HTTP error code,
    since those mean two very different things about what broke.
    """
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}/search",
            headers={"X-API-Key": key},
            json={"query": query, "k": 5},
            timeout=15.0,
        )
        return resp.status_code, time.perf_counter() - start
    except httpx.RequestError:
        return None, time.perf_counter() - start


def summarize(label: str, results: list, wall_seconds: float):
    statuses = [s for s, _ in results]
    ok = statuses.count(200)
    rate_limited = statuses.count(429)
    conn_failed = statuses.count(None)
    other = len(statuses) - ok - rate_limited - conn_failed

    ok_durations = sorted(d for s, d in results if s == 200)
    p50 = statistics.median(ok_durations) if ok_durations else None
    p99 = (ok_durations[int(len(ok_durations) * 0.99)]
           if len(ok_durations) >= 2 else (ok_durations[0] if ok_durations else None))

    print(f"\n=== {label} ===")
    print(f"  requests: {len(results)}  |  wall time: {wall_seconds:.2f}s  "
          f"|  throughput: {len(results)/wall_seconds:.1f} req/s")
    print(f"  200 OK: {ok}   429 rate-limited: {rate_limited}   "
          f"connection failures: {conn_failed}   other: {other}")
    if p50 is not None:
        print(f"  p50 (successful only): {p50*1000:.1f} ms   "
              f"p99: {p99*1000:.1f} ms")


async def concurrency_stage(label: str, n_keys: int, requests_per_key: int):
    """Fire n_keys * requests_per_key requests, ALL AT ONCE, spread across
    n_keys distinct callers — simulates that many different real users
    searching simultaneously, not one client looping.
    """
    keys = LOAD_TEST_KEYS[:n_keys]
    async with httpx.AsyncClient() as client:
        tasks = [
            one_request(client, key, QUERIES[i % len(QUERIES)])
            for key in keys
            for i in range(requests_per_key)
        ]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    summarize(label, results, wall)


async def single_key_hammer(n_requests: int = 30):
    """One key, sequential rapid-fire requests — isolates the rate
    limiter's own behavior from server capacity entirely, since this
    never exceeds what a single client could do alone.
    """
    key = LOAD_TEST_KEYS[0]
    results = []
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        for i in range(n_requests):
            results.append(await one_request(client, key, QUERIES[i % len(QUERIES)]))
        wall = time.perf_counter() - start
    summarize(f"single key, {n_requests} rapid sequential requests", results, wall)


REFILL_WAIT = 12  # capacity=10, refill_rate=1/sec on the live server — 12s
                   # guarantees every key's bucket is back to full before the
                   # next stage, so each stage measures a clean baseline
                   # instead of buckets already drained by the previous one.


async def main():
    # Stage 1: sanity baseline — 1 key, exactly at its own burst capacity (10).
    await concurrency_stage("baseline: 1 key x 10 requests", n_keys=1, requests_per_key=10)
    print(f"\n(waiting {REFILL_WAIT}s for buckets to fully refill...)")
    await asyncio.sleep(REFILL_WAIT)

    # Stage 2: 10 distinct callers, each exactly at their own capacity —
    # 100 concurrent requests total, none of which SHOULD be rejected by
    # the rate limiter. Whatever breaks here is server capacity, not policy.
    await concurrency_stage("10 keys x 10 requests (100 concurrent, at limit)",
                             n_keys=10, requests_per_key=10)
    print(f"\n(waiting {REFILL_WAIT}s for buckets to fully refill...)")
    await asyncio.sleep(REFILL_WAIT)

    # Stage 3: deliberately over each key's capacity — expect a clean split
    # between 200s (up to each key's 10-token bucket) and 429s (the rest).
    await concurrency_stage("10 keys x 20 requests (200 concurrent, over limit)",
                             n_keys=10, requests_per_key=20)
    print(f"\n(waiting {REFILL_WAIT}s for buckets to fully refill...)")
    await asyncio.sleep(REFILL_WAIT)

    # Stage 4: the rate limiter in isolation, single caller, fresh bucket.
    await single_key_hammer(30)


if __name__ == "__main__":
    asyncio.run(main())
