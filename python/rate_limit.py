"""Step 45 — per-key rate limiting, token bucket algorithm.

Yours to write. This gets imported into app.py and wired in as another
FastAPI dependency, chained right after require_api_key from step 44 —
the `caller` string that function returns is the identity this file
tracks bucket state against.
"""

import threading
import time


class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimiter:
    """Owns one TokenBucket per caller, created lazily on first request.

    TODO __init__(self, capacity: float, refill_rate: float):
    - store both (every bucket this limiter creates uses the same
      capacity/refill_rate — one limit policy for now, not per-key
      policies)
    - self._buckets: dict[str, TokenBucket] = {}
    - self._lock = threading.Lock()
      WHY A LOCK: uvicorn runs your synchronous route functions in a
      thread pool (you saw "run_in_threadpool" in tracebacks back in
      step 43) — two requests from the SAME key can genuinely execute
      concurrently on different threads. allow() does a read-modify-write
      on self.tokens; without a lock, two threads could both read
      "1 token left" before either writes anything back, both decide to
      allow, and both proceed. That's a real race condition, not a
      theoretical one, given how uvicorn actually runs your code.

    TODO allow(self, key: str) -> bool:
    - with self._lock:
        - look up key in self._buckets; if missing, create a new
          TokenBucket(self.capacity, self.refill_rate) and store it
          (dict.setdefault does exactly this "get or create" in one line)
        - call .allow() on that bucket and return the result

    Note the lock wraps bucket lookup/creation AND the .allow() call
    together — if you only locked the dict access and not the token
    math, you'd still have the same race on the token count itself.
    """
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            if key in self._buckets:
                return self._buckets[key].allow()
            self._buckets[key] = TokenBucket(capacity=self.capacity, refill_rate=self.refill_rate)
            return self._buckets[key].allow()

    


# TODO: pick starting numbers. Something like capacity=10, refill_rate=1
# (10-request burst allowed, sustained rate of 60/minute after that) is a
# reasonable starting guess, not a measured one — step 51 (load testing)
# is where you'll find out if it's actually right for this service, and
# you should be ready to say why you picked whatever you land on here.
limiter = RateLimiter(capacity=10, refill_rate=1)
