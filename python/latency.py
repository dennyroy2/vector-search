"""Step 46 — latency instrumentation and percentile computation.

Yours to write. Two separate jobs:
  1. Record how long each request actually took.
  2. Turn those recorded numbers into p50/p99 — the numbers a latency
     budget gets judged against.

Wires into app.py two ways: HTTP middleware that times every request
(see the note at the bottom of this file for the decision that's yours
to make there), and a small /stats route that exposes the numbers.
"""

import numpy as np
from collections import deque


class LatencyTracker:
    def __init__(self, maxlen: int = 1000):
        self.durations = deque(maxlen=maxlen)

    def record(self, duration_seconds):
        self.durations.append(duration_seconds)

    def percentile(self, p):
        if len(self.durations) == 0:
            return None
        return float(np.percentile(list(self.durations), p))

    def summary(self):
        return {"count": len(self.durations), "p50": self.percentile(50), "p99": self.percentile(99)}


tracker = LatencyTracker()
