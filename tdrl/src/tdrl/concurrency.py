"""Two dialogue workers, isolated requests and shared-allocation accounting."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .provenance import atomic_json


class ActiveShare:
    """Allocate elapsed busy time equally among active dialogues, without double counting.

    This is an explicit accounting convention, not per-kernel GPU attribution.
    Cold load and work outside dialogues are reported separately by the stage.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.active, self.shares = set(), {}
        self.last = clock()

    def _tick(self):
        now = self.clock()
        if self.active:
            for key in self.active:
                self.shares[key] += (now - self.last) / len(self.active)
        self.last = now

    def enter(self, key):
        with self.lock:
            self._tick()
            if key in self.active:
                raise ValueError("duplicate active dialogue")
            self.active.add(key)
            self.shares[key] = 0.0

    def leave(self, key):
        with self.lock:
            self._tick()
            self.active.remove(key)
            return self.shares[key]


def run_method_pair(callback, directory):
    """Run exactly PPDPP and natural-budget TDRL Full; return canonical order."""
    meter, barrier = ActiveShare(), threading.Barrier(2)
    timing = directory / "pair-timing.json"
    started = time.monotonic()

    def worker(method):
        barrier.wait()
        meter.enter(method)
        try:
            result, path = callback(method)
        finally:
            share = meter.leave(method)
        if result.get("compute_accounting") != "equal_active_dialogue_share_v1":
            result = {
                **result,
                "h100_seconds": share,
                "compute_accounting": "equal_active_dialogue_share_v1",
                "cold_load_accounting": "separate stage/server metadata",
            }
            atomic_json(path / "dialogue.json", result)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, method) for method in ("PPDPP", "TDRL")]
        # Drain both workers even when one fails. Retain journals and failed outcomes.
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
    if not timing.exists():
        atomic_json(
            timing,
            {
                "schema_version": 1,
                "methods": ["PPDPP", "TDRL"],
                "wall_s": time.monotonic() - started,
                "h100_seconds": sum(r["h100_seconds"] for r in results),
                "accounting": "equal_active_dialogue_share_v1",
            },
        )
    return results
