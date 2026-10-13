"""Tiny process-pool map with shared read-only context (works with fork and spawn)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

CTX: dict = {}


def _init(ctx):
    CTX.clear()
    CTX.update(ctx)


def pmap(fn, items, ctx, workers=None):
    """map(fn, items) where fn reads shared data from parallel.CTX."""
    items = list(items)
    workers = workers or os.cpu_count() or 1
    if workers <= 1 or len(items) < 4:
        _init(ctx)
        return [fn(i) for i in items]
    chunk = max(1, len(items) // (workers * 6))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
        return list(ex.map(fn, items, chunksize=chunk))
