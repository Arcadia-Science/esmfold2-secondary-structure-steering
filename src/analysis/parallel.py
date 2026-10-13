"""Ordered CPU work in spawned processes, callable from notebooks and scripts."""

from __future__ import annotations
import itertools
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
Result = TypeVar("Result")


def validate_workers(workers: int) -> None:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")


def starmap(
    function: Callable[..., Result], tasks: Iterable[tuple], *, workers: int = DEFAULT_WORKERS
) -> Iterator[Result]:
    """Apply an importable function to argument tuples, preserving input order.

    Worker functions must live in modules, not notebook cells or closures.
    Explicit spawn avoids forking a running Jupyter kernel or native libraries.
    ``workers=1`` runs locally. Pools are closed before returning to the caller;
    exceptions propagate rather than triggering a silent serial fallback.
    """
    validate_workers(workers)
    if workers == 1:
        yield from itertools.starmap(function, tasks)
        return
    tasks = list(tasks)
    if not tasks:
        return
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        yield from pool.map(function, *zip(*tasks, strict=True))
