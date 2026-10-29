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


def starmap(
    function: Callable[..., Result], tasks: Iterable[tuple], *, workers: int = DEFAULT_WORKERS
) -> Iterator[Result]:
    """Yield ordered results from an importable function applied to argument tuples.

    Use spawned workers, or run locally with ``workers=1``. Worker functions must
    be importable from modules. Exhaust or close the iterator to release the pool;
    worker exceptions propagate.
    """
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
