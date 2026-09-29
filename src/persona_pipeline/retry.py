"""Retry with exponential backoff and jitter, for flaky model APIs."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger("persona_pipeline")


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    is_retryable: Callable[[Exception], bool] = lambda exc: True,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "call",
) -> T:
    """Call `fn` until it succeeds, retrying only errors that `is_retryable` accepts.

    Waits base_delay * 2^n seconds (capped at max_delay, plus up to 25% jitter)
    between attempts. `sleep` is injectable so tests run instantly.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)
            log.warning(
                "  %s failed (%s), retry %d/%d in %.1fs",
                label,
                type(exc).__name__,
                attempt,
                attempts - 1,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
