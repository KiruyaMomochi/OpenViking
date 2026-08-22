# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Hotness scoring for cold/hot memory lifecycle management (#296).

Provides a pure function to compute a 0.0–1.0 hotness score based on
access frequency (active_count) and recency (updated_at). The memory
health statistics API uses this score to classify cold, warm, and hot memories.
"""

import math
from datetime import datetime, timezone
from typing import Optional

# Default half-life in days for the exponential time-decay component.
DEFAULT_HALF_LIFE_DAYS: float = 7.0


def hotness_score(
    active_count: int,
    updated_at: Optional[datetime],
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Compute a 0.0–1.0 hotness score.

    Formula::

        score = sigmoid(log1p(active_count)) * time_decay(updated_at)

    * **sigmoid** maps ``log1p(active_count)`` into (0, 1).
    * **time_decay** is an exponential decay with configurable half-life;
      returns 0.0 when *updated_at* is ``None``.

    Args:
        active_count: Number of times this context was retrieved/accessed.
        updated_at: Last update / access timestamp (preferably UTC).
        now: Current time override (useful for deterministic tests).
        half_life_days: Half-life for the recency decay, in days.

    Returns:
        A float in [0.0, 1.0].
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # --- frequency component ---
    freq = 1.0 / (1.0 + math.exp(-math.log1p(active_count)))

    # --- recency component ---
    if updated_at is None:
        return 0.0

    # Normalise to aware UTC so subtraction always works.
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = max((now - updated_at).total_seconds() / 86400.0, 0.0)
    decay_rate = math.log(2) / half_life_days
    recency = math.exp(-decay_rate * age_days)

    return freq * recency


# Default half-life in days for episodic (one-off event) memory decay.
# Deliberately longer than the hotness half-life: hotness models access
# frequency, this models how long a single past event stays relevant.
DEFAULT_EPISODIC_HALF_LIFE_DAYS: float = 30.0


def is_episodic_uri(uri: str, uri_segment: str = "/memories/events/") -> bool:
    """Return True when *uri* points at an episodic (one-off event) memory.

    Retrieval output fields carry no structured memory type -- the vector
    index exposes only ``uri``, ``level``, ``context_type``, ``abstract``,
    ``active_count``, ``created_at`` and ``updated_at``, and ``context_type``
    is coarse (memories / resources / skills).  Classification is therefore
    path-based, with the segment kept configurable so a future layout change
    is a config edit rather than a code change.

    Args:
        uri: Context URI, e.g. ``viking://user/x/memories/events/2026/05/31/foo.md``.
        uri_segment: Path segment marking episodic memories.

    Returns:
        True if the URI is episodic, False for stable memory types
        (preferences, entities, experiences, trajectories) and non-memories.
    """
    if not uri or not uri_segment:
        return False
    return uri_segment in uri


def episodic_decay_factor(
    recorded_at: Optional[datetime],
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_EPISODIC_HALF_LIFE_DAYS,
) -> float:
    """Exponential age-decay factor in (0.0, 1.0] for a one-off event memory.

    Unlike :func:`hotness_score` this has no frequency term: a past event does
    not become more current by being retrieved often.  Returns ``1.0`` (no
    decay) when *recorded_at* is unknown, so a missing timestamp can never
    silently suppress a memory.

    Args:
        recorded_at: When the event was recorded (``created_at`` preferred over
            ``updated_at``; the latter also moves on re-indexing).
        now: Current time override, for deterministic tests.
        half_life_days: Age at which the factor reaches 0.5. Must be > 0.

    Returns:
        A float in (0.0, 1.0]; 1.0 means "do not decay".
    """
    if recorded_at is None:
        return 1.0
    if half_life_days <= 0:
        return 1.0

    if now is None:
        now = datetime.now(timezone.utc)

    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = max((now - recorded_at).total_seconds() / 86400.0, 0.0)
    decay_rate = math.log(2) / half_life_days
    return math.exp(-decay_rate * age_days)
