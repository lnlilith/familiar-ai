"""ExplorationTracker — ICL-driven novelty-aware exploration.

Inspired by EMPO² (MSR, 2026): instead of gradient updates, we inject
exploration history into the LLM's context (in-context learning) so that
the model autonomously steers toward unexplored directions.

The pan/tilt deltas from Tapo camera moves are:
  left  → pan_delta = +degrees/180   (positive x = physical LEFT)
  right → pan_delta = -degrees/180
  up    → tilt_delta = -degrees/90   (negative y = physical UP)
  down  → tilt_delta = +degrees/90
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# Directions that the camera can face
_DIRECTIONS = ("left", "right", "up", "down")
_PAN_DELTA = {"left": 1.0, "right": -1.0}
_TILT_DELTA = {"up": -1.0, "down": 1.0}


@dataclass
class ExplorationRecord:
    timestamp: str
    direction_label: str  # left | right | up | down | center
    pan_accum: float
    tilt_accum: float
    novelty: float | None = None  # filled after observation is saved


@dataclass
class MovementRecord:
    """Record of a body movement (walk).

    Duration is the raw command parameter (seconds).  Estimated distance
    and rotation are placeholders for future body-parameter learning —
    once calibrated, the agent can think in metres / degrees instead of
    raw seconds.
    """

    timestamp: str
    direction: str  # forward | backward | left | right
    duration: float  # seconds commanded
    # Future: populated by body-parameter calibration
    estimated_distance: float | None = None  # metres
    estimated_rotation: float | None = None  # degrees


class ExplorationTracker:
    """Track camera exploration history within a session.

    Enables ICL-based exploration: by injecting this context into the
    system prompt, the LLM naturally favors under-explored directions.
    In-memory only — no persistence needed (session-scoped state).
    """

    def __init__(self) -> None:
        self._records: list[ExplorationRecord] = []
        self._movements: list[MovementRecord] = []
        self._pan_accum: float = 0.0
        self._tilt_accum: float = 0.0

    def record_move(self, direction: str, degrees: int) -> None:
        """Record a camera movement. Called whenever 'look' tool is used."""
        direction = direction.lower()
        pan_delta = _PAN_DELTA.get(direction, 0.0) * (degrees / 180.0)
        tilt_delta = _TILT_DELTA.get(direction, 0.0) * (degrees / 90.0)
        self._pan_accum += pan_delta
        self._tilt_accum += tilt_delta

        label = direction if direction in _DIRECTIONS else "center"
        self._records.append(
            ExplorationRecord(
                timestamp=datetime.now().strftime("%H:%M"),
                direction_label=label,
                pan_accum=self._pan_accum,
                tilt_accum=self._tilt_accum,
            )
        )

    def record_walk(self, direction: str, duration: float) -> None:
        """Record a body movement. Called whenever 'walk' tool is used."""
        self._movements.append(
            MovementRecord(
                timestamp=datetime.now().strftime("%H:%M"),
                direction=direction.lower(),
                duration=duration,
            )
        )

    def record_novelty(self, novelty: float) -> None:
        """Fill novelty score into the most recent record after observation."""
        if not self._records:
            return
        self._records[-1].novelty = novelty

    def unvisited_hint(self) -> str:
        """Return a natural-language hint about under-explored directions."""
        if not self._records:
            return ""

        counts: dict[str, int] = {d: 0 for d in _DIRECTIONS}
        for r in self._records:
            if r.direction_label in counts:
                counts[r.direction_label] += 1

        max_count = max(counts.values())
        if max_count == 0:
            return ""

        unvisited = [d for d, c in counts.items() if c == 0]
        rare = [d for d, c in counts.items() if 0 < c <= max_count // 2]

        candidates = unvisited or rare
        if not candidates:
            return ""

        directions_str = ", ".join(candidates)
        return f"Unexplored directions: {directions_str} — consider looking there."

    def context_for_prompt(self, n: int = 5) -> str:
        """Return a compact exploration summary for LLM context injection."""
        has_looks = bool(self._records)
        has_walks = bool(self._movements)
        if not has_looks and not has_walks:
            return ""

        lines = ["[Exploration context — recent activity]"]

        if has_looks:
            recent = self._records[-n:]
            for r in recent:
                if r.novelty is None:
                    novelty_str = "?"
                elif r.novelty >= 0.7:
                    novelty_str = "HIGH"
                elif r.novelty <= 0.35:
                    novelty_str = "LOW"
                else:
                    novelty_str = "MED"
                lines.append(
                    f"  - {r.timestamp} looked {r.direction_label} (novelty: {novelty_str})"
                )

        if has_walks:
            recent_walks = self._movements[-n:]
            for m in recent_walks:
                dur_str = f"{m.duration:.1f}s" if m.duration else "brief"
                lines.append(f"  - {m.timestamp} walked {m.direction} ({dur_str})")

        hint = self.unvisited_hint()
        if hint:
            lines.append(hint)

        return "\n".join(lines)
