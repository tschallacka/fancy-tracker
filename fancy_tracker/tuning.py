"""Whether the settings let you actually reach every monitor.

Stickiness is subtracted from the monitor you are already on, so a move happens
only when the destination beats the origin by more than stickiness plus margin.
Monitor pairs differ enormously in how far apart they are - on one real desk the
easiest move had thirty times the room of the hardest - so a value that steadies
a loose pair can silently make a tight one unreachable. That failure is invisible
from the inside: nothing errors, the cursor simply does not follow, and only a
particular pair is affected.

This measures the room each move actually has, using the calibration's own dots
as the places you might be looking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration, Classifier


@dataclass
class Transition:
    src_label: str
    dst_label: str
    worst_dot: str
    lead: float  # how much the destination beats the origin, at its weakest dot
    margin: float

    @property
    def usable_stickiness(self) -> float:
        """The most stickiness that still leaves this move possible."""
        return self.lead - self.margin

    def blocked_at(self, stickiness: float) -> bool:
        return stickiness >= self.usable_stickiness


def transitions(calibration: Calibration, margin: float = 0.35) -> list[Transition]:
    classifier = Classifier(calibration)
    with_dots = [p for p in calibration.profiles if p.targets]
    out: list[Transition] = []

    for src in with_dots:
        for dst in with_dots:
            if src.display_id == dst.display_id:
                continue
            leads = []
            for t in dst.targets:
                d = classifier.distances(np.asarray(t.mean))
                leads.append((d[src.display_id] - d[dst.display_id], t.name))
            lead, dot = min(leads)
            out.append(Transition(src.label, dst.label, dot, float(lead), margin))
    return out


def recommend(moves: list[Transition]) -> float:
    """A stickiness that leaves every achievable move room to happen.

    Pairs with no room even at zero are excluded. Those two monitors sit at
    nearly the same angle where they face each other and no setting rescues
    them; letting one drag the recommendation to zero would give up all the
    steadiness the other pairs can afford.
    """
    achievable = [m.usable_stickiness for m in moves if m.usable_stickiness > 0.0]
    if not achievable:
        return 0.0
    return max(0.0, round(min(achievable) * 0.6, 2))


def describe(moves: list[Transition], stickiness: float) -> str:
    if not moves:
        return "No per-dot calibration data; run `fancy-tracker calibrate` to check tuning."

    lines = ["How much room each move has, measured at the destination's own dots.", ""]
    lines.append(f"  {'from -> to':44s} {'weakest spot':14s} {'lead':>6} {'max sticky':>11}")
    for m in sorted(moves, key=lambda m: m.usable_stickiness):
        flag = "  <-- BLOCKED by current setting" if m.blocked_at(stickiness) else ""
        lines.append(
            f"  {m.src_label[:20] + ' -> ' + m.dst_label[:20]:44s} "
            f"{m.worst_dot:14s} {m.lead:>6.2f} {m.usable_stickiness:>11.2f}{flag}"
        )

    blocked = [m for m in moves if m.blocked_at(stickiness)]
    lines.append("")
    if blocked:
        lines.append(f"  --stickiness {stickiness} blocks {len(blocked)} move(s).")
        lines.append("  Those moves will not happen: the cursor simply stays where it is,")
        lines.append("  and going via a third monitor is the only way across.")
        lines.append(f"  Try --stickiness {recommend(moves)}.")
    else:
        lines.append(f"  --stickiness {stickiness} leaves every move usable.")

    tight = [m for m in moves if m.lead < 0.5]
    if tight:
        lines.append("")
        lines.append("  These pairs are barely distinguishable where they face each other,")
        lines.append("  whatever the settings - they sit at nearly the same angle from here:")
        for m in tight:
            lines.append(
                f"    {m.src_label[:22]:24s} -> {m.dst_label[:22]:24s} "
                f"lead {m.lead:.2f} at its {m.worst_dot}"
            )
    return "\n".join(lines)
