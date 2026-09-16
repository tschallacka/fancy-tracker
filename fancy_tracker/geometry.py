"""Where the monitors actually are, inferred from where your head pointed.

macOS's display arrangement is a bookkeeping fiction: it butts panels together
in one pixel plane whatever their real position, so two monitors with a hand's
width of desk between them share an edge as far as CGDisplayBounds is concerned.
Classifying against that pretend geometry costs accuracy exactly at the seams.

Calibration measures the truth instead. Five dots per monitor, each with a known
position on that panel and a measured head pose, are enough to fit the monitor's
own gradient - how much the head turns per pixel travelled - and from there to
place its edges in angular space. Comparing one monitor's far edge with its
neighbour's near edge then shows whether they really adjoin or whether there is
a gap, in degrees, regardless of what the arrangement claims.

The same fit removes the axis coupling that five-point solvePnP suffers from.
Each monitor's map is a full linear transform from panel position to the whole
feature vector, so "moving down this column also reads as a little yaw" is
represented rather than fought.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# A monitor needs at least this many usable dots before a plane can be fitted
# through them. Three is the bare minimum for x, y and a constant.
MIN_POINTS = 4

# Degrees of angular separation between two monitors' facing edges above which
# they are treated as physically apart rather than adjoining.
GAP_DEGREES = 2.0


@dataclass
class DisplayModel:
    """Linear map from a position on one panel to the head pose that looks at it."""

    display_id: int
    label: str
    width: float
    height: float
    coeffs: np.ndarray  # (n_features, 3): features = coeffs @ [x, y, 1]
    residual: float  # rms fit error, in feature units
    n_points: int

    @classmethod
    def fit(
        cls,
        display_id: int,
        label: str,
        width: float,
        height: float,
        points: list[tuple[float, float, np.ndarray]],
    ) -> DisplayModel | None:
        """Least-squares plane through (x, y) -> features. None if underdetermined."""
        if len(points) < MIN_POINTS:
            return None

        design = np.array([[x, y, 1.0] for x, y, _f in points], dtype=np.float64)
        values = np.array([f for _x, _y, f in points], dtype=np.float64)
        if np.linalg.matrix_rank(design) < 3:
            return None  # all the dots fell on one line; no plane to fit

        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        predicted = design @ coeffs
        residual = float(np.sqrt(np.mean((predicted - values) ** 2)))
        return cls(
            display_id=display_id,
            label=label,
            width=float(width),
            height=float(height),
            coeffs=coeffs.T,
            residual=residual,
            n_points=len(points),
        )

    def features_at(self, x: float, y: float) -> np.ndarray:
        """The head pose that looks at this spot on this panel."""
        return self.coeffs @ np.array([x, y, 1.0])

    @property
    def gradient(self) -> np.ndarray:
        """Feature change per pixel, as a (n_features, 2) matrix of dx and dy."""
        return self.coeffs[:, :2]

    @property
    def degrees_per_pixel(self) -> tuple[float, float]:
        """Yaw per horizontal pixel, pitch per vertical pixel."""
        return (abs(float(self.coeffs[0, 0])), abs(float(self.coeffs[1, 1])))

    @property
    def yaw_pitch_coupling(self) -> tuple[float, float]:
        """The off-axis terms: yaw per vertical pixel, pitch per horizontal pixel.

        Ideally zero. What is left is the axis coupling in the pose estimate, and
        because it is part of the fit the classifier is no longer misled by it.
        """
        return (float(self.coeffs[0, 1]), float(self.coeffs[1, 0]))

    def locate(
        self, features: np.ndarray, weights: np.ndarray | None = None
    ) -> tuple[float, float, float]:
        """Best (x, y) on this panel for a live pose, plus how well it fits.

        The residual is what says "this pose is not explained by this monitor at
        all", which a nearest-centroid distance cannot distinguish from "you are
        looking at its edge".

        `weights` puts the residual in units of measured noise rather than raw
        degrees, so a pose off by one noisy feature is not judged the same as one
        off by a steady feature. The coefficients stay unweighted, so the layout
        report keeps its degrees.
        """
        w = np.ones_like(features) if weights is None else weights
        basis = self.coeffs[:, :2] * w[:, None]
        offset = self.coeffs[:, 2] * w
        target = features * w
        solution, *_ = np.linalg.lstsq(basis, target - offset, rcond=None)
        residual = float(np.linalg.norm(basis @ solution + offset - target))
        return (float(solution[0]), float(solution[1]), residual)

    def outside_by(self, x: float, y: float, weights: np.ndarray | None = None) -> float:
        """How far a point lies beyond this panel's edges, in the same units."""
        dx = max(0.0, -x, x - self.width)
        dy = max(0.0, -y, y - self.height)
        if dx == 0.0 and dy == 0.0:
            return 0.0
        w = np.ones(self.coeffs.shape[0]) if weights is None else weights
        gx = float(np.linalg.norm(self.coeffs[:, 0] * w))
        gy = float(np.linalg.norm(self.coeffs[:, 1] * w))
        return float(np.hypot(dx * gx, dy * gy))

    def score(
        self, features: np.ndarray, weights: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """(score, x, y, outside) - lower score is a better explanation."""
        x, y, residual = self.locate(features, weights)
        outside = self.outside_by(x, y, weights)
        return (residual + outside, x, y, outside)

    def edges(self) -> dict[str, float]:
        """Angular extent of this panel: yaw of its sides, pitch of top and bottom."""
        corners = [
            self.features_at(0.0, 0.0),
            self.features_at(self.width, 0.0),
            self.features_at(0.0, self.height),
            self.features_at(self.width, self.height),
        ]
        yaws = [c[0] for c in corners]
        pitches = [c[1] for c in corners]
        return {
            "yaw_min": min(yaws),
            "yaw_max": max(yaws),
            # Pitch is positive when the chin drops, so the TOP edge of a panel
            # is its smallest pitch, not its largest.
            "pitch_top": min(pitches),
            "pitch_bottom": max(pitches),
        }


@dataclass
class Neighbours:
    left: DisplayModel
    right: DisplayModel
    gap_degrees: float
    equivalent_pixels: float

    @property
    def adjoining(self) -> bool:
        return self.gap_degrees < GAP_DEGREES


def analyse(models: list[DisplayModel]) -> dict:
    """Angular layout: where each panel sits, and what lies between them."""
    ordered = sorted(models, key=lambda m: m.edges()["yaw_min"])

    gaps: list[Neighbours] = []
    for left, right in itertools.pairwise(ordered):
        gap = right.edges()["yaw_min"] - left.edges()["yaw_max"]
        # Express the gap in the neighbouring panels' own terms, since degrees
        # alone mean little without knowing how big a panel looks from here.
        per_pixel = np.mean([left.degrees_per_pixel[0], right.degrees_per_pixel[0]])
        equivalent = float(gap / per_pixel) if per_pixel > 1e-9 else 0.0
        gaps.append(Neighbours(left, right, float(gap), equivalent))

    tops = {m.display_id: m.edges()["pitch_top"] for m in ordered}
    spread = (max(tops.values()) - min(tops.values())) if tops else 0.0

    return {"ordered": ordered, "gaps": gaps, "tops": tops, "top_spread": float(spread)}


def describe(models: list[DisplayModel]) -> str:
    """The layout report printed after calibration."""
    if len(models) < 2:
        return ""

    report = analyse(models)
    lines = ["\nInferred layout (from where your head actually pointed, not the arrangement):"]
    lines.append(f"  {'monitor':26s} {'yaw span':>16} {'pitch span':>16} {'fit':>7}")
    for m in report["ordered"]:
        e = m.edges()
        lines.append(
            f"  {m.label[:24]:26s} "
            f"{e['yaw_min']:+6.1f}..{e['yaw_max']:+6.1f}  "
            f"{e['pitch_top']:+6.1f}..{e['pitch_bottom']:+6.1f}  "
            f"{m.residual:6.2f}"
        )

    lines.append("\n  Between neighbours:")
    for n in report["gaps"]:
        if n.adjoining:
            verdict = "adjoining"
        else:
            verdict = f"gap, about {abs(n.equivalent_pixels):.0f}px of screen would fill it"
        lines.append(
            f"    {n.left.label[:20]:22s} -> {n.right.label[:20]:22s} "
            f"{n.gap_degrees:+5.1f} deg  {verdict}"
        )

    lines.append("\n  Top edges:")
    for m in report["ordered"]:
        lines.append(f"    {m.label[:24]:26s} {m.edges()['pitch_top']:+6.1f} deg")
    if report["top_spread"] < 3.0:
        lines.append("    -> all aligned within 3 degrees")

    lines.append("\n  Axis coupling (yaw per vertical pixel - zero would be ideal):")
    for m in report["ordered"]:
        yaw_per_y, pitch_per_x = m.yaw_pitch_coupling
        lines.append(
            f"    {m.label[:24]:26s} {yaw_per_y * 1000:+6.2f} deg/1000px vertical"
            f"   (pitch per horizontal: {pitch_per_x * 1000:+.2f})"
        )
    return "\n".join(lines)
