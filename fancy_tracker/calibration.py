"""Calibration storage and the nearest-centroid gaze classifier.

The camera sits off to one side rather than straight ahead, so there is no
clean geometric mapping from head pose to screen. Calibration sidesteps that:
it records what the feature vector actually looks like while looking at each
display, and classification is then just "which of those does this resemble".
Any fixed camera offset is absorbed into the centroids.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .pose import FEATURE_NAMES, N_FEATURES

SCHEMA_VERSION = 1


def state_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "fancy-tracker"


def calibration_path() -> Path:
    return state_dir() / "calibration.json"


@dataclass
class DisplayProfile:
    display_id: int
    label: str
    mean: list[float]
    std: list[float]
    samples: int


@dataclass
class Calibration:
    version: int
    profiles: list[DisplayProfile]
    scale: list[float]  # pooled within-display spread, per feature

    @classmethod
    def build(cls, samples_by_display: dict[int, tuple[str, np.ndarray]]) -> Calibration:
        profiles = []
        variances = []
        for display_id, (label, samples) in samples_by_display.items():
            arr = np.asarray(samples, dtype=np.float64)
            profiles.append(
                DisplayProfile(
                    display_id=display_id,
                    label=label,
                    mean=np.median(arr, axis=0).tolist(),
                    std=arr.std(axis=0).tolist(),
                    samples=int(arr.shape[0]),
                )
            )
            variances.append(arr.var(axis=0))

        # Pooled within-display spread. Floor it so a feature that happened to be
        # perfectly still during calibration cannot dominate the distance metric.
        pooled = np.sqrt(np.mean(np.asarray(variances), axis=0))
        centroids = np.asarray([p.mean for p in profiles])
        between = centroids.std(axis=0)
        scale = np.maximum(pooled, np.maximum(between * 0.05, 1e-3))

        return cls(version=SCHEMA_VERSION, profiles=profiles, scale=scale.tolist())

    def save(self, path: Path | None = None) -> Path:
        path = path or calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "features": list(FEATURE_NAMES),
            "scale": self.scale,
            "profiles": [asdict(p) for p in self.profiles],
        }
        # Written through a temporary file so a reader never sees half a
        # document. The running tracker polls this path and would otherwise
        # occasionally catch it mid-write.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> Calibration:
        path = path or calibration_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"No calibration at {path}. Run `fancy-tracker calibrate` first."
            )
        data = json.loads(path.read_text())
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Calibration at {path} is version {data.get('version')}, "
                f"expected {SCHEMA_VERSION}. Re-run `fancy-tracker calibrate`."
            )
        profiles = [DisplayProfile(**p) for p in data["profiles"]]
        return cls(version=data["version"], profiles=profiles, scale=data["scale"])


class Classifier:
    """Nearest centroid, each display judged against its own spread.

    A single shared scale punishes a display whose samples are legitimately
    spread out. A tall portrait monitor is the clearest case: looking from its
    top edge to its bottom edge sweeps a wide pitch range, so honest samples sit
    far from its centroid, while a compact neighbour keeps a tight one and wins
    the comparison. Measured on a real four-monitor setup, the portrait monitor's
    median margin was 0.90 against 2.70 for a landscape monitor of the same
    pixel count.

    Each display therefore gets its own scale, shrunk toward the pooled one by a
    geometric mean. Full per-display scaling would over-correct: with no
    normalising term, the widest display would start winning everything.
    """

    # How far a display's own spread may fall below the pooled spread before it
    # is floored. Without this, a display that happened to be sampled very still
    # gets an impossibly tight metric and can never win.
    SCALE_FLOOR = 0.35

    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        self._ids = [p.display_id for p in calibration.profiles]
        self._labels = {p.display_id: p.label for p in calibration.profiles}
        self._centroids = np.asarray([p.mean for p in calibration.profiles], dtype=np.float64)
        if self._centroids.shape[1] != N_FEATURES:
            raise ValueError("Calibration feature width does not match this build")

        pooled = np.asarray(calibration.scale, dtype=np.float64)
        own = np.asarray([p.std for p in calibration.profiles], dtype=np.float64)
        own = np.maximum(own, pooled * self.SCALE_FLOOR)
        self._scales = np.sqrt(own * pooled)

    @property
    def display_ids(self) -> list[int]:
        return list(self._ids)

    def label(self, display_id: int) -> str:
        return self._labels.get(display_id, str(display_id))

    def distances(self, features: np.ndarray) -> dict[int, float]:
        """Scaled distance from the live pose to every calibrated display."""
        d = np.linalg.norm((self._centroids - features) / self._scales, axis=1)
        return {display_id: float(dist) for display_id, dist in zip(self._ids, d)}

    def classify(self, features: np.ndarray) -> tuple[int, float, float]:
        """Return (display_id, distance, margin over the runner-up)."""
        d = self.distances(features)
        ranked = sorted(d.items(), key=lambda kv: kv[1])
        best_id, best = ranked[0]
        margin = (ranked[1][1] - best) if len(ranked) > 1 else float("inf")
        return best_id, best, float(margin)
