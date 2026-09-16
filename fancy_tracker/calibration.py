"""Calibration storage and the gaze classifier.

The camera sits off to one side rather than straight ahead, so there is no clean
geometric mapping from head pose to screen that could be written down in
advance. Calibration measures one instead.

Two classifiers live here. The centroid one asks "which monitor's average does
this pose most resemble", which is all a pooled profile can answer. The
geometric one fits each monitor's own plane through its five dots and asks
"which monitor best explains this pose, and whereabouts on it" - that can tell
the edge of a panel apart from somewhere off it entirely, which a distance to an
average cannot. The geometric one is used whenever the calibration carries
per-dot data; older files fall back to centroids rather than being rejected.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import DisplayModel
from .pose import FEATURE_NAMES, N_FEATURES

SCHEMA_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)


def state_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "fancy-tracker"


def calibration_path() -> Path:
    return state_dir() / "calibration.json"


@dataclass
class TargetProfile:
    """One dot: where it was on the panel, and the pose that looked at it."""

    name: str
    x: float
    y: float
    mean: list[float]
    std: list[float]
    samples: int


@dataclass
class DisplayProfile:
    display_id: int
    label: str
    mean: list[float]
    std: list[float]
    samples: int
    width: float = 0.0
    height: float = 0.0
    targets: list[TargetProfile] = field(default_factory=list)


@dataclass
class Calibration:
    version: int
    profiles: list[DisplayProfile]
    scale: list[float]  # pooled within-display spread, per feature

    @staticmethod
    def _pooled_scale(per_display: list[np.ndarray], centroids: np.ndarray) -> np.ndarray:
        # Floor it so a feature that happened to be perfectly still during
        # calibration cannot dominate the distance metric.
        pooled = np.sqrt(np.mean(np.asarray(per_display), axis=0))
        between = centroids.std(axis=0)
        return np.maximum(pooled, np.maximum(between * 0.05, 1e-3))

    @classmethod
    def build(cls, samples_by_display: dict[int, tuple[str, np.ndarray]]) -> Calibration:
        """Pooled profiles only - no per-dot geometry."""
        profiles, variances = [], []
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

        centroids = np.asarray([p.mean for p in profiles])
        scale = cls._pooled_scale(variances, centroids)
        return cls(version=SCHEMA_VERSION, profiles=profiles, scale=scale.tolist())

    @classmethod
    def build_with_targets(cls, collected: dict) -> Calibration:
        """Full calibration, keeping each dot so the layout can be fitted.

        `collected` maps display id to
        (label, width, height, [(name, x, y, samples), ...]).
        """
        profiles, variances = [], []
        for display_id, (label, width, height, targets) in collected.items():
            rows = [s for _n, _x, _y, samples in targets for s in samples]
            arr = np.asarray(rows, dtype=np.float64)
            profiles.append(
                DisplayProfile(
                    display_id=display_id,
                    label=label,
                    mean=np.median(arr, axis=0).tolist(),
                    std=arr.std(axis=0).tolist(),
                    samples=int(arr.shape[0]),
                    width=float(width),
                    height=float(height),
                    targets=[
                        TargetProfile(
                            name=name,
                            x=float(x),
                            y=float(y),
                            mean=np.median(np.asarray(samples), axis=0).tolist(),
                            std=np.asarray(samples).std(axis=0).tolist(),
                            samples=len(samples),
                        )
                        for name, x, y, samples in targets
                    ],
                )
            )
            variances.append(arr.var(axis=0))

        centroids = np.asarray([p.mean for p in profiles])
        scale = cls._pooled_scale(variances, centroids)
        return cls(version=SCHEMA_VERSION, profiles=profiles, scale=scale.tolist())

    def models(self) -> list[DisplayModel]:
        """A fitted plane per monitor, for those that carry per-dot data."""
        out = []
        for p in self.profiles:
            if not p.targets or p.width <= 0 or p.height <= 0:
                continue
            model = DisplayModel.fit(
                p.display_id,
                p.label,
                p.width,
                p.height,
                [(t.x, t.y, np.asarray(t.mean)) for t in p.targets],
            )
            if model is not None:
                out.append(model)
        return out

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
        version = data.get("version")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Calibration at {path} is version {version}, expected one of "
                f"{SUPPORTED_VERSIONS}. Re-run `fancy-tracker calibrate`."
            )
        profiles = []
        for raw in data["profiles"]:
            targets = [TargetProfile(**t) for t in raw.get("targets", [])]
            profiles.append(
                DisplayProfile(
                    display_id=raw["display_id"],
                    label=raw["label"],
                    mean=raw["mean"],
                    std=raw["std"],
                    samples=raw["samples"],
                    width=raw.get("width", 0.0),
                    height=raw.get("height", 0.0),
                    targets=targets,
                )
            )
        return cls(version=version, profiles=profiles, scale=data["scale"])


class Classifier:
    """Picks the monitor a live pose belongs to.

    Geometric when the calibration carries per-dot data, centroid otherwise.
    Both report distances on the same footing - multiples of the measured noise
    - so --margin and --stickiness keep their meaning either way.
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

        # Scoring happens in units of the measured noise, while the models stay
        # in degrees so the layout report means something to a person.
        self._weights = 1.0 / np.maximum(pooled, 1e-6)
        models = calibration.models()
        self._models = {m.display_id: m for m in models} if len(models) == len(self._ids) else {}

    @property
    def geometric(self) -> bool:
        return bool(self._models)

    @property
    def display_ids(self) -> list[int]:
        return list(self._ids)

    def label(self, display_id: int) -> str:
        return self._labels.get(display_id, str(display_id))

    def model_list(self) -> list[DisplayModel]:
        return list(self._models.values())

    def distances(self, features: np.ndarray) -> dict[int, float]:
        if self._models:
            return {
                did: model.score(features, self._weights)[0] for did, model in self._models.items()
            }
        d = np.linalg.norm((self._centroids - features) / self._scales, axis=1)
        return {display_id: float(dist) for display_id, dist in zip(self._ids, d)}

    def locate(self, features: np.ndarray) -> tuple[int, float, float, float] | None:
        """Which monitor, whereabouts on it, and how far outside it - if geometric."""
        if not self._models:
            return None
        best_id, best = None, None
        for did, model in self._models.items():
            score, x, y, outside = model.score(features, self._weights)
            if best is None or score < best[0]:
                best_id, best = did, (score, x, y, outside)
        return (best_id, best[1], best[2], best[3])

    def classify(self, features: np.ndarray) -> tuple[int, float, float]:
        """Return (display_id, distance, margin over the runner-up)."""
        d = self.distances(features)
        ranked = sorted(d.items(), key=lambda kv: kv[1])
        best_id, best = ranked[0]
        margin = (ranked[1][1] - best) if len(ranked) > 1 else float("inf")
        return best_id, best, float(margin)
