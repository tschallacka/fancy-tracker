"""Guided calibration.

Every display is dimmed and shows five dots - its four corners and its centre.
The dot to look at pulses yellow, then goes solid while samples are taken. The
cursor is parked on it too, as a second cue.

Each dot is judged as soon as it is sampled. A good one turns green with a
tick. One whose samples were unusable flashes red and goes back to yellow to be
taken again. One that turns out to sit too close to a dot already recorded on a
*different* display is an unclear edge: both are suspect, so the earlier one is
ringed amber and queued to be taken again once the pass is done.

All five targets feed one profile per display. That is deliberate: a display is
an area, not a point, and a profile built only from its centre would fail to
recognise a glance at its far edge.
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np

from .calibration import Calibration
from .detector import FaceDetector
from .displays import Display, active_displays, warp_cursor
from .overlay import ACTIVE, BAD, GOOD, REVISIT, CalibrationOverlay, Target
from .pose import estimate
from .tracker import Settings, open_camera

MIN_SAMPLES_PER_TARGET = 8
MIN_SAMPLES_PER_DISPLAY = 20

# Degrees of yaw or pitch wobble within one target's sampling window before the
# samples are treated as a smeared aim rather than a steady look.
MAX_JITTER_DEG = 6.0

# Scaled distance below which two targets on different displays are too alike to
# tell apart. Roughly "within the noise of a single target".
CONFLICT_DISTANCE = 2.5

MAX_ATTEMPTS = 3
RED_FLASH_SECONDS = 0.9
RED_FLASH_HZ = 5.0

# Verdicts.
OK = "ok"
INSUFFICIENT = "insufficient"
UNSTEADY = "unsteady"


def target_to_global(display: Display, target: Target) -> tuple[float, float]:
    """Dot position in macOS global cursor coordinates.

    Targets are in the display's own space with y up from the bottom, which is
    AppKit's convention. The cursor lives in CoreGraphics space, y down from the
    top of the main display, so the vertical axis has to be flipped.
    """
    return (display.x + target.x, display.y + (display.height - target.y))


def assess(samples: list[np.ndarray]) -> tuple[str, str]:
    """Judge one target's samples on their own, before comparing to any other."""
    if len(samples) < MIN_SAMPLES_PER_TARGET:
        return INSUFFICIENT, f"only {len(samples)} usable frames"

    arr = np.asarray(samples)
    yaw_jitter = float(arr[:, 0].std())
    pitch_jitter = float(arr[:, 1].std())
    if yaw_jitter > MAX_JITTER_DEG or pitch_jitter > MAX_JITTER_DEG:
        return UNSTEADY, f"aim wandered ({yaw_jitter:.1f}/{pitch_jitter:.1f} deg)"
    return OK, f"{len(samples)} samples"


def feature_scale(collected: dict) -> np.ndarray | None:
    """Typical within-target spread, used as the yardstick for conflicts.

    Built from the targets recorded so far rather than fixed up front, because
    how steadily any given person holds a look is exactly what it has to measure.
    """
    spreads = [np.asarray(s).std(axis=0) for s in collected.values() if len(s) >= 2]
    if len(spreads) < 3:
        return None  # too early to have a meaningful sense of scale
    scale = np.median(np.asarray(spreads), axis=0)
    return np.maximum(scale, 1e-3)


def find_conflict(key, collected: dict) -> tuple | None:
    """An already-recorded target on another display that this one resembles.

    Only across displays: neighbouring dots on the same display are supposed to
    be close, that is what makes the profile cover the whole panel.
    """
    scale = feature_scale(collected)
    if scale is None:
        return None

    here = np.median(np.asarray(collected[key]), axis=0)
    nearest, best = None, np.inf
    for other, samples in collected.items():
        if other == key or other[0] == key[0]:
            continue
        distance = float(np.linalg.norm((np.median(np.asarray(samples), axis=0) - here) / scale))
        if distance < best:
            nearest, best = other, distance
    if nearest is not None and best < CONFLICT_DISTANCE:
        return nearest
    return None


def run(settings: Settings) -> int:
    displays = active_displays()
    if len(displays) < 2:
        print("Only one display found - there is nothing to switch between.", file=sys.stderr)
        return 1

    detector = FaceDetector(score_threshold=settings.min_score)
    cap = open_camera(settings)
    by_id = {d.id: d for d in displays}

    print(f"Calibrating {len(displays)} displays, five points each.")
    print("Look straight at whichever dot is pulsing, and hold still while it is solid.")
    print("Green tick means good. Red means it will be taken again.\n")

    collected: dict[tuple[int, int], list[np.ndarray]] = {}
    attempts: dict[tuple[int, int], int] = {}
    revisited: set[tuple[int, int]] = set()

    try:
        overlay = CalibrationOverlay(displays)
        try:
            queue = deque((d.id, i) for d in displays for i in range(len(overlay.targets[d.id])))
            while queue:
                key = queue.popleft()
                display_id, index = key
                display = by_id[display_id]
                target = overlay.targets[display_id][index]
                attempts[key] = attempts.get(key, 0) + 1

                overlay.set_active(display_id, index)
                warp_cursor(*target_to_global(display, target))
                _blink(overlay, cap, settings.settle_seconds, settings.blink_hz)

                overlay.set_flash(True)
                samples = _collect(overlay, cap, detector, settings.sample_seconds)
                verdict, detail = assess(samples)

                label = f"{display.label[:26]:28s} {target.name:<13}"
                if verdict != OK and attempts[key] < MAX_ATTEMPTS:
                    print(f"    {label} {detail} - retrying")
                    _flash_bad(overlay, cap, display_id, index)
                    queue.appendleft(key)
                    continue

                if verdict == INSUFFICIENT:
                    print(
                        f"\n  Gave up on {display.label} / {target.name}: {detail}.\n"
                        "  Your face is probably outside the camera's view at this angle. "
                        "Reposition the camera so it still sees you when you look here.",
                        file=sys.stderr,
                    )
                    return 1

                collected[key] = samples
                overlay.set_state(display_id, index, GOOD)
                suffix = "" if verdict == OK else f"  (accepted anyway: {detail})"
                print(f"    {label} {detail}{suffix}")

                clash = find_conflict(key, collected)
                if clash is not None and clash not in revisited:
                    revisited.add(clash)
                    overlay.set_state(*clash, REVISIT)
                    queue.append(clash)
                    other = by_id[clash[0]]
                    print(
                        f"      ^ too close to {other.label[:22]} / "
                        f"{overlay.targets[clash[0]][clash[1]].name} - will retake that one"
                    )
        finally:
            overlay.close()
    finally:
        cap.release()

    by_display: dict[int, tuple[str, np.ndarray]] = {}
    for (display_id, _index), samples in collected.items():
        pooled = by_display.get(display_id)
        rows = samples if pooled is None else list(pooled[1]) + samples
        by_display[display_id] = (by_id[display_id].label, np.asarray(rows))

    for display_id, (label, rows) in by_display.items():
        if len(rows) < MIN_SAMPLES_PER_DISPLAY:
            print(
                f"\n  Only {len(rows)} usable samples for {label}; not enough to build a profile.",
                file=sys.stderr,
            )
            return 1

    calibration = Calibration.build(by_display)
    path = calibration.save()
    print(f"\nSaved calibration to {path}")
    if revisited:
        print(f"Retook {len(revisited)} target(s) that sat too close to another display.")
    _report_separation(calibration)
    return 0


def _blink(overlay: CalibrationOverlay, cap, seconds: float, hz: float) -> None:
    """Pulse the active dot, keeping the camera draining so it stays current."""
    end = time.monotonic() + seconds
    period = 1.0 / (max(hz, 0.25) * 2.0)
    on = True
    while time.monotonic() < end:
        overlay.set_flash(on)
        on = not on
        phase_end = time.monotonic() + period
        while time.monotonic() < phase_end:
            cap.read()
            overlay.pump()
    overlay.set_flash(True)


def _flash_bad(overlay: CalibrationOverlay, cap, display_id: int, index: int) -> None:
    """Blink the dot red, then hand it back to the yellow pulse for another go."""
    overlay.set_state(display_id, index, BAD)
    end = time.monotonic() + RED_FLASH_SECONDS
    period = 1.0 / (RED_FLASH_HZ * 2.0)
    on = True
    while time.monotonic() < end:
        overlay.set_flash(on)
        on = not on
        phase_end = time.monotonic() + period
        while time.monotonic() < phase_end:
            cap.read()
            overlay.pump()
    overlay.set_state(display_id, index, ACTIVE)
    overlay.set_flash(True)


def _collect(overlay: CalibrationOverlay, cap, detector, seconds: float) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        overlay.pump()
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        detection = detector.detect(frame)
        if detection is None:
            continue
        pose = estimate(detection.landmarks, frame.shape[1], frame.shape[0])
        if pose is not None:
            samples.append(pose.features)
    return samples


def _report_separation(calibration: Calibration) -> None:
    """How far apart the displays landed, in units of within-display noise.

    Anything under about 2 will misfire; that is the number to watch if the
    tracker feels twitchy.
    """
    scale = np.asarray(calibration.scale)
    centroids = np.asarray([p.mean for p in calibration.profiles])
    labels = [p.label for p in calibration.profiles]

    print("\nSeparation between displays (higher is better, under 2.0 is marginal):")
    worst = float("inf")
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = float(np.linalg.norm((centroids[i] - centroids[j]) / scale))
            worst = min(worst, d)
            flag = "  <-- weak" if d < 2.0 else ""
            print(f"  {labels[i]}  vs  {labels[j]}:  {d:.1f}{flag}")
    if worst < 2.0:
        print(
            "\nAt least one pair is hard to tell apart. Those two displays are "
            "probably too close together in angle from where you sit."
        )
