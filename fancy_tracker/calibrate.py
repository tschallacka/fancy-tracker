"""Guided calibration.

Every display is dimmed and shows five dots - its four corners and its centre.
The dot to look at blinks, then goes solid while samples are taken, so there is
never any doubt about where to point your head. The cursor is parked on the dot
too, as a second cue.

All five targets feed one profile per display. That is deliberate: a display is
an area, not a point, and a profile built only from its centre would fail to
recognise a glance at its far edge.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .calibration import Calibration
from .detector import FaceDetector
from .displays import Display, active_displays, warp_cursor
from .overlay import CalibrationOverlay, Target
from .pose import estimate
from .tracker import Settings, open_camera

MIN_SAMPLES_PER_DISPLAY = 20


def target_to_global(display: Display, target: Target) -> tuple[float, float]:
    """Dot position in macOS global cursor coordinates.

    Targets are in the display's own space with y up from the bottom, which is
    AppKit's convention. The cursor lives in CoreGraphics space, y down from the
    top of the main display, so the vertical axis has to be flipped.
    """
    return (display.x + target.x, display.y + (display.height - target.y))


def run(settings: Settings) -> int:
    displays = active_displays()
    if len(displays) < 2:
        print("Only one display found - there is nothing to switch between.", file=sys.stderr)
        return 1

    detector = FaceDetector(score_threshold=settings.min_score)
    cap = open_camera(settings)

    print(f"Calibrating {len(displays)} displays, five points each.")
    print("Look straight at whichever dot is blinking, and hold still while it is solid.\n")

    collected: dict[int, tuple[str, np.ndarray]] = {}
    try:
        overlay = CalibrationOverlay(displays)
        try:
            for index, display in enumerate(displays, start=1):
                label = f"display {index} - {display.label}"
                print(f"[{index}/{len(displays)}] {label}")

                samples: list[np.ndarray] = []
                for t_index, target in enumerate(overlay.targets[display.id]):
                    overlay.show_target(display.id, t_index)
                    warp_cursor(*target_to_global(display, target))

                    _blink(overlay, cap, settings.settle_seconds, settings.blink_hz)
                    overlay.set_flash(True)
                    got = _collect(overlay, cap, detector, settings.sample_seconds)
                    samples.extend(got)
                    print(f"    {target.name:<13} {len(got):>3} samples")

                if len(samples) < MIN_SAMPLES_PER_DISPLAY:
                    print(
                        f"\n  Only {len(samples)} usable samples for {label}.\n"
                        "  Your face is probably outside the camera's view at this angle. "
                        "Reposition the camera so it still sees you when you look here.",
                        file=sys.stderr,
                    )
                    return 1
                collected[display.id] = (label, np.asarray(samples))
        finally:
            overlay.close()
    finally:
        cap.release()

    calibration = Calibration.build(collected)
    path = calibration.save()
    print(f"\nSaved calibration to {path}")
    _report_separation(calibration)
    return 0


def _blink(overlay: CalibrationOverlay, cap, seconds: float, hz: float) -> None:
    """Blink the active dot, keeping the camera draining so it stays current."""
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
