"""Guided calibration.

Every display is dimmed and shows five dots - its four corners and its centre.
The dot to look at pulses yellow, and goes solid only once your head has
actually stopped moving, so a long swing to a far corner is waited out rather
than sampled through. The cursor is parked on it too, as a second cue.

Each dot is judged as it is taken. A good one turns green with a tick. One whose
samples were unusable flashes red and goes back to yellow to be taken again. One
that lands too close to a dot already recorded on a *different* display is an
unclear edge, so the earlier one is ringed amber and queued to be retaken.

After the pass, the monitors are compared as wholes. Any pair that is still hard
to tell apart has the dots along their facing edges retaken, because those are
the ones carrying the ambiguity. The per-dot check cannot see this: it measures
in units of how steady one hold is, while the pair problem lives in units of how
wide a whole monitor is.
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np

from .calibration import Calibration
from .detector import FaceDetector
from .displays import Display, active_displays, warp_cursor
from .geometry import describe
from .overlay import ACTIVE, BAD, GOOD, REVISIT, CalibrationOverlay, Target
from .pose import estimate
from .tracker import Settings, open_camera

MIN_SAMPLES_PER_TARGET = 8
MIN_SAMPLES_PER_DISPLAY = 20

# Degrees of yaw or pitch wobble within one target's sampling window before the
# samples are treated as a smeared aim rather than a steady look.
MAX_JITTER_DEG = 6.0

# Stillness gate. Settled means the window's two halves agree to within this
# many degrees - drift, not spread, because landmark noise alone moves a still
# head by a couple of degrees and a spread test would never pass.
STILL_FRAMES = 8
STILL_DEGREES = 2.5
STILL_TIMEOUT = 4.0

# The window must also cover this much wall-clock time. Without it the test is
# frame-rate dependent: on a fast capture, eight frames can span a few
# milliseconds, over which any movement looks like no movement at all.
STILL_WINDOW_SECONDS = 0.35

# Scaled distance below which two targets on different displays are too alike to
# tell apart. Roughly "within the noise of a single target".
CONFLICT_DISTANCE = 2.5

# Display-pair separation below which the pair is reworked after the first pass.
WEAK_PAIR = 2.0
MAX_PAIR_PASSES = 2

MAX_ATTEMPTS = 3
RED_FLASH_SECONDS = 0.9
RED_FLASH_HZ = 5.0

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
    """Typical within-target spread, used as the yardstick for conflicts."""
    spreads = [np.asarray(s).std(axis=0) for s in collected.values() if len(s) >= 2]
    if len(spreads) < 3:
        return None  # too early to have a meaningful sense of scale
    return np.maximum(np.median(np.asarray(spreads), axis=0), 1e-3)


def find_conflict(key, collected: dict) -> tuple | None:
    """An already-recorded target on another display that this one resembles."""
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
    return nearest if nearest is not None and best < CONFLICT_DISTANCE else None


def weak_pairs(calibration: Calibration) -> list[tuple[int, int, float]]:
    """Display pairs still too alike after a full pass."""
    scale = np.asarray(calibration.scale)
    out = []
    for i, a in enumerate(calibration.profiles):
        for b in calibration.profiles[i + 1 :]:
            d = float(np.linalg.norm((np.asarray(a.mean) - np.asarray(b.mean)) / scale))
            if d < WEAK_PAIR:
                out.append((a.display_id, b.display_id, d))
    return sorted(out, key=lambda t: t[2])


def facing_targets(
    a: Display, b: Display, targets_by_display: dict[int, list[Target]]
) -> list[tuple[int, int]]:
    """The dots along the edge each of two displays presents to the other.

    Those are the dots carrying the ambiguity: the far side of a monitor is
    never confusable with its neighbour, only the side facing it is.
    """
    keys = []
    for near, far in ((a, b), (b, a)):
        towards_right = far.center[0] > near.center[0]
        for index, target in enumerate(targets_by_display[near.id]):
            on_right_half = target.x > near.width / 2.0
            if target.name == "centre" or on_right_half == towards_right:
                keys.append((near.id, index))
    return keys


def run(settings: Settings) -> int:
    displays = active_displays()
    if len(displays) < 2:
        print("Only one display found - there is nothing to switch between.", file=sys.stderr)
        return 1

    detector = FaceDetector(score_threshold=settings.min_score)
    cap = open_camera(settings)
    by_id = {d.id: d for d in displays}

    print(f"Calibrating {len(displays)} displays, five points each.")
    print("Look straight at whichever dot is pulsing. It goes solid once you hold still.")
    print("Green tick means good. Red means it will be taken again.\n")

    collected: dict[tuple[int, int], list[np.ndarray]] = {}
    revisited: set[tuple[int, int]] = set()

    try:
        overlay = CalibrationOverlay(displays)
        # Kept so the calibration can still be assembled once the overlay is gone.
        targets_by_display = {did: list(ts) for did, ts in overlay.targets.items()}
        try:
            queue = deque(
                (did, i) for did, ts in targets_by_display.items() for i in range(len(ts))
            )
            if not _work_queue(
                queue, overlay, cap, detector, settings, by_id, collected, revisited
            ):
                return 1

            # Second pass: whole-monitor ambiguity the per-dot check cannot see.
            for _ in range(MAX_PAIR_PASSES):
                trial = _build(collected, by_id, targets_by_display)
                weak = weak_pairs(trial)
                if not weak:
                    break
                retake: deque = deque()
                for a_id, b_id, distance in weak:
                    a, b = by_id[a_id], by_id[b_id]
                    print(
                        f"\n  {a.label[:24]} and {b.label[:24]} are still hard to tell "
                        f"apart ({distance:.1f}). Retaking their facing edges."
                    )
                    for key in facing_targets(a, b, targets_by_display):
                        if key not in retake:
                            retake.append(key)
                            overlay.set_state(*key, REVISIT)
                if not _work_queue(
                    retake, overlay, cap, detector, settings, by_id, collected, revisited
                ):
                    return 1
        finally:
            overlay.close()
    finally:
        cap.release()

    for display_id in {k[0] for k in collected}:
        rows = sum(len(collected[k]) for k in collected if k[0] == display_id)
        if rows < MIN_SAMPLES_PER_DISPLAY:
            print(
                f"\n  Only {rows} usable samples for {by_id[display_id].label}; "
                "not enough to build a profile.",
                file=sys.stderr,
            )
            return 1

    calibration = _build(collected, by_id, targets_by_display)
    path = calibration.save()
    print(f"\nSaved calibration to {path}")
    if revisited:
        print(f"Retook {len(revisited)} target(s) that sat too close to another display.")
    _report_separation(calibration)
    print(describe(calibration.models()))
    return 0


def _build(collected, by_id, targets_by_display) -> Calibration:
    """Assemble a calibration from whatever has been collected so far."""
    payload = {}
    for display_id in {k[0] for k in collected}:
        display = by_id[display_id]
        entries = []
        for (did, index), samples in collected.items():
            if did != display_id:
                continue
            target = targets_by_display[display_id][index]
            entries.append((target.name, target.x, target.y, samples))
        payload[display_id] = (display.label, display.width, display.height, entries)
    return Calibration.build_with_targets(payload)


def _work_queue(queue, overlay, cap, detector, settings, by_id, collected, revisited) -> bool:
    """Take every target in the queue, retrying and enqueuing conflicts. False aborts."""
    attempts: dict[tuple[int, int], int] = {}

    while queue:
        key = queue.popleft()
        display_id, index = key
        display = by_id[display_id]
        target = overlay.targets[display_id][index]
        attempts[key] = attempts.get(key, 0) + 1

        overlay.set_active(display_id, index)
        warp_cursor(*target_to_global(display, target))
        _await_stillness(overlay, cap, detector, settings)

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
            return False

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
    return True


def _await_stillness(overlay: CalibrationOverlay, cap, detector, settings) -> None:
    """Pulse the dot until the head stops moving, then stop pulsing.

    A fixed settle time has to be guessed, and a guess long enough for a swing
    across four monitors is tedious everywhere else.

    Two things this has to get right. It must not measure the stillness of a
    head that has not started moving yet: when a new dot lights up you are still
    parked perfectly still on the previous one, and a naive gate opens
    immediately and then samples straight through the movement. So nothing
    counts until a minimum settling period has passed. And it tests for drift
    between the halves of the window rather than spread across it, because
    landmark noise alone moves a motionless head a couple of degrees and a
    spread test would simply never pass.
    """
    start = time.monotonic()
    min_wait = max(0.0, settings.settle_seconds)
    deadline = start + min_wait + STILL_TIMEOUT
    period = 1.0 / (max(settings.blink_hz, 0.25) * 2.0)
    # Held by time rather than by frame count: eight frames is a quarter second
    # at 30fps but a blink at 200, and the question is about seconds either way.
    recent: deque = deque(maxlen=1024)
    on = True
    next_toggle = 0.0

    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        if now >= next_toggle:
            overlay.set_flash(on)
            on = not on
            next_toggle = now + period
        overlay.pump()

        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        detection = detector.detect(frame)
        if detection is None:
            recent.clear()
            continue
        pose = estimate(detection.landmarks, frame.shape[1], frame.shape[0])
        if pose is None:
            continue

        if now - start < min_wait:
            # Still on the way to the dot; anything measured here is the old aim.
            recent.clear()
            continue

        recent.append((now, pose.features[:2]))
        while recent and now - recent[0][0] > STILL_WINDOW_SECONDS:
            recent.popleft()
        if _settled(recent):
            break

    overlay.set_flash(True)


def _settled(window) -> bool:
    """True when the aim has stopped drifting over a long enough stretch.

    Entries are (timestamp, [yaw, pitch]). Both conditions matter: the window
    has to cover real time, and its two halves have to agree. Drift rather than
    spread, because landmark noise alone moves a motionless head a couple of
    degrees.
    """
    if len(window) < STILL_FRAMES:
        return False
    times = [t for t, _f in window]
    # A little under the full window, so an ordinary frame-rate jitter does not
    # keep pushing the decision out.
    if times[-1] - times[0] < STILL_WINDOW_SECONDS * 0.6:
        return False
    arr = np.asarray([f for _t, f in window])
    half = len(arr) // 2
    drift = np.abs(arr[:half].mean(axis=0) - arr[half:].mean(axis=0))
    return bool(drift.max() < STILL_DEGREES)


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
    """How far apart the displays landed, in units of within-display noise."""
    scale = np.asarray(calibration.scale)
    centroids = np.asarray([p.mean for p in calibration.profiles])
    labels = [p.label for p in calibration.profiles]

    print("\nSeparation between displays (higher is better, under 2.0 is marginal):")
    worst = float("inf")
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = float(np.linalg.norm((centroids[i] - centroids[j]) / scale))
            worst = min(worst, d)
            flag = "  <-- weak" if d < WEAK_PAIR else ""
            print(f"  {labels[i]}  vs  {labels[j]}:  {d:.1f}{flag}")
    if worst < WEAK_PAIR:
        print(
            "\nThose two are genuinely at nearly the same angle from where you sit - "
            "usually two monitors that share a physical edge. The layout below says "
            "whether that is really the case."
        )
