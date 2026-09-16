"""The run loop: remember a cursor position per display, jump on a gaze change."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, Classifier, calibration_path, state_dir
from .detector import FaceDetector
from .displays import Display, active_displays, cursor_position, display_at, warp_cursor
from .emphasis import CursorEmphasis
from .pose import Pose, estimate
from .prompt import Question, describe_change


def positions_path() -> Path:
    return state_dir() / "positions.json"


@dataclass
class Settings:
    camera: int = 0
    width: int = 640
    height: int = 480
    min_score: float = 0.6
    smoothing: float = 0.35  # EMA weight on each new sample
    dwell: int = 6  # consecutive agreeing frames before a gaze counts
    margin: float = 0.35  # required lead over the runner-up display
    cooldown: float = 0.6  # seconds between jumps
    mouse_grace: float = 0.5  # defer a jump this long after a manual mouse move
    # Head start for the display you are already on. Kept modest because a
    # value that steadies a well-separated pair can make a tight pair
    # unreachable, and that failure is silent - `fancy-tracker check` measures
    # how much room each move actually has.
    stickiness: float = 0.35

    # How far outside every monitor a gaze may land before it is treated as
    # aimed between them - at the desk, at a gap - rather than at any of them.
    gap_tolerance: float = 2.0
    prompt_on_change: bool = True
    dry_run: bool = False
    preview: bool = False

    # Jump to the middle of the display rather than to wherever the cursor was
    # left. A fixed landing spot is far easier to find again than a moving one.
    recall_position: bool = False

    # A cursor that teleports is easy to lose, so it swells on arrival the way
    # a shaken one does, then settles back.
    emphasis_seconds: float = 1.1
    emphasis_scale: float = 4.0

    # Calibration pacing. A slow blink is easier to follow to a new corner than
    # a fast one, and the settle time has to cover actually turning your head.
    blink_hz: float = 2.0
    settle_seconds: float = 1.8
    sample_seconds: float = 1.5


class CursorMemory:
    """Last known cursor position per display, persisted between runs."""

    def __init__(self, displays: list[Display]):
        self._pos: dict[int, tuple[float, float]] = {d.id: d.center for d in displays}
        self._displays = {d.id: d for d in displays}
        self._load()

    def _load(self) -> None:
        path = positions_path()
        if not path.is_file():
            return
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        for key, value in saved.items():
            did = int(key)
            display = self._displays.get(did)
            # A display may have moved or been unplugged since; clamp it back in.
            if display and isinstance(value, list) and len(value) == 2:
                self._pos[did] = display.clamp(float(value[0]), float(value[1]))

    def save(self) -> None:
        path = positions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({str(k): list(v) for k, v in self._pos.items()}, indent=2) + "\n"
        )

    def update_displays(self, displays: list[Display]) -> None:
        """Adopt a new display layout, keeping any position still on-screen."""
        self._displays = {d.id: d for d in displays}
        for d in displays:
            if d.id in self._pos:
                self._pos[d.id] = d.clamp(*self._pos[d.id])
            else:
                self._pos[d.id] = d.center

    def remember(self, display_id: int, x: float, y: float) -> None:
        self._pos[display_id] = (x, y)

    def recall(self, display_id: int) -> tuple[float, float]:
        if display_id in self._pos:
            return self._pos[display_id]
        return self._displays[display_id].center


class Tracker:
    DISPLAY_REFRESH_SECONDS = 2.0
    CALIBRATION_REFRESH_SECONDS = 2.0

    def __init__(self, classifier: Classifier, settings: Settings):
        self.classifier = classifier
        self.settings = settings
        self.displays = active_displays()
        self.memory = CursorMemory(self.displays)
        self._by_id = {d.id: d for d in self.displays}
        self._displays_checked = time.monotonic()

        self._smoothed: np.ndarray | None = None
        self._candidate: int | None = None
        self._streak = 0
        self._stable_gaze: int | None = None
        self._last_jump = 0.0
        self._last_cursor = cursor_position()
        self._last_user_move = 0.0
        self._warp_target: tuple[float, float] | None = None
        self._calibration_mtime = self._calibration_stamp()
        self._calibration_checked = time.monotonic()
        self._question: Question | None = None
        self._prompted_for: dict | None = None
        # The first layout check belongs to run(), not here. Constructing a
        # Tracker must not put a dialog on screen: it is done in tests, and a
        # constructor that raises UI is a side effect nobody asked for.
        self._layout_checked = False
        self._emphasis = CursorEmphasis(
            seconds=settings.emphasis_seconds, scale=settings.emphasis_scale
        )

    @staticmethod
    def _calibration_stamp() -> float:
        try:
            return calibration_path().stat().st_mtime
        except OSError:
            return 0.0

    def _reset_gaze(self) -> None:
        self._smoothed = None
        self._candidate = None
        self._streak = 0
        self._stable_gaze = None

    def _reload_calibration_if_changed(self) -> None:
        """Adopt a new calibration without needing a restart.

        This usually runs as a login agent, so requiring a restart after
        recalibrating would mean quietly tracking against stale profiles - the
        one failure that looks exactly like the tracker being bad at its job.
        """
        now = time.monotonic()
        if now - self._calibration_checked < self.CALIBRATION_REFRESH_SECONDS:
            return
        self._calibration_checked = now

        stamp = self._calibration_stamp()
        if stamp == self._calibration_mtime:
            return

        try:
            classifier = Classifier(Calibration.load())
        except (OSError, ValueError, KeyError, TypeError):
            return  # mid-write or malformed; try again on the next tick

        self._calibration_mtime = stamp
        self.classifier = classifier
        self._reset_gaze()
        print(f"  calibration reloaded ({len(classifier.display_ids)} displays)")

    def _layout_signature(self, displays: list[Display]) -> dict[int, tuple[float, float]]:
        return {d.id: (d.width, d.height) for d in displays}

    def _calibration_matches(self, displays: list[Display]) -> tuple[bool, list, list, list]:
        """Does the saved calibration still describe these monitors?

        Compared against what calibration recorded rather than against the
        previous poll, so an arrangement changed while the tracker was not
        running is caught too.
        """
        known = {p.display_id: p for p in self.classifier.calibration.profiles}
        here = {d.id: d for d in displays}

        added = [here[i].label for i in here.keys() - known.keys()]
        removed = [known[i].label for i in known.keys() - here.keys()]
        moved = []
        for i in here.keys() & known.keys():
            profile = known[i]
            # Width and height are only recorded from calibration v2 onwards.
            if profile.width <= 0 or profile.height <= 0:
                continue
            if (
                abs(profile.width - here[i].width) > 1.0
                or abs(profile.height - here[i].height) > 1.0
            ):
                moved.append(here[i].label)
        return (not (added or removed or moved), added, removed, moved)

    def _refresh_displays(self) -> None:
        """Re-read the display list periodically.

        macOS reports zero active displays while the screens are asleep, and the
        arrangement can change under a running tracker when a monitor is
        plugged, unplugged or moved. Enumerating once at startup would leave the
        tracker aiming at a layout that no longer exists - or, if it started
        while the screens were asleep, at no layout at all.
        """
        now = time.monotonic()
        if now - self._displays_checked < self.DISPLAY_REFRESH_SECONDS:
            return
        self._displays_checked = now

        current = active_displays()
        if [d.id for d in current] == [d.id for d in self.displays]:
            return

        self.displays = current
        self._by_id = {d.id: d for d in current}
        self.memory.update_displays(current)

        # Asleep screens report as no displays at all; that is not a change of
        # arrangement and must not trigger a prompt.
        if current:
            self._check_layout(current)

    def _check_layout(self, displays: list[Display]) -> None:
        """Offer a recalibration when the monitors no longer match the profile."""
        matches, added, removed, moved = self._calibration_matches(displays)
        signature = self._layout_signature(displays)
        if matches:
            self._prompted_for = None
            return
        if self._prompted_for == signature or self._question is not None:
            return  # already asked about exactly this layout

        self._prompted_for = signature
        if self.settings.prompt_on_change:
            print("  monitor layout changed; asking whether to recalibrate")
            self._question = Question(describe_change(added, removed, moved))
        else:
            print("  monitor layout changed; calibration is stale")

    def _poll_question(self) -> bool:
        """True when the user asked for a recalibration."""
        if self._question is None:
            return False
        answer = self._question.answered()
        if answer is None:
            return False
        self._question = None
        if answer:
            print("  recalibration accepted")
            return True
        print("  recalibration declined")
        return False

    def _observe_cursor(self) -> None:
        """Track the cursor and note whether the move was ours or the user's."""
        pos = cursor_position()
        if pos != self._last_cursor:
            moved_by_us = (
                self._warp_target is not None
                and abs(pos[0] - self._warp_target[0]) < 2.0
                and abs(pos[1] - self._warp_target[1]) < 2.0
            )
            if not moved_by_us:
                self._last_user_move = time.monotonic()
            self._last_cursor = pos

        here = display_at(self.displays, *pos)
        if here is not None:
            self.memory.remember(here.id, *pos)

    def _update_gaze(self, features: np.ndarray) -> tuple[int, float]:
        if self._smoothed is None:
            self._smoothed = features.copy()
        else:
            a = self.settings.smoothing
            self._smoothed = a * features + (1.0 - a) * self._smoothed

        distances = self.classifier.distances(self._smoothed)

        # The display you are already on gets a head start, so leaving it costs
        # more than arriving did. Without this the two closest displays trade
        # places whenever the pose sits near the boundary between them, which
        # shows up in the logs as A -> B -> A -> B at margins barely over the
        # threshold.
        if self._stable_gaze in distances:
            distances[self._stable_gaze] -= self.settings.stickiness

        ranked = sorted(distances.items(), key=lambda kv: kv[1])
        display_id, best = ranked[0]
        margin = (ranked[1][1] - best) if len(ranked) > 1 else float("inf")

        if display_id == self._candidate and margin >= self.settings.margin:
            self._streak += 1
        else:
            self._candidate = display_id
            self._streak = 1 if margin >= self.settings.margin else 0
        return display_id, float(margin)

    def _maybe_jump(self, gaze_id: int) -> str | None:
        """Jump if the gaze has newly settled on a different display."""
        if self._streak < self.settings.dwell:
            return None
        if gaze_id == self._stable_gaze:
            return None

        now = time.monotonic()
        if now - self._last_jump < self.settings.cooldown:
            return None
        if now - self._last_user_move < self.settings.mouse_grace:
            return None  # the user is working the mouse; do not fight them

        # A calibrated display can be unplugged while running, and there is
        # nowhere to jump to on a display that is not there.
        display = self._by_id.get(gaze_id)
        if display is None:
            return None

        # Monitors that are physically apart have a gap between them that the
        # arrangement does not model. A gaze landing in that gap belongs to
        # neither, so moving the cursor anywhere would be a guess.
        #
        # Both tests have to agree before a look is dismissed as aimed between
        # monitors. The surface fit alone is not enough: on a monitor viewed at
        # a steep angle the fitted plane can miss its own corners by more than
        # this tolerance, and suppressing there means never being able to look
        # at that corner at all.
        placed = self.classifier.locate(self._smoothed)
        if (
            placed is not None
            and placed[3] > self.settings.gap_tolerance
            and self.classifier.nearest_dot_distance(self._smoothed) > self.settings.gap_tolerance
        ):
            return None
        previous, self._stable_gaze = self._stable_gaze, gaze_id
        if previous is None:
            return None  # first lock-on only establishes where we are

        # The centre is the default because it is the only spot that is the same
        # every time: you already know where to look before the cursor arrives.
        # Restoring the last position makes the cursor harder to reacquire, which
        # is the whole problem this is meant to solve.
        target = self.memory.recall(gaze_id) if self.settings.recall_position else display.center

        # Charge the cooldown either way, so a dry run reports exactly the jumps
        # a real run would make rather than a more permissive superset of them.
        self._last_jump = now

        if self.settings.dry_run:
            return f"would jump {self.classifier.label(gaze_id)} -> {target[0]:.0f},{target[1]:.0f}"

        warp_cursor(*target)
        self._emphasis.show(*target)
        self._warp_target = target
        self._last_cursor = cursor_position()
        return f"jumped to {self.classifier.label(gaze_id)} at {target[0]:.0f},{target[1]:.0f}"

    def _recalibrate(self, cap) -> cv2.VideoCapture:
        """Run calibration in place, then carry on tracking with the result.

        The camera is handed over rather than shared: two processes reading it
        halves the frame rate, and the tracker warping the cursor would fight
        the dot calibration is asking you to look at.
        """
        from . import calibrate  # imported here; calibrate imports this module

        cap.release()
        try:
            calibrate.run(self.settings)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad. Recalibration touches the camera, the window
            # server and the filesystem; whatever it fails on, the tracker still
            # has a working calibration loaded and should carry on with it
            # rather than exit and be restarted in a loop by launchd.
            print(f"  recalibration failed: {exc}")
        self._calibration_checked = 0.0
        self._reload_calibration_if_changed()
        self._prompted_for = None
        return open_camera(self.settings)

    def _warn_if_unreachable(self) -> None:
        """Say so when the settings make a monitor impossible to reach.

        Nothing errors when stickiness is too high for a pair; the cursor just
        does not follow, and only between those two monitors. Left to be noticed
        in use, it reads as the tracking being unreliable rather than as a
        setting being wrong.
        """
        from .tuning import recommend, transitions

        try:
            moves = transitions(self.classifier.calibration, margin=self.settings.margin)
        except (ValueError, KeyError):
            return
        blocked = [m for m in moves if m.blocked_at(self.settings.stickiness)]
        if not blocked:
            return
        print(f"  warning: --stickiness {self.settings.stickiness} blocks {len(blocked)} move(s):")
        for m in blocked[:4]:
            print(
                f"    {m.src_label[:24]} -> {m.dst_label[:24]} (room for {m.usable_stickiness:.2f})"
            )
        print(f"    try --stickiness {recommend(moves)}, or run `fancy-tracker check`")

    def run(self) -> int:
        detector = FaceDetector(score_threshold=self.settings.min_score)
        cap = open_camera(self.settings)
        paused = False
        self._warn_if_unreachable()
        print(
            "tracking - ctrl-c to stop"
            + (" ('p' pauses in the preview window)" if self.settings.preview else "")
        )

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                if not self._layout_checked and self.displays:
                    self._layout_checked = True
                    self._check_layout(self.displays)

                self._emphasis.tick()
                self._refresh_displays()
                self._reload_calibration_if_changed()
                if self._poll_question():
                    cap = self._recalibrate(cap)
                    continue
                self._observe_cursor()

                pose: Pose | None = None
                detection = detector.detect(frame)
                if detection is not None:
                    pose = estimate(detection.landmarks, frame.shape[1], frame.shape[0])

                note = None
                if pose is not None and not paused:
                    gaze_id, margin = self._update_gaze(pose.features)
                    note = self._maybe_jump(gaze_id)
                    if note:
                        print(f"  {note}  (margin {margin:.2f})")

                if self.settings.preview:
                    key = show_preview(frame, detection, pose, self, paused)
                    if key == ord("q"):
                        break
                    if key == ord("p"):
                        paused = not paused
                        print("  paused" if paused else "  resumed")
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            cap.release()
            if self.settings.preview:
                cv2.destroyAllWindows()
            self._emphasis.close()
            if self.settings.recall_position:
                self.memory.save()
        return 0


def open_camera(settings: Settings) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(settings.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {settings.camera}. On macOS the terminal running "
            "this needs Camera permission: System Settings > Privacy & Security > Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    return cap


def show_preview(frame, detection, pose, tracker: Tracker, paused: bool) -> int:
    view = frame.copy()
    if detection is not None:
        x, y, w, h = (int(v) for v in detection.box)
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
        for px, py in detection.landmarks.astype(int):
            cv2.circle(view, (int(px), int(py)), 2, (0, 160, 255), -1)

    lines = []
    if pose is not None:
        lines.append(f"yaw {pose.yaw:+6.1f}  pitch {pose.pitch:+6.1f}")
    else:
        lines.append("no face")
    if tracker._stable_gaze is not None:
        lines.append(tracker.classifier.label(tracker._stable_gaze))
    lines.append(f"streak {tracker._streak}" + ("  PAUSED" if paused else ""))

    for i, text in enumerate(lines):
        cv2.putText(
            view, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
    cv2.imshow("fancy-tracker", view)
    return cv2.waitKey(1) & 0xFF
