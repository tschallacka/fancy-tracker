"""Everything that can be checked without a live camera.

Run inside the dev shell, which supplies both the model and the fixture face:

    nix develop --command python tests/selftest.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fancy_tracker.calibrate import target_to_global
from fancy_tracker.calibration import Calibration, Classifier
from fancy_tracker.detector import FaceDetector, model_path
from fancy_tracker.displays import Display, active_displays
from fancy_tracker.overlay import targets_for
from fancy_tracker.pose import FEATURE_NAMES, estimate
from fancy_tracker.tracker import CursorMemory, Settings, Tracker

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    print("1. model")
    check("model path ends in .onnx", model_path().endswith(".onnx"), model_path())

    face = os.environ.get("FANCY_TRACKER_TEST_FACE")
    if face and os.path.isfile(face):
        detector = FaceDetector()

        print("2. detection and pose on a real face")
        img = cv2.imread(face)
        det = detector.detect(img)
        check("face detected", det is not None)
        if det is not None:
            check("score plausible", det.score > 0.7, f"{det.score:.3f}")
            lm = det.landmarks
            check("eyes sit above mouth corners", lm[0][1] < lm[3][1] and lm[1][1] < lm[4][1])
            check("right eye is left of left eye in frame", lm[0][0] < lm[1][0])

            pose = estimate(lm, img.shape[1], img.shape[0])
            check("pose computed", pose is not None)
            if pose is not None:
                check(
                    "features finite",
                    bool(np.all(np.isfinite(pose.features))),
                    f"yaw={pose.yaw:.1f} pitch={pose.pitch:.1f}",
                )

                # The real proof that pose tracks orientation rather than
                # returning something constant: mirroring the face must flip yaw.
                print("3. yaw actually responds to orientation")
                flipped = cv2.flip(img, 1)
                det2 = detector.detect(flipped)
                pose2 = (
                    estimate(det2.landmarks, flipped.shape[1], flipped.shape[0]) if det2 else None
                )
                check(
                    "yaw sign flips on a mirrored face",
                    pose2 is not None and np.sign(pose2.yaw) != np.sign(pose.yaw),
                    f"{pose.yaw:.1f} -> {pose2.yaw:.1f}" if pose2 else "",
                )
    else:
        print("2-3. skipped (FANCY_TRACKER_TEST_FACE not set)")

    print("4. calibration round-trip and classification")
    rng = np.random.default_rng(0)
    centres = {
        101: [-40.0, 0.0, -0.5, 0.1],
        102: [0.0, 0.0, 0.0, 0.1],
        103: [25.0, 5.0, 0.3, 0.1],
        104: [50.0, 10.0, 0.6, 0.1],
    }
    collected = {
        did: (f"display {did}", np.asarray(c) + rng.normal(0, 1.5, size=(60, 4)))
        for did, c in centres.items()
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cal.json"
        Calibration.build(collected).save(path)
        loaded = Calibration.load(path)
    check("round-trips through JSON", len(loaded.profiles) == 4)

    clf = Classifier(loaded)
    correct = sum(clf.classify(np.asarray(c))[0] == did for did, c in centres.items())
    check("each centroid classifies to its own display", correct == 4, f"{correct}/4")
    _id, _dist, margin = clf.classify(np.asarray(centres[101]))
    check("clear case has a real margin", margin > 0.5, f"margin={margin:.2f}")

    # macOS reports zero active displays while the screens are asleep or locked,
    # so every display-dependent check below has to tolerate an empty list.
    live = active_displays()
    if not live:
        print("\n(no active displays - screens asleep? skipping display-dependent checks)")

    print("5. overlay targets land inside their display")
    for d in live:
        outside = [t.name for t in targets_for(d) if not d.contains(*target_to_global(d, t))]
        check(f"all 5 targets inside display {d.id}", not outside, ", ".join(outside))

    print("6. target geometry is the right way up")
    if live:
        by_name = {t.name: target_to_global(live[0], t) for t in targets_for(live[0])}
        tl, br = by_name["top-left"], by_name["bottom-right"]
        check(
            "top-left has smaller x and y than bottom-right",
            tl[0] < br[0] and tl[1] < br[1],
            f"{tl} vs {br}",
        )

    print("7. cursor memory defaults to display centres")
    fake = [Display(9, 0, 0, 800, 600, False, True), Display(8, 800, 0, 1000, 1000, False, False)]
    mem = CursorMemory(fake)
    check(
        "centre of a display with no history", mem.recall(9) == (400.0, 300.0), str(mem.recall(9))
    )
    check("centre of a second display", mem.recall(8) == (1300.0, 500.0), str(mem.recall(8)))

    # Synthetic displays, so these run whether or not the screens are awake.
    # macOS reports no displays while asleep, and a test that silently skips is
    # a test that is not protecting anything.
    panels = [
        Display(9, 0, 0, 800, 600, False, True),
        Display(8, 800, 0, 1000, 1000, False, False),
    ]

    def tracker_on(panels_, settings):
        # Never let a test put a dialog on a real screen. Constructing a Tracker
        # no longer prompts, but these classifiers describe synthetic panels that
        # will never match the real monitors, so say so explicitly too.
        settings.prompt_on_change = False
        t = Tracker(clf_panels, settings)
        t.displays = panels_
        t._by_id = {d.id: d for d in panels_}
        t.memory = CursorMemory(panels_)
        t._streak = 10_000
        t._stable_gaze = panels_[0].id
        return t

    rng2 = np.random.default_rng(1)
    clf_panels = Classifier(
        Calibration.build(
            {
                d.id: (f"panel {d.id}", np.full(4, i * 25.0) + rng2.normal(0, 1.0, (40, 4)))
                for i, d in enumerate(panels)
            }
        )
    )

    print("8. a jump lands on the display centre")
    destination = panels[1]
    note = tracker_on(panels, Settings(dry_run=True))._maybe_jump(destination.id)
    cx, cy = destination.center
    check(
        "jumps to centre, not to a remembered spot",
        note is not None and note.endswith(f"-> {cx:.0f},{cy:.0f}"),
        note or "no jump produced",
    )

    # --recall-position must still honour the stored spot.
    t2 = tracker_on(panels, Settings(dry_run=True, recall_position=True))
    spot = (destination.x + 11.0, destination.y + 13.0)
    t2.memory.remember(destination.id, *spot)
    note2 = t2._maybe_jump(destination.id)
    check(
        "--recall-position restores the stored spot",
        note2 is not None and note2.endswith(f"-> {spot[0]:.0f},{spot[1]:.0f}"),
        note2 or "no jump produced",
    )

    # A calibrated display that is no longer connected must not crash.
    t3 = tracker_on(panels, Settings(dry_run=True))
    check("unplugged display is skipped, not fatal", t3._maybe_jump(999_999) is None)

    print("9. a wide display is not punished for being wide")
    # Two displays the same distance apart, but one sampled three times as
    # spread out - a tall portrait monitor against a compact laptop. Under one
    # shared scale the wide one loses its own edge; under per-display scale it
    # keeps it.
    wide = np.asarray([0.0, 0.0, 0.0, 0.0]) + rng.normal(0, 3.0, (200, 4))
    tight = np.asarray([12.0, 0.0, 0.0, 0.0]) + rng.normal(0, 1.0, (200, 4))
    cal2 = Calibration.build({201: ("wide", wide), 202: ("tight", tight)})
    clf3 = Classifier(cal2)
    scales = clf3._scales
    check(
        "wide display gets the larger scale",
        scales[0][0] > scales[1][0],
        f"{scales[0][0]:.2f} vs {scales[1][0]:.2f}",
    )
    edge = np.asarray([3.0, 0.0, 0.0, 0.0])  # one std out along the wide display
    check(
        "its own edge still classifies to it",
        clf3.classify(edge)[0] == 201,
        str(clf3.classify(edge)),
    )

    print("10. stickiness keeps you on the display you are already on")
    a_id, b_id = panels[0].id, panels[1].id
    clf4 = Classifier(
        Calibration.build(
            {
                a_id: ("A", np.zeros((40, 4)) + rng.normal(0, 1.0, (40, 4))),
                b_id: ("B", np.full((40, 4), 10.0) + rng.normal(0, 1.0, (40, 4))),
            }
        )
    )
    # A pose just past the midpoint, so B wins outright but only narrowly.
    probe = np.full(4, 5.2)
    d_probe = clf4.distances(probe)
    raw_lead = d_probe[a_id] - d_probe[b_id]

    held = Tracker(clf4, Settings(dry_run=True, stickiness=raw_lead + 0.5, smoothing=1.0))
    held._stable_gaze = a_id
    check(
        "marginal challenger does not win",
        held._update_gaze(probe)[0] == a_id,
        f"B's raw lead was {raw_lead:.2f}",
    )

    loose = Tracker(clf4, Settings(dry_run=True, stickiness=0.0, smoothing=1.0))
    loose._stable_gaze = a_id
    check("without stickiness the same pose switches", loose._update_gaze(probe)[0] == b_id)

    print("11. display list survives sleep and re-arrangement")
    a = Display(9, 0, 0, 800, 600, False, True)
    b = Display(8, 800, 0, 1000, 1000, False, False)
    mem2 = CursorMemory([a, b])
    mem2.remember(8, 1700.0, 900.0)
    # b moves and shrinks; the stored spot is now outside it and must be pulled in.
    b_moved = Display(8, 800, 0, 400, 400, False, False)
    mem2.update_displays([a, b_moved])
    x, y = mem2.recall(8)
    check("position clamped into the resized display", b_moved.contains(x, y), f"({x:.0f},{y:.0f})")
    # A newly attached display starts at its centre.
    c = Display(7, -900, 0, 900, 900, False, False)
    mem2.update_displays([a, b_moved, c])
    check("new display starts centred", mem2.recall(7) == (-450.0, 450.0), str(mem2.recall(7)))

    print("12. recalibrating is picked up without a restart")
    import time as _time

    from fancy_tracker import calibration as cal_mod

    with tempfile.TemporaryDirectory() as td:
        original = cal_mod.state_dir
        cal_mod.state_dir = lambda _td=td: Path(_td)
        try:
            first = {
                panels[0].id: ("A", np.zeros((40, 4)) + rng.normal(0, 1.0, (40, 4))),
                panels[1].id: ("B", np.full((40, 4), 10.0) + rng.normal(0, 1.0, (40, 4))),
            }
            Calibration.build(first).save()
            t = Tracker(Classifier(Calibration.load()), Settings(dry_run=True))
            t.displays = panels
            t._by_id = {d.id: d for d in panels}
            before = t.classifier

            # Rewrite with a third display and force the poll.
            second = dict(first)
            second[4242] = ("C", np.full((40, 4), -10.0) + rng.normal(0, 1.0, (40, 4)))
            _time.sleep(0.01)
            Calibration.build(second).save()
            t._calibration_checked = 0.0
            t._reload_calibration_if_changed()

            check("classifier was swapped", t.classifier is not before)
            check(
                "new display is known after reload",
                4242 in t.classifier.display_ids,
                str(sorted(t.classifier.display_ids)),
            )
            check("gaze state reset on reload", t._stable_gaze is None and t._streak == 0)

            # A truncated file must not take down a running tracker.
            kept = t.classifier
            calibration_file = Path(td) / "calibration.json"
            calibration_file.write_text('{"version": 1, "prof')
            t._calibration_checked = 0.0
            t._reload_calibration_if_changed()
            check("malformed calibration is ignored", t.classifier is kept)
        finally:
            cal_mod.state_dir = original

    print("13. per-target verdicts")
    from fancy_tracker.calibrate import (
        INSUFFICIENT,
        OK,
        UNSTEADY,
        assess,
        find_conflict,
    )

    steady = [np.array([10.0, 2.0, 0.1, 0.5]) + rng.normal(0, 0.4, 4) for _ in range(30)]
    check("steady samples pass", assess(steady)[0] == OK, assess(steady)[1])
    check("too few frames is insufficient", assess(steady[:4])[0] == INSUFFICIENT)
    wobbly = [np.array([10.0, 2.0, 0.1, 0.5]) + rng.normal(0, 12.0, 4) for _ in range(30)]
    check("a wandering aim is unsteady", assess(wobbly)[0] == UNSTEADY, assess(wobbly)[1])

    print("14. unclear edges between displays are found, within one are not")

    def cloud(centre, n=30, sd=0.5):
        return [np.asarray(centre, float) + rng.normal(0, sd, 4) for _ in range(n)]

    # (display_id, target_index). Display 1's dots are spread out; display 2 has
    # one dot sitting right on top of one of display 1's.
    coll = {
        (1, 0): cloud([0, 0, 0, 0]),
        (1, 1): cloud([20, 0, 0, 0]),
        (1, 2): cloud([40, 0, 0, 0]),
        (2, 0): cloud([200, 0, 0, 0]),
    }
    check(
        "a well-separated target reports no conflict",
        find_conflict((2, 0), coll) is None,
        str(find_conflict((2, 0), coll)),
    )
    coll[(2, 1)] = cloud([40.2, 0, 0, 0])  # all but on top of (1, 2)
    check(
        "an overlapping target on another display is caught",
        find_conflict((2, 1), coll) == (1, 2),
        str(find_conflict((2, 1), coll)),
    )
    check(
        "neighbours on the same display are not a conflict",
        find_conflict((1, 1), coll) is None,
        str(find_conflict((1, 1), coll)),
    )

    print("15. overlay dot states")
    from fancy_tracker import overlay as ov

    check(
        "every state has a distinct name",
        len({ov.PENDING, ov.ACTIVE, ov.GOOD, ov.BAD, ov.REVISIT}) == 5,
    )

    print("16. layout is recovered from where the head pointed")
    from fancy_tracker.geometry import DisplayModel, analyse, describe

    # Two panels 1000px wide. In pixel space they adjoin at x=1000, but the
    # synthetic poses put a 6-degree gap between them: a physical separation the
    # OS arrangement does not model.
    def panel_points(x0_deg, deg_per_px, w=1000, h=800, pitch0=0.0, coupling=0.0):
        pts = []
        for px, py in [(0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)]:
            yaw = x0_deg + px * deg_per_px + py * coupling
            pitch = pitch0 + py * 0.01
            pts.append((px, py, np.array([yaw, pitch, yaw / 100.0, pitch / 100.0])))
        return pts

    left = DisplayModel.fit(1, "left", 1000, 800, panel_points(-30.0, 0.02))
    right = DisplayModel.fit(2, "right", 1000, 800, panel_points(-4.0, 0.02))
    check("a plane is fitted per panel", left is not None and right is not None)
    check("fit is near-exact on clean data", left.residual < 1e-6, f"{left.residual:.2e}")
    check(
        "gradient recovered (0.02 deg/px)",
        abs(left.degrees_per_pixel[0] - 0.02) < 1e-6,
        f"{left.degrees_per_pixel[0]:.4f}",
    )

    report = analyse([left, right])
    gap = report["gaps"][0]
    # left spans -30..-10, right spans -4..+16, so the gap is 6 degrees.
    check("physical gap detected", abs(gap.gap_degrees - 6.0) < 1e-6, f"{gap.gap_degrees:.2f} deg")
    check("gap reported as not adjoining", not gap.adjoining)
    # Both panels span 20 degrees, so a 6-degree gap is 0.3x their width. Stated
    # this way it is comparable between monitors; pixels are not, since a dense
    # laptop panel and a 27-inch one disagree about what a pixel is worth.
    check(
        "gap expressed against the narrower neighbour's own width",
        abs(gap.relative_width - 0.3) < 0.01,
        f"{gap.relative_width:.2f}x",
    )
    check("no pixel figure is offered", not hasattr(gap, "equivalent_pixels"))

    touching = DisplayModel.fit(3, "touching", 1000, 800, panel_points(-10.0, 0.02))
    adj = analyse([left, touching])["gaps"][0]
    check("adjoining panels report no gap", adj.adjoining, f"{adj.gap_degrees:.2f} deg")

    # A pair that overlaps in angle is physically impossible, so it has to be
    # reported as a bad fit rather than quietly rounded to "adjoining".
    overlapped = DisplayModel.fit(5, "overlapped", 1000, 800, panel_points(-22.0, 0.02))
    bad = analyse([left, overlapped])["gaps"][0]
    check("an impossible overlap is flagged", bad.overlapping, f"{bad.gap_degrees:.1f} deg")
    check("an overlap is not called adjoining", not bad.adjoining)
    text = describe([left, overlapped])
    check("the report says so in words", "OVERLAP" in text)
    check("the report offers no pixel figure", "px" not in text, text[:0])

    print("17. axis coupling is measured, not fought")
    coupled = DisplayModel.fit(4, "coupled", 1000, 800, panel_points(-30.0, 0.02, coupling=0.005))
    yaw_per_y, _pitch_per_x = coupled.yaw_pitch_coupling
    check("coupling recovered", abs(yaw_per_y - 0.005) < 1e-6, f"{yaw_per_y:.4f} deg/px")
    # Despite the coupling, a pose from the panel still locates back onto it.
    probe = coupled.features_at(750.0, 200.0)
    x, y, residual = coupled.locate(probe)
    check(
        "a coupled pose still maps to the right spot",
        abs(x - 750) < 1.0 and abs(y - 200) < 1.0 and residual < 1e-6,
        f"({x:.0f},{y:.0f}) residual {residual:.2e}",
    )

    print("18. looking into a gap is not attributed to a monitor")
    inside = left.features_at(500.0, 400.0)
    _s, _x, _y, outside_in = left.score(inside)
    check("a pose on the panel is not outside it", outside_in == 0.0)
    # Middle of the gap: left spans -30..-10 deg, right spans -4..+16, so -7 deg
    # belongs to neither. On the left panel's own scale that is x = 1150.
    between = left.features_at(1150.0, 400.0)
    best = min((left, right), key=lambda m: m.score(between)[0])
    outside_gap = best.score(between)[3]
    # 150px beyond an edge at ~0.02 deg/px, so ~3.0 - comfortably over the
    # default --gap-tolerance of 2.0, which is what makes the jump suppressible.
    check(
        "a pose in the gap lands well outside whichever panel claims it",
        outside_gap > 2.0,
        f"outside by {outside_gap:.2f}",
    )
    edge = left.features_at(1000.0, 400.0)  # the panel's own edge, not the gap
    check(
        "the panel's own edge is not treated as a gap",
        left.score(edge)[3] < 1e-6,
        f"outside by {left.score(edge)[3]:.4f}",
    )

    print("19. calibration keeps per-dot data and still reads old files")
    with tempfile.TemporaryDirectory() as td:
        payload = {
            d.id: (
                d.label,
                d.width,
                d.height,
                [
                    (
                        t.name,
                        t.x,
                        t.y,
                        [np.array([i * 20.0 + t.x * 0.01, t.y * 0.01, 0.0, 0.0])] * 12,
                    )
                    for t in targets_for(d)
                ],
            )
            for i, d in enumerate(panels)
        }
        full = Calibration.build_with_targets(payload)
        p = Path(td) / "v2.json"
        full.save(p)
        reloaded = Calibration.load(p)
        check("v2 round-trips with targets", len(reloaded.profiles[0].targets) == 5)
        check("models are buildable from it", len(reloaded.models()) == 2)
        check("classifier goes geometric", Classifier(reloaded).geometric)

        # A v1 file (no targets) must still load and fall back to centroids.
        legacy = Path(td) / "v1.json"
        legacy.write_text(
            json.dumps(
                {
                    "version": 1,
                    "features": list(FEATURE_NAMES),
                    "scale": [1.0] * 4,
                    "profiles": [
                        {
                            "display_id": 9,
                            "label": "a",
                            "mean": [0.0] * 4,
                            "std": [1.0] * 4,
                            "samples": 50,
                        },
                        {
                            "display_id": 8,
                            "label": "b",
                            "mean": [10.0] * 4,
                            "std": [1.0] * 4,
                            "samples": 50,
                        },
                    ],
                }
            )
        )
        old = Calibration.load(legacy)
        check("v1 still loads", old.version == 1)
        check("v1 falls back to centroids", not Classifier(old).geometric)

    print("20. the recalibration prompt is only raised on a real change")
    from fancy_tracker.prompt import describe_change

    wording = describe_change(["new"], [], [])
    check("prompt names what changed", "1 new monitor" in wording, wording.split("\n")[0])
    t4 = Tracker(Classifier(full), Settings(dry_run=True, prompt_on_change=False))
    t4.displays = panels
    matches, added, removed, moved = t4._calibration_matches(panels)
    check("unchanged layout matches", matches, f"{added} {removed} {moved}")
    shrunk = [Display(panels[0].id, 0, 0, 640, 480, False, True), panels[1]]
    matches2, _a, _r, moved2 = t4._calibration_matches(shrunk)
    check("a resized monitor is spotted", not matches2 and moved2, str(moved2))
    gone = [panels[0]]
    matches3, _a3, removed3, _m3 = t4._calibration_matches(gone)
    check("an unplugged monitor is spotted", not matches3 and removed3, str(removed3))

    print("21. constructing a Tracker never puts a dialog on screen")
    import fancy_tracker.tracker as tracker_mod

    raised: list[str] = []

    class SpyQuestion:
        def __init__(self, message, **_kw):
            raised.append(message)
            self._answer: bool | None = None

        def answered(self):
            return self._answer

        def cancel(self):
            pass

    original_question = tracker_mod.Question
    tracker_mod.Question = SpyQuestion
    try:
        # A classifier describing panels that are not the real monitors. Before,
        # this alone raised a dialog; a constructor must not.
        t5 = Tracker(Classifier(full), Settings(dry_run=True, prompt_on_change=True))
        check("no dialog raised by the constructor", raised == [], str(raised))

        # It is raised on the first loop pass instead, and only once per layout.
        t5.displays = panels
        t5._check_layout([Display(4242, 0, 0, 900, 900, False, True)])
        check("dialog raised on a genuine mismatch", len(raised) == 1, str(len(raised)))
        t5._check_layout([Display(4242, 0, 0, 900, 900, False, True)])
        check("not raised again for the same layout", len(raised) == 1, str(len(raised)))

        # Answering yes is what run() acts on.
        check("pending question reports nothing yet", t5._poll_question() is False)
        t5._question._answer = True
        check("confirming asks for a recalibration", t5._poll_question() is True)
        check("question is cleared after answering", t5._question is None)

        # Answering no must not trigger one.
        t5._prompted_for = None
        t5._check_layout([Display(4242, 0, 0, 900, 900, False, True)])
        t5._question._answer = False
        check("declining does not recalibrate", t5._poll_question() is False)

        # And --no-prompt stays silent.
        before = len(raised)
        t6 = Tracker(Classifier(full), Settings(dry_run=True, prompt_on_change=False))
        t6._check_layout([Display(4242, 0, 0, 900, 900, False, True)])
        check("--no-prompt raises nothing", len(raised) == before, str(len(raised)))
    finally:
        tracker_mod.Question = original_question

    print("22. the stillness gate waits for movement before judging stillness")
    from fancy_tracker.calibrate import _await_stillness, _settled

    def window(values, span=1.0):
        return [(i * span / max(len(values) - 1, 1), np.asarray(v)) for i, v in enumerate(values)]

    check("drift between halves is caught", not _settled(window([[0, 0]] * 4 + [[9, 0]] * 4)))
    check("pure noise is not mistaken for drift", _settled(window([[0, 0], [2, 1], [-2, -1]] * 3)))
    check(
        "a window spanning no real time is never settled",
        not _settled(window([[0, 0]] * 8, span=0.001)),
        "eight frames in a millisecond say nothing about stillness",
    )

    class FakePose:
        def __init__(self, yaw):
            self.features = np.array([yaw, 0.0, 0.0, 0.0])

    SWING_START, SWING_END = 0.20, 0.60

    class FakeRig:
        """A head parked on the old dot, then swinging, then settling.

        Driven by the clock rather than a frame count, so the timings under test
        are the ones that actually matter.
        """

        def __init__(self):
            self.t0 = time.monotonic()
            self.frames = 0

        def read(self):
            self.frames += 1
            time.sleep(0.005)  # a camera delivers frames, it does not spin
            return True, np.zeros((4, 4, 3), np.uint8)

        def detect(self, _frame):
            return self  # stands in for a Detection

        @property
        def landmarks(self):
            return np.zeros((5, 2), np.float32)

        def yaw_now(self):
            elapsed = time.monotonic() - self.t0
            if elapsed < SWING_START:
                return 0.0  # still parked on the previous dot
            if elapsed < SWING_END:
                return (elapsed - SWING_START) / (SWING_END - SWING_START) * 50.0
            return 50.0

    rig = FakeRig()

    class FakeOverlay:
        def set_flash(self, _on):
            pass

        def pump(self, _s=0.0):
            pass

    import fancy_tracker.calibrate as cal_mod

    real_estimate = cal_mod.estimate
    cal_mod.estimate = lambda _lm, _w, _h: FakePose(rig.yaw_now())
    try:
        settings_fast = Settings(settle_seconds=0.25)
        t0 = time.monotonic()
        _await_stillness(FakeOverlay(), rig, rig, settings_fast)
        waited = time.monotonic() - t0
        # The old gate returned at ~0.0s, latching onto the stillness of a head
        # that had not started moving, and then sampled through the swing.
        check(
            "it does not open during the pre-movement stillness",
            waited >= settings_fast.settle_seconds,
            f"waited {waited:.2f}s, minimum {settings_fast.settle_seconds}s",
        )
        check(
            "it waits out the whole swing",
            waited > SWING_END,
            f"waited {waited:.2f}s, swing ended at {SWING_END}s",
        )
        check("it settles rather than timing out", waited < 4.0, f"{waited:.2f}s")
    finally:
        cal_mod.estimate = real_estimate

    print("23. a monitor keeps its own corners even when its surface fit is poor")
    # A monitor viewed at a steep angle: its dots do not lie on one plane, so
    # the fitted surface misses its own corners and a neighbour with a tidier
    # fit can claim them. Observed on a real four-monitor desk, where the
    # leftmost panel (67 degrees of head turn) lost two of its five dots.
    steep_dots = [
        ("centre", 500.0, 400.0, np.array([-67.0, 18.0, 1.5, 0.3])),
        ("top-left", 100.0, 700.0, np.array([-79.0, 8.0, 1.9, 0.1])),
        ("top-right", 900.0, 700.0, np.array([-60.0, 10.0, 1.2, 0.1])),
        # These two bend away from the plane the other three define.
        ("bottom-right", 900.0, 100.0, np.array([-57.0, 26.0, 1.0, 0.5])),
        ("bottom-left", 100.0, 100.0, np.array([-72.0, 27.0, 1.8, 0.5])),
    ]
    tidy_dots = [
        ("centre", 500.0, 400.0, np.array([-48.0, 9.0, 0.7, 0.15])),
        ("top-left", 100.0, 700.0, np.array([-60.0, 6.0, 0.9, 0.1])),
        ("top-right", 900.0, 700.0, np.array([-36.0, 6.0, 0.5, 0.1])),
        ("bottom-right", 900.0, 100.0, np.array([-36.0, 12.0, 0.5, 0.2])),
        ("bottom-left", 100.0, 100.0, np.array([-60.0, 12.0, 0.9, 0.2])),
    ]

    def as_payload(did, label, dots):
        return (label, 1000.0, 800.0, [(n, x, y, [m] * 12) for n, x, y, m in dots])

    steep_cal = Calibration.build_with_targets(
        {
            1: as_payload(1, "steep", steep_dots),
            2: as_payload(2, "tidy", tidy_dots),
        }
    )
    steep_clf = Classifier(steep_cal)
    check("this calibration is geometric", steep_clf.geometric)

    misplaced = []
    for p in steep_cal.profiles:
        for t in p.targets:
            if steep_clf.classify(np.asarray(t.mean))[0] != p.display_id:
                misplaced.append(f"{p.label}/{t.name}")
    check("every measured dot lands on its own monitor", not misplaced, ", ".join(misplaced))

    # A pose can land well outside a monitor's fitted surface and still be a
    # real look at that monitor, when the surface is the thing that is wrong.
    # Suppressing there makes that corner unreachable - which is exactly what
    # happened to the top-left of the steeply-viewed monitor on a real desk.
    class StubClassifier:
        """Reports a pose far outside every surface, but right on a measured dot."""

        def __init__(self, outside, nearest):
            self.outside, self.nearest = outside, nearest
            self.calibration = steep_cal

        def locate(self, _f):
            return (panels[1].id, 0.0, 0.0, self.outside)

        def nearest_dot_distance(self, _f):
            return self.nearest

        def label(self, did):
            return str(did)

    def jump_with(outside, nearest):
        t = Tracker(steep_clf, Settings(dry_run=True, prompt_on_change=False, gap_tolerance=2.0))
        t.displays = panels
        t._by_id = {d.id: d for d in panels}
        t.memory = CursorMemory(panels)
        t._streak = 10_000
        t._stable_gaze = panels[0].id
        t._smoothed = np.zeros(4)
        t.classifier = StubClassifier(outside, nearest)
        return t._maybe_jump(panels[1].id)

    check(
        "a look into a real gap is suppressed",
        jump_with(outside=5.0, nearest=5.0) is None,
        "far from every surface and every measured dot",
    )
    check(
        "a look off a bad surface but on a measured dot still jumps",
        jump_with(outside=5.0, nearest=0.0) is not None,
        "this is the unreachable-corner regression",
    )
    check(
        "a look on the surface jumps",
        jump_with(outside=0.0, nearest=0.0) is not None,
    )

    print("24. stickiness that makes a monitor unreachable is detected")
    from fancy_tracker.tuning import Transition, recommend, transitions
    from fancy_tracker.tuning import describe as tuning_describe

    # Two pairs with very different room: steep/tidy are close together, and a
    # third monitor sits far from both. A single stickiness cannot serve both
    # unless it is chosen for the tighter pair, which is the trap.
    far_dots = [(n, x, y, m + np.array([60.0, 0.0, 0.0, 0.0])) for n, x, y, m in tidy_dots]
    spread_cal = Calibration.build_with_targets(
        {
            1: as_payload(1, "steep", steep_dots),
            2: as_payload(2, "tidy", tidy_dots),
            3: as_payload(3, "far", far_dots),
        }
    )
    moves = transitions(spread_cal, margin=0.35)
    check("a move is measured in both directions for every pair", len(moves) == 6, str(len(moves)))

    rooms = {(m.src_label, m.dst_label): m.usable_stickiness for m in moves}
    tightest = min(rooms.values())
    loosest = max(rooms.values())
    check(
        "pairs differ a lot in how much room they have",
        loosest > tightest * 2,
        f"tightest {tightest:.2f}, loosest {loosest:.2f}",
    )

    too_much = loosest - 0.01  # fine for the easy pair, fatal for the tight one
    blocked = [m for m in moves if m.blocked_at(too_much)]
    check(
        "a stickiness tuned for the easy pair blocks the tight one",
        blocked,
        f"{len(blocked)} move(s) blocked at {too_much:.2f}",
    )
    check(
        "the recommendation leaves every move usable",
        not [m for m in moves if m.blocked_at(recommend(moves))],
        f"recommends {recommend(moves)}",
    )
    text = tuning_describe(moves, too_much)
    check("the report names the blocked moves", "BLOCKED" in text)
    check("and suggests a value", "Try --stickiness" in text)

    # A pair that is impossible at any setting must not drag the recommendation
    # to zero and throw away the steadiness the other pairs can afford.
    impossible = [*moves, Transition("a", "b", "corner", lead=0.10, margin=0.35)]
    check(
        "an impossible pair does not veto the recommendation",
        recommend(impossible) == recommend(moves),
        f"{recommend(impossible)} vs {recommend(moves)}",
    )
    check("an impossible pair is still reported as blocked", impossible[-1].blocked_at(0.0))

    print("25. the enlarged cursor is drawn the right way up")
    from AppKit import NSCursor, NSMakeRect

    from fancy_tracker.emphasis import CursorEmphasis, EmphasisView, cg_to_cocoa

    system_cursor = NSCursor.currentSystemCursor()
    if system_cursor is None or system_cursor.image() is None:
        check("no system cursor available to test", True, "skipped")
    else:

        def opaque_centroid(scale):
            """Row/column centre of mass of what gets drawn, in bitmap pixels."""
            side = 240
            view = EmphasisView.alloc().initWithFrame_(NSMakeRect(0, 0, side, side))
            view.image = system_cursor.image()
            hs = system_cursor.hotSpot()
            view.hotspot = (float(hs.x), float(hs.y))
            view.scale, view.halo = scale, 0.0  # halo off; the cursor is the subject
            rep = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
            view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), rep)

            rows, cols, n = 0.0, 0.0, 0
            step = 4
            for py in range(0, rep.pixelsHigh(), step):
                for px in range(0, rep.pixelsWide(), step):
                    if rep.colorAtX_y_(px, py).alphaComponent() > 0.35:
                        rows += py
                        cols += px
                        n += 1
            return (rows / n, cols / n, n) if n else (0.0, 0.0, 0)

        small_row, small_col, small_n = opaque_centroid(1.0)
        big_row, big_col, big_n = opaque_centroid(4.0)
        check("the cursor is actually drawn", small_n > 0 and big_n > 0, f"{small_n} / {big_n} px")
        check("scaling it draws more of it", big_n > small_n * 4, f"{small_n} -> {big_n} px")

        # The hotspot is the arrow's tip and sits at the top-left of the
        # artwork, so the body extends down and to the right of the point it
        # marks. Enlarging must therefore push the centre of mass down-right.
        # Drawn upside down - the bug this catches - it would move up instead.
        check(
            "enlarging extends the cursor downwards, not upwards",
            big_row > small_row,
            f"centre of mass row {small_row:.0f} -> {big_row:.0f} (bitmap rows count down)",
        )
        check(
            "and to the right",
            big_col > small_col,
            f"column {small_col:.0f} -> {big_col:.0f}",
        )

    print("26. the swell runs and then gets out of the way")
    main_bounds = active_displays()
    if main_bounds:
        m = next((d for d in main_bounds if d.main), main_bounds[0])
        _cx, cocoa_y = cg_to_cocoa(m.x, m.y)
        check(
            "the top of the main display maps to its own height",
            abs(cocoa_y - m.height) < 1.0,
            f"cg y={m.y:.0f} -> cocoa y={cocoa_y:.0f}, height {m.height:.0f}",
        )

    em = CursorEmphasis(seconds=0.12, scale=3.0)
    check("nothing is running before it is shown", em.tick() is False)
    em.show(400.0, 300.0)
    ran = em.tick()
    time.sleep(0.2)
    check("it finishes on its own", em.tick() is False, f"was running: {ran}")
    em.close()

    disabled = CursorEmphasis(seconds=0.0)
    disabled.show(400.0, 300.0)
    check("zero seconds means it never appears", disabled.tick() is False)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
