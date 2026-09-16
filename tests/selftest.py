"""Everything that can be checked without a live camera.

Run inside the dev shell, which supplies both the model and the fixture face:

    nix develop --command python tests/selftest.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fancy_tracker.calibrate import target_to_global
from fancy_tracker.calibration import Calibration, Classifier
from fancy_tracker.detector import FaceDetector, model_path
from fancy_tracker.displays import Display, active_displays
from fancy_tracker.overlay import targets_for
from fancy_tracker.pose import estimate
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

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
