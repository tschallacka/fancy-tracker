from __future__ import annotations

import argparse
import sys
import time

from .calibration import Calibration, Classifier, calibration_path
from .displays import active_displays, cursor_position, display_at
from .tracker import Settings, Tracker, positions_path


def _camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=0, help="capture device index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="face detection confidence floor; lower it if steep angles lose the face",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fancy-tracker",
        description="Remember a cursor position per monitor and jump to it when you look there.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("displays", help="list the displays as macOS reports them")

    cal = sub.add_parser("calibrate", help="record what each display looks like")
    _camera_args(cal)
    cal.add_argument("--blink-hz", type=float, default=2.0, help="how fast the target dot blinks")
    cal.add_argument(
        "--settle", type=float, default=1.8, help="seconds to find the dot before sampling"
    )
    cal.add_argument("--sample", type=float, default=1.5, help="seconds of sampling per dot")

    diag = sub.add_parser("diag", help="live pose readout; never moves the cursor")
    _camera_args(diag)

    run = sub.add_parser("run", help="track and move the cursor")
    _camera_args(run)
    run.add_argument("--dwell", type=int, default=6, help="agreeing frames before a gaze counts")
    run.add_argument("--margin", type=float, default=0.35, help="required lead over runner-up")
    run.add_argument("--smoothing", type=float, default=0.35, help="EMA weight, 0-1")
    run.add_argument("--cooldown", type=float, default=0.6, help="seconds between jumps")
    run.add_argument(
        "--stickiness",
        type=float,
        default=0.5,
        help="head start for the display you are on; raise it if it flips between two",
    )
    run.add_argument(
        "--mouse-grace",
        type=float,
        default=0.5,
        help="hold off this long after you move the mouse yourself",
    )
    run.add_argument(
        "--recall-position",
        action="store_true",
        help="jump to where the cursor last was on that monitor, instead of its centre",
    )
    run.add_argument(
        "--gap-tolerance",
        type=float,
        default=2.0,
        help="how far off every monitor a look may land before it counts as between them",
    )
    run.add_argument(
        "--no-prompt",
        action="store_true",
        help="do not offer a recalibration when the monitor layout changes",
    )
    run.add_argument("--preview", action="store_true", help="show the camera window")
    run.add_argument("--dry-run", action="store_true", help="log jumps without making them")
    return parser


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        camera=args.camera,
        width=args.width,
        height=args.height,
        min_score=args.min_score,
        smoothing=getattr(args, "smoothing", 0.35),
        dwell=getattr(args, "dwell", 6),
        margin=getattr(args, "margin", 0.35),
        cooldown=getattr(args, "cooldown", 0.6),
        mouse_grace=getattr(args, "mouse_grace", 0.5),
        stickiness=getattr(args, "stickiness", 0.5),
        gap_tolerance=getattr(args, "gap_tolerance", 2.0),
        prompt_on_change=not getattr(args, "no_prompt", False),
        dry_run=getattr(args, "dry_run", False),
        preview=getattr(args, "preview", False),
        recall_position=getattr(args, "recall_position", False),
        blink_hz=getattr(args, "blink_hz", 2.0),
        settle_seconds=getattr(args, "settle", 1.8),
        sample_seconds=getattr(args, "sample", 1.5),
    )


def cmd_displays() -> int:
    displays = active_displays()
    x, y = cursor_position()
    here = display_at(displays, x, y)
    print(f"{len(displays)} active displays (left to right):")
    for i, d in enumerate(displays, start=1):
        mark = " <- cursor" if here and here.id == d.id else ""
        print(f"  {i}. id={d.id:<4} {d.label}{mark}")
    print(f"\ncursor at ({x:.0f}, {y:.0f})")
    print(f"calibration: {calibration_path()}")
    print(f"positions:   {positions_path()}")
    return 0


def cmd_diag(settings: Settings) -> int:
    from .detector import FaceDetector
    from .pose import estimate
    from .tracker import open_camera

    classifier = None
    try:
        classifier = Classifier(Calibration.load())
    except (FileNotFoundError, ValueError) as exc:
        print(f"({exc})\nShowing raw pose only.\n", file=sys.stderr)

    detector = FaceDetector(score_threshold=settings.min_score)
    cap = open_camera(settings)
    print("ctrl-c to stop\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            detection = detector.detect(frame)
            if detection is None:
                print("\rno face                                             ", end="", flush=True)
                continue
            pose = estimate(detection.landmarks, frame.shape[1], frame.shape[0])
            if pose is None:
                continue
            line = (
                f"yaw {pose.yaw:+7.1f}  pitch {pose.pitch:+7.1f}  "
                f"dx {pose.nose_dx:+5.2f}  dy {pose.nose_dy:+5.2f}"
            )
            if classifier is not None:
                did, dist, margin = classifier.classify(pose.features)
                line += f"  -> {classifier.label(did)}  (d {dist:.2f}, margin {margin:.2f})"
            print(f"\r{line}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        cap.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "displays":
            return cmd_displays()

        settings = settings_from(args)
        if args.command == "calibrate":
            from . import calibrate

            return calibrate.run(settings)
        if args.command == "diag":
            return cmd_diag(settings)
        if args.command == "run":
            return Tracker(Classifier(Calibration.load()), settings).run()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
