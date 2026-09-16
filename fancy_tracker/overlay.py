"""Full-screen calibration targets drawn as borderless Cocoa overlays.

One transparent, click-through window per display, each showing five dots: the
four corners and the centre. Sampling the corners as well as the centre matters
- a 2560px-wide monitor subtends a wide angle from where you sit, so a profile
built only from its centre would be far too tight to recognise a glance at its
edge.

Each dot carries its own state, so the whole run is legible at a glance: the
dot being sampled pulses yellow, a good one turns green with a tick, one whose
samples were unusable flashes red before going back to yellow to be retried,
and one queued for another look because it sits too close to a dot already
recorded is ringed amber.

AppKit needs its run loop pumped to redraw. Rather than hand control to
NSApplication.run(), the capture loop calls pump() each frame, which keeps the
camera loop in charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSScreen,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSDate, NSMakeRect, NSRunLoop

from .displays import Display

# Corner dots sit this far in from the edges, so they are never clipped and
# never sit under the menu bar or the Dock.
INSET = 90.0
DOT_RADIUS = 26.0
SCREEN_DIM = 0.72

TARGET_NAMES = ("centre", "top-left", "top-right", "bottom-right", "bottom-left")

# Dot states.
PENDING = "pending"
ACTIVE = "active"
GOOD = "good"
BAD = "bad"
REVISIT = "revisit"


@dataclass(frozen=True)
class Target:
    """A calibration dot, in the display's own local coordinates (y up)."""

    name: str
    x: float
    y: float


def targets_for(display: Display) -> list[Target]:
    w, h = display.width, display.height
    inset_x = min(INSET, w / 2.0 - DOT_RADIUS)
    inset_y = min(INSET, h / 2.0 - DOT_RADIUS)
    return [
        Target("centre", w / 2.0, h / 2.0),
        Target("top-left", inset_x, h - inset_y),
        Target("top-right", w - inset_x, h - inset_y),
        Target("bottom-right", w - inset_x, inset_y),
        Target("bottom-left", inset_x, inset_y),
    ]


class TargetView(NSView):
    """Draws the dim backdrop and the five dots in their current states."""

    def initWithFrame_(self, frame):
        # Reassigning self is the PyObjC initialiser idiom: the superclass may
        # return a different instance, and that one is the object to carry on with.
        self = objc.super(TargetView, self).initWithFrame_(frame)  # noqa: PLW0642
        if self is None:
            return None
        self.targets = []
        self.states = []
        self.flash_on = True
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        NSColor.colorWithCalibratedWhite_alpha_(0.0, SCREEN_DIM).set()
        NSBezierPath.fillRect_(self.bounds())

        for i, target in enumerate(self.targets):
            state = self.states[i] if i < len(self.states) else PENDING
            if state == PENDING:
                self._dot(target, DOT_RADIUS * 0.45, (1.0, 1.0, 1.0, 0.18))
            elif state == GOOD:
                self._dot(target, DOT_RADIUS * 0.8, (0.20, 0.80, 0.35, 1.0))
                self._tick(target, DOT_RADIUS * 0.8)
            elif state == REVISIT:
                self._dot(target, DOT_RADIUS * 0.55, (1.0, 0.60, 0.0, 0.85))
                self._ring(target, DOT_RADIUS * 0.55, (1.0, 0.60, 0.0, 0.9))
            elif state == BAD:
                # Only the flash-on half is drawn, so this reads as a blink.
                if self.flash_on:
                    self._dot(target, DOT_RADIUS, (0.95, 0.20, 0.20, 1.0))
                    self._ring(target, DOT_RADIUS, (1.0, 0.45, 0.45, 0.9))
            elif state == ACTIVE and self.flash_on:
                self._dot(target, DOT_RADIUS, (1.0, 0.85, 0.10, 1.0))
                self._ring(target, DOT_RADIUS, (1.0, 1.0, 1.0, 0.9))

    # PyObjC turns every method on an NSView subclass into an ObjC selector, and
    # derives the expected argument count from the name. These helpers are plain
    # Python and have to say so, or the class fails to build.
    @objc.python_method
    def _dot(self, target, radius, rgba):
        NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(target.x - radius, target.y - radius, radius * 2, radius * 2)
        ).fill()

    @objc.python_method
    def _ring(self, target, radius, rgba):
        NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba).set()
        outer = radius * 1.8
        path = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(target.x - outer, target.y - outer, outer * 2, outer * 2)
        )
        path.setLineWidth_(3.0)
        path.stroke()

    @objc.python_method
    def _tick(self, target, radius):
        """A check mark inside the dot. Cocoa's y axis points up."""
        NSColor.colorWithCalibratedWhite_alpha_(1.0, 1.0).set()
        path = NSBezierPath.bezierPath()
        path.setLineWidth_(max(3.0, radius * 0.26))
        path.setLineCapStyle_(1)  # round
        path.setLineJoinStyle_(1)
        path.moveToPoint_((target.x - radius * 0.42, target.y + radius * 0.05))
        path.lineToPoint_((target.x - radius * 0.12, target.y - radius * 0.30))
        path.lineToPoint_((target.x + radius * 0.45, target.y + radius * 0.38))
        path.stroke()


class CalibrationOverlay:
    """Overlay windows for every display, driven target by target."""

    def __init__(self, displays: list[Display]):
        self._app = NSApplication.sharedApplication()
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        screens = {}
        for screen in NSScreen.screens():
            number = screen.deviceDescription().get("NSScreenNumber")
            if number is not None:
                screens[int(number)] = screen

        self._windows: dict[int, tuple[NSWindow, TargetView]] = {}
        self.targets: dict[int, list[Target]] = {}

        for display in displays:
            screen = screens.get(display.id)
            if screen is None:
                continue  # display vanished between enumeration and now
            frame = screen.frame()
            # initWithContentRect:...screen: reads the rect in THAT screen's own
            # coordinate space, so passing a global frame here double-offsets every
            # display whose origin is not (0,0) and throws them off-screen. Build at
            # the origin, then place with setFrame:, which is always global.
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_screen_(
                NSMakeRect(0, 0, frame.size.width, frame.size.height),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
                screen,
            )
            window.setFrameOrigin_(frame.origin)
            window.setOpaque_(False)
            window.setBackgroundColor_(NSColor.clearColor())
            window.setIgnoresMouseEvents_(True)
            window.setLevel_(25)  # above normal windows, below the screen saver
            window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            view = TargetView.alloc().initWithFrame_(
                NSMakeRect(0, 0, frame.size.width, frame.size.height)
            )
            targets = targets_for(display)
            view.targets = targets
            view.states = [PENDING] * len(targets)
            window.setContentView_(view)
            window.orderFrontRegardless()

            self._windows[display.id] = (window, view)
            self.targets[display.id] = targets

        self.pump()

    def state(self, display_id: int, index: int) -> str:
        _window, view = self._windows[display_id]
        return view.states[index]

    def set_state(self, display_id: int, index: int, state: str) -> None:
        entry = self._windows.get(display_id)
        if entry is None:
            return
        _window, view = entry
        view.states[index] = state
        view.setNeedsDisplay_(True)
        self.pump()

    def set_active(self, display_id: int, index: int) -> None:
        """Make one dot the active one, leaving every finished dot as it is."""
        for did, (_window, view) in self._windows.items():
            for i, state in enumerate(view.states):
                if state in (ACTIVE, BAD) and not (did == display_id and i == index):
                    view.states[i] = PENDING
            if did == display_id:
                view.states[index] = ACTIVE
            view.setNeedsDisplay_(True)
        self.pump()

    def set_flash(self, on: bool) -> None:
        for _window, view in self._windows.values():
            view.flash_on = on
            view.setNeedsDisplay_(True)
        self.pump()

    def pump(self, seconds: float = 0.005) -> None:
        """Let AppKit draw. Must be called regularly or nothing appears."""
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))

    def close(self) -> None:
        for window, _view in self._windows.values():
            window.orderOut_(None)
        self._windows.clear()
        self.pump()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
