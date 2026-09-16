"""Full-screen calibration targets drawn as borderless Cocoa overlays.

One transparent, click-through window per display, each showing five dots: the
four corners and the centre. The dot being calibrated flashes; the rest stay
dim. Sampling the corners as well as the centre matters - a 2560px-wide monitor
subtends a wide angle from where you sit, so a profile built only from its
centre would be far too tight to recognise a glance at its edge.

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
    """Draws the dim backdrop, the five dots, and a caption."""

    def initWithFrame_(self, frame):
        # Reassigning self is the PyObjC initialiser idiom: the superclass may
        # return a different instance, and that one is the object to carry on with.
        self = objc.super(TargetView, self).initWithFrame_(frame)  # noqa: PLW0642
        if self is None:
            return None
        self.targets = []
        self.active_index = -1
        self.flash_on = True
        self.caption = ""
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        NSColor.colorWithCalibratedWhite_alpha_(0.0, SCREEN_DIM).set()
        NSBezierPath.fillRect_(self.bounds())

        for i, target in enumerate(self.targets):
            active = i == self.active_index
            if active and not self.flash_on:
                continue
            if active:
                NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.85, 0.1, 1.0).set()
                radius = DOT_RADIUS
            else:
                NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.18).set()
                radius = DOT_RADIUS * 0.45

            path = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(target.x - radius, target.y - radius, radius * 2, radius * 2)
            )
            path.fill()

            if active:
                NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.9).set()
                ring = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(
                        target.x - radius * 1.8,
                        target.y - radius * 1.8,
                        radius * 3.6,
                        radius * 3.6,
                    )
                )
                ring.setLineWidth_(3.0)
                ring.stroke()


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
            view.targets = targets_for(display)
            window.setContentView_(view)
            window.orderFrontRegardless()

            self._windows[display.id] = (window, view)
            self.targets[display.id] = view.targets

        self.pump()

    def show_target(self, display_id: int, index: int) -> None:
        """Light the given dot on one display, dim every other display."""
        for did, (_window, view) in self._windows.items():
            view.active_index = index if did == display_id else -1
            view.setNeedsDisplay_(True)
        self.pump()

    def set_flash(self, on: bool) -> None:
        for _window, view in self._windows.values():
            if view.active_index >= 0:
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
