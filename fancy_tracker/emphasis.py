"""Make the cursor findable at the moment it moves.

A cursor that teleports is harder to follow than one that was dragged there,
and on a busy background it can be lost entirely. macOS already answers this
with shake-to-locate, so this borrows the same vocabulary: the cursor swells to
roughly the size a shake gives it, holds for an instant, then settles back.
Someone who has ever shaken a mouse recognises it without being told.

It is drawn rather than applied. The system's own pointer magnification is the
`mouseDriverCursorSize` accessibility preference, which is global and
persistent - a crash midway through an animation would leave the cursor stuck
large and the setting silently changed. A click-through window borrowing the
current cursor's own artwork costs nothing if this process dies: the window
goes with it.
"""

from __future__ import annotations

import time

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSCompositingOperationSourceOver,
    NSCursor,
    NSMakeRect,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSDate, NSRunLoop
from Quartz import CGDisplayBounds, CGMainDisplayID

# Big enough to hold the cursor at full magnification with room for the halo.
CANVAS = 420.0

GROW_FRACTION = 0.14  # of the total, spent swelling
HOLD_FRACTION = 0.18  # then held, so the eye has time to land on it


def cg_to_cocoa(x: float, y: float) -> tuple[float, float]:
    """CoreGraphics global coordinates to AppKit's.

    Both are global and share an x axis; they disagree about y. CG counts down
    from the top of the main display, AppKit up from its bottom.
    """
    main = CGDisplayBounds(CGMainDisplayID())
    return (x, float(main.size.height) - y)


def _ease_out(t: float) -> float:
    """Fast at first, slow at the end - how a thing settles rather than stops."""
    return 1.0 - (1.0 - t) * (1.0 - t)


class EmphasisView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(EmphasisView, self).initWithFrame_(frame)  # noqa: PLW0642
        if self is None:
            return None
        self.image = None
        self.hotspot = (0.0, 0.0)
        self.scale = 1.0
        self.halo = 0.0
        return self

    def isOpaque(self):
        return False

    def isFlipped(self):
        # Not flipped. An NSImage drawn into a flipped context comes out
        # mirrored, which stands the arrow on its head and throws the hotspot to
        # the wrong corner, so the y arithmetic below is done by hand instead.
        return False

    def drawRect_(self, rect):
        if self.image is None:
            return
        cx, cy = self.bounds().size.width / 2.0, self.bounds().size.height / 2.0
        size = self.image.size()
        w, h = size.width * self.scale, size.height * self.scale

        if self.halo > 0.0:
            # A ring, not a disc: this sits on top of whatever you were trying
            # to look at, so it has to draw the eye without hiding anything.
            radius = max(w, h) * 0.7
            ring = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
            )
            ring.setLineWidth_(3.0)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, self.halo * 0.35).set()
            ring.stroke()
            ring.setLineWidth_(1.5)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.9, 0.3, self.halo * 0.9).set()
            ring.stroke()

        # The hotspot is measured from the top-left of the artwork, and the draw
        # rect is anchored at its bottom-left, so the vertical offset is the
        # image height less the hotspot.
        hx = self.hotspot[0] * self.scale
        hy = (size.height - self.hotspot[1]) * self.scale
        self.image.drawInRect_fromRect_operation_fraction_(
            NSMakeRect(cx - hx, cy - hy, w, h),
            ((0, 0), (0, 0)),
            NSCompositingOperationSourceOver,
            1.0,
        )


class CursorEmphasis:
    """A one-shot swell of the cursor, advanced by tick() from the caller's loop."""

    def __init__(self, seconds: float = 1.1, scale: float = 4.0):
        self.seconds = seconds
        self.scale = scale
        self._window = None
        self._view = None
        self._started = 0.0
        self._active = False

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, CANVAS, CANVAS),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setIgnoresMouseEvents_(True)
        window.setLevel_(25)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
        )
        view = EmphasisView.alloc().initWithFrame_(NSMakeRect(0, 0, CANVAS, CANVAS))
        window.setContentView_(view)
        self._window, self._view = window, view

    def show(self, cg_x: float, cg_y: float) -> None:
        """Begin the swell, centred on a point in CoreGraphics global coordinates."""
        if self.seconds <= 0.0:
            return
        try:
            cursor = NSCursor.currentSystemCursor()
            if cursor is None or cursor.image() is None:
                return
            self._ensure_window()
            self._view.image = cursor.image()
            hs = cursor.hotSpot()
            self._view.hotspot = (float(hs.x), float(hs.y))

            x, y = cg_to_cocoa(cg_x, cg_y)
            self._window.setFrameOrigin_((x - CANVAS / 2.0, y - CANVAS / 2.0))
            self._window.orderFrontRegardless()
        except Exception:  # noqa: BLE001 - decoration must never break tracking
            self._active = False
            return
        self._started = time.monotonic()
        self._active = True
        self.tick()

    def tick(self) -> bool:
        """Advance the animation. Returns True while it is still running."""
        if not self._active or self._window is None:
            return False

        elapsed = (time.monotonic() - self._started) / self.seconds
        if elapsed >= 1.0:
            self._finish()
            return False

        grow_end = GROW_FRACTION
        hold_end = GROW_FRACTION + HOLD_FRACTION
        if elapsed < grow_end:
            progress = _ease_out(elapsed / grow_end)
            scale = 1.0 + (self.scale - 1.0) * progress
            halo = progress
        elif elapsed < hold_end:
            scale, halo = self.scale, 1.0
        else:
            progress = _ease_out((elapsed - hold_end) / (1.0 - hold_end))
            scale = self.scale - (self.scale - 1.0) * progress
            halo = 1.0 - progress

        self._view.scale = scale
        self._view.halo = halo
        self._view.setNeedsDisplay_(True)
        self._pump()
        return True

    def _finish(self) -> None:
        self._active = False
        if self._window is not None:
            self._window.orderOut_(None)
        self._pump()

    def _pump(self) -> None:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

    def close(self) -> None:
        self._finish()
        self._window = None
        self._view = None
