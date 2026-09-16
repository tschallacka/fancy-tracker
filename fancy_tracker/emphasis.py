"""Announce the cursor's arrival with a burst of sparks.

A cursor that teleports is harder to follow than one that was dragged, and on a
busy background it can be lost entirely. A burst centred on the landing point
answers that: the eye is drawn to motion, and motion that converges on a point
says where to look rather than merely that something happened.

So the sparks fly outward and the rings expand, but the embers curl back inward
and the brightest thing on screen is always the centre. Enlarging the cursor
itself was tried first and was redundant once the rings were there.

It is drawn, not applied. The system's own pointer magnification is the
`mouseDriverCursorSize` accessibility preference, which is global and
persistent - a crash midway would leave the cursor stuck large and that setting
silently changed. A click-through window costs nothing if this process dies: the
window goes with it.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSMakeRect,
    NSPoint,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSDate, NSRunLoop
from Quartz import CGDisplayBounds, CGMainDisplayID

BASE_RADIUS = 34.0  # multiplied by `scale` to size the whole burst
SPARKS = 46
EMBERS = 9

# Violet through magenta, with a white core. Ender-ish, but warmer at the
# edges so it reads against both dark and light desktops.
PALETTE = (
    (0.62, 0.36, 1.00),
    (0.80, 0.40, 1.00),
    (0.95, 0.45, 0.95),
    (0.55, 0.65, 1.00),
    (1.00, 0.85, 1.00),
)


def cg_to_cocoa(x: float, y: float) -> tuple[float, float]:
    """CoreGraphics global coordinates to AppKit's.

    Both are global and share an x axis; they disagree about y. CG counts down
    from the top of the main display, AppKit up from its bottom.
    """
    main = CGDisplayBounds(CGMainDisplayID())
    return (x, float(main.size.height) - y)


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    colour: tuple[float, float, float]
    born: float  # fraction of the burst elapsed when it appears
    life: float  # fraction of the burst it lasts
    swirl: float
    twinkle: float
    inward: float  # pulls back to the centre, so the eye ends up there


class SparkView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(SparkView, self).initWithFrame_(frame)  # noqa: PLW0642
        if self is None:
            return None
        self.sparks = []
        self.progress = 0.0
        self.ring_scale = 1.0
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        bounds = self.bounds()
        cx, cy = bounds.size.width / 2.0, bounds.size.height / 2.0
        t = self.progress

        # Ordinary source-over, not PlusLighter. Adding light is invisible on a
        # light desktop, and this has to work on both, so every bright mark is
        # laid over a darker one of its own instead.
        self._flash(cx, cy, t)
        self._rings(cx, cy, t)
        for spark in self.sparks:
            self._spark(cx, cy, spark, t)

    @objc.python_method
    def _flash(self, cx, cy, t):
        """A brief bloom at the centre, so the eye is told where before it is told what."""
        if t > 0.22:
            return
        local = t / 0.22
        radius = self.ring_scale * (0.10 + 0.22 * local)
        alpha = (1.0 - local) ** 2
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.65, 1.0, alpha * 0.55).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
        ).fill()

    @objc.python_method
    def _stroke_twice(self, path, width, colour, alpha):
        """A dark pass under a bright one, so the mark survives any background."""
        r, g, b = colour
        path.setLineWidth_(width + 1.6)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.02, 0.16, alpha * 0.55).set()
        path.stroke()
        path.setLineWidth_(width)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, alpha).set()
        path.stroke()

    @objc.python_method
    def _rings(self, cx, cy, t):
        """Two shockwaves, the second a beat behind, so the burst has depth.

        They start small and are overtaken by the sparks early on, which is the
        way round an explosion actually looks.
        """
        for delay, weight, width in ((0.0, 1.0, 2.6), (0.18, 0.5, 1.4)):
            local = (t - delay) / (1.0 - delay) if t > delay else -1.0
            if not 0.0 <= local <= 1.0:
                continue
            radius = self.ring_scale * (0.06 + 1.05 * _ease_out(local))
            alpha = (1.0 - local) ** 1.8 * weight
            path = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
            )
            self._stroke_twice(path, width + 1.5 * (1.0 - local), (0.78, 0.52, 1.0), alpha * 0.9)

    @objc.python_method
    def _spark(self, cx, cy, s: Spark, t: float):
        local = (t - s.born) / s.life
        if not 0.0 <= local <= 1.0:
            return

        # Outward hard and decelerating, curling as it goes, easing back towards
        # the centre late so the burst settles onto the cursor rather than
        # abandoning it.
        travel = _ease_out(local)
        angle = math.atan2(s.vy, s.vx) + s.swirl * travel
        speed = math.hypot(s.vx, s.vy) * (1.0 - s.inward * local**3)
        px = cx + s.x + math.cos(angle) * speed * travel
        py = cy + s.y + math.sin(angle) * speed * travel

        fade = (1.0 - local) ** 1.3
        flicker = 0.7 + 0.3 * math.sin(local * s.twinkle)
        alpha = fade * flicker
        size = s.size * (0.5 + 0.5 * (1.0 - local))

        # A trail in the direction of travel reads as speed at any frame rate,
        # where a bare dot only ever reads as a dot. Its length follows the
        # spark's own speed, so the slow ones stay motes and only the fast ones
        # streak - uniform trails are what made this look like a dial.
        pace = min(1.0, math.hypot(s.vx, s.vy) / max(self.ring_scale, 1.0))
        tail = pace * (8.0 + 34.0 * (1.0 - local))
        trail = NSBezierPath.bezierPath()
        trail.setLineCapStyle_(1)
        trail.moveToPoint_(NSPoint(px - math.cos(angle) * tail, py - math.sin(angle) * tail))
        trail.lineToPoint_(NSPoint(px, py))
        self._stroke_twice(trail, max(1.0, size * 0.7), s.colour, alpha * 0.7)

        r, g, b = s.colour
        outer = size + 1.8
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.02, 0.16, alpha * 0.5).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(px - outer / 2.0, py - outer / 2.0, outer, outer)
        ).fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            min(1.0, r + 0.25), min(1.0, g + 0.25), min(1.0, b + 0.25), alpha
        ).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(px - size / 2.0, py - size / 2.0, size, size)
        ).fill()


def build_sparks(scale: float, rng: random.Random) -> list[Spark]:
    """One burst's worth of sparks, plus slower embers that linger."""
    reach = BASE_RADIUS * scale
    out: list[Spark] = []

    for i in range(SPARKS):
        # Spread evenly and then jitter hard. Even spacing alone leaves visible
        # spokes, and the speeds matter more than the angles: sparks that all
        # travel the same distance land on one circle and read as a clock face
        # rather than an explosion, so the range here is deliberately wide.
        angle = (i / SPARKS) * math.tau + rng.uniform(-0.55, 0.55)
        speed = reach * rng.triangular(0.12, 1.35, 0.45)
        out.append(
            Spark(
                x=math.cos(angle) * reach * rng.uniform(0.0, 0.12),
                y=math.sin(angle) * reach * rng.uniform(0.0, 0.12),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=rng.uniform(2.2, 7.5),
                colour=rng.choice(PALETTE),
                born=rng.uniform(0.0, 0.20),
                life=rng.uniform(0.32, 1.0),
                swirl=rng.uniform(-1.7, 1.7),
                twinkle=rng.uniform(8.0, 26.0),
                inward=rng.uniform(0.05, 0.25),
            )
        )

    for _ in range(EMBERS):
        angle = rng.uniform(0.0, math.tau)
        out.append(
            Spark(
                x=0.0,
                y=0.0,
                vx=math.cos(angle) * reach * rng.uniform(0.15, 0.4),
                vy=math.sin(angle) * reach * rng.uniform(0.15, 0.4),
                size=rng.uniform(1.4, 2.6),
                colour=PALETTE[-1],
                born=rng.uniform(0.05, 0.3),
                life=rng.uniform(0.6, 1.0),
                swirl=rng.uniform(-2.2, 2.2),
                twinkle=rng.uniform(14.0, 30.0),
                inward=rng.uniform(0.5, 0.9),
            )
        )
    return out


class CursorEmphasis:
    """A one-shot burst at a point, advanced by tick() from the caller's loop."""

    def __init__(self, seconds: float = 0.9, scale: float = 2.0):
        self.seconds = seconds
        self.scale = scale
        self.canvas = max(360.0, BASE_RADIUS * scale * 2.9)
        self._window = None
        self._view = None
        self._rng = random.Random()
        self._started = 0.0
        self._active = False

    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.canvas, self.canvas),
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
        view = SparkView.alloc().initWithFrame_(NSMakeRect(0, 0, self.canvas, self.canvas))
        view.ring_scale = BASE_RADIUS * self.scale
        window.setContentView_(view)
        self._window, self._view = window, view

    def show(self, cg_x: float, cg_y: float) -> None:
        """Begin a burst centred on a point in CoreGraphics global coordinates."""
        if self.seconds <= 0.0:
            return
        try:
            self._ensure_window()
            self._view.sparks = build_sparks(self.scale, self._rng)
            self._view.progress = 0.0
            x, y = cg_to_cocoa(cg_x, cg_y)
            self._window.setFrameOrigin_((x - self.canvas / 2.0, y - self.canvas / 2.0))
            self._window.orderFrontRegardless()
        except Exception:  # noqa: BLE001 - decoration must never break tracking
            self._active = False
            return
        self._started = time.monotonic()
        self._active = True
        self.tick()

    def tick(self) -> bool:
        """Advance the burst. Returns True while it is still running."""
        if not self._active or self._window is None:
            return False

        elapsed = (time.monotonic() - self._started) / self.seconds
        if elapsed >= 1.0:
            self._finish()
            return False

        self._view.progress = elapsed
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
