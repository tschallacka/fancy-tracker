"""Display enumeration and cursor control through Quartz.

Everything here works in macOS global display space: origin at the top-left of
the main display, y increasing downwards. CGDisplayBounds, CGEventGetLocation
and CGWarpMouseCursorPosition all agree on that space, so no flipping is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from Quartz import (
    CGAssociateMouseAndMouseCursorPosition,
    CGDisplayBounds,
    CGDisplayIsBuiltin,
    CGEventCreate,
    CGEventGetLocation,
    CGGetActiveDisplayList,
    CGMainDisplayID,
    CGWarpMouseCursorPosition,
)

MAX_DISPLAYS = 16


@dataclass(frozen=True)
class Display:
    id: int
    x: float
    y: float
    width: float
    height: float
    builtin: bool
    main: bool

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def clamp(self, px: float, py: float) -> tuple[float, float]:
        """Nearest point inside this display, kept a pixel off the far edges."""
        cx = min(max(px, self.x), self.x + self.width - 1)
        cy = min(max(py, self.y), self.y + self.height - 1)
        return (cx, cy)

    @property
    def label(self) -> str:
        tags = []
        if self.main:
            tags.append("main")
        if self.builtin:
            tags.append("built-in")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        return f"{self.width:.0f}x{self.height:.0f} @ ({self.x:.0f},{self.y:.0f}){suffix}"


def active_displays() -> list[Display]:
    """Active displays, ordered left to right by their global x origin."""
    err, ids, count = CGGetActiveDisplayList(MAX_DISPLAYS, None, None)
    if err != 0:
        raise RuntimeError(f"CGGetActiveDisplayList failed with error {err}")

    main = CGMainDisplayID()
    out = []
    for did in list(ids)[:count]:
        b = CGDisplayBounds(did)
        out.append(
            Display(
                id=int(did),
                x=float(b.origin.x),
                y=float(b.origin.y),
                width=float(b.size.width),
                height=float(b.size.height),
                builtin=bool(CGDisplayIsBuiltin(did)),
                main=(int(did) == int(main)),
            )
        )
    out.sort(key=lambda d: d.x)
    return out


def cursor_position() -> tuple[float, float]:
    loc = CGEventGetLocation(CGEventCreate(None))
    return (float(loc.x), float(loc.y))


def warp_cursor(x: float, y: float) -> None:
    """Move the cursor without needing Accessibility permission.

    CGWarpMouseCursorPosition suppresses local mouse input for about a quarter
    second afterwards, which reads as the mouse briefly going dead. Re-associating
    immediately cancels that suppression.
    """
    CGWarpMouseCursorPosition((x, y))
    CGAssociateMouseAndMouseCursorPosition(True)


def display_at(displays: list[Display], x: float, y: float) -> Display | None:
    for d in displays:
        if d.contains(x, y):
            return d
    return None
