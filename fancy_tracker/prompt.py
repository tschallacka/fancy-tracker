"""Asking a question from a background agent.

The tracker normally runs as a login agent with no terminal attached, so a
question has to be asked through the window server. osascript is the only thing
already on every Mac that can do it, and it is run as a detached subprocess so a
dialog nobody is looking at cannot wedge the tracking loop.
"""

from __future__ import annotations

import shlex
import subprocess

TITLE = "fancy-tracker"


class Question:
    """A dialog asked in the background; poll answered() until it is done."""

    def __init__(self, message: str, confirm: str = "Recalibrate", dismiss: str = "Later"):
        script = (
            f"display dialog {_quote(message)} "
            f'with title "{TITLE}" '
            f"buttons {{{_quote(dismiss)}, {_quote(confirm)}}} "
            f"default button {_quote(confirm)} "
            f"cancel button {_quote(dismiss)} "
            "with icon caution"
        )
        self._confirm = confirm
        try:
            self._process = subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self._process = None  # no osascript; treat as never answered

    @property
    def pending(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def answered(self) -> bool | None:
        """True if confirmed, False if dismissed, None while still open."""
        if self._process is None:
            return False
        code = self._process.poll()
        if code is None:
            return None
        # osascript exits non-zero when the cancel button is used, and prints
        # the chosen button otherwise.
        if code != 0:
            return False
        out = (self._process.stdout.read() if self._process.stdout else "") or ""
        return self._confirm in out

    def cancel(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()


def _quote(text: str) -> str:
    """AppleScript string literal. Quoting is the whole attack surface here."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def notify(message: str) -> None:
    """A non-blocking banner, for things that do not need an answer."""
    script = f"display notification {_quote(message)} with title {_quote(TITLE)}"
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def describe_change(added: list[str], removed: list[str], moved: list[str]) -> str:
    """The wording of the prompt, given what changed about the monitors."""
    parts = []
    if added:
        parts.append(f"{len(added)} new monitor(s)")
    if removed:
        parts.append(f"{len(removed)} monitor(s) gone")
    if moved:
        parts.append(f"{len(moved)} moved or resized")
    summary = ", ".join(parts) if parts else "the monitor layout changed"
    return (
        f"Your monitors have changed: {summary}.\n\n"
        "The saved calibration no longer matches, so head tracking will be "
        "inaccurate until it is redone.\n\nRecalibrate now?"
    )


def shlex_join(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)
