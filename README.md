# fancy-tracker

Moves the mouse cursor to the middle of whichever monitor you turn your head
towards.

The centre is deliberate: it is the same spot every time, so you know where the
cursor will be before it arrives. Restoring the position it was last left at
sounds more helpful but is harder to reacquire, because the landing spot moves.
`--recall-position` turns that older behaviour back on.

macOS only. It needs a webcam and does not need the camera to be centred in
front of you — a camera off to one side is handled by calibration rather than
by geometry.

## How it works

1. **Where the monitors are** comes from macOS itself (`CGDisplayBounds`), so
   the arrangement you set in System Settings → Displays → Arrangement is what
   the program uses. Nothing is hardcoded or guessed.
2. **Where you are looking** is inferred from your head, not your eyes. OpenCV's
   bundled YuNet detector finds your face and five landmarks; `solvePnP` turns
   those into yaw and pitch.
3. **Which monitor that means** is decided by calibration, not trigonometry.
   Because the camera is off to one side there is no clean geometric mapping, so
   the program instead records what the pose actually looks like while you face
   each monitor, and later picks whichever recorded profile the live pose most
   resembles. Any fixed camera offset is absorbed into those profiles.
4. **The cursor** is moved with `CGWarpMouseCursorPosition`, which needs no
   Accessibility permission.

## Setup

Install Nix, if you have not already — this installer turns on flakes by
default, which this project needs:

```sh
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

Camera access has to be granted to whatever terminal you run this from:
**System Settings → Privacy & Security → Camera**. Without it the program stops
with `Could not open camera 0`.

```sh
nix run github:tschallacka/fancy-tracker -- displays     # check the monitors
nix run github:tschallacka/fancy-tracker -- calibrate    # roughly a minute
nix run github:tschallacka/fancy-tracker -- run
```

From a local clone, `nix run .#fancy-tracker -- …` does the same.

## Running it at login

Install it into your profile first, so there is a stable path for launchd to
call and the build cannot be garbage-collected out from under it:

```sh
nix profile add github:tschallacka/fancy-tracker
```

On Nix older than 2.28 that subcommand is still called `nix profile install`.

Then write a launch agent. The path has to be absolute — launchd expands
neither `~` nor environment variables — so let the shell fill it in:

```sh
mkdir -p ~/Library/LaunchAgents ~/Library/Logs
cat > ~/Library/LaunchAgents/nl.tschallacka.fancy-tracker.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>nl.tschallacka.fancy-tracker</string>
  <key>ProgramArguments</key>
  <array>
    <string>$HOME/.nix-profile/bin/fancy-tracker</string>
    <string>run</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/fancy-tracker.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/fancy-tracker.log</string>
</dict>
</plist>
PLIST

launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/nl.tschallacka.fancy-tracker.plist
```

It starts at login from then on. To check on it, stop it, or remove it:

```sh
launchctl print gui/$(id -u)/nl.tschallacka.fancy-tracker | head -20
tail -f ~/Library/Logs/fancy-tracker.log
launchctl bootout gui/$(id -u)/nl.tschallacka.fancy-tracker
```

Three things that will bite otherwise:

- **Calibrate before you enable the agent.** With no calibration the program
  exits immediately, and `KeepAlive` will restart it forever. `ThrottleInterval`
  holds that to once every 30 seconds rather than a spin, but the log will fill
  with the same error until you calibrate.
- **Camera permission is per-executable.** Granting it to your terminal does not
  grant it to the launchd-started process, which macOS treats separately. It
  should appear in **Privacy & Security → Camera** the first time the agent
  tries; if it never does, the log will show `Could not open camera 0`.
- **`nix profile upgrade` is what updates it.** The agent runs the copy in your
  profile, not your working tree, so editing the checkout changes nothing until
  you reinstall.


## Calibrating

Every monitor dims and shows five dots — its four corners and its centre. The
dot to look at blinks, then goes solid; look straight at it and hold still
while it is solid. The cursor is parked on the dot as a second cue.

All five points feed one profile per monitor, on purpose. A 2560px-wide monitor
covers a wide angle from where you sit, so a profile built only from its centre
would not recognise a glance at its far edge.

Calibration ends by printing how far apart the monitors landed, in units of the
noise within a single monitor:

```
Separation between displays (higher is better, under 2.0 is marginal):
  display 1 ... vs display 2 ...:  6.3
  display 3 ... vs display 4 ...:  1.4  <-- weak
```

A weak pair is two monitors that are too close together in angle to tell apart
from where you sit. That is the number to look at first if the tracker misfires.

If the blink is hard to follow, slow it down:

```sh
nix run . -- calibrate --blink-hz 1 --settle 3 --sample 2
```

Calibration is stored at `~/.config/fancy-tracker/calibration.json`. With
`--recall-position`, cursor positions are kept in `positions.json` next to it,
and a monitor with no stored position — a first run, or one just plugged in —
falls back to its centre.

## Tuning the run

```sh
nix run . -- run --preview     # camera window, 'p' pauses, 'q' quits
nix run . -- run --dry-run     # log the jumps without making them
nix run . -- diag              # live yaw/pitch and the current classification
```

`diag` is the one to reach for when something is off: it shows the live pose and
which monitor it resolves to, and never touches the cursor.

| flag | default | what it does |
| --- | --- | --- |
| `--dwell` | 6 | agreeing frames before a look counts as settled |
| `--margin` | 0.35 | how far the winning monitor must beat the runner-up |
| `--smoothing` | 0.35 | EMA weight; lower is steadier but slower |
| `--cooldown` | 0.6 | seconds between jumps |
| `--mouse-grace` | 0.5 | hold off this long after you move the mouse yourself |
| `--stickiness` | 0.5 | head start for the monitor you are already on |
| `--min-score` | 0.6 | face-detection confidence floor |
| `--recall-position` | off | land on the last cursor spot instead of the centre |

Two behaviours worth knowing, because they are what stop it fighting you:

- A jump happens only when your gaze **changes** monitor, never because the
  cursor and your gaze merely disagree. So dragging a window onto another
  monitor while still looking at this one does not get yanked back.
- `--mouse-grace` defers a jump while you are actively using the mouse.

If steep angles lose your face — most likely the monitor furthest from the
camera — lower `--min-score`, and check the detection counts that calibration
prints per dot.

### When it flips between two monitors

Two monitors that sit side by side share an edge, and looking at the last
column of one is very nearly the same head angle as looking at the first column
of the next. A narrow monitor between two others is the hardest case, because
both of its edges are seams.

Two things work against that:

- **Each monitor is judged against its own spread**, not a shared one. A tall
  portrait monitor legitimately covers a wide pitch range, and under a single
  shared yardstick it loses to its more compact neighbours — measured on a real
  four-monitor setup, a portrait monitor's median confidence was 0.90 where a
  landscape monitor of the same pixel count scored 2.70.
- **`--stickiness`** gives the monitor you are already on a head start, so
  leaving costs more than arriving did. Without it, two neighbours trade places
  whenever your pose sits near their shared edge, which shows up in the log as
  `A -> B -> A -> B` at margins barely over `--margin`.

Raise `--stickiness` if it still flips; lower it if a monitor becomes hard to
reach. Watch the margins `--dry-run` prints: a monitor whose margins sit below
about 1.0 is one whose profile is too close to a neighbour's.

## Development

```sh
nix develop
python -m fancy_tracker --help
nix develop --command python tests/selftest.py
```

The self-test covers everything that does not need a live camera: detection and
pose on a pinned fixture face, that yaw genuinely responds to orientation, the
calibration round-trip and classifier, and the overlay target geometry.

## Notes

- The YuNet model is fetched by the flake and pinned by hash, so runtime needs
  no network. It is stored under a `.onnx` name deliberately: OpenCV picks its
  DNN importer from the file extension, and a `.bin` name sends it looking for
  an OpenVINO backend that is not there.
- The calibration overlay builds each window at the origin and then positions it
  with `setFrameOrigin:`. `initWithContentRect:…screen:` reads its rectangle in
  the target screen's own coordinate space, so passing a global frame puts every
  monitor except the one at `(0,0)` off-screen.
