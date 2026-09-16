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
- **The first launch will fail, and that is expected.** Camera permission is not
  inherited from the terminal you granted it to; the agent asks for its own, and
  the log shows `Could not open camera 0` while the prompt is still sitting
  there. Click allow, and `KeepAlive` picks it up on the next retry — which is
  what `ThrottleInterval` is really for. Confirm with
  `launchctl print gui/$(id -u)/nl.tschallacka.fancy-tracker | grep -E 'state|pid'`;
  you want `state = running`, and a `last exit code = 1` left over from the
  denied first attempt is harmless.
- **`nix profile upgrade` is what updates it.** The agent runs the copy in your
  profile, not your working tree, so editing the checkout changes nothing until
  you reinstall. Changing flags in the plist needs a `bootout` and `bootstrap`
  too — editing the file alone does nothing.

### Recalibrating while the agent is running

Recalibrating is picked up on its own, within a couple of seconds, with no
restart and no `bootout`. The tracker watches `calibration.json` and swaps
profiles in place.

Stop the agent first anyway:

```sh
launchctl bootout gui/$(id -u)/nl.tschallacka.fancy-tracker
nix run github:tschallacka/fancy-tracker -- calibrate
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/nl.tschallacka.fancy-tracker.plist
```

Not because the reload needs it, but because a running agent moves the cursor to
the centre of whatever monitor it thinks you are looking at — while calibration
is trying to park that same cursor on the dot it wants you to look at. The two
fight over it, and the cue you are supposed to follow becomes useless. They also
share the webcam, which halves the frame rate and so the sample count.


## Calibrating

Every monitor dims and shows five dots — its four corners and its centre. The
dot to look at blinks, then goes solid; look straight at it and hold still
while it is solid. The cursor is parked on the dot as a second cue.

All five points feed one profile per monitor, on purpose. A 2560px-wide monitor
covers a wide angle from where you sit, so a profile built only from its centre
would not recognise a glance at its far edge.

Each dot tells you how it went as soon as it is taken:

| dot | meaning |
| --- | --- |
| small grey | not taken yet |
| yellow, pulsing, white ring | being sampled right now |
| **green with a tick** | good — steady aim, enough frames |
| **red, flashing** | unusable; it goes straight back to yellow to be retaken |
| **amber ring** | queued for a retake, because it sits too close to a dot already recorded on another monitor |

A dot is unusable if too few frames found your face — you looked far enough
away that the camera lost you — or if your aim wandered more than about six
degrees while it was solid. Either way it is simply taken again, up to three
times.

The amber case is the interesting one. If a dot ends up looking almost
identical to one already recorded on a *different* monitor, the boundary
between those two monitors is unclear, and the earlier dot is queued to be
taken again at the end of the pass. Often the first take was just sloppy and
the retake separates them. Sometimes it does not, because the two really are at
the same angle from where you sit — two monitors that share a physical edge are
the usual culprit. The separation table printed at the end is what tells you
which case you are in.

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

### Finding the cursor after it moves

A cursor that teleports is harder to follow than one that was dragged, and on a
busy background it can be lost entirely. So arriving sets off a burst of sparks
centred on the landing point: the eye is drawn to motion, and motion that
converges on a point says *where* to look rather than merely that something
happened.

```sh
fancy-tracker flash                          # preview it, no camera needed
fancy-tracker flash --emphasis-scale 7       # bigger
fancy-tracker flash --emphasis-seconds 2     # slower
```

Two details are load-bearing rather than decorative. Every bright mark is laid
over a darker one of its own, because additive blending is invisible on a pale
desktop and this has to work on both. And the sparks' speeds vary widely — when
they all travel the same distance they land on one circle and the whole thing
reads as a clock face rather than an explosion.

It is drawn, not applied. The system's own pointer magnification is the
`mouseDriverCursorSize` accessibility preference, which is global and
persistent: a crash midway through would leave the cursor stuck large and that
setting quietly changed. A click-through window leaves nothing behind if the
process dies. `--emphasis-seconds 0` turns it off.

| flag | default | what it does |
| --- | --- | --- |
| `--dwell` | 6 | agreeing frames before a look counts as settled |
| `--margin` | 0.35 | how far the winning monitor must beat the runner-up |
| `--smoothing` | 0.35 | EMA weight; lower is steadier but slower |
| `--cooldown` | 0.6 | seconds between jumps |
| `--mouse-grace` | 0.5 | hold off this long after you move the mouse yourself |
| `--stickiness` | 0.5 | head start for the monitor you are already on |
| `--gap-tolerance` | 2.0 | how far off every monitor a look may land before it counts as *between* them |
| `--emphasis-seconds` | 1.1 | how long the cursor swells on arrival; `0` turns it off |
| `--emphasis-scale` | 2.0 | how far the sparks reach |
| `--no-prompt` | off | don't offer a recalibration when the monitors change |
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

## Where it thinks your monitors are

macOS's arrangement is bookkeeping, not geography. It butts panels together in
one pixel plane whatever their real position, so two monitors with a hand's
width of desk between them share an edge as far as the system is concerned.
Classifying against that pretend geometry costs accuracy exactly at the seams.

Calibration measures the truth instead. Five dots per monitor, each at a known
spot with a measured head pose, are enough to fit that monitor's **gradient** —
how far the head turns per pixel travelled — and from there to place its edges
in angular space. Comparing one monitor's far edge against its neighbour's near
edge then shows whether they really adjoin:

```
Inferred layout (from where your head actually pointed, not the arrangement):
  monitor                      yaw span       pitch span     fit
  2560x1440 @ (-2560,0)     -41.2..-18.6    -7.1.. +8.4     0.31
  2560x1440 @ (0,0)         -16.9.. +8.2    -6.8.. +9.1     0.28
  1440x2560 @ (2560,0)       +9.4..+22.7   -11.2..+24.6     0.44
  1512x982  @ (4000,0)      +29.1..+41.0    +6.3..+19.8     0.35

  Between neighbours:
    2560x1440 @ (-2560,0)  -> 2560x1440 @ (0,0)    +1.7 deg  adjoining
    2560x1440 @ (0,0)      -> 1440x2560 @ (2560,0) +1.2 deg  adjoining
    1440x2560 @ (2560,0)   -> 1512x982 @ (4000,0)  +6.4 deg  gap, about 470px
                                                              of screen would fill it
```

That last line is a physical gap the arrangement claims does not exist. Two
things follow from knowing about it. A look that lands *in* the gap belongs to
no monitor, so the cursor stays where it is rather than being sent somewhere on
a guess — that is `--gap-tolerance`. And the monitors on either side are no
longer forced to explain poses that fall between them, which is what made that
pair hard to separate in the first place.

The angles it recovers are compressed, typically to about half of true, because
you sweep a panel mostly with your eyes and only partly with your head. Panels
therefore come out narrower than they are and the gaps between them wider. The
comparative findings still hold — which monitors adjoin, and where a real gap
is — but it is not a tape measure.

Eyes would close that gap and are deliberately not used. Past roughly 45° of
head turn the far iris is occluded by the nose and brow, and on a wide desk the
outer monitors sit well past that: measured on a four-monitor arrangement, the
leftmost was viewed at 67° of turn, near profile. Glasses make it worse again,
with frames occluding the iris, lenses refracting its apparent position, and
coatings throwing highlights back at the sensor. None of this affects which
monitor you are judged to be facing, which only needs angles that differ from
each other, not angles that are true.

The report also prints **axis coupling** — how much yaw is read when only pitch
changed. Five-point `solvePnP` leaks one into the other, and looking up and down
a tall portrait monitor is where it shows. The per-monitor fit is a full linear
map rather than two independent axes, so the coupling is represented rather than
fought.

### When the monitors change

If a monitor is plugged, unplugged, moved or resized, the saved calibration no
longer describes reality. The tracker notices and asks whether to recalibrate.
Say yes and it hands the camera over, runs calibration, and picks the result up
without the agent needing a restart. `--no-prompt` turns the offer off.

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
