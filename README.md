# Pole vault stick figures from video

Tools for turning a side-on pole vault clip into a stick figure animation and
a JSON of per-frame joint angles that plays in `pole-vault-positions.html`.

## Pipeline

```
video.mp4
  -> tools/extract_pose.py   MediaPipe Pose on a crop around the athlete, 4 rotations per frame
  -> tools/process_pose.py   side consistency, outlier removal, smoothing, angles, pole, key frames
  -> tools/render_stick.py   MP4 / GIF / key position sheet on the 1000x430 canvas
  -> pole-vault-positions.html   sequence view, animate, scrub, load and export JSON
```

Install with `pip install -r requirements.txt`. MediaPipe 0.10.14 with
`model_complexity=1` runs offline; `imageio-ffmpeg` supplies the ffmpeg binary.

Run the whole thing on a clip:

```
python tools/extract_pose.py clip.mp4 data/clip_raw.json --debug-dir debug
python tools/process_pose.py data/clip_raw.json data/clip.json --box 360 640 --bar 430 --name clip
python tools/render_stick.py data/clip.json output/clip_stick.mp4 --gif output/clip_stick.gif --slow 2 --keys output/clip_keys.png
```

`--box X Y` is the plant box in video pixels (where the pole tip sits). If it
is hidden, extend the visible pole lines from a few frames after the plant;
they converge at the box. Without `--box` the script guesses ground level
under the athlete at mid clip, which is usually wrong, so give it.
`--bar PX` is the crossbar height above the box in pixels, only used for
drawing. Add `--video clip.mp4` to the render step to get a side-by-side
check with the mirrored source frame above the stick figure.

### What each step does

**extract_pose.py** runs a full-frame pass to find the frame where the athlete
is clearest, then tracks outward from that anchor in both directions. Each
frame is cropped around the previous track and the crop is run through
MediaPipe at 0, 90, 180 and 270 degrees. The rotation with the best mix of
landmark visibility and continuity with the previous frame wins. This is what
makes the inverted phases work; the detector alone loses the athlete once the
hips pass the hands. Frames with visibility under 0.7 or poor continuity are
marked missing. `--debug-dir` writes an overlay PNG per frame.

**process_pose.py**
1. Re-labels left and right per frame (arms and legs separately) so limbs do
   not swap between frames.
2. Drops joints more than 0.6 torso lengths from a running median, fills the
   gaps by interpolation, then smooths every track with a Savitzky-Golay
   filter (window 7, order 3).
3. Mirrors the clip if the athlete runs right to left, so the pit is always
   on the right, and maps video pixels to the canvas: box at (560, 340), scale
   chosen so the hip-to-shoulder distance is 44 units.
4. Computes segment angles from the smoothed joints and unwraps them so a
   swing reads as a continuous rotation.
5. Models the pole. Before the plant it is a straight line from the top hand
   with the tip easing down from `--carry-angle` to the box. From the plant to
   the release it runs from the top hand to the box with a bend estimated from
   the chord shortening (grip length is the hand-to-box distance at takeoff).
   The tip is never detected, only the hands are.
6. Picks the nine key positions from the torso rotation and the hip and foot
   heights. Override any of them with `--keys "Swing=18,Rockback=33"`.

**render_stick.py** rebuilds the figure from hip position and angles with the
fixed bone lengths (forward kinematics), so it shows exactly what the HTML
tool plays. When the vault goes above the canvas it shrinks the scene about
the box so the top stays visible (`--no-fit` disables this).

## Conventions

Canvas 1000x430, ground y=340, plant box x=560, standards at 600 and 640, mat
from x=610, crossbar y=120 unless the JSON carries `canvas.barY`.

Angles in degrees: 0 = straight up, 90 = forward toward the pit, 180 = down.
They are unwrapped, so a torso that rotates backward through the swing runs
from about 0 down to about -150 and back rather than flipping sign at 180.

Bone lengths: torso 44, head 16, upper arm 24, forearm 24, thigh 32, shin 32.

Left and right are consistent within a clip but which leg is called left is
arbitrary on a side view. Right-side limbs draw in grey.

## JSON format

`data/<name>.json` (full clip) has:

```
source     video name, fps, size, mirrored, box_px, ground_px, torso_px, scale, first and last frame
canvas     width, height, groundY, boxX, barY
bones      the six bone lengths
pole       gripLength, plantFrame, takeoffFrame, releaseFrame
keys       [{name, frame}] for the nine positions
frames     [{frame, t, hip:[x,y], angles:{torso, head, lUpperArm, lForearm, rUpperArm,
             rForearm, lThigh, lShin, rThigh, rShin}, pole:{top, tip, bend, sag, angle, state},
             joints:{all smoothed joints in canvas units}, quality}]
```

`pole.state` is `carry`, `planted` or `released`. `bend` is the chord
shortening (0 = straight); `sag` is the control point offset for a quadratic
curve from tip to top hand. The HTML tool draws the pole from the top hand of
the rebuilt figure, so the pole always meets the hand.

`data/<name>_keys.json` holds the nine key positions in the same per-frame
format under `positions`. This is the drill library format: a drill is a list
of named positions with `t`, `hip`, `angles` and optionally `pole`.

## Tracing and fixing frames: pose-editor.html

Open `pose-editor.html`, load the vault JSON, then load the video (or a set of
frame images named `f_0034.jpg` style). The chosen video frame is shown
mirrored with the measured joints on top (green = left side, magenta = right,
red bones, yellow torso line), the fixed-bone figure in blue and the modelled
pole in orange. Drag any joint to correct it; the angles, hip and pole for that
frame are recomputed at once and the panel on the right shows the result on
the 1000x430 canvas. Keys 1 to 9 jump to the nine positions, arrows step
frames, "Set this frame as" re-assigns a key position, "Copy previous frame"
seeds a lost frame. Pole controls: bend direction, bend scale, grip length and
per-frame state (carry, planted, released). Save writes the corrected vault
JSON or the keys JSON; render them with `tools/render_stick.py` or open them
in the positions tool.

Served over http the editor loads `data/<name>.json` automatically and uses
`data/frames/<name>/f_%04d.jpg` when present (write them with
`extract_pose.py --frames-dir data/frames/<name>`; the folder is ignored by git).
Opened from disk, use the file pickers instead.

## The HTML tool

Open `pole-vault-positions.html`. It embeds the nine key positions from the
sample clip. Served over http (for example `python -m http.server`) it also
loads `data/vault_2018-05-24.json` for frame-by-frame playback. Otherwise use
Load JSON with either a full clip file or a keys file.

Sequence shows all key positions side by side. Animate plays either the
Hermite spline through the keys or the measured video frames. Space plays and
pauses, arrow keys step. Export writes the current keys as a drill JSON.

## Sample clip

`data/vault_2018-05-24*.json` and `output/vault_2018-05-24_stick.*` come from
a 2.75 s backyard clip, 82 frames at 30 fps, 1280x720, athlete running right
to left. Box estimated at pixel (360, 640) and the bar 430 px above it.
Frames 77 to 81 are dropped: the athlete lands on a trampoline behind the
foreground and the detector loses the body. The athlete does not push off and
rotate over a bar in this clip, so Fly away is taken as the frame where the
body unrolls back past 90 degrees of rotation after the peak.

## Known limits and next steps

- The pole tip and bend are modelled, not measured. The pole bows toward the
  pit (`poleStyle.bendToward`), with `poleStyle.bendScale` to exaggerate or
  soften it.
- Frames with the body edge-on to the camera get foreshortened limbs; the
  angles are still taken in the image plane.
- Fast phases at 30 fps blur; 60 fps clips will track better.
- Next: a drill library built from `_keys.json` files.
