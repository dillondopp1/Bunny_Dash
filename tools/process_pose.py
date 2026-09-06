#!/usr/bin/env python3
"""Turn raw MediaPipe landmarks into a smoothed stick-figure vault.

Steps
1. Keep the left/right labels consistent from frame to frame (MediaPipe swaps
   sides freely on a side view, especially when the body is inverted).
2. Drop outlier joints (far from a running median), interpolate the gaps.
3. Smooth every joint track with a Savitzky-Golay filter.
4. Mirror if needed so the athlete runs left to right, then map video pixels
   to the 1000x430 canvas used by pole-vault-positions.html: plant box at
   (560, 340), scale so the hip-to-shoulder distance is 44 canvas units.
5. Compute segment angles (degrees, 0 = straight up, 90 = forward toward the
   pit, 180 = down), unwrapped so they stay continuous through the inversion.
6. Model the pole from the top hand to the box with a bend estimated from the
   chord shortening.
7. Pick the nine key positions automatically (overridable with --keys).

Usage:
    python tools/process_pose.py raw.json vault.json [--box X Y] [--keys Run=3,Plant=8,...]
"""
import argparse
import json
import math

import numpy as np
from scipy.signal import savgol_filter


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

CANVAS_W, CANVAS_H = 1000, 430
GROUND_Y, BOX_X = 340, 560
BONES = {"torso": 44, "head": 16, "upperArm": 24, "forearm": 24, "thigh": 32, "shin": 32}
KEY_NAMES = ["Run", "Plant", "Takeoff", "Swing", "Rockback", "Extension",
             "Turn + clear", "Fly away", "Land"]

ARM = ["shoulder", "elbow", "wrist"]
LEG = ["hip", "knee", "ankle", "heel", "foot"]
JOINTS = ["nose", "l_ear", "r_ear"] + [f"{s}_{j}" for s in "lr" for j in ARM + LEG]


def load_raw(path):
    raw = json.load(open(path))
    n = len(raw["frames"])
    tracks = {j: np.full((n, 3), np.nan) for j in JOINTS}
    ok = np.zeros(n, bool)
    for i, fr in enumerate(raw["frames"]):
        if fr["ok"] and fr["landmarks"]:
            ok[i] = True
            for j in JOINTS:
                tracks[j][i] = fr["landmarks"][j]
    return raw, tracks, ok


def fix_sides(tracks, ok):
    """Swap l/r labels per frame (arms and legs separately) for continuity."""
    n = len(ok)
    for group in (ARM, LEG):
        prev = None
        for i in range(n):
            if not ok[i]:
                continue
            L = {j: tracks[f"l_{j}"][i].copy() for j in group}
            R = {j: tracks[f"r_{j}"][i].copy() for j in group}
            if prev is not None:
                pl, pr = prev
                keep = sum(np.hypot(*(L[j][:2] - pl[j][:2])) + np.hypot(*(R[j][:2] - pr[j][:2])) for j in group)
                swap = sum(np.hypot(*(L[j][:2] - pr[j][:2])) + np.hypot(*(R[j][:2] - pl[j][:2])) for j in group)
                if swap < keep:
                    for j in group:
                        tracks[f"l_{j}"][i], tracks[f"r_{j}"][i] = R[j], L[j]
                    L, R = R, L
            prev = (L, R)


def clean_and_smooth(tracks, ok, torso_px, window=7, poly=3, max_dev=0.6, min_vis=0.3):
    """Outlier rejection against a running median, gap interpolation, smoothing."""
    n = len(ok)
    idx = np.arange(n)
    good_range = idx[ok]
    first, last = good_range.min(), good_range.max()
    out = {}
    for j, tr in tracks.items():
        xy = tr[:, :2].copy()
        vis = tr[:, 2]
        xy[(~ok) | (vis < min_vis)] = np.nan
        # running median over valid neighbours
        for k in range(2):
            col = xy[:, k]
            med = np.full(n, np.nan)
            for i in range(n):
                lo, hi = max(0, i - 3), min(n, i + 4)
                win = col[lo:hi]
                win = win[~np.isnan(win)]
                if len(win):
                    med[i] = np.median(win)
            bad = np.abs(col - med) > max_dev * torso_px
            xy[bad, k] = np.nan
        both_bad = np.isnan(xy).any(1)
        xy[both_bad] = np.nan
        valid = ~np.isnan(xy[:, 0])
        res = np.full((n, 2), np.nan)
        if valid.sum() >= 4:
            for k in range(2):
                res[first:last + 1, k] = np.interp(idx[first:last + 1], idx[valid], xy[valid, k])
            seg = res[first:last + 1]
            w = min(window, len(seg) if len(seg) % 2 else len(seg) - 1)
            if w > poly + 1:
                res[first:last + 1] = savgol_filter(seg, w, poly, axis=0)
        out[j] = res
    return out, first, last


def angle_deg(a, b):
    """Angle of segment a->b in canvas coords: 0 up, 90 forward (+x), 180 down."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dx, -dy))


def unwrap(series):
    return np.degrees(np.unwrap(np.radians(np.array(series))))


def pole_sag(chord, length):
    """Perpendicular offset of a quadratic Bezier control point so the curve
    has roughly the given arc length. Returns the control offset d (the curve
    midpoint sits at d/2 from the chord)."""
    if length <= chord:
        return 0.0
    return math.sqrt(3.0 * chord * (length - chord) / 8.0)


def guess_box(sm, ok, first, last, torso_px, height):
    """Estimate the plant box in video pixels.

    Both hands are on the pole, so the pole runs along the line from the top
    hand through the bottom hand. Extending that line to ground level lands on
    the box. Averaged over the frames just before the athlete leaves the
    ground, where the pole is still straight and both hands are gripping.
    """
    feet = np.nanmax(np.stack([sm[f"{s}_{j}"][:, 1] for s in "lr" for j in ("heel", "foot", "ankle")]), 0)
    early = slice(first, first + max(3, (last - first) // 4))
    ground = float(np.nanpercentile(feet[early][~np.isnan(feet[early])], 90))
    hips = (sm["l_hip"][:, 1] + sm["r_hip"][:, 1]) / 2
    # ground contact ends when the feet lift clear of ground level
    contact = feet > ground - 0.15 * torso_px
    peak = int(np.nanargmin(hips[first:last + 1])) + first
    onground = np.where(contact[first:peak])[0]
    takeoff = int(onground.max()) + first if len(onground) else first
    xs = []
    for i in range(max(first, takeoff - 4), takeoff + 1):
        a, b = sm["l_wrist"][i], sm["r_wrist"][i]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        top, bot = (a, b) if a[1] < b[1] else (b, a)   # smaller y is higher
        dy = bot[1] - top[1]
        if dy < 0.15 * torso_px:                        # hands level, no direction
            continue
        t = (ground - top[1]) / dy
        xs.append(top[0] + (bot[0] - top[0]) * t)
    if not xs:
        hip_x = (sm["l_hip"][:, 0] + sm["r_hip"][:, 0]) / 2
        return np.array([hip_x[takeoff], ground])
    return np.array([float(np.median(xs)), ground])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--box", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="plant box position in video pixels (where the pole tip sits)")
    ap.add_argument("--mirror", choices=["auto", "yes", "no"], default="auto",
                    help="mirror horizontally so the athlete runs left to right")
    ap.add_argument("--keys", default="", help="override key frames, e.g. Run=3,Plant=8")
    ap.add_argument("--release", type=int, default=None, help="frame where the hands leave the pole")
    ap.add_argument("--bar", type=float, default=None,
                    help="crossbar height in video pixels above the box (optional, for drawing)")
    ap.add_argument("--carry-angle", type=float, default=25.0,
                    help="pole tip angle above horizontal at the first frame of the run (degrees)")
    ap.add_argument("--name", default="vault")
    args = ap.parse_args()

    raw, tracks, ok = load_raw(args.raw)
    W, H, fps = raw["width"], raw["height"], raw["fps"]
    n = len(ok)
    fix_sides(tracks, ok)

    # Body scale from the hip-to-shoulder distance over well-tracked frames.
    hip = (tracks["l_hip"][:, :2] + tracks["r_hip"][:, :2]) / 2
    sho = (tracks["l_shoulder"][:, :2] + tracks["r_shoulder"][:, :2]) / 2
    torso_len = np.hypot(*(sho - hip).T)
    # The longest apparent torso is the least foreshortened (side-on, upright).
    torso_px = float(np.nanpercentile(torso_len[ok], 85))
    scale = BONES["torso"] / torso_px

    sm, first, last = clean_and_smooth(tracks, ok, torso_px)

    # Run direction: sign of hip x displacement over the first half of the track.
    hip_x = (sm["l_hip"][:, 0] + sm["r_hip"][:, 0]) / 2
    mid = first + (last - first) // 2
    runs_left = hip_x[mid] < hip_x[first]
    mirror = (args.mirror == "yes") or (args.mirror == "auto" and runs_left)

    # Box in video pixels. Default: below the median takeoff-side hand, on the ground.
    if args.box:
        box_px = np.array(args.box, float)
    else:
        box_px = guess_box(sm, ok, first, last, torso_px, H)
        print("no --box given, estimating box at", box_px.round(0))

    def to_canvas(p):
        x = (W - p[0]) if mirror else p[0]
        bx = (W - box_px[0]) if mirror else box_px[0]
        return np.array([BOX_X + (x - bx) * scale, GROUND_Y + (p[1] - box_px[1]) * scale])

    frames_out = []
    torso_series, head_series = [], []
    seg_series = {k: [] for k in ["lUpperArm", "lForearm", "rUpperArm", "rForearm",
                                  "lThigh", "lShin", "rThigh", "rShin"]}
    hips_c, hands_c, feet_c, joints_c = [], [], [], []
    for i in range(first, last + 1):
        J = {j: to_canvas(sm[j][i]) for j in JOINTS}
        hip_c = (J["l_hip"] + J["r_hip"]) / 2
        sho_c = (J["l_shoulder"] + J["r_shoulder"]) / 2
        head_c = (J["l_ear"] + J["r_ear"]) / 2
        if np.isnan(head_c).any():
            head_c = J["nose"]
        torso_series.append(angle_deg(hip_c, sho_c))
        head_series.append(angle_deg(sho_c, head_c))
        for side, key in (("l", "l"), ("r", "r")):
            seg_series[f"{key}UpperArm"].append(angle_deg(J[f"{side}_shoulder"], J[f"{side}_elbow"]))
            seg_series[f"{key}Forearm"].append(angle_deg(J[f"{side}_elbow"], J[f"{side}_wrist"]))
            seg_series[f"{key}Thigh"].append(angle_deg(J[f"{side}_hip"], J[f"{side}_knee"]))
            seg_series[f"{key}Shin"].append(angle_deg(J[f"{side}_knee"], J[f"{side}_ankle"]))
        hips_c.append(hip_c)
        hands_c.append((J["l_wrist"], J["r_wrist"]))
        feet_c.append(min(J["l_ankle"][1], J["r_ankle"][1], J["l_foot"][1], J["r_foot"][1]))
        joints_c.append({j: [round(float(v[0]), 1), round(float(v[1]), 1)] for j, v in J.items()})

    torso_u = unwrap(torso_series)
    head_u = unwrap(head_series)
    seg_u = {k: unwrap(v) for k, v in seg_series.items()}
    hips_c = np.array(hips_c)
    m = len(hips_c)
    box_c = np.array([BOX_X, GROUND_Y], float)

    # Pole geometry: the top hand is the wrist farther from the box.
    top_hand = np.array([max(h, key=lambda p: np.hypot(*(p - box_c))) for h in hands_c])
    chord = np.hypot(*(top_hand - box_c).T)
    lowest_foot_px = np.nanmax(np.stack([sm[f"{s}_{j}"][first:last + 1, 1]
                                         for s in "lr" for j in ("heel", "foot", "ankle")]), 0)
    ground_px = float(np.nanpercentile(lowest_foot_px[:max(3, m // 4)], 90))
    contact = lowest_foot_px > ground_px - 0.15 * torso_px
    peak = int(np.argmin(hips_c[:, 1]))
    takeoff = int(np.where(contact[:peak])[0].max()) if contact[:peak].any() else 0
    grip = float(chord[takeoff])
    plant_candidates = np.where(chord[:takeoff + 1] <= grip * 1.03)[0]
    plant = int(plant_candidates.min()) if len(plant_candidates) else max(0, takeoff - 2)
    if args.release is not None:
        release = args.release - first
    else:
        after = np.where(chord[peak:] > grip * 1.05)[0]
        release = int(peak + after.min()) if len(after) else m - 1

    poles = []
    rel_angle = None
    for i in range(m):
        if i < plant:
            # Straight pole carried in the hands. The tip starts carry_angle above
            # horizontal and drops smoothly so that it reaches the box at the plant.
            plant_dir = box_c - top_hand[plant]
            plant_ang = math.atan2(plant_dir[1], plant_dir[0])          # image angle, y down
            carry_ang = -math.radians(args.carry_angle)                  # tip up and forward
            u = i / max(plant, 1)
            u = u * u * (3 - 2 * u)                                       # smoothstep
            ang = carry_ang + (plant_ang - carry_ang) * u
            tip = top_hand[i] + grip * np.array([math.cos(ang), math.sin(ang)])
            poles.append({"top": top_hand[i].round(1).tolist(), "tip": tip.round(1).tolist(),
                          "bend": 0.0, "sag": 0.0, "angle": round(angle_deg(tip, top_hand[i]), 1),
                          "state": "carry"})
        elif i < release:
            bend = max(0.0, 1.0 - chord[i] / grip)
            sag = pole_sag(chord[i], grip)
            poles.append({"top": top_hand[i].round(1).tolist(), "tip": box_c.tolist(),
                          "bend": round(float(bend), 3), "sag": round(float(sag), 1),
                          "angle": round(angle_deg(box_c, top_hand[i]), 1), "state": "planted"})
        else:
            if rel_angle is None:
                rel_angle = angle_deg(box_c, top_hand[i - 1]) if i > 0 else 0.0
            ang = rel_angle - 2.5 * (i - release)
            a = math.radians(ang)
            top = box_c + grip * np.array([math.sin(a), -math.cos(a)])
            poles.append({"top": top.round(1).tolist(), "tip": box_c.tolist(), "bend": 0.0, "sag": 0.0,
                          "angle": round(ang, 1), "state": "released"})

    # Key positions.
    def first_cross(series, thr, start):
        idx = np.where(series[start:] >= thr)[0]
        return int(start + idx.min()) if len(idx) else None

    # Body rotation relative to the takeoff torso angle. The swing rotates the
    # torso backward (hips pass the shoulders), which is negative in this
    # convention, so work with the magnitude of the rotation.
    rot = torso_u - torso_u[takeoff]
    if abs(rot[peak]) > 0 and rot[peak] < 0:
        rot = -rot
    swing = first_cross(rot, 45, takeoff) or takeoff + 3
    rockback = first_cross(rot, 120, swing) or swing + 4
    inv = [i for i in range(rockback, release + 1) if 120 <= rot[i] <= 240]
    extension = int(min(inv, key=lambda i: feet_c[i])) if inv else rockback + 3
    turn = peak
    # Fly away: either the body keeps rotating over the bar (past 270) or, on a
    # vault that unrolls back to upright, the rotation drops back under 90.
    fly = first_cross(rot, 270, turn)
    if fly is None:
        back = np.where(rot[turn:] < 90)[0]
        fly = int(turn + back.min()) if len(back) else min(m - 1, turn + 4)
    land = m - 1
    keys = {"Run": max(0, plant - 5), "Plant": plant, "Takeoff": takeoff, "Swing": swing,
            "Rockback": rockback, "Extension": extension, "Turn + clear": turn,
            "Fly away": fly, "Land": land}
    keys = {k: v + first for k, v in keys.items()}
    for item in filter(None, args.keys.split(",")):
        k, v = item.split("=")
        keys[k.strip()] = int(v)
    keys = {k: int(min(max(keys[k], first), last)) for k in KEY_NAMES}

    for i in range(m):
        fi = i + first
        angles = {"torso": round(float(torso_u[i]), 1), "head": round(float(head_u[i]), 1)}
        angles.update({k: round(float(v[i]), 1) for k, v in seg_u.items()})
        frames_out.append({
            "frame": fi, "t": round(fi / fps, 4),
            "hip": hips_c[i].round(1).tolist(),
            "angles": angles,
            "pole": poles[i],
            "joints": joints_c[i],
            "quality": round(float(raw["frames"][fi].get("visibility", 0) if raw["frames"][fi]["ok"] else 0), 2),
        })

    bar_y = None
    if args.bar is not None:
        bar_y = round(GROUND_Y - args.bar * scale, 1)

    out = {
        "name": args.name,
        "source": {"video": raw["video"], "fps": fps, "width": W, "height": H,
                   "mirrored": bool(mirror), "box_px": box_px.round(1).tolist(),
                   "ground_px": round(ground_px, 1), "torso_px": round(torso_px, 1),
                   "scale": round(scale, 5), "first_frame": int(first), "last_frame": int(last)},
        "canvas": {"width": CANVAS_W, "height": CANVAS_H, "groundY": GROUND_Y, "boxX": BOX_X,
                   "barY": bar_y},
        "bones": BONES,
        "angleConvention": "degrees, 0 = straight up, 90 = forward toward the pit, 180 = down; unwrapped (continuous)",
        "poleStyle": {"bendToward": "pit", "bendScale": 1.0},
        "pole": {"gripLength": round(grip, 1), "plantFrame": int(plant + first),
                 "takeoffFrame": int(takeoff + first), "releaseFrame": int(release + first)},
        "keys": [{"name": k, "frame": keys[k]} for k in KEY_NAMES],
        "frames": frames_out,
    }
    json.dump(out, open(args.out, "w"), indent=None, cls=NumpyEncoder)
    print(f"wrote {args.out}: frames {first}..{last}, mirror={mirror}, scale={scale:.4f}, "
          f"torso_px={torso_px:.1f}, ground_px={ground_px:.0f}, grip={grip:.1f}")
    print("pole: plant", plant + first, "takeoff", takeoff + first, "release", release + first)
    print("keys:", ", ".join(f"{k}={keys[k]}" for k in KEY_NAMES))

    # Key-position library file (same per-frame objects, tagged by name).
    lib = {"name": args.name, "bones": BONES, "angleConvention": out["angleConvention"],
           "canvas": out["canvas"],
           "positions": [dict(frames_out[keys[k] - first], name=k) for k in KEY_NAMES]}
    lib_path = args.out.replace(".json", "_keys.json")
    json.dump(lib, open(lib_path, "w"), indent=1, cls=NumpyEncoder)
    print("wrote", lib_path)


if __name__ == "__main__":
    main()
