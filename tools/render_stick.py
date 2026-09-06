#!/usr/bin/env python3
"""Render a processed vault JSON as a stick-figure MP4 and/or GIF.

Draws the same scene as pole-vault-positions.html (1000x430 canvas, ground at
y=340, plant box at x=560, standards at 600 and 640, crossbar, mat from 610)
and rebuilds the figure from the hip position plus segment angles with the
fixed bone lengths, so what you see is exactly what the HTML tool will play.

Usage:
    python tools/render_stick.py vault.json out.mp4 [--gif out.gif] [--fps 30]
        [--video vault.mp4]   side-by-side with the (mirrored) source frame
        [--keys]              also write a sequence PNG of the nine key positions
"""
import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np

W, H = 1000, 430
GROUND, BOX_X = 340, 560
BG = (250, 248, 245)
INK = (40, 40, 40)
POLE = (30, 110, 200)
SKEL = (20, 20, 20)
HEAD_FILL = (230, 230, 230)


def seg_end(p, ang, length):
    a = math.radians(ang)
    return (p[0] + length * math.sin(a), p[1] - length * math.cos(a))


def figure_points(fr, bones):
    """Forward kinematics from hip and angles. Returns dict of point pairs."""
    hip = tuple(fr["hip"])
    A = fr["angles"]
    sho = seg_end(hip, A["torso"], bones["torso"])
    head = seg_end(sho, A["head"], bones["head"])
    segs = [("torso", hip, sho)]
    for s in "lr":
        el = seg_end(sho, A[f"{s}UpperArm"], bones["upperArm"])
        wr = seg_end(el, A[f"{s}Forearm"], bones["forearm"])
        kn = seg_end(hip, A[f"{s}Thigh"], bones["thigh"])
        an = seg_end(kn, A[f"{s}Shin"], bones["shin"])
        segs += [(f"{s}UpperArm", sho, el), (f"{s}Forearm", el, wr),
                 (f"{s}Thigh", hip, kn), (f"{s}Shin", kn, an)]
    return hip, sho, head, segs


def top_hand_from_figure(fr, bones):
    """Top hand for the pole: the wrist farther from the box (in FK space)."""
    _, _, _, segs = figure_points(fr, bones)
    wrists = [b for name, a, b in segs if name.endswith("Forearm")]
    return max(wrists, key=lambda p: math.hypot(p[0] - BOX_X, p[1] - GROUND))


def draw_scene(img, bar_y):
    img[:] = BG
    cv2.line(img, (0, GROUND), (W, GROUND), INK, 2)
    # plant box
    cv2.fillPoly(img, [np.array([[BOX_X - 40, GROUND], [BOX_X, GROUND], [BOX_X, GROUND + 12],
                                 [BOX_X - 30, GROUND + 12]])], (120, 120, 120))
    # mat
    cv2.rectangle(img, (610, GROUND - 40), (W, GROUND), (200, 215, 235), -1)
    cv2.rectangle(img, (610, GROUND - 40), (W, GROUND), (150, 165, 190), 1)
    # standards and bar
    for x in (600, 640):
        cv2.line(img, (x, GROUND), (x, 30), (150, 150, 150), 2)
    if bar_y is not None:
        cv2.line(img, (596, int(bar_y)), (644, int(bar_y)), (0, 120, 220), 3)


def draw_pole(img, fr, bones, use_fk_hand=True):
    pole = fr.get("pole")
    if not pole:
        return
    tip = tuple(pole["tip"])
    top = top_hand_from_figure(fr, bones) if (use_fk_hand and pole["state"] != "released") else tuple(pole["top"])
    if pole["state"] == "carry":
        # straight, length preserved from the hand toward the tip
        d = (tip[0] - top[0], tip[1] - top[1])
        cv2.line(img, (int(top[0]), int(top[1])), (int(tip[0]), int(tip[1])), POLE, 3, cv2.LINE_AA)
        return
    sag = pole.get("sag", 0.0)
    cx, cy = (top[0] + tip[0]) / 2, (top[1] + tip[1]) / 2
    dx, dy = top[0] - tip[0], top[1] - tip[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    if nx > 0 or (abs(nx) < 1e-6 and ny > 0):
        nx, ny = -nx, -ny
    ctrl = (cx + nx * sag, cy + ny * sag)
    pts = []
    for t in np.linspace(0, 1, 24):
        x = (1 - t) ** 2 * tip[0] + 2 * (1 - t) * t * ctrl[0] + t * t * top[0]
        y = (1 - t) ** 2 * tip[1] + 2 * (1 - t) * t * ctrl[1] + t * t * top[1]
        pts.append([int(round(x)), int(round(y))])
    cv2.polylines(img, [np.array(pts)], False, POLE, 3, cv2.LINE_AA)


def draw_figure(img, fr, bones, color=SKEL, thick=4):
    hip, sho, head, segs = figure_points(fr, bones)
    for name, a, b in segs:
        c = color if not name.startswith("r") else tuple(int(v * 0.55 + 90) for v in color)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, thick, cv2.LINE_AA)
    cv2.circle(img, (int(head[0]), int(head[1])), int(bones["head"] * 0.6), HEAD_FILL, -1, cv2.LINE_AA)
    cv2.circle(img, (int(head[0]), int(head[1])), int(bones["head"] * 0.6), color, 2, cv2.LINE_AA)
    for p in (hip, sho):
        cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1, cv2.LINE_AA)


def fit_data(data, margin=10):
    """Uniformly shrink the scene about the plant box (ground level) when the
    vault goes above the canvas. Bone lengths shrink by the same factor so the
    figure keeps its proportions. Returns (data copy, factor)."""
    ys = []
    for fr in data["frames"]:
        _, _, head, segs = figure_points(fr, data["bones"])
        ys.append(head[1] - data["bones"]["head"])
        ys += [b[1] for _, _, b in segs]
        if fr.get("pole"):
            ys.append(fr["pole"]["top"][1])
    top = min(ys)
    if top >= margin:
        return data, 1.0
    f = (GROUND - margin) / (GROUND - top)
    out = json.loads(json.dumps(data))
    out["bones"] = {k: v * f for k, v in data["bones"].items()}
    if out["canvas"].get("barY") is not None:
        out["canvas"]["barY"] = GROUND + (out["canvas"]["barY"] - GROUND) * f

    def tf(p):
        return [BOX_X + (p[0] - BOX_X) * f, GROUND + (p[1] - GROUND) * f]

    for fr in out["frames"]:
        fr["hip"] = tf(fr["hip"])
        if fr.get("pole"):
            fr["pole"]["top"] = tf(fr["pole"]["top"])
            fr["pole"]["tip"] = tf(fr["pole"]["tip"])
            fr["pole"]["sag"] = fr["pole"].get("sag", 0) * f
        if fr.get("joints"):
            fr["joints"] = {k: tf(v) for k, v in fr["joints"].items()}
    return out, f


def render_frame(data, fr, label=None):
    img = np.zeros((H, W, 3), np.uint8)
    draw_scene(img, data["canvas"].get("barY") or 120)
    draw_pole(img, fr, data["bones"])
    draw_figure(img, fr, data["bones"])
    txt = f"frame {fr['frame']}  t={fr['t']:.2f}s"
    if label:
        txt = f"{label}   " + txt
    cv2.putText(img, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)
    return img


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("out_mp4")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--video", default=None, help="source video for a side-by-side check")
    ap.add_argument("--keys", default=None, help="write a PNG sequence of the key positions here")
    ap.add_argument("--slow", type=float, default=1.0, help="playback slowdown factor")
    ap.add_argument("--no-fit", action="store_true", help="do not shrink the scene to fit the canvas")
    args = ap.parse_args()

    data = json.load(open(args.json))
    if not args.no_fit:
        data, f = fit_data(data)
        if f < 1:
            print(f"scene shrunk by {f:.3f} to fit the canvas")
    fps = (args.fps or data["source"]["fps"]) / args.slow
    key_by_frame = {k["frame"]: k["name"] for k in data["keys"]}

    src = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        src = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            src.append(im)

    tmpdir = os.path.splitext(args.out_mp4)[0] + "_frames"
    os.makedirs(tmpdir, exist_ok=True)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    for k, fr in enumerate(data["frames"]):
        img = render_frame(data, fr, key_by_frame.get(fr["frame"]))
        if src is not None and fr["frame"] < len(src):
            v = src[fr["frame"]]
            if data["source"]["mirrored"]:
                v = v[:, ::-1]
            v = cv2.resize(v, (W, int(v.shape[0] * W / v.shape[1])))
            img = np.vstack([v, img])
        cv2.imwrite(os.path.join(tmpdir, f"f_{k:04d}.png"), img)

    ff = ffmpeg()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out_mp4], check=True)
    print("wrote", args.out_mp4)
    if args.gif:
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(tmpdir, "f_%04d.png"),
                        "-vf", "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", args.gif], check=True)
        print("wrote", args.gif)
    if args.keys:
        tiles = []
        by_frame = {fr["frame"]: fr for fr in data["frames"]}
        for k in data["keys"]:
            fr = by_frame[k["frame"]]
            img = render_frame(data, fr, k["name"])
            tiles.append(img)
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        if len(rows[-1].shape) and rows[-1].shape[1] < rows[0].shape[1]:
            pad = np.full((H, rows[0].shape[1] - rows[-1].shape[1], 3), BG, np.uint8)
            rows[-1] = np.hstack([rows[-1], pad])
        cv2.imwrite(args.keys, np.vstack(rows))
        print("wrote", args.keys)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
