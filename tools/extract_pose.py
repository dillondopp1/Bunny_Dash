#!/usr/bin/env python3
"""Extract 2D pose landmarks from a side-on pole vault video.

Runs MediaPipe Pose (model_complexity=1, works offline) on a crop around the
athlete. For every frame the crop is tried at 0, 90, 180 and 270 degrees of
rotation and the best result is kept, because the detector fails on inverted
bodies far more often than on upright ones. Output is a raw landmark JSON in
video pixel coordinates that process_pose.py smooths and converts.

Usage:
    python tools/extract_pose.py video.mp4 out_raw.json [--debug-dir DIR]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import mediapipe as mp

MP_POSE = mp.solutions.pose

# Landmarks we keep (MediaPipe indices) and the names used downstream.
KEEP = {
    0: "nose", 7: "l_ear", 8: "r_ear",
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow", 14: "r_elbow",
    15: "l_wrist", 16: "r_wrist",
    23: "l_hip", 24: "r_hip",
    25: "l_knee", 26: "r_knee",
    27: "l_ankle", 28: "r_ankle",
    29: "l_heel", 30: "r_heel",
    31: "l_foot", 32: "r_foot",
}
CORE = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    return frames, fps


def landmarks_to_dict(res, w, h):
    if res is None or res.pose_landmarks is None:
        return None
    out = {}
    for idx, name in KEEP.items():
        lm = res.pose_landmarks.landmark[idx]
        out[name] = [lm.x * w, lm.y * h, lm.visibility]
    return out


def rotate_image(im, k):
    """Rotate by k*90 degrees counter-clockwise."""
    return np.ascontiguousarray(np.rot90(im, k))


def unrotate_point(x, y, k, w, h):
    """Map a point in the k*90-degree rotated image back to the unrotated crop.

    w, h are the unrotated crop size. np.rot90 rotates counter-clockwise:
    k=1: new(x', y') = (y, w-1-x)  so x = w-1-y', y = x'
    """
    k %= 4
    if k == 0:
        return x, y
    if k == 1:
        return (w - y), x
    if k == 2:
        return (w - x), (h - y)
    return y, (h - x)


def score(lms, prev, scale):
    """Higher is better: mean visibility of core joints, minus a motion penalty."""
    if lms is None:
        return -1.0
    vis = np.mean([lms[n][2] for n in CORE])
    if prev is None:
        return vis
    d = []
    for n in CORE:
        if prev[n][2] > 0.5:
            d.append(np.hypot(lms[n][0] - prev[n][0], lms[n][1] - prev[n][1]))
    if not d:
        return vis
    motion = np.median(d) / max(scale, 1.0)
    return vis - 0.35 * motion


def bbox_from(lms, w, h):
    pts = np.array([[v[0], v[1]] for k, v in lms.items() if v[2] > 0.3])
    if len(pts) < 4:
        return None
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return x0, y0, x1, y1


def square_crop(bbox, w, h, pad=1.9, min_size=260):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0) * pad
    size = max(size, min_size)
    size = int(min(size, min(w, h)))
    x = int(round(cx - size / 2))
    y = int(round(cy - size / 2))
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return x, y, size


def detect_in_crop(crop_pose, rgb, cx, cy, size, prev):
    """Try the crop at four rotations, return (landmarks, rotation_k, score)."""
    crop = rgb[cy:cy + size, cx:cx + size]
    best, best_k, best_s = None, 0, -9
    for k in range(4):
        rot = rotate_image(crop, k)
        res = crop_pose.process(rot)
        lms = landmarks_to_dict(res, rot.shape[1], rot.shape[0])
        if lms is None:
            continue
        for n, v in lms.items():
            x, y = unrotate_point(v[0], v[1], k, size, size)
            lms[n] = [x + cx, y + cy, v[2]]
        s = score(lms, prev, size)
        if s > best_s:
            best, best_k, best_s = lms, k, s
    return best, best_k, best_s


def search_frame(crop_pose, rgb, w, h, prev, win=420, step=210):
    """Sliding-window search over the whole frame. Used when the track is lost."""
    best, best_k, best_s = None, 0, -9
    for y in range(0, max(1, h - win + 1), step):
        for x in range(0, max(1, w - win + 1), step):
            lms, k, s = detect_in_crop(crop_pose, rgb, x, y, min(win, w - x, h - y), prev)
            if lms is not None and s > best_s:
                best, best_k, best_s = lms, k, s
    return best, best_k, best_s


def track_frame(crop_pose, rgb, w, h, prev, prev_box):
    """Track one frame from the previous track. Returns (lms, k, score, crop)."""
    tries = []
    if prev is not None:
        tries.append(square_crop(bbox_from(prev, w, h), w, h, pad=1.9))
        tries.append(square_crop(bbox_from(prev, w, h), w, h, pad=2.8))
    elif prev_box is not None:
        tries.append(prev_box)
    for (cx, cy, size) in tries:
        lms, k, s = detect_in_crop(crop_pose, rgb, cx, cy, size, prev)
        if lms is not None and np.mean([lms[n][2] for n in CORE]) >= 0.5:
            return lms, k, s, (cx, cy, size)
    lms, k, s = search_frame(crop_pose, rgb, w, h, prev)
    if lms is not None:
        return lms, k, s, square_crop(bbox_from(lms, w, h), w, h)
    return None, 0, -9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--debug-dir", default=None, help="write per-frame overlay PNGs here")
    ap.add_argument("--min-det", type=float, default=0.3)
    ap.add_argument("--frames-dir", default=None,
                    help="also write every frame as JPEG here (underlay for pose-editor.html)")
    ap.add_argument("--min-vis", type=float, default=0.7,
                    help="frames whose core-joint visibility is below this are marked missing")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="frames whose continuity score is below this are marked missing")
    args = ap.parse_args()

    frames, fps = read_frames(args.video)
    if not frames:
        sys.exit("no frames read")
    h, w = frames[0].shape[:2]
    n = len(frames)
    print(f"{n} frames, {w}x{h}, {fps:.2f} fps")
    rgbs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in frames]
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        for i, im in enumerate(frames):
            cv2.imwrite(os.path.join(args.frames_dir, f"f_{i:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    full = MP_POSE.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=args.min_det, min_tracking_confidence=0.3)
    crop_pose = MP_POSE.Pose(static_image_mode=True, model_complexity=1,
                             min_detection_confidence=args.min_det)

    # Pass 1: full-frame tracking to find the best anchor frame.
    rough = []
    for si, rgb in enumerate(rgbs):
        lms = landmarks_to_dict(full.process(rgb), w, h)
        vis = float(np.mean([lms[k][2] for k in CORE])) if lms else 0.0
        rough.append((lms, vis))
        print(f"scan {si + 1}/{n}", flush=True)
    # Anchor: frame with the highest visibility, preferring a run of good frames.
    vis_arr = np.array([v for _, v in rough])
    run_score = np.convolve(vis_arr, np.ones(5) / 5, mode="same")
    anchor = int(np.argmax(run_score))
    print(f"anchor frame {anchor} (visibility {vis_arr[anchor]:.2f})")

    results = [None] * n
    for order in (range(anchor, n), range(anchor - 1, -1, -1)):
        prev, prev_box = None, None
        if order.start != anchor:
            prev = results[anchor]["landmarks"]
        for i in order:
            if i == anchor and results[anchor] is not None:
                prev = results[anchor]["landmarks"]
                continue
            if prev is None and rough[i][0] is not None and rough[i][1] >= 0.5:
                prev_box = square_crop(bbox_from(rough[i][0], w, h), w, h)
            lms, k, s, crop = track_frame(crop_pose, rgbs[i], w, h, prev, prev_box)
            vis = float(np.mean([lms[c][2] for c in CORE])) if lms else 0.0
            ok = bool(lms is not None and vis >= args.min_vis and s >= args.min_score)
            results[i] = {"frame": i, "t": i / fps, "ok": ok, "rotation": k * 90,
                          "score": float(s), "visibility": float(vis),
                          "crop": list(crop) if crop else None, "landmarks": lms if ok else None}
            print(f"frame {i}: {'ok ' if ok else 'BAD'} rot {k * 90:3d} vis {vis:.2f} score {s:.2f}", flush=True)
            if ok:
                prev, prev_box = lms, crop
            if args.debug_dir and lms is not None:
                dbg = frames[i].copy()
                if crop:
                    cv2.rectangle(dbg, (crop[0], crop[1]), (crop[0] + crop[2], crop[1] + crop[2]), (0, 255, 255), 1)
                for a, b in MP_POSE.POSE_CONNECTIONS:
                    if a in KEEP and b in KEEP:
                        pa, pb = lms[KEEP[a]], lms[KEEP[b]]
                        cv2.line(dbg, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 0, 255), 2)
                for name, v in lms.items():
                    cv2.circle(dbg, (int(v[0]), int(v[1])), 3, (0, 255, 0), -1)
                cv2.putText(dbg, f"{i} rot{k * 90} vis{vis:.2f}", (10, 30), 0, 0.8, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(args.debug_dir, f"pose_{i:04d}.png"), dbg)

    with open(args.out, "w") as f:
        json.dump({"video": os.path.basename(args.video), "width": w, "height": h,
                   "fps": fps, "anchor_frame": anchor, "frames": results}, f)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"wrote {args.out}: {n_ok}/{n} frames detected")


if __name__ == "__main__":
    main()
