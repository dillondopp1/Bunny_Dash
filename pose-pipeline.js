// Browser-side version of tools/extract_pose.py and tools/process_pose.py.
//
// Runs MediaPipe Pose as WebAssembly in the page, so a phone can turn a video
// into a vault JSON with nothing installed. The output matches what the Python
// pipeline writes, so the editor treats both the same.

// The runtime and models ship with the project so this works with no internet
// (a track with no signal), falling back to the CDN if they are missing.
const LOCAL_VISION = new URL('./vendor/tasks-vision', import.meta.url).href;
const LOCAL_MODELS = new URL('./vendor/models', import.meta.url).href;
const CDN_VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const CDN_MODELS = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker';

async function exists(url) {
  try { const r = await fetch(url, { method: 'HEAD' }); return r.ok; } catch (e) { return false; }
}

// MediaPipe landmark indices, named the way the rest of the project names them.
const KEEP = {
  0: 'nose', 7: 'l_ear', 8: 'r_ear', 11: 'l_shoulder', 12: 'r_shoulder',
  13: 'l_elbow', 14: 'r_elbow', 15: 'l_wrist', 16: 'r_wrist',
  23: 'l_hip', 24: 'r_hip', 25: 'l_knee', 26: 'r_knee',
  27: 'l_ankle', 28: 'r_ankle', 29: 'l_heel', 30: 'r_heel', 31: 'l_foot', 32: 'r_foot',
};
const JOINTS = Object.values(KEEP);
const CORE = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
              'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle'];
const ARM = ['shoulder', 'elbow', 'wrist'];
const LEG = ['hip', 'knee', 'ankle', 'heel', 'foot'];
const KEY_NAMES = ['Run', 'Plant', 'Takeoff', 'Swing', 'Rockback', 'Extension',
                   'Turn + clear', 'Fly away', 'Land'];
const BONES = { torso: 44, head: 16, upperArm: 24, forearm: 24, thigh: 32, shin: 32 };
const CANVAS_W = 1000, CANVAS_H = 430, GROUND_Y = 340, BOX_X = 560;

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const hypot = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);

// --------------------------------------------------------------------------
// Frames out of the video
// --------------------------------------------------------------------------
function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.preload = 'auto'; v.muted = true; v.playsInline = true;
    v.src = URL.createObjectURL(file);
    v.onloadedmetadata = () => resolve(v);
    v.onerror = () => reject(new Error('That video could not be read by this browser.'));
  });
}

async function guessFps(video) {
  // Play a moment and time the frames the browser actually presents.
  if (!video.requestVideoFrameCallback) return 30;
  const times = [];
  await new Promise(res => {
    let done = false;
    const finish = () => { if (!done) { done = true; video.pause(); res(); } };
    const step = (now, meta) => {
      times.push(meta.mediaTime);
      if (times.length >= 8) return finish();
      video.requestVideoFrameCallback(step);
    };
    video.requestVideoFrameCallback(step);
    video.play().catch(finish);
    setTimeout(finish, 1200);
  });
  const deltas = [];
  for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) deltas.push(times[i] - times[i - 1]);
  if (!deltas.length) return 30;
  deltas.sort((a, b) => a - b);
  const fps = 1 / deltas[Math.floor(deltas.length / 2)];
  // snap to a sensible camera rate
  return [24, 25, 30, 50, 60, 120, 240].reduce((a, b) => Math.abs(b - fps) < Math.abs(a - fps) ? b : a, 30);
}

function seek(video, t) {
  return new Promise(res => {
    const done = () => { video.removeEventListener('seeked', done); res(); };
    video.addEventListener('seeked', done);
    video.currentTime = t;
  });
}

async function extractFrames(video, fps, onProgress) {
  const w = video.videoWidth, h = video.videoHeight;
  const n = Math.max(1, Math.floor(video.duration * fps));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const frames = [];
  video.pause();
  for (let i = 0; i < n; i++) {
    await seek(video, Math.min((i + 0.5) / fps, Math.max(0, video.duration - 1e-3)));
    ctx.drawImage(video, 0, 0, w, h);
    const bitmap = await createImageBitmap(cv);
    const blob = await new Promise(r => cv.toBlob(r, 'image/jpeg', 0.85));
    frames.push({ bitmap, url: URL.createObjectURL(blob) });
    if (i % 3 === 0) onProgress(`frame ${i + 1} of ${n}`, 3 + 12 * (i + 1) / n);
  }
  return { frames, width: w, height: h };
}

// --------------------------------------------------------------------------
// Detection, with the same crop-and-four-rotations trick as the Python version
// --------------------------------------------------------------------------
const ROT = {                       // (x,y) in the crop -> where it is drawn
  0: (s) => [1, 0, 0, 1, 0, 0],
  1: (s) => [0, -1, 1, 0, 0, s],
  2: (s) => [-1, 0, 0, -1, s, s],
  3: (s) => [0, 1, -1, 0, s, 0],
};
function unrotate(x, y, k, s) {
  if (k === 0) return [x, y];
  if (k === 1) return [s - y, x];
  if (k === 2) return [s - x, s - y];
  return [y, s - x];
}

function toLandmarks(res, w, h) {
  if (!res || !res.landmarks || !res.landmarks.length) return null;
  const lm = res.landmarks[0];
  const out = {};
  for (const [idx, name] of Object.entries(KEEP)) {
    const p = lm[idx];
    if (!p) return null;
    out[name] = [p.x * w, p.y * h, p.visibility === undefined ? 1 : p.visibility];
  }
  return out;
}

function meanVis(l) { return CORE.reduce((a, n) => a + l[n][2], 0) / CORE.length; }

function score(l, prev, scale) {
  if (!l) return -1;
  const vis = meanVis(l);
  if (!prev) return vis;
  const d = [];
  for (const n of CORE) if (prev[n][2] > 0.5) d.push(hypot(l[n][0], l[n][1], prev[n][0], prev[n][1]));
  if (!d.length) return vis;
  d.sort((a, b) => a - b);
  return vis - 0.35 * (d[Math.floor(d.length / 2)] / Math.max(scale, 1));
}

function bboxOf(l) {
  const pts = JOINTS.map(n => l[n]).filter(p => p[2] > 0.3);
  if (pts.length < 4) return null;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function squareCrop(bbox, w, h, pad = 1.9, minSize = 260) {
  const [x0, y0, x1, y1] = bbox;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  let size = Math.max(Math.max(x1 - x0, y1 - y0) * pad, minSize);
  size = Math.round(Math.min(size, Math.min(w, h)));
  return [clamp(Math.round(cx - size / 2), 0, w - size), clamp(Math.round(cy - size / 2), 0, h - size), size];
}

class Detector {
  constructor(landmarker, w, h) {
    this.lm = landmarker; this.w = w; this.h = h;
    this.cv = document.createElement('canvas');
    this.ctx = this.cv.getContext('2d', { willReadFrequently: true });
  }
  inCrop(bitmap, cx, cy, size, prev) {
    let best = null, bestK = 0, bestS = -9;
    this.cv.width = size; this.cv.height = size;
    for (let k = 0; k < 4; k++) {
      const [a, b, c, d, e, f] = ROT[k](size);
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.clearRect(0, 0, size, size);
      this.ctx.setTransform(a, b, c, d, e, f);
      this.ctx.drawImage(bitmap, cx, cy, size, size, 0, 0, size, size);
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      const l = toLandmarks(this.lm.detect(this.cv), size, size);
      if (!l) continue;
      for (const n of JOINTS) {
        const [x, y] = unrotate(l[n][0], l[n][1], k, size);
        l[n] = [x + cx, y + cy, l[n][2]];
      }
      const s = score(l, prev, size);
      if (s > bestS) { best = l; bestK = k; bestS = s; }
    }
    return [best, bestK, bestS];
  }
  search(bitmap, prev, win = 420, step = 210) {
    let best = null, bestK = 0, bestS = -9;
    for (let y = 0; y < Math.max(1, this.h - win + 1); y += step) {
      for (let x = 0; x < Math.max(1, this.w - win + 1); x += step) {
        const size = Math.min(win, this.w - x, this.h - y);
        const [l, k, s] = this.inCrop(bitmap, x, y, size, prev);
        if (l && s > bestS) { best = l; bestK = k; bestS = s; }
      }
    }
    return [best, bestK, bestS];
  }
  track(bitmap, prev, prevBox) {
    const tries = [];
    if (prev) {
      const bb = bboxOf(prev);
      if (bb) { tries.push(squareCrop(bb, this.w, this.h, 1.9)); tries.push(squareCrop(bb, this.w, this.h, 2.8)); }
    } else if (prevBox) tries.push(prevBox);
    for (const [cx, cy, size] of tries) {
      const [l, k, s] = this.inCrop(bitmap, cx, cy, size, prev);
      if (l && meanVis(l) >= 0.5) return [l, k, s, [cx, cy, size]];
    }
    const [l, k, s] = this.search(bitmap, prev);
    if (l) {
      const bb = bboxOf(l);
      return [l, k, s, bb ? squareCrop(bb, this.w, this.h) : null];
    }
    return [null, 0, -9, null];
  }
}

// --------------------------------------------------------------------------
// Smoothing and geometry (the process_pose.py half)
// --------------------------------------------------------------------------
const SG7 = [-2, 3, 6, 7, 6, 3, -2].map(v => v / 21);   // Savitzky-Golay, window 7, cubic

function smoothSeries(v) {
  const n = v.length;
  if (n < 7) return v.slice();
  const out = v.slice();
  for (let i = 3; i < n - 3; i++) {
    let s = 0;
    for (let j = -3; j <= 3; j++) s += SG7[j + 3] * v[i + j];
    out[i] = s;
  }
  return out;
}

function medianOf(a) {
  const b = a.filter(v => Number.isFinite(v)).sort((x, y) => x - y);
  return b.length ? b[Math.floor(b.length / 2)] : NaN;
}

function fixSides(tracks, ok) {
  const n = ok.length;
  for (const group of [ARM, LEG]) {
    let prev = null;
    for (let i = 0; i < n; i++) {
      if (!ok[i]) continue;
      const L = {}, R = {};
      for (const j of group) { L[j] = tracks['l_' + j][i].slice(); R[j] = tracks['r_' + j][i].slice(); }
      if (prev) {
        let keep = 0, swap = 0;
        for (const j of group) {
          keep += hypot(L[j][0], L[j][1], prev[0][j][0], prev[0][j][1]) + hypot(R[j][0], R[j][1], prev[1][j][0], prev[1][j][1]);
          swap += hypot(L[j][0], L[j][1], prev[1][j][0], prev[1][j][1]) + hypot(R[j][0], R[j][1], prev[0][j][0], prev[0][j][1]);
        }
        if (swap < keep) {
          for (const j of group) { tracks['l_' + j][i] = R[j]; tracks['r_' + j][i] = L[j]; }
          prev = [R, L];
          continue;
        }
      }
      prev = [L, R];
    }
  }
}

function cleanAndSmooth(tracks, ok, torsoPx) {
  const n = ok.length;
  let first = ok.indexOf(true), last = ok.lastIndexOf(true);
  const out = {};
  for (const j of JOINTS) {
    const tr = tracks[j];
    const xy = tr.map((p, i) => (ok[i] && p[2] >= 0.3) ? [p[0], p[1]] : [NaN, NaN]);
    for (let k = 0; k < 2; k++) {
      for (let i = 0; i < n; i++) {
        const win = [];
        for (let d = -3; d <= 3; d++) { const q = xy[i + d]; if (q && Number.isFinite(q[k])) win.push(q[k]); }
        const med = medianOf(win);
        if (Number.isFinite(med) && Math.abs(xy[i][k] - med) > 0.6 * torsoPx) xy[i] = [NaN, NaN];
      }
    }
    const res = [];
    for (let k = 0; k < 2; k++) {
      const idx = [], val = [];
      xy.forEach((p, i) => { if (Number.isFinite(p[k])) { idx.push(i); val.push(p[k]); } });
      const col = [];
      for (let i = 0; i < n; i++) {
        if (!idx.length) { col.push(NaN); continue; }
        if (i <= idx[0]) { col.push(val[0]); continue; }
        if (i >= idx[idx.length - 1]) { col.push(val[val.length - 1]); continue; }
        let a = 0; while (idx[a + 1] < i) a++;
        const t = (i - idx[a]) / (idx[a + 1] - idx[a]);
        col.push(val[a] + (val[a + 1] - val[a]) * t);
      }
      res.push(smoothSeries(col.slice(first, last + 1)));
    }
    out[j] = [];
    for (let i = 0; i < n; i++) {
      out[j].push(i < first || i > last ? [NaN, NaN] : [res[0][i - first], res[1][i - first]]);
    }
  }
  return [out, first, last];
}

const angleDeg = (a, b) => Math.atan2(b[0] - a[0], -(b[1] - a[1])) * 180 / Math.PI;
function unwrapDeg(v) {
  const out = [v[0]];
  for (let i = 1; i < v.length; i++) {
    let d = v[i] - v[i - 1];
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    out.push(out[i - 1] + d);
  }
  return out;
}
function poleSag(chord, length) {
  return length <= chord ? 0 : Math.sqrt(3 * chord * (length - chord) / 8);
}

function buildVault(det, meta, name) {
  const { width: W, height: H, fps } = meta;
  const n = det.length;
  const tracks = {}; JOINTS.forEach(j => tracks[j] = det.map(d => d.ok ? d.landmarks[j] : [NaN, NaN, 0]));
  const ok = det.map(d => d.ok);
  if (!ok.some(Boolean)) throw new Error('No athlete was found in that clip.');
  fixSides(tracks, ok);

  const torsoLens = [];
  for (let i = 0; i < n; i++) {
    if (!ok[i]) continue;
    const hip = [(tracks.l_hip[i][0] + tracks.r_hip[i][0]) / 2, (tracks.l_hip[i][1] + tracks.r_hip[i][1]) / 2];
    const sho = [(tracks.l_shoulder[i][0] + tracks.r_shoulder[i][0]) / 2, (tracks.l_shoulder[i][1] + tracks.r_shoulder[i][1]) / 2];
    torsoLens.push(hypot(hip[0], hip[1], sho[0], sho[1]));
  }
  torsoLens.sort((a, b) => a - b);
  const torsoPx = torsoLens[Math.floor(0.85 * (torsoLens.length - 1))];
  const scale = BONES.torso / torsoPx;

  const [sm, first, last] = cleanAndSmooth(tracks, ok, torsoPx);
  const m = last - first + 1;
  const hipX = i => (sm.l_hip[i][0] + sm.r_hip[i][0]) / 2;
  const mid = first + Math.floor((last - first) / 2);
  const mirror = hipX(mid) < hipX(first);

  // ground level and the plant box, estimated from the line through both hands
  const footY = i => Math.max(...['l', 'r'].flatMap(s => ['heel', 'foot', 'ankle'].map(j => sm[s + '_' + j][i][1])));
  const early = [];
  for (let i = first; i < first + Math.max(3, Math.floor((last - first) / 4)); i++) early.push(footY(i));
  early.sort((a, b) => a - b);
  const groundPx = early[Math.min(early.length - 1, Math.floor(0.9 * (early.length - 1)))];
  let peak = first, best = Infinity;
  for (let i = first; i <= last; i++) {
    const hy = (sm.l_hip[i][1] + sm.r_hip[i][1]) / 2;
    if (hy < best) { best = hy; peak = i; }
  }
  let takeoff = first;
  for (let i = first; i < peak; i++) if (footY(i) > groundPx - 0.15 * torsoPx) takeoff = i;
  const xs = [];
  for (let i = Math.max(first, takeoff - 4); i <= takeoff; i++) {
    const a = sm.l_wrist[i], b = sm.r_wrist[i];
    if (!Number.isFinite(a[0]) || !Number.isFinite(b[0])) continue;
    const [top, bot] = a[1] < b[1] ? [a, b] : [b, a];
    const dy = bot[1] - top[1];
    if (dy < 0.15 * torsoPx) continue;
    xs.push(top[0] + (bot[0] - top[0]) * ((groundPx - top[1]) / dy));
  }
  const boxPx = [xs.length ? medianOf(xs) : hipX(takeoff), groundPx];

  const toCanvas = p => {
    const x = mirror ? W - p[0] : p[0];
    const bx = mirror ? W - boxPx[0] : boxPx[0];
    return [BOX_X + (x - bx) * scale, GROUND_Y + (p[1] - boxPx[1]) * scale];
  };

  const jointsC = [], hipsC = [], handsC = [], feetC = [];
  const torsoS = [], headS = [], segS = {};
  for (const k of ['lUpperArm', 'lForearm', 'rUpperArm', 'rForearm', 'lThigh', 'lShin', 'rThigh', 'rShin']) segS[k] = [];
  for (let i = first; i <= last; i++) {
    const J = {}; JOINTS.forEach(j => J[j] = toCanvas(sm[j][i]));
    const hip = [(J.l_hip[0] + J.r_hip[0]) / 2, (J.l_hip[1] + J.r_hip[1]) / 2];
    const sho = [(J.l_shoulder[0] + J.r_shoulder[0]) / 2, (J.l_shoulder[1] + J.r_shoulder[1]) / 2];
    let head = [(J.l_ear[0] + J.r_ear[0]) / 2, (J.l_ear[1] + J.r_ear[1]) / 2];
    if (!Number.isFinite(head[0])) head = J.nose;
    torsoS.push(angleDeg(hip, sho)); headS.push(angleDeg(sho, head));
    for (const s of ['l', 'r']) {
      segS[s + 'UpperArm'].push(angleDeg(J[s + '_shoulder'], J[s + '_elbow']));
      segS[s + 'Forearm'].push(angleDeg(J[s + '_elbow'], J[s + '_wrist']));
      segS[s + 'Thigh'].push(angleDeg(J[s + '_hip'], J[s + '_knee']));
      segS[s + 'Shin'].push(angleDeg(J[s + '_knee'], J[s + '_ankle']));
    }
    hipsC.push(hip); handsC.push([J.l_wrist, J.r_wrist]);
    feetC.push(Math.min(J.l_ankle[1], J.r_ankle[1], J.l_foot[1], J.r_foot[1]));
    const round = p => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10];
    const jj = {}; JOINTS.forEach(j => jj[j] = round(J[j]));
    jointsC.push(jj);
  }
  const torsoU = unwrapDeg(torsoS), headU = unwrapDeg(headS);
  const segU = {}; for (const k in segS) segU[k] = unwrapDeg(segS[k]);

  const boxC = [BOX_X, GROUND_Y];
  const topHand = handsC.map(h => hypot(h[0][0], h[0][1], boxC[0], boxC[1]) >= hypot(h[1][0], h[1][1], boxC[0], boxC[1]) ? h[0] : h[1]);
  const chord = topHand.map(p => hypot(p[0], p[1], boxC[0], boxC[1]));
  const tkLocal = takeoff - first, peakLocal = peak - first;
  const grip = chord[tkLocal];
  let plant = Math.max(0, tkLocal - 2);
  for (let i = 0; i <= tkLocal; i++) if (chord[i] <= grip * 1.03) { plant = i; break; }
  let release = m - 1;
  for (let i = peakLocal; i < m; i++) if (chord[i] > grip * 1.05) { release = i; break; }

  const poles = [];
  let relAngle = null;
  for (let i = 0; i < m; i++) {
    if (i < plant) {
      const plantDir = [boxC[0] - topHand[plant][0], boxC[1] - topHand[plant][1]];
      const plantAng = Math.atan2(plantDir[1], plantDir[0]);
      const carryAng = -25 * Math.PI / 180;
      let u = i / Math.max(plant, 1); u = u * u * (3 - 2 * u);
      const ang = carryAng + (plantAng - carryAng) * u;
      const tip = [topHand[i][0] + grip * Math.cos(ang), topHand[i][1] + grip * Math.sin(ang)];
      poles.push({ top: topHand[i], tip, bend: 0, sag: 0, angle: angleDeg(tip, topHand[i]), state: 'carry' });
    } else if (i < release) {
      poles.push({
        top: topHand[i], tip: boxC, bend: Math.max(0, 1 - chord[i] / grip),
        sag: poleSag(chord[i], grip), angle: angleDeg(boxC, topHand[i]), state: 'planted',
      });
    } else {
      if (relAngle === null) relAngle = angleDeg(boxC, topHand[Math.max(0, i - 1)]);
      const ang = relAngle - 2.5 * (i - release);
      const a = ang * Math.PI / 180;
      poles.push({
        top: [boxC[0] + grip * Math.sin(a), boxC[1] - grip * Math.cos(a)],
        tip: boxC, bend: 0, sag: 0, angle: ang, state: 'released',
      });
    }
  }

  const rot = torsoU.map(v => v - torsoU[tkLocal]);
  const sign = rot[peakLocal] < 0 ? -1 : 1;
  const r = rot.map(v => v * sign);
  const cross = (thr, from) => { for (let i = from; i < m; i++) if (r[i] >= thr) return i; return null; };
  const swing = cross(45, tkLocal) ?? Math.min(m - 1, tkLocal + 3);
  const rockback = cross(120, swing) ?? Math.min(m - 1, swing + 4);
  let extension = Math.min(m - 1, rockback + 3), bestFoot = Infinity;
  for (let i = rockback; i <= release; i++) if (r[i] >= 120 && r[i] <= 240 && feetC[i] < bestFoot) { bestFoot = feetC[i]; extension = i; }
  let fly = cross(270, peakLocal);
  if (fly === null) { fly = Math.min(m - 1, peakLocal + 4); for (let i = peakLocal; i < m; i++) if (r[i] < 90) { fly = i; break; } }
  const keys = { Run: Math.max(0, plant - 5), Plant: plant, Takeoff: tkLocal, Swing: swing, Rockback: rockback,
                 Extension: extension, 'Turn + clear': peakLocal, 'Fly away': fly, Land: m - 1 };

  const round1 = v => Math.round(v * 10) / 10;
  const frames = [];
  for (let i = 0; i < m; i++) {
    const angles = { torso: round1(torsoU[i]), head: round1(headU[i]) };
    for (const k in segU) angles[k] = round1(segU[k][i]);
    const p = poles[i];
    frames.push({
      frame: i + first, t: Math.round((i + first) / fps * 1e4) / 1e4,
      hip: [round1(hipsC[i][0]), round1(hipsC[i][1])],
      angles,
      pole: { top: [round1(p.top[0]), round1(p.top[1])], tip: [round1(p.tip[0]), round1(p.tip[1])],
              bend: Math.round(p.bend * 1000) / 1000, sag: round1(p.sag), angle: round1(p.angle), state: p.state },
      joints: jointsC[i],
      quality: Math.round((det[i + first].visibility || 0) * 100) / 100,
    });
  }

  return {
    name,
    source: { video: meta.filename, fps, width: W, height: H, mirrored: mirror,
              box_px: [round1(boxPx[0]), round1(boxPx[1])], ground_px: round1(groundPx),
              torso_px: round1(torsoPx), scale: Math.round(scale * 1e5) / 1e5,
              first_frame: first, last_frame: last },
    canvas: { width: CANVAS_W, height: CANVAS_H, groundY: GROUND_Y, boxX: BOX_X, barY: null },
    bones: BONES,
    poleStyle: { bendToward: 'pit', bendScale: 1 },
    angleConvention: 'degrees, 0 = straight up, 90 = forward toward the pit, 180 = down; unwrapped (continuous)',
    pole: { gripLength: round1(grip), plantFrame: plant + first, takeoffFrame: tkLocal + first, releaseFrame: release + first },
    keys: KEY_NAMES.map(k => ({ name: k, frame: clamp(keys[k], 0, m - 1) + first })),
    frames,
  };
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------
let landmarkerPromise = null;
async function getLandmarker(quality, onProgress) {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    onProgress('Loading the pose model (first time only)', 1);
    const local = await exists(`${LOCAL_VISION}/vision_bundle.mjs`);
    const visionBase = local ? LOCAL_VISION : CDN_VISION;
    const q = quality === 'lite' ? 'lite' : 'full';
    const model = local
      ? `${LOCAL_MODELS}/pose_landmarker_${q}.task`
      : `${CDN_MODELS}/pose_landmarker_${q}/float16/1/pose_landmarker_${q}.task`;
    const vision = await import(/* webpackIgnore: true */ `${visionBase}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`${visionBase}/wasm`);
    return vision.PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: model, delegate: 'CPU' },
      runningMode: 'IMAGE', numPoses: 1,
      minPoseDetectionConfidence: 0.3, minPosePresenceConfidence: 0.3, minTrackingConfidence: 0.3,
    });
  })();
  return landmarkerPromise;
}

export async function processVideoInBrowser(file, opts, onProgress) {
  const quality = (opts && opts.quality) || 'full';
  const minVis = 0.7, minScore = 0.5;
  const report = (stage, pct, message) => onProgress({ stage, pct, message: message || '' });

  const landmarker = await getLandmarker(quality, (s, p) => report(s, p));
  report('Reading the video', 2);
  const video = await loadVideo(file);
  const fps = (opts && opts.fps) || await guessFps(video);
  report('Splitting the clip into frames', 3, `${fps} frames a second`);
  const { frames, width, height } = await extractFrames(video, fps, (msg, pct) => report('Splitting the clip into frames', pct, msg));
  const n = frames.length;

  const detector = new Detector(landmarker, width, height);
  // Pass 1: find the frame where the athlete reads most clearly.
  const rough = [];
  for (let i = 0; i < n; i++) {
    const [l] = detector.inCrop(frames[i].bitmap, 0, 0, Math.min(width, height), null);
    rough.push(l);
    if (i % 2 === 0) report('Looking for the athlete', 15 + 25 * (i + 1) / n, `frame ${i + 1} of ${n}`);
    if (i % 8 === 0) await new Promise(r => setTimeout(r));    // let the page breathe
  }
  const visArr = rough.map(l => l ? meanVis(l) : 0);
  let anchor = 0, bestRun = -1;
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let d = -2; d <= 2; d++) if (visArr[i + d] !== undefined) { s += visArr[i + d]; c++; }
    if (s / c > bestRun) { bestRun = s / c; anchor = i; }
  }

  const det = new Array(n);
  let processed = 0;
  for (const forward of [true, false]) {
    let prev = null, prevBox = null;
    const order = [];
    if (forward) for (let i = anchor; i < n; i++) order.push(i);
    else { for (let i = anchor - 1; i >= 0; i--) order.push(i); prev = det[anchor] && det[anchor].ok ? det[anchor].landmarks : null; }
    for (const i of order) {
      if (!prev && rough[i] && visArr[i] >= 0.5) {
        const bb = bboxOf(rough[i]);
        if (bb) prevBox = squareCrop(bb, width, height);
      }
      const [l, k, s, crop] = detector.track(frames[i].bitmap, prev, prevBox);
      const vis = l ? meanVis(l) : 0;
      const good = !!l && vis >= minVis && s >= minScore;
      det[i] = { frame: i, ok: good, rotation: k * 90, score: s, visibility: vis, landmarks: good ? l : null };
      if (good) { prev = l; prevBox = crop; }
      processed++;
      if (processed % 2 === 0) report('Tracking the body through the vault', 40 + 55 * processed / n, `frame ${processed} of ${n}`);
      if (processed % 8 === 0) await new Promise(r => setTimeout(r));
    }
  }

  report('Smoothing and building the vault', 96);
  const name = (file.name || 'vault').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '-').toLowerCase() || 'vault';
  const vault = buildVault(det, { width, height, fps, filename: file.name || 'clip' }, name);
  const images = {};
  vault.frames.forEach(f => images[f.frame] = frames[f.frame].url);
  frames.forEach(f => f.bitmap.close && f.bitmap.close());
  URL.revokeObjectURL(video.src);
  report('Ready', 100, `${vault.frames.length} frames tracked`);
  return { vault, images };
}
