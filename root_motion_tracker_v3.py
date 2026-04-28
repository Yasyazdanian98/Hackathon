"""
Loco Tracker V3 - MotionBuilder Plugin
---------------------------------------
World-space yaw via FBMatrix forward axis (column 2).

Yaw chain:
    raw world yaw  ->  unwrap
                   ->  median(window=11)            spike rejection
                   ->  rolling-mean(~gait cycle)    sway rejection
                   ->  windowed net yaw diff(T)     ang_vel ground truth
                   ->  TurnEventDetector

Foot anchoring (live, retroactive):
    Per-frame foot contact detection (height + speed + hysteresis).
    On contact: compare foot world yaw vs trajectory yaw (signed cross).
    If |delta| > threshold, retroactively rewrite buffered yaw history
    inside a backward smoothstep blend window toward foot yaw.

Lag correction:
    PathCanvas turn markers placed at (event_start_t - group_delay), where
    group_delay = (M-1)/2/fps + mean_secs/2 + diff_T/2.

Advanced thresholds collapsible at bottom.

Drop into MotionBuilder Python Editor and press Execute.
"""

from pyfbsdk import *
from collections import deque
import math
import time
import bisect
import traceback

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    _PYSIDE = 6
except ImportError:
    from PySide2 import QtWidgets, QtGui, QtCore
    _PYSIDE = 2


TOOL_NAME = "Loco Tracker V3"

# ---------------------------------------------------------------------------
# JOINT CANDIDATES (no mixamo)
# ---------------------------------------------------------------------------
ROOT_CANDIDATES = [
    "Hips", "hips",
    "Reference", "reference",
    "Root", "root",
    "Character_Hips",
    "Bip01",
]
LEFT_FOOT_CANDIDATES  = ["LeftFoot",  "leftFoot",  "L_Foot", "LFoot",
                         "Bip01_L_Foot", "LeftToeBase"]
RIGHT_FOOT_CANDIDATES = ["RightFoot", "rightFoot", "R_Foot", "RFoot",
                         "Bip01_R_Foot", "RightToeBase"]


# ---------------------------------------------------------------------------
# CONFIG (mutable - bound to advanced spinboxes)
# ---------------------------------------------------------------------------
CFG = {
    "sample_min_dt":         0.020,
    "units_per_meter":       100.0,

    # Yaw chain
    "median_window":         11,
    "mean_secs":             0.50,    # gait sway rejection window
    "diff_T":                1.00,    # windowed net yaw differentiation
    "gait_auto":             True,

    # Turn detection
    "turn_start_angvel":     15.0,
    "turn_end_angvel":       5.0,
    "turn_end_hold":         0.30,
    "turn_flip_hold":        0.15,
    "turn_min_angle":        25.0,
    "turn_soft_angle":       60.0,
    "turn_min_peak_angvel":  110.0,

    # Stop detection
    "stop_moving_mps":       0.80,
    "stop_still_mps":        0.50,
    "stop_hold_secs":        1.00,

    # Foot contact
    "foot_height_max_cm":    8.0,
    "foot_speed_max_cmps":   30.0,
    "foot_hold_frames":      2,

    # Foot turn classifier
    "foot_turn_thresh_deg":  15.0,
    "foot_blend_frames":     6,
}

SPEED_BUCKETS = [
    ("Walk",   0.05, 1.50, 60.0),
    ("Jog",    1.50, 3.00, 45.0),
    ("Run",    3.00, 5.00, 30.0),
    ("Sprint", 5.00, 1e9,  5.0),
]
TURN_MAG_BUCKETS = [("30-60",   30.0,  60.0),
                    ("60-120",  60.0, 120.0),
                    ("120-160", 120.0, 160.0),
                    ("160+",    160.0, 1e9)]

PATH_MAX_POINTS = 10000
VIEW_HEIGHT     = 300
VIEW_MARGIN     = 0.12
VIEW_MIN_SPAN   = 200.0


def log(msg):
    print("[LocoTrackerV3] {}".format(msg))


# =============================================================================
# HELPERS
# =============================================================================
def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def signed_yaw_delta(yaw_a, yaw_b):
    """Shortest signed delta from yaw_a to yaw_b (radians). Result in [-pi, pi]."""
    d = yaw_b - yaw_a
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def speed_to_color(speed):
    if speed is None or speed < 0.05:
        return (128, 128, 128)
    elif speed < 1.50:
        return (51, 230, 51)
    elif speed < 3.00:
        return (230, 230, 26)
    elif speed < 5.00:
        return (255, 128, 0)
    else:
        return (255, 26, 26)


def find_joint(candidates):
    for n in candidates:
        m = FBFindModelByLabelName(n)
        if m is not None:
            return m, n
    return None, None


def world_matrix_list(node):
    m = FBMatrix()
    node.GetMatrix(m, FBModelTransformationType.kModelTransformation, True)
    return [m[i] for i in range(16)]


def world_pos_yaw(node):
    """Return ((x, y, z), yaw_rad). yaw=None if forward-XZ degenerate."""
    m = world_matrix_list(node)
    px, py, pz = m[12], m[13], m[14]
    fx, fz = m[8], m[10]
    L = math.sqrt(fx * fx + fz * fz)
    if L < 1e-4:
        return (px, py, pz), None
    return (px, py, pz), math.atan2(fx / L, fz / L)


# =============================================================================
# SMOOTHERS
# =============================================================================
class RollingSmoother:
    def __init__(self, window):
        self.buf = deque(maxlen=window)
    def smooth(self, v):
        self.buf.append(v)
        return sum(self.buf) / len(self.buf)
    def reset(self):
        self.buf.clear()


# =============================================================================
# YAW PROCESSOR (median -> time-windowed mean -> windowed net yaw diff)
# =============================================================================
class YawProcessor:
    def __init__(self):
        self.median_buf     = deque()
        self.mean_buf       = deque()   # (t, after_median_unwrapped)
        self.history        = deque()   # (t, filtered_unwrapped) - rewriteable
        self.last_unwrapped = None

    def reset(self):
        self.median_buf.clear()
        self.mean_buf.clear()
        self.history.clear()
        self.last_unwrapped = None

    def push(self, t, raw_yaw_rad):
        # 1. Unwrap relative to running unwrapped value
        if self.last_unwrapped is None:
            unwrapped = raw_yaw_rad
        else:
            base_w = ((self.last_unwrapped + math.pi) % (2.0 * math.pi)) - math.pi
            d = signed_yaw_delta(base_w, raw_yaw_rad)
            unwrapped = self.last_unwrapped + d
        self.last_unwrapped = unwrapped

        # 2. Median over last M unwrapped samples
        M = max(1, int(CFG["median_window"]))
        self.median_buf.append(unwrapped)
        while len(self.median_buf) > M:
            self.median_buf.popleft()
        med = sorted(self.median_buf)[len(self.median_buf) // 2]

        # 3. Time-windowed mean (gait-cycle low-pass)
        win_s = max(0.05, float(CFG["mean_secs"]))
        self.mean_buf.append((t, med))
        cutoff = t - win_s
        while self.mean_buf and self.mean_buf[0][0] < cutoff:
            self.mean_buf.popleft()
        avg = sum(v for _, v in self.mean_buf) / len(self.mean_buf)

        # 4. Append to history; trim
        self.history.append((t, avg))
        T = float(CFG["diff_T"])
        keep = T + max(2.0, win_s + 1.0)
        while self.history and self.history[0][0] < t - keep:
            self.history.popleft()

        # 5. Windowed net yaw differentiation -> ang_vel (deg/s)
        ang_vel = self._windowed_ang_vel(t)
        return (avg, ang_vel)

    def _windowed_ang_vel(self, now_t):
        T = float(CFG["diff_T"])
        target = now_t - T
        if not self.history or self.history[0][0] > target:
            return None
        times = [s[0] for s in self.history]
        idx = bisect.bisect_left(times, target)
        if idx >= len(times):
            return None
        if idx > 0 and (target - times[idx - 1]) < (times[idx] - target):
            idx -= 1
        old_t, old_y = self.history[idx]
        cur_t, cur_y = self.history[-1]
        dt = cur_t - old_t
        if dt <= 1e-6:
            return None
        return math.degrees(cur_y - old_y) / dt

    def apply_anchor(self, contact_t, foot_yaw_rad, blend_secs):
        """Smoothstep-blend buffered yaw in [contact_t - blend_secs, contact_t] toward foot_yaw."""
        if not self.history or blend_secs <= 0:
            return
        # Align foot_yaw to nearest history sample's unwrap frame
        nearest_y = None
        nearest_d = None
        for (t, y) in self.history:
            d = abs(t - contact_t)
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest_y = y
        if nearest_y is None:
            return
        wrapped = ((nearest_y + math.pi) % (2.0 * math.pi)) - math.pi
        target_yaw = nearest_y + signed_yaw_delta(wrapped, foot_yaw_rad)

        new_hist = deque()
        for (t, y) in self.history:
            if t < contact_t - blend_secs or t > contact_t:
                new_hist.append((t, y))
                continue
            u = 1.0 - (contact_t - t) / blend_secs   # 0 at edge, 1 at contact
            w = smoothstep(u)
            new_hist.append((t, y * (1.0 - w) + target_yaw * w))
        self.history = new_hist


# =============================================================================
# FOOT CONTACT DETECTOR (height + speed + hysteresis)
# =============================================================================
class FootContactDetector:
    def __init__(self, candidates, side):
        self.candidates = candidates
        self.side = side
        self.joint = None
        self.in_contact = False
        self.satisfy = 0
        self.unsatisfy = 0
        self.prev_pos = None
        self.prev_t = None

    def resolve(self):
        if self.joint is not None:
            return self.joint
        m, name = find_joint(self.candidates)
        if m is not None:
            self.joint = m
            log("Foot ({}) joint: '{}'".format(self.side, name))
        return m

    def update(self, scene_t):
        j = self.resolve()
        if j is None:
            return None
        m = world_matrix_list(j)
        wx, wy, wz = m[12], m[13], m[14]
        fx, fz = m[8], m[10]
        L = math.sqrt(fx * fx + fz * fz)
        foot_yaw = math.atan2(fx / L, fz / L) if L > 1e-4 else None

        speed = 0.0
        if self.prev_pos is not None and self.prev_t is not None:
            dt = scene_t - self.prev_t
            if 0.0 < dt < 0.5:
                dx = wx - self.prev_pos[0]
                dy = wy - self.prev_pos[1]
                dz = wz - self.prev_pos[2]
                speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
        self.prev_pos = (wx, wy, wz)
        self.prev_t = scene_t

        h_max = float(CFG["foot_height_max_cm"])
        s_max = float(CFG["foot_speed_max_cmps"])
        hold  = max(1, int(CFG["foot_hold_frames"]))

        ok = (wy < h_max) and (speed < s_max)
        contact = None
        if ok:
            self.satisfy += 1
            self.unsatisfy = 0
            if not self.in_contact and self.satisfy >= hold:
                self.in_contact = True
                contact = {
                    "side": self.side,
                    "scene_t": scene_t,
                    "pos_xz": (wx, wz),
                    "foot_yaw": foot_yaw,
                }
        else:
            self.unsatisfy += 1
            self.satisfy = 0
            if self.in_contact and self.unsatisfy >= hold:
                self.in_contact = False
        return contact

    def reset(self):
        self.in_contact = False
        self.satisfy = 0
        self.unsatisfy = 0
        self.prev_pos = None
        self.prev_t = None


# =============================================================================
# GAIT CYCLE ESTIMATOR (median of same-foot intervals)
# =============================================================================
class GaitCycleEstimator:
    def __init__(self):
        self.last = {"Left": None, "Right": None}
        self.intervals = deque(maxlen=8)
        self.estimate = None

    def push(self, side, t):
        prev = self.last[side]
        if prev is not None:
            interval = t - prev
            if 0.15 < interval < 2.0:
                self.intervals.append(interval)
                s = sorted(self.intervals)
                self.estimate = s[len(s) // 2]
        self.last[side] = t

    def reset(self):
        self.last = {"Left": None, "Right": None}
        self.intervals.clear()
        self.estimate = None


# =============================================================================
# ROOT SAMPLER
# =============================================================================
class RootSampler:
    def __init__(self):
        self.candidates = ROOT_CANDIDATES
        self.joint = None
        self.prev_pos = None
        self.prev_t = None

    def resolve(self):
        if self.joint is not None:
            return self.joint
        m, name = find_joint(self.candidates)
        if m is not None:
            self.joint = m
            log("Root joint: '{}'".format(name))
        return m

    def sample(self):
        j = self.resolve()
        if j is None:
            return None
        (px, py, pz), yaw = world_pos_yaw(j)
        scene_t = FBSystem().LocalTime.GetSecondDouble()
        speed_mps = None
        dt = 0.0
        if self.prev_pos is not None and self.prev_t is not None:
            dt = scene_t - self.prev_t
            if dt <= 0.0 or dt > 0.2:
                dt = 0.0
            else:
                dx = px - self.prev_pos[0]
                dz = pz - self.prev_pos[2]
                dist_cm = math.sqrt(dx * dx + dz * dz)
                speed_mps = (dist_cm / float(CFG["units_per_meter"])) / dt
        self.prev_pos = (px, py, pz)
        self.prev_t = scene_t
        return {
            "pos_xyz": (px, py, pz),
            "yaw": yaw,
            "scene_t": scene_t,
            "speed_mps": speed_mps,
            "dt": dt,
            "name": j.LongName,
        }

    def reset(self):
        self.prev_pos = None
        self.prev_t = None

    def get_current_pos_xz(self):
        j = self.resolve()
        if j is None:
            return None
        m = world_matrix_list(j)
        return (m[12], m[14])


# =============================================================================
# COVERAGE / TURN / STOP
# =============================================================================
class SpeedCoverage:
    def __init__(self, buckets):
        self.buckets = buckets
        self.time_in = {b[0]: 0.0 for b in buckets}
        self.targets = {b[0]: b[3] for b in buckets}
    def update(self, speed, dt):
        for b in self.buckets:
            if b[1] <= speed < b[2]:
                self.time_in[b[0]] += dt
                return b[0]
        return None
    def pct(self, name):
        t = self.targets.get(name, 1.0)
        return min(self.time_in.get(name, 0.0) / t, 1.0)
    def reset(self):
        for k in self.time_in:
            self.time_in[k] = 0.0


class TurnEventDetector:
    def __init__(self):
        self.active = False
        self.direction = 0
        self.accum_deg = 0.0
        self.duration = 0.0
        self.below_time = 0.0
        self.flip_time = 0.0
        self.peak_angvel = 0.0
        self.start_t = 0.0
        self.end_t = 0.0
        self.counts = {"Left": {}, "Right": {}}
        for side in ("Left", "Right"):
            for (lbl, _, _) in TURN_MAG_BUCKETS:
                self.counts[side][lbl] = 0

    def _start(self, ang_vel, dt, scene_t):
        self.active = True
        self.direction = 1 if ang_vel > 0 else -1
        self.accum_deg = abs(ang_vel) * dt
        self.duration = dt
        self.below_time = 0.0
        self.flip_time = 0.0
        self.peak_angvel = abs(ang_vel)
        self.start_t = scene_t - dt
        self.end_t = scene_t

    def update(self, ang_vel, dt, scene_t):
        if ang_vel is None:
            return None
        START     = float(CFG["turn_start_angvel"])
        END       = float(CFG["turn_end_angvel"])
        END_HOLD  = float(CFG["turn_end_hold"])
        FLIP_HOLD = float(CFG["turn_flip_hold"])
        MIN_A     = float(CFG["turn_min_angle"])
        SOFT      = float(CFG["turn_soft_angle"])
        MIN_PEAK  = float(CFG["turn_min_peak_angvel"])

        mag = abs(ang_vel)
        if not self.active:
            if mag >= START:
                self._start(ang_vel, dt, scene_t)
            return None
        sign = 1 if ang_vel > 0 else -1
        if sign != self.direction and mag >= START:
            self.flip_time += dt
            if self.flip_time >= FLIP_HOLD:
                ev = self._close(MIN_A, SOFT, MIN_PEAK)
                self._start(ang_vel, dt, scene_t)
                return ev
            return None
        else:
            self.flip_time = 0.0
        self.accum_deg += mag * dt
        self.duration  += dt
        if mag > self.peak_angvel:
            self.peak_angvel = mag
        if mag < END:
            self.below_time += dt
            if self.below_time >= END_HOLD:
                return self._close(MIN_A, SOFT, MIN_PEAK)
        else:
            self.below_time = 0.0
            self.end_t = scene_t
        return None

    def _close(self, MIN_A, SOFT, MIN_PEAK):
        total = self.accum_deg
        duration = self.duration
        direction = self.direction
        peak = self.peak_angvel
        start_t = self.start_t
        end_t = self.end_t
        self.active = False
        self.accum_deg = 0.0
        self.duration = 0.0
        self.below_time = 0.0
        self.flip_time = 0.0
        self.peak_angvel = 0.0
        if total < MIN_A:
            return None
        if total < SOFT and peak < MIN_PEAK:
            return None
        side = "Left" if direction > 0 else "Right"
        mag_label = TURN_MAG_BUCKETS[0][0]
        for (lbl, lo, hi) in TURN_MAG_BUCKETS:
            if lo <= total < hi:
                mag_label = lbl
                break
        self.counts[side][mag_label] += 1
        log("Turn: {} {}  {:.0f} deg  {:.2f}s  peak {:.0f} deg/s".format(
            side, mag_label, total, duration, peak))
        return {"side": side, "mag_label": mag_label,
                "total_deg": total, "duration": duration,
                "peak_angvel": peak,
                "start_t": start_t, "end_t": end_t}

    def reset(self):
        self.__init__()


class StopDetector:
    def __init__(self):
        self.was_moving = False
        self.still_time = 0.0
        self.count = 0
    def update(self, speed, dt):
        if speed is None:
            return False
        MOV  = float(CFG["stop_moving_mps"])
        STL  = float(CFG["stop_still_mps"])
        HOLD = float(CFG["stop_hold_secs"])
        if speed >= MOV:
            self.was_moving = True
            self.still_time = 0.0
            return False
        if speed < STL and self.was_moving:
            self.still_time += dt
            if self.still_time >= HOLD:
                self.count += 1
                self.was_moving = False
                self.still_time = 0.0
                log("Stop detected. Total: {}".format(self.count))
                return True
        else:
            self.still_time = 0.0
        return False
    def reset(self):
        self.__init__()


# =============================================================================
# 2D PATH CANVAS (Qt) - lag-corrected turn marker placement
# =============================================================================
class PathCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PathCanvas, self).__init__(parent)
        self.setMinimumHeight(VIEW_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor(33, 33, 33))
        self.setPalette(pal)

    @staticmethod
    def _lag_secs():
        fps = max(1.0, _STATE.get("fps", 30.0))
        M = max(1, int(CFG["median_window"]))
        return ((M - 1) / 2.0) / fps + float(CFG["mean_secs"]) / 2.0 + float(CFG["diff_T"]) / 2.0

    @staticmethod
    def _path_pos_at(target_t, pts, times):
        """Given path_points list and pre-extracted times list, find (x,z) closest to target_t."""
        if not pts:
            return None
        if not times:
            return (pts[-1][0], pts[-1][1])
        idx = bisect.bisect_left(times, target_t)
        if idx <= 0:
            return (pts[0][0], pts[0][1])
        if idx >= len(times):
            return (pts[-1][0], pts[-1][1])
        if (target_t - times[idx - 1]) < (times[idx] - target_t):
            idx -= 1
        return (pts[idx][0], pts[idx][1])

    def paintEvent(self, ev):
        try:
            pts = list(_STATE["path_points"])
            cur = _STATE.get("current_pos")
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            w = self.width()
            h = self.height()
            p.fillRect(0, 0, w, h, QtGui.QColor(33, 33, 33))

            if len(pts) < 2 and cur is None:
                p.setPen(QtGui.QPen(QtGui.QColor(64, 64, 64), 1))
                p.drawLine(w // 2, 0, w // 2, h)
                p.drawLine(0, h // 2, w, h // 2)
                p.setPen(QtGui.QColor(120, 120, 120))
                p.drawText(10, 18, "Waiting for character... (scrub or Play)")
                return

            xs = [pt[0] for pt in pts]
            zs = [pt[1] for pt in pts]
            if cur is not None:
                xs.append(cur[0])
                zs.append(cur[1])
            min_x, max_x = min(xs), max(xs)
            min_z, max_z = min(zs), max(zs)
            span_x = max(max_x - min_x, VIEW_MIN_SPAN) * (1.0 + 2.0 * VIEW_MARGIN)
            span_z = max(max_z - min_z, VIEW_MIN_SPAN) * (1.0 + 2.0 * VIEW_MARGIN)
            cx = (min_x + max_x) * 0.5
            cz = (min_z + max_z) * 0.5
            scale = min(w / span_x, h / span_z)

            def to_screen(wx, wz):
                return (w * 0.5 + (wx - cx) * scale,
                        h * 0.5 - (wz - cz) * scale)

            ox, oy = to_screen(0.0, 0.0)
            p.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70), 1))
            p.drawLine(int(ox), 0, int(ox), h)
            p.drawLine(0, int(oy), w, int(oy))

            # 1m scale bar
            bar_px = int(float(CFG["units_per_meter"]) * scale)
            if 8 < bar_px < w - 40:
                p.setPen(QtGui.QPen(QtGui.QColor(160, 160, 160), 2))
                bx = w - bar_px - 12
                by = h - 14
                p.drawLine(bx, by, bx + bar_px, by)
                p.drawLine(bx, by - 4, bx, by + 4)
                p.drawLine(bx + bar_px, by - 4, bx + bar_px, by + 4)
                p.setPen(QtGui.QColor(180, 180, 180))
                p.drawText(bx, by - 6, "1 m")

            # Path
            if len(pts) >= 2:
                prev_sx = prev_sy = None
                for (wx, wz, spd, _t) in pts:
                    sx, sy = to_screen(wx, wz)
                    if prev_sx is not None:
                        r, g, b = speed_to_color(spd)
                        p.setPen(QtGui.QPen(QtGui.QColor(r, g, b), 2))
                        p.drawLine(int(prev_sx), int(prev_sy), int(sx), int(sy))
                    prev_sx = sx
                    prev_sy = sy

            # Foot contacts (optional)
            if _STATE["show_contacts"][0]:
                for c in _STATE["foot_contacts"]:
                    sx, sy = to_screen(c["pos_xz"][0], c["pos_xz"][1])
                    col = QtGui.QColor(80, 160, 255) if c["side"] == "Left" else QtGui.QColor(255, 160, 80)
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(col)
                    p.drawEllipse(QtCore.QPointF(sx, sy), 2.5, 2.5)

            # Stops - red squares
            if _STATE["show_stops"][0]:
                p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
                p.setBrush(QtGui.QColor(230, 40, 40))
                for ev in _STATE["stop_events"]:
                    sx, sy = to_screen(ev["pos_xz"][0], ev["pos_xz"][1])
                    p.drawRect(QtCore.QRectF(sx - 4, sy - 4, 8, 8))

            # Turns - hollow circles, lag-corrected position from path lookup
            if _STATE["show_turns"][0] and _STATE["turn_events"]:
                lag = self._lag_secs()
                times = [pt[3] for pt in pts]
                font = p.font()
                font.setPointSize(7)
                p.setFont(font)
                for ev in _STATE["turn_events"]:
                    target_t = ev["start_t"] - lag
                    placed = self._path_pos_at(target_t, pts, times)
                    if placed is None:
                        placed = ev["pos_xz"]
                    sx, sy = to_screen(placed[0], placed[1])
                    if ev["side"] == "Left":
                        col = QtGui.QColor(0, 200, 230)
                    else:
                        col = QtGui.QColor(230, 60, 200)
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.setPen(QtGui.QPen(col, 2))
                    p.drawEllipse(QtCore.QPointF(sx, sy), 6, 6)
                    p.setPen(col)
                    p.drawText(int(sx + 8), int(sy + 4),
                               "{}{:.0f}".format(ev["side"][0], ev["total_deg"]))

            # Foot anchor marks (small triangles where retroactive yaw rewrite fired)
            if _STATE["show_anchors"][0]:
                for a in _STATE["anchor_events"]:
                    sx, sy = to_screen(a["pos_xz"][0], a["pos_xz"][1])
                    col = QtGui.QColor(255, 220, 0)
                    p.setPen(QtGui.QPen(col, 1))
                    p.setBrush(col)
                    poly = QtGui.QPolygonF([
                        QtCore.QPointF(sx,     sy - 5),
                        QtCore.QPointF(sx - 5, sy + 4),
                        QtCore.QPointF(sx + 5, sy + 4),
                    ])
                    p.drawPolygon(poly)

            # Start dot - green
            if pts:
                sx0, sy0 = to_screen(pts[0][0], pts[0][1])
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(0, 220, 0))
                p.drawEllipse(QtCore.QPointF(sx0, sy0), 5, 5)

            # Live marker
            if cur is not None:
                mx, my = to_screen(cur[0], cur[1])
            elif pts:
                mx, my = to_screen(pts[-1][0], pts[-1][1])
            else:
                mx = my = None
            if mx is not None:
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                p.setBrush(QtGui.QColor(255, 255, 255))
                p.drawEllipse(QtCore.QPointF(mx, my), 7, 7)
        except Exception:
            log("PathCanvas paint error:")
            traceback.print_exc()


# =============================================================================
# MAIN WINDOW
# =============================================================================
class LocoTrackerWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(LocoTrackerWindow, self).__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle(TOOL_NAME)
        self.resize(520, 920)
        self._build_ui()

    @staticmethod
    def _section(text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #888; margin-top: 4px;")
        return lbl

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        # Header
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Root:"))
        self.root_label = QtWidgets.QLabel("(searching)")
        row.addWidget(self.root_label, 1)
        v.addLayout(row)

        self.play_label   = QtWidgets.QLabel("Status:   STOPPED (click Play to start)")
        self.speed_label  = QtWidgets.QLabel("Speed:    -- m/s")
        self.angvel_label = QtWidgets.QLabel("Turn:     -- deg/s")
        self.state_label  = QtWidgets.QLabel("State:    --")
        self.gait_label   = QtWidgets.QLabel("Gait:     -- s")
        for w in (self.play_label, self.speed_label, self.angvel_label,
                  self.state_label, self.gait_label):
            v.addWidget(w)

        # Path viewer
        v.addWidget(self._section("--- Path (top-down XZ view) ---"))
        self.canvas = PathCanvas()
        v.addWidget(self.canvas, 1)

        prow = QtWidgets.QHBoxLayout()
        self.show_chk = QtWidgets.QCheckBox("Path")
        self.show_chk.setChecked(True)
        self.show_chk.toggled.connect(self._on_toggle_path)
        prow.addWidget(self.show_chk)

        self.show_stops_chk = QtWidgets.QCheckBox("Stops")
        self.show_stops_chk.setChecked(True)
        self.show_stops_chk.toggled.connect(self._on_toggle_stops)
        prow.addWidget(self.show_stops_chk)

        self.show_turns_chk = QtWidgets.QCheckBox("Turns")
        self.show_turns_chk.setChecked(True)
        self.show_turns_chk.toggled.connect(self._on_toggle_turns)
        prow.addWidget(self.show_turns_chk)

        self.show_contacts_chk = QtWidgets.QCheckBox("Contacts")
        self.show_contacts_chk.setChecked(False)
        self.show_contacts_chk.toggled.connect(self._on_toggle_contacts)
        prow.addWidget(self.show_contacts_chk)

        self.show_anchors_chk = QtWidgets.QCheckBox("Anchors")
        self.show_anchors_chk.setChecked(False)
        self.show_anchors_chk.toggled.connect(self._on_toggle_anchors)
        prow.addWidget(self.show_anchors_chk)

        prow.addStretch(1)
        self.pt_label = QtWidgets.QLabel("0 pts")
        prow.addWidget(self.pt_label)
        clr_btn = QtWidgets.QPushButton("Clear")
        clr_btn.clicked.connect(self._on_clear_path)
        prow.addWidget(clr_btn)
        v.addLayout(prow)

        leg = QtWidgets.QLabel(
            "speed: green=walk yellow=jog orange=run red=sprint    "
            "stops=red sq    turns: cyan=L magenta=R    "
            "contacts: blue=L orange=R    anchors=yellow tri"
        )
        leg.setStyleSheet("color: #aaa;")
        leg.setWordWrap(True)
        v.addWidget(leg)

        # Export
        exp_row = QtWidgets.QHBoxLayout()
        exp_btn = QtWidgets.QPushButton("Export Stops + Turns to Timeline Markers")
        exp_btn.clicked.connect(_on_export_markers)
        exp_row.addWidget(exp_btn)
        clr_marks_btn = QtWidgets.QPushButton("Clear Take Marks")
        clr_marks_btn.clicked.connect(_on_clear_take_marks)
        exp_row.addWidget(clr_marks_btn)
        v.addLayout(exp_row)

        # Speed coverage
        v.addWidget(self._section("--- Speed coverage ---"))
        self.speed_bars = {}
        self.speed_pcts = {}
        for (name, _, _, _) in SPEED_BUCKETS:
            r = QtWidgets.QHBoxLayout()
            nm = QtWidgets.QLabel(name)
            nm.setMinimumWidth(60)
            r.addWidget(nm)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            r.addWidget(bar, 1)
            pct = QtWidgets.QLabel("0%")
            pct.setMinimumWidth(40)
            r.addWidget(pct)
            v.addLayout(r)
            self.speed_bars[name] = bar
            self.speed_pcts[name] = pct

        # Turn counts
        v.addWidget(self._section("--- Turn counts ---"))
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.addWidget(QtWidgets.QLabel(""), 0, 0)
        for col, (lbl, _, _) in enumerate(TURN_MAG_BUCKETS, 1):
            h = QtWidgets.QLabel(lbl)
            h.setStyleSheet("color: #aaa;")
            grid.addWidget(h, 0, col)
        self.turn_labels = {"Left": {}, "Right": {}}
        for ridx, side in enumerate(("Left", "Right"), 1):
            grid.addWidget(QtWidgets.QLabel(side), ridx, 0)
            for col, (mag_lbl, _, _) in enumerate(TURN_MAG_BUCKETS, 1):
                cl = QtWidgets.QLabel("0")
                grid.addWidget(cl, ridx, col)
                self.turn_labels[side][mag_lbl] = cl
        v.addLayout(grid)

        # Stops + contacts
        v.addWidget(self._section("--- Events ---"))
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Stops:"))
        self.stop_count = QtWidgets.QLabel("0")
        srow.addWidget(self.stop_count)
        srow.addSpacing(20)
        srow.addWidget(QtWidgets.QLabel("Contacts:"))
        self.contact_count = QtWidgets.QLabel("L0  R0")
        srow.addWidget(self.contact_count)
        srow.addSpacing(20)
        srow.addWidget(QtWidgets.QLabel("Anchors:"))
        self.anchor_count = QtWidgets.QLabel("0")
        srow.addWidget(self.anchor_count)
        srow.addStretch(1)
        v.addLayout(srow)

        # Reset
        rst = QtWidgets.QPushButton("Reset All")
        rst.clicked.connect(_on_reset)
        v.addWidget(rst)

        # Advanced (collapsible)
        self.adv_btn = QtWidgets.QToolButton()
        self.adv_btn.setText(" Advanced thresholds")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setChecked(False)
        self.adv_btn.setArrowType(QtCore.Qt.RightArrow)
        self.adv_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.adv_btn.setStyleSheet(
            "QToolButton { border: none; color: #ccc; padding: 4px; }"
            "QToolButton:hover { color: #fff; }"
        )
        self.adv_btn.toggled.connect(self._on_adv_toggle)
        v.addWidget(self.adv_btn)

        self.adv_widget = QtWidgets.QWidget()
        self.adv_widget.setVisible(False)
        adv_v = QtWidgets.QVBoxLayout(self.adv_widget)
        adv_v.setContentsMargins(12, 0, 0, 0)
        adv_v.setSpacing(2)
        self._build_advanced(adv_v)
        v.addWidget(self.adv_widget)

        v.addStretch(1)

    # ---- Advanced thresholds builder ----
    def _build_advanced(self, parent):
        self._spinboxes = {}

        def add_section(title):
            s = QtWidgets.QLabel(title)
            s.setStyleSheet("color: #888; margin-top: 6px; font-weight: bold;")
            parent.addWidget(s)

        def add_spin(label, key, decimals=2, step=0.1, low=0.0, high=10000.0, is_int=False):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(180)
            row.addWidget(lbl)
            if is_int:
                sb = QtWidgets.QSpinBox()
                sb.setRange(int(low), int(high))
                sb.setSingleStep(int(step))
                sb.setValue(int(CFG[key]))
                sb.valueChanged.connect(lambda v, k=key: CFG.__setitem__(k, int(v)))
            else:
                sb = QtWidgets.QDoubleSpinBox()
                sb.setDecimals(decimals)
                sb.setRange(low, high)
                sb.setSingleStep(step)
                sb.setValue(float(CFG[key]))
                sb.valueChanged.connect(lambda v, k=key: CFG.__setitem__(k, float(v)))
            row.addWidget(sb)
            row.addStretch(1)
            parent.addLayout(row)
            self._spinboxes[key] = sb
            return sb

        def add_check(label, key):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(bool(CFG[key]))
            cb.toggled.connect(lambda v, k=key: CFG.__setitem__(k, bool(v)))
            parent.addWidget(cb)
            self._spinboxes[key] = cb
            return cb

        add_section("Yaw chain")
        add_spin("Median window (samples)",      "median_window",  is_int=True, step=1, low=1, high=51)
        add_spin("Mean window (s)",              "mean_secs",      decimals=2, step=0.05, low=0.05, high=2.0)
        add_check("  auto-estimate from gait",   "gait_auto")
        add_spin("Diff window T (s)",            "diff_T",         decimals=2, step=0.1,  low=0.1,  high=4.0)

        add_section("Turn detection")
        add_spin("Start ang_vel (deg/s)",        "turn_start_angvel",    decimals=1, step=1.0,  low=1.0,  high=360.0)
        add_spin("End ang_vel (deg/s)",          "turn_end_angvel",      decimals=1, step=1.0,  low=0.0,  high=360.0)
        add_spin("End hold (s)",                 "turn_end_hold",        decimals=2, step=0.05, low=0.0,  high=2.0)
        add_spin("Flip hold (s)",                "turn_flip_hold",       decimals=2, step=0.05, low=0.0,  high=2.0)
        add_spin("Min total angle (deg)",        "turn_min_angle",       decimals=1, step=1.0,  low=0.0,  high=720.0)
        add_spin("Soft angle (deg)",             "turn_soft_angle",      decimals=1, step=1.0,  low=0.0,  high=720.0)
        add_spin("Min peak ang_vel (deg/s)",     "turn_min_peak_angvel", decimals=1, step=5.0,  low=0.0,  high=720.0)

        add_section("Stop detection")
        add_spin("Moving threshold (m/s)",       "stop_moving_mps",      decimals=2, step=0.05, low=0.0,  high=20.0)
        add_spin("Still threshold (m/s)",        "stop_still_mps",       decimals=2, step=0.05, low=0.0,  high=20.0)
        add_spin("Hold (s)",                     "stop_hold_secs",       decimals=2, step=0.05, low=0.0,  high=10.0)

        add_section("Foot contact")
        add_spin("Height max (cm)",              "foot_height_max_cm",   decimals=2, step=0.5,  low=0.0,  high=100.0)
        add_spin("Speed max (cm/s)",             "foot_speed_max_cmps",  decimals=1, step=2.0,  low=0.0,  high=500.0)
        add_spin("Hold frames",                  "foot_hold_frames",     is_int=True, step=1, low=1, high=20)

        add_section("Foot turn classifier")
        add_spin("Foot vs traj threshold (deg)", "foot_turn_thresh_deg", decimals=1, step=1.0,  low=0.0,  high=180.0)
        add_spin("Anchor blend window (frames)", "foot_blend_frames",    is_int=True, step=1, low=1, high=30)

        add_section("Sampling")
        add_spin("Min sample dt (s)",            "sample_min_dt",        decimals=3, step=0.005, low=0.0, high=0.2)
        add_spin("Units per meter",              "units_per_meter",      decimals=1, step=10.0, low=1.0, high=10000.0)

    def _on_adv_toggle(self, checked):
        self.adv_widget.setVisible(checked)
        self.adv_btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def _on_toggle_path(self, c):     _STATE["show_path"][0]     = bool(c)
    def _on_toggle_stops(self, c):    _STATE["show_stops"][0]    = bool(c); self.canvas.update()
    def _on_toggle_turns(self, c):    _STATE["show_turns"][0]    = bool(c); self.canvas.update()
    def _on_toggle_contacts(self, c): _STATE["show_contacts"][0] = bool(c); self.canvas.update()
    def _on_toggle_anchors(self, c):  _STATE["show_anchors"][0]  = bool(c); self.canvas.update()

    def _on_clear_path(self):
        _STATE["path_points"].clear()
        _STATE["stop_events"]   = []
        _STATE["turn_events"]   = []
        _STATE["foot_contacts"] = []
        _STATE["anchor_events"] = []
        self.canvas.update()
        log("Path + events cleared.")

    def closeEvent(self, e):
        _remove_old_callback()
        _STATE["window"] = None
        super(LocoTrackerWindow, self).closeEvent(e)


# =============================================================================
# STATE
# =============================================================================
_STATE = globals().setdefault("_LOCO_STATE_V3", {
    "window":         None,
    "sampler":        None,
    "speed_sm":       None,
    "yaw_proc":       None,
    "speed_cov":      None,
    "turn_det":       None,
    "stop_det":       None,
    "foot_left":      None,
    "foot_right":     None,
    "gait_est":       None,
    "last_tick":      [None],
    "cb":             None,
    "fps":            30.0,

    "path_points":    deque(maxlen=PATH_MAX_POINTS),  # (x, z, speed, scene_t)
    "stop_events":    [],   # [{scene_t, pos_xz}]
    "turn_events":    [],   # [{scene_t, pos_xz, side, mag_label, total_deg, start_t, end_t}]
    "foot_contacts":  [],   # [{scene_t, pos_xz, side}]
    "anchor_events":  [],   # [{scene_t, pos_xz, side, delta_deg}]

    "show_path":      [True],
    "show_stops":     [True],
    "show_turns":     [True],
    "show_contacts":  [False],
    "show_anchors":   [False],
    "current_pos":    None,
})

# Re-execution safety: ensure all keys exist on cached state
for _k, _v in [
    ("foot_left", None), ("foot_right", None), ("gait_est", None),
    ("foot_contacts", []), ("anchor_events", []),
    ("show_contacts", [False]), ("show_anchors", [False]),
    ("yaw_proc", None), ("fps", 30.0), ("current_pos", None),
]:
    _STATE.setdefault(_k, _v)


def _remove_old_callback():
    if _STATE["cb"] is not None:
        try:
            FBSystem().OnUIIdle.Remove(_STATE["cb"])
            log("Removed previous OnUIIdle callback.")
        except Exception:
            pass
        _STATE["cb"] = None


def _close_old_window():
    w = _STATE["window"]
    if w is None:
        return
    try:
        w.close()
        w.deleteLater()
    except Exception:
        pass
    _STATE["window"] = None


def _get_mobu_main_window():
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    for w in app.topLevelWidgets():
        try:
            t = w.windowTitle() or ""
        except Exception:
            t = ""
        if "MotionBuilder" in t:
            return w
    return None


# =============================================================================
def _trajectory_yaw_lookback(now_t, lookback=0.2):
    pts = _STATE["path_points"]
    if len(pts) < 2:
        return None
    target = now_t - lookback
    last = pts[-1]
    for pt in reversed(pts):
        if pt[3] <= target:
            dx = last[0] - pt[0]
            dz = last[1] - pt[1]
            if abs(dx) > 0.01 or abs(dz) > 0.01:
                return math.atan2(dx, dz)
            return None
    return None


# =============================================================================
def _update_ui(speed, ang_vel, state, root_name):
    win = _STATE["window"]
    if win is None:
        return
    try:
        win.root_label.setText(root_name or "(not found - edit ROOT_CANDIDATES)")
        win.speed_label.setText("Speed:    " + ("{:.2f} m/s".format(speed) if speed is not None else "--"))
        win.angvel_label.setText("Turn:    " + ("{:+.1f} deg/s".format(ang_vel) if ang_vel is not None else "--"))
        win.state_label.setText("State:    " + (state or "--"))
        ge = _STATE["gait_est"]
        gait_str = "{:.2f} s".format(ge.estimate) if ge and ge.estimate else "--"
        win.gait_label.setText("Gait:     " + gait_str)

        for (name, _, _, _) in SPEED_BUCKETS:
            pct = _STATE["speed_cov"].pct(name)
            win.speed_bars[name].setValue(int(round(pct * 100.0)))
            win.speed_pcts[name].setText("{:.0f}%".format(pct * 100.0))

        counts = _STATE["turn_det"].counts
        for side in ("Left", "Right"):
            for (mag_lbl, _, _) in TURN_MAG_BUCKETS:
                win.turn_labels[side][mag_lbl].setText(str(counts[side][mag_lbl]))

        win.stop_count.setText(str(_STATE["stop_det"].count))
        nL = sum(1 for c in _STATE["foot_contacts"] if c["side"] == "Left")
        nR = sum(1 for c in _STATE["foot_contacts"] if c["side"] == "Right")
        win.contact_count.setText("L{}  R{}".format(nL, nR))
        win.anchor_count.setText(str(len(_STATE["anchor_events"])))
        win.pt_label.setText("{} pts".format(len(_STATE["path_points"])))

        # If gait_auto, reflect estimate in mean_secs spinbox (read-only-ish)
        if CFG["gait_auto"] and ge and ge.estimate:
            sb = win._spinboxes.get("mean_secs")
            if sb is not None:
                sb.blockSignals(True)
                sb.setValue(float(CFG["mean_secs"]))
                sb.blockSignals(False)
    except RuntimeError:
        _STATE["window"] = None


def _tick(control, event):
    try:
        now_real = time.time()
        last = _STATE["last_tick"][0]
        if last is not None and (now_real - last) < float(CFG["sample_min_dt"]):
            return
        _STATE["last_tick"][0] = now_real

        win = _STATE["window"]
        if win is None:
            return

        sampler = _STATE["sampler"]
        cur = sampler.get_current_pos_xz()
        prev_cur = _STATE["current_pos"]
        _STATE["current_pos"] = cur

        playing = FBPlayerControl().IsPlaying
        try:
            status = "PLAYING" if playing else "STOPPED (click Play to start)"
            win.play_label.setText("Status:   " + status)
        except RuntimeError:
            _STATE["window"] = None
            return

        if not playing:
            sampler.reset()
            _STATE["yaw_proc"].reset()
            root_name = sampler.joint.LongName if sampler.joint else None
            _update_ui(None, None, None, root_name)
            if cur != prev_cur:
                try:
                    win.canvas.update()
                except RuntimeError:
                    _STATE["window"] = None
            return

        res = sampler.sample()
        if res is None:
            _update_ui(None, None, None, None)
            return
        pos_xyz   = res["pos_xyz"]
        yaw       = res["yaw"]
        scene_t   = res["scene_t"]
        speed_mps = res["speed_mps"]
        dt        = res["dt"]
        if speed_mps is None or dt == 0.0 or yaw is None:
            _update_ui(None, None, None, res["name"])
            return

        # Path point (timestamped) - feed before contact lookback
        if _STATE["show_path"][0]:
            _STATE["path_points"].append((pos_xyz[0], pos_xyz[2], None, scene_t))

        # Speed
        speed = _STATE["speed_sm"].smooth(speed_mps)

        # Patch the just-appended path point with smoothed speed
        if _STATE["path_points"]:
            x, z, _spd, t_ = _STATE["path_points"][-1]
            _STATE["path_points"][-1] = (x, z, speed, t_)

        # Yaw chain
        filtered_yaw, ang_vel = _STATE["yaw_proc"].push(scene_t, yaw)

        # Foot contacts + anchor + gait estimate
        for det in (_STATE["foot_left"], _STATE["foot_right"]):
            if det is None:
                continue
            contact = det.update(scene_t)
            if contact is None:
                continue
            _STATE["foot_contacts"].append({
                "scene_t": contact["scene_t"],
                "pos_xz":  contact["pos_xz"],
                "side":    contact["side"],
            })
            _STATE["gait_est"].push(contact["side"], contact["scene_t"])
            est = _STATE["gait_est"].estimate
            if CFG["gait_auto"] and est is not None:
                CFG["mean_secs"] = max(0.20, min(1.20, est))

            # Foot turn classifier
            if contact["foot_yaw"] is None:
                continue
            traj_yaw = _trajectory_yaw_lookback(scene_t, lookback=0.20)
            if traj_yaw is None:
                continue
            signed = signed_yaw_delta(traj_yaw, contact["foot_yaw"])
            delta_deg = math.degrees(abs(signed))
            if delta_deg > float(CFG["foot_turn_thresh_deg"]):
                blend_secs = float(CFG["foot_blend_frames"]) / max(1.0, _STATE["fps"])
                _STATE["yaw_proc"].apply_anchor(scene_t, contact["foot_yaw"], blend_secs)
                _STATE["anchor_events"].append({
                    "scene_t":   scene_t,
                    "pos_xz":    contact["pos_xz"],
                    "side":      contact["side"],
                    "delta_deg": delta_deg,
                })

        # Coverage / turn / stop
        state = _STATE["speed_cov"].update(speed, dt)
        turn_ev = _STATE["turn_det"].update(ang_vel, dt, scene_t)
        stop_ev = _STATE["stop_det"].update(speed, dt)

        if turn_ev is not None:
            _STATE["turn_events"].append({
                "scene_t":   scene_t,
                "pos_xz":    (pos_xyz[0], pos_xyz[2]),
                "side":      turn_ev["side"],
                "mag_label": turn_ev["mag_label"],
                "total_deg": turn_ev["total_deg"],
                "start_t":   turn_ev["start_t"],
                "end_t":     turn_ev["end_t"],
            })
        if stop_ev:
            _STATE["stop_events"].append({
                "scene_t": scene_t,
                "pos_xz":  (pos_xyz[0], pos_xyz[2]),
            })

        try:
            win.canvas.update()
        except RuntimeError:
            _STATE["window"] = None
            return

        _update_ui(speed, ang_vel, state, res["name"])
    except Exception:
        log("tick error:")
        traceback.print_exc()


# =============================================================================
def _on_export_markers():
    take = FBSystem().CurrentTake
    if take is None:
        log("Export aborted: no current take.")
        return
    n_stops = 0
    for i, ev in enumerate(_STATE["stop_events"], 1):
        try:
            ftime = FBTime(0)
            ftime.SetSecondDouble(float(ev["scene_t"]))
            take.AddTimeMark(ftime)
            idx = take.GetTimeMarkCount() - 1
            take.SetTimeMarkName(idx, "Stop {}".format(i))
            n_stops += 1
        except Exception:
            traceback.print_exc()
    n_turns = 0
    lag = PathCanvas._lag_secs()
    for i, ev in enumerate(_STATE["turn_events"], 1):
        try:
            ftime = FBTime(0)
            corrected_t = ev["start_t"] - lag
            ftime.SetSecondDouble(float(corrected_t))
            take.AddTimeMark(ftime)
            idx = take.GetTimeMarkCount() - 1
            take.SetTimeMarkName(idx, "Turn {} {} {:.0f}d".format(
                ev["side"], ev["mag_label"], ev["total_deg"]))
            n_turns += 1
        except Exception:
            traceback.print_exc()
    log("Exported {} stops + {} turns (lag-corrected) to take time marks.".format(n_stops, n_turns))


def _on_clear_take_marks():
    take = FBSystem().CurrentTake
    if take is None:
        return
    try:
        n = take.GetTimeMarkCount()
        for _ in range(n):
            take.RemoveTimeMark(0)
        log("Removed {} take time marks.".format(n))
    except Exception:
        traceback.print_exc()


def _on_reset():
    for k in ("sampler", "speed_sm", "yaw_proc", "speed_cov",
              "turn_det", "stop_det", "foot_left", "foot_right", "gait_est"):
        obj = _STATE[k]
        if obj is not None:
            try:
                obj.reset()
            except Exception:
                pass
    _STATE["last_tick"][0]  = None
    _STATE["path_points"].clear()
    _STATE["stop_events"]   = []
    _STATE["turn_events"]   = []
    _STATE["foot_contacts"] = []
    _STATE["anchor_events"] = []
    win = _STATE["window"]
    if win is not None:
        try:
            win.canvas.update()
        except RuntimeError:
            _STATE["window"] = None
    log("All counters and events reset.")


# =============================================================================
def create_tool():
    log("=" * 50)
    log("Loco Tracker V3 starting (PySide{}).".format(_PYSIDE))
    log("=" * 50)

    _remove_old_callback()
    _close_old_window()

    try:
        try:
            fps = FBPlayerControl().GetTransportFpsValue()
        except Exception:
            fps = 30.0
        _STATE["fps"] = float(fps)
        log("Scene fps: {:.2f}".format(fps))

        _STATE["sampler"]    = RootSampler()
        _STATE["speed_sm"]   = RollingSmoother(5)
        _STATE["yaw_proc"]   = YawProcessor()
        _STATE["speed_cov"]  = SpeedCoverage(SPEED_BUCKETS)
        _STATE["turn_det"]   = TurnEventDetector()
        _STATE["stop_det"]   = StopDetector()
        _STATE["foot_left"]  = FootContactDetector(LEFT_FOOT_CANDIDATES,  "Left")
        _STATE["foot_right"] = FootContactDetector(RIGHT_FOOT_CANDIDATES, "Right")
        _STATE["gait_est"]   = GaitCycleEstimator()
        _STATE["last_tick"][0] = None

        win = LocoTrackerWindow(_get_mobu_main_window())
        _STATE["window"] = win
        win.show()
        win.raise_()

        FBSystem().OnUIIdle.Add(_tick)
        _STATE["cb"] = _tick

        log("Window shown. Title: '{}'.".format(TOOL_NAME))
        return win
    except Exception:
        log("FAILED:")
        traceback.print_exc()
        return None


create_tool()
