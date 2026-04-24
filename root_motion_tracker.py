"""
Root Motion Tracker - MotionBuilder Plugin
-------------------------------------------
Live tool that extracts root motion from an actor, computes speed + angular
velocity, and tracks locomotion coverage across speed buckets.

Inspired by the structure of the offline LSTM rootbone_extractor:
  - Scaler           -> RollingSmoother (normalizes noisy live data)
  - in_feats/out_feats -> Feature dict {speed, ang_vel}
  - forward() loop   -> OnUIIdle callback
  - synthesize()     -> live tick()

Drop into:  <MotionBuilder>/bin/config/PythonStartup/
Or run via: File > Python Editor > open this file > Execute
"""

from pyfbsdk import *
from pyfbsdk_additions import *
from collections import deque
import math
import time


# =============================================================================
# CONFIG  (mirrors configs/ folder in the LSTM project)
# =============================================================================
CONFIG = {
    "root_joint_name": "Hips",          # change to "Reference" / "Root" if needed
    "smooth_window": 5,                  # frames of rolling avg (like Scaler)
    "target_coverage_seconds": 10.0,     # per-bucket goal
    "sample_min_dt": 0.016,              # ~60fps max sample rate

    # Speed buckets in cm/s (MotionBuilder default units)
    "speed_buckets": [
        {"label": "Idle",   "min": 0.0,   "max": 5.0,   "color": FBColor(0.4, 0.4, 0.4)},
        {"label": "Walk",   "min": 5.0,   "max": 150.0, "color": FBColor(0.3, 0.7, 0.3)},
        {"label": "Jog",    "min": 150.0, "max": 300.0, "color": FBColor(0.9, 0.8, 0.2)},
        {"label": "Run",    "min": 300.0, "max": 500.0, "color": FBColor(0.9, 0.5, 0.2)},
        {"label": "Sprint", "min": 500.0, "max": 1e9,   "color": FBColor(0.9, 0.2, 0.2)},
    ],
}


# =============================================================================
# SMOOTHER  (mirrors the Scaler in LitLSTM — stabilizes live signal)
# =============================================================================
class RollingSmoother:
    def __init__(self, window=5):
        self.buf = deque(maxlen=window)

    def smooth(self, value):
        self.buf.append(value)
        return sum(self.buf) / len(self.buf)

    def reset(self):
        self.buf.clear()


# =============================================================================
# SAMPLER  (the actual root motion math — finite differences on transforms)
# =============================================================================
class RootSampler:
    """
    Extracts root motion live from the scene.

    This is the piece the offline LSTM doesn't have — it learns a mapping
    from features to root motion. Here we compute root motion directly from
    MotionBuilder transforms using finite differences:

        speed   = ||delta_xz|| / dt
        ang_vel = delta_yaw / dt
    """

    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.prev_pos = None
        self.prev_yaw = None
        self.prev_time = None

    def _find_root(self):
        return FBFindModelByLabelName(self.joint_name)

    def sample(self):
        root = self._find_root()
        if root is None:
            return None

        # Global transforms - matches what the dataset pre-proc would read from BVH
        pos = FBVector3d()
        rot = FBVector3d()
        root.GetVector(pos, FBModelTransformationType.kModelTranslation, True)
        root.GetVector(rot, FBModelTransformationType.kModelRotation, True)

        now = time.time()
        out = {
            "pos": (pos[0], pos[1], pos[2]),
            "yaw": rot[1],
            "time": now,
            "speed": None,
            "ang_vel": None,
        }

        if self.prev_pos is not None:
            dt = now - self.prev_time
            if dt > 0:
                # Forward/locomotion speed on XZ plane (ignore vertical motion)
                dx = pos[0] - self.prev_pos[0]
                dz = pos[2] - self.prev_pos[2]
                out["speed"] = math.sqrt(dx * dx + dz * dz) / dt

                # Angular velocity around Y (yaw rate)
                dyaw = rot[1] - self.prev_yaw
                # wrap to [-180, 180] to avoid jumps at 360
                dyaw = (dyaw + 180.0) % 360.0 - 180.0
                out["ang_vel"] = dyaw / dt

        self.prev_pos = out["pos"]
        self.prev_yaw = out["yaw"]
        self.prev_time = now
        return out

    def reset(self):
        self.prev_pos = None
        self.prev_yaw = None
        self.prev_time = None


# =============================================================================
# COVERAGE TRACKER  (mirrors the label/target concept from training config)
# =============================================================================
class CoverageTracker:
    def __init__(self, buckets, target_seconds):
        self.buckets = buckets
        self.target = target_seconds
        self.time_in_bucket = {b["label"]: 0.0 for b in buckets}

    def classify(self, speed):
        for b in self.buckets:
            if b["min"] <= speed < b["max"]:
                return b["label"]
        return None

    def update(self, speed, dt):
        label = self.classify(speed)
        if label is not None:
            self.time_in_bucket[label] += dt
        return label

    def pct(self, label):
        return min(self.time_in_bucket[label] / self.target, 1.0)

    def reset(self):
        for k in self.time_in_bucket:
            self.time_in_bucket[k] = 0.0


# =============================================================================
# UI PANEL  (MotionBuilder FBTool with live fill bars)
# =============================================================================
class TrackerUI:
    def __init__(self, tool, config):
        self.tool = tool
        self.config = config
        self.bars = {}        # label -> FBEditNumber or progress widget
        self.labels = {}      # label -> FBLabel for coverage text
        self.speed_label = None
        self.state_label = None
        self.angvel_label = None
        self.joint_input = None
        self._build()

    def _build(self):
        main = FBVBoxLayout()
        self.tool.AddRegion("main", "main", 5, FBAttachType.kFBAttachLeft, "", 1.0,
                                           5, FBAttachType.kFBAttachTop, "", 1.0,
                                          -5, FBAttachType.kFBAttachRight, "", 1.0,
                                          -5, FBAttachType.kFBAttachBottom, "", 1.0)
        self.tool.SetControl("main", main)

        # Root joint selector
        row_joint = FBHBoxLayout()
        lbl = FBLabel(); lbl.Caption = "Root joint:"
        row_joint.Add(lbl, 80)
        self.joint_input = FBEdit()
        self.joint_input.Text = self.config["root_joint_name"]
        row_joint.Add(self.joint_input, 150)
        main.Add(row_joint, 25)

        # Live readouts
        self.speed_label = FBLabel(); self.speed_label.Caption = "Speed:   -- cm/s"
        main.Add(self.speed_label, 22)

        self.state_label = FBLabel(); self.state_label.Caption = "State:   --"
        main.Add(self.state_label, 22)

        self.angvel_label = FBLabel(); self.angvel_label.Caption = "Ang Vel: -- deg/s"
        main.Add(self.angvel_label, 22)

        # Separator
        sep = FBLabel(); sep.Caption = "── Coverage ──────────────"
        main.Add(sep, 20)

        # Per-bucket rows
        for b in self.config["speed_buckets"]:
            row = FBHBoxLayout()

            name = FBLabel(); name.Caption = b["label"]
            row.Add(name, 60)

            bar = FBEditNumber()                # used as numeric readout 0-100
            bar.Value = 0.0
            bar.Enabled = False
            row.Add(bar, 70)

            pct_lbl = FBLabel(); pct_lbl.Caption = "0%"
            row.Add(pct_lbl, 50)

            main.Add(row, 24)
            self.bars[b["label"]] = bar
            self.labels[b["label"]] = pct_lbl

        # Reset button
        btn = FBButton(); btn.Caption = "Reset Coverage"
        btn.OnClick.Add(self._on_reset)
        main.Add(btn, 28)

        self._reset_cb = None

    def set_reset_callback(self, cb):
        self._reset_cb = cb

    def _on_reset(self, control, event):
        if self._reset_cb:
            self._reset_cb()

    def get_joint_name(self):
        return self.joint_input.Text

    def update(self, speed, ang_vel, active_label, coverage_map):
        if speed is not None:
            self.speed_label.Caption = "Speed:   {:.1f} cm/s".format(speed)
        if ang_vel is not None:
            self.angvel_label.Caption = "Ang Vel: {:.1f} deg/s".format(ang_vel)
        self.state_label.Caption = "State:   {}".format(active_label or "--")

        for label, pct in coverage_map.items():
            self.bars[label].Value = pct * 100.0
            self.labels[label].Caption = "{:.0f}%".format(pct * 100.0)


# =============================================================================
# LIVE LOOP  (mirrors the training loop in LitLSTM — sample, process, output)
# =============================================================================
class LiveLoop:
    def __init__(self, config):
        self.config = config
        self.sampler = RootSampler(config["root_joint_name"])
        self.speed_smoother = RollingSmoother(config["smooth_window"])
        self.angvel_smoother = RollingSmoother(config["smooth_window"])
        self.tracker = CoverageTracker(config["speed_buckets"], config["target_coverage_seconds"])
        self.ui = None
        self.last_tick_time = None
        self.min_dt = config["sample_min_dt"]

    def bind_ui(self, ui):
        self.ui = ui
        ui.set_reset_callback(self.reset)

    def reset(self):
        self.sampler.reset()
        self.speed_smoother.reset()
        self.angvel_smoother.reset()
        self.tracker.reset()
        self.last_tick_time = None

    def tick(self, control, event):
        # Sync joint name from UI (allows live re-targeting)
        if self.ui:
            current = self.ui.get_joint_name()
            if current != self.sampler.joint_name:
                self.sampler.joint_name = current
                self.reset()

        now = time.time()
        if self.last_tick_time and (now - self.last_tick_time) < self.min_dt:
            return

        result = self.sampler.sample()
        self.last_tick_time = now

        if result is None or result["speed"] is None:
            if self.ui:
                self.ui.update(None, None, None, {b["label"]: self.tracker.pct(b["label"])
                                                   for b in self.config["speed_buckets"]})
            return

        speed = self.speed_smoother.smooth(result["speed"])
        ang_vel = self.angvel_smoother.smooth(result["ang_vel"])

        dt = now - (self.last_tick_time - self.min_dt) if self.last_tick_time else 0.0
        # Use actual dt from sampler for accuracy
        dt = self.min_dt if dt <= 0 else dt

        active = self.tracker.update(speed, dt)

        if self.ui:
            coverage = {b["label"]: self.tracker.pct(b["label"])
                        for b in self.config["speed_buckets"]}
            self.ui.update(speed, ang_vel, active, coverage)


# =============================================================================
# TOOL REGISTRATION
# =============================================================================
TOOL_NAME = "Root Motion Tracker"
_tool_instance = None
_loop_instance = None


def _on_tool_show(control, event):
    pass


def create_tool():
    global _tool_instance, _loop_instance

    tool = FBCreateUniqueTool(TOOL_NAME)
    tool.StartSizeX = 340
    tool.StartSizeY = 380

    ui = TrackerUI(tool, CONFIG)
    loop = LiveLoop(CONFIG)
    loop.bind_ui(ui)

    FBSystem().OnUIIdle.Add(loop.tick)

    _tool_instance = tool
    _loop_instance = loop

    ShowTool(tool)
    return tool


if __name__ in ("__main__", "__builtin__"):
    create_tool()
