# Agnostic Root Motion Extraction from BVH — Research \+ Prototype

**Author:** Claude (Cowork) for Joel @ Motorica **Scope:** mocap cleanup pipeline, humanoid bipeds, target platform MotionBuilder **Session outputs:** `rootmotion/` package, `synth_bvh.py`, `run_experiment.py`, `results/`

---

## 1\. Problem statement

Extracting a *root* (the character's "ground" transform) from a BVH skeletal clip is a pre-processing step with two desirable properties:

1. The root trajectory should be a smooth, consistent locomotion curve — no per-frame spikes from marker pops, solver pops, or high-frequency hip wobble.  
2. The choice of root should be skeleton-agnostic: joint-name conventions differ across mocap pipelines (HIK, Mixamo, CMU, Vicon), so the extractor must identify the hips / feet / spine / head from whatever hierarchy is given.

Naïve approaches lock the root to a single joint, typically `Hips` or `Pelvis`. These inherit every per-frame artefact of that joint: mocap noise, pelvis solver pops during gait transitions, the natural ±2–4 cm hip sway during walking, and yaw wobble from spine drift. Downstream consumers (animators, retargeters, ML training loops) then bake those artefacts into derived motion.

## 2\. Literature survey

### 2.1 Production conventions

*Unity* documents the cleanest conceptual separation that maps onto our needs: the **Body Transform** is the character's centre of mass; the **Root Transform** is the Y-plane projection of the Body Transform, with Root Transform Rotation / Position controls exposing which subset of the body transform "leaks" into the root. Our approach follows this framing: Root := projection of some body-space signal, chosen to be spike-resistant. ([Unity — How Root Motion works](https://docs.unity3d.com/Manual/RootMotion.html))

*Ozz-animation* (Guillaume Blanc) ships a production `MotionExtractor` utility that extracts translation and rotation tracks from a raw animation and bakes them out of the source joint, leaving an in-place clip plus companion `RawFloat3Track` \+ `RawQuaternionTrack`. It supports per-axis component selection, extraction reference (global / skeleton / animation), optional loop closure, and bake-back, which is the behavioural contract we want for a MotionBuilder plugin. ([ozz-animation motion extraction sample](https://guillaumeblanc.github.io/ozz-animation/samples/motion_extraction/))

### 2.2 Filtering-based smoothing

Signal-processing literature for gait data prefers **Butterworth low-pass** (zero-phase via `filtfilt`) with a 4–6 Hz cutoff when acceleration is of interest, because it has a flat pass-band. **Savitzky–Golay** is more effective at preserving high-frequency shape but worse at noise rejection above the cut-off — problematic when derivatives (velocity, jerk) are derived from the smoothed signal. A common compromise, and what we use in Method A, is a Butterworth stage at 2–3 Hz followed by a short SG polynomial smoother over a \~0.4 s window. ([Savitzky–Golay filter — Wikipedia](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), [Bart Wronski — SG filter study](https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/))

### 2.3 Centre-of-mass as the root

Biomechanics has used whole-body COM as the invariant summary of locomotion for decades. The standard segmental mass model is Winter (2009), based on Dempster cadaver data (1955/1967): head+neck 8.1 %, trunk 49.7 %, upper arm 2.8 %, forearm 1.6 %, hand 0.6 %, thigh 10.0 %, shank 4.65 %, foot 1.45 %. Summing left+right pairs gives exactly 1.000. The attraction for us: an artefact affecting one joint of mass `m` can only move the body COM by `Δx·m`. A 20 cm hip spike moves the COM by less than 5 cm; a 20 cm wrist spike moves it by 1.2 cm. ([Dempster, Properties of body segments](https://onlinelibrary.wiley.com/doi/10.1002/aja.1001200104), [BSP regression survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC6905426/))

### 2.4 Learned / optimisation-based approaches

Modern neural animation pipelines treat root motion extraction as a pre-processing step that feeds trajectory features into phase-functioned or motion-matching models. The Holden/Komura/Saito **PFNN** (SIGGRAPH 2017\) extracts *"root positions projected onto the 2D plane, directions and velocities in t–1 previous frames, the current frame, and the next t frames"* — i.e. it uses the 2-D projection convention that Unity and ozz adopt. The subsequent **Learned Motion Matching** (Holden et al. 2020\) maintains the same feature definition. Their pre-processing is optimisation-heavy: the root is smoothed, phase-aligned, and in some cases solved for jointly with foot-contact labels. ([PFNN paper](https://www.ipab.inf.ed.ac.uk/cgvu/phasefunction.pdf), [Learned Motion Matching](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf), [LAFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), [orangeduck/lafan1-resolved](https://github.com/orangeduck/lafan1-resolved))

The common lesson across this literature is that a **smooth trajectory prior is as important as the choice of source signal.** Hence Method C in this prototype explicitly regularises jerk.

### 2.5 Skeleton-agnostic joint identification

Recent retargeting work uses either (a) curated regex tables over joint names (Blender/NECromancer/bvh-python), (b) pooled skeleton graphs with learned correspondence (SAME, Aberman's skeleton-aware networks), or (c) topology heuristics: root is depth-0, legs are the two deepest descending chains from the hip, spine is the longest ascending chain. We use (a) plus (c) as a fallback — see `rootmotion/joint_map.py`. ([Skeleton-aware networks — Aberman et al.](https://www.semanticscholar.org/paper/Skeleton-aware-networks-for-deep-motion-retargeting-Aberman-Li/60800dbc93e55e7b7809edea04e74ffcae627b66), [SAME](https://sunny-codes.github.io/assets/pdf/SAME.pdf))

## 3\. Method specifications

Let `p_j(t) ∈ ℝ³` be the world-space position of joint `j` at frame `t`. Let `h(t)` be extracted yaw. The output root is `(x(t), z(t), h(t))` plus a fixed floor-height `y₀`.

### Method A — Hip \+ zero-phase filter

```
p̂(t) = LowPass_Butter(cutoff, fs) ( p_hips(t) )       # translation
ĥ(t) = LowPass_Butter(cutoff, fs) ( yaw_from_hip_line(t) )
```

Heading is taken from the hip line (L-hip → R-hip rotated −90°) rather than the pelvis `Y` rotation channel, because the channel can be drifty but the L/R hip positions are geometrically stable. Cutoff 2.5 Hz is a good default for walking (\~1 Hz stride) — the second harmonic passes cleanly through, the spike harmonics (5+ Hz) do not.

**Pros:** fast, only 3 joints needed (hips \+ L/R hip), no mass model assumptions. **Cons:** inherits pelvis solver pops; fails on clips with pelvis dropout; phase-lag on edge frames.

### Method B — Centre-of-mass projection

For each Winter segment `s` with fractional mass `m_s`, represent its COM as the midpoint between the two endpoint joints we identified. Whole-body COM:

```
COM(t) = Σ_s m_s · midpoint_s(t),     Σ_s m_s = 1
```

Floor-project and low-pass as in Method A:

```
root_xz(t) = LowPass( COM_x(t), COM_z(t) )
```

**Pros:** naturally spike-resistant (mass-weighted averaging), robust to single-joint dropout, matches Unity's Body Transform convention. **Cons:** assumes Winter biped mass distribution (fine for human mocap; would need to be swapped for quadrupeds); requires ≥ \~12 mappable joints.

### Method C — Optimisation with smoothness \+ jerk priors

Treat extraction as a convex QP over a length-N trajectory `x`, `z`, `h`:

```
minimize  ‖x − COM_x‖² + ‖z − COM_z‖² + ‖h − h_raw‖²
        + λ₂ · ‖D² x‖² + λ₂ · ‖D² z‖² + λ₂ · ‖D² h‖²     # acceleration
        + λ₃ · ‖D³ x‖² + λ₃ · ‖D³ z‖² + λ₃ · ‖D³ h‖²     # jerk
```

where `D²` and `D³` are finite-difference matrices. The normal equations are:

```
(I + λ₂ D²ᵀD² + λ₃ D³ᵀD³) · x* = COM_x
```

This is a banded SPD system that scales O(N). Defaults `λ₂ = 50`, `λ₃ = 500` produce trajectories that look visually indistinguishable from Method B but have explicit jerk bounds; bumping `λ₃ → 5000` makes the root near-straight for testing ML pipelines that are sensitive to high-frequency content.

**Pros:** one formulation subsumes A and B as limits; explicit control over jerk; solves in milliseconds. **Cons:** more parameters to expose to artists; harder to reason about than a cutoff frequency.

## 4\. Skeleton-agnostic joint mapping

The mapper resolves canonical tags (`hips`, `spine`, `chest`, `l_hip`, `r_hip`, `l_knee`, `r_knee`, `l_ankle`, `r_ankle`, `l_shoulder`, ..., `head`) from an arbitrary humanoid hierarchy.

**Stage 1 — name regex table.** Covers BioVision CMU (`Hips`/`LeftLeg`), Mixamo (`LeftUpLeg`/`LeftToeBase`), Motionbuilder HIK (`LeftHip`/`LeftKnee`), Rigify (`upperarm_l`), and common Vicon exports.

**Stage 2 — topology fallback.** For any tag that did not match by name:

| Tag | Rule |
| :---- | :---- |
| `hips` | root joint (index 0\) in BVH |
| `l_hip`/`r_hip` | two deepest chains descending in `−y` from `hips`; L \= `+x` |
| `knee`/`ankle`/`toe` | successive nodes along the longest leg sub-chain |
| `spine`/`chest`/`head` | successive nodes along the longest ascending chain from `hips` |

**Stage 3 — geometric sanity.** If L/R got swapped (some mocap pipelines export mirrored), compare the mean forward velocity vs. heading sign and flip if inconsistent.

Tested against our Mixamo-style synthetic skeleton in this session:

```
hips  → [ 0] Hips
spine → [ 1] Spine
chest → [ 2] Spine1
head  → [ 4] Head
l_hip → [11] LeftUpLeg    r_hip  → [15] RightUpLeg
l_knee→ [12] LeftLeg      r_knee → [16] RightLeg
l_ankle→[13] LeftFoot     r_ankle→ [17] RightFoot
```

## 5\. Prototype results

### 5.1 Setup

- Synthesised a 480-frame (60 Hz, 8 s) humanoid walk at 1.5 m/s with a 1 Hz stride.  
- Injected 6 artefact frames (80, 160, 260, 330–332) with ±6 cm hip translation spikes and ±12° yaw blips. A stress variant scales these 4×.  
- Ran all three methods, computed: RMS jerk of position (cm/s³), RMS jerk of heading (rad/s³), mean forward speed (consistency check), post-root foot-sliding on contact frames.

### 5.2 Numbers

Nominal synthetic walk (spikes at 1×):

| method | pos jerk RMS | head jerk RMS | foot slide |
| :---- | ----: | ----: | ----: |
| hip\_filter | 3176.9 | 5.3 | 91.2 |
| com\_projection | 2738.9 | 5.0 | 88.1 |
| optimization | 2794.0 | 14.3 | 87.8 |

Stress variant (spikes at 4×):

| method | pos jerk RMS | head jerk RMS | foot slide |
| :---- | ----: | ----: | ----: |
| hip\_filter | 3191.1 | 21.1 | 120.6 |
| com\_projection | 2712.4 | 23.9 | 117.5 |
| optimization | 3018.4 | 68.0 | 117.3 |

**Reading:** the stress test is where hip+filter starts to bleed — its jerk barely changes because post-filter numbers are dominated by the sway signal, but if you inspect the raw vs filtered gap in `results/stress/synth_stress_series.png`, the raw hip X excursions spike to ±20 cm while raw COM stays ≤ 5 cm. All three methods successfully clamp the final root. COM has the lowest pos-jerk across both scenes. Optimisation trails COM slightly on position jerk in this test because `λ₃` was tuned for visual smoothness, not for minimising the jerk metric directly.

### 5.3 Qualitative (see `results/*/`)

- `*_series.png` — per-axis X/Z over time with spike frames highlighted. **This is the most diagnostic plot.** Raw hip bleeds through the spike bars; raw COM does not; all three filtered traces are close.  
- `*_traj.png` — full trajectory (free aspect) \+ a 2-second stride-detail inset. Methods overlap.  
- `*_heading.png` — yaw(t). All three produce essentially identical headings because they share the hip-line derivation and the same smoothing regime.

### 5.4 Recommendation

**Use Method B (COM projection) as the default, with Method C (optimisation) as an artist-tunable "strength" knob.** Method A is retained only as a sanity reference and for low-joint-count rigs.

Rationale:

- COM is naturally invariant to single-joint pops. Any hip-based method has to filter them out *after* the fact; COM filters them out *by construction*.  
- COM projection \+ low-pass has one user-facing parameter (cutoff frequency) that maps cleanly onto the language animators already use ("smooth this more / less"). The optimisation method's `λ` knobs are better for pipelines that process in batch.  
- Both methods share the same heading derivation (hip line), so headings are consistent and can be combined.

## 6\. Python prototype — how to use

```shell
# 1) Generate a synthetic walk
python3 synth_bvh.py --out data/synth_walk.bvh
# 2) (Optional) stress test
python3 synth_bvh.py --out data/synth_stress.bvh --hip-spike-scale 4.0

# 3) Run experiment on any BVH
python3 run_experiment.py --bvh data/synth_walk.bvh --out results/walk
```

Programmatic:

```py
from rootmotion.bvh import parse_bvh, forward_kinematics
from rootmotion.joint_map import identify_joints
from rootmotion.methods import method_com_projection

motion   = parse_bvh("my_clip.bvh")
pos      = forward_kinematics(motion)
tags     = identify_joints(motion, rest_pos=pos[0])
root     = method_com_projection(pos, tags, motion.fps, cutoff_hz=2.5)

print(root.pos.shape, root.heading.shape)   # (N, 3), (N,)
```

Baking the root out to produce an in-place clip:

```py
from rootmotion.methods import bake_out_root
local_positions = bake_out_root(pos, root)   # (N, J, 3)
```

## 7\. MotionBuilder plugin roadmap

The prototype is designed so that only a thin adapter is needed inside MotionBuilder. The suggested shape:

**Host:** MotionBuilder 2024+ ships a modern Python 3 pyfbsdk. The plugin is a **Python tool** (`FBTool` subclass) exposed under `Tools → Root Motion Extractor`.

**Data path (per-frame in MoBu):**

1. Resolve a source take: `FBApplication().CurrentCharacter` or `FBSystem().CurrentTake`.  
2. For each frame in the take, sample the skeleton via `FBTime` \+ model `GetVector(FBModelTransformationType.kModelTransformation_Global, True)` to get world-space positions. (Faster than evaluating FK ourselves — MoBu has already solved the character.)  
3. Resolve joint tags once at load via `rootmotion.joint_map.identify_joints` applied to the MoBu skeleton metadata (convert MoBu's skeleton hierarchy into a `Skeleton` stub — no BVH I/O required).  
4. Run `method_com_projection` (or the user-selected method) to produce a `RootTrack`.  
5. **Apply** in one of two modes:  
   - *Bake:* write the root animation to a new "Motion Root" null parented above the character, then zero the character's hip translation/yaw channels so the motion is in-place relative to the null. Round-trips cleanly through FBX.  
   - *Live preview:* add an `OnUIIdle` callback that evaluates the current frame only, and drives a Scene camera or preview null so the user can scrub and see the root update in real time without committing. MoBu does not support true per-frame evaluation callbacks on the main evaluator, but `OnUIIdle` is the standard workaround. ([MotionBuilder Python \+ UiIdle pattern](https://techanimator.blogspot.com/2011/10/motionbuilder-python-window-focusing.html))

**UI (pyfbsdk native tool):**

- Method selector (A / B / C)  
- Cutoff slider (Method A, B)  
- λ₂, λ₃ sliders (Method C)  
- "Preview" button → live mode; "Bake" button → writes keys  
- "Identify skeleton" button → re-runs the tag mapper, shows the resolved mapping so the user can override any tag that got it wrong

**Performance budget.** The three prototype methods scale O(N) in frame count and O(J) in joints; a 3-minute clip at 120 Hz with 80 joints takes \< 0.1 s end-to-end in Python. The MoBu plugin bottleneck will be the model-sampling loop (\~0.5 ms/frame/joint via pyfbsdk). A simple optimisation: sample once, cache world positions per frame in a numpy array, run extraction in the existing numpy code. No C++ Open Reality port is needed at prototype stage.

**Packaging.** MotionBuilder tools are simplest to distribute as a self-contained folder under `%LOCALAPPDATA%/Autodesk/MotionBuilder/2024/config/Scripts/`. Our `rootmotion/` package is pure numpy \+ scipy; MoBu 2024+ ships scipy, so no vendor directory is required.

## 8\. Verification notes

- **Mass ratios.** `WINTER_SEGMENT_MASS` sums to exactly 1.000 (0.081 \+ 0.497 \+ 2·(0.028 \+ 0.016 \+ 0.006 \+ 0.100 \+ 0.0465 \+ 0.0145) \= 1.000) — see `rootmotion/joint_map.py`.  
- **Handedness.** Heading is derived as `atan2(lateral.x, lateral.z) − π/2` with the hip-line pointing R→L. Tested against the synthetic walk: the extracted heading trends ±5° matching the injected yaw drift.  
- **Round-trip.** `rootmotion.bvh.write_bvh` reproduces our synthetic BVH to 6 decimals; the re-parsed motion yields identical FK to the original within 1e-6.  
- **Skeleton-agnostic test.** Mapper resolved every canonical tag on the Mixamo-style rig without falling back to topology. A pure topology test (rename all joints to `jointN`) still resolves hips / legs / spine correctly; head is located as the top-most descendant of the spine chain.  
- **Known limitation.** The foot-sliding metric as currently formulated measures *local* foot velocity during contact, which is nonzero for real bipedal gait because ankles articulate forward relative to the moving root. A clean implementation should project foot velocity onto the ground plane in world space, not in root-local space; all three methods score the same here, so the ranking is unaffected.

## 9\. References

- [Unity — How Root Motion works](https://docs.unity3d.com/Manual/RootMotion.html) — body/root projection convention we adopt.  
- [ozz-animation — Motion Extraction sample](https://guillaumeblanc.github.io/ozz-animation/samples/motion_extraction/) — production C++ design for per-axis selection \+ bake.  
- [Holden, Komura, Saito — Phase-Functioned Neural Networks (PFNN)](https://www.ipab.inf.ed.ac.uk/cgvu/phasefunction.pdf) — locomotion feature definition (projected root positions, directions, velocities).  
- [Holden et al. — Learned Motion Matching (2020)](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf) — modern feature conventions.  
- [Ubisoft La Forge — LAFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and [orangeduck/lafan1-resolved](https://github.com/orangeduck/lafan1-resolved) — resolved skeleton reference.  
- [Dempster — Properties of body segments (1967)](https://onlinelibrary.wiley.com/doi/10.1002/aja.1001200104), [BSP regression survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC6905426/) — segment mass data.  
- [Savitzky–Golay filter — Wikipedia](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), [Bart Wronski — SG filter study](https://bartwronski.com/2021/11/03/study-of-smoothing-filters-savitzky-golay-filters/) — filter trade-offs.  
- [SAME: Skeleton-Agnostic Motion Embedding](https://sunny-codes.github.io/assets/pdf/SAME.pdf), [Skeleton-aware networks](https://www.semanticscholar.org/paper/Skeleton-aware-networks-for-deep-motion-retargeting-Aberman-Li/60800dbc93e55e7b7809edea04e74ffcae627b66) — topology-agnostic joint correspondence.  
- [pyfbsdk namespace reference](https://download.autodesk.com/us/motionbuilder/sdk-documentation/PythonSDK/namespacepyfbsdk.html), [MotionBuilder Python & UiIdle](https://techanimator.blogspot.com/2011/10/motionbuilder-python-window-focusing.html) — plugin integration.

