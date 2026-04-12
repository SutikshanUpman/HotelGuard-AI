# MIGRATION.md — MediGuard → HotelGuard

**This document is the sole reference for adapting MediGuard-AI into HotelGuard-AI.**  
Read it before touching any file. Every decision here is based on reading the actual source code.

---

## Core principle

The RL environment architecture, reward logic, grader math, inference pipeline, and Gradio UI structure are all domain-agnostic. They don't care whether they're watching a patient or a hotel zone. The only things that are medical-specific are: signal names, system prompts, and terminology in strings/comments. Everything else reuses directly.

**Estimated migration effort: 2 days for a focused solo builder.**

---

## File-by-file decisions

### DELETE these files entirely

| File | Reason |
|------|--------|
| `openenv.yaml` | Hackathon spec file, no value here |
| `validation.sh` | OpenEnv validator script, not needed |
| `pyproject.toml` | uv/pyproject config for the hackathon, use plain requirements.txt |
| `uv.lock` | Same — lockfile for uv, not needed |
| `entrypoint.sh` | Was specific to the OpenEnv docker setup, rewrite Dockerfile entrypoint directly |

---

### RENAME + LIGHT REWRITE (keep structure, swap terminology)

#### `patient_simulator.py` → `venue_simulator.py`

This is the most work but it's mechanical. The class structure (`__init__`, `tick()`, `get_vitals()`, `get_activity()`, `get_state()`, `reset()`) stays **100% identical**. The environment calls these exact method names — don't rename them.

What changes:

**Class name:** `PatientSimulator` → `ZoneSimulator`

**`patient_type` parameter → `zone_type` parameter.** Replace the 5 patient types with 4 zone types:

| Remove | Replace with | Used in |
|--------|-------------|---------|
| `"healthy"` | `"lobby_normal"` | Triage background zones |
| `"hypertensive"` | `"event_ballroom"` | Task 1 — suppression |
| `"deteriorating"` | `"silent_room"` | Task 2 — deterioration |
| `"post_op"` | `"restricted_zone"` | Triage second zone |
| `"unstable"` | remove | Not needed |

**`VitalRanges` dataclass → `ZoneSignalRanges`.** Replace the 6 vital fields with 6 sensor signals:

| Remove (medical) | Replace with (hospitality) | Raw range |
|------------------|---------------------------|-----------|
| `heart_rate` | `motion_level` | 0–100 |
| `systolic_bp` | `sound_db` | 30–120 dB |
| `diastolic_bp` | `door_events` | 0–20/min |
| `spo2` | `panic_score` | 0–1.0 |
| `respiratory_rate` | `occupancy_delta` | 0–50 |
| `temperature` | `smoke_co_level` | 0–1.0 |

**`_initialize_baselines()`.** Keep the method, rename to `_initialize_zone_baselines()`. Rewrite the per-type baseline values for hospitality:

```python
# event_ballroom — noisy by design
self.baseline_motion  = 75.0   # high activity
self.baseline_sound   = 88.0   # loud
self.baseline_doors   = 12.0   # frequent entry/exit
self.baseline_panic   = 0.05   # near zero
self.baseline_occ     = 40.0   # high occupancy
self.baseline_smoke   = 0.02   # clean air

# silent_room — quiet baseline, will deteriorate
self.baseline_motion  = 15.0
self.baseline_sound   = 35.0
self.baseline_doors   = 2.0
self.baseline_panic   = 0.02
self.baseline_occ     = 5.0
self.baseline_smoke   = 0.01
```

**Activity codes → Venue context codes.** The 5 activity codes (0–4) stay as integers. Just rename what they mean:

| Code | Was | Now |
|:----:|-----|-----|
| 0 | resting | quiet hours (2am–6am) |
| 1 | eating | meal service |
| 2 | walking | event in progress |
| 3 | distressed | guest distress signal |
| 4 | falling | physical emergency confirmed |

The `_apply_activity_effects()` method stays — just retune the delta values for hospitality signals instead of vital signs.

**The emergency spike injection (lines 325–335 in patient_simulator.py).** Keep this exactly. It fires between steps 30–55 for the `event_ballroom` zone type, forcing a genuine panic event so the suppression grader always has a real emergency to score sensitivity against. Just change the signal values:

```python
if self.zone_type == "event_ballroom" and not self._spike_injected and 30 <= self.timestep <= 55:
    vitals["panic_score"]   = 0.95
    vitals["motion_level"]  = 5.0    # everyone has stopped
    vitals["sound_db"]      = 45.0   # sudden quiet — crowd reacting
    self.current_activity   = 0      # force quiet hours context
    self._spike_injected    = True
```

---

#### `mediguard_env.py` → `hotelguard_env.py`

**Class name:** `MediGuardEnv` → `HotelGuardEnv`

**`NORM_RANGES` dict.** Update the 6 keys and ranges to match the new signals:

```python
NORM_RANGES = {
    "motion_level":    (0.0,   100.0),
    "sound_db":        (30.0,  120.0),
    "door_events":     (0.0,    20.0),
    "panic_score":     (0.0,     1.0),
    "occupancy_delta": (0.0,    50.0),
    "smoke_co_level":  (0.0,     1.0),
}
```

**`VITAL_KEYS` → `SIGNAL_KEYS`.** Update the list to the 6 new signal names. This list drives the normalization loop and `vitals_history` — it must match exactly.

**`TASK_PATIENT_MAP` → `TASK_ZONE_MAP`.** Update values:
```python
TASK_ZONE_MAP = {
    "suppression":   "event_ballroom",
    "deterioration": "silent_room",
}
```

**`TRIAGE_PATIENTS` → `TRIAGE_ZONES`.** Update the 4 zone types:
```python
TRIAGE_ZONES = [
    ("lobby_normal",    0),
    ("restricted_zone", 1),
    ("silent_room",     2),
    ("lobby_normal",    3),
]
```

**`_classify_condition()`.** Keep the logic completely. Just update the absolute vital-sign safety net at the bottom (lines 291–299) to use the new signal names:

```python
# was: spo2 < 80 or temp > 41.0 or hr > 170
# now:
panic  = vitals.get("panic_score",   0.0)
smoke  = vitals.get("smoke_co_level", 0.0)
motion = vitals.get("motion_level",  50.0)

if panic > 0.8 or smoke > 0.7 or motion < 2.0:
    return ZoneCondition.EMERGENCY
if panic > 0.5 or smoke > 0.4 or motion < 5.0:
    return ZoneCondition.BORDERLINE
```

**Grader method names.** Rename for clarity but keep the logic:
- `false_alarm_rate_grader()` → `suppression_grader()`
- `deterioration_grader()` → same (keep)
- `triage_grader()` → same (keep)

**Everything else in this file** — `_PatientTracker` (rename to `_ZoneTracker`), `build_observation()`, `_compute_reward()`, `reset()`, `step()`, `state()` — is structurally identical. Only string/comment changes needed.

---

#### `reward_function.py`

**Zero structural changes.** This file is completely domain-agnostic. The reward table values and multiplier logic work identically for hospitality.

Only rename:
- `PatientCondition` → `ZoneCondition`
- `DRUG_MASKED` → `ESCALATED` (same semantics — condition masked/suppressed, hardest to detect)
- Update docstring comments from patient/medical language

The actual numbers in `REWARD_TABLE` and `ACTIVITY_CONTEXT` stay the same on Day 1. Tune them after you have a working baseline if needed.

---

#### `task1_suppression.py`, `task2_deterioration.py`, `task3_triage.py`

**Zero logic changes.** These graders operate entirely on the action/condition/activity history from `RewardFunction.get_stats()`. They don't know or care what the signals are called.

Only changes needed:
- Update docstrings and any string references to patient/medical terms
- Rename any `patient` variables to `zone` for clarity

---

#### `inference.py`

**Structure:** Keep everything. The LLM call machinery, fallback logic, conversation history, `run_episode()`, logging — all identical.

**What actually changes:**

1. Replace the model endpoint. Swap from HuggingFace router + Qwen to Gemini:

```python
# Remove:
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Add:
import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
```

2. Replace `ACTIVITY_NAMES` dict:
```python
CONTEXT_NAMES = {
    0: "quiet hours",
    1: "meal service",
    2: "event in progress",
    3: "guest distress signal",
    4: "physical emergency",
}
```

3. Rewrite the three system prompts (`SUPPRESSION_PROMPT`, `DETERIORATION_PROMPT`, `TRIAGE_PROMPT`) in hospitality language. The structure — rules, thresholds, JSON output format — stays identical. Just swap ICU → hotel, vitals → sensor signals, patient → zone.

4. Update `obs_to_user_message()` — rename the vitals fields in the display string to match the new signal names.

5. Remove:
   - `openenv_validate()` call in `main()`
   - `BASELINE_SCORES` dict
   - `[IMPROVEMENT] vs_baseline` log line
   - `[VALIDATE]` log line

6. Update import: `from mediguard_env import MediGuardEnv` → `from hotelguard_env import HotelGuardEnv`

---

#### `server/app.py`

**Keep the entire FastAPI + Gradio skeleton.** This is the most complex file and the least work — almost everything is structural.

**What changes:**

1. Update all import references (`MediGuardEnv` → `HotelGuardEnv`, signal names in format functions)

2. Rename `_fmt_single()` and `_fmt_triage()` display strings — swap vital sign labels for sensor signal labels

3. Update `_risk_tag()` — the threshold logic stays, just change which fields it reads:
```python
# was: delta, spo2, temp
# now: delta, panic_score, smoke_co_level
def _risk_tag(delta, panic, smoke):
    if delta > 0.5 or panic > 0.6 or smoke > 0.5:
        return "CRITICAL"
    elif delta > 0.25 or panic > 0.3 or smoke > 0.2:
        return "BORDERLINE"
    return "STABLE"
```

4. Update `ACTION_LABELS` and `ACTION_EMOJI`:
```python
ACTION_LABELS = {0: "MONITOR", 1: "DISPATCH", 2: "EMERGENCY"}
ACTION_EMOJI  = {0: "✅",      1: "👁️",       2: "🚨"}
```

5. Remove the `/reset` and `/step` FastAPI REST endpoints — these were for the OpenEnv validator. Keep `/health` and `/score`. The Gradio UI is all you need for the demo.

6. Add a zone floor plan panel to the Gradio UI — a simple 2×2 grid of zone cards showing current status, color coded by risk level. This is the demo money shot and takes maybe 30 lines of Gradio HTML.

---

### Summary table

| File | Action | Effort |
|------|--------|--------|
| `patient_simulator.py` | Rewrite signals and zone types, keep all method signatures | ~2 hours |
| `mediguard_env.py` | Rename class + signal keys, update condition classifier | ~1 hour |
| `reward_function.py` | Rename enums only | ~15 min |
| `task1/2/3_*.py` | Docstrings + variable names only | ~30 min |
| `inference.py` | New model endpoint + new system prompts + remove OpenEnv calls | ~2 hours |
| `server/app.py` | Signal labels + action labels + remove REST endpoints + add floor plan | ~2 hours |
| `openenv.yaml` | DELETE | 0 |
| `validation.sh` | DELETE | 0 |
| `pyproject.toml` | DELETE | 0 |
| `uv.lock` | DELETE | 0 |
| `entrypoint.sh` | DELETE | 0 |

**Total realistic estimate: 1 full focused day to get a working running prototype.**

---

## First commit checklist

Do this in order on Day 1:

1. Fork/copy the MediGuard-AI repo into `HotelGuard-AI`
2. Delete the 5 files listed above
3. Global find-replace: `MediGuardEnv` → `HotelGuardEnv`, `PatientSimulator` → `ZoneSimulator`, `patient_type` → `zone_type`, `mediguard` → `hotelguard`
4. Update the 6 signal names in `NORM_RANGES` and `SIGNAL_KEYS`
5. Run `python hotelguard_env.py` — smoke test should pass with zero other changes
6. Commit: `init: hotelguard scaffold from mediguard base`

If the smoke test passes after step 5, the core environment is working. Everything after that is signal tuning, prompt writing, and UI polish.

---

## What NOT to change (ever)

- The `tick()` / `get_vitals()` / `get_activity()` interface on `ZoneSimulator` — the env calls these exactly
- The `reset()` / `step()` / `state()` interface on `HotelGuardEnv` — the inference script calls these exactly  
- The `RewardFunction.compute()` signature — the env calls this exactly
- The grader method signatures — inference.py calls these exactly
- Episode length of 60 steps — all three graders assume this
- Seed 42 — ensures reproducible demo scores
