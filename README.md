---
title: MediGuard AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏨 HotelGuard AI

**Context-aware crisis detection and triage for hospitality venues — an AI agent that learns what normal looks like per zone, suppresses false alarms intelligently, and ranks simultaneous incidents so staff respond to the right emergency first.**

> Built for Google Solution Challenge 2026 · Rapid Crisis Response track  
> Solo project · Prototype submission by April 24, 2026

---

## The Problem

Hospitality venues — hotels, resorts, event spaces — generate constant sensor noise. A crowded ballroom during a wedding will spike sound levels, motion sensors, and door counts. A guest room at 2am is dead quiet. A naive alert system treats both the same way and drowns staff in false alarms, causing alarm fatigue. When a real emergency happens — a guest collapses, a fire starts slowly building, multiple incidents hit simultaneously across floors — staff either miss it or can't rank it.

The result: fragmented, delayed, uncoordinated response. The communication gap between a distressed guest, on-site staff, and emergency services costs critical minutes.

---

## What HotelGuard Does

HotelGuard is a reinforcement learning environment where an AI agent monitors multiple hotel zones simultaneously and must choose one of three actions at every timestep:

| Action | Code | Meaning |
|--------|:----:|---------|
| Monitor | `0` | Zone reads normal for this time and context — no action |
| Dispatch | `1` | Something is off — send staff to check |
| Emergency | `2` | Genuine crisis — call emergency services immediately |

**The core insight:** sensor readings mean completely different things depending on context. Sound at 95dB during a wedding reception is expected. Sound at 95dB in a corridor at 3am is not. Motion dropping to zero in a guest room during the afternoon is a warning sign. Motion at zero at 3am is normal.

HotelGuard scores signals as deviations from each zone's own rolling baseline — not from a population average — and weights them by venue context (event in progress, quiet hours, meal service, guest distress signal). An agent that learns this outperforms any threshold-based system.

---

## Three Scenarios — Easy to Hard

### Scenario 1 — Suppression (Easy)

**Setup:** The hotel ballroom during a three-hour wedding reception. Sound levels, motion, occupancy, and door events are all elevated throughout. This is completely expected. Between steps 30–55, a genuine panic event is injected — a guest in distress.

**Challenge:** Suppress the constant noise of a busy event while still catching the real emergency when it happens.

**Why this is hard for rule-based systems:** Any threshold agent that fires on high sound/motion will generate continuous false alarms during the reception. Only an agent that learns this zone's event-time baseline can achieve both high specificity and high sensitivity.

**Grader:** F1 score — harmonic mean of sensitivity (catching the real panic event) and specificity (not firing on wedding noise).

**Score range:** Rule-based ~0.55 · AI target ~0.75+

---

### Scenario 2 — Deterioration (Medium)

**Setup:** A guest room where something is slowly going wrong over 60 timesteps. Starting at step 30: motion drops gradually, the door stops opening, smoke/CO level begins a slow climb, panic score inches up. Classic signature of a medical emergency or early fire buildup.

**Challenge:** Detect the trend early. Single-step thresholds miss this — the individual readings stay within plausible range for many steps. Only an agent reading the signal history table can catch the drift before it becomes a full crisis.

**Grader:** Onset-delay scoring: `score = 0.4 + 0.6 × (1 − delay/30)`. Catching it at step 32 scores near 1.0. Catching it at step 58 scores near 0.4. Missing it entirely scores 0.

**Score range:** Rule-based ~0.30–0.45 · AI target ~0.55+

---

### Scenario 3 — Triage (Hard)

**Setup:** Four simultaneous zones: lobby (normal), event corridor (elevated but expected), guest room floor (slow deterioration in progress), pool area (healthy baseline). The agent receives all four observations at once and must allocate response correctly.

**Challenge:** This is a ranking problem, not four independent decisions. A MONITOR-everything agent scores ~0.22. Getting it right means comparing zones against each other and escalating the right one.

**Grader:** Composite score:
- NDCG@4 (50%) — did the agent rank zones by actual urgency?
- Emergency F1 (30%) — precision and recall on committing EMERGENCY to the right zone
- Responsiveness (20%) — how quickly did the agent escalate after the situation changed?
- Concentration penalty — same action for every zone is penalised
- Hesitation penalty — DISPATCH on an EMERGENCY zone wastes critical time

**Score range:** Rule-based ~0.22–0.28 · AI target ~0.65+

---

## Observation Space — 6 Signals per Zone

All signals normalized 0–1 against each zone's own 3-hour rolling baseline.

| Field | Raw Range | Description |
|-------|-----------|-------------|
| `motion_level` | 0–100 | Movement activity in zone |
| `sound_db` | 30–120 dB | Ambient sound level |
| `door_events` | 0–20/min | Door open/close frequency |
| `panic_score` | 0–1.0 | Panic button / distress signal strength |
| `occupancy_delta` | 0–50 | Deviation from expected occupancy |
| `smoke_co_level` | 0–1.0 | Combined smoke and CO sensor reading |
| `baseline_delta` | 0–1.0 | Combined deviation from zone's personal baseline |
| `hours_observed` | ≥0 | Elapsed time (step / 10.0) |
| `context` | 0–4 | Venue context code (see below) |
| `signal_history` | [10][6] | Last 10 timesteps of all 6 signals |

**Venue context codes:**

| Code | Context | Effect |
|:----:|---------|--------|
| 0 | Quiet hours (2am–6am) | 1.00× — full weight on anomalies |
| 1 | Meal service | 0.40× — elevated occupancy expected |
| 2 | Event in progress | 0.50× — noise/motion expected |
| 3 | Guest distress signal | 1.25× — amplified concern |
| 4 | Physical emergency confirmed | 1.60× — maximum weight |

---

## Reward Design

```
Base reward table (action × zone condition):

              | STABLE | BORDERLINE | EMERGENCY | ESCALATED |
EMERGENCY (2) |  -0.5  |    +0.2    |   +1.0    |   +1.0    |
DISPATCH  (1) |  -0.1  |    +0.7    |   +0.3    |   +0.3    |
MONITOR   (0) |  +0.2  |    -0.2    |   -1.0    |   -1.0    |

Venue context multipliers:
  Quiet hours    → 1.00× (anomaly unexplained — full weight)
  Meal service   → 0.40× (activity expected)
  Event progress → 0.50× (noise/motion expected)
  Guest distress → 1.25× (distress compounds risk)
  Physical emerg → 1.60× (maximum weight)

Alarm fatigue:  >5 emergency calls in last 30 steps → 0.6× multiplier
Zone learning:  Correctly monitoring a stable zone after step 20 → +0.2 bonus
```

---

## Google Technology Integration

| Technology | How it's used |
|------------|---------------|
| **Gemini API** | Powers the AI triage agent — replaces rule-based thresholds with context-aware reasoning across all three scenarios |
| **Firebase Realtime Database** | Live sensor event streaming — zone readings flow through Firebase, enabling real-time dashboard updates without polling |
| **MediaPipe** | Camera-based distress detection — pose landmark model classifies guest activity (standing, lying still, distressed posture) and feeds directly into the `context` observation field |
| **Google Maps Platform** | Venue floor plan — zones plotted on a hotel map, alert levels shown as color overlays |

---

## Project Structure

```
HotelGuard-AI/
├── app.py                  # Gradio UI + FastAPI dashboard (entry point)
├── hotelguard_env.py       # RL environment — reset() / step() / state()
├── venue_simulator.py      # Zone signal generator (4 zone types, seeded)
├── reward_function.py      # Stateful reward calculator
├── task1_suppression.py    # Grader: F1 (sensitivity + specificity)
├── task2_deterioration.py  # Grader: onset-delay scoring
├── task3_triage.py         # Grader: NDCG@4 + F1 + responsiveness
├── inference.py            # Gemini agent + rule-based fallback
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/SutikshanUpman/HotelGuard-AI
cd HotelGuard-AI
pip install -r requirements.txt

# Run the dashboard (no API key needed — use Rule-Based mode)
python app.py

# Run Gemini inference across all 3 scenarios (requires GEMINI_API_KEY)
python inference.py
```

**Docker:**
```bash
docker build -t hotelguard-ai .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=your_key \
  -e FIREBASE_URL=your_firebase_url \
  hotelguard-ai
```

**Use the environment directly:**
```python
from hotelguard_env import HotelGuardEnv

# Scenario 1 — ballroom event suppression
env = HotelGuardEnv(task="suppression", seed=42)
obs = env.reset()
while True:
    obs, reward, done, info = env.step(0)  # Monitor
    if done:
        break
print(f"Score: {env.suppression_grader():.4f}")

# Scenario 3 — four simultaneous zones
env = HotelGuardEnv(task="triage", seed=42)
obs = env.reset()  # list of 4 observation dicts
obs, reward, done, info = env.step([2, 0, 1, 0])
print(f"Score: {env.triage_grader():.4f}")
```

---

## Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `GEMINI_API_KEY` | Yes* | Google Gemini API key (*Rule-Based mode works without it) |
| `FIREBASE_URL` | No | Firebase Realtime DB URL for live streaming |

---

## Results

| Scenario | Rule-Based Baseline | HotelGuard AI (Gemini) | Improvement |
|----------|:-------------------:|:----------------------:|:-----------:|
| Suppression (F1) | 0.4188 | — | — |
| Deterioration (onset-delay) | 0.7533 | — | — |
| Triage (composite) | 0.2330 | — | — |

*Rule-based baseline verified via local Gradio UI (seed 42). Gemini scores pending — will update after API key test.*

---

## Why This Matters

Standard hotel emergency systems fire on absolute thresholds. They don't know that the ballroom is always loud on Friday nights. They don't know that Room 412 usually has high occupancy at 8pm. They treat every spike the same.

HotelGuard learns the normal for each zone. The same reading that triggers an alert at 3am in a quiet corridor gets suppressed at 10pm in an active event space. This is the difference between a system that drowns staff in noise and one they actually trust.

When a real emergency happens — and it will — a trusted system gets the right response to the right place in the minimum time possible.

---

## Author

**Sutikshan Upman**  
Built for Google Solution Challenge 2026 — Rapid Crisis Response track.

*HotelGuard AI — because the difference between 2 minutes and 8 minutes is everything.*