"""
MediGuard-AI Environment — OpenEnv-compliant RL environment.

Wraps PatientSimulator to produce normalized observations, compute rewards,
and support 3 hackathon tasks: suppression, deterioration, triage.

OpenEnv spec: reset() -> obs, step(action) -> (obs, reward, done, info), state() -> dict
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from patient_simulator import PatientSimulator
from reward_function import RewardFunction, Action, PatientCondition
from task1_suppression import grade_suppression
from task2_deterioration import grade_deterioration
from task3_triage import grade_triage

# ------------------------------------------------------------------ #
# Pydantic models (required by OpenEnv spec)                         #
# ------------------------------------------------------------------ #

class ObservationModel(BaseModel):
    """Schema for a single-patient observation dict."""
    heart_rate:        float = Field(..., ge=0.0, le=1.0, description="Normalized heart rate")
    systolic_bp:       float = Field(..., ge=0.0, le=1.0, description="Normalized systolic BP")
    diastolic_bp:      float = Field(..., ge=0.0, le=1.0, description="Normalized diastolic BP")
    spo2:              float = Field(..., ge=0.0, le=1.0, description="Normalized SpO2")
    respiratory_rate:  float = Field(..., ge=0.0, le=1.0, description="Normalized respiratory rate")
    temperature:       float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    baseline_delta:    float = Field(..., ge=0.0, le=1.0, description="Rolling deviation from personal baseline")
    hours_observed:    float = Field(..., ge=0.0,          description="Hours elapsed (step / 10.0)")
    activity:          int   = Field(..., ge=0,  le=4,     description="Current activity code")
    vitals_history:    list  = Field(...,                  description="Last 10 timesteps of normalized vitals [10][6]")


class ActionModel(BaseModel):
    action: Union[int, List[int]] = Field(
        ...,
        description="0=Ignore, 1=Verify, 2=Alert. List[int] of length 4 for triage."
    )


class RewardModel(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)
    done:   bool  = Field(...)
    step:   int   = Field(..., ge=0)


# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

NORM_RANGES = {
    "heart_rate":       (30.0,  200.0),
    "systolic_bp":      (60.0,  220.0),
    "diastolic_bp":     (30.0,  140.0),
    "spo2":             (70.0,  100.0),
    "respiratory_rate": (5.0,    40.0),
    "temperature":      (34.0,   42.0),
}

VITAL_KEYS = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate", "temperature"]

TASK_PATIENT_MAP = {
    "suppression":  "hypertensive",
    "deterioration": "deteriorating",
}

TRIAGE_PATIENTS = [
    ("healthy",      0),
    ("post_op",      1),
    ("deteriorating", 2),
    ("healthy",      3),
]

EPISODE_LENGTH = 60
NUM_ACTIONS    = 3
HISTORY_LEN    = 10


# ------------------------------------------------------------------ #
# Helper: per-patient observation tracker                             #
# ------------------------------------------------------------------ #

class _PatientTracker:
    def __init__(self, sim: PatientSimulator):
        self.sim = sim
        self.vitals_history: deque = deque(maxlen=HISTORY_LEN)
        self._running_sum   = np.zeros(len(VITAL_KEYS), dtype=np.float64)
        self._running_count = 0

    def reset(self, sim: PatientSimulator):
        self.sim = sim
        self.vitals_history.clear()
        self._running_sum[:] = 0.0
        self._running_count  = 0

    @staticmethod
    def _normalize(raw: Dict[str, float]) -> np.ndarray:
        arr = np.empty(len(VITAL_KEYS), dtype=np.float64)
        for i, key in enumerate(VITAL_KEYS):
            lo, hi = NORM_RANGES[key]
            arr[i] = np.clip((raw[key] - lo) / (hi - lo), 0.0, 1.0)
        return arr

    def build_observation(self, step: int) -> Dict:
        raw_vitals = self.sim.get_vitals()
        norm       = self._normalize(raw_vitals)

        self._running_sum   += norm
        self._running_count += 1
        rolling_mean         = self._running_sum / self._running_count
        baseline_delta       = float(np.clip(np.mean(np.abs(norm - rolling_mean)), 0.0, 1.0))

        self.vitals_history.append(norm.tolist())
        padded_history  = [[0.0] * len(VITAL_KEYS)] * (HISTORY_LEN - len(self.vitals_history))
        padded_history += list(self.vitals_history)

        obs = {
            "heart_rate":       float(norm[0]),
            "systolic_bp":      float(norm[1]),
            "diastolic_bp":     float(norm[2]),
            "spo2":             float(norm[3]),
            "respiratory_rate": float(norm[4]),
            "temperature":      float(norm[5]),
            "baseline_delta":   baseline_delta,
            "hours_observed":   step / 10.0,
            "activity":         int(self.sim.get_activity()),
            "vitals_history":   padded_history,
        }
        return obs


# ------------------------------------------------------------------ #
# MediGuardEnv                                                        #
# ------------------------------------------------------------------ #

class MediGuardEnv:
    """
    OpenEnv-compliant RL environment for MediGuard-AI.
    Episode length: 60 steps.
    """

    def __init__(self, task: str = "suppression", seed: int = 42):
        if task not in ("suppression", "deterioration", "triage"):
            raise ValueError(
                f"Unknown task '{task}'. Must be one of: suppression, deterioration, triage."
            )
        self._task    = task
        self._seed    = seed
        self._step    = 0
        self._trackers: List[_PatientTracker] = []
        self._is_triage = (task == "triage")

        if self._is_triage:
            self._reward_fns = [RewardFunction() for _ in TRIAGE_PATIENTS]
        else:
            self._reward_fns = [RewardFunction()]

    # -------------------------------------------------------------- #
    # Public API                                                       #
    # -------------------------------------------------------------- #

    def reset(self) -> Union[Dict, List[Dict]]:
        self._step = 0
        for rf in self._reward_fns:
            rf.reset()

        if self._is_triage:
            sims = [
                PatientSimulator(patient_type=pt, seed=self._seed + offset)
                for pt, offset in TRIAGE_PATIENTS
            ]
        else:
            patient_type = TASK_PATIENT_MAP[self._task]
            sims = [PatientSimulator(patient_type=patient_type, seed=self._seed)]

        self._trackers = [_PatientTracker(sim) for sim in sims]
        for tr in self._trackers:
            tr.sim.tick()

        obs_list = [tr.build_observation(self._step) for tr in self._trackers]
        return obs_list if self._is_triage else obs_list[0]

    def step(self, action: Union[int, List[int]]):
        if self._is_triage:
            if isinstance(action, (list, tuple)):
                actions = [max(0, min(2, int(a))) for a in action]
            else:
                actions = [max(0, min(2, int(action)))] * len(self._trackers)
            while len(actions) < len(self._trackers):
                actions.append(0)
            actions = actions[:len(self._trackers)]
        else:
            if isinstance(action, (list, tuple)):
                actions = [max(0, min(2, int(action[0])))]
            else:
                actions = [max(0, min(2, int(action)))]

        # 1. Tick all simulators
        for tr in self._trackers:
            tr.sim.tick()

        # 2. Build observations
        obs_list = [tr.build_observation(self._step) for tr in self._trackers]

        # 3. Compute reward
        obs_for_reward    = obs_list if self._is_triage else obs_list[0]
        action_for_reward = actions  if self._is_triage else actions[0]
        reward = self._compute_reward(action_for_reward, obs_for_reward)

        # 4. Increment step and check termination
        self._step += 1
        done = self._step >= EPISODE_LENGTH

        # 5. Build info dict
        if self._is_triage:
            patient_types = [pt for pt, _ in TRIAGE_PATIENTS]
        else:
            patient_types = TASK_PATIENT_MAP[self._task]

        info = {
            "step":         self._step,
            "patient_type": patient_types,
            "task":         self._task,
        }

        obs_out = obs_list if self._is_triage else obs_list[0]
        return obs_out, reward, done, info

    def state(self) -> Dict:
        if self._is_triage:
            patient_type    = [pt for pt, _ in TRIAGE_PATIENTS]
            det_severity    = [tr.sim.get_state().get("deterioration_severity", 0.0)
                               for tr in self._trackers]
            current_activity = [tr.sim.get_activity() for tr in self._trackers]
        else:
            patient_type     = TASK_PATIENT_MAP[self._task]
            det_severity     = self._trackers[0].sim.get_state().get("deterioration_severity", 0.0)
            current_activity = self._trackers[0].sim.get_activity()

        return {
            "step":                   self._step,
            "task":                   self._task,
            "patient_type":           patient_type,
            "done":                   self._step >= EPISODE_LENGTH,
            "current_activity":       current_activity,
            "deterioration_severity": det_severity,
        }

    # -------------------------------------------------------------- #
    # Condition classifier                                             #
    # -------------------------------------------------------------- #

    def _classify_condition(self, tracker: _PatientTracker) -> PatientCondition:
        state        = tracker.sim.get_state()
        det_severity = state.get("deterioration_severity", 0.0)

        # P1 fix: DRUG_MASKED previously required activity in (2, 3) which made
        # it nearly unreachable (~15% combined probability while also at high
        # severity). Clinically, "drug-masked" means medications are chemically
        # suppressing vital-sign expression — it is independent of activity.
        #
        # New thresholds:
        #   det_severity > 0.7  → DRUG_MASKED  (severe + masked by medication)
        #   det_severity > 0.5  → EMERGENCY    (severe, unmasked)
        #   det_severity > 0.2  → BORDERLINE
        #   else                → check absolute vitals, then STABLE
        #
        # DRUG_MASKED fires at a *higher* severity than EMERGENCY so it remains
        # rare but reachable, and represents the hardest detection case (vitals
        # may look deceptively near-normal despite life-threatening severity).
        if det_severity > 0.7:
            return PatientCondition.DRUG_MASKED

        if det_severity > 0.5:
            return PatientCondition.EMERGENCY

        if det_severity > 0.2:
            return PatientCondition.BORDERLINE

        # Absolute vital-sign safety net (catches non-deteriorating emergencies
        # such as the hypertensive spike injected by patient_simulator.py)
        vitals = tracker.sim.get_vitals()
        spo2   = vitals.get("spo2",         97)
        temp   = vitals.get("temperature",  37.0)
        hr     = vitals.get("heart_rate",   75)

        if spo2 < 80 or temp > 41.0 or hr > 170 or hr < 35:
            return PatientCondition.EMERGENCY
        if spo2 < 85 or temp > 39.5 or hr > 150 or hr < 40:
            return PatientCondition.BORDERLINE

        return PatientCondition.STABLE

    # -------------------------------------------------------------- #
    # Reward function (integrated)                                     #
    # -------------------------------------------------------------- #

    def _compute_reward(self, action, obs) -> float:
        if self._is_triage:
            rewards = []
            actions = action if isinstance(action, list) else [action]
            for act, tracker, rf in zip(actions, self._trackers, self._reward_fns):
                condition  = self._classify_condition(tracker)
                activity   = tracker.sim.get_activity()
                action_enum = Action(act)
                r = rf.compute(action_enum, condition, activity=activity)
                rewards.append(r)
            raw_reward = sum(rewards) / len(rewards)
        else:
            condition   = self._classify_condition(self._trackers[0])
            activity    = self._trackers[0].sim.get_activity()
            action_enum = Action(action)
            raw_reward  = self._reward_fns[0].compute(action_enum, condition, activity=activity)

        normalized = (raw_reward + 1.6) / 3.2
        return max(0.0, min(1.0, normalized))

    # -------------------------------------------------------------- #
    # Graders (integrated)                                             #
    # -------------------------------------------------------------- #

    def false_alarm_rate_grader(self) -> float:
        stats = self._reward_fns[0].get_stats()
        return grade_suppression(stats)

    def deterioration_grader(self) -> float:
        stats = self._reward_fns[0].get_stats()
        return grade_deterioration(stats)

    def triage_grader(self) -> float:
        stats_list = [rf.get_stats() for rf in self._reward_fns]
        return grade_triage(stats_list)


# ------------------------------------------------------------------ #
# Smoke test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 65)
    print("MediGuardEnv — Smoke Test")
    print("=" * 65)

    for task in ("suppression", "deterioration", "triage"):
        print(f"\n{'─' * 65}")
        print(f"Task: {task}")
        print(f"{'─' * 65}")

        env = MediGuardEnv(task=task, seed=42)
        obs = env.reset()

        for s in range(1, 6):
            action = [1, 0, 1, 0] if task == "triage" else 1
            obs, reward, done, info = env.step(action)

            if task == "triage":
                p0 = obs[0]
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"P0 HR={p0['heart_rate']:.3f} SpO2={p0['spo2']:.3f} "
                    f"delta={p0['baseline_delta']:.3f} act={p0['activity']}"
                )
            else:
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"HR={obs['heart_rate']:.3f} SpO2={obs['spo2']:.3f} "
                    f"delta={obs['baseline_delta']:.3f} act={obs['activity']}"
                )

        st = env.state()
        print(f"  state -> step={st['step']}, done={st['done']}, "
              f"deterioration_severity={st['deterioration_severity']}")

    print(f"\n{'=' * 65}")
    print("All smoke tests passed.")
    print(f"{'=' * 65}")
