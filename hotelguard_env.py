"""
HotelGuard-AI Environment — RL environment for hospitality crisis detection.

Wraps ZoneSimulator to produce normalized observations, compute rewards,
and support 3 tasks: suppression, deterioration, triage.

API: reset() -> obs, step(action) -> (obs, reward, done, info), state() -> dict
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from venue_simulator import ZoneSimulator
from reward_function import RewardFunction, Action, ZoneCondition
from task1_suppression import grade_suppression
from task2_deterioration import grade_deterioration
from task3_triage import grade_triage

# ------------------------------------------------------------------ #
# Pydantic models                                                     #
# ------------------------------------------------------------------ #

class ObservationModel(BaseModel):
    """Schema for a single-zone observation dict."""
    motion_level:      float = Field(..., ge=0.0, le=1.0, description="Normalized motion level")
    sound_db:          float = Field(..., ge=0.0, le=1.0, description="Normalized sound dB")
    door_events:       float = Field(..., ge=0.0, le=1.0, description="Normalized door events")
    panic_score:       float = Field(..., ge=0.0, le=1.0, description="Normalized panic score")
    occupancy_delta:   float = Field(..., ge=0.0, le=1.0, description="Normalized occupancy delta")
    smoke_co_level:    float = Field(..., ge=0.0, le=1.0, description="Normalized smoke/CO level")
    baseline_delta:    float = Field(..., ge=0.0, le=1.0, description="Rolling deviation from zone baseline")
    hours_observed:    float = Field(..., ge=0.0,          description="Hours elapsed (step / 10.0)")
    activity:          int   = Field(..., ge=0,  le=4,     description="Current context code")
    signal_history:    list  = Field(...,                  description="Last 10 timesteps of normalized signals [10][6]")


class ActionModel(BaseModel):
    action: Union[int, List[int]] = Field(
        ...,
        description="0=Monitor, 1=Dispatch, 2=Emergency. List[int] of length 4 for triage."
    )


class RewardModel(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)
    done:   bool  = Field(...)
    step:   int   = Field(..., ge=0)


# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

NORM_RANGES = {
    "motion_level":    (0.0,  100.0),
    "sound_db":        (30.0, 120.0),
    "door_events":     (0.0,   20.0),
    "panic_score":     (0.0,    1.0),
    "occupancy_delta": (0.0,   50.0),
    "smoke_co_level":  (0.0,    1.0),
}

SIGNAL_KEYS = ["motion_level", "sound_db", "door_events", "panic_score", "occupancy_delta", "smoke_co_level"]

TASK_ZONE_MAP = {
    "suppression":   "event_ballroom",
    "deterioration": "silent_room",
}

TRIAGE_ZONES = [
    ("lobby_normal",    0),
    ("restricted_zone", 1),
    ("silent_room",     2),
    ("lobby_normal",    3),
]

EPISODE_LENGTH = 60
NUM_ACTIONS    = 3
HISTORY_LEN    = 10


# ------------------------------------------------------------------ #
# Helper: per-zone observation tracker                                #
# ------------------------------------------------------------------ #

class _ZoneTracker:
    def __init__(self, sim: ZoneSimulator):
        self.sim = sim
        self.signal_history: deque = deque(maxlen=HISTORY_LEN)
        self._running_sum   = np.zeros(len(SIGNAL_KEYS), dtype=np.float64)
        self._running_count = 0

    def reset(self, sim: ZoneSimulator):
        self.sim = sim
        self.signal_history.clear()
        self._running_sum[:] = 0.0
        self._running_count  = 0

    @staticmethod
    def _normalize(raw: Dict[str, float]) -> np.ndarray:
        arr = np.empty(len(SIGNAL_KEYS), dtype=np.float64)
        for i, key in enumerate(SIGNAL_KEYS):
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

        self.signal_history.append(norm.tolist())
        padded_history  = [[0.0] * len(SIGNAL_KEYS)] * (HISTORY_LEN - len(self.signal_history))
        padded_history += list(self.signal_history)

        obs = {
            "motion_level":    float(norm[0]),
            "sound_db":        float(norm[1]),
            "door_events":     float(norm[2]),
            "panic_score":     float(norm[3]),
            "occupancy_delta": float(norm[4]),
            "smoke_co_level":  float(norm[5]),
            "baseline_delta":  baseline_delta,
            "hours_observed":  step / 10.0,
            "activity":        int(self.sim.get_activity()),
            "signal_history":  padded_history,
        }
        return obs


# ------------------------------------------------------------------ #
# HotelGuardEnv                                                       #
# ------------------------------------------------------------------ #

class HotelGuardEnv:
    """
    RL environment for HotelGuard-AI.
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
        self._trackers: List[_ZoneTracker] = []
        self._is_triage = (task == "triage")

        if self._is_triage:
            self._reward_fns = [RewardFunction() for _ in TRIAGE_ZONES]
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
                ZoneSimulator(zone_type=zt, seed=self._seed + offset)
                for zt, offset in TRIAGE_ZONES
            ]
        else:
            zone_type = TASK_ZONE_MAP[self._task]
            sims = [ZoneSimulator(zone_type=zone_type, seed=self._seed)]

        self._trackers = [_ZoneTracker(sim) for sim in sims]
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
            zone_types = [zt for zt, _ in TRIAGE_ZONES]
        else:
            zone_types = TASK_ZONE_MAP[self._task]

        info = {
            "step":      self._step,
            "zone_type": zone_types,
            "task":      self._task,
        }

        obs_out = obs_list if self._is_triage else obs_list[0]
        return obs_out, reward, done, info

    def state(self) -> Dict:
        if self._is_triage:
            zone_type        = [zt for zt, _ in TRIAGE_ZONES]
            det_severity     = [tr.sim.get_state().get("deterioration_severity", 0.0)
                               for tr in self._trackers]
            current_activity = [tr.sim.get_activity() for tr in self._trackers]
        else:
            zone_type        = TASK_ZONE_MAP[self._task]
            det_severity     = self._trackers[0].sim.get_state().get("deterioration_severity", 0.0)
            current_activity = self._trackers[0].sim.get_activity()

        return {
            "step":                   self._step,
            "task":                   self._task,
            "zone_type":              zone_type,
            "done":                   self._step >= EPISODE_LENGTH,
            "current_activity":       current_activity,
            "deterioration_severity": det_severity,
        }

    # -------------------------------------------------------------- #
    # Condition classifier                                             #
    # -------------------------------------------------------------- #

    def _classify_condition(self, tracker: _ZoneTracker) -> ZoneCondition:
        state        = tracker.sim.get_state()
        det_severity = state.get("deterioration_severity", 0.0)

        # P1 fix: ESCALATED previously required activity in (2, 3) which made
        # it nearly unreachable (~15% combined probability while also at high
        # severity). In hospitality, "escalated" means the situation has been
        # suppressed by context — it is independent of activity.
        #
        # New thresholds:
        #   det_severity > 0.7  → ESCALATED   (severe + masked by context)
        #   det_severity > 0.5  → EMERGENCY   (severe, unmasked)
        #   det_severity > 0.2  → BORDERLINE
        #   else                → check absolute signals, then STABLE
        #
        # ESCALATED fires at a *higher* severity than EMERGENCY so it remains
        # rare but reachable, and represents the hardest detection case (signals
        # may look deceptively near-normal despite life-threatening severity).
        if det_severity > 0.7:
            return ZoneCondition.ESCALATED

        if det_severity > 0.5:
            return ZoneCondition.EMERGENCY

        if det_severity > 0.2:
            return ZoneCondition.BORDERLINE

        # Absolute signal safety net (catches non-deteriorating emergencies
        # such as the event_ballroom spike injected by venue_simulator.py)
        vitals = tracker.sim.get_vitals()
        panic  = vitals.get("panic_score",    0.0)
        smoke  = vitals.get("smoke_co_level", 0.0)
        motion = vitals.get("motion_level",  50.0)

        if panic > 0.8 or smoke > 0.7 or motion < 2.0:
            return ZoneCondition.EMERGENCY
        if panic > 0.5 or smoke > 0.4 or motion < 5.0:
            return ZoneCondition.BORDERLINE

        return ZoneCondition.STABLE

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

    def suppression_grader(self) -> float:
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
    print("HotelGuardEnv — Smoke Test")
    print("=" * 65)

    for task in ("suppression", "deterioration", "triage"):
        print(f"\n{'─' * 65}")
        print(f"Task: {task}")
        print(f"{'─' * 65}")

        env = HotelGuardEnv(task=task, seed=42)
        obs = env.reset()

        for s in range(1, 6):
            action = [1, 0, 1, 0] if task == "triage" else 1
            obs, reward, done, info = env.step(action)

            if task == "triage":
                z0 = obs[0]
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"Z0 Motion={z0['motion_level']:.3f} Panic={z0['panic_score']:.3f} "
                    f"delta={z0['baseline_delta']:.3f} ctx={z0['activity']}"
                )
            else:
                print(
                    f"  step {s} | reward={reward:.2f} | done={done} | "
                    f"Motion={obs['motion_level']:.3f} Panic={obs['panic_score']:.3f} "
                    f"delta={obs['baseline_delta']:.3f} ctx={obs['activity']}"
                )

        st = env.state()
        print(f"  state -> step={st['step']}, done={st['done']}, "
              f"deterioration_severity={st['deterioration_severity']}")

    print(f"\n{'=' * 65}")
    print("All smoke tests passed.")
    print(f"{'=' * 65}")
