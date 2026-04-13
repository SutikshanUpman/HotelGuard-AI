import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ZoneSignalRanges:
    """
    Normal ranges for each sensor signal.
    We'll use these as baselines and add noise/drift.
    """
    motion_level_range: Tuple[float, float] = (0, 100)         # Motion level (0–100)
    sound_db_range: Tuple[float, float] = (30, 120)            # Sound dB (30–120)
    door_events_range: Tuple[float, float] = (0, 20)           # Door events (0–20/min)
    panic_score_range: Tuple[float, float] = (0.0, 1.0)        # Panic score (0–1.0)
    occupancy_delta_range: Tuple[float, float] = (0, 50)       # Occupancy delta (0–50)
    smoke_co_level_range: Tuple[float, float] = (0.0, 1.0)     # Smoke/CO level (0–1.0)


class ZoneSimulator:
    """
    Simulates a single hotel zone's sensor signals over time.

    A realistic zone data generator.
    No decision-making logic, just produces believable sensor signal streams.
    """

    def __init__(
        self,
        zone_type: str = "lobby_normal",
        seed: int = None,
        baseline_motion: float = None,
        baseline_sound: float = None,
        baseline_doors: float = None
    ):
        """
        Initialize the zone simulator.

        Args:
            zone_type: Type of zone simulation
                - "lobby_normal":      Normal lobby signals with small noise
                - "event_ballroom":    High baseline sound/motion (busy event)
                - "silent_room":       Slow deterioration over time
                - "restricted_zone":   Low activity, restricted access
            seed: Random seed for reproducibility (important for testing!)
            baseline_motion, baseline_sound, baseline_doors:
                Override default baselines for personalization
        """
        self.rng = np.random.default_rng(seed)
        self.zone_type = zone_type
        self.timestep = 0

        self._custom_baseline_motion = baseline_motion
        self._custom_baseline_sound = baseline_sound
        self._custom_baseline_doors = baseline_doors

        self.activity_weights = [70, 10, 10, 5, 5]
        self.current_activity = 0

        self.deterioration_start_time = None
        self.deterioration_severity = 0.0

        # P0 fix: guaranteed emergency spike for event_ballroom zone (Task 1).
        # Fires exactly once between steps 30-55 so grade_suppression() always
        # has at least one real emergency to score sensitivity against,
        # preventing a silent agent from trivially scoring 1.0.
        self._spike_injected = False

        self._initialize_zone_baselines(baseline_motion, baseline_sound, baseline_doors)

        # Initial signals are at quiet hours state (activity 0)
        self.current_vitals = self._generate_baseline_vitals()
        self.last_vitals = self.current_vitals.copy()

    def _initialize_zone_baselines(
        self,
        baseline_motion: float = None,
        baseline_sound: float = None,
        baseline_doors: float = None
    ):
        """
        Set zone's personal baseline signals.
        """
        if self.zone_type == "lobby_normal":
            self.baseline_motion = baseline_motion if baseline_motion is not None else 40.0
            self.baseline_sound  = baseline_sound  if baseline_sound  is not None else 55.0
            self.baseline_doors  = baseline_doors  if baseline_doors  is not None else 6.0
            self.baseline_panic  = 0.02
            self.baseline_occ    = 20.0
            self.baseline_smoke  = 0.01

        elif self.zone_type == "event_ballroom":
            self.baseline_motion = baseline_motion if baseline_motion is not None else 75.0
            self.baseline_sound  = baseline_sound  if baseline_sound  is not None else 88.0
            self.baseline_doors  = baseline_doors  if baseline_doors  is not None else 12.0
            self.baseline_panic  = 0.05
            self.baseline_occ    = 40.0
            self.baseline_smoke  = 0.02

        elif self.zone_type == "silent_room":
            self.baseline_motion = baseline_motion if baseline_motion is not None else 15.0
            self.baseline_sound  = baseline_sound  if baseline_sound  is not None else 35.0
            self.baseline_doors  = baseline_doors  if baseline_doors  is not None else 2.0
            self.baseline_panic  = 0.02
            self.baseline_occ    = 5.0
            self.baseline_smoke  = 0.01
            self.deterioration_start_time = 30

        elif self.zone_type == "restricted_zone":
            self.baseline_motion = baseline_motion if baseline_motion is not None else 10.0
            self.baseline_sound  = baseline_sound  if baseline_sound  is not None else 40.0
            self.baseline_doors  = baseline_doors  if baseline_doors  is not None else 3.0
            self.baseline_panic  = 0.03
            self.baseline_occ    = 8.0
            self.baseline_smoke  = 0.01

        else:
            self.baseline_motion = baseline_motion if baseline_motion is not None else 40.0
            self.baseline_sound  = baseline_sound  if baseline_sound  is not None else 55.0
            self.baseline_doors  = baseline_doors  if baseline_doors  is not None else 6.0
            self.baseline_panic  = 0.02
            self.baseline_occ    = 20.0
            self.baseline_smoke  = 0.01

    def _generate_baseline_vitals(self) -> Dict[str, float]:
        """
        Generate signals around the baseline with small random noise.
        """
        return {
            "motion_level":    self.baseline_motion + self.rng.normal(0, 3),
            "sound_db":        self.baseline_sound  + self.rng.normal(0, 5),
            "door_events":     self.baseline_doors  + self.rng.normal(0, 1),
            "panic_score":     np.clip(self.baseline_panic + self.rng.normal(0, 0.02), 0.0, 1.0),
            "occupancy_delta": self.baseline_occ    + self.rng.normal(0, 2),
            "smoke_co_level":  np.clip(self.baseline_smoke + self.rng.normal(0, 0.01), 0.0, 1.0),
        }

    def _apply_activity_effects(self, vitals: Dict[str, float], activity: int) -> Dict[str, float]:
        """
        Modify signals based on the current venue context.
        Activity codes:
            0: quiet hours (2am–6am)
            1: meal service
            2: event in progress
            3: guest distress signal
            4: physical emergency confirmed
        """
        modified = vitals.copy()

        if activity == 0:    # Quiet hours (2am–6am) — no adjustment
            pass
        elif activity == 1:  # Meal service — moderate increase in motion and sound
            modified["motion_level"]    += self.rng.uniform(5, 15)      # 5–15% of 0–100 range
            modified["sound_db"]        += self.rng.uniform(5, 10)      # 5–10 dB bump
            modified["occupancy_delta"] += self.rng.uniform(3, 8)       # more people moving
        elif activity == 2:  # Event in progress — high motion, sound, door traffic
            modified["motion_level"]    += self.rng.uniform(15, 30)     # significant activity
            modified["sound_db"]        += self.rng.uniform(10, 25)     # loud event
            modified["door_events"]     += self.rng.uniform(3, 8)       # frequent entry/exit
            modified["occupancy_delta"] += self.rng.uniform(5, 15)      # crowd influx
        elif activity == 3:  # Guest distress signal — panic rises, motion spikes
            modified["panic_score"]     += self.rng.uniform(0.15, 0.35)
            modified["motion_level"]    += self.rng.uniform(10, 25)
            modified["sound_db"]        += self.rng.uniform(10, 20)
            modified["door_events"]     += self.rng.uniform(2, 5)
        elif activity == 4:  # Physical emergency confirmed — extreme readings
            modified["panic_score"]     += self.rng.uniform(0.30, 0.60)
            modified["motion_level"]    += self.rng.uniform(20, 40)
            modified["sound_db"]        += self.rng.uniform(15, 30)
            modified["door_events"]     += self.rng.uniform(4, 10)

        return modified

    def _apply_deterioration(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply slow deterioration for "silent_room" zone type (e.g. medical
        emergency or early fire buildup in a guest room).

        Severity goes from 0.0 to 1.0 over 30 timesteps after
        deterioration_start_time. Each signal drifts proportionally:
            smoke_co_level → slowly rises (fire buildup)
            panic_score    → climbs (guest distress)
            motion_level   → drops (guest incapacitated)
            sound_db       → drops (silence)
            door_events    → drops (no one entering/leaving)
        """
        # Event ballroom zone: inject a genuine emergency spike at a fixed
        # window (timesteps 38–45) so the Task 1 grader always has at least
        # one real emergency to detect, preventing the trivial "always MONITOR"
        # exploit that scored ~1.0 on the no-emergency grader branch.
        if self.zone_type == "event_ballroom":
            if 38 <= self.timestep <= 45:
                modified = vitals.copy()
                # Sudden crisis: panic spikes, motion crashes, sound drops
                modified["panic_score"]     += 0.70 + self.rng.normal(0, 0.03)
                modified["motion_level"]    -= 50.0 + self.rng.normal(0, 3.0)
                modified["sound_db"]        -= 30.0 + self.rng.normal(0, 4.0)
                self.deterioration_severity = 0.75   # triggers EMERGENCY in classifier
            else:
                self.deterioration_severity = 0.0
                modified = vitals.copy()
            return modified

        if self.zone_type != "silent_room":
            return vitals

        if self.deterioration_start_time is None or self.timestep < self.deterioration_start_time:
            return vitals

        time_since_start = self.timestep - self.deterioration_start_time
        self.deterioration_severity = min(time_since_start / 30.0, 1.0)

        modified    = vitals.copy()
        noise_scale = 1.0 + self.deterioration_severity * 2.0

        modified["smoke_co_level"]  += (self.deterioration_severity * 0.6
                                         + self.rng.normal(0, 0.02 * noise_scale))
        modified["panic_score"]     += (self.deterioration_severity * 0.5
                                         + self.rng.normal(0, 0.02 * noise_scale))
        modified["motion_level"]    -= (self.deterioration_severity * 12.0
                                         + self.rng.normal(0, 1.0 * noise_scale))
        modified["sound_db"]        -= (self.deterioration_severity * 15.0
                                         + self.rng.normal(0, 2.0 * noise_scale))
        modified["door_events"]     -= (self.deterioration_severity * 1.5
                                         + self.rng.normal(0, 0.3 * noise_scale))
        modified["occupancy_delta"] -= (self.deterioration_severity * 4.0
                                         + self.rng.normal(0, 0.5 * noise_scale))

        return modified

    def _smooth_transition(self, new_vitals: Dict[str, float], smoothing: float = 0.3) -> Dict[str, float]:
        """
        Smooth transitions between timesteps.

        Blends 70% of the old value with 30% of the new target so signals
        can't jump unrealistically in a single step.
        """
        smoothed = {}
        for key in new_vitals:
            if key in self.last_vitals:
                smoothed[key] = (1 - smoothing) * self.last_vitals[key] + smoothing * new_vitals[key]
            else:
                smoothed[key] = new_vitals[key]
        return smoothed

    def _sample_new_activity(self) -> int:
        """
        Randomly pick a new activity context for this timestep.
        """
        activity = self.rng.choice(
            [0, 1, 2, 3, 4],
            p=np.array(self.activity_weights) / sum(self.activity_weights)
        )
        self.current_activity = activity
        return activity

    def get_activity(self) -> int:
        """
        Read the current activity/context state. Read-only.
        """
        return self.current_activity

    def get_vitals(self) -> Dict[str, float]:
        """
        Get current sensor signals for this timestep. Read-only.
        """
        return self.current_vitals.copy()

    def tick(self):
        """
        Advance simulation time by 1 timestep and update signals.

        Pipeline:
            1. Generate fresh baseline signals with noise
            2. Apply activity effects (what context is the zone in?)
            3. Apply deterioration (are conditions getting worse?)
            4. Smooth the transition from last reading
            5. Clip everything to valid signal ranges
            6. Inject guaranteed emergency spike (event_ballroom only, once, steps 30-55)
            7. Save the result
        """
        self.timestep += 1
        self._sample_new_activity()

        # 1. Generate target baseline with noise
        vitals = self._generate_baseline_vitals()

        # 2. Apply activity effects
        vitals = self._apply_activity_effects(vitals, self.current_activity)

        # 3. Apply deterioration
        vitals = self._apply_deterioration(vitals)

        # 4. Smooth transition from last timestep
        vitals = self._smooth_transition(vitals)

        # 5. Clip to valid signal ranges
        vitals["motion_level"]    = np.clip(vitals["motion_level"],     0,  100)
        vitals["sound_db"]        = np.clip(vitals["sound_db"],        30,  120)
        vitals["door_events"]     = np.clip(vitals["door_events"],      0,   20)
        vitals["panic_score"]     = np.clip(vitals["panic_score"],    0.0,  1.0)
        vitals["occupancy_delta"] = np.clip(vitals["occupancy_delta"],  0,   50)
        vitals["smoke_co_level"]  = np.clip(vitals["smoke_co_level"], 0.0,  1.0)

        # 6. Guaranteed emergency spike for event_ballroom zone (Task 1 / P0 fix).
        #    Forces at least one unambiguous EMERGENCY between steps 30-55 so
        #    grade_suppression() always has real sensitivity to score — preventing
        #    a silent agent from trivially achieving 1.0 with no emergencies in
        #    the episode. Fires exactly once per episode (reset() clears the flag).
        #    Activity is forced to quiet hours (0) so the spike is not activity-discounted
        #    and _classify_condition() returns EMERGENCY (not ESCALATED).
        if (
            self.zone_type == "event_ballroom"
            and not self._spike_injected
            and 30 <= self.timestep <= 55
        ):
            vitals["panic_score"]   = 0.95
            vitals["motion_level"]  = 5.0
            vitals["sound_db"]      = 45.0
            self.current_activity   = 0   # force quiet hours so condition → EMERGENCY
            self._spike_injected    = True

        # 7. Store for next iteration and for observation
        self.last_vitals    = vitals.copy()
        self.current_vitals = vitals.copy()

    def reset(self, zone_type: str = None, seed: int = None):
        """
        Reset simulator to initial state.
        """
        if zone_type is not None:
            self.zone_type = zone_type
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep         = 0
        self.current_activity = 0
        self.deterioration_severity    = 0.0
        self.deterioration_start_time  = None
        self._spike_injected           = False  # allow spike to fire again next episode

        self._initialize_zone_baselines(
            self._custom_baseline_motion,
            self._custom_baseline_sound,
            self._custom_baseline_doors
        )
        self.current_vitals = self._generate_baseline_vitals()
        self.last_vitals    = self.current_vitals.copy()

    def get_state(self) -> Dict:
        """
        Get full simulator state for debugging/logging. Read-only.
        """
        return {
            "timestep":     self.timestep,
            "zone_type":    self.zone_type,
            "baselines": {
                "motion": self.baseline_motion,
                "sound":  self.baseline_sound,
                "doors":  self.baseline_doors,
                "panic":  self.baseline_panic,
                "occ":    self.baseline_occ,
                "smoke":  self.baseline_smoke,
            },
            "current_vitals":          self.get_vitals(),
            "current_activity":        self.current_activity,
            "deterioration_severity":  self.deterioration_severity,
        }


# ============================================================
# TESTING CODE - Quality Check
# ============================================================

if __name__ == "__main__":
    """
    Test all zone types and verify realistic behavior.
    """
    print("=" * 60)
    print("ZONE SIMULATOR TEST")
    print("=" * 60)

    # Test 1: Lobby normal zone
    print("\n[TEST 1] Lobby Normal Zone - 10 timesteps")
    print("-" * 60)
    zone = ZoneSimulator(zone_type="lobby_normal", seed=42)

    vitals = zone.get_vitals()
    print(f"Initial | Motion: {vitals['motion_level']:5.1f} | Sound: {vitals['sound_db']:5.1f} dB")

    vitals2 = zone.get_vitals()
    assert vitals == vitals2, "ERROR: get_vitals() is not idempotent!"
    print("✅ get_vitals() is idempotent.")

    context_names = ["Quiet Hours", "Meal Service", "Event", "Distress", "Emergency"]
    for i in range(1, 11):
        zone.tick()
        vitals   = zone.get_vitals()
        activity = zone.get_activity()
        print(f"Step {i:02d} | Context: {context_names[activity]:14s} | "
              f"Motion: {vitals['motion_level']:5.1f} | Sound: {vitals['sound_db']:5.1f} dB | "
              f"Panic: {vitals['panic_score']:.3f} | Doors: {vitals['door_events']:4.1f} | Smoke: {vitals['smoke_co_level']:.3f}")

    # Test 2: Event ballroom zone (Task 1) — verify spike fires
    print("\n[TEST 2] Event Ballroom Zone - Baseline high activity + guaranteed spike")
    print("-" * 60)
    zone = ZoneSimulator(zone_type="event_ballroom", seed=42)
    spike_seen = False
    for i in range(1, 61):
        zone.tick()
        vitals = zone.get_vitals()
        marker = ""
        if vitals["panic_score"] > 0.8:
            marker = "  ← EMERGENCY SPIKE"
            spike_seen = True
        if i <= 5 or vitals["panic_score"] > 0.8:
            print(f"Step {i:02d} | Motion: {vitals['motion_level']:5.1f} Sound: {vitals['sound_db']:5.1f} "
                  f"Panic: {vitals['panic_score']:.3f} Smoke: {vitals['smoke_co_level']:.3f}{marker}")
    assert spike_seen, "ERROR: Emergency spike never fired for event_ballroom zone!"
    print("✅ Emergency spike fired correctly.")

    # Test 3: Silent room zone (Task 2)
    print("\n[TEST 3] Silent Room Zone - Deterioration simulation")
    print("-" * 60)
    print("Watching smoke rise, panic increase, motion drop over time...")
    zone = ZoneSimulator(zone_type="silent_room", seed=42)
    for i in range(0, 361, 60):
        zone.timestep = i - 1 if i > 0 else 0
        zone.tick()
        vitals   = zone.get_vitals()
        state    = zone.get_state()
        severity = state["deterioration_severity"]
        print(f"Hour {i//60} | Severity: {severity:4.2f} | "
              f"Smoke: {vitals['smoke_co_level']:.3f} | Panic: {vitals['panic_score']:.3f} | "
              f"Motion: {vitals['motion_level']:5.1f} | Sound: {vitals['sound_db']:5.1f}")

    print("\n✅ All tests passed!")
    print("=" * 60)