import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class VitalRanges:
    """
    Normal ranges for each vital sign.
    We'll use these as baselines and add noise/drift.
    """
    hr_range: Tuple[float, float] = (60, 100)       # Heart rate (bpm)
    sys_bp_range: Tuple[float, float] = (90, 140)   # Systolic BP (mmHg)
    dia_bp_range: Tuple[float, float] = (60, 90)    # Diastolic BP (mmHg)
    spo2_range: Tuple[float, float] = (95, 100)     # Oxygen saturation (%)
    rr_range: Tuple[float, float] = (12, 20)        # Respiratory rate (breaths/min)
    temp_range: Tuple[float, float] = (36.5, 37.5)  # Temperature (°C)


class PatientSimulator:
    """
    Simulates a single ICU patient's vital signs over time.

    A realistic patient data generator.
    No decision-making logic, just produces believable vital sign streams.
    """

    def __init__(
        self,
        patient_type: str = "healthy",
        seed: int = None,
        baseline_hr: float = None,
        baseline_sys_bp: float = None,
        baseline_dia_bp: float = None
    ):
        """
        Initialize the patient simulator.

        Args:
            patient_type: Type of patient simulation
                - "healthy":       Normal vitals with small noise
                - "hypertensive":  High baseline BP (chronic condition)
                - "deteriorating": Slow sepsis-like decline over hours
                - "post_op":       Low BP, recovering from surgery
                - "unstable":      Random spikes and drops
            seed: Random seed for reproducibility (important for testing!)
            baseline_hr, baseline_sys_bp, baseline_dia_bp:
                Override default baselines for personalization
        """
        self.rng = np.random.default_rng(seed)
        self.patient_type = patient_type
        self.timestep = 0

        self._custom_baseline_hr = baseline_hr
        self._custom_baseline_sys_bp = baseline_sys_bp
        self._custom_baseline_dia_bp = baseline_dia_bp

        self.activity_weights = [70, 10, 10, 5, 5]
        self.current_activity = 0

        self.deterioration_start_time = None
        self.deterioration_severity = 0.0

        # P0 fix: guaranteed emergency spike for hypertensive patient (Task 1).
        # Fires exactly once between steps 30-55 so grade_suppression() always
        # has at least one real emergency to score sensitivity against,
        # preventing a silent agent from trivially scoring 1.0.
        self._spike_injected = False

        self._initialize_baselines(baseline_hr, baseline_sys_bp, baseline_dia_bp)

        # Initial vitals are at resting state (activity 0)
        self.current_vitals = self._generate_baseline_vitals()
        self.last_vitals = self.current_vitals.copy()

    def _initialize_baselines(
        self,
        baseline_hr: float = None,
        baseline_sys_bp: float = None,
        baseline_dia_bp: float = None
    ):
        """
        Set patient's personal baseline vitals.
        """
        if self.patient_type == "healthy":
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 75.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 120.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 80.0
            self.baseline_spo2   = 98.0
            self.baseline_rr     = 16.0
            self.baseline_temp   = 37.0

        elif self.patient_type == "hypertensive":
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 78.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 150.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 95.0
            self.baseline_spo2   = 97.0
            self.baseline_rr     = 16.0
            self.baseline_temp   = 37.0

        elif self.patient_type == "deteriorating":
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 72.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 118.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 78.0
            self.baseline_spo2   = 98.0
            self.baseline_rr     = 15.0
            self.baseline_temp   = 37.1
            self.deterioration_start_time = 30

        elif self.patient_type == "post_op":
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 88.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 100.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 65.0
            self.baseline_spo2   = 96.0
            self.baseline_rr     = 18.0
            self.baseline_temp   = 36.8

        elif self.patient_type == "unstable":
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 80.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 125.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 82.0
            self.baseline_spo2   = 97.0
            self.baseline_rr     = 17.0
            self.baseline_temp   = 37.2

        else:
            self.baseline_hr     = baseline_hr     if baseline_hr     is not None else 75.0
            self.baseline_sys_bp = baseline_sys_bp if baseline_sys_bp is not None else 120.0
            self.baseline_dia_bp = baseline_dia_bp if baseline_dia_bp is not None else 80.0
            self.baseline_spo2   = 98.0
            self.baseline_rr     = 16.0
            self.baseline_temp   = 37.0

    def _generate_baseline_vitals(self) -> Dict[str, float]:
        """
        Generate vitals around the baseline with small random noise.
        """
        return {
            "heart_rate":       self.baseline_hr     + self.rng.normal(0, 3),
            "systolic_bp":      self.baseline_sys_bp + self.rng.normal(0, 5),
            "diastolic_bp":     self.baseline_dia_bp + self.rng.normal(0, 3),
            "spo2":             np.clip(self.baseline_spo2 + self.rng.normal(0, 1), 70, 100),
            "respiratory_rate": self.baseline_rr     + self.rng.normal(0, 2),
            "temperature":      self.baseline_temp   + self.rng.normal(0, 0.2)
        }

    def _apply_activity_effects(self, vitals: Dict[str, float], activity: int) -> Dict[str, float]:
        """
        Modify vitals based on what the patient is currently doing.
        """
        modified = vitals.copy()

        if activity == 0:    # Resting
            pass
        elif activity == 1:  # Eating
            modified["heart_rate"]       += self.rng.uniform(10, 15)
            modified["systolic_bp"]      += self.rng.uniform(5, 10)
            modified["respiratory_rate"] += self.rng.uniform(2, 3)
        elif activity == 2:  # Ambulating
            modified["heart_rate"]       += self.rng.uniform(20, 40)
            modified["systolic_bp"]      += self.rng.uniform(10, 20)
            modified["respiratory_rate"] += self.rng.uniform(5, 8)
            modified["spo2"]             -= self.rng.uniform(1, 2)
        elif activity == 3:  # Distressed
            modified["heart_rate"]       += self.rng.uniform(15, 30)
            modified["systolic_bp"]      += self.rng.uniform(15, 25)
            modified["diastolic_bp"]     += self.rng.uniform(10, 15)
            modified["respiratory_rate"] += self.rng.uniform(5, 10)
        elif activity == 4:  # Falling
            modified["heart_rate"]       += self.rng.uniform(30, 50)
            modified["systolic_bp"]      += self.rng.uniform(20, 35)
            modified["respiratory_rate"] += self.rng.uniform(8, 12)

        return modified

    def _apply_deterioration(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sepsis-like deterioration for "deteriorating" patient type,
        and random spikes for "unstable" patient type.

        Severity goes from 0.0 to 1.0 over 30 timesteps after
        deterioration_start_time. Each vital drifts proportionally:
            Temp  → slowly rises (fever)
            HR    → climbs (body working harder)
            BP    → drops (circulatory failure)
            SpO2  → falls (oxygen delivery failing)
            RR    → rises (breathing faster to compensate)
        """
        if self.patient_type == "unstable":
            modified = vitals.copy()
            if self.rng.random() < 0.10:
                spike_targets = ["heart_rate", "systolic_bp", "spo2", "respiratory_rate"]
                spike_target  = spike_targets[self.rng.integers(0, len(spike_targets))]
                if spike_target == "spo2":
                    modified["spo2"] -= self.rng.uniform(5, 15)
                else:
                    modified[spike_target] += self.rng.uniform(20, 40)
            return modified

        # Hypertensive patient: inject a genuine emergency spike at a fixed
        # window (timesteps 38–45) so the Task 1 grader always has at least
        # one real emergency to detect, preventing the trivial "always IGNORE"
        # exploit that scored ~1.0 on the no-emergency grader branch.
        if self.patient_type == "hypertensive":
            if 38 <= self.timestep <= 45:
                modified = vitals.copy()
                # Sudden hypertensive crisis: extreme HR + SpO2 crash
                modified["heart_rate"] += 65.0 + self.rng.normal(0, 3.0)
                modified["spo2"]       -= 14.0 + self.rng.normal(0, 1.5)
                modified["systolic_bp"] += 35.0 + self.rng.normal(0, 4.0)
                self.deterioration_severity = 0.75   # triggers EMERGENCY in classifier
            else:
                self.deterioration_severity = 0.0
                modified = vitals.copy()
            return modified

        if self.patient_type != "deteriorating":
            return vitals

        if self.deterioration_start_time is None or self.timestep < self.deterioration_start_time:
            return vitals

        time_since_start = self.timestep - self.deterioration_start_time
        self.deterioration_severity = min(time_since_start / 30.0, 1.0)

        modified    = vitals.copy()
        noise_scale = 1.0 + self.deterioration_severity * 2.0

        modified["temperature"]      += (self.deterioration_severity * 3.0
                                         + self.rng.normal(0, 0.1 * noise_scale))
        modified["heart_rate"]       += (self.deterioration_severity * 50.0
                                         + self.rng.normal(0, 2.0 * noise_scale))
        modified["systolic_bp"]      -= (self.deterioration_severity * 30.0
                                         + self.rng.normal(0, 3.0 * noise_scale))
        modified["diastolic_bp"]     -= (self.deterioration_severity * 20.0
                                         + self.rng.normal(0, 2.0 * noise_scale))
        modified["spo2"]             -= (self.deterioration_severity * 6.0
                                         + self.rng.normal(0, 0.5 * noise_scale))
        modified["respiratory_rate"] += (self.deterioration_severity * 10.0
                                         + self.rng.normal(0, 1.0 * noise_scale))

        return modified

    def _smooth_transition(self, new_vitals: Dict[str, float], smoothing: float = 0.3) -> Dict[str, float]:
        """
        Smooth transitions between timesteps.

        Blends 70% of the old value with 30% of the new target so vitals
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
        Randomly pick a new activity for this timestep.
        """
        activity = self.rng.choice(
            [0, 1, 2, 3, 4],
            p=np.array(self.activity_weights) / sum(self.activity_weights)
        )
        self.current_activity = activity
        return activity

    def get_activity(self) -> int:
        """
        Read the current activity state. Read-only.
        """
        return self.current_activity

    def get_vitals(self) -> Dict[str, float]:
        """
        Get current vital signs for this timestep. Read-only.
        """
        return self.current_vitals.copy()

    def tick(self):
        """
        Advance simulation time by 1 timestep and update vitals.

        Pipeline:
            1. Generate fresh baseline vitals with noise
            2. Apply activity effects (what's the patient doing?)
            3. Apply deterioration (are they getting worse?)
            4. Smooth the transition from last reading
            5. Clip everything to valid physiological ranges
            6. Inject guaranteed emergency spike (hypertensive only, once, steps 30-55)
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

        # 5. Clip to valid physiological ranges
        vitals["heart_rate"]       = np.clip(vitals["heart_rate"],       30,  200)
        vitals["systolic_bp"]      = np.clip(vitals["systolic_bp"],      60,  220)
        vitals["diastolic_bp"]     = np.clip(vitals["diastolic_bp"],     30,  140)
        vitals["spo2"]             = np.clip(vitals["spo2"],             70,  100)
        vitals["respiratory_rate"] = np.clip(vitals["respiratory_rate"],  5,   40)
        vitals["temperature"]      = np.clip(vitals["temperature"],      34,   42)

        # 6. Guaranteed emergency spike for hypertensive patient (Task 1 / P0 fix).
        #    Forces at least one unambiguous EMERGENCY between steps 30-55 so
        #    grade_suppression() always has real sensitivity to score — preventing
        #    a silent agent from trivially achieving 1.0 with no emergencies in
        #    the episode. Fires exactly once per episode (reset() clears the flag).
        #    Activity is forced to resting (0) so the spike is not activity-discounted
        #    and _classify_condition() returns EMERGENCY (not DRUG_MASKED).
        if (
            self.patient_type == "hypertensive"
            and not self._spike_injected
            and 30 <= self.timestep <= 55
        ):
            vitals["heart_rate"]   = 185.0
            vitals["spo2"]         = 76.0
            vitals["temperature"]  = 41.3
            vitals["systolic_bp"]  = 210.0
            self.current_activity  = 0   # force resting so condition → EMERGENCY
            self._spike_injected   = True

        # 7. Store for next iteration and for observation
        self.last_vitals    = vitals.copy()
        self.current_vitals = vitals.copy()

    def reset(self, patient_type: str = None, seed: int = None):
        """
        Reset simulator to initial state.
        """
        if patient_type is not None:
            self.patient_type = patient_type
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep         = 0
        self.current_activity = 0
        self.deterioration_severity    = 0.0
        self.deterioration_start_time  = None
        self._spike_injected           = False  # allow spike to fire again next episode

        self._initialize_baselines(
            self._custom_baseline_hr,
            self._custom_baseline_sys_bp,
            self._custom_baseline_dia_bp
        )
        self.current_vitals = self._generate_baseline_vitals()
        self.last_vitals    = self.current_vitals.copy()

    def get_state(self) -> Dict:
        """
        Get full simulator state for debugging/logging. Read-only.
        """
        return {
            "timestep":     self.timestep,
            "patient_type": self.patient_type,
            "baselines": {
                "hr":     self.baseline_hr,
                "sys_bp": self.baseline_sys_bp,
                "dia_bp": self.baseline_dia_bp,
                "spo2":   self.baseline_spo2,
                "rr":     self.baseline_rr,
                "temp":   self.baseline_temp
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
    Test all patient types and verify realistic behavior.
    """
    print("=" * 60)
    print("PATIENT SIMULATOR TEST")
    print("=" * 60)

    # Test 1: Healthy patient
    print("\n[TEST 1] Healthy Patient - 10 timesteps")
    print("-" * 60)
    patient = PatientSimulator(patient_type="healthy", seed=42)

    vitals = patient.get_vitals()
    print(f"Initial | HR: {vitals['heart_rate']:5.1f} | BP: {vitals['systolic_bp']:5.1f}/{vitals['diastolic_bp']:5.1f}")

    vitals2 = patient.get_vitals()
    assert vitals == vitals2, "ERROR: get_vitals() is not idempotent!"
    print("✅ get_vitals() is idempotent.")

    for i in range(1, 11):
        patient.tick()
        vitals   = patient.get_vitals()
        activity = patient.get_activity()
        activity_names = ["Resting", "Eating", "Walking", "Distressed", "Falling"]
        print(f"Step {i:02d} | Activity: {activity_names[activity]:10s} | "
              f"HR: {vitals['heart_rate']:5.1f} | BP: {vitals['systolic_bp']:5.1f}/{vitals['diastolic_bp']:5.1f} | "
              f"SpO2: {vitals['spo2']:5.1f} | RR: {vitals['respiratory_rate']:4.1f} | Temp: {vitals['temperature']:4.1f}")

    # Test 2: Hypertensive patient (Task 1) — verify spike fires
    print("\n[TEST 2] Hypertensive Patient - Baseline BP ~150/95 + guaranteed spike")
    print("-" * 60)
    patient = PatientSimulator(patient_type="hypertensive", seed=42)
    spike_seen = False
    for i in range(1, 61):
        patient.tick()
        vitals = patient.get_vitals()
        marker = ""
        if vitals["spo2"] < 80:
            marker = "  ← EMERGENCY SPIKE"
            spike_seen = True
        if i <= 5 or vitals["spo2"] < 80:
            print(f"Step {i:02d} | BP: {vitals['systolic_bp']:5.1f}/{vitals['diastolic_bp']:5.1f} "
                  f"HR: {vitals['heart_rate']:5.1f} SpO2: {vitals['spo2']:5.1f}{marker}")
    assert spike_seen, "ERROR: Emergency spike never fired for hypertensive patient!"
    print("✅ Emergency spike fired correctly.")

    # Test 3: Deteriorating patient (Task 2)
    print("\n[TEST 3] Deteriorating Patient - 6-hour sepsis simulation")
    print("-" * 60)
    print("Watching temp rise, HR increase, BP drop over time...")
    patient = PatientSimulator(patient_type="deteriorating", seed=42)
    for i in range(0, 361, 60):
        patient.timestep = i - 1 if i > 0 else 0
        patient.tick()
        vitals   = patient.get_vitals()
        state    = patient.get_state()
        severity = state["deterioration_severity"]
        print(f"Hour {i//60} | Severity: {severity:4.2f} | "
              f"Temp: {vitals['temperature']:4.1f}°C | HR: {vitals['heart_rate']:5.1f} | "
              f"BP: {vitals['systolic_bp']:5.1f}/{vitals['diastolic_bp']:5.1f} | SpO2: {vitals['spo2']:5.1f}")

    print("\n✅ All tests passed!")
    print("=" * 60)