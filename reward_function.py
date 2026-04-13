"""
Reward Function for MediGuard-AI
================================
Maps (action × patient_condition × activity) to a reward signal with:
  - Base reward table
  - Activity context multipliers (the novel mechanic)
  - Alarm fatigue modifier
  - Personalization bonus
"""

from enum import Enum


class Action(Enum):
    IGNORE = 0
    VERIFY = 1
    ALERT  = 2


class PatientCondition(Enum):
    STABLE      = "stable"
    BORDERLINE  = "borderline"
    EMERGENCY   = "emergency"
    DRUG_MASKED = "drug_masked"


# Base reward table: REWARD_TABLE[action][condition]
REWARD_TABLE = {
    Action.ALERT: {
        PatientCondition.EMERGENCY:   +1.0,
        PatientCondition.BORDERLINE:  +0.2,
        PatientCondition.STABLE:      -0.5,
        PatientCondition.DRUG_MASKED: +1.0,
    },
    Action.VERIFY: {
        PatientCondition.EMERGENCY:   +0.3,
        PatientCondition.BORDERLINE:  +0.7,
        PatientCondition.STABLE:      -0.1,
        PatientCondition.DRUG_MASKED: +0.3,
    },
    Action.IGNORE: {
        PatientCondition.EMERGENCY:   -1.0,
        PatientCondition.BORDERLINE:  -0.2,
        PatientCondition.STABLE:      +0.2,
        PatientCondition.DRUG_MASKED: -1.0,
    },
}

# Activity context multipliers — the key insight:
# Same vital sign means different things depending on what the patient is doing.
#   HR 130 while walking    → expected (low multiplier, discount the anomaly)
#   HR 130 while lying still → emergency (high multiplier, amplify the anomaly)
ACTIVITY_CONTEXT = {
    0: 1.00,  # resting (lying in bed) — baseline, no discount
    1: 0.40,  # eating — slight HR/BP increase expected
    2: 0.50,  # walking/ambulating — elevated vitals expected
    3: 1.25,  # distressed — amplify concern
    4: 1.60,  # falling — immediate concern
}

# Alarm fatigue settings
FATIGUE_WINDOW      = 30   # look at last 30 steps
FATIGUE_THRESHOLD   = 5    # more than 5 alerts in the window
FATIGUE_MULTIPLIER  = 0.6  # reduce reward to 60%

# Personalization settings
# P0 fix: was 200, which is unreachable in a 60-step episode.
# Now 20 so the bonus fires from step 21 onward — matching the README description.
PERSONALIZATION_STEP  = 20   # kicks in after this many steps
PERSONALIZATION_BONUS = 0.2  # bonus for correctly ignoring known-normal


class RewardFunction:
    """
    Stateful reward calculator.
    Tracks action history for alarm fatigue and personalization bonuses.
    """

    def __init__(self):
        self.action_history    = []
        self.condition_history = []
        self.activity_history  = []
        self.step_count        = 0

    def reset(self):
        """Clear all history for a new episode."""
        self.action_history    = []
        self.condition_history = []
        self.activity_history  = []
        self.step_count        = 0

    def compute(self, action: Action, condition: PatientCondition,
                activity: int = 0) -> float:
        """
        Compute the reward for a single (action, condition, activity) tuple.

        Parameters
        ----------
        action : Action
            The agent's action (IGNORE, VERIFY, ALERT).
        condition : PatientCondition
            The patient's current condition.
        activity : int
            The patient's current activity code (0-4).

        Returns
        -------
        float
            The reward value (can be negative).
        """
        self.step_count += 1
        self.action_history.append(action)
        self.condition_history.append(condition)
        self.activity_history.append(activity)

        # 1. Base reward from the table
        base_reward = REWARD_TABLE[action][condition]

        # 2. Activity context multiplier
        #    Only apply to penalty situations — don't discount correct actions
        ctx = ACTIVITY_CONTEXT.get(activity, 1.0)
        if base_reward < 0:
            # Penalties are reduced during expected-high-vitals activities
            # e.g., alerting during walking gets less penalty (ctx=0.5)
            base_reward *= ctx
        elif condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED):
            # Correct emergency responses amplified during dangerous activities
            base_reward *= ctx

        # 3. Alarm fatigue modifier
        fatigue_modifier = 1.0
        if len(self.action_history) >= FATIGUE_WINDOW:
            recent_alerts = sum(
                1 for a in self.action_history[-FATIGUE_WINDOW:]
                if a == Action.ALERT
            )
            if recent_alerts > FATIGUE_THRESHOLD:
                fatigue_modifier = FATIGUE_MULTIPLIER

        # 4. Personalization bonus
        personalization = 0.0
        if (self.step_count > PERSONALIZATION_STEP
                and action == Action.IGNORE
                and condition == PatientCondition.STABLE):
            personalization = PERSONALIZATION_BONUS

        # Final reward
        reward = base_reward * fatigue_modifier + personalization
        return reward

    def get_stats(self) -> dict:
        """Return episode statistics for graders."""
        total_alerts  = sum(1 for a in self.action_history if a == Action.ALERT)
        total_verifies = sum(1 for a in self.action_history if a == Action.VERIFY)
        total_ignores  = sum(1 for a in self.action_history if a == Action.IGNORE)
        return {
            "total_steps":     self.step_count,
            "total_alerts":    total_alerts,
            "total_verifies":  total_verifies,
            "total_ignores":   total_ignores,
            "action_history":    list(self.action_history),
            "condition_history": list(self.condition_history),
            "activity_history":  list(self.activity_history),
        }
