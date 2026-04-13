"""
Reward Function for HotelGuard-AI
==================================
Maps (action × zone_condition × activity) to a reward signal with:
  - Base reward table
  - Activity context multipliers (the novel mechanic)
  - Alarm fatigue modifier
  - Personalization bonus
"""

from enum import Enum


class Action(Enum):
    MONITOR   = 0
    DISPATCH  = 1
    EMERGENCY = 2


class ZoneCondition(Enum):
    STABLE    = "stable"
    BORDERLINE = "borderline"
    EMERGENCY  = "emergency"
    ESCALATED  = "escalated"


# Base reward table: REWARD_TABLE[action][condition]
REWARD_TABLE = {
    Action.EMERGENCY: {
        ZoneCondition.EMERGENCY:  +1.0,
        ZoneCondition.BORDERLINE: +0.2,
        ZoneCondition.STABLE:     -0.5,
        ZoneCondition.ESCALATED:  +1.0,
    },
    Action.DISPATCH: {
        ZoneCondition.EMERGENCY:  +0.3,
        ZoneCondition.BORDERLINE: +0.7,
        ZoneCondition.STABLE:     -0.1,
        ZoneCondition.ESCALATED:  +0.3,
    },
    Action.MONITOR: {
        ZoneCondition.EMERGENCY:  -1.0,
        ZoneCondition.BORDERLINE: -0.2,
        ZoneCondition.STABLE:     +0.2,
        ZoneCondition.ESCALATED:  -1.0,
    },
}

# Activity context multipliers — the key insight:
# Same sensor reading means different things depending on venue context.
#   High sound_db during event     → expected (low multiplier, discount the anomaly)
#   High sound_db during quiet hrs → emergency (high multiplier, amplify the anomaly)
ACTIVITY_CONTEXT = {
    0: 1.00,  # quiet hours (2am–6am) — baseline, no discount
    1: 0.40,  # meal service — slight motion/sound increase expected
    2: 0.50,  # event in progress — elevated signals expected
    3: 1.25,  # guest distress signal — amplify concern
    4: 1.60,  # physical emergency confirmed — immediate concern
}

# Alarm fatigue settings
FATIGUE_WINDOW      = 30   # look at last 30 steps
FATIGUE_THRESHOLD   = 5    # more than 5 emergency calls in the window
FATIGUE_MULTIPLIER  = 0.6  # reduce reward to 60%

# Personalization settings
# P0 fix: was 200, which is unreachable in a 60-step episode.
# Now 20 so the bonus fires from step 21 onward — matching the README description.
PERSONALIZATION_STEP  = 20   # kicks in after this many steps
PERSONALIZATION_BONUS = 0.2  # bonus for correctly monitoring known-normal zone


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

    def compute(self, action: Action, condition: ZoneCondition,
                activity: int = 0) -> float:
        """
        Compute the reward for a single (action, condition, activity) tuple.

        Parameters
        ----------
        action : Action
            The agent's action (MONITOR, DISPATCH, EMERGENCY).
        condition : ZoneCondition
            The zone's current condition.
        activity : int
            The zone's current activity/context code (0-4).

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
            # Penalties are reduced during expected-high-signal activities
            # e.g., dispatching during an event gets less penalty (ctx=0.5)
            base_reward *= ctx
        elif condition in (ZoneCondition.EMERGENCY, ZoneCondition.ESCALATED):
            # Correct emergency responses amplified during dangerous contexts
            base_reward *= ctx

        # 3. Alarm fatigue modifier
        fatigue_modifier = 1.0
        if len(self.action_history) >= FATIGUE_WINDOW:
            recent_emergencies = sum(
                1 for a in self.action_history[-FATIGUE_WINDOW:]
                if a == Action.EMERGENCY
            )
            if recent_emergencies > FATIGUE_THRESHOLD:
                fatigue_modifier = FATIGUE_MULTIPLIER

        # 4. Personalization bonus
        personalization = 0.0
        if (self.step_count > PERSONALIZATION_STEP
                and action == Action.MONITOR
                and condition == ZoneCondition.STABLE):
            personalization = PERSONALIZATION_BONUS

        # Final reward
        reward = base_reward * fatigue_modifier + personalization
        return reward

    def get_stats(self) -> dict:
        """Return episode statistics for graders."""
        total_emergencies = sum(1 for a in self.action_history if a == Action.EMERGENCY)
        total_dispatches  = sum(1 for a in self.action_history if a == Action.DISPATCH)
        total_monitors    = sum(1 for a in self.action_history if a == Action.MONITOR)
        return {
            "total_steps":     self.step_count,
            "total_alerts":    total_emergencies,
            "total_verifies":  total_dispatches,
            "total_ignores":   total_monitors,
            "action_history":    list(self.action_history),
            "condition_history": list(self.condition_history),
            "activity_history":  list(self.activity_history),
        }
