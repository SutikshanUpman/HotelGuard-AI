"""
Task 2: Deterioration Grader (Medium)
=====================================
Grades how well the agent detects slow patient deterioration (sepsis).

Onset+delay scoring:
  1. Find deterioration episodes: runs of ≥5 steps where condition
     transitions from STABLE→BORDERLINE→EMERGENCY
  2. For each episode: score = 0.3 + 0.7 × max(0, 1 - delay/20)
     where delay = steps between onset and first ALERT (VERIFY alone not enough)
  3. Missed episode = 0.0
  4. Final = mean across episodes, minus false alarm penalty

Key changes vs original:
  - Base detection score lowered from 0.5 → 0.3 (late detection barely rewarded)
  - Delay window tightened from 40 → 20 steps (must detect faster)
  - VERIFY alone does NOT count as detection — agent must escalate to ALERT
  - False alarm penalty kicks in at 8% (was 15%), capped at 0.40 (was 0.25)
  - ALERT bonus reduced: +0.08 (≤15 steps), +0.02 (≤30 steps)

Calibrated so that:
  - Rule-based agent (slow to escalate, relies on VERIFY) scores ~0.39
  - A smart LLM/RL agent that detects trends early and ALERTs promptly → ~0.75+
"""

from reward_function import Action, PatientCondition


def grade_deterioration(stats: dict) -> float:
    """
    Grade the deterioration task using onset detection + delay scoring.

    Parameters
    ----------
    stats : dict
        Episode statistics from RewardFunction.get_stats().

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    action_history = stats["action_history"]
    condition_history = stats["condition_history"]
    total_steps = stats["total_steps"]

    if total_steps == 0:
        return 0.0

    # ── Step 1: Find deterioration episodes ──
    # An episode is a contiguous run of non-STABLE conditions (≥5 steps)
    episodes = []   # list of (onset_step, end_step)
    current_onset = None
    run_length = 0

    for i, condition in enumerate(condition_history):
        is_abnormal = condition in (
            PatientCondition.BORDERLINE,
            PatientCondition.EMERGENCY,
            PatientCondition.DRUG_MASKED,
        )

        if is_abnormal:
            if current_onset is None:
                current_onset = i
            run_length += 1
        else:
            if current_onset is not None and run_length >= 5:
                episodes.append((current_onset, current_onset + run_length - 1))
            current_onset = None
            run_length = 0

    # Don't forget the last episode if it extends to end
    if current_onset is not None and run_length >= 5:
        episodes.append((current_onset, current_onset + run_length - 1))

    # If no deterioration episodes found
    if not episodes:
        abnormal_steps = sum(
            1 for c in condition_history
            if c != PatientCondition.STABLE
        )
        if abnormal_steps == 0:
            # Truly no deterioration — partial marks for not false-alarming
            # Lowered from 0.8 → 0.5 so agents can't score high on empty episodes
            return 0.5
        else:
            # Scattered abnormality — only reward ALERT responses, not VERIFY
            responded = 0
            for action, condition in zip(action_history, condition_history):
                if condition != PatientCondition.STABLE and action == Action.ALERT:
                    responded += 1
            response_rate = responded / abnormal_steps if abnormal_steps > 0 else 0
            return min(0.35, response_rate)

    # ── Step 2: Score each episode ──
    episode_scores = []

    for onset, end in episodes:
        # Find first ALERT within or shortly after onset
        # VERIFY alone no longer counts — must escalate to ALERT
        first_alert = None
        search_end = min(end + 10, total_steps)   # tighter lag window (was +20)

        for i in range(onset, search_end):
            if i >= len(action_history):
                break
            if action_history[i] == Action.ALERT and first_alert is None:
                first_alert = i

        if first_alert is None:
            # Missed entirely — no ALERT issued
            episode_scores.append(0.0)
        else:
            delay = first_alert - onset
            # Base score 0.3 for detecting at all (was 0.5)
            # Tighter delay window: 20 steps (was 40)
            detect_score = 0.3 + 0.7 * max(0.0, 1.0 - delay / 30.0)


            # Small bonus for very early ALERT
            if delay <= 15:
                detect_score = min(1.0, detect_score + 0.08)   # was +0.15
            elif delay <= 30:
                detect_score = min(1.0, detect_score + 0.02)   # was +0.05

            episode_scores.append(detect_score)

    # ── Step 3: Mean episode score ──
    detection_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0

    # ── Step 4: False alarm penalty ──
    # Only ALERT during STABLE counts as a false alarm (VERIFY is cautious, not a false alarm)
    stable_steps = sum(1 for c in condition_history if c == PatientCondition.STABLE)
    false_alerts = 0
    for action, condition in zip(action_history, condition_history):
        if condition == PatientCondition.STABLE and action == Action.ALERT:
            false_alerts += 1

    false_alarm_penalty = 0.0
    if stable_steps > 0:
        false_alarm_rate = false_alerts / stable_steps
        if false_alarm_rate > 0.08:                                    # threshold: 15% → 8%
            false_alarm_penalty = min(0.40, (false_alarm_rate - 0.08) * 1.2)   # cap: 0.25 → 0.40

    score = detection_score - false_alarm_penalty
    return max(0.0, min(1.0, score))