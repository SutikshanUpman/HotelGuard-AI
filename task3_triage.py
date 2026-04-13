"""
Task 3: Triage Grader (Hard)
=============================
Grades multi-patient triage using NDCG@4 (Normalized Discounted Cumulative Gain)
combined with F1 detection accuracy, a temporal responsiveness score, and a
concentration penalty.

Score = 0.50 × NDCG@4      (priority ordering across patients)
      + 0.30 × F1           (ALERT-only detection on emergency patients)
      + 0.20 × Responsiveness (how quickly agent reacts after condition changes)
      - concentration_penalty (spamming same action to all patients)
      - hesitation_penalty    (VERIFYing when patient is in EMERGENCY is costly)

Calibration targets:
  - Rule-based agent (uniform VERIFY/IGNORE, no urgency differentiation) → 0.18–0.28
  - Mediocre LLM (mostly correct ordering but slow to ALERT)             → 0.40–0.55
  - Strong LLM agent (correct ordering, fast ALERT, differentiates well) → 0.65–0.80

Key design decisions vs prior version:
  - Responsiveness component added (20%): rule-based agents can't adapt to
    condition changes mid-episode; LLMs that read observations can.
  - Hesitation penalty added: VERIFY on an EMERGENCY patient is partial
    credit in the old grader (reward +0.3) but is explicitly penalised here.
    A good agent must *commit* to ALERT when the signs are unambiguous.
  - BORDERLINE urgency weight: 0.05 (was 0.1) — makes urgency ordering even
    harder to approximate with a static rule.
  - Concentration threshold tightened: 50% uniformity now triggers penalty
    (was 60%), and the slope is steeper.
  - NDCG weight reduced 0.60 → 0.50 to make room for Responsiveness.
  - F1 stays at 0.30 (reduced from 0.40) but hesitation_penalty applies
    separately so the effective F1 pressure is higher.
"""

import math
from reward_function import Action, PatientCondition


# ── Urgency weights (lower = harder to accidentally match true ordering) ──
URGENCY_WEIGHT = {
    PatientCondition.EMERGENCY:   1.0,
    PatientCondition.DRUG_MASKED: 1.0,
    PatientCondition.BORDERLINE:  0.05,   # was 0.3 then 0.1, now even lower
    PatientCondition.STABLE:      0.0,
}

# ── Action urgency mapping (for NDCG predicted-urgency) ──
ACTION_VALUE = {
    Action.ALERT:  2,
    Action.VERIFY: 1,
    Action.IGNORE: 0,
}


def _compute_ndcg(true_relevance: list, predicted_relevance: list) -> float:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain).

    Parameters
    ----------
    true_relevance : list[float]
        True urgency scores per patient (higher = more urgent).
    predicted_relevance : list[float]
        Predicted urgency scores per patient (derived from action history).

    Returns
    -------
    float
        NDCG score in [0.0, 1.0].
    """
    n = len(true_relevance)
    if n == 0:
        return 0.0

    predicted_ranking = sorted(range(n), key=lambda i: predicted_relevance[i], reverse=True)

    dcg = sum(
        true_relevance[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(predicted_ranking)
    )

    ideal_ranking = sorted(range(n), key=lambda i: true_relevance[i], reverse=True)
    idcg = sum(
        true_relevance[idx] / math.log2(rank + 2)
        for rank, idx in enumerate(ideal_ranking)
    )

    if idcg == 0.0:
        return 1.0  # all patients equally urgent → any ordering is correct

    return dcg / idcg


def _compute_responsiveness(stats_list: list) -> float:
    """
    Score how quickly the agent reacts after a patient's condition worsens.

    For every step where a patient transitions from STABLE/BORDERLINE into
    EMERGENCY or DRUG_MASKED, we measure how many subsequent steps until the
    agent first issues ALERT for that patient. Shorter lag = better score.

    A rule-based agent with a fixed policy has infinite lag (it never adapts),
    so it scores 0.0 here. An LLM reading the observation can react within
    1–3 steps.

    Parameters
    ----------
    stats_list : list[dict]
        Per-patient episode stats from RewardFunction.get_stats().

    Returns
    -------
    float
        Responsiveness score in [0.0, 1.0].
    """
    lag_scores = []
    max_lag = 10  # steps beyond this are treated as "never responded"

    for stats in stats_list:
        conditions = stats["condition_history"]
        actions = stats["action_history"]
        n = len(conditions)

        for t in range(1, n):
            prev = conditions[t - 1]
            curr = conditions[t]

            # Detect a worsening transition
            prev_ok = prev in (PatientCondition.STABLE, PatientCondition.BORDERLINE)
            curr_bad = curr in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED)

            if not (prev_ok and curr_bad):
                continue

            # Find first ALERT after this transition
            first_alert_lag = None
            for lag in range(0, min(max_lag, n - t)):
                if actions[t + lag] == Action.ALERT:
                    first_alert_lag = lag
                    break

            if first_alert_lag is None:
                lag_scores.append(0.0)  # never alerted → zero credit
            else:
                # lag=0 → 1.0, lag=1 → 0.85, lag=5 → 0.30, lag=9 → ~0.05
                lag_scores.append(math.exp(-0.35 * first_alert_lag))

    if not lag_scores:
        # No condition transitions in this episode → score based on
        # whether ALERTs correctly targeted persistent emergencies
        # (flat 0.5 — neither rewards nor heavily penalises)
        return 0.5

    return sum(lag_scores) / len(lag_scores)


def grade_triage(stats_list: list) -> float:
    """
    Grade the triage task.

    Parameters
    ----------
    stats_list : list[dict]
        List of per-patient episode stats from RewardFunction.get_stats().
        Expects at least 2 patients (meaningful triage requires comparison).

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    if not stats_list or len(stats_list) < 2:
        return 0.0

    total_steps = stats_list[0]["total_steps"] if stats_list[0]["total_steps"] > 0 else 1

    # ── Component 1: NDCG@N — priority ordering (50%) ────────────────────────
    # True urgency: weighted fraction of steps each patient was in bad condition
    true_urgency = []
    for stats in stats_list:
        weighted = sum(
            URGENCY_WEIGHT.get(c, 0.0)
            for c in stats["condition_history"]
        )
        true_urgency.append(weighted / total_steps)

    # Predicted urgency: mean action value (ALERT=2, VERIFY=1, IGNORE=0)
    predicted_urgency = []
    for stats in stats_list:
        vals = [ACTION_VALUE[a] for a in stats["action_history"]]
        mean_val = sum(vals) / len(vals) if vals else 0.0
        predicted_urgency.append(mean_val / 2.0)  # normalise to [0, 1]

    ndcg = _compute_ndcg(true_urgency, predicted_urgency)

    # ── Component 2: F1 Detection Accuracy (30%) ──────────────────────────────
    # Only ALERT counts as a positive detection.
    # VERIFY is explicitly neutral-to-bad (see hesitation_penalty below).
    tp, tn, fp, fn = 0, 0, 0, 0

    for stats in stats_list:
        for action, condition in zip(stats["action_history"], stats["condition_history"]):
            is_emergency = condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED)
            is_alert = (action == Action.ALERT)

            if is_emergency and is_alert:
                tp += 1
            elif not is_emergency and not is_alert:
                tn += 1
            elif not is_emergency and is_alert:
                fp += 1
            elif is_emergency and not is_alert:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # ── Component 3: Responsiveness — lag to ALERT after worsening (20%) ─────
    responsiveness = _compute_responsiveness(stats_list)

    # ── Penalty A: Concentration ───────────────────────────────────────────────
    # Agents that apply the same action to every patient every step are doing
    # zero triage (triage = differentiation). Threshold tightened to 50%.
    concentration_penalty = 0.0
    if total_steps > 0 and len(stats_list) >= 2:
        uniform_steps = 0
        for t in range(total_steps):
            step_actions = [
                stats["action_history"][t]
                for stats in stats_list
                if t < len(stats["action_history"])
            ]
            if len(set(a.value for a in step_actions)) == 1:
                uniform_steps += 1

        uniformity_rate = uniform_steps / total_steps
        # Tightened threshold: 50% (was 60%), steeper slope: 0.50 (was 0.375)
        if uniformity_rate > 0.50:
            concentration_penalty = min(0.20, (uniformity_rate - 0.50) * 0.50)

    # ── Penalty B: Hesitation ─────────────────────────────────────────────────
    # VERIFY on an EMERGENCY patient wastes critical time.
    # Each VERIFY-on-emergency step contributes a fractional penalty.
    # A rule-based VERIFY-everything agent typically runs VERIFY on emergencies
    # for the entire episode → maximum hesitation penalty.
    total_emergency_steps = 0
    verify_on_emergency = 0

    for stats in stats_list:
        for action, condition in zip(stats["action_history"], stats["condition_history"]):
            if condition in (PatientCondition.EMERGENCY, PatientCondition.DRUG_MASKED):
                total_emergency_steps += 1
                if action == Action.VERIFY:
                    verify_on_emergency += 1

    hesitation_penalty = 0.0
    if total_emergency_steps > 0:
        hesitation_rate = verify_on_emergency / total_emergency_steps
        # Only penalise beyond 20% hesitation (allow for early-episode uncertainty)
        if hesitation_rate > 0.20:
            hesitation_penalty = min(0.12, (hesitation_rate - 0.20) * 0.15)

    # ── Weighted sum ───────────────────────────────────────────────────────────
    score = (
        0.50 * ndcg
        + 0.30 * f1
        + 0.20 * responsiveness
        - concentration_penalty
        - hesitation_penalty
    )

    return max(0.0, min(1.0, score))