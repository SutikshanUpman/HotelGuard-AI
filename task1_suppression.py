"""
Task 1: Suppression Grader (Easy)
=================================
Grades how well the agent suppresses false alarms for an event ballroom zone
while still catching real emergencies.

Uses F1 harmonic mean of sensitivity and specificity:
  - sensitivity = correctly responded to emergencies / total emergencies
    (EMERGENCY counts as full TP, DISPATCH counts as 0.5 TP — must commit to EMERGENCY)
  - specificity = correctly monitored stable periods / total stable periods
    (DISPATCH counts as 0.7 FP — spamming DISPATCH is penalized more heavily)
  - score = 2 × sensitivity × specificity / (sensitivity + specificity)

When no emergencies occur (possible at 60 steps), scores purely on
false alarm suppression (specificity) so the grader never returns 0
for a well-behaved agent.
"""

from reward_function import Action, ZoneCondition


def grade_suppression(stats: dict) -> float:
    action_history = stats["action_history"]
    condition_history = stats["condition_history"]
    total_steps = stats["total_steps"]

    if total_steps == 0:
        return 0.0

    true_positives  = 0.0
    false_negatives = 0.0
    true_negatives  = 0.0
    false_positives = 0.0

    for action, condition in zip(action_history, condition_history):
        is_emergency = condition in (ZoneCondition.EMERGENCY, ZoneCondition.ESCALATED)

        if is_emergency:
            if action == Action.EMERGENCY:
                true_positives += 1.0
            elif action == Action.DISPATCH:
                true_positives += 0.5
                false_negatives += 0.5
            else:
                false_negatives += 1.0
        else:
            if action == Action.MONITOR:
                if condition == ZoneCondition.BORDERLINE:
                    true_negatives += 0.5
                else:
                    true_negatives += 1.0
            elif action == Action.EMERGENCY:
                false_positives += 1.0
            else:  # DISPATCH
                if condition == ZoneCondition.BORDERLINE:
                    false_positives += 0.1
                else:
                    false_positives += 0.7

    total_emergencies = true_positives + false_negatives
    total_stable = true_negatives + false_positives

    if total_emergencies > 0:
        sensitivity = true_positives / total_emergencies
    else:
        # No emergencies in this episode (can happen at 60 steps).
        # Score purely on false alarm suppression — the core task goal.
        if total_stable > 0:
            return max(0.0, min(1.0, true_negatives / total_stable))
        return 0.5

    if total_stable > 0:
        specificity = true_negatives / total_stable
    else:
        specificity = 1.0

    if (sensitivity + specificity) > 0:
        f1 = 2.0 * sensitivity * specificity / (sensitivity + specificity)
    else:
        f1 = 0.0

    return max(0.0, min(1.0, f1))