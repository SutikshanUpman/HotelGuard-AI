"""
HotelGuard-AI — LLM-Based Inference Script

Runs an LLM agent (via Google Gemini) against all 3 tasks.
Falls back to a rule-based baseline agent on API errors or missing key.

Runtime guarantee: 60 steps × 3 tasks × 4s timeout = 720s = 12 min worst case.
Well within the 30-minute eval time limit.

Environment variables:
  GEMINI_API_KEY — Google Gemini authentication token
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Union

import httpx
from google import genai
from google.genai import types
from hotelguard_env import HotelGuardEnv

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)

client = _get_client()

# Always attempt LLM if any key is present.
HAS_API_KEY = bool(GEMINI_API_KEY)

MODEL_BY_TASK = {
    "suppression":   "gemini-2.0-flash",
    "deterioration": "gemini-2.0-flash",
    "triage":        "gemini-2.0-flash",
}

MONITOR   = 0
DISPATCH  = 1
EMERGENCY = 2

CONTEXT_NAMES = {
    0: "quiet hours",
    1: "meal service",
    2: "event in progress",
    3: "guest distress signal",
    4: "physical emergency",
}

# ------------------------------------------------------------------ #
#  System Prompts                                                     #
# ------------------------------------------------------------------ #

SUPPRESSION_PROMPT = """You are an AI safety agent monitoring a hotel ballroom during an event.
Sensor signals: motion_level, sound_db, door_events, panic_score, occupancy_delta, smoke_co_level. All normalized 0–1 against this zone's rolling baseline. Context 2 = event in progress (noise/motion expected — suppress normal event activity). Context 0 = quiet hours (full weight on anomalies). A genuine emergency is a sudden panic_score spike combined with motion drop and sound drop (crowd reacting).
Rules:
- MONITOR (0): zone reads normal for its current context
- DISPATCH (1): something is off — send staff to investigate
- EMERGENCY (2): genuine crisis — call emergency services NOW
Respond ONLY with valid JSON: {"action": <0|1|2>, "reason": "<brief>"}"""

DETERIORATION_PROMPT = """You are an AI safety agent monitoring a hotel guest room.
Watch for slow deterioration: motion dropping, door_events stopping, smoke_co_level climbing, panic_score rising. These signals together over multiple steps indicate a medical emergency or early fire buildup.
Single readings within range are not alarming — you must detect the TREND using signal_history (last 10 timesteps provided).
Rules:
- MONITOR (0): readings stable, trend normal
- DISPATCH (1): trend is concerning — send staff to check
- EMERGENCY (2): trend indicates imminent crisis — escalate immediately
Respond ONLY with valid JSON: {"action": <0|1|2>, "reason": "<brief>"}"""

TRIAGE_PROMPT = """You are an AI safety agent managing 4 hotel zones simultaneously.
You receive observations for all 4 zones and must rank them by urgency and allocate responses. This is a RANKING problem — do not apply the same action to every zone. Compare zones against each other.
Urgency signals: high panic_score, rising smoke_co_level, motion near zero.
Rules:
- MONITOR (0): zone is stable — no action needed
- DISPATCH (1): zone is borderline — send staff
- EMERGENCY (2): zone is in crisis — call emergency services
Respond ONLY with valid JSON: {"actions": [<0|1|2>, <0|1|2>, <0|1|2>, <0|1|2>], "reason": "<brief>"}"""

SYSTEM_PROMPTS = {
    "suppression":   SUPPRESSION_PROMPT,
    "deterioration": DETERIORATION_PROMPT,
    "triage":        TRIAGE_PROMPT,
}

# ------------------------------------------------------------------ #
#  Observation formatting                                             #
# ------------------------------------------------------------------ #

def obs_to_user_message(obs: Dict, task: str, signal_history: list,
                        conversation_history: list) -> str:
    context_name = CONTEXT_NAMES.get(obs.get("activity", 0), "unknown")
    hours = obs.get("hours_observed", 0.0)

    lines = [
        f"Step: {int(hours * 60)}/{60}  |  Hours observed: {hours:.1f}h",
        "",
        "CURRENT SIGNALS (normalized 0-1):",
        f"  Motion Level:    {obs.get('motion_level', 0):.3f}",
        f"  Sound dB:        {obs.get('sound_db', 0):.3f}",
        f"  Door Events:     {obs.get('door_events', 0):.3f}",
        f"  Panic Score:     {obs.get('panic_score', 0):.3f}",
        f"  Occupancy Δ:     {obs.get('occupancy_delta', 0):.3f}",
        f"  Smoke/CO:        {obs.get('smoke_co_level', 0):.3f}",
        "",
        f"Baseline Delta:  {obs.get('baseline_delta', 0):.3f} (deviation from zone norm)",
        f"Context:         {context_name}",
    ]

    if task == "deterioration" and len(signal_history) >= 3:
        lines.append("")
        lines.append("SIGNAL TREND (last readings, oldest→newest):")
        lines.append("  Step   Motion Sound  Panic  Smoke")
        n = len(signal_history)
        for i, reading in enumerate(signal_history):
            offset = -(n - i)
            if len(reading) >= 6:
                lines.append(
                    f"  {offset:+4d}   {reading[0]:.3f}  {reading[1]:.3f}  "
                    f"{reading[3]:.3f}  {reading[5]:.3f}"
                )

    if conversation_history:
        lines.append("")
        lines.append("YOUR RECENT DECISIONS:")
        for entry in conversation_history[-3:]:
            lines.append(f"  Action={entry['action']} → reward={entry['reward']:.2f}")

    return "\n".join(lines)


def triage_obs_to_message(obs_list: List[Dict], conversation_history: list) -> str:
    lines = []
    for i, obs in enumerate(obs_list):
        context_name = CONTEXT_NAMES.get(obs.get("activity", 0), "unknown")
        lines.append(f"ZONE {i} :")
        lines.append(
            f"  Signals:  Motion={obs.get('motion_level', 0):.3f}  "
            f"Panic={obs.get('panic_score', 0):.3f}  "
            f"Sound={obs.get('sound_db', 0):.3f}  "
            f"Smoke={obs.get('smoke_co_level', 0):.3f}"
        )
        lines.append(
            f"  Context: {context_name}  |  "
            f"Baseline Delta: {obs.get('baseline_delta', 0):.3f}  |  "
            f"Hours: {obs.get('hours_observed', 0):.1f}h"
        )
        lines.append("")
    lines.append("Rank these zones by urgency. Assign actions:")
    lines.append("  Most critical → EMERGENCY(2). Second → DISPATCH(1). Stable → MONITOR(0).")
    if conversation_history:
        lines.append("")
        lines.append("YOUR RECENT DECISIONS:")
        for entry in conversation_history[-2:]:
            lines.append(f"  Actions={entry['action']} → reward={entry['reward']:.2f}")
    return "\n".join(lines)

# ------------------------------------------------------------------ #
#  LLM agents                                                         #
# ------------------------------------------------------------------ #

def llm_agent(obs: Dict, task: str, conversation_history: list,
              signal_history: list, model_name: str) -> Tuple[int, str]:
    system_prompt = SYSTEM_PROMPTS[task]
    user_message = obs_to_user_message(obs, task, signal_history, conversation_history)

    c = client or _get_client()
    if not c:
        raise ValueError("no_api_key")

    # Build conversation for Gemini
    full_prompt = system_prompt + "\n\n" + user_message
    for entry in conversation_history[-2:]:
        full_prompt += f"\n\nPrevious observation:\n{entry['obs_text'][:500]}"
        full_prompt += f"\nYour response: {entry['response']}"

    response = c.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=120,
        ),
    )
    raw_response = response.text.strip()
    return _parse_single_response(raw_response)


def _parse_single_response(raw: str) -> Tuple[int, str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()
    data = json.loads(text)
    action = int(data.get("action", 1))
    reasoning = str(data.get("reason", data.get("reasoning", "")))[:80]
    if action not in (0, 1, 2):
        action = max(0, min(2, action))
    return action, reasoning


def triage_llm_agent(obs_list: List[Dict], conversation_history: list,
                     model_name: str) -> Tuple[List[int], str]:
    system_prompt = SYSTEM_PROMPTS["triage"]
    user_message = triage_obs_to_message(obs_list, conversation_history)

    c = client or _get_client()
    if not c:
        raise ValueError("no_api_key")

    full_prompt = system_prompt + "\n\n" + user_message
    for entry in conversation_history[-2:]:
        full_prompt += f"\n\nPrevious observation:\n{entry['obs_text'][:500]}"
        full_prompt += f"\nYour response: {entry['response']}"

    response = c.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=120,
        ),
    )
    raw_response = response.text.strip()
    return _parse_triage_response(raw_response)


def _parse_triage_response(raw: str) -> Tuple[List[int], str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()
    data = json.loads(text)
    actions = list(data.get("actions", [0, 0, 0, 0]))
    reasoning = str(data.get("reason", data.get("reasoning", "")))[:100]
    while len(actions) < 4:
        actions.append(0)
    actions = actions[:4]
    actions = [max(0, min(2, int(a))) for a in actions]
    return actions, reasoning

# ------------------------------------------------------------------ #
#  Rule-based fallback                                                #
# ------------------------------------------------------------------ #

def baseline_agent(obs: Dict) -> int:
    activity = obs["activity"]
    delta    = obs["baseline_delta"]
    hours    = obs["hours_observed"]
    panic    = obs["panic_score"]
    motion   = obs["motion_level"]
    smoke    = obs.get("smoke_co_level", 0.0)
    history  = obs.get("signal_history", [])

    if hours < 1.0:
        return DISPATCH
    if activity == 2:
        return EMERGENCY if panic > 0.75 else MONITOR
    if delta > 0.6 and activity == 0:
        return EMERGENCY
    if delta > 0.35 and activity == 0:
        return DISPATCH
    if panic > 0.65:
        return EMERGENCY
    if panic > 0.50:
        return DISPATCH
    if smoke > 0.60:
        return EMERGENCY
    if smoke > 0.40:
        return DISPATCH
    if motion < 0.10:
        return EMERGENCY
    if motion < 0.20:
        return DISPATCH

    if len(history) >= 6:
        try:
            recent = history[-3:]
            oldest = history[:3]
            recent_panic = sum(r[3] for r in recent) / 3
            oldest_panic = sum(r[3] for r in oldest) / 3
            panic_rise = recent_panic - oldest_panic
            recent_smoke = sum(r[5] for r in recent) / 3
            oldest_smoke = sum(r[5] for r in oldest) / 3
            smoke_rise = recent_smoke - oldest_smoke
            if panic_rise > 0.08: return EMERGENCY
            if panic_rise > 0.04: return DISPATCH
            if smoke_rise > 0.08: return EMERGENCY
            if smoke_rise > 0.04: return DISPATCH
        except (IndexError, TypeError):
            pass

    if hours > 4.0:
        if delta > 0.20 and activity == 0:
            return DISPATCH
        if panic > 0.45:
            return DISPATCH
    return MONITOR


def triage_baseline(obs_list: List[Dict]) -> List[int]:
    return [baseline_agent(obs) for obs in obs_list]

# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #

def log_start(task: str, model_name: str):
    print(f"[START] task={task} env=hotelguard model={model_name}", flush=True)

def log_agent(model_name: str):
    print(f"[AGENT] type=llm model={model_name} temperature=0.0", flush=True)

def log_step(step: int, action, reward: float, done: bool, error=None):
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)
    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_fallback(step: int, reason: str):
    print(f"[FALLBACK] step={step} reason={reason} using=rule_based", flush=True)

def log_end(success: bool, steps: int, rewards: List[float], score: float):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

def log_reasoning(action, reasoning: str):
    if isinstance(action, (list, tuple)):
        action_str = ",".join(str(a) for a in action)
    else:
        action_str = str(action)
    print(f"[REASONING] last_action={action_str} reasoning=\"{reasoning}\"", flush=True)

# ------------------------------------------------------------------ #
#  Episode runner                                                     #
# ------------------------------------------------------------------ #
def run_episode(task: str, seed: int = 42) -> Tuple[List[float], float]:
    model_name = MODEL_BY_TASK.get(task, "gemini-2.0-flash")
    log_start(task, model_name)
    log_agent(model_name)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    last_action: Union[int, List[int]] = 0
    last_reasoning = ""
    fallback_count = 0
    conversation_history: List[Dict] = []
    signal_history: List[list] = []

    try:
        env = HotelGuardEnv(task=task, seed=seed)
        obs = env.reset()
        done = False

        while not done:
            action = None
            reasoning = ""
            used_fallback = False

            if not HAS_API_KEY:
                used_fallback = True
                reasoning = "no_api_key"
            else:
                try:
                    if task == "triage":
                        action, reasoning = triage_llm_agent(obs, conversation_history, model_name)
                    else:
                        action, reasoning = llm_agent(obs, task, conversation_history, signal_history, model_name)
                except json.JSONDecodeError:
                    used_fallback = True
                    reasoning = "parse_error"
                except Exception as e:
                    used_fallback = True
                    reasoning = f"error:{type(e).__name__}"

            if used_fallback or action is None:
                if task == "triage":
                    action = triage_baseline(obs)
                else:
                    action = baseline_agent(obs)
                fallback_count += 1
                log_fallback(steps + 1, reasoning)

            obs, reward, done, info = env.step(action)
            steps = info["step"]
            rewards.append(reward)
            last_action = action
            last_reasoning = reasoning

            # ── LOGGING FOR EVERY STEP ──
            log_step(steps, action, reward, done, error=None)
            log_reasoning(action, reasoning)

            if task == "triage":
                obs_text = triage_obs_to_message(obs, conversation_history)
            else:
                obs_text = obs_to_user_message(obs, task, signal_history, conversation_history)
                history = obs.get("signal_history", [])
                if history:
                    for entry in reversed(history):
                        if any(v != 0.0 for v in entry):
                            signal_history.append(entry)
                            break
                    signal_history = signal_history[-8:]

            response_text = json.dumps({"action": action, "reasoning": reasoning})
            conversation_history.append({
                "obs_text": obs_text,
                "response": response_text + f" → reward: {reward:.2f}",
                "action": action,
                "reward": reward,
            })
            if len(conversation_history) > 4:
                conversation_history = conversation_history[-4:]

        grader_map = {
            "suppression": env.suppression_grader,
            "deterioration": env.deterioration_grader,
            "triage": env.triage_grader,
        }
        score = grader_map[task]()
        success = score > 0.50

    except Exception as exc:
        steps += 1
        log_step(steps, 0, 0.0, True, error=str(exc))
        success = False
        score = 0.0

    # ── FINAL SUMMARY ──
    log_end(success, steps, rewards, score)

    if fallback_count > 0:
        print(
            f"[INFO] fallback_count={fallback_count}/{steps} "
            f"({100*fallback_count/max(steps,1):.1f}% of steps used rule-based)",
            flush=True,
        )

    return rewards, score

# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main():
    key_source = "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "none"
    print(f"[CONFIG] api_key_source={key_source} model=gemini-2.0-flash", flush=True)
    print(f"[LLM_CHECK] has_key={HAS_API_KEY} client_ready=true", flush=True)

    if not HAS_API_KEY:
        print("[INFO] No API key — running fully rule-based (fast mode)", flush=True)
    else:
        print(f"[INFO] LLM mode: gemini-2.0-flash | 60 steps/task | 4s timeout", flush=True)
        print(f"[INFO] Worst-case runtime: 60 x 3 x 4s = 720s = 12 min", flush=True)

    tasks = ["suppression", "deterioration", "triage"]
    all_results = {}

    for task in tasks:
        rewards, score = run_episode(task, seed=42)
        all_results[task] = {"rewards": rewards, "score": score}

    print(flush=True)
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task in tasks:
        rews = all_results[task]["rewards"]
        score = all_results[task]["score"]
        mean_r = sum(rews) / len(rews) if rews else 0.0
        print(f"  {task:15s}  steps={len(rews):4d}  mean_reward={mean_r:.4f}  score={score:.4f}", flush=True)
    print("=" * 60, flush=True)

    s = all_results["suppression"]["score"]
    d = all_results["deterioration"]["score"]
    t = all_results["triage"]["score"]

    print(f"\n[SUMMARY] suppression={s:.4f} deterioration={d:.4f} triage={t:.4f}", flush=True)


if __name__ == "__main__":
    main()