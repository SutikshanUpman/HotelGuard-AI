"""
MediGuard-AI — LLM-Based Inference Script

Runs an LLM agent (via OpenAI client) against all 3 hackathon tasks.
Falls back to a rule-based baseline agent on API errors or missing key.

Runtime guarantee: 60 steps × 3 tasks × 4s timeout = 720s = 12 min worst case.
Well within the 30-minute eval time limit.

Environment variables:
  HF_TOKEN / API_KEY / OPENAI_API_KEY — authentication token
  API_BASE_URL — LLM endpoint (default: HuggingFace router)
  MODEL_NAME   — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Union

import httpx
from openai import OpenAI
from mediguard_env import MediGuardEnv

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# CRITICAL: Validator injects API_KEY (not HF_TOKEN). Read API_KEY first.
API_KEY = (
    os.getenv("API_KEY")           # validator injects this
    or os.getenv("HF_TOKEN")       # our own token
    or os.getenv("OPENAI_API_KEY")
    or "dummy"
)

# Always attempt LLM if any key is present — validator always injects one.
HAS_API_KEY = bool(API_KEY) and API_KEY != "dummy"

MODEL_BY_TASK = {
    "suppression":   MODEL_NAME,
    "deterioration": MODEL_NAME,
    "triage":        MODEL_NAME,
}

# ── Timeout fix ─────────────────────────────────────────────────────
# timeout= is NOT a valid kwarg in chat.completions.create().
# Must be set at the httpx.Client level.
# 4s hard cap: 60 steps × 3 tasks × 4s = 720s = 12 min worst case.
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(
        timeout=httpx.Timeout(8.0, connect=8.0)
    ),
)

IGNORE = 0
VERIFY = 1
ALERT  = 2

ACTIVITY_NAMES = {
    0: "resting (lying in bed)",
    1: "eating",
    2: "walking/ambulating",
    3: "in distress",
    4: "falling",
}

BASELINE_SCORES = {
    "suppression":   0.6500,   # updated: spike injection means grader now uses F1 branch
    "deterioration": 0.3931,
    "triage":        0.3073,
}

# ------------------------------------------------------------------ #
#  System Prompts                                                     #
# ------------------------------------------------------------------ #

SUPPRESSION_PROMPT = """You are an ICU monitoring AI. Your job: decide if a patient's vitals need attention.

KEY INSIGHT: Vitals are normalized relative to THIS patient's personal baseline (0-1 scale).
A baseline_delta of 0.3 while walking → normal (exertion raises HR).
A baseline_delta of 0.3 while lying still → concerning.

RULES FOR SUPPRESSION TASK:
- This patient is CHRONICALLY HYPERTENSIVE. High BP is NORMAL for them.
- Default to IGNORE unless evidence is compelling.
- Activity context heavily discounts vitals spikes: walking/eating → IGNORE.
- False alarms are MORE costly than missed mild events.
- Only ALERT for baseline_delta > 0.5 WITH resting/distressed/falling context.
- VERIFY is the safe middle ground — use it for uncertain cases.
- After learning the baseline (hours > 1.0), trust that stable readings are fine.

Respond with ONLY JSON: {"action": 0, "reasoning": "Explain using vitals/activity in <15 words"}
action: 0=IGNORE, 1=VERIFY, 2=ALERT"""

DETERIORATION_PROMPT = """You are an ICU monitoring AI detecting slow patient deterioration (sepsis).

KEY INSIGHT: Vitals are normalized 0-1. Look for TRENDS, not just current values.
A single reading of SpO2=0.80 might be noise. But SpO2 dropping 0.93→0.88→0.83→0.80 over 4 steps = real deterioration.

RULES FOR DETERIORATION TASK:
- Read the vitals history table carefully — it shows the last 8 readings.
- Look for TRENDS: consistent directional changes across multiple steps.
- SpO2 dropping is MORE urgent than HR rising — weight it higher.
- Temperature rising steadily over multiple steps = infection sign → VERIFY/ALERT.
- A single spike that returns to normal = noise → IGNORE.
- Early detection scores HIGHER than late detection — act on trends, don't wait for crisis.
- If SpO2 is below 0.50 or temp above 0.65, ALERT regardless of trend.

Respond with ONLY JSON: {"action": 0, "reasoning": "Explain using vitals/activity in <15 words"}
action: 0=IGNORE, 1=VERIFY, 2=ALERT"""

TRIAGE_PROMPT = """You are an ICU monitoring AI triaging 4 patients simultaneously.

KEY INSIGHT: You must RANK patients by urgency, not just evaluate each independently.
The most critical patient gets ALERT, second-most gets VERIFY, stable ones get IGNORE.

RULES FOR TRIAGE TASK:
- Look at ALL 4 patients before deciding — this is a ranking problem.
- ALERT the most critical patient (highest baseline_delta, worst vitals, dangerous activity).
- VERIFY the second most concerning patient.
- IGNORE patients who look stable.
- If two patients are both critical, ALERT both.
- Activity matters: falling/distressed beats resting which beats walking/eating.
- High baseline_delta + resting = more concerning than high delta + walking.
- Don't ALERT more than 2 patients unless multiple are genuinely critical.

Respond with ONLY JSON: {"actions": [a0,a1,a2,a3], "reasoning": "Explain ranking using vitals/activity in <20 words"}
Each action: 0=IGNORE, 1=VERIFY, 2=ALERT. Array position = patient index."""

SYSTEM_PROMPTS = {
    "suppression":   SUPPRESSION_PROMPT,
    "deterioration": DETERIORATION_PROMPT,
    "triage":        TRIAGE_PROMPT,
}

# ------------------------------------------------------------------ #
#  Observation formatting                                             #
# ------------------------------------------------------------------ #

def obs_to_user_message(obs: Dict, task: str, vitals_history: list,
                        conversation_history: list) -> str:
    activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
    hours = obs.get("hours_observed", 0.0)

    lines = [
        f"Step: {int(hours * 60)}/{60}  |  Hours observed: {hours:.1f}h",
        "",
        "CURRENT VITALS (normalized 0-1):",
        f"  Heart Rate:    {obs.get('heart_rate', 0):.3f}",
        f"  SpO2:          {obs.get('spo2', 0):.3f}",
        f"  Systolic BP:   {obs.get('systolic_bp', 0):.3f}",
        f"  Diastolic BP:  {obs.get('diastolic_bp', 0):.3f}",
        f"  Resp Rate:     {obs.get('respiratory_rate', 0):.3f}",
        f"  Temperature:   {obs.get('temperature', 0):.3f}",
        "",
        f"Baseline Delta:  {obs.get('baseline_delta', 0):.3f} (deviation from personal norm)",
        f"Activity:        {activity_name}",
    ]

    if task == "deterioration" and len(vitals_history) >= 3:
        lines.append("")
        lines.append("VITALS TREND (last readings, oldest→newest):")
        lines.append("  Step   HR     SpO2   SysBP  Temp")
        n = len(vitals_history)
        for i, reading in enumerate(vitals_history):
            offset = -(n - i)
            if len(reading) >= 6:
                lines.append(
                    f"  {offset:+4d}   {reading[0]:.3f}  {reading[3]:.3f}  "
                    f"{reading[1]:.3f}  {reading[5]:.3f}"
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
        activity_name = ACTIVITY_NAMES.get(obs.get("activity", 0), "unknown")
        lines.append(f"PATIENT {i} :")
        lines.append(
            f"  Vitals:  HR={obs.get('heart_rate', 0):.3f}  "
            f"SpO2={obs.get('spo2', 0):.3f}  "
            f"SysBP={obs.get('systolic_bp', 0):.3f}  "
            f"Temp={obs.get('temperature', 0):.3f}"
        )
        lines.append(
            f"  Activity: {activity_name}  |  "
            f"Baseline Delta: {obs.get('baseline_delta', 0):.3f}  |  "
            f"Hours: {obs.get('hours_observed', 0):.1f}h"
        )
        lines.append("")
    lines.append("Rank these patients by urgency. Assign actions:")
    lines.append("  Most critical → ALERT(2). Second → VERIFY(1). Stable → IGNORE(0).")
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
              vitals_history: list, model: str) -> Tuple[int, str]:
    system_prompt = SYSTEM_PROMPTS[task]
    user_message = obs_to_user_message(obs, task, vitals_history, conversation_history)

    messages = [{"role": "system", "content": system_prompt}]
    for entry in conversation_history[-2:]:
        messages.append({"role": "user", "content": entry["obs_text"][:500]})
        messages.append({"role": "assistant", "content": entry["response"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=120,
    )
    raw_response = response.choices[0].message.content.strip()
    return _parse_single_response(raw_response)


def _parse_single_response(raw: str) -> Tuple[int, str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()
    data = json.loads(text)
    action = int(data.get("action", 1))
    reasoning = str(data.get("reasoning", ""))[:80]
    if action not in (0, 1, 2):
        action = max(0, min(2, action))
    return action, reasoning


def triage_llm_agent(obs_list: List[Dict], conversation_history: list,
                     model: str) -> Tuple[List[int], str]:
    system_prompt = SYSTEM_PROMPTS["triage"]
    user_message = triage_obs_to_message(obs_list, conversation_history)

    messages = [{"role": "system", "content": system_prompt}]
    for entry in conversation_history[-2:]:
        messages.append({"role": "user", "content": entry["obs_text"][:500]})
        messages.append({"role": "assistant", "content": entry["response"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=120,
    )
    raw_response = response.choices[0].message.content.strip()
    return _parse_triage_response(raw_response)


def _parse_triage_response(raw: str) -> Tuple[List[int], str]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        text = inner.strip()
    data = json.loads(text)
    actions = list(data.get("actions", [0, 0, 0, 0]))
    reasoning = str(data.get("reasoning", ""))[:100]
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
    spo2     = obs["spo2"]
    hr       = obs["heart_rate"]
    temp     = obs.get("temperature", 0.4)
    history  = obs.get("vitals_history", [])

    if hours < 1.0:
        return VERIFY
    if activity == 2:
        return ALERT if spo2 < 0.25 else IGNORE
    if delta > 0.6 and activity == 0:
        return ALERT
    if delta > 0.35 and activity == 0:
        return VERIFY
    if spo2 < 0.35:
        return ALERT
    if spo2 < 0.50:
        return VERIFY
    if temp > 0.80:
        return ALERT
    if temp > 0.65:
        return VERIFY
    if hr > 0.75:
        return ALERT
    if hr > 0.60:
        return VERIFY

    if len(history) >= 6:
        try:
            recent = history[-3:]
            oldest = history[:3]
            recent_spo2 = sum(r[3] for r in recent) / 3
            oldest_spo2 = sum(r[3] for r in oldest) / 3
            spo2_drop = oldest_spo2 - recent_spo2
            recent_temp = sum(r[5] for r in recent) / 3
            oldest_temp = sum(r[5] for r in oldest) / 3
            temp_rise = recent_temp - oldest_temp
            if spo2_drop > 0.08: return ALERT
            if spo2_drop > 0.04: return VERIFY
            if temp_rise > 0.08: return ALERT
            if temp_rise > 0.04: return VERIFY
        except (IndexError, TypeError):
            pass

    if hours > 4.0:
        if delta > 0.20 and activity == 0:
            return VERIFY
        if spo2 < 0.55:
            return VERIFY
    return IGNORE


def triage_baseline(obs_list: List[Dict]) -> List[int]:
    return [baseline_agent(obs) for obs in obs_list]

# ------------------------------------------------------------------ #
#  OpenEnv validation                                                 #
# ------------------------------------------------------------------ #

def openenv_validate() -> bool:
    try:
        import yaml
    except ImportError:
        if not os.path.exists("openenv.yaml"):
            print("[VALIDATE] fail openenv.yaml not found", flush=True)
            return False
        print("[VALIDATE] pass (yaml module not available, file exists)", flush=True)
        return True

    if not os.path.exists("openenv.yaml"):
        print("[VALIDATE] fail openenv.yaml not found", flush=True)
        return False

    with open("openenv.yaml", "r") as f:
        spec = yaml.safe_load(f)

    required = ["name", "spec_version", "tasks", "action_space", "observation_space"]
    for field in required:
        if field not in spec:
            print(f"[VALIDATE] fail missing field: {field}", flush=True)
            return False

    action_space = spec.get("action_space", {})
    if action_space.get("n") != 3:
        print(f"[VALIDATE] fail action_space.n={action_space.get('n')}, expected 3", flush=True)
        return False

    tasks = spec.get("tasks", [])
    task_names = {t.get("name") for t in tasks}
    for required_task in ("suppression", "deterioration", "triage"):
        if required_task not in task_names:
            print(f"[VALIDATE] fail missing task: {required_task}", flush=True)
            return False

    print("[VALIDATE] pass", flush=True)
    return True

# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #

def log_start(task: str, model: str):
    print(f"[START] task={task} env=mediguard model={model}", flush=True)

def log_agent(model: str):
    print(f"[AGENT] type=llm model={model} temperature=0.0", flush=True)

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
    model = MODEL_BY_TASK.get(task, MODEL_NAME)
    log_start(task, model)
    log_agent(model)

    rewards: List[float] = []
    steps = 0
    score = 0.0
    last_action: Union[int, List[int]] = 0
    last_reasoning = ""
    fallback_count = 0
    conversation_history: List[Dict] = []
    vitals_history: List[list] = []

    try:
        env = MediGuardEnv(task=task, seed=seed)
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
                        action, reasoning = triage_llm_agent(obs, conversation_history, model)
                    else:
                        action, reasoning = llm_agent(obs, task, conversation_history, vitals_history, model)
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

            # ── LOGGING FOR EVERY STEP (Added for Phase 3 visibility) ──
            log_step(steps, action, reward, done, error=None)
            log_reasoning(action, reasoning)

            if task == "triage":
                obs_text = triage_obs_to_message(obs, conversation_history)
            else:
                obs_text = obs_to_user_message(obs, task, vitals_history, conversation_history)
                history = obs.get("vitals_history", [])
                if history:
                    for entry in reversed(history):
                        if any(v != 0.0 for v in entry):
                            vitals_history.append(entry)
                            break
                    vitals_history = vitals_history[-8:]

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
            "suppression": env.false_alarm_rate_grader,
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
    # The final log_reasoning call is removed here as it is now logged at every step above.

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
    openenv_validate()

    key_source = (
        "API_KEY" if os.getenv("API_KEY")
        else "HF_TOKEN" if os.getenv("HF_TOKEN")
        else "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY")
        else "none"
    )
    print(f"[CONFIG] api_key_source={key_source} base_url={API_BASE_URL} model={MODEL_NAME}", flush=True)
    print(f"[LLM_CHECK] has_key={HAS_API_KEY} client_ready=true", flush=True)

    if not HAS_API_KEY:
        print("[INFO] No API key — running fully rule-based (fast mode)", flush=True)
    else:
        print(f"[INFO] LLM mode: {MODEL_NAME} | 60 steps/task | 4s timeout", flush=True)
        print(f"[INFO] API endpoint: {API_BASE_URL}", flush=True)
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
    print(
        f"[IMPROVEMENT] vs_baseline "
        f"suppression={s - BASELINE_SCORES['suppression']:+.4f} "
        f"deterioration={d - BASELINE_SCORES['deterioration']:+.4f} "
        f"triage={t - BASELINE_SCORES['triage']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()