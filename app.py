"""
HotelGuard-AI — Gradio App
Interactive UI for hospitality crisis detection.
"""

import json
import os
import sys
import threading
import html
import gradio as gr

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import os
import sys

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "dashboard.html"), "r", encoding="utf-8") as f:
        _DASHBOARD_HTML_CONTENT = f.read()
except Exception as e:
    _DASHBOARD_HTML_CONTENT = f"<h2>dashboard.html not found: {e}</h2>"

from hotelguard_env import HotelGuardEnv
from inference import (
    baseline_agent, triage_baseline,
    obs_to_user_message, triage_obs_to_message,
    CONTEXT_NAMES, MONITOR, DISPATCH, EMERGENCY,
    MODEL_BY_TASK,
    llm_agent, triage_llm_agent,
)

# ── Token check ────────────────────────────────────────────────────
_llm_available = bool(os.getenv("GEMINI_API_KEY"))

# ── Firebase Realtime Database ─────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, db as firebase_db

_FIREBASE_URL = os.getenv("FIREBASE_URL")
_firebase_enabled = False

if _FIREBASE_URL:
    try:
        # None credentials works for Firebase Realtime Database with default/anonymous
        # access rules. On HuggingFace Spaces, this may fail silently — that's OK,
        # _firebase_enabled will stay False and the app works without it.
        # For full Firebase integration, set GOOGLE_APPLICATION_CREDENTIALS to a
        # service account JSON path.
        firebase_admin.initialize_app(None, {"databaseURL": _FIREBASE_URL})
        _firebase_enabled = True
    except Exception:
        _firebase_enabled = False


def _push_to_firebase(obs, task):
    if not _firebase_enabled:
        return
    try:
        if isinstance(obs, list):
            for i, o in enumerate(obs):
                firebase_db.reference(f"/zones/{task}/{i}/latest").set(
                    {k: v for k, v in o.items() if k != "signal_history"}
                )
        else:
            firebase_db.reference(f"/zones/{task}/0/latest").set(
                {k: v for k, v in obs.items() if k != "signal_history"}
            )
    except Exception:
        pass


# ── Global episode state ───────────────────────────────────────────
_state_lock   = threading.Lock()

_env          = None
_current_task = None
_step_count   = 0
_total_reward = 0.0
_episode_log  = []
_conv_history = []
_signal_history = []
_last_obs     = None

ACTION_LABELS = {0: "MONITOR",  1: "DISPATCH",  2: "EMERGENCY"}
ACTION_EMOJI  = {0: "✅",        1: "👁️",         2: "🚨"}
CONTEXT_EMOJI = {
    0: "Quiet Hours",
    1: "Meal Service",
    2: "Event",
    3: "Distress",
    4: "Emergency",
}

import time
try:
    with open(os.path.join(base_dir, "results_cache.json"), "r") as f:
        _CACHE_DATA = json.load(f)
except Exception:
    _CACHE_DATA = {}

_agent_choices = ["Simulation Replay (Offline)", "Rule-Based", "Manual"]
_agent_default = "Simulation Replay (Offline)"
if _llm_available:
    _agent_choices = ["LLM Agent"] + _agent_choices


# ══════════════════════════════════════════════════════════════════
# Observation formatting
# ══════════════════════════════════════════════════════════════════

def _risk_tag(delta, panic, smoke):
    if delta > 0.5 or panic > 0.6 or smoke > 0.5:
        return "CRITICAL"
    elif delta > 0.25 or panic > 0.3 or smoke > 0.2:
        return "BORDERLINE"
    return "STABLE"


def _fmt_single(obs: dict) -> str:
    motion = obs.get("motion_level",    0)
    sound  = obs.get("sound_db",        0)
    doors  = obs.get("door_events",     0)
    panic  = obs.get("panic_score",     0)
    occ    = obs.get("occupancy_delta", 0)
    smoke  = obs.get("smoke_co_level",  0)
    delta  = obs.get("baseline_delta",  0)
    hours  = obs.get("hours_observed",  0)
    ctx    = CONTEXT_EMOJI.get(obs.get("activity", 0), "Unknown")

    motion_raw = round(motion * 100, 1)
    sound_raw  = round(30 + sound * 90, 1)
    doors_raw  = round(doors * 20, 1)
    panic_raw  = round(panic, 3)
    occ_raw    = round(occ * 50, 1)
    smoke_raw  = round(smoke, 3)

    risk     = _risk_tag(delta, panic, smoke)
    risk_map = {"CRITICAL": "[CRITICAL]", "BORDERLINE": "[BORDERLINE]", "STABLE": "[STABLE]"}

    lines = [
        "  ZONE SENSOR READINGS",
        "  " + "─"*40,
        f"  Motion Level      {motion_raw:>6}",
        f"  Sound dB          {sound_raw:>6} dB",
        f"  Door Events       {doors_raw:>6} /min",
        f"  Panic Score       {panic_raw:>6}",
        f"  Occupancy Δ       {occ_raw:>6}",
        f"  Smoke/CO          {smoke_raw:>6}",
        "  " + "─"*40,
        f"  Baseline Delta    {delta:.3f}",
        f"  Time Observed     {hours:.1f} hours",
        f"  Context           {ctx}",
        "  " + "─"*40,
        f"  Status            {risk_map[risk]}",
    ]
    return "\n".join(lines)


def _fmt_triage(obs_list: list) -> str:
    zone_names = ["Lobby", "Event Hall", "Guest Room", "Pool Area"]
    risk_weights = {"CRITICAL": 2, "BORDERLINE": 1, "STABLE": 0}
    risk_labels = {"CRITICAL": "EMERGENCY", "BORDERLINE": "DISPATCH", "STABLE": "MONITOR"}
    risk_emoji = {"CRITICAL": "🚨", "BORDERLINE": "⚠️", "STABLE": "✅"}
    
    ranked = []
    for i, obs in enumerate(obs_list):
        panic  = obs.get("panic_score", 0)
        smoke  = obs.get("smoke_co_level", 0)
        delta  = obs.get("baseline_delta", 0)
        risk   = _risk_tag(delta, panic, smoke)
        ranked.append({
            "idx": i,
            "name": zone_names[i] if i < len(zone_names) else f"Zone {i}",
            "risk": risk,
            "weight": risk_weights[risk],
            "panic": panic,
            "delta": delta,
            "obs": obs
        })
        
    # Sort by urgency
    ranked.sort(key=lambda x: (x["weight"], x["panic"], x["delta"]), reverse=True)
    
    shifts = []
    for r in ranked:
        tag = risk_labels[r['risk']]
        shifts.append(f"Zone {r['idx']} ({r['name']}): {tag}")
        
    lines = [
        "  [DYNAMIC TRIAGE RANKING]",
        "  " + " ➔ ".join(shifts),
        "  " + "═"*80,
        "  DETAILED SENSOR FEED"
    ]
    
    for r in ranked:
        obs = r["obs"]
        motion = obs.get("motion_level", 0)
        panic = obs.get("panic_score", 0)
        smoke = obs.get("smoke_co_level", 0)
        delta = obs.get("baseline_delta", 0)
        ctx = CONTEXT_EMOJI.get(obs.get("activity", 0), "Unknown")
        risk_tag = risk_labels[r["risk"]]
        emoji = risk_emoji[r["risk"]]
        
        lines += [
            f"  {emoji} Zone {r['idx']} ({r['name']}) — {risk_tag} ",
            f"      Activity: {ctx:<12} | Motion: {motion*100:3.0f} | Panic: {panic:.3f} | Smoke: {smoke:.3f} | Δ: {delta:.3f}",
            "  " + "─"*55,
        ]
        
    return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════
# Zone Floor Plan (2×2 grid)
# ══════════════════════════════════════════════════════════════════

def _build_floor_plan(obs_data, actions_taken=None):
    """Build a dynamically sorted zone floor plan HTML list, color-coded by risk level."""
    if obs_data is None:
        return '<div style="text-align:center;color:#94a3b8;padding:40px">Reset to see zone floor plan</div>'

    if not isinstance(obs_data, list):
        obs_data = [obs_data]

    zone_names = ["Lobby", "Event Hall", "Guest Room", "Pool Area"]
    if len(obs_data) == 1:
        zone_names = [_current_task.replace("_", " ").title() if _current_task else "Zone 0"]

    cards = []
    risk_weights = {"CRITICAL": 2, "BORDERLINE": 1, "STABLE": 0}

    for i, obs in enumerate(obs_data):
        panic  = obs.get("panic_score", 0)
        smoke  = obs.get("smoke_co_level", 0)
        delta  = obs.get("baseline_delta", 0)
        motion = obs.get("motion_level", 0)
        risk   = _risk_tag(delta, panic, smoke)
        weight = risk_weights[risk]

        color_map = {
            "STABLE":     ("#dcfce7", "#15803d", "#166534"),
            "BORDERLINE": ("#fef3c7", "#d97706", "#92400e"),
            "CRITICAL":   ("#fee2e2", "#dc2626", "#991b1b"),
        }
        bg, accent, text = color_map[risk]

        act_emoji = ""
        if actions_taken is not None:
            if isinstance(actions_taken, list) and i < len(actions_taken):
                act_emoji = ACTION_EMOJI.get(actions_taken[i], "")
            elif not isinstance(actions_taken, list):
                act_emoji = ACTION_EMOJI.get(actions_taken, "") if i == 0 else ""

        # Top 2 signals: pick panic and smoke as most safety-critical
        sig1 = f"Panic: {panic:.3f}"
        sig2 = f"Smoke: {smoke:.3f}"

        name = zone_names[i] if i < len(zone_names) else f"Zone {i}"
        card_html = f"""
        <div style="background:{bg};border:2px solid {accent};border-radius:14px;padding:16px;min-height:100px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <strong style="color:{text};font-family:'Rajdhani',sans-serif;font-size:1.1em">{name}</strong>
                <span style="font-size:1.3em">{act_emoji}</span>
            </div>
            <div style="display:inline-block;background:{accent};color:white;padding:2px 10px;border-radius:20px;font-size:0.78em;font-weight:700;font-family:'JetBrains Mono',monospace;margin-bottom:8px">{risk}</div>
            <div style="color:{text};font-size:0.85em;font-family:'JetBrains Mono',monospace;line-height:1.6">{sig1}<br>{sig2}</div>
        </div>
        """
        cards.append({
            "html": card_html,
            "weight": weight,
            "panic": panic,
            "delta": delta,
            "idx": i
        })

    # Sort by urgency
    cards.sort(key=lambda x: (x["weight"], x["panic"], x["delta"]), reverse=True)

    sorted_html = "".join([c["html"] for c in cards])

    # If empty or fewer than expected, pad (though normally 1 or 4)
    if len(cards) == 0:
        sorted_html = '<div style="border:2px dashed #d0d8f0;border-radius:14px;padding:16px;min-height:100px;display:flex;align-items:center;justify-content:center;color:#94a3b8">—</div>'

    html = f"""
    <div style="display:flex;flex-direction:column;gap:12px;padding:4px">
        {sorted_html}
    </div>
    """
    return html


# ══════════════════════════════════════════════════════════════════
# Agent helpers
# ══════════════════════════════════════════════════════════════════

def _agent_action(obs, task, agent_mode):
    global _conv_history, _signal_history, _step_count
    
    if agent_mode == "Simulation Replay (Offline)":
        time.sleep(1.2) # simulated latency
        try:
            record = _CACHE_DATA[task]["steps"][_step_count]
            used_llm = record.get("agent") == "llm"
            return record["action"], record["reasoning"], used_llm
        except Exception:
            pass # fallback to regular agent if cache fails

    if agent_mode == "LLM Agent" and _llm_available:
        model_name = MODEL_BY_TASK.get(task, "gemini-flash-latest")
        try:
            if task == "triage":
                action, reasoning = triage_llm_agent(obs, _conv_history, model_name)
            else:
                action, reasoning = llm_agent(obs, task, _conv_history, _signal_history, model_name)
            return action, reasoning, True
        except Exception as e:
            err = f"llm_err:{type(e).__name__}"
            return (triage_baseline(obs) if task == "triage" else baseline_agent(obs)), err, False
    if task == "triage":
        return triage_baseline(obs), "rule-based", False
    return baseline_agent(obs), "rule-based", False


def _compute_score(env, task):
    fn = {
        "suppression":  env.suppression_grader,
        "deterioration": env.deterioration_grader,
        "triage":        env.triage_grader,
    }.get(task)
    return float(fn()) if fn else 0.0


# ══════════════════════════════════════════════════════════════════
# Demo functions (Gradio callbacks)
# ══════════════════════════════════════════════════════════════════

def demo_reset(task, seed, agent_mode):
    global _env, _current_task, _step_count, _total_reward
    global _episode_log, _conv_history, _signal_history, _last_obs

    _env          = HotelGuardEnv(task=task, seed=int(seed))
    _current_task = task
    _step_count   = 0
    _total_reward = 0.0
    _episode_log  = []
    _conv_history = []
    _signal_history = []
    obs           = _env.reset()
    _last_obs     = obs

    obs_display = _fmt_triage(obs) if isinstance(obs, list) else _fmt_single(obs)
    _episode_log.append(f"[RESET] task={task} seed={seed} agent={agent_mode}")

    manual     = agent_mode == "Manual"
    is_tri     = task == "triage"
    difficulty = {"deterioration": "Easy", "suppression": "Medium", "triage": "Hard"}[task]
    status     = f"READY  {task.upper()} [{difficulty}]  Seed {seed}  {agent_mode}"

    floor_plan = _build_floor_plan(obs)

    return (
        status, obs_display,
        "—", "—", "—", "0 / 60",
        "\n".join(_episode_log),
        floor_plan,
        gr.update(interactive=True),
        gr.update(visible=manual and not is_tri),
        gr.update(visible=manual and is_tri),
    )


def demo_step(action_radio, triage_txt, agent_mode):
    global _env, _step_count, _total_reward
    global _episode_log, _conv_history, _signal_history, _last_obs

    if _env is None or _last_obs is None:
        return ("Reset the environment first!",) + ("",)*6 + ("",) + ("",) + (gr.update(),)*3

    task = _current_task
    obs  = _last_obs

    if agent_mode == "Manual":
        if task == "triage":
            try:
                parsed = [max(0, min(2, int(a.strip()))) for a in triage_txt.split(",")]
                while len(parsed) < 4:
                    parsed.append(0)
                action = parsed[:4]
            except ValueError:
                action = [0, 0, 0, 0]
        else:
            try:
                action = int(action_radio[0]) if action_radio else 1
            except Exception:
                action = 1
        reasoning, used_llm = "manual", False
    else:
        action, reasoning, used_llm = _agent_action(obs, task, agent_mode)

    obs_next, reward, done, info = _env.step(action)
    _step_count   += 1
    _total_reward += reward
    mean_r         = _total_reward / _step_count
    _last_obs      = obs_next

    # ── Push observation to Firebase RTDB ──
    _push_to_firebase(obs_next, task)

    if task == "triage":
        obs_text = triage_obs_to_message(
            obs_next if isinstance(obs_next, list) else [obs_next], _conv_history)
    else:
        obs_text = obs_to_user_message(obs_next, task, _signal_history, _conv_history)

    hist = obs_next.get("signal_history", []) if not isinstance(obs_next, list) else []
    if hist:
        for entry in reversed(hist):
            if any(v != 0.0 for v in entry):
                _signal_history.append(entry)
                break
    _signal_history = _signal_history[-8:]

    _conv_history.append({
        "obs_text": obs_text,
        "response": json.dumps({"action": action, "reasoning": reasoning}),
        "action":   action,
        "reward":   reward,
    })
    if len(_conv_history) > 4:
        _conv_history = _conv_history[-4:]

    if isinstance(action, list):
        act_str = " ".join(
            f"Z{i}:{ACTION_EMOJI.get(a,'?')}{ACTION_LABELS.get(a,'?')}"
            for i, a in enumerate(action))
    else:
        act_str = f"{ACTION_EMOJI.get(action,'?')} {ACTION_LABELS.get(action,'?')}"

    obs_display = _fmt_triage(obs_next) if isinstance(obs_next, list) else _fmt_single(obs_next)
    tag         = "LLM" if used_llm else ("RULE" if agent_mode == "Rule-Based" else "USER")
    _episode_log.append(f"[{_step_count:03d}] {tag}  {act_str[:38]}  R={reward:+.3f}")
    if len(_episode_log) > 200:
        _episode_log = _episode_log[-200:]

    manual = agent_mode == "Manual"
    is_tri = task == "triage"

    floor_plan = _build_floor_plan(obs_next, actions_taken=action)

    if done:
        score  = _compute_score(_env, task)
        status = f"COMPLETE  Score: {score:.4f}  Mean Reward: {mean_r:.3f}  Steps: {_step_count}"
        _episode_log += [
            "=" * 50,
            f"  FINAL SCORE : {score:.4f}",
            f"  MEAN REWARD : {mean_r:.4f}",
            "=" * 50,
        ]
        return (status, obs_display, act_str, f"{reward:+.3f}", f"{mean_r:.4f}",
                f"{_step_count}/60", "\n".join(_episode_log[-80:]),
                floor_plan,
                gr.update(interactive=False),
                gr.update(visible=manual and not is_tri),
                gr.update(visible=manual and is_tri))

    status = f"Step {_step_count}/60  Last Reward: {reward:+.3f}  Mean: {mean_r:.3f}"
    return (status, obs_display, act_str, f"{reward:+.3f}", f"{mean_r:.4f}",
            f"{_step_count}/60", "\n".join(_episode_log[-80:]),
            floor_plan,
            gr.update(interactive=True),
            gr.update(visible=manual and not is_tri),
            gr.update(visible=manual and is_tri))


def demo_run_all(_):
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True, text=True,
            env=os.environ.copy(), timeout=55,
        )
        return (r.stdout + ("\n" + r.stderr if r.stderr else "")) or "No output."
    except subprocess.TimeoutExpired:
        return "Timed out after 15 min."
    except Exception as e:
        return f"Error: {e}"


def on_agent_change(agent_mode, task):
    m, t = agent_mode == "Manual", task == "triage"
    return gr.update(visible=m and not t), gr.update(visible=m and t)


def on_task_change(task, agent_mode):
    m, t = agent_mode == "Manual", task == "triage"
    return gr.update(visible=m and not t), gr.update(visible=m and t)


# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Nunito:wght@400;600;700;800&display=swap');

html, body,
.gradio-container,
.gradio-container * {
    color-scheme: dark !important;
}
body,
.gradio-container {
    background: #0f172a !important;
    font-family: 'Nunito', sans-serif !important;
    color: #e2e8f0 !important;
}

.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 30px 36px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(29,78,216,0.25);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.07), transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5em;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0 0 5px;
    letter-spacing: 1px;
}
.app-header .tagline { color: rgba(255,255,255,0.75) !important; font-size: 0.9em; margin: 0 0 18px; }
.pill {
    display: inline-block;
    border-radius: 30px;
    padding: 4px 13px;
    font-size: 0.75em;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 5px 0 0;
    letter-spacing: 0.5px;
}
.p-white  { background: rgba(255,255,255,0.18); color: #ffffff !important; }
.p-green  { background: #dcfce7; color: #15803d !important; }
.p-yellow { background: #fef9c3; color: #a16207 !important; }
.p-red    { background: #fee2e2; color: #b91c1c !important; }
.p-llm-on  { background: #dcfce7; color: #15803d !important; }
.p-llm-off { background: #fee2e2; color: #b91c1c !important; }

.status-bar textarea, .status-bar input {
    background: #1e3a8a !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.87em !important;
    font-weight: 600 !important;
    color: #bfdbfe !important;
}

.vitals-box textarea, .vitals-box input {
    background: #0f172a !important;
    border: 2px solid #2563eb !important;
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92em !important;
    font-weight: 500 !important;
    color: #e0f2fe !important;
    line-height: 1.8 !important;
}

.log-box textarea, .log-box input {
    background: #f8fafc !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8em !important;
    color: #334155 !important;
    line-height: 1.6 !important;
}

.full-log textarea, .full-log input {
    background: #fffbeb !important;
    border: 2px solid #fde047 !important;
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8em !important;
    color: #78350f !important;
    line-height: 1.6 !important;
}

.btn-reset {
    background: linear-gradient(135deg,#1d4ed8,#4f46e5) !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 1.05em !important;
    font-weight: 700 !important; color: #ffffff !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(79,70,229,.4) !important;
    transition: all .2s !important;
}
.btn-reset:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(79,70,229,.55) !important; }
.btn-step {
    background: linear-gradient(135deg,#059669,#10b981) !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 1.05em !important;
    font-weight: 700 !important; color: #ffffff !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(16,185,129,.35) !important;
    transition: all .2s !important;
}
.btn-step:hover { transform: translateY(-1px) !important; }
.btn-run {
    background: linear-gradient(135deg,#b45309,#d97706) !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 1em !important;
    font-weight: 700 !important; color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(217,119,6,.3) !important;
}

.sec-h {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1em; font-weight: 700;
    color: #e2e8f0 !important;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin: 14px 0 8px;
}

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .block { color: #1a1f3a !important; }
.gradio-container input,
.gradio-container select,
.gradio-container textarea {
    font-family: 'Nunito', sans-serif !important;
    border: 2px solid #d0d8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    color: #1a1f3a !important;
}
.gradio-container .options,
.gradio-container ul.options {
    background: #ffffff !important;
    border: 2px solid #d0d8f0 !important;
    color: #1a1f3a !important;
}
.gradio-container ul.options li:hover { background: #eff6ff !important; }

.tab-nav { border-bottom: 2px solid #d0d8f0 !important; }
.tab-nav button {
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    font-size: 1em !important; color: #6b7280 !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
}

.gradio-container .wrap .wrap-inner label {
    border: 2px solid #d0d8f0 !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    background: #ffffff !important;
    color: #6b7280 !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}
.gradio-container .wrap .wrap-inner label:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
    color: #2563eb !important;
}
.gradio-container .wrap .wrap-inner label:has(input[type="radio"]:checked) {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
}
.gradio-container .wrap .wrap-inner label input[type="radio"] {
    accent-color: #ffffff !important;
    width: 14px !important; height: 14px !important;
}


.hw-card {
    background: #1e293b !important;
    border: 2px solid #334155;
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 14px;
}
.hw-card h3 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2em; font-weight: 700;
    color: #3b82f6 !important;
    margin: 0 0 14px;
}
.score-block { border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }
.sb-g { background: rgba(22, 163, 74, 0.15) !important; border-left: 5px solid #16a34a; }
.sb-y { background: rgba(217, 119, 6, 0.15) !important; border-left: 5px solid #d97706; }
.sb-r { background: rgba(220, 38, 38, 0.15) !important; border-left: 5px solid #dc2626; }

::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: #e8edf8; }
::-webkit-scrollbar-thumb { background: #d0d8f0; border-radius: 3px; }
"""


# ══════════════════════════════════════════════════════════════════
# HTML blocks
# ══════════════════════════════════════════════════════════════════

_llm_pill = (
    '<span class="pill p-llm-on">LLM CONNECTED</span>'
    if _llm_available else
    '<span class="pill p-llm-off">LLM OFFLINE — set GEMINI_API_KEY</span>'
)

HEADER_HTML = f"""
<div class="app-header">
  <h1>🏨 HotelGuard-AI</h1>
  <p class="tagline">Context-Aware Hospitality Crisis Detection &nbsp;·&nbsp; AI Safety Monitoring</p>
  <span class="pill p-green">🟢 Deterioration · Easy</span>
  <span class="pill p-yellow">🟡 Suppression · Medium</span>
  <span class="pill p-red">🔴 Triage · Hard</span>
  <span class="pill p-white">Gemini Flash</span>
  <span class="pill p-white">NDCG@4</span>
  {_llm_pill}
</div>
"""

HOW_INSIGHT_HTML = """
<div class="hw-card">
  <h3>The Core Insight</h3>
  <p style="color:#cbd5e1;line-height:1.7;margin:0 0 16px">
    Sensor signals are z-scores from <strong style="color:#f1f5f9;">this zone's own baseline</strong> — not global norms.
    The same reading means something entirely different depending on the venue context.
  </p>
  <table style="width:100%;border-collapse:collapse;font-size:0.9em;color:#e2e8f0;">
    <tr style="border-bottom:2px solid #334155">
      <th style="text-align:left;padding:8px 6px;color:#94a3b8;font-size:0.83em;font-family:JetBrains Mono,monospace">Situation</th>
      <th style="text-align:center;padding:8px;color:#94a3b8;font-size:0.83em;font-family:JetBrains Mono,monospace">Sound 95 dB</th>
      <th style="text-align:center;padding:8px;color:#94a3b8;font-size:0.83em;font-family:JetBrains Mono,monospace">Correct Action</th>
    </tr>
    <tr style="border-bottom:1px solid #1e293b">
      <td style="padding:10px 6px">🎉 Event in progress</td>
      <td style="text-align:center;color:#4ade80;font-weight:700">Expected</td>
      <td style="text-align:center"><span style="background:rgba(22,163,74,0.25);color:#4ade80;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">✅ MONITOR</span></td>
    </tr>
    <tr style="border-bottom:1px solid #1e293b">
      <td style="padding:10px 6px">🍽 Meal service</td>
      <td style="text-align:center;color:#fbbf24;font-weight:700">Possible</td>
      <td style="text-align:center"><span style="background:rgba(217,119,6,0.25);color:#fbbf24;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">👁️ DISPATCH</span></td>
    </tr>
    <tr>
      <td style="padding:10px 6px">🌙 Quiet hours</td>
      <td style="text-align:center;color:#f87171;font-weight:700">DANGER</td>
      <td style="text-align:center"><span style="background:rgba(220,38,38,0.25);color:#f87171;padding:4px 16px;border-radius:20px;font-weight:700;font-size:0.88em">🚨 EMERGENCY</span></td>
    </tr>
  </table>
</div>
"""

SCORING_HTML = f"""
<div class="hw-card">
  <h3>Scoring System</h3>
  <div class="score-block sb-g">
    <strong style="color:#4ade80;font-size:1.05em">🟢 Deterioration &nbsp;·&nbsp; Easy</strong>
    <p style="margin:6px 0 0;color:#cbd5e1;font-size:0.87em;line-height:1.6">
      Onset-delay: score = 0.3 + 0.7 × (1 − delay/30).
      Detect crisis drift early for high score. Miss it completely → 0.
      <br><strong style="color:#f1f5f9;">Baseline ~0.75 &nbsp;·&nbsp; LLM target ~0.85+</strong>
    </p>
  </div>
  <div class="score-block sb-y">
    <strong style="color:#fbbf24;font-size:1.05em">🟡 Suppression &nbsp;·&nbsp; Medium</strong>
    <p style="margin:6px 0 0;color:#cbd5e1;font-size:0.87em;line-height:1.6">
      F1 score — harmonic mean of sensitivity and specificity.
      Monitoring everything gives 0. Emergency-calling everything also penalised.
      <br><strong style="color:#f1f5f9;">Baseline ~0.42 &nbsp;·&nbsp; LLM target ~0.75+</strong>
    </p>
  </div>
  <div class="score-block sb-r">
    <strong style="color:#f87171;font-size:1.05em">🔴 Triage &nbsp;·&nbsp; Hard</strong>
    <p style="margin:6px 0 0;color:#cbd5e1;font-size:0.87em;line-height:1.6">
      NDCG@4 (50%) + EMERGENCY-F1 (30%) + Responsiveness (20%) − penalties.
      Sending DISPATCH to every zone is penalised. Must differentiate.
      <br><strong style="color:#f1f5f9;">Baseline ~0.23 &nbsp;·&nbsp; LLM target ~0.65+</strong>
    </p>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════
# Gradio layout
# ══════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="HotelGuard-AI — Hospitality Safety",
) as gradio_app:

    if not _llm_available:
        gr.HTML(
            '<div style="background: rgba(220, 38, 38, 0.15); border: 1px solid rgba(220, 38, 38, 0.4); color:#f87171; padding:12px; margin-bottom:15px; border-radius:14px; text-align:center; font-weight:600; font-family:\'Nunito\', sans-serif;">'
            '⚠️ <span style="font-family:\'JetBrains Mono\',monospace; font-weight:700;">[Rule-Based Mode — API key not set]</span> Offline fallback active. Provide <code>GEMINI_API_KEY</code> for LLM features.'
            '</div>'
        )

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        with gr.Tab("🖥️ Simulation Dashboard"):
            gr.HTML(
                f'<iframe style="width: 100%; height: 850px; border: none; border-radius: 12px; overflow: hidden;" '
                f'srcdoc="{html.escape(_DASHBOARD_HTML_CONTENT, quote=True)}"></iframe>'
            )

        # ── TAB 1: PLAY ──────────────────────────────────────────
        with gr.Tab("🎮 Interactive Demo"):

            with gr.Row():
                metric_step   = gr.Textbox(value="0 / 60", label="Step",        interactive=False)
                metric_reward = gr.Textbox(value="—",      label="Last Reward", interactive=False)
                metric_mean   = gr.Textbox(value="—",      label="Mean Reward", interactive=False)
                metric_action = gr.Textbox(value="—",      label="Last Action", interactive=False)

            status_out = gr.Textbox(
                label="Status",
                value="Select a task below and hit Reset to start",
                interactive=False,
                elem_classes=["status-bar"],
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=270):
                    gr.HTML('<div class="sec-h">Configuration</div>')

                    task_dd = gr.Dropdown(
                        choices=["suppression", "deterioration", "triage"],
                        value="suppression",
                        label="Task",
                        info="Easy → Medium → Hard",
                    )
                    seed_num = gr.Number(value=42, label="Random Seed", precision=0)
                    agent_radio = gr.Radio(
                        choices=_agent_choices,
                        value=_agent_default,
                        label="Agent Mode",
                        info=(
                            "GEMINI_API_KEY detected — LLM active"
                            if _llm_available
                            else "LLM requires GEMINI_API_KEY (not set) — using Rule-Based"
                        ),
                    )
                    reset_btn = gr.Button(
                        "🔄 Reset Environment",
                        variant="primary", size="lg",
                        elem_classes=["btn-reset"],
                    )

                    gr.HTML('<div class="sec-h" style="margin-top:18px">Action</div>')
                    gr.Markdown(
                        "_LLM and Rule-Based agents decide automatically. \n"
                        "Action controls only appear in Manual mode._"
                    )

                    with gr.Column(visible=False) as single_action_row:
                        action_radio = gr.Radio(
                            choices=["0 — ✅ Monitor", "1 — 👁️ Dispatch", "2 — 🚨 Emergency"],
                            value="1 — 👁️ Dispatch",
                            label="Single-Zone Action",
                        )

                    with gr.Column(visible=False) as triage_action_row:
                        triage_txt = gr.Textbox(
                            value="1,0,2,0",
                            label="Triage Actions [Z0, Z1, Z2, Z3]",
                            info="0=Monitor  1=Dispatch  2=Emergency",
                        )

                    step_btn = gr.Button(
                        "▶ Next Step",
                        variant="secondary", size="lg",
                        interactive=False,
                        elem_classes=["btn-step"],
                    )

                with gr.Column(scale=2):
                    # Zone Floor Plan (2×2 grid)
                    gr.HTML('<div class="sec-h">Zone Floor Plan</div>')
                    floor_plan_html = gr.HTML(
                        value='<div style="text-align:center;color:#94a3b8;padding:40px">Reset to see zone floor plan</div>',
                        elem_id="floor-plan-panel",
                    )

                    obs_out = gr.Textbox(
                        label="Sensor Monitor",
                        lines=15, max_lines=22,
                        interactive=False,
                        elem_classes=["vitals-box"],
                        placeholder="Reset to see live sensor data...",
                    )
                    log_out = gr.Textbox(
                        label="Episode Log",
                        lines=9,
                        interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Step decisions appear here...",
                    )

            gr.HTML('<hr style="border:none;border-top:2px solid #e5e7eb;margin:22px 0">')
            gr.HTML('<div class="sec-h">Full Inference Pipeline</div>')
            gr.Markdown(
                "Runs `inference.py` end-to-end (all 3 tasks, LLM agent, ~5–15 min). "
                "Requires `GEMINI_API_KEY`."
            )
            run_all_btn = gr.Button(
                "🏃 Run Full Inference (All 3 Tasks)",
                variant="stop", elem_classes=["btn-run"],
            )
            full_log = gr.Textbox(
                label="Inference Output",
                lines=20, interactive=False,
                elem_classes=["full-log"],
                placeholder="[START] / [STEP] / [END] logs appear here...",
            )

        # ── TAB 2: HOW IT WORKS ──────────────────────────────────
        with gr.Tab("💡 How It Works"):
            with gr.Row():
                with gr.Column():
                    gr.HTML(HOW_INSIGHT_HTML)
                    gr.Markdown("""
### Three Tasks — Easy to Hard

| Task | Zones | Challenge |
|------|:-----:|-----------| 
| 🟢 Deterioration | 1 guest room | Catch the trend before crisis |
| 🟡 Suppression   | 1 event ballroom | High noise is *normal* — don't alarm |
| 🔴 Triage        | 4 simultaneous | Rank by urgency — ordering matters |

### Action Space `Discrete(3)`

| Code | Action | Use when |
|:----:|--------|----------|
| `0` | ✅ MONITOR  | Zone signals match this zone's normal baseline |
| `1` | 👁️ DISPATCH | Mild deviation — send staff to investigate |
| `2` | 🚨 EMERGENCY | Genuine crisis — call emergency services now |

### Observation Fields (10 per zone)

`motion_level` · `sound_db` · `door_events` · `panic_score` · `occupancy_delta` · `smoke_co_level`
— all normalized 0–1 relative to this zone's rolling baseline

`baseline_delta` — combined deviation score (0 = normal, 1 = extreme)

`hours_observed` · `activity` (0=quiet hrs 1=meal 2=event 3=distress 4=emergency)

`signal_history` — last 10 readings for trend detection
""")

                with gr.Column():
                    gr.HTML(SCORING_HTML)
                    gr.Markdown(f"""
### LLM Agent Design

Task-specific system prompts + 4-turn sliding conversation window for pseudo-online learning.

```
System Prompt  (task rules + thresholds)
+ Signal trend table  (last 8 readings)
+ Recent decisions and their rewards
→ LLM picks: MONITOR / DISPATCH / EMERGENCY
→ Falls back to rule-based on any error
```

**Model:** Gemini 2.0 Flash
**LLM:** {"✅ Connected" if _llm_available else "❌ Offline — set `GEMINI_API_KEY`"}
""")

    # ── Event wiring ──────────────────────────────────────────────
    _shared = [
        status_out, obs_out,
        metric_action, metric_reward, metric_mean, metric_step,
        log_out,
        floor_plan_html,
        step_btn,
        single_action_row, triage_action_row,
    ]

    reset_btn.click(demo_reset, [task_dd, seed_num, agent_radio], _shared)
    step_btn.click(demo_step,   [action_radio, triage_txt, agent_radio], _shared)
    agent_radio.change(on_agent_change, [agent_radio, task_dd], [single_action_row, triage_action_row])
    task_dd.change(on_task_change,      [task_dd, agent_radio], [single_action_row, triage_action_row])
    run_all_btn.click(demo_run_all, [agent_radio], [full_log])



# ══════════════════════════════════════════════════════════════════
# Entry point — Gradio native launch (no FastAPI mount)
# gr.mount_gradio_app causes unhashable dict crash on Python 3.13
# ══════════════════════════════════════════════════════════════════

def main():
    port = int(os.getenv("PORT", 7860))
    print(f"[STARTUP] HotelGuard-AI starting on 0.0.0.0:{port}", flush=True)
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ).set(
            body_background_fill="#0f172a",
            body_text_color="#e2e8f0",
            background_fill_primary="#1e293b",
            background_fill_secondary="#0f172a",
            border_color_primary="#334155",
            color_accent="#3b82f6",
            color_accent_soft="#1e3a8a",
            block_background_fill="#1e293b",
            block_border_color="#334155",
            block_label_text_color="#e2e8f0",
            input_background_fill="#0f172a",
            input_border_color="#334155",
            input_placeholder_color="#64748b",
            code_background_fill="#1e293b",
            code_background_fill_dark="#1e293b",
        )
    )

if __name__ == "__main__":
    main()