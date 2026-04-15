# inference.py — Hybrid Inference Upgrade Guide

**What changed, why it changed, and exactly how to implement it.**

---

## Why These Changes Are Needed

Your current `inference.py` calls Gemini on every single step of the episode — 60 calls per task, 180 calls for a full 3-task run. On the Gemini Flash free tier (15 RPM), this causes a `429 Too Many Requests` crash within the first ~15 seconds of the first task, because the RL loop processes steps much faster than 4 seconds each.

Beyond just avoiding crashes, calling the LLM every step is wasteful. Most steps are clearly stable or clearly escalating — the rule-based agent already handles these well. The LLM adds the most value at uncertain inflection points: when signals are trending but ambiguous, when context is suppressing readings, when zones need to be ranked against each other. Calling it selectively is smarter architecture, not a compromise.

---

## The Two Changes

### Change 1 — Rate Limiter

**File:** `inference.py`  
**Where:** Add near the top of the file, after your existing imports and config constants.

**Add this function:**

```python
import time

# Free tier: 15 RPM = 1 call per 4 seconds. Buffer to 4.2s to be safe.
RATE_LIMIT_DELAY = 4.2
_last_api_call_time: float = 0.0

def wait_if_needed() -> None:
    """Block until enough time has passed since the last Gemini API call."""
    global _last_api_call_time
    elapsed = time.time() - _last_api_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_api_call_time = time.time()
```

**Call it** immediately before every `c.models.generate_content(...)` call. In your current `llm_agent()` and `triage_llm_agent()` functions, add `wait_if_needed()` one line above the API call:

```python
# In llm_agent():
wait_if_needed()                          # ← ADD THIS LINE
response = c.models.generate_content(
    model=model_name,
    ...
)

# In triage_llm_agent():
wait_if_needed()                          # ← ADD THIS LINE
response = c.models.generate_content(
    model=model_name,
    ...
)
```

That's all for rate limiting. The function is self-contained — if the last call was more than 4.2 seconds ago it returns immediately, so it never adds unnecessary delay.

---

### Change 2 — Hybrid Mode (LLM every N steps)

**File:** `inference.py`  
**Where:** Add the constant near your other config constants at the top of the file. Modify the step loop inside `run_episode()`.

**Step 1 — Add the constant** (put it near `MODEL_BY_TASK`):

```python
# Call the LLM every N steps. Rule-based handles the rest.
# This reduces 60 API calls/task → ~20, and runtime from ~4 min → ~1.5 min.
USE_LLM_EVERY_N = 3
```

**Step 2 — Modify the step loop** inside `run_episode()`.

Find the block that currently looks like this (around line 378):

```python
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
```

Replace it with:

```python
if not HAS_API_KEY:
    used_fallback = True
    reasoning = "no_api_key"
elif steps % USE_LLM_EVERY_N != 0:
    # Non-LLM step — use rule-based to conserve API quota
    used_fallback = True
    reasoning = "hybrid_skip"
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
```

The only addition is the `elif steps % USE_LLM_EVERY_N != 0:` branch. When that fires, `used_fallback = True`, which means the existing code below it will call `baseline_agent(obs)` or `triage_baseline(obs)` as it already does for API errors. No other code needs to change.

---

### Change 3 — Improve the Fallback Log (optional but useful for demo)

The current `log_fallback` function prints `hybrid_skip` as the reason for skipped LLM steps, which will flood your console. Change the `log_fallback` call to only print when it's a real error, not a scheduled skip:

Find in `run_episode()`:

```python
if used_fallback or action is None:
    if task == "triage":
        action = triage_baseline(obs)
    else:
        action = baseline_agent(obs)
    fallback_count += 1
    log_fallback(steps + 1, reasoning)    # ← currently logs every hybrid skip
```

Replace with:

```python
if used_fallback or action is None:
    if task == "triage":
        action = triage_baseline(obs)
    else:
        action = baseline_agent(obs)
    fallback_count += 1
    if reasoning != "hybrid_skip":        # ← only log real errors
        log_fallback(steps + 1, reasoning)
```

---

## Final Result

After these changes:

| Metric | Before | After |
|--------|--------|-------|
| API calls per task | 60 | ~20 |
| API calls per full run | 180 | ~60 |
| Runtime per task (free tier) | ~4 min | ~1.5 min |
| Full run runtime | ~12 min | ~4.5 min |
| Free tier daily capacity | ~1 full run | ~3–4 full runs |
| Crash risk on free tier | High (429 within 15s) | None |

The fallback count printed at the end will now read something like:

```
[INFO] fallback_count=40/60 (66.7% of steps used rule-based)
```

This is expected and correct — 2/3 of steps use rule-based by design. The LLM is handling the 20 most consequential decision points per task.

---

## What NOT to Change

- Do not change the episode length (60 steps), seed (42), or any grader logic — the baseline scores in the README depend on these being stable.
- Do not change `RATE_LIMIT_DELAY` below 4.0 — the free tier hard limit is 15 RPM and there is no burst allowance.
- Do not change `USE_LLM_EVERY_N` below 3 without testing — at N=2 you get 30 calls per task, which is still within free-tier limits but leaves less headroom.
- The `wait_if_needed()` function uses a global timestamp so it works correctly even if LLM calls happen from different code paths. Do not replace it with a simple `time.sleep(4.2)` — that would add delay even when calls are already spaced far apart.

---

## Prompt for IDE Implementation

If you are implementing this in an IDE with a more powerful model (Claude Opus, GPT-4, etc.), use this prompt:

---

> I have an existing `inference.py` for a project called HotelGuard-AI. I need you to make exactly three targeted changes to it — do not restructure or rewrite anything else.
>
> **Change 1 — Rate limiter:** Add a module-level float `_last_api_call_time = 0.0` and a function `wait_if_needed()` that sleeps until 4.2 seconds have elapsed since the last call, then updates `_last_api_call_time`. Call this function immediately before every `c.models.generate_content(...)` call (there are two: one in `llm_agent()`, one in `triage_llm_agent()`).
>
> **Change 2 — Hybrid mode:** Add a constant `USE_LLM_EVERY_N = 3` near the top. In the `run_episode()` step loop, add an `elif steps % USE_LLM_EVERY_N != 0:` branch (with `used_fallback = True` and `reasoning = "hybrid_skip"`) between the `if not HAS_API_KEY:` branch and the `else:` LLM branch.
>
> **Change 3 — Suppress hybrid log noise:** In the fallback handling block inside `run_episode()`, wrap the `log_fallback(steps + 1, reasoning)` call in an `if reasoning != "hybrid_skip":` guard so hybrid skips are not logged to stdout (only real errors are).
>
> Make only these three changes. Show me a diff or the complete modified file.
