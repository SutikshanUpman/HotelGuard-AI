import os
from google import genai
from google.genai import types

def test_suppression_logic():
    key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    
    system_prompt = """You are an AI safety agent. Monitor a hotel zone.
Signals: motion, sound, doors, panic, occ, smoke.
If Context 2 (event): Ignore high sound/motion. 
If Panic spike + Motion drop: EMERGENCY (2).
Else if stable: MONITOR (0).
Output JSON only: {"action": <number>, "reason": "<text>"}"""

    user_msg = """Step: 1/60 | Context: event in progress
Signals: Motion: 0.5, Sound: 0.47, Panic: 0.05, Smoke: 0.02"""

    full_prompt = system_prompt + "\n\n" + user_msg
    
    print("Testing with gemini-flash-latest...")
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1000, # Increased significantly
            ),
        )
        print("--- RESPONSE INFO ---")
        print(f"Model Version: {response.model_version}")
        print(f"Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
        print(f"Text Content: '{response.text}'")
        
        # Check for thoughts
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for i, part in enumerate(response.candidates[0].content.parts):
                if part.thought:
                    print(f"Thought Part {i}: {part.text}") # Note: in some versions it might be part.text or part.thought
                elif part.text:
                    print(f"Text Part {i}: {part.text}")

        print("--- FULL OBJECT (repr) ---")
        print(repr(response))
        print("---------------------")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_suppression_logic()
