FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY patient_simulator.py .
COPY reward_function.py .
COPY task1_suppression.py .
COPY task2_deterioration.py .
COPY task3_triage.py .
COPY mediguard_env.py .
COPY inference.py .
COPY app.py .
COPY openenv.yaml .
COPY README.md .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# entrypoint.sh decides what to run:
#   - If RUN_INFERENCE=1 (set by validator): python inference.py
#   - Otherwise: uvicorn app:app (HuggingFace Spaces default)
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
