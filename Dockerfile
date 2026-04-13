FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY venue_simulator.py .
COPY reward_function.py .
COPY task1_suppression.py .
COPY task2_deterioration.py .
COPY task3_triage.py .
COPY hotelguard_env.py .
COPY inference.py .
COPY app.py .
COPY server/ server/
COPY README.md .

# HotelGuard-AI uses port 7860
EXPOSE 7860

CMD ["python", "app.py"]
