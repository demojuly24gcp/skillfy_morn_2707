from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import random
import time

app = FastAPI(title="ML App with Metrics")

# Metrics
PREDICTION_COUNTER = Counter("prediction_requests_total", "Total prediction requests")
LATENCY_HISTOGRAM = Histogram("prediction_latency_seconds", "Latency for predictions")
MODEL_DRIFT_GAUGE = Gauge("model_drift_score", "Simulated model drift score (0=no drift, 1=high drift)")

# Baseline distribution (for simulation)
BASELINE_DISTRIBUTION = {"cat": 0.5, "dog": 0.5}

@app.get("/")
def root():
    return {"message": "ML App Running"}

@app.get("/predict")
def predict():
    start_time = time.time()
    time.sleep(random.uniform(0.1, 0.5))  # simulate latency

    # Random prediction
    prediction = random.choice(["cat", "dog"])
    PREDICTION_COUNTER.inc()
    LATENCY_HISTOGRAM.observe(time.time() - start_time)

    # --- Simulate drift calculation ---
    drift_score = simulate_drift(prediction)
    MODEL_DRIFT_GAUGE.set(drift_score)

    return {"prediction": prediction, "model_drift_score": drift_score}


def simulate_drift(prediction: str):
    """
    Simulates drift by computing deviation between 
    baseline distribution and a random noisy prediction ratio.
    """
    # Randomly vary distribution to mimic real-world drift
    noisy_dist = {
        "cat": BASELINE_DISTRIBUTION["cat"] + random.uniform(-0.2, 0.2),
        "dog": BASELINE_DISTRIBUTION["dog"] + random.uniform(-0.2, 0.2)
    }

    # Ensure they sum to 1
    total = noisy_dist["cat"] + noisy_dist["dog"]
    noisy_dist["cat"] /= total
    noisy_dist["dog"] /= total

    # Simple drift score = sum of abs differences
    drift_score = abs(noisy_dist["cat"] - BASELINE_DISTRIBUTION["cat"]) + \
                  abs(noisy_dist["dog"] - BASELINE_DISTRIBUTION["dog"])
    return round(drift_score, 3)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
