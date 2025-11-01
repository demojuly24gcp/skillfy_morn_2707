FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mlapp.py .

# Expose port for app
EXPOSE 8000

# Start FastAPI (mlapp.py defines app = FastAPI())
CMD ["uvicorn", "mlapp:app", "--host", "0.0.0.0", "--port", "8000"]
