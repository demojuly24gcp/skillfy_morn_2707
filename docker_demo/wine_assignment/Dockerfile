# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all teh files related to the project
COPY . .

# Install dependencies from requirements.txt file
RUN pip install -r requirements.txt


# Expose port for FastaAPI port
EXPOSE 8000

# Run the app
CMD ["python", "main.py"]
