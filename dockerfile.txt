FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
