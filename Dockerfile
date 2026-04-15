FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached unless pyproject.toml changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Chainlit listens on 8000 by default
EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
