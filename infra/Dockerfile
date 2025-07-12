# infra/Dockerfile
FROM python:3.11-slim

WORKDIR /code

# OS build tools for numpy/pydantic etc. – comment out if not needed
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
