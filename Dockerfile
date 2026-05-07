FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero (aprovecha la caché de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
