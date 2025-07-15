FROM python:3.11-slim
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements desde backend
COPY backend/requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorios necesarios
RUN mkdir -p /app/data /app/ml-model

# Copiar código del backend
COPY backend/ .

# Copiar código del modelo ML
COPY ml-model/ ./ml-model

# Variables de entorno
ENV PYTHONPATH=/app:/app/ml-model
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV DATABASE_PATH=/app/data/patients.db

# Exponer puerto
EXPOSE 8000

# Comando para producción
CMD ["sh", "-c", "python auto_train.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
