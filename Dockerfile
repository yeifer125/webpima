# ---------- Imagen base ----------
FROM python:3.10-slim

# ---------- Variables de entorno ----------
ENV PYTHONUNBUFFERED=1 \
    API_PIMA_URL=https://apiparagit.onrender.com/precios

# ---------- Directorio de trabajo ----------
WORKDIR /app

# ---------- Instalar dependencias ----------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copiar código ----------
COPY . .

# ---------- Exponer puerto ----------
EXPOSE 5000

# ---------- Comando de ejecución ----------
CMD ["python", "main.py"]
