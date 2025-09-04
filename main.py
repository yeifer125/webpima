from flask import Flask, render_template, jsonify, request
import requests
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

API_PIMA = os.environ.get("API_PIMA_URL", "https://apiparagit.onrender.com/precios")

# Middleware para mostrar la IP en logs
@app.before_request
def log_ip():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    print(f"[LOG] Conexión desde IP: {ip} - Endpoint: {request.path}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/precios")
def precios():
    try:
        res = requests.get(API_PIMA, timeout=10)
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        print(f"[ERROR] No se pudo obtener datos de PIMA: {e}")
        return jsonify({"error": f"No se pudo obtener datos de PIMA: {e}"}), 500

@app.route("/api/prediccion")
def prediccion():
    producto = request.args.get("producto")
    if not producto:
        return jsonify({"error": "Debe indicar un producto ?producto=XYZ"}), 400

    try:
        # Obtener datos en línea desde API PIMA
        res = requests.get(API_PIMA, timeout=10)
        res.raise_for_status()
        data = res.json()

        df = pd.DataFrame(data)
        df = df[df["producto"] == producto].copy()

        if df.empty:
            return jsonify({"error": f"No hay datos para {producto}"}), 404

        # Asegurar tipos correctos
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha", "moda"])
        df["moda"] = pd.to_numeric(df["moda"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("fecha")

        # Crear variable de tiempo
        df["dias"] = (df["fecha"] - df["fecha"].min()).dt.days
        X = df[["dias"]].values
        y = df["moda"].values

        if len(df) < 3:
            return jsonify({"error": "No hay suficientes datos históricos para predecir"}), 400

        # Modelo IA simple
        model = LinearRegression()
        model.fit(X, y)

        # Predicción próximos 5 días
        ult_fecha = df["fecha"].max()
        dias_futuros = np.array([df["dias"].max() + i for i in range(1, 6)]).reshape(-1, 1)
        predicciones = model.predict(dias_futuros)

        fechas_futuras = [(ult_fecha + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 6)]

        return jsonify({
            "producto": producto,
            "historico": df.to_dict(orient="records"),
            "prediccion": [{"fecha": f, "precio": round(float(p), 2)} for f, p in zip(fechas_futuras, predicciones)]
        })

    except Exception as e:
        print(f"[ERROR] Predicción falló: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/pima")
def pima():
    return render_template("pima.html")

@app.route("/info")
def info():
    return render_template("info.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
 