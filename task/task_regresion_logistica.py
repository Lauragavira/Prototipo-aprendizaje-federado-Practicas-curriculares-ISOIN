# task_regresion_logistica.py

import glob
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


COLUMNAS_ENTRADA = [
    "hora_salida_min",
    "distancia_km",
    "numero_entregas",
    "peso_total_kg",
    "duracion_estimada_min",
    "hora_llegada_estimada_min",
]

INPUT_DIM = len(COLUMNAS_ENTRADA)


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, 2)

    def forward(self, x):
        return self.linear(x)


def get_model():
    return LogisticRegressionModel()


def preparar_datos(df):
    df = df.copy()

    df["puntual"] = (
        df["puntual"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map({
            "si": 1,
            "sí": 1,
            "no": 0,
        })
    )

    df = df.dropna(subset=COLUMNAS_ENTRADA + ["puntual"])

    X = df[COLUMNAS_ENTRADA].astype(float)
    y = df["puntual"].astype(int)

    return X, y


def obtener_ruta_csv(cid=0, distributor_id=None):
    """
    Busca el CSV del cliente.

    Prioridad:
    1. Si hay distributor_id: data/distribuidor_N.csv
    2. Si no: data/distribuidor_{cid + 1}.csv
    3. Si no existe: primer data/distribuidor_*.csv
    """

    if distributor_id is not None:
        ruta = os.path.join("data", f"distribuidor_{int(distributor_id)}.csv")
        if os.path.exists(ruta):
            return ruta

    ruta_cliente = os.path.join("data", f"distribuidor_{int(cid) + 1}.csv")
    if os.path.exists(ruta_cliente):
        return ruta_cliente

    archivos_csv = sorted(glob.glob(os.path.join("data", "distribuidor_*.csv")))

    if not archivos_csv:
        raise FileNotFoundError(
            "No se encontró ningún archivo con formato data/distribuidor_*.csv"
        )

    return archivos_csv[0]


def load_data(batch_size, cid=0, num_clients=2, distributor_id=None):
    ruta_csv = obtener_ruta_csv(cid=cid, distributor_id=distributor_id)

    print(f"Cliente {cid} usando CSV: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    if distributor_id is not None and "distribuidor_id" in df.columns:
        distribuidor = f"D{int(distributor_id)}"
        df_filtrado = df[df["distribuidor_id"] == distribuidor]

        if not df_filtrado.empty:
            df = df_filtrado

    if len(df) < 5:
        raise ValueError(
            f"Hay muy pocos datos para entrenar este cliente en {ruta_csv}."
        )

    X, y = preparar_datos(df)

    if len(X) < 5:
        raise ValueError(
            f"No quedan suficientes filas válidas después de preparar los datos en {ruta_csv}."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.long),
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    return trainloader, testloader, len(train_dataset)


def train(net, trainloader, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    net.train()

    for _ in range(5):
        for features, labels in trainloader:
            optimizer.zero_grad()

            outputs = net(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()

    net.eval()

    total_loss = 0.0
    num_batches = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, labels in testloader:
            outputs = net(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

            num_batches += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    avg_loss = total_loss / max(num_batches, 1)

    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = 0.0

    puntual_real_medio = float(np.mean(y_true))
    puntual_predicho_medio = float(np.mean(y_pred))

    diferencia_media = abs(
        puntual_real_medio - puntual_predicho_medio
    )

    return float(avg_loss), float(accuracy), len(y_true), {
        "accuracy": float(accuracy),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),

        # Se mantienen estos nombres para que tu dashboard los pueda reutilizar
        "retraso_real_medio": puntual_real_medio,
        "retraso_predicho_medio": puntual_predicho_medio,
        "diferencia_media": float(diferencia_media),
    }