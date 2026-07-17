# task_regresion.py

import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


COLUMNAS_ENTRADA = [
    "hora_salida_min",
    "distancia_km",
    "numero_entregas",
    "peso_total_kg",
    "duracion_estimada_min",
    "hora_llegada_estimada_min",
]

COLUMNA_OBJETIVO = "retraso_min"
INPUT_DIM = len(COLUMNAS_ENTRADA)

# Escalado fijo y compartido por todos los clientes. En aprendizaje federado
# no conviene ajustar un StandardScaler diferente en cada distribuidor, porque
# los mismos pesos globales representarían escalas distintas.
CENTROS_ENTRADA = np.array([720.0, 15.0, 10.0, 175.0, 75.0, 720.0], dtype=np.float32)
ESCALAS_ENTRADA = np.array([360.0, 10.0, 5.0, 100.0, 30.0, 360.0], dtype=np.float32)

# El archivo está dentro de task/, por lo que la raíz del proyecto es su carpeta padre.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LinearRegressionModel(nn.Module):
    """Regresión lineal para predecir minutos de retraso."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def get_model():
    return LinearRegressionModel()


def preparar_datos(df):
    """Convierte entradas y objetivo a valores numéricos y elimina filas inválidas."""

    df = df.copy()

    columnas_necesarias = COLUMNAS_ENTRADA + [COLUMNA_OBJETIVO]
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            + ", ".join(columnas_faltantes)
        )

    for columna in columnas_necesarias:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = df.dropna(subset=columnas_necesarias)

    X = df[COLUMNAS_ENTRADA].astype(np.float32)
    y = df[COLUMNA_OBJETIVO].astype(np.float32)

    return X, y


def obtener_ruta_csv(cid=0, distributor_id=None):
    """Busca el CSV asociado al distribuidor.

    Se admiten las carpetas ``data/`` y ``distribuidores/`` para mantener
    compatibilidad con ambas estructuras del proyecto.
    """

    distribuidor = int(distributor_id) if distributor_id is not None else int(cid) + 1
    nombre_archivo = f"distribuidor_{distribuidor}.csv"

    carpetas_candidatas = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "distribuidores"),
        os.path.abspath("data"),
        os.path.abspath("distribuidores"),
    ]

    for carpeta in carpetas_candidatas:
        ruta = os.path.join(carpeta, nombre_archivo)
        if os.path.exists(ruta):
            return ruta

    # Fallback: primer CSV disponible, evitando que una diferencia de carpetas
    # impida arrancar el cliente.
    for carpeta in carpetas_candidatas:
        archivos_csv = sorted(glob.glob(os.path.join(carpeta, "distribuidor_*.csv")))
        if archivos_csv:
            return archivos_csv[0]

    raise FileNotFoundError(
        "No se encontró ningún archivo distribuidor_*.csv en las carpetas "
        "data/ o distribuidores/."
    )


def load_data(batch_size, cid=0, num_clients=2, distributor_id=None):
    del num_clients  # Se mantiene en la firma por compatibilidad con client.py.

    ruta_csv = obtener_ruta_csv(cid=cid, distributor_id=distributor_id)
    print(f"Cliente {cid} usando CSV: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    if distributor_id is not None and "distribuidor_id" in df.columns:
        distribuidor = f"D{int(distributor_id)}"
        df_filtrado = df[df["distribuidor_id"].astype(str) == distribuidor]
        if not df_filtrado.empty:
            df = df_filtrado

    if len(df) < 5:
        raise ValueError(
            f"Hay muy pocos datos para entrenar este cliente en {ruta_csv}."
        )

    X, y = preparar_datos(df)

    if len(X) < 5:
        raise ValueError(
            "No quedan suficientes filas válidas después de preparar los "
            f"datos en {ruta_csv}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Todos los distribuidores aplican exactamente la misma transformación.
    X_train_scaled = (X_train.to_numpy(dtype=np.float32) - CENTROS_ENTRADA) / ESCALAS_ENTRADA
    X_test_scaled = (X_test.to_numpy(dtype=np.float32) - CENTROS_ENTRADA) / ESCALAS_ENTRADA

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32),
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    return trainloader, testloader, len(train_dataset)


def train(net, trainloader, lr):
    """Entrena localmente el modelo de regresión."""

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=float(lr), momentum=0.9)

    net.train()

    for _ in range(5):
        for features, targets in trainloader:
            optimizer.zero_grad()

            predictions = net(features)
            loss = criterion(predictions, targets)

            loss.backward()
            # Evita gradientes excesivos cuando se selecciona un learning rate alto.
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()


def test(net, testloader):
    """Evalúa MAE, MSE, RMSE, R² y retrasos medios."""

    criterion = nn.MSELoss(reduction="sum")
    net.eval()

    total_squared_error = 0.0
    total_examples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, targets in testloader:
            predictions = net(features)

            total_squared_error += criterion(predictions, targets).item()
            total_examples += targets.size(0)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    if total_examples == 0:
        raise ValueError("El conjunto de prueba está vacío.")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # R² puede no estar definido si el conjunto de prueba tiene menos de dos
    # muestras o si todos los valores reales son iguales.
    if len(y_true) >= 2 and not np.allclose(y_true, y_true[0]):
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = 0.0

    retraso_real_medio = float(np.mean(y_true))
    retraso_predicho_medio = float(np.mean(y_pred))
    sesgo_medio = retraso_predicho_medio - retraso_real_medio
    diferencia_media = abs(sesgo_medio)

    avg_loss = total_squared_error / total_examples

    # El segundo valor se conserva por compatibilidad con la interfaz existente.
    return float(avg_loss), float(r2), int(total_examples), {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "retraso_real_medio": retraso_real_medio,
        "retraso_predicho_medio": retraso_predicho_medio,
        "diferencia_media": float(diferencia_media),
        # Positivo: sobreestima el retraso. Negativo: lo infraestima.
        "sesgo_medio": float(sesgo_medio),
    }