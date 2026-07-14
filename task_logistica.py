# task_logistica.py
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "retraso_min"

NUMERIC_FEATURES = [
    "hora_salida_min",
    "distancia_km",
    "numero_entregas",
    "peso_total_kg",
    "duracion_estimada_min",
    "hora_llegada_estimada_min",
]

CATEGORICAL_FEATURES = [
    "dia_semana",
    "zona_origen",
    "trafico",
    "clima",
]

DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado"]
TRAFICO = ["bajo", "medio", "alto"]
CLIMA = ["soleado", "nublado", "lluvia"]

ZONAS = [
    "Nervion",
    "Sevilla Este",
    "Alcala de Guadaira",
    "Torreblanca",
    "San Pablo",
    "Triana",
    "Los Remedios",
    "Camas",
    "Dos Hermanas",
    "Montequinto",
    "Pino Montano",
    "San Jeronimo",
    "Macarena",
    "Alamillo",
    "Cartuja",
    "La Rinconada",
    "Santiponce",
    "Aeropuerto",
    "Parque Alcosa",
]

FEATURES = (
    NUMERIC_FEATURES
    + [f"dia_semana_{x}" for x in DIAS]
    + [f"zona_origen_{x}" for x in ZONAS]
    + [f"trafico_{x}" for x in TRAFICO]
    + [f"clima_{x}" for x in CLIMA]
)


def preprocess_dataframe(df):
    df = df.copy()

    df["hora_salida_min"] = df["hora_salida_min"] / 1440
    df["distancia_km"] = df["distancia_km"] / 30
    df["numero_entregas"] = df["numero_entregas"] / 20
    df["peso_total_kg"] = df["peso_total_kg"] / 350
    df["duracion_estimada_min"] = df["duracion_estimada_min"] / 150
    df["hora_llegada_estimada_min"] = df["hora_llegada_estimada_min"] / 1440

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES)
    X = X.reindex(columns=FEATURES, fill_value=0)

    y = df[TARGET]

    return X.values.astype(np.float32), y.values.astype(np.float32)


def get_model():
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="constant",
        eta0=0.01,
        max_iter=2,
        warm_start=True,
        random_state=42,
    )

    model.coef_ = np.zeros(len(FEATURES), dtype=np.float32)
    model.intercept_ = np.zeros(1, dtype=np.float32)
    model.n_features_in_ = len(FEATURES)

    return model


def get_model_params(model):
    return [model.coef_, model.intercept_]


def set_model_params(model, parameters):
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.n_features_in_ = len(FEATURES)
    return model


def load_data(
    batch_size=32,
    cid=0,
    num_clients=4,
    distributor_id=None,
):
    # distributor_id permite elegir, por ejemplo, los distribuidores 2 y 4
    # aunque los clientes virtuales de Flower tengan los identificadores 0 y 1.
    if distributor_id is None:
        distributor_id = int(cid) + 1
    else:
        distributor_id = int(distributor_id)

    if distributor_id not in {1, 2, 3, 4}:
        raise ValueError(
            f"Distribuidor no válido: {distributor_id}. "
            "Debe estar entre 1 y 4."
        )

    csv_path = Path("distribuidores") / f"distribuidor_{distributor_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)

    X, y = preprocess_dataframe(df)

    split_idx = int(0.8 * len(X))

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    return train_data, test_data, len(X_train)


def train(model, train_data, lr):
    X_train, y_train = train_data

    model.eta0 = float(lr)

    model.fit(X_train, y_train)


def test(model, test_data):
    X_test, y_test = test_data

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    retraso_real_medio = float(np.mean(y_test))
    retraso_predicho_medio = float(np.mean(y_pred))

    # Diferencia absoluta entre ambas medias.
    diferencia_media = abs(
        retraso_real_medio - retraso_predicho_medio
    )

    # Flower necesita un valor principal de loss; para regresión usamos el MAE.
    return float(mae), float(r2), len(X_test), {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "retraso_real_medio": retraso_real_medio,
        "retraso_predicho_medio": retraso_predicho_medio,
        "diferencia_media": float(diferencia_media),
    }