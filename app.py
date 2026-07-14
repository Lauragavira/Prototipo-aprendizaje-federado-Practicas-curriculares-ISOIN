import json
import os
import subprocess
import time

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Panel de Aprendizaje Federado", layout="wide")
st.title("🌸 Panel de Control: Entrenamiento Federado")


ALL_METRICS = ["loss", "accuracy", "mae", "rmse", "r2"]

METRICAS_PREDETERMINADAS = {
    "task_mnist": ["loss", "accuracy"],
    "task_noIID_mnist": ["loss", "accuracy"],
    "task_logistica": ["mae", "rmse", "r2"],
}

METRICAS_DISPONIBLES = {
    "task_mnist": {"loss", "accuracy"},
    "task_noIID_mnist": {"loss", "accuracy"},
    # En regresión, Flower usa MAE como loss principal.
    "task_logistica": {"loss", "mae", "rmse", "r2"},
}

CONFIG_METRICAS = {
    "loss": {
        "nombre": "Loss",
        "kpi": "Pérdida global (Loss)",
        "titulo": "📉 Evolución de la pérdida",
        "formato": ".4f",
        "delta_color": "inverse",
        "color": "#ff4b4b",
    },
    "accuracy": {
        "nombre": "Accuracy",
        "kpi": "Precisión global (Accuracy)",
        "titulo": "🎯 Evolución de la precisión",
        "formato": ".2%",
        "delta_color": "normal",
        "color": "#21c354",
    },
    "mae": {
        "nombre": "MAE",
        "kpi": "MAE medio",
        "titulo": "📉 Evolución del MAE",
        "formato": ".2f",
        "sufijo": " min",
        "delta_color": "inverse",
        "color": "#ff4b4b",
    },
    "rmse": {
        "nombre": "RMSE",
        "kpi": "RMSE medio",
        "titulo": "📊 Evolución del RMSE",
        "formato": ".2f",
        "sufijo": " min",
        "delta_color": "inverse",
        "color": "#ff9f1c",
    },
    "r2": {
        "nombre": "R²",
        "kpi": "R² global",
        "titulo": "🎯 Evolución del R²",
        "formato": ".4f",
        "delta_color": "normal",
        "color": "#21c354",
    },
}


if "server_process" not in st.session_state:
    st.session_state.server_process = None

if "client_processes" not in st.session_state:
    st.session_state.client_processes = []


with st.sidebar:
    st.header("📦 Definición del Problema")

    tarea = st.selectbox(
        "Tarea a ejecutar",
        ["task_mnist", "task_noIID_mnist", "task_logistica"],
        key="task_select",
    )

    st.caption("Añade más archivos 'task_algo.py' y ponlos en esta lista.")

    st.subheader("📏 Métricas a medir")
    selected_metrics = []

    for metrica in ALL_METRICS:
        marcada = st.checkbox(
            CONFIG_METRICAS[metrica]["nombre"],
            value=metrica in METRICAS_PREDETERMINADAS[tarea],
            key=f"metric_{tarea}_{metrica}",
        )
        if marcada:
            selected_metrics.append(metrica)

    metricas_no_disponibles = [
        metrica
        for metrica in selected_metrics
        if metrica not in METRICAS_DISPONIBLES[tarea]
    ]

    if not selected_metrics:
        st.warning("Selecciona al menos una métrica.")
    elif metricas_no_disponibles:
        nombres_no_disponibles = ", ".join(
            CONFIG_METRICAS[metrica]["nombre"]
            for metrica in metricas_no_disponibles
        )
        st.warning(
            f"{nombres_no_disponibles} no se calcula en {tarea} y no tendrá gráfica."
        )

    st.header("⚙️ Parámetros del Servidor")

    num_rounds = st.slider(
        "Número de Rondas",
        1,
        50,
        5,
        key="rounds_slider",
    )

    selected_distributors = []
    distributor_selection_valid = True

    if tarea == "task_logistica":
        data_distribution = "No-IID real"
        st.info(
            "Esta tarea usa los CSV reales de distribuidores. "
            "Selecciona libremente 2, 3 o 4 distribuidores."
        )

        selected_distributors = st.multiselect(
            "Distribuidores a comparar",
            options=[1, 2, 3, 4],
            default=[1, 2],
            max_selections=4,
            format_func=lambda distributor_id: (
                f"Distribuidor {distributor_id}"
            ),
            key="selected_distributors",
        )
        selected_distributors = [
            int(distributor_id)
            for distributor_id in selected_distributors
        ]

        # En logística, cada distribuidor seleccionado es un cliente Flower.
        min_clients = len(selected_distributors)
        distributor_selection_valid = 2 <= min_clients <= 4

        if not distributor_selection_valid:
            st.warning("Selecciona entre 2 y 4 distribuidores.")
        else:
            selected_text = ", ".join(
                f"D{distributor_id}"
                for distributor_id in selected_distributors
            )
            st.success(
                f"Clientes seleccionados: {min_clients}. "
                f"Participarán: {selected_text}"
            )
    else:
        min_clients = int(
            st.selectbox(
                "Clientes a simular",
                options=[2, 3, 4],
                index=0,
                key="clients_input",
                help=(
                    "El simulador requiere un mínimo de 2 clientes y "
                    "permite un máximo de 4."
                ),
            )
        )

        if tarea == "task_noIID_mnist":
            data_distribution = "No-IID"
            st.info(
                "Esta tarea usa datos No-IID: cada cliente recibe "
                "clases distintas."
            )
        else:
            data_distribution = "IID"
            st.info(
                "Esta tarea usa datos IID: los clientes reciben datos "
                "equilibrados."
            )

    privacy_budget = st.selectbox(
        "Presupuesto de privacidad ε",
        [
            "Sin DP",
            "ε = 20.0  → privacidad muy baja",
            "ε = 10.0  → privacidad baja",
            "ε = 5.0   → privacidad media",
            "ε = 1.0   → privacidad alta",
            "ε = 0.5   → privacidad muy alta",
            "ε = 0.1   → privacidad extrema",
        ],
        key="epsilon_select",
    )

    epsilon_values = {
        "Sin DP": None,
        "ε = 20.0  → privacidad muy baja": 20.0,
        "ε = 10.0  → privacidad baja": 10.0,
        "ε = 5.0   → privacidad media": 5.0,
        "ε = 1.0   → privacidad alta": 1.0,
        "ε = 0.5   → privacidad muy alta": 0.5,
        "ε = 0.1   → privacidad extrema": 0.1,
    }

    epsilon_to_noise = {
        None: 0.0,
        20.0: 0.001,
        10.0: 0.005,
        5.0: 0.02,
        1.0: 0.08,
        0.5: 0.15,
        0.1: 0.30,
    }

    privacy_epsilon = epsilon_values[privacy_budget]
    dp_noise = epsilon_to_noise[privacy_epsilon]

    st.caption("Cuanto menor es ε, mayor privacidad y más ruido añadido.")

    porcentaje = st.slider(
        "Participación por ronda (%)",
        10,
        100,
        100,
        step=10,
        key="fraction_slider",
    )
    fraction_val = porcentaje / 100.0

    st.header("🧮 Estrategia de Agregación")

    estrategia = st.selectbox(
        "Algoritmo",
        ["FedAvg", "FedProx", "FedMedian"],
        key="strategy_select",
    )

    mu_val = 0.0
    if estrategia == "FedProx":
        mu_val = st.number_input(
            "Término Proximal (μ)",
            0.0,
            5.0,
            0.1,
            step=0.1,
            key="mu_input",
        )

    st.header("🧠 Hiperparámetros Locales")

    lr = st.selectbox(
        "Learning Rate",
        [0.1, 0.01, 0.001],
        index=1,
        key="lr_select",
    )

    batch = st.select_slider(
        "Batch Size",
        options=[8, 16, 32, 64],
        value=32,
        key="batch_select",
    )

    st.divider()
    start_button = st.button(
        "🚀 Iniciar Entrenamiento Automático",
        width="stretch",
        disabled=(
            not selected_metrics
            or not distributor_selection_valid
        ),
    )


st.subheader("📊 Métricas de la Última Ronda")

kpi_placeholders = {}
if selected_metrics:
    columnas_kpi = st.columns(len(selected_metrics))
    for columna, metrica in zip(columnas_kpi, selected_metrics):
        kpi_placeholders[metrica] = columna.empty()
else:
    st.info("Marca al menos una métrica en el panel lateral.")

st.divider()
placeholder_tiempo = st.empty()

grafica_placeholders = {
    metrica: st.empty()
    for metrica in selected_metrics
}

# Este contenedor se usa únicamente en task_logistica.
panel_distribuidores = st.empty()


def formatear_valor(metrica, valor):
    configuracion = CONFIG_METRICAS[metrica]
    texto = format(valor, configuracion["formato"])
    return f"{texto}{configuracion.get('sufijo', '')}"


def pintar_panel_distribuidores(data, key_suffix):
    """Muestra los resultados individuales de la última ronda logística."""

    if data.get("task") != "task_logistica":
        return

    historial = data.get("distribuidores", [])
    if not historial:
        return

    ultima_ronda = historial[-1]
    resultados = ultima_ronda.get("resultados", [])

    if not resultados:
        return

    df = pd.DataFrame(resultados)

    # Seguridad adicional: aunque el servidor ya guarda únicamente los
    # distribuidores elegidos, filtramos por la selección de esta ejecución.
    configured_distributors = {
        int(distributor_id)
        for distributor_id in data.get("selected_distributors", [])
    }
    if configured_distributors and "distribuidor" in df.columns:
        df = df[
            df["distribuidor"].isin(configured_distributors)
        ].copy()

    if df.empty:
        return

    columnas_numericas = [
        "mae",
        "rmse",
        "r2",
        "retraso_real_medio",
        "retraso_predicho_medio",
        "diferencia_media",
    ]
    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    columnas_tabla = [
        "distribuidor",
        "mae",
        "rmse",
        "r2",
        "retraso_real_medio",
        "retraso_predicho_medio",
        "diferencia_media",
    ]
    columnas_tabla = [
        columna for columna in columnas_tabla
        if columna in df.columns
    ]

    nombres_columnas = {
        "distribuidor": "Distribuidor",
        "mae": "MAE (min)",
        "rmse": "RMSE (min)",
        "r2": "R²",
        "retraso_real_medio": "Retraso real medio (min)",
        "retraso_predicho_medio": "Retraso predicho medio (min)",
        "diferencia_media": "Diferencia media (min)",
    }

    panel_distribuidores.empty()

    with panel_distribuidores.container():
        st.divider()
        st.subheader("🚚 Resultados por distribuidor")
        selected_text = ", ".join(
            f"Distribuidor {int(distributor_id)}"
            for distributor_id in sorted(df["distribuidor"].tolist())
        )
        st.caption(
            f"Resultados individuales de la ronda "
            f"{ultima_ronda.get('round', '-')}: {selected_text}"
        )

        tabla = df[columnas_tabla].rename(
            columns=nombres_columnas
        )
        st.dataframe(
            tabla,
            width="stretch",
            hide_index=True,
        )

        if (
            "diferencia_media" in df.columns
            and not df["diferencia_media"].dropna().empty
        ):
            indice_peor = df["diferencia_media"].idxmax()
            peor = df.loc[indice_peor]

            st.warning(
                f"El distribuidor {int(peor['distribuidor'])} "
                f"presenta la mayor diferencia media: "
                f"{peor['diferencia_media']:.2f} minutos."
            )

        columna_1, columna_2 = st.columns(2)

        with columna_1:
            if (
                "mae" in df.columns
                and not df["mae"].dropna().empty
            ):
                figura_mae = px.bar(
                    df.dropna(subset=["mae"]),
                    x="distribuidor",
                    y="mae",
                    text_auto=".2f",
                    title="MAE por distribuidor",
                    labels={
                        "distribuidor": "Distribuidor",
                        "mae": "MAE (minutos)",
                    },
                )
                figura_mae.update_layout(
                    xaxis=dict(
                        tickmode="linear",
                        dtick=1,
                    )
                )
                st.plotly_chart(
                    figura_mae,
                    width="stretch",
                    key=f"mae_distribuidor_{key_suffix}",
                )

        with columna_2:
            columnas_comparacion = {
                "retraso_real_medio",
                "retraso_predicho_medio",
            }

            if columnas_comparacion.issubset(df.columns):
                comparacion = df[
                    [
                        "distribuidor",
                        "retraso_real_medio",
                        "retraso_predicho_medio",
                    ]
                ].melt(
                    id_vars="distribuidor",
                    value_vars=[
                        "retraso_real_medio",
                        "retraso_predicho_medio",
                    ],
                    var_name="tipo",
                    value_name="retraso",
                )

                comparacion["tipo"] = comparacion["tipo"].map(
                    {
                        "retraso_real_medio": "Retraso real",
                        "retraso_predicho_medio": "Retraso predicho",
                    }
                )

                comparacion = comparacion.dropna(
                    subset=["retraso"]
                )

                if not comparacion.empty:
                    figura_comparacion = px.bar(
                        comparacion,
                        x="distribuidor",
                        y="retraso",
                        color="tipo",
                        barmode="group",
                        text_auto=".2f",
                        title="Retraso real frente al predicho",
                        labels={
                            "distribuidor": "Distribuidor",
                            "retraso": "Retraso medio (minutos)",
                            "tipo": "Valor",
                        },
                    )
                    figura_comparacion.update_layout(
                        xaxis=dict(
                            tickmode="linear",
                            dtick=1,
                        )
                    )
                    st.plotly_chart(
                        figura_comparacion,
                        width="stretch",
                        key=f"retrasos_distribuidor_{key_suffix}",
                    )


def pintar_metricas(data, key_suffix):
    rondas = data.get("round", [])
    if not rondas:
        return 0

    columnas_permitidas = {"round", "time", *CONFIG_METRICAS.keys()}
    columnas = {
        clave: valor
        for clave, valor in data.items()
        if (
            clave in columnas_permitidas
            and isinstance(valor, list)
            and len(valor) == len(rondas)
        )
    }
    df = pd.DataFrame(columnas)

    if "time" in df.columns and not df["time"].dropna().empty:
        tiempos = df["time"].dropna().tolist()
        duraciones_ronda = [tiempos[0]] + [
            tiempos[i] - tiempos[i - 1]
            for i in range(1, len(tiempos))
        ]
        tiempo_medio = sum(duraciones_ronda) / len(duraciones_ronda)
        placeholder_tiempo.metric(
            label="⌛ Tiempo medio por ronda",
            value=f"{tiempo_medio:.2f} segundos",
        )

    for metrica in selected_metrics:
        if metrica not in df.columns:
            continue

        df_metrica = df.dropna(subset=[metrica])
        if df_metrica.empty:
            continue

        configuracion = CONFIG_METRICAS[metrica]
        ultimo_valor = float(df_metrica[metrica].iloc[-1])
        delta = (
            ultimo_valor - float(df_metrica[metrica].iloc[-2])
            if len(df_metrica) > 1
            else 0.0
        )

        kpi_placeholders[metrica].metric(
            label=configuracion["kpi"],
            value=formatear_valor(metrica, ultimo_valor),
            delta=formatear_valor(metrica, delta),
            delta_color=configuracion["delta_color"],
        )

        figura = px.line(
            df_metrica,
            x="round",
            y=metrica,
            markers=True,
            title=configuracion["titulo"],
            template="plotly_white",
        )
        figura.update_traces(
            line=dict(width=3, color=configuracion["color"]),
            marker=dict(size=8),
        )
        figura.update_layout(
            xaxis_title="Ronda",
            yaxis_title=configuracion["nombre"],
        )

        grafica_placeholders[metrica].plotly_chart(
            figura,
            width="stretch",
            key=f"{metrica}_{key_suffix}",
        )

    pintar_panel_distribuidores(data, key_suffix)

    return len(rondas)


if start_button:
    if not selected_metrics:
        st.error("Debes seleccionar al menos una métrica.")
        st.stop()

    if tarea == "task_logistica" and not distributor_selection_valid:
        st.error("Debes seleccionar entre 2 y 4 distribuidores.")
        st.stop()

    if (
        st.session_state.server_process is None
        or st.session_state.server_process.poll() is not None
    ):
        ruta_absoluta_config = os.path.abspath("run_config.json")
        ruta_absoluta_metrics = os.path.abspath("metrics.json")

        if os.path.exists(ruta_absoluta_metrics):
            os.remove(ruta_absoluta_metrics)

        config_data = {
            "rounds": num_rounds,
            "min_clients": int(min_clients),
            "lr": lr,
            "batch_size": batch,
            "fraction": fraction_val,
            "strategy": estrategia,
            "mu": mu_val,
            "task": tarea,
            "selected_metrics": selected_metrics,
            "data_distribution": data_distribution,
            "privacy_budget": privacy_budget,
            "privacy_epsilon": privacy_epsilon,
            "dp_noise": dp_noise,
            # En las tareas MNIST queda vacío. En logística contiene los
            # identificadores reales de los distribuidores seleccionados.
            "selected_distributors": selected_distributors,
        }

        with open(ruta_absoluta_config, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        entorno = os.environ.copy()
        entorno["FLWR_RUN_CONFIG_PATH"] = ruta_absoluta_config
        entorno["FLWR_METRICS_PATH"] = ruta_absoluta_metrics
        entorno["FLWR_TASK_NAME"] = tarea
        entorno["PYTHONPATH"] = os.getcwd() + os.pathsep + entorno.get("PYTHONPATH", "")

        flwr_exe = r"C:\Flower\.venv\Scripts\flwr.exe"

        if not os.path.exists(flwr_exe):
            st.error(f"No se encuentra Flower en esta ruta: {flwr_exe}")
            st.stop()

        comando_servidor = [flwr_exe, "run", ".", "--stream"]

        st.session_state.server_process = subprocess.Popen(
            comando_servidor,
            env=entorno,
        )

        nombres_metricas = ", ".join(
            CONFIG_METRICAS[metrica]["nombre"]
            for metrica in selected_metrics
        )
        st.toast(
            f"🚀 Iniciando {tarea}: {nombres_metricas}",
            icon="🌸",
        )

        ultimo_round_dibujado = 0

        while st.session_state.server_process.poll() is None:
            try:
                with open(ruta_absoluta_metrics, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if len(data.get("round", [])) > ultimo_round_dibujado:
                    ultimo_round_dibujado = pintar_metricas(
                        data,
                        key_suffix=f"live_{len(data['round'])}",
                    )
            except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
                pass

            time.sleep(1)

        try:
            with open(ruta_absoluta_metrics, "r", encoding="utf-8") as f:
                data = json.load(f)
            pintar_metricas(data, key_suffix="final")
        except Exception as error:
            st.warning(f"No se pudieron cargar las métricas finales: {error}")

        st.session_state.server_process = None
        st.success("✨ Entrenamiento finalizado.")