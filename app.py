import sys
import streamlit as st
import subprocess
import json
import pandas as pd
import numpy as np
import os
import plotly.express as px
import glob
import socket
import shutil

st.set_page_config(page_title="Panel de Aprendizaje Federado", layout="wide")
st.title("🌸 Panel de Control: Entrenamiento Federado")

# --- FUNCIÓN AUXILIAR PARA OBTENER LA IP DE TU MÁQUINA ---
def obtener_ip_local():
    try:
        # Se abre una conexión UDP temporal para detectar la interfaz de red activa
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# --- INICIALIZACIÓN DE PROCESOS ---
# Ahora controlamos dos procesos independientes: el SuperLink (red) y la ServerApp (entrenamiento)
if "superlink_process" not in st.session_state:
    st.session_state.superlink_process = None

if "server_app_process" not in st.session_state:
    st.session_state.server_app_process = None

# Comprobamos los estados de ambos procesos de forma asíncrona
superlink_activo = False
if st.session_state.superlink_process is not None:
    if st.session_state.superlink_process.poll() is None:
        superlink_activo = True
    else:
        st.session_state.superlink_process = None

entrenamiento_en_curso = False
if st.session_state.server_app_process is not None:
    if st.session_state.server_app_process.poll() is None:
        entrenamiento_en_curso = True
    else:
        st.session_state.server_app_process = None

ALL_METRICS = ["loss", "accuracy", "mae", "mse", "rmse", "r2"]
METRICAS_PREDETERMINADAS = {
    "task_mnist": ["loss", "accuracy"],
    "task_noIID_mnist": ["loss", "accuracy"],
    "task_logistica": ["mae", "rmse", "r2"],
    "task_regresion": ["mae", "rmse", "r2"],
}
METRICAS_DISPONIBLES = {
    "task_mnist": {"loss", "accuracy", "rmse", "r2"},
    "task_noIID_mnist": {"loss", "accuracy"},
    "task_logistica": {"loss", "mae", "mse", "rmse", "r2"},
    "task_regresion": {"loss", "mae", "mse", "rmse", "r2"},
}

CONFIG_METRICAS = {
    "loss": {"nombre": "Loss", "kpi": "Pérdida global (Loss)", "titulo": "📉 Evolución de la pérdida", "formato": ".4f", "delta_color": "inverse", "color": "#ff4b4b"},
    "accuracy": {"nombre": "Accuracy", "kpi": "Precisión global", "titulo": "🎯 Evolución de la precisión", "formato": ".2%", "delta_color": "normal", "color": "#21c354"},
    "mae": {"nombre": "MAE", "kpi": "MAE medio", "titulo": "📉 Evolución del MAE", "formato": ".2f", "sufijo": " min", "delta_color": "inverse", "color": "#ff4b4b"},
    "mse": {"nombre": "MSE", "kpi": "MSE medio", "titulo": "📉 Evolución del MSE", "formato": ".2f", "sufijo": " min²", "delta_color": "inverse", "color": "#7b2cbf"},
    "rmse": {"nombre": "RMSE", "kpi": "RMSE medio", "titulo": "📊 Evolución del RMSE", "formato": ".2f", "sufijo": " min", "delta_color": "inverse", "color": "#ff9f1c"},
    "r2": {"nombre": "R²", "kpi": "R² global", "titulo": "🎯 Evolución del R²", "formato": ".4f", "delta_color": "normal", "color": "#21c354"},
}

# --- BARRA LATERAL ---
with st.sidebar:
    # Mostrar datos de conexión para las otras portátiles
    ip_servidor = obtener_ip_local()
    st.success(f"🖥️ **Tu IP Local:** `{ip_servidor}`")
    
    st.info(
        f"**Comando de conexión:**\n\n"
        f"```bash\n"
        f"flower-supernode --insecure --superlink {ip_servidor}:9092\n"
        f"```"
    )
    st.divider()

    st.header("🌐 Control de Red (SuperLink)")
    # Gestión del proceso SuperLink en el puerto 9092
    if not superlink_activo:
        start_superlink = st.button("🌐 Levantar SuperLink", width='stretch')
        stop_superlink = False
    else:
        st.write("🟢 SuperLink en ejecución (Puerto 9092)")
        stop_superlink = st.button("🛑 Detener SuperLink", type="primary", width='stretch')
        start_superlink = False

    st.divider()

    st.header("📦 Definición del Problema")
    ruta_tareas_carpeta = os.path.join("task", "*.py")
    archivos_tarea = glob.glob(ruta_tareas_carpeta)
    
    nombres_tarea = sorted(list(set([
        os.path.basename(f).replace(".py", "") 
        for f in archivos_tarea 
        if not os.path.basename(f).startswith("__")
    ])))
    
    if not nombres_tarea:
        nombres_tarea = ["task_mnist", "task_noIID_mnist", "task_logistica"]
    
    tarea = st.selectbox("Tarea a ejecutar", nombres_tarea, key="task_select")
    
    st.subheader("Métricas a mostrar")
    selected_metrics = []
    predeterminadas = METRICAS_PREDETERMINADAS.get(tarea, ["loss", "accuracy"])
    disponibles = METRICAS_DISPONIBLES.get(tarea, set(ALL_METRICS))

    for m in ALL_METRICS:
        if m not in disponibles:
            continue
        marcada = st.checkbox(
            CONFIG_METRICAS[m]["nombre"],
            value=(m in predeterminadas),
            key=f"chk_{tarea}_{m}",
        )
        if marcada:
            selected_metrics.append(m)

    st.header("⚙️ Parámetros del Servidor")
    num_rounds = st.slider("Número de Rondas", 1, 50, 5)
    usar_checkpoints = st.checkbox("💾 Usar checkpoints (Reanudar)", value=False)

    selected_distributors = []
    distributor_selection_valid = True

    if tarea in ["task_logistica", "task_regresion"]:
        data_distribution = "No-IID real"

        selected_distributors = st.multiselect(
            "Distribuidores a comparar",
            [1, 2, 3, 4],
            default=[1, 2],
            format_func=lambda x: f"Distribuidor {x}",
        )

        min_clients = len(selected_distributors)
        distributor_selection_valid = 2 <= min_clients <= 4

        if not distributor_selection_valid:
            st.warning("Selecciona entre 2 y 4 distribuidores.")
    else:
        min_clients = st.selectbox("Clientes esperados", [2, 3, 4], index=0)
        data_distribution = "No-IID" if "noIID" in tarea else "IID"

    # --- Privacidad Diferencial ---
    st.subheader("🛡️ Privacidad Diferencial")
    privacy_budget = st.selectbox(
        "Presupuesto de privacidad ε",
        ["Sin DP", "ε = 20.0 (Baja)", "ε = 10.0", "ε = 5.0 (Media)", "ε = 1.0 (Alta)", "ε = 0.5", "ε = 0.1 (Extrema)"]
    )
    epsilon_values = {"Sin DP": None, "ε = 20.0 (Baja)": 20.0, "ε = 10.0": 10.0, "ε = 5.0 (Media)": 5.0, "ε = 1.0 (Alta)": 1.0, "ε = 0.5": 0.5, "ε = 0.1 (Extrema)": 0.1}
    epsilon_to_noise = {None: 0.0, 20.0: 0.001, 10.0: 0.005, 5.0: 0.02, 1.0: 0.08, 0.5: 0.15, 0.1: 0.30}
    privacy_epsilon = epsilon_values[privacy_budget]
    dp_noise = epsilon_to_noise[privacy_epsilon]

    porcentaje = st.slider("Participación por ronda (%)", 10, 100, 100, step=10)
    fraction_val = porcentaje / 100.0

    st.header("🧮 Estrategia")
    estrategia = st.selectbox("Algoritmo", ["FedAvg", "FedProx", "FedMedian"])
    mu_val = st.number_input("Término Proximal (μ)", 0.0, 5.0, 0.1, step=0.1) if estrategia == "FedProx" else 0.0

    st.header("🧠 Hiperparámetros Locales")
    lr = st.selectbox("Learning Rate", [0.1, 0.01, 0.001], index=1)
    batch = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32)

    st.divider()
    if entrenamiento_en_curso:
        stop_button = st.button("🛑 Detener Entrenamiento", type="primary", width='stretch')
        start_button = False
    else:
        # El entrenamiento solo se habilita si el SuperLink está activo
        start_button = st.button(
            "🚀 Iniciar ServerApp (Entrenamiento)", 
            width='stretch', 
            disabled=(not superlink_activo or not selected_metrics or not distributor_selection_valid)
        )
        stop_button = False

# --- GESTIÓN DE EXECUTABLES DE FLOWER ---
import sys
import shutil

def buscar_ejecutable(nombres_posibles):
    python_dir = os.path.dirname(sys.executable)
    for nombre in nombres_posibles:
        # Buscamos directamente en la carpeta de tu entorno
        ruta_directa = os.path.join(python_dir, f"{nombre}.exe")
        if os.path.exists(ruta_directa):
            return ruta_directa
        # Fallback al buscador global de tu sistema
        ruta_shutil = shutil.which(nombre)
        if ruta_shutil:
            return ruta_shutil
    return None

superlink_exe = buscar_ejecutable(["flower-superlink", "flwr-superlink"])
flwr_exe = buscar_ejecutable(["flwr"])

if not superlink_exe or not flwr_exe:
    st.error(
        f"❌ No se encontraron los ejecutables de Flower.\n\n"
        f"**SuperLink detectado:** `{superlink_exe}`\n"
        f"**flwr detectado:** `{flwr_exe}`"
    )
    st.stop()

# --- DETENCIÓN Y ARRANQUE DEL SUPERLINK ---
if start_superlink:
    st.session_state.superlink_process = subprocess.Popen(
    [
        superlink_exe,
        "--insecure",
        "--fleet-api-address", "0.0.0.0:9092",
        "--control-api-address", "0.0.0.0:9093",
    ]
    )
    st.toast("🌐 ¡SuperLink abierto! Clientes en 9092, ServerApp en 9091", icon="🌍")
    st.rerun()

if stop_superlink and st.session_state.superlink_process is not None:
    st.session_state.superlink_process.terminate()
    st.session_state.superlink_process = None
    st.toast("🛑 SuperLink detenido de forma segura.", icon="🔌")
    st.rerun()

# --- DETENER ENTRENAMIENTO (ServerApp) ---
ruta_txt_parada = os.path.abspath("stop_training.txt")
if stop_button and st.session_state.server_app_process is not None:
    st.toast("🛑 Enviando señal de parada a la ServerApp...", icon="⏳")
    with open(ruta_txt_parada, "w") as f:
        f.write("stop")
    st.rerun()

# --- ÁREA PRINCIPAL ---
st.subheader("📊 Métricas del Entrenamiento")


def _convertir_columnas_numericas(df, columnas):
    for columna in columnas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
    return df


def _fila_optima(df, metrica, maximizar=False):
    if metrica not in df.columns:
        return None
    df_valido = df.dropna(subset=[metrica])
    if df_valido.empty:
        return None
    indice = df_valido[metrica].idxmax() if maximizar else df_valido[metrica].idxmin()
    return df_valido.loc[indice]


def dibujar_resultados_distribuidores(data):
    historial = data.get("distribuidores", [])
    if not historial:
        return

    filas = []
    for resultado_ronda in historial:
        ronda = resultado_ronda.get("round")
        for resultado in resultado_ronda.get("resultados", []):
            fila = dict(resultado)
            fila["round"] = ronda
            filas.append(fila)

    if not filas:
        return

    df_hist = pd.DataFrame(filas)
    columnas_numericas = [
        "round",
        "distribuidor",
        "num_examples",
        "mae",
        "mse",
        "rmse",
        "r2",
        "retraso_real_medio",
        "retraso_predicho_medio",
        "diferencia_media",
        "sesgo_medio",
    ]
    df_hist = _convertir_columnas_numericas(df_hist, columnas_numericas)
    df_hist = df_hist.dropna(subset=["round", "distribuidor"])

    if df_hist.empty:
        return

    df_hist["Distribuidor"] = df_hist["distribuidor"].astype(int).map(
        lambda value: f"Distribuidor {value}"
    )

    ultima_ronda = int(df_hist["round"].max())
    df_final = (
        df_hist[df_hist["round"] == ultima_ronda]
        .sort_values("distribuidor")
        .copy()
    )

    if df_final.empty:
        return

    # El sesgo se puede reconstruir aunque proceda de un metrics.json anterior.
    if {
        "retraso_real_medio",
        "retraso_predicho_medio",
    }.issubset(df_final.columns):
        sesgo_calculado = (
            df_final["retraso_predicho_medio"]
            - df_final["retraso_real_medio"]
        )
        if "sesgo_medio" not in df_final.columns:
            df_final["sesgo_medio"] = sesgo_calculado
        else:
            df_final["sesgo_medio"] = df_final["sesgo_medio"].fillna(
                sesgo_calculado
            )

        if "diferencia_media" not in df_final.columns:
            df_final["diferencia_media"] = sesgo_calculado.abs()
        else:
            df_final["diferencia_media"] = df_final[
                "diferencia_media"
            ].fillna(sesgo_calculado.abs())

    if {"mae", "rmse"}.issubset(df_final.columns):
        df_final["brecha_rmse_mae"] = df_final["rmse"] - df_final["mae"]

    st.divider()
    st.subheader(f"🚚 Comparativa por distribuidor · ronda {ultima_ronda}")

    mejor_mae = _fila_optima(df_final, "mae")
    mejor_rmse = _fila_optima(df_final, "rmse")
    mejor_r2 = _fila_optima(df_final, "r2", maximizar=True)
    mejor_diferencia = _fila_optima(df_final, "diferencia_media")

    columnas_kpi = st.columns(4)
    if mejor_mae is not None:
        columnas_kpi[0].metric(
            "Menor MAE",
            f"{mejor_mae['mae']:.2f} min",
            mejor_mae["Distribuidor"],
            delta_color="off",
        )
    if mejor_rmse is not None:
        columnas_kpi[1].metric(
            "Menor RMSE",
            f"{mejor_rmse['rmse']:.2f} min",
            mejor_rmse["Distribuidor"],
            delta_color="off",
        )
    if mejor_r2 is not None:
        columnas_kpi[2].metric(
            "Mayor R²",
            f"{mejor_r2['r2']:.4f}",
            mejor_r2["Distribuidor"],
            delta_color="off",
        )
    if mejor_diferencia is not None:
        columnas_kpi[3].metric(
            "Menor diferencia media",
            f"{mejor_diferencia['diferencia_media']:.2f} min",
            mejor_diferencia["Distribuidor"],
            delta_color="off",
        )

    tab_resumen, tab_retrasos, tab_evolucion, tab_tabla = st.tabs(
        [
            "Comparación de errores",
            "Retraso real y predicho",
            "Evolución por rondas",
            "Tabla completa",
        ]
    )

    with tab_resumen:
        col_errores, col_r2 = st.columns(2)

        metricas_error = [
            metrica
            for metrica in ["mae", "rmse"]
            if metrica in df_final.columns
            and not df_final[metrica].dropna().empty
        ]
        if metricas_error:
            df_errores = df_final.melt(
                id_vars=["Distribuidor"],
                value_vars=metricas_error,
                var_name="Métrica",
                value_name="Minutos",
            )
            df_errores["Métrica"] = df_errores["Métrica"].str.upper()
            fig_errores = px.bar(
                df_errores,
                x="Distribuidor",
                y="Minutos",
                color="Métrica",
                barmode="group",
                title="MAE y RMSE por distribuidor",
                template="plotly_white",
                text_auto=".2f",
            )
            fig_errores.update_layout(yaxis_title="Error (minutos)")
            col_errores.plotly_chart(
                fig_errores,
                width="stretch",
                key="dist_errores_finales",
            )

        if "r2" in df_final.columns and not df_final["r2"].dropna().empty:
            fig_r2 = px.bar(
                df_final,
                x="Distribuidor",
                y="r2",
                title="R² por distribuidor",
                template="plotly_white",
                text_auto=".4f",
            )
            fig_r2.add_hline(y=0, line_dash="dash")
            fig_r2.update_layout(yaxis_title="R²")
            col_r2.plotly_chart(fig_r2, width="stretch", key="dist_r2_final")
            col_r2.caption(
                "Un R² cercano a 1 indica mejor ajuste; un valor negativo "
                "indica que el modelo rinde peor que predecir la media."
            )

        if (
            "brecha_rmse_mae" in df_final.columns
            and not df_final["brecha_rmse_mae"].dropna().empty
        ):
            fig_brecha = px.bar(
                df_final,
                x="Distribuidor",
                y="brecha_rmse_mae",
                title="Diferencia entre RMSE y MAE",
                template="plotly_white",
                text_auto=".2f",
            )
            fig_brecha.update_layout(
                yaxis_title="RMSE − MAE (minutos)"
            )
            st.plotly_chart(fig_brecha, width="stretch", key="dist_brecha")
            st.caption(
                "Una brecha grande entre RMSE y MAE suele indicar que existen "
                "algunos errores especialmente altos."
            )

    with tab_retrasos:
        columnas_medias = [
            columna
            for columna in [
                "retraso_real_medio",
                "retraso_predicho_medio",
            ]
            if columna in df_final.columns
            and not df_final[columna].dropna().empty
        ]

        if len(columnas_medias) == 2:
            nombres_medias = {
                "retraso_real_medio": "Retraso real medio",
                "retraso_predicho_medio": "Retraso predicho medio",
            }
            df_medias = df_final.melt(
                id_vars=["Distribuidor"],
                value_vars=columnas_medias,
                var_name="Tipo",
                value_name="Minutos",
            )
            df_medias["Tipo"] = df_medias["Tipo"].map(nombres_medias)

            fig_medias = px.bar(
                df_medias,
                x="Distribuidor",
                y="Minutos",
                color="Tipo",
                barmode="group",
                title="Retraso real medio frente al predicho",
                template="plotly_white",
                text_auto=".2f",
            )
            st.plotly_chart(fig_medias, width="stretch", key="dist_medias")

            col_diferencia, col_sesgo = st.columns(2)

            if "diferencia_media" in df_final.columns:
                fig_diferencia = px.bar(
                    df_final,
                    x="Distribuidor",
                    y="diferencia_media",
                    title="Diferencia absoluta entre las medias",
                    template="plotly_white",
                    text_auto=".2f",
                )
                fig_diferencia.update_layout(yaxis_title="Diferencia (minutos)")
                col_diferencia.plotly_chart(
                    fig_diferencia,
                    width="stretch",
                    key="dist_diferencia_media",
                )

            if "sesgo_medio" in df_final.columns:
                fig_sesgo = px.bar(
                    df_final,
                    x="Distribuidor",
                    y="sesgo_medio",
                    title="Sesgo medio de la predicción",
                    template="plotly_white",
                    text_auto=".2f",
                )
                fig_sesgo.add_hline(y=0, line_dash="dash")
                fig_sesgo.update_layout(yaxis_title="Predicho − real (minutos)")
                col_sesgo.plotly_chart(
                    fig_sesgo,
                    width="stretch",
                    key="dist_sesgo_medio",
                )
                col_sesgo.caption(
                    "Valor positivo: el modelo sobreestima el retraso. "
                    "Valor negativo: lo infraestima."
                )

            fig_dispersion = px.scatter(
                df_final,
                x="retraso_real_medio",
                y="retraso_predicho_medio",
                text="Distribuidor",
                title="Correspondencia entre retraso real y predicho",
                template="plotly_white",
            )
            valores = pd.concat(
                [
                    df_final["retraso_real_medio"],
                    df_final["retraso_predicho_medio"],
                ]
            ).dropna()
            if not valores.empty:
                minimo = float(valores.min())
                maximo = float(valores.max())
                margen = max((maximo - minimo) * 0.1, 1.0)
                fig_dispersion.add_shape(
                    type="line",
                    x0=minimo - margen,
                    y0=minimo - margen,
                    x1=maximo + margen,
                    y1=maximo + margen,
                    line=dict(dash="dash"),
                )
            fig_dispersion.update_traces(textposition="top center")
            fig_dispersion.update_layout(
                xaxis_title="Retraso real medio (min)",
                yaxis_title="Retraso predicho medio (min)",
            )
            st.plotly_chart(
                fig_dispersion,
                width="stretch",
                key="dist_real_predicho_scatter",
            )
            st.caption(
                "Cuanto más cerca esté cada distribuidor de la diagonal, "
                "más próxima es su media predicha a la real."
            )

    with tab_evolucion:
        if df_hist["round"].nunique() < 2:
            st.info("Se necesitan al menos dos rondas para mostrar la evolución.")
        else:
            metricas_evolucion = [
                ("mae", "Evolución del MAE", "MAE (minutos)"),
                ("rmse", "Evolución del RMSE", "RMSE (minutos)"),
                ("r2", "Evolución del R²", "R²"),
                (
                    "diferencia_media",
                    "Evolución de la diferencia media",
                    "Diferencia (minutos)",
                ),
            ]

            for indice in range(0, len(metricas_evolucion), 2):
                columnas = st.columns(2)
                for columna_ui, (metrica, titulo, eje_y) in zip(
                    columnas,
                    metricas_evolucion[indice:indice + 2],
                ):
                    if (
                        metrica not in df_hist.columns
                        or df_hist[metrica].dropna().empty
                    ):
                        continue
                    fig_evolucion = px.line(
                        df_hist.dropna(subset=[metrica]),
                        x="round",
                        y=metrica,
                        color="Distribuidor",
                        markers=True,
                        title=titulo,
                        template="plotly_white",
                    )
                    if metrica == "r2":
                        fig_evolucion.add_hline(y=0, line_dash="dash")
                    fig_evolucion.update_layout(
                        xaxis_title="Ronda",
                        yaxis_title=eje_y,
                    )
                    columna_ui.plotly_chart(
                        fig_evolucion,
                        width="stretch",
                        key=f"evolucion_dist_{metrica}",
                    )

    with tab_tabla:
        nombres_columnas = {
            "Distribuidor": "Distribuidor",
            "num_examples": "Ejemplos de test",
            "mae": "MAE (min)",
            "mse": "MSE (min²)",
            "rmse": "RMSE (min)",
            "r2": "R²",
            "retraso_real_medio": "Retraso real medio (min)",
            "retraso_predicho_medio": "Retraso predicho medio (min)",
            "diferencia_media": "Diferencia absoluta (min)",
            "sesgo_medio": "Sesgo predicho-real (min)",
        }
        columnas_tabla = [
            columna
            for columna in nombres_columnas
            if columna in df_final.columns
        ]
        tabla = df_final[columnas_tabla].rename(columns=nombres_columnas)
        formatos = {
            columna: "{:.2f}"
            for columna in tabla.columns
            if columna not in ["Distribuidor", "Ejemplos de test", "R²"]
        }
        if "R²" in tabla.columns:
            formatos["R²"] = "{:.4f}"
        st.dataframe(
            tabla.style.format(formatos, na_rep="—"),
            width="stretch",
            hide_index=True,
        )


def dibujar_dashboard():
    ruta_metrics = os.path.abspath("metrics.json")
    if not os.path.exists(ruta_metrics):
        st.info(
            "No hay métricas registradas. Inicia un entrenamiento cuando "
            "el SuperLink esté activo."
        )
        return

    try:
        with open(ruta_metrics, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        st.warning("No se pudo leer metrics.json.")
        return

    rondas = data.get("round", [])
    if not rondas:
        return

    columnas_globales = ["round", "time"] + ALL_METRICS
    df_data = {
        columna: data[columna]
        for columna in columnas_globales
        if isinstance(data.get(columna), list)
        and len(data[columna]) == len(rondas)
    }
    df = pd.DataFrame(df_data)

    metricas_validas = [
        metrica
        for metrica in selected_metrics
        if metrica in df.columns and not df.dropna(subset=[metrica]).empty
    ]

    for i in range(0, len(metricas_validas), 2):
        columnas_kpis = st.columns(2)

        for col, metrica in zip(columnas_kpis, metricas_validas[i:i + 2]):
            df_m = df.dropna(subset=[metrica])
            val = float(df_m[metrica].iloc[-1])
            prev = float(df_m[metrica].iloc[-2]) if len(df_m) > 1 else val
            delta = val - prev
            config = CONFIG_METRICAS[metrica]

            texto_val = format(val, config["formato"]) + config.get("sufijo", "")
            texto_del = format(delta, config["formato"]) + config.get("sufijo", "")

            col.metric(
                label=config["kpi"],
                value=texto_val,
                delta=texto_del,
                delta_color=config["delta_color"],
            )

    for metrica in selected_metrics:
        if metrica not in df.columns:
            continue
        df_m = df.dropna(subset=[metrica])
        if df_m.empty:
            continue

        config = CONFIG_METRICAS[metrica]
        fig = px.line(
            df_m,
            x="round",
            y=metrica,
            markers=True,
            title=config["titulo"],
            template="plotly_white",
        )
        fig.update_traces(
            line=dict(width=3, color=config["color"]),
            marker=dict(size=8),
        )
        fig.update_layout(uirevision="constant", xaxis_title="Ronda")
        st.plotly_chart(fig, width="stretch", key=f"chart_{metrica}")

    if data.get("task") in ["task_logistica", "task_regresion"]:
        dibujar_resultados_distribuidores(data)

if entrenamiento_en_curso:
    @st.fragment(run_every="1s")
    def monitor_vivo():
        dibujar_dashboard()
        if st.session_state.server_app_process.poll() is not None:
            st.rerun()
    monitor_vivo()
else:
    dibujar_dashboard()
    if st.session_state.server_app_process is not None:
        if st.session_state.server_app_process.poll() == 0:
            st.success("✨ Entrenamiento completado con éxito en la red.")
        else:
            st.warning("🛑 ServerApp detenido.")
        st.session_state.server_app_process = None

# --- LANZAMIENTO DEL PROCESO SERVERAPP ---
if start_button:
    ruta_config = os.path.abspath("run_config.json")
    ruta_metrics = os.path.abspath("metrics.json")

    if os.path.exists(ruta_metrics):
        os.remove(ruta_metrics)
    if os.path.exists(ruta_txt_parada):
        os.remove(ruta_txt_parada)

    if not usar_checkpoints:
        checkpoints_viejos = glob.glob(
            os.path.join("checkpoint", "**", "checkpoint_round_*.npz"),
            recursive=True,
        )
        for f_old in checkpoints_viejos:
            try: os.remove(f_old)
            except: pass

    config_data = {
        "rounds": num_rounds, "min_clients": int(min_clients), "lr": lr, "batch_size": batch,
        "fraction": fraction_val, "strategy": estrategia, "mu": mu_val, "task": tarea,
        "selected_metrics": selected_metrics, "data_distribution": data_distribution,
        "privacy_epsilon": privacy_epsilon, "dp_noise": dp_noise, "selected_distributors": selected_distributors,
        "use_checkpoints": usar_checkpoints
    }

    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    entorno = os.environ.copy()
    entorno["FLWR_RUN_CONFIG_PATH"] = ruta_config
    entorno["FLWR_METRICS_PATH"] = ruta_metrics
    entorno["FLWR_TASK_NAME"] = tarea
    entorno["PYTHONPATH"] = os.getcwd() + os.pathsep + entorno.get("PYTHONPATH", "")


    st.session_state.server_app_process = subprocess.Popen(
    [
        flwr_exe,
        "run",
        ".",
        "red-fisica",
        "--stream",
    ],
    env=entorno,
    cwd=os.getcwd()
)
    st.toast("🚀 Iniciar flwr run", icon="🌸")
    st.rerun()
