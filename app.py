import sys
import streamlit as st
import subprocess
import json
import pandas as pd
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

ALL_METRICS = ["loss", "accuracy", "mae", "rmse", "r2"]
METRICAS_PREDETERMINADAS = {
    "task_mnist": ["loss", "accuracy"],
    "task_noIID_mnist": ["loss", "accuracy"],
    "task_logistica": ["mae", "rmse", "r2"],
    "task_regresion_logistica": ["loss", "accuracy"],
}
METRICAS_DISPONIBLES = {
    "task_mnist": {"loss", "accuracy","rmse", "r2"},
    "task_noIID_mnist": {"loss", "accuracy"},
    "task_logistica": {"loss", "mae", "rmse", "r2"},
    "task_regresion_logistica": {"loss", "accuracy","rmse", "r2"},
}

CONFIG_METRICAS = {
    "loss": {"nombre": "Loss", "kpi": "Pérdida global (Loss)", "titulo": "📉 Evolución de la pérdida", "formato": ".4f", "delta_color": "inverse", "color": "#ff4b4b"},
    "accuracy": {"nombre": "Accuracy", "kpi": "Precisión global", "titulo": "🎯 Evolución de la precisión", "formato": ".2%", "delta_color": "normal", "color": "#21c354"},
    "mae": {"nombre": "MAE", "kpi": "MAE medio", "titulo": "📉 Evolución del MAE", "formato": ".2f", "sufijo": " min", "delta_color": "inverse", "color": "#ff4b4b"},
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
        f"flwr-supernode --insecure --superlink {ip_servidor}:9092\n"
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
    
    st.subheader("Measuring Metrics")
    selected_metrics = []
    predeterminadas = METRICAS_PREDETERMINADAS.get(tarea, ["loss", "accuracy"])
    for m in ALL_METRICS:
        marcada = st.checkbox(CONFIG_METRICAS[m]["nombre"], value=(m in predeterminadas), key=f"chk_{m}")
        if marcada:
            selected_metrics.append(m)

    st.header("⚙️ Parámetros del Servidor")
    num_rounds = st.slider("Número de Rondas", 1, 50, 5)
    usar_checkpoints = st.checkbox("💾 Usar checkpoints (Reanudar)", value=False)

    selected_distributors = []
    distributor_selection_valid = True

    if tarea == "task_logistica":
        data_distribution = "No-IID real"

        selected_distributors = st.multiselect(
            "Distribuidores a comparar",
            [1, 2, 3, 4],
            default=[1, 2],
            format_func=lambda x: f"Distribuidor {x}"
        )

        min_clients = len(selected_distributors)
        distributor_selection_valid = 2 <= min_clients <= 4

    elif tarea == "task_regresion_logistica":
        data_distribution = "No-IID real"

        min_clients = st.selectbox(
            "Clientes esperados",
            [2, 3, 4],
            index=0
        )

        selected_distributors = list(range(1, min_clients + 1))
        distributor_selection_valid = True

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

def dibujar_dashboard():
    ruta_metrics = os.path.abspath("metrics.json")
    if not os.path.exists(ruta_metrics):
        st.info("No hay métricas registradas. ¡Inicia un entrenamiento cuando el SuperLink esté activo!")
        return

    try:
        with open(ruta_metrics, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return

    rondas = data.get("round", [])
    if not rondas:
        return

    df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, list) and len(v) == len(rondas)})

    metricas_validas = [
        m for m in selected_metrics
        if m in df.columns and not df.dropna(subset=[m]).empty
    ]

    for i in range(0, len(metricas_validas), 2):
        columnas_kpis = st.columns(2)

        for col, metrica in zip(columnas_kpis, metricas_validas[i:i + 2]):
            df_m = df.dropna(subset=[metrica])
            val = df_m[metrica].iloc[-1]
            prev = df_m[metrica].iloc[-2] if len(df_m) > 1 else val
            delta = val - prev
            config = CONFIG_METRICAS[metrica]

            texto_val = format(val, config["formato"]) + config.get("sufijo", "")
            texto_del = format(delta, config["formato"]) + config.get("sufijo", "")

            col.metric(
                label=config["kpi"],
                value=texto_val,
                delta=texto_del,
                delta_color=config["delta_color"]
            )

    for metrica in selected_metrics:
        if metrica in df.columns:
            df_m = df.dropna(subset=[metrica])
            if not df_m.empty:
                config = CONFIG_METRICAS[metrica]
                fig = px.line(df_m, x="round", y=metrica, markers=True, title=config["titulo"], template="plotly_white")
                fig.update_traces(line=dict(width=3, color=config["color"]), marker=dict(size=8))
                fig.update_layout(uirevision="constant")
                st.plotly_chart(fig, width='stretch', key=f"chart_{metrica}")

    if data.get("task") == "task_logistica" and data.get("distribuidores"):
        historial = data["distribuidores"]
        if historial:
            resultados = historial[-1].get("resultados", [])
            if resultados:
                st.divider()
                st.subheader("🚚 Resultados por distribuidor")
                df_dist = pd.DataFrame(resultados)
                st.dataframe(df_dist, width='stretch', hide_index=True)
    

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
        checkpoints_viejos = glob.glob(os.path.join("checkpoint", "checkpoint_round_*.npz"))
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