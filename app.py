import sys
import streamlit as st
import subprocess
import time
import json
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Panel de Aprendizaje Federado", layout="wide")

st.title("🌸 Panel de Control: Entrenamiento Federado")

# --- INICIALIZACIÓN DE PROCESOS EN MEMORIA ---
if "server_process" not in st.session_state:
    st.session_state.server_process = None
if "client_processes" not in st.session_state:
    st.session_state.client_processes = [] # Lista para guardar los clientes

# --- BARRA LATERAL (PARÁMETROS) ---
with st.sidebar:
    st.header("📦 Definición del Problema")
    # Este selector permite cambiar de tarea sin tocar el código
    tarea = st.selectbox("Tarea a ejecutar", ["task_mnist"], key="task_select")
    st.caption("Añade más archivos 'task_algo.py' y ponlos en esta lista.")
    
    st.header("⚙️ Parámetros del Servidor")
    num_rounds = st.slider("Número de Rondas", 1, 50, 5, key="rounds_slider")
    min_clients = st.number_input("Clientes a simular", 1, 10, 2, key="clients_input")
    porcentaje = st.slider("Participación por ronda (%)", 10, 100, 100, step=10, key="fraction_slider")
    fraction_val = porcentaje / 100.0
    
    st.header("🧮 Estrategia de Agregación")
    estrategia = st.selectbox("Algoritmo", ["FedAvg", "FedProx", "FedMedian"], key="strategy_select")
    
    mu_val = 0.0 
    if estrategia == "FedProx":
        mu_val = st.number_input("Término Proximal (μ)", 0.0, 5.0, 0.1, step=0.1, key="mu_input")
        
    st.header("🧠 Hiperparámetros Locales")
    lr = st.selectbox("Learning Rate", [0.1, 0.01, 0.001], index=1, key="lr_select")
    batch = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32, key="batch_select")
    
    st.divider()
    start_button = st.button("🚀 Iniciar Entrenamiento Automático", width='stretch')

# --- ÁREA PRINCIPAL ---
st.subheader("📊 Métricas de la Última Ronda")
kpi_col1, kpi_col2 = st.columns(2)
kpi_loss = kpi_col1.empty()
kpi_acc = kpi_col2.empty()
st.divider()

placeholder_tiempo = st.empty()
grafica_loss = st.empty()
grafica_acc = st.empty()

if start_button:
    if st.session_state.server_process is None or st.session_state.server_process.poll() is not None:
        
        # 1. Aseguramos rutas absolutas
        ruta_absoluta_config = os.path.abspath("run_config.json")
        ruta_absoluta_metrics = os.path.abspath("metrics.json")
        
        if os.path.exists(ruta_absoluta_metrics):
            os.remove(ruta_absoluta_metrics)
            
        # 2. Guardamos la configuración de hiperparámetros
        config_data = {
            "rounds": num_rounds,
            "min_clients": min_clients,
            "lr": lr,
            "batch_size": batch,
            "fraction": fraction_val,
            "strategy": estrategia,
            "mu": mu_val
        }
        with open(ruta_absoluta_config, "w") as f:
            json.dump(config_data, f)
            
        # 3. Inyectamos las rutas en las variables de entorno de Windows
        entorno = os.environ.copy()
        entorno["FLWR_RUN_CONFIG_PATH"] = ruta_absoluta_config
        entorno["FLWR_METRICS_PATH"] = ruta_absoluta_metrics
            
        # 4. RASTREAMOS EL EJECUTABLE 'flwr.exe'
        carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        carpeta_proyecto = os.path.dirname(carpeta_actual)
        
        flwr_exe = os.path.join(carpeta_proyecto, "venv", "Scripts", "flwr.exe")
        if not os.path.exists(flwr_exe):
            flwr_exe = os.path.join(carpeta_proyecto, "venv", "Scripts", "flwr")
            

        # Como Flower lee automáticamente tu carpeta juan.vazquez/.flwr, 
        # solo tenemos que pasarle el nombre de tu perfil: "red-fisica"
        comando_servidor = [flwr_exe, "run", ".", "red-fisica", "--stream"]
        
        st.session_state.server_process = subprocess.Popen(comando_servidor, env=entorno)
        st.toast(f"🚀 Empaquetando código y enviando a la red-fisica...", icon="🌸")
                
        
        ultimo_round_dibujado = 0
        
        # 4. MONITORIZACIÓN (Igual que antes)
        while st.session_state.server_process.poll() is None:
            try:
                with open("metrics.json", "r") as f:
                    data = json.load(f)

                if data["time"] and len(data["time"]) > 0:
                    tiempos = data["time"]
                    duraciones_ronda = [tiempos[0]] + [tiempos[i] - tiempos[i-1] for i in range(1, len(tiempos))]
                    tiempo_medio = sum(duraciones_ronda) / len(duraciones_ronda)
                    placeholder_tiempo.metric(label="⌛ Tiempo medio por ronda", value=f"{tiempo_medio:.2f} segundos")
                
                if len(data["round"]) > ultimo_round_dibujado:
                    df = pd.DataFrame(data)
                    
                    ultima_loss = df["loss"].iloc[-1]
                    delta_loss = ultima_loss - df["loss"].iloc[-2] if len(df) > 1 else 0.0
                    kpi_loss.metric(label="Pérdida Global (Loss)", value=f"{ultima_loss:.4f}", delta=f"{delta_loss:.4f}", delta_color="inverse")
                    
                    fig_loss = px.line(df, x="round", y="loss", markers=True, title="📉 Evolución de la Pérdida", template="plotly_white")
                    fig_loss.update_traces(line=dict(width=3, color="#ff4b4b"), marker=dict(size=8))
                    grafica_loss.plotly_chart(fig_loss, width='stretch', key=f"loss_{len(data['round'])}")
                    
                    df_acc = df.dropna(subset=["accuracy"])
                    if not df_acc.empty:
                        ultima_acc = df_acc["accuracy"].iloc[-1]
                        delta_acc = ultima_acc - df_acc["accuracy"].iloc[-2] if len(df_acc) > 1 else 0.0
                        kpi_acc.metric(label="Precisión Global (Accuracy)", value=f"{ultima_acc:.2%}", delta=f"{delta_acc:.2%}")
                        
                        fig_acc = px.line(df_acc, x="round", y="accuracy", markers=True, title="🎯 Evolución de la Precisión", template="plotly_white")
                        fig_acc.update_traces(line=dict(width=3, color="#21c354"), marker=dict(size=8))
                        grafica_acc.plotly_chart(fig_acc, width='stretch', key=f"acc_{len(data['round'])}")
                    
                    ultimo_round_dibujado = len(data["round"])
            except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
                pass 
                
            time.sleep(1)
            
        st.session_state.server_process = None
        st.success("✨ Entrenamiento finalizado. Los clientes siguen conectados esperando el próximo.")