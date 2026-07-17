import json
import os
import time
import glob
import numpy as np
import flwr as fl

config_path = os.environ.get("FLWR_RUN_CONFIG_PATH", "run_config.json")
metrics_path = os.environ.get("FLWR_METRICS_PATH", "metrics.json")

try:
    with open(config_path, "r", encoding="utf-8") as f:
        args = json.load(f)
except FileNotFoundError:
    args = {
        "rounds": 5,
        "min_clients": 2,
        "lr": 0.01,
        "batch_size": 32,
        "fraction": 1.0,
        "strategy": "FedAvg",
        "mu": 0.1,
        "task": "task_mnist",
        "selected_metrics": ["loss", "accuracy"],
        "data_distribution": "IID",
        "privacy_epsilon": None,
        "dp_noise": 0.0,
        "use_checkpoints": False
    }

ALL_METRICS = ["loss", "accuracy", "mae", "mse", "rmse", "r2"]
TASK_NAME = args.get("task", os.environ.get("FLWR_TASK_NAME", "task_mnist"))
DISTRIBUTOR_METRICS = ["mae", "mse", "rmse", "r2", "retraso_real_medio", "retraso_predicho_medio", "diferencia_media", "sesgo_medio"]

def parse_selected_distributors():
    if TASK_NAME not in ["task_logistica", "task_regresion"]:
        return []
    raw_distributors = args.get("selected_distributors", [])
    try:
        return [int(d_id) for d_id in raw_distributors]
    except (TypeError, ValueError):
        return []

SELECTED_DISTRIBUTORS = parse_selected_distributors()
NUM_CLIENTS = (
    len(SELECTED_DISTRIBUTORS)
    if TASK_NAME in ["task_logistica", "task_regresion"] and SELECTED_DISTRIBUTORS
    else int(args.get("min_clients", 2))
)

selected_metrics = [m for m in args.get("selected_metrics", ["loss", "accuracy"]) if m in ALL_METRICS]

def weighted_average(metrics):
    aggregated = {}
    for metric_name in selected_metrics:
        if metric_name == "loss":
            continue
        valid_values = [(num_examples, client_metrics[metric_name]) 
                        for num_examples, client_metrics in metrics if metric_name in client_metrics]
        if not valid_values:
            continue
        total_examples = sum(num_examples for num_examples, _ in valid_values)
        if total_examples == 0:
            continue
        aggregated[metric_name] = sum(num_examples * float(val) for num_examples, val in valid_values) / total_examples
    return aggregated

def common_config():
    return {
        "num_clients": NUM_CLIENTS,
        "data_distribution": args.get("data_distribution", "IID"),
        "task_name": TASK_NAME,
        "selected_metrics": ",".join(selected_metrics),
        "selected_distributors": ",".join(str(d_id) for d_id in SELECTED_DISTRIBUTORS),
    }

def fit_config(server_round: int):
    config = common_config()
    config.update({
        "lr": float(args["lr"]),
        "batch_size": int(args["batch_size"]),
        "privacy_epsilon": float(args["privacy_epsilon"]) if args.get("privacy_epsilon") is not None else -1.0,
        "dp_noise": float(args.get("dp_noise", 0.0)),
    })
    return config

def evaluate_config(server_round: int):
    return common_config()

def empty_metrics_data():
    data = {
        "task": TASK_NAME,
        "selected_metrics": selected_metrics,
        "selected_distributors": SELECTED_DISTRIBUTORS,
        "round": [],
        "time": [],
        "distribuidores": [],
    }
    for metric in ALL_METRICS:
        data[metric] = []
    return data

def safe_float(value):
    try: return float(value)
    except (TypeError, ValueError): return None

def extract_distributor_results(server_round, results):
    if TASK_NAME not in ["task_logistica", "task_regresion"]:
        return None
    distributor_results = []
    for pos, (_, eval_res) in enumerate(results, start=1):
        client_metrics = eval_res.metrics or {}
        d_id = int(safe_float(client_metrics.get("distribuidor_id", pos)) or pos)
        if SELECTED_DISTRIBUTORS and d_id not in SELECTED_DISTRIBUTORS:
            continue
        row = {"distribuidor": d_id, "num_examples": int(eval_res.num_examples)}
        for metric_name in DISTRIBUTOR_METRICS:
            row[metric_name] = safe_float(client_metrics.get(metric_name))
        distributor_results.append(row)
    distributor_results.sort(key=lambda item: item["distribuidor"])
    return {"round": int(server_round), "resultados": distributor_results}

def cargar_parametros_iniciales(ruta_archivo: str):
    if os.path.exists(ruta_archivo):
        print(f"📦 Checkpoint detectado: Cargando {ruta_archivo}...")
        datos_cargados = np.load(ruta_archivo)
        lista_ndarrays = [datos_cargados[clave] for clave in datos_cargados.files]
        return fl.common.ndarrays_to_parameters(lista_ndarrays)
    return None

def server_fn(context: fl.common.Context):
    # --- Gestión de Checkpoints (Tu lógica) ---
    pesos_recuperados = None
    carpeta_checkpoints = os.path.join("checkpoint", TASK_NAME)
    activar_reanudacion = args.get("use_checkpoints", True)
    ronda_inicial = 0

    if activar_reanudacion:
        checkpoints = glob.glob(os.path.join(carpeta_checkpoints, "checkpoint_round_*.npz"))
        if checkpoints:
            def extraer_numero_ronda(path):
                try: return int(os.path.basename(path).split('_')[-1].split('.')[0])
                except: return -1
            ultimo_checkpoint = max(checkpoints, key=extraer_numero_ronda)
            pesos_recuperados = cargar_parametros_iniciales(ultimo_checkpoint)
            ronda_inicial = max(0, extraer_numero_ronda(ultimo_checkpoint))

    base_kwargs = {
        "fraction_fit": float(args["fraction"]),
        "fraction_evaluate": float(args["fraction"]),
        "min_available_clients": NUM_CLIENTS,
        "min_fit_clients": max(1, int(NUM_CLIENTS * args["fraction"])),
        "min_evaluate_clients": max(1, int(NUM_CLIENTS * args["fraction"])),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "initial_parameters": pesos_recuperados
    }

    BaseStrategy = fl.server.strategy.FedProx if args["strategy"] == "FedProx" else (
        fl.server.strategy.FedMedian if args["strategy"] == "FedMedian" else fl.server.strategy.FedAvg
    )
    if args["strategy"] == "FedProx":
        base_kwargs["proximal_mu"] = float(args["mu"])

    class SaveMetricsStrategy(BaseStrategy):
        def __init__(self, ronda_inicial=0, **kwargs):
            super().__init__(**kwargs)
            self.start_time = time.time()
            self.ronda_inicial = ronda_inicial

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            if loss is None:
                return loss, metrics
            
            elapsed_time = time.time() - self.start_time
            metrics = metrics or {}
            ronda_real = self.ronda_inicial + server_round

            if results:
                client_metrics = [(ev_res.num_examples, ev_res.metrics or {}) for _, ev_res in results]
                metrics.update(weighted_average(client_metrics))

            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = empty_metrics_data()

            # Asegurar consistencia de listas
            if int(ronda_real) not in data["round"]:
                data["round"].append(int(ronda_real))
                data["time"].append(float(elapsed_time))
                for m in ALL_METRICS:
                    val = float(loss) if m == "loss" else safe_float(metrics.get(m)) if m in selected_metrics else None
                    data[m].append(val)

                dist_round = extract_distributor_results(ronda_real, results)
                if dist_round:
                    data.setdefault("distribuidores", [])
                    data["distribuidores"].append(dist_round)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return loss, metrics

        def aggregate_fit(self, server_round, results, failures):
            if os.path.exists("stop_training.txt"):
                print("🛑 Abortando entrenamiento de forma segura...")
                raise InterruptedError("Entrenamiento detenido desde la web.")
            
            agg_weights, metrics = super().aggregate_fit(server_round, results, failures)
            if agg_weights is not None:
                os.makedirs(carpeta_checkpoints, exist_ok=True)
                ndarrays = fl.common.parameters_to_ndarrays(agg_weights)
                checkpoint_path = os.path.join(carpeta_checkpoints, f"checkpoint_round_{self.ronda_inicial + server_round}.npz")
                np.savez(checkpoint_path, *ndarrays)
            return agg_weights, metrics

    if ronda_inicial == 0:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(empty_metrics_data(), f, ensure_ascii=False, indent=2)

    rondas_restantes = max(1, int(args["rounds"]) - ronda_inicial)

    return fl.server.ServerAppComponents(
        config=fl.server.ServerConfig(num_rounds=rondas_restantes),
        strategy=SaveMetricsStrategy(ronda_inicial=ronda_inicial, **base_kwargs),
    )

app = fl.server.ServerApp(server_fn=server_fn)
