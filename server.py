import flwr as fl
import time
import json
import os

config_path = os.environ.get("FLWR_RUN_CONFIG_PATH", "run_config.json")
metrics_path = os.environ.get("FLWR_METRICS_PATH", "metrics.json")

try:
    with open(config_path, "r") as f:
        args = json.load(f)
except FileNotFoundError:
    args = {"rounds": 5, "min_clients": 2, "lr": 0.01, "batch_size": 32, "fraction": 1.0, "strategy": "FedAvg", "mu": 0.1}  

def weighted_average(metrics):
    # 1. Filtramos y nos quedamos solo con los clientes que sí enviaron "accuracy"
    metricas_validas = [(num_examples, m) for num_examples, m in metrics if "accuracy" in m]
    
    # 2. Si la lista está vacía (como pasa en la fase 'fit'), devolvemos un diccionario vacío sin rompernos
    if not metricas_validas:
        return {}
        
    # 3. Si hay datos (fase 'evaluate'), hacemos la media matemática normal
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metricas_validas]
    examples = [num_examples for num_examples, m in metricas_validas]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    return {"lr": args["lr"], "batch_size": args["batch_size"]}

# --- NUEVA API DE FLOWER ---
def server_fn(context: fl.common.Context):
    """Flower llama a esta función cuando Streamlit inicia el trabajo."""
    
    # 1. Configuración base
    base_kwargs = {
        "fraction_fit": args["fraction"],
        "fraction_evaluate": args["fraction"],
        "min_available_clients": args["min_clients"],
        "min_fit_clients": max(1, int(args["min_clients"] * args["fraction"])),
        "min_evaluate_clients": max(1, int(args["min_clients"] * args["fraction"])),
        "on_fit_config_fn": fit_config,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    # 2. Selección de Estrategia
    if args["strategy"] == "FedProx":
        BaseStrategy = fl.server.strategy.FedProx
        base_kwargs["proximal_mu"] = args["mu"]
    elif args["strategy"] == "FedMedian":
        BaseStrategy = fl.server.strategy.FedMedian
    else:
        BaseStrategy = fl.server.strategy.FedAvg

    # 3. Estrategia personalizada para guardar en metrics.json
    class SaveMetricsStrategy(BaseStrategy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Iniciamos el reloj interno del servidor
            self.start_time = time.time()

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            
            # Calculamos cuánto tiempo ha pasado desde que arrancó el servidor
            tiempo_transcurrido = time.time() - self.start_time
            
            if loss is not None:
                try:
                    with open(metrics_path, "r") as f: 
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"round": [], "loss": [], "accuracy": [], "time": []}
                
                data["round"].append(server_round)
                data["loss"].append(loss)
                data["time"].append(tiempo_transcurrido) # Añadimos el tiempo
                
                if metrics and "accuracy" in metrics:
                    data["accuracy"].append(metrics["accuracy"])
                else:
                    data["accuracy"].append(None)
                    
                with open(metrics_path, "w") as f: 
                    json.dump(data, f)
            return loss, metrics

    # Limpiamos métricas usando la ruta absoluta
    with open(metrics_path, "w") as f:
        json.dump({"round": [], "loss": [], "accuracy": [], "time": []}, f)

    return fl.server.ServerAppComponents(
        config=fl.server.ServerConfig(num_rounds=args["rounds"]),
        strategy=SaveMetricsStrategy(**base_kwargs),
    )

app = fl.server.ServerApp(server_fn=server_fn)