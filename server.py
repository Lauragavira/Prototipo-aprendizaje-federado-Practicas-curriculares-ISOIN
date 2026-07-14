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
    args = {
        "rounds": 5,
        "min_clients": 2,
        "lr": 0.01,
        "batch_size": 32,
        "fraction": 1.0,
        "strategy": "FedAvg",
        "mu": 0.1,
        "data_distribution": "IID",
        "privacy_epsilon": None,
        "dp_noise": 0.0,
    }


def weighted_average(metrics):
    # 1. Filtramos y nos quedamos solo con los clientes que sí enviaron "accuracy"
    metricas_validas = [(num_examples, m) for num_examples, m in metrics if "accuracy" in m]

    # 2. Si la lista está vacía, devolvemos un diccionario vacío
    if not metricas_validas:
        return {}

    # 3. Media ponderada de accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metricas_validas]
    examples = [num_examples for num_examples, m in metricas_validas]

    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    return {
        "lr": args["lr"],
        "batch_size": args["batch_size"],
        "num_clients": args["min_clients"],
        "data_distribution": args.get("data_distribution", "IID"),
        "privacy_epsilon": args.get("privacy_epsilon", None),
        "dp_noise": args.get("dp_noise", 0.0),
    }


def evaluate_config(server_round: int):
    return {
        "num_clients": args["min_clients"],
        "data_distribution": args.get("data_distribution", "IID"),
    }


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
        "on_evaluate_config_fn": evaluate_config,
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
            self.start_time = time.time()

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)

            tiempo_transcurrido = time.time() - self.start_time

            # Si Flower no devuelve accuracy agregada, la calculamos desde los clientes
            accuracy_global = None

            if metrics and "accuracy" in metrics:
                accuracy_global = metrics["accuracy"]
            elif results:
                total_examples = 0
                total_accuracy = 0.0

                for _, evaluate_res in results:
                    if "accuracy" in evaluate_res.metrics:
                        num_examples = evaluate_res.num_examples
                        acc = float(evaluate_res.metrics["accuracy"])

                        total_accuracy += acc * num_examples
                        total_examples += num_examples

                if total_examples > 0:
                    accuracy_global = total_accuracy / total_examples

            if loss is not None:
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"round": [], "loss": [], "accuracy": [], "time": []}

                data["round"].append(server_round)
                data["loss"].append(float(loss))
                data["time"].append(tiempo_transcurrido)
                data["accuracy"].append(accuracy_global)

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