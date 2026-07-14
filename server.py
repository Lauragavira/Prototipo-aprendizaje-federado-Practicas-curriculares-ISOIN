import json
import os
import time

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
    }


ALL_METRICS = ["loss", "accuracy", "mae", "rmse", "r2"]

TASK_DEFAULT_METRICS = {
    "task_mnist": ["loss", "accuracy"],
    "task_noIID_mnist": ["loss", "accuracy"],
    "task_logistica": ["mae", "rmse", "r2"],
}

DISTRIBUTOR_METRICS = [
    "mae",
    "rmse",
    "r2",
    "retraso_real_medio",
    "retraso_predicho_medio",
    "diferencia_media",
]

TASK_NAME = args.get("task", os.environ.get("FLWR_TASK_NAME", "task_mnist"))


def parse_selected_distributors():
    """Valida los distribuidores elegidos desde Streamlit."""

    if TASK_NAME != "task_logistica":
        return []

    raw_distributors = args.get("selected_distributors", [])

    try:
        distributors = [
            int(distributor_id)
            for distributor_id in raw_distributors
        ]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selected_distributors debe ser una lista de números enteros."
        ) from error

    if not 2 <= len(distributors) <= 4:
        raise ValueError(
            "En logística se deben seleccionar entre 2 y 4 "
            f"distribuidores. Se recibieron {len(distributors)}."
        )

    if len(set(distributors)) != len(distributors):
        raise ValueError("No se pueden repetir distribuidores.")

    invalid = [
        distributor_id
        for distributor_id in distributors
        if distributor_id not in {1, 2, 3, 4}
    ]
    if invalid:
        raise ValueError(
            f"Distribuidores no válidos: {invalid}. "
            "Solo existen los distribuidores 1, 2, 3 y 4."
        )

    return distributors


SELECTED_DISTRIBUTORS = parse_selected_distributors()

if TASK_NAME == "task_logistica":
    # Cada distribuidor seleccionado representa un cliente Flower.
    NUM_CLIENTS = len(SELECTED_DISTRIBUTORS)
else:
    NUM_CLIENTS = int(args.get("min_clients", 2))
    if NUM_CLIENTS < 2 or NUM_CLIENTS > 4:
        raise ValueError(
            "El número de clientes debe estar entre 2 y 4. "
            f"Se recibió: {NUM_CLIENTS}."
        )

selected_metrics = args.get(
    "selected_metrics",
    TASK_DEFAULT_METRICS.get(TASK_NAME, ["loss", "accuracy"]),
)
selected_metrics = [
    metric for metric in selected_metrics if metric in ALL_METRICS
]

if not selected_metrics:
    selected_metrics = TASK_DEFAULT_METRICS.get(
        TASK_NAME,
        ["loss", "accuracy"],
    )




def weighted_average(metrics):
    """Agrega únicamente las métricas seleccionadas en Streamlit."""

    aggregated = {}

    for metric_name in selected_metrics:
        if metric_name == "loss":
            continue

        valid_values = [
            (num_examples, client_metrics[metric_name])
            for num_examples, client_metrics in metrics
            if metric_name in client_metrics
        ]

        if not valid_values:
            continue

        total_examples = sum(
            num_examples for num_examples, _ in valid_values
        )
        if total_examples == 0:
            continue

        aggregated[metric_name] = sum(
            num_examples * float(value)
            for num_examples, value in valid_values
        ) / total_examples

    return aggregated


def common_config():
    return {
        "num_clients": NUM_CLIENTS,
        "data_distribution": args.get("data_distribution", "IID"),
        "task_name": TASK_NAME,
        # Flower solo permite valores escalares en config.
        "selected_metrics": ",".join(selected_metrics),
        "selected_distributors": ",".join(
            str(distributor_id)
            for distributor_id in SELECTED_DISTRIBUTORS
        ),
    }


def fit_config(server_round: int):
    config = common_config()
    config.update(
        {
            "lr": float(args["lr"]),
            "batch_size": int(args["batch_size"]),
            "privacy_epsilon": (
                float(args["privacy_epsilon"])
                if args.get("privacy_epsilon") is not None
                else -1.0
            ),
            "dp_noise": float(args.get("dp_noise", 0.0)),
        }
    )
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
        # Una entrada por ronda con los resultados individuales.
        "distribuidores": [],
    }

    for metric in ALL_METRICS:
        data[metric] = []

    return data


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_distributor_results(server_round, results):
    """Extrae las métricas individuales de cada distribuidor."""

    if TASK_NAME != "task_logistica":
        return None

    distributor_results = []

    for position, (_, evaluate_res) in enumerate(results, start=1):
        client_metrics = evaluate_res.metrics or {}

        distributor_id = safe_float(
            client_metrics.get("distribuidor_id", position)
        )

        actual_distributor_id = int(distributor_id or position)

        # Solo se guardan distribuidores seleccionados en esta ejecución.
        if (
            SELECTED_DISTRIBUTORS
            and actual_distributor_id not in SELECTED_DISTRIBUTORS
        ):
            continue

        row = {
            "distribuidor": actual_distributor_id,
            "num_examples": int(evaluate_res.num_examples),
        }

        for metric_name in DISTRIBUTOR_METRICS:
            row[metric_name] = safe_float(
                client_metrics.get(metric_name)
            )

        distributor_results.append(row)

    distributor_results.sort(
        key=lambda item: item["distribuidor"]
    )

    return {
        "round": int(server_round),
        "resultados": distributor_results,
    }


def server_fn(context: fl.common.Context):
    """Flower llama a esta función al iniciar el entrenamiento."""

    base_kwargs = {
        "fraction_fit": float(args["fraction"]),
        "fraction_evaluate": float(args["fraction"]),
        "min_available_clients": NUM_CLIENTS,
        "min_fit_clients": max(
            1,
            int(NUM_CLIENTS * args["fraction"]),
        ),
        "min_evaluate_clients": max(
            1,
            int(NUM_CLIENTS * args["fraction"]),
        ),
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
    }

    if args["strategy"] == "FedProx":
        base_strategy = fl.server.strategy.FedProx
        base_kwargs["proximal_mu"] = float(args["mu"])
    elif args["strategy"] == "FedMedian":
        base_strategy = fl.server.strategy.FedMedian
    else:
        base_strategy = fl.server.strategy.FedAvg

    class SaveMetricsStrategy(base_strategy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.start_time = time.time()

        def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
        ):
            loss, metrics = super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )

            if loss is None:
                return loss, metrics

            elapsed_time = time.time() - self.start_time
            metrics = metrics or {}

            # Respaldo para versiones de Flower que devuelven vacío
            # el diccionario agregado.
            if results:
                client_metrics = [
                    (
                        evaluate_res.num_examples,
                        evaluate_res.metrics or {},
                    )
                    for _, evaluate_res in results
                ]
                metrics.update(weighted_average(client_metrics))

            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = empty_metrics_data()

            data["round"].append(int(server_round))
            data["time"].append(float(elapsed_time))

            for metric in ALL_METRICS:
                value = None

                if metric in selected_metrics:
                    if metric == "loss":
                        value = float(loss)
                    elif metric in metrics:
                        value = safe_float(metrics[metric])

                data[metric].append(value)

            distributor_round = extract_distributor_results(
                server_round,
                results,
            )
            if distributor_round is not None:
                data.setdefault("distribuidores", [])
                data["distribuidores"].append(distributor_round)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            return loss, metrics

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            empty_metrics_data(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    return fl.server.ServerAppComponents(
        config=fl.server.ServerConfig(
            num_rounds=int(args["rounds"])
        ),
        strategy=SaveMetricsStrategy(**base_kwargs),
    )


app = fl.server.ServerApp(server_fn=server_fn)