import importlib
import json
import os
import sys
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
from flwr.app import Context


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
DEFAULT_TASK = "task_regresion_logistica"

def get_requested_task_name(config=None):
    """Obtiene la tarea desde la configuración enviada por el servidor.

    La lectura dinámica evita que los clientes se queden cargados con MNIST
    cuando Streamlit ha seleccionado la tarea logística.
    """

    if config:
        task_name = config.get("task_name")
        if task_name:
            return str(task_name)

    config_path = os.environ.get("FLWR_RUN_CONFIG_PATH", "run_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            run_config = json.load(file)
        task_name = run_config.get("task")
        if task_name:
            return str(task_name)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    return os.environ.get("FLWR_TASK_NAME", DEFAULT_TASK)


def parse_selected_metrics(config):
    metrics_text = str(config.get("selected_metrics", ""))
    return {
        metric.strip()
        for metric in metrics_text.split(",")
        if metric.strip()
    }


def parse_selected_distributors(config):
    """Convierte la selección enviada por el servidor en una lista."""

    distributors_text = str(
        config.get("selected_distributors", "")
    )

    distributors = []
    for value in distributors_text.split(","):
        value = value.strip()
        if not value:
            continue
        distributors.append(int(value))

    return distributors


def add_noise_to_parameters(parameters, noise_std):
    """Añade ruido gaussiano a los parámetros antes de enviarlos al servidor."""

    if noise_std <= 0:
        return parameters

    noisy_parameters = []

    for param in parameters:
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=param.shape,
        ).astype(param.dtype)
        noisy_parameters.append(param + noise)

    return noisy_parameters


class GenericFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.task_name = None
        self.task = None
        self.net = None

    def ensure_task(self, config=None):
        requested_task = get_requested_task_name(config)

        if self.task_name != requested_task or self.task is None or self.net is None:
            self.task_name = requested_task
            
            try:
                # Intento 1: Buscar dentro de la carpeta 'task'
                self.task = importlib.import_module(f"task.{requested_task}")
            except Exception as error_carpeta:
                try:
                    # Intento 2: Fallback por si el archivo está en la raíz
                    self.task = importlib.import_module(requested_task)
                except Exception as error_raiz:
                    # Si falla, lanzamos un error claro a la consola para saber exactamente el porqué
                    raise RuntimeError(
                        f"\n❌ CRÍTICO: No se pudo cargar el archivo '{requested_task}.py'.\n"
                        f"Detalle al buscar en carpeta 'task/': {error_carpeta}\n"
                        f"Detalle al buscar en la raíz: {error_raiz}\n"
                    )
            
            self.net = self.task.get_model()

    def get_distributor_id(self, config):
        """Mapea el cliente virtual con el distribuidor real elegido.

        Ejemplo: si se seleccionan [2, 4], el cliente virtual 0 usa el
        distribuidor 2 y el cliente virtual 1 usa el distribuidor 4.
        """

        selected_distributors = parse_selected_distributors(config)

        if not selected_distributors:
            return self.cid + 1

        if self.cid < 0 or self.cid >= len(selected_distributors):
            raise ValueError(
                f"El cliente virtual {self.cid} no tiene un "
                "distribuidor asignado."
            )

        return int(selected_distributors[self.cid])

    def load_local_data(self, config, batch_size, num_clients):
        load_kwargs = {
            "batch_size": batch_size,
            "cid": self.cid,
            "num_clients": num_clients,
        }

        if self.task_name in ["task_logistica", "task_regresion_logistica"]:
            load_kwargs["distributor_id"] = self.get_distributor_id(
                config
            )

        return self.task.load_data(**load_kwargs)

    def get_parameters(self, config):
        if not config:
            config = {"task_name": os.environ.get("FLWR_TASK_NAME", DEFAULT_TASK)}

        self.ensure_task(config)

        print(f"Cliente {self.cid} usando task en get_parameters: {self.task_name}")

        if hasattr(self.task, "get_model_params"):
            return self.task.get_model_params(self.net)

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        self.ensure_task(config)

        if hasattr(self.task, "set_model_params"):
            self.task.set_model_params(self.net, parameters)
            return

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {key: torch.tensor(value) for key, value in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)

        lr = float(config.get("lr", 0.01))
        batch_size = int(config.get("batch_size", 32))
        num_clients = int(config.get("num_clients", 2))
        dp_noise = float(config.get("dp_noise", 0.0))

        trainloader, _, num_examples = self.load_local_data(
            config=config,
            batch_size=batch_size,
            num_clients=num_clients,
        )

        self.task.train(self.net, trainloader, lr)

        parameters_to_send = self.get_parameters(config={"task_name": self.task_name})
        parameters_to_send = add_noise_to_parameters(parameters_to_send, dp_noise)

        return parameters_to_send, num_examples, {"dp_noise": dp_noise}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)

        num_clients = int(config.get("num_clients", 2))
        selected_metrics = parse_selected_metrics(config)

        _, testloader, _ = self.load_local_data(
            config=config,
            batch_size=32,
            num_clients=num_clients,
        )

        result = self.task.test(self.net, testloader)

        if len(result) == 4:
            loss, _, num_examples, extra_metrics = result
            available_metrics = {"loss": float(loss)}
            available_metrics.update(
                {
                    name: float(value)
                    for name, value in extra_metrics.items()
                }
            )
        else:
            loss, accuracy, num_examples = result
            available_metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
            }

        # Flower recibe la pérdida en el primer campo obligatorio.
        # Las demás métricas viajan en el diccionario para su agregación.
        metrics_to_send = {
            name: value
            for name, value in available_metrics.items()
            if name in selected_metrics and name != "loss"
        }

        # Estas métricas auxiliares se envían siempre en logística.
        # El servidor las guarda por distribuidor y no las agrega globalmente.
        if self.task_name in ["task_logistica", "task_regresion_logistica"]:
            actual_distributor_id = self.get_distributor_id(config)
            metrics_to_send["distribuidor_id"] = float(
                actual_distributor_id
            )

            for auxiliary_metric in [
                "retraso_real_medio",
                "retraso_predicho_medio",
                "diferencia_media",
            ]:
                if auxiliary_metric in available_metrics:
                    metrics_to_send[auxiliary_metric] = available_metrics[
                        auxiliary_metric
                    ]

            for distributor_metric in ["mae", "rmse", "r2"]:
                if distributor_metric in available_metrics:
                    metrics_to_send[distributor_metric] = available_metrics[
                        distributor_metric
                    ]

        return float(loss), int(num_examples), metrics_to_send


def client_fn(context: Context):
    cid = context.node_config.get("partition-id", 0)
    
    return GenericFlowerClient(str(cid)).to_client()


app = fl.client.ClientApp(client_fn=client_fn)