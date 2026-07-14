import flwr as fl
import torch
import importlib
import os
import numpy as np
from collections import OrderedDict


TASK_NAME = os.environ.get("FLWR_TASK_NAME", "task_mnist")
task = importlib.import_module(TASK_NAME)


def add_noise_to_parameters(parameters, noise_std):
    """Añade ruido gaussiano a los parámetros antes de enviarlos al servidor."""

    if noise_std <= 0:
        return parameters

    noisy_parameters = []

    for param in parameters:
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=param.shape
        ).astype(param.dtype)

        noisy_parameters.append(param + noise)

    return noisy_parameters


class GenericFlowerClient(fl.client.NumPyClient):
    def __init__(self, net, cid):
        self.net = net
        self.cid = int(cid)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = float(config.get("lr", 0.01))
        batch_size = int(config.get("batch_size", 32))
        num_clients = int(config.get("num_clients", 2))
        dp_noise = float(config.get("dp_noise", 0.0))

        trainloader, _, num_examples = task.load_data(
            batch_size=batch_size,
            cid=self.cid,
            num_clients=num_clients,
        )

        task.train(self.net, trainloader, lr)

        parameters_to_send = self.get_parameters(config={})
        parameters_to_send = add_noise_to_parameters(parameters_to_send, dp_noise)

        return parameters_to_send, num_examples, {"dp_noise": dp_noise}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        num_clients = int(config.get("num_clients", 2))

        _, testloader, _ = task.load_data(
            batch_size=32,
            cid=self.cid,
            num_clients=num_clients,
        )

        loss, accuracy, num_examples = task.test(self.net, testloader)

        return float(loss), num_examples, {"accuracy": float(accuracy)}


def client_fn(cid: str):
    modelo_local = task.get_model()
    return GenericFlowerClient(modelo_local, cid).to_client()


app = fl.client.ClientApp(client_fn=client_fn)