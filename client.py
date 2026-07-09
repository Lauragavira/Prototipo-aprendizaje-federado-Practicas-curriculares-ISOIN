import flwr as fl
import torch
import importlib
from collections import OrderedDict

# Por simplicidad, definimos la tarea aquí (puedes cambiarlo a task_finanzas, etc.)
TASK_NAME = "task_mnist"
task = importlib.import_module(TASK_NAME)

class GenericFlowerClient(fl.client.NumPyClient):
    def __init__(self, net):
        self.net = net

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
        
        trainloader, _, num_examples = task.load_data(batch_size)
        task.train(self.net, trainloader, lr)
        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, testloader, _ = task.load_data(batch_size=32)
        loss, accuracy, num_examples = task.test(self.net, testloader)
        return float(loss), num_examples, {"accuracy": float(accuracy)}

# --- NUEVA API DE FLOWER ---
def client_fn(cid: str):
    """Flower llama a esta función cada vez que el servidor solicita entrenar."""
    # Instanciamos un modelo nuevo y fresco para cada ronda/entrenamiento
    modelo_local = task.get_model()
    # Retornamos el cliente envuelto para la nueva API
    return GenericFlowerClient(modelo_local).to_client()

# Definimos la aplicación cliente (no hace falta if __name__ == "__main__")
app = fl.client.ClientApp(client_fn=client_fn)