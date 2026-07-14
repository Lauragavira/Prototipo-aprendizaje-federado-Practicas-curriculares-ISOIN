# task_noIID_mnist.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

# Reutilizamos el mismo modelo, entrenamiento y test de MNIST
from task_mnist import get_model, train, test


def labels_for_client(cid, num_clients):
    """Define qué números verá cada cliente en modo No-IID."""

    grupos = {
        1: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        2: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        3: [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]],
        4: [[0, 1], [2, 3], [4, 5], [6, 7, 8, 9]],
    }

    return grupos[num_clients][cid]


def load_data_non_iid(batch_size, cid, num_clients):
    """Carga MNIST dando a cada cliente clases distintas."""

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    etiquetas_cliente = labels_for_client(cid, num_clients)

    train_targets = torch.as_tensor(trainset.targets)
    test_targets = torch.as_tensor(testset.targets)

    train_mask = torch.isin(train_targets, torch.tensor(etiquetas_cliente))
    test_mask = torch.isin(test_targets, torch.tensor(etiquetas_cliente))

    train_indices = torch.where(train_mask)[0].tolist()
    test_indices = torch.where(test_mask)[0].tolist()

    generator = torch.Generator().manual_seed(1234 + cid)

    train_indices = [
        train_indices[i]
        for i in torch.randperm(len(train_indices), generator=generator).tolist()[:1000]
    ]

    test_indices = [
        test_indices[i]
        for i in torch.randperm(len(test_indices), generator=generator).tolist()[:500]
    ]

    cliente_trainset = Subset(trainset, train_indices)
    cliente_testset = Subset(testset, test_indices)

    trainloader = DataLoader(cliente_trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(cliente_testset, batch_size=32)

    return trainloader, testloader, len(cliente_trainset)


def load_data(batch_size, cid=0, num_clients=2):
    """Alias por si se llama a load_data directamente."""
    return load_data_non_iid(batch_size, cid, num_clients)


def get_label_distribution(cid, num_clients):
    """Devuelve la distribución de clases del cliente."""

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST("./data", train=True, download=True, transform=transform)

    etiquetas_cliente = labels_for_client(cid, num_clients)
    targets = torch.as_tensor(trainset.targets)

    mask = torch.isin(targets, torch.tensor(etiquetas_cliente))
    indices = torch.where(mask)[0].tolist()

    generator = torch.Generator().manual_seed(1234 + cid)
    indices = [
        indices[i]
        for i in torch.randperm(len(indices), generator=generator).tolist()[:1000]
    ]

    selected_targets = targets[indices]
    counts = torch.bincount(selected_targets, minlength=10)

    return {str(i): int(counts[i]) for i in range(10)}