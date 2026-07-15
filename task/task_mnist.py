# task_mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose


class SimpleMNIST(nn.Module):
    def __init__(self):
        super(SimpleMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_model():
    """Devuelve el modelo inicializado."""
    return SimpleMNIST()


def load_data(batch_size, cid=0, num_clients=2):
    """
    Carga MNIST en modo IID.

    Cada cliente recibe una partición aleatoria del dataset,
    pero con una mezcla equilibrada de clases.
    """

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    num_train = 1000
    num_test = 500

    # Semilla distinta por cliente para que no todos usen exactamente los mismos datos
    generator_train = torch.Generator().manual_seed(1234 + int(cid))
    generator_test = torch.Generator().manual_seed(4321 + int(cid))

    cliente_trainset, _ = random_split(
        trainset,
        [num_train, len(trainset) - num_train],
        generator=generator_train
    )

    cliente_testset, _ = random_split(
        testset,
        [num_test, len(testset) - num_test],
        generator=generator_test
    )

    trainloader = DataLoader(
        cliente_trainset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        cliente_testset,
        batch_size=32
    )

    return trainloader, testloader, len(cliente_trainset)


def train(net, trainloader, lr):
    """Entrena el modelo localmente."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    net.train()

    for _ in range(3):
        for images, labels in trainloader:
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Evalúa el modelo y devuelve loss media, accuracy y número de ejemplos."""

    criterion = nn.CrossEntropyLoss()

    net.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)
            num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return float(avg_loss), float(accuracy), total