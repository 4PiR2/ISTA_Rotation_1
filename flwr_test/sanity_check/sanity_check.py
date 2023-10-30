from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pefll_dataset import gen_random_loaders, get_datasets

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=2e-3, momentum=.9, weight_decay=5e-5)
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # correct += (torch.max(outputs.data, 1)[1] == torch.max(labels, 1)[1]).sum().item()
    return loss / len(testloader.dataset), correct / total


def load_data():
    """Load CIFAR-10 (training and test set)."""
    # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trf = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = CIFAR10("./dataset", train=True, download=True, transform=trf)
    testset = CIFAR10("./dataset", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset, batch_size=32, shuffle=False)


NUM_CLIENTS = 5
trainloaders, valloaders, testloaders = gen_random_loaders(
    data_name='cifar10', data_path='./dataset', num_users=NUM_CLIENTS, num_train_users=NUM_CLIENTS, bz=32,
    partition_type='by_class', classes_per_user=2, alpha_train=None, alpha_test=None, embedding_dir_path=None
)

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

state_dict = torch.load('output/model_round_10.pth')
net.load_state_dict(state_dict)

t_all, v_all, tt_all = get_datasets('cifar10', './dataset', one_hot=False)
# out = test(net, DataLoader(t_all, batch_size=32))
# print(out)
# out = test(net, DataLoader(v_all, batch_size=32))
# print(out)
out = test(net, DataLoader(tt_all, batch_size=32))
print(out)
# out = test(net, trainloader)
# print(out)
# out = test(net, testloader)
# print(out)


for epoch in range(100):
    train(net, trainloader, 1)
    ret = test(net, testloader)
    print(epoch, ret)

exit(0)
