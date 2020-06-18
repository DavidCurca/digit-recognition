import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import numpy as np

train = datasets.MNIST("", train=True, download=False,
                       transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=False,
                       transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
net = Net()
X = torch.rand((28, 28))
X = X.view(-1, 28*28)
output = net(X)
optimizer = optim.Adam(net.parameters(), lr=0.001)


EPOCHS = int(input("Number of epochs?: "))
for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print("Loss: ", loss)

correct = 0
total = 0

#net = Net()
#net.load_state_dict(torch.load('model.pth'))
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if(torch.argmax(i) == y[idx]):
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))
torch.save(net.state_dict(), "model.pth")
