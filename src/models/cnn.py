import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.l1 = nn.Linear(7*7*64, 1)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(2,2)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h = self.pool(self.act(self.bn1(self.conv1(x))))
        h = self.pool(self.act(self.bn2(self.conv2(h)))) 
        h = self.flat(h)
        h = self.l1(h)
        h = self.sigmoid(h)
        return h
        

if __name__ == "__main__":

    model = CNN()

    dummy = torch.randn(32, 1, 28, 28)

    output = model(dummy)

    print("Output shape:", output.shape)