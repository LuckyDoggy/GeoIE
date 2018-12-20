import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from model import GeoIE
net=GeoIE(2,5,4,2)
optimizer = optim.SGD(net.parameters(), lr=0.005,momentum=0.9)
for i in range(1000):
    optimizer.zero_grad()
    loss=net.forward(2,[0],[0],[2,3],[2,3],[[1,2],[1,2],[1,2]])
    print(loss)
    loss.backward()
    optimizer.step()