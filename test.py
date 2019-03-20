import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from models import GeoIE
from models import testmodel
net=testmodel(2,5,4)
"""
optimizer = optim.SGD(net.parameters(), lr=0.005,momentum=0.9)
for i in range(1000):
    optimizer.zero_grad()
    loss=net.forward([0],[0],[2,3],[[1.2,2]])
    print(loss)
    loss.backward()
    optimizer.step()
"""
loss=net.forward([0])
print(loss)
print(net.UserPreference.weight)
