import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math
import csv
import random
from models import GeoIE

def cal_distance(pid,hid,POI_info):
    result=[]
    for id in hid:
        d = math.sqrt((float(POI_info[str(id)][0]) - float(POI_info[str(pid)][0])) ** 2 + (float(POI_info[str(id)][1]) - float(POI_info[str(pid)][1])) ** 2)
        if d==0:
            d=0.01
        result.append(d)
    return result

emb_dimension=200
user_count=2580
poi_count=2578
net=GeoIE(user_count,poi_count,emb_dimension,5)
memory_rate=0.7
print("模型初始化完成")
file1=open('./data/POI_info.csv','r')
POI_info={}
reader=csv.reader(file1)
for line in reader:
    POI_info.setdefault(line[0],line[2:4])
f=open('./data/attentiondata.csv','r')
reader=csv.reader(f)
data=[]
reader=csv.reader(f)
for line in reader:
    if len(line)<3:
        continue
    l=[]
    user_id=[]
    history_id=[]
    label=[]
    for i in range(len(line)):
        if i==0:
            user_id.append(int(line[i]))
            continue
        if i<=(len(line)-1)*memory_rate:
            history_id.append(int(line[i]))
            continue
        else:
            label.append(int(line[i]))
    for target in label:
        c=label.count(target)
        neg=[]
        while(len(neg)<5):
            n=random.randint(0,2577)
            if n not in label and n not in history_id:
                neg.append(n)
        distance=[]
        distance.append(cal_distance(target,history_id,POI_info))
        for id in neg:
            distance.append(cal_distance(id,history_id,POI_info))
        l = [c,user_id, [target],neg,history_id,distance]
    data.append(l)

print("数据加载完成")
optimizer = optim.SGD(net.parameters(), lr=0.005,momentum=0.9)
total_loss=0.0
total_loss=0.0
for epoch in range(5):
    random.shuffle(data)
    for i in range(len(data)):
        if i%1000==999:
            print(total_loss/1000)
            total_loss=0
        optimizer.zero_grad()
        loss=net.forward(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
print("训练完成")
filename='./model/GeoIEmodel(canshu)'+str(emb_dimension)+'.pkl'
torch.save(net.state_dict(), filename)
