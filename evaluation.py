#coding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import csv
from models import GeoIE
from models import testmodel
emb_dimension=200
user_count=2580
poi_count=2578
trainnet=GeoIE(user_count,poi_count,emb_dimension,5)
trainnet.load_state_dict(torch.load('./model/GeoIEmodel(canshu)200.pkl'))
net=testmodel(user_count,poi_count,emb_dimension)
net.UserPreference=trainnet.UserPreference
net.PoiPreference=trainnet.PoiPreference
net.GeoSusceptibility=trainnet.GeoSusceptibility
net.GeoInfluence=trainnet.GeoInfluence
print("模型初始化完成")

topK=20
memory_rate=0.7
'''
file1=open('./data/POI_info.csv','r')
POI_info={}
reader=csv.reader(file1)
for line in reader:
    POI_info.setdefault(line[0],line[2:4])
f=open('./data/testset.csv','r')
reader=csv.reader(f)
user={}
for line in reader:
    user.setdefault(line[0],[])
    user[line[0]].append(int(line[4]))
data=[]
for key in user.keys():
    user_id = []
    history_id = []
    label = []
    if len(user[key])<5:
        continue
    user_id.append(int(key))
    for i in range(len(user[key])):
        if i<=(len(line))*memory_rate:
            history_id.append(int(user[key][i]))
        else:
            label.append(int(user[key][i]))
    label=set(label)
    label=list(label)
    l = [user_id, history_id, label]
    data.append(l)
print("数据加载完成")

Rec=0
T=0
count=0
for line in data:
    y=net(line[0],line[1])
    _, index = torch.topk(y, topK)
    index = index.data.numpy()
    index = index.tolist()
    target=line[2]
    Rec += topK
    T += len(target)
    for i in index:
        for j in target:
            if i==j:
                count+=1
                break
Precision=count/(Rec*1.0)
Recall=count/(T*1.0)
print(count)
print(Rec)
print(T)
print(Precision)
print(Recall)
'''
file1=open('./data/POI_info.csv','r')
POI_info={}
reader=csv.reader(file1)
for line in reader:
    POI_info.setdefault(line[0],line[2:4])
f=open('./data/testset.csv','r')
reader=csv.reader(f)
user={}
for line in reader:
    user.setdefault(line[0],[])
    user[line[0]].append(int(line[4]))
data=[]
for key in user.keys():
    user_id = []
    history_id = []
    label = []
    if len(user[key])<5:
        continue
    user_id.append(int(key))
    for i in range(len(user[key])):
        if i<=(len(line))*memory_rate:
            history_id.append(int(user[key][i]))
        else:
            label.append(int(user[key][i]))
    label=set(label)
    label=list(label)
    l = [user_id, history_id, label]
    data.append(l)
print("数据加载完成")
Rec=0
T=0
count=0
for line in data:
    y=net(line[0])
    _, index = torch.topk(y, topK)
    index = index.numpy()
    index = index.tolist()
    target = line[2]
    Rec += topK
    T += len(target)
    for i in index:
        for j in target:
            if i == j:
                count += 1
                break
Precision = count / (Rec * 1.0)
Recall = count / (T * 1.0)
print(count)
print(Rec)
print(T)
print(Precision)
print(Recall)