import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class GeoIE(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,scaling,neg_num):
        super(GeoIE,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=scaling
        self.negnum=neg_num
        self.a=0.1
        self.b=2
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.init_emb()
    def init_emb(self):
        initrange=0.5/self.emb_dimension
        self.UserPreference.weight.data.uniform_(-initrange,initrange)
        self.PoiPreference.weight.data.uniform_(-initrange, initrange)
        self.GeoInfluence.weight.data.uniform_(-initrange, initrange)
        self.GeoSusceptibility.weight.data.uniform_(-initrange, initrange)
    def forward(self,cuj,pos_u,pos_p,neg_u,neg_p,H_pos,distance):
        wuj=1+math.log(1+cuj*(10**self.scaling))
        fij=[]
        for i in range(len(distance)):
            line=[]
            for j in range(len(distance[0])):
                tmp=self.a*distance[i][j]**self.b
                line.append(tmp)
            fij.append(line)


