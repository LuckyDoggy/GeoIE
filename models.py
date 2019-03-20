import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
class GeoIE(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,neg_num):
        super(GeoIE,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.negnum=neg_num
        self.a=0.1
        self.b=-0.2
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
        nn.init.xavier_normal(self.GeoInfluence.weight)
        nn.init.xavier_normal(self.GeoSusceptibility.weight)
    def forward(self,cuj,pos_u,pos_p,neg_p,History,distance):
        wuj=1+math.log(1+cuj*(10**self.scaling))
        fij=[]
        for i in range(len(distance)):
            line=[]
            for j in range(len(distance[0])):
                tmp=self.a*distance[i][j]**self.b
                line.append(tmp)
            fij.append(line)
        UPre=self.UserPreference(Variable(torch.LongTensor(pos_u)))
        PPre=self.PoiPreference(Variable(torch.LongTensor(pos_p)))
        #NegPPre = self.PoiPreference(torch.LongTensor(neg_p))
        Hnum=len(History)
        loss=[]
        if Hnum:
            hj=self.GeoSusceptibility(Variable(torch.LongTensor(pos_p)))
            g=self.GeoInfluence(Variable(torch.LongTensor(History)))
            f=torch.FloatTensor([fij[0]])
            posresult=UPre.mm(PPre.t())+(f.mul((hj.mm(g.t())))).sum()/float(Hnum)
            posresult=F.logsigmoid(posresult)
            loss.append(posresult)
            for j in range(self.negnum):
                f=torch.FloatTensor([fij[j+1]])
                NegPPre = self.PoiPreference(Variable(torch.LongTensor([neg_p[j]])))
                hj=self.GeoSusceptibility(Variable(torch.LongTensor([neg_p[j]])))
                negresult=UPre.mm(NegPPre.t())+(f.mul((hj.mm(g.t())))).sum()/float(Hnum)
                negresult=F.logsigmoid(-negresult)
                loss.append(negresult)
        else:
            posresult = UPre.mm(PPre.t())
            posresult = F.logsigmoid(posresult)
            loss.append(posresult)
            for j in range(self.negnum):
                NegPPre = self.PoiPreference(Variable(torch.LongTensor([neg_p[j]])))
                negresult=UPre.mm(NegPPre.t())
                negresult=F.logsigmoid(-negresult)
                loss.append(negresult)
        return -1*wuj*sum(loss)

'''
class testmodel(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension):
        super(testmodel,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.a=0.1
        self.b=-0.2
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
        nn.init.xavier_normal(self.GeoInfluence.weight)
        nn.init.xavier_normal(self.GeoSusceptibility.weight)
    def forward(self,pos_u,pos_p,History,distance):
        fij=[]
        for i in range(len(distance)):
            line=[]
            for j in range(len(distance[0])):
                tmp=self.a*distance[i][j]**self.b
                line.append(tmp)
            fij.append(line)
        UPre=self.UserPreference(Variable(torch.LongTensor(pos_u)))
        PPre=self.PoiPreference(Variable(torch.LongTensor(pos_p)))
        #NegPPre = self.PoiPreference(torch.LongTensor(neg_p))
        Hnum=len(History)
        if Hnum:
            hj=self.GeoSusceptibility(Variable(torch.LongTensor(pos_p)))
            g=self.GeoInfluence(Variable(torch.LongTensor(History)))
            f=torch.FloatTensor([fij[0]])
            posresult=UPre.mm(PPre.t())+(f.mul((hj.mm(g.t())))).sum()/float(Hnum)
            posresult=F.logsigmoid(posresult)
        else:
            posresult = UPre.mm(PPre.t())
            posresult = F.logsigmoid(posresult)
        return posresult
'''
class testmodel(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension):
        super(testmodel,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.a=0.1
        self.b=-0.2
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
        nn.init.xavier_normal(self.GeoInfluence.weight)
        nn.init.xavier_normal(self.GeoSusceptibility.weight)
    def forward(self,user):
        emb_u=self.UserPreference(torch.LongTensor(user))
        list=torch.mm(self.PoiPreference.weight,emb_u.t())
        list=list.view(-1)
        return list