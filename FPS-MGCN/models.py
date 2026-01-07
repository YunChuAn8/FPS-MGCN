import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from utils import *

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class FPS_MGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, n_clusters,fourier_alpha):
        super(FPS_MGCN, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.FRGCN = GCN(2 * nfeat, nhid1, nhid2, dropout)
        self.FRCGCN = GCN(2 * nfeat, nhid1, nhid2, dropout)

        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )

        self.cls = nn.Sequential(
            nn.Linear(nhid2, n_clusters),
        )

        self.alpha = fourier_alpha

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN(x, sadj)
        com1 = self.CGCN(x, sadj)
        com2 = self.CGCN(x, fadj)
        emb2 = self.FGCN(x, fadj)

        fourier_x = fourier_transform(x, alpha=self.alpha)
        emb_e = self.FRGCN(fourier_x, sadj)

        com_avg = (com1 + com2) / 2

        emb_stack = torch.stack([emb1, com_avg, emb2, emb_e], dim=1)
        emb, att = self.att(emb_stack)
        emb = self.MLP(emb)
        [pi, disp, mean] = self.ZINB(emb)

        cluster_logits = self.cls(emb)
        
        return com1, com2,emb, pi, disp, mean, cluster_logits

    def __repr__(self):
        return self.__class__.__name__ + ' (nfeat=' + str(self.SGCN.gc1.in_features) + ' -> nhid2=' + str(
            self.SGCN.gc2.out_features) + ')'

