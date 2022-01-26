import pdb
import scipy.sparse as sp
import numpy as np
import dgl

import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch


def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

def generate_auxiliary(feats):

    def compute_similarity(a, b, eps=1e-8):
        """
        eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    sim_mat = compute_similarity(feats,feats)
    thresh = torch.mean(sim_mat)  + 0.75*torch.std(sim_mat)
    sim_mat = torch.where(sim_mat>thresh, sim_mat, torch.zeros_like(sim_mat))
    adj = sp.coo_matrix(sim_mat.cpu().detach().numpy(), dtype=np.float32)
    G_aux = dgl.from_scipy(adj, eweight_name='sim')

    return G_aux


def total_var_loss(x,y):
    res = torch.abs(x.repeat(1,x.shape[0]) - y.T.repeat(y.shape[0],1))
    res = res.T*res
    return res

def get_dense_adj(g):
    g = g.to(torch.device('cuda:0'))
    src, dst = g.edges()
    block_adj = torch.zeros(g.num_src_nodes(), g.num_dst_nodes(), device='cuda:0')
    block_adj[src, dst] = g.edata['sim'].squeeze(-1)

    return block_adj


class VarFairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(VarFairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.aux_graph = generate_auxiliary(args.data)
        # self.vdnet = VarDropoutNetwork(self.aux_graph)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.alpha = 1e-4

        self.G_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)

        return y

    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ### update E, G
        self.optimizer_G.zero_grad()

        h = self.GNN(g,x)
        y = self.classifier(h)

        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())

        # pdb.set_trace()
        adj_mat = get_dense_adj(self.aux_graph)
        self.lip_loss = torch.clamp(torch.sum(total_var_loss(y,y) * adj_mat)-1, min=0)

        self.G_loss = self.cls_loss + self.alpha*self.lip_loss
        self.G_loss.backward()
        self.optimizer_G.step()
