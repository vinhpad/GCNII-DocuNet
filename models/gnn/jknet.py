import dgl
import torch
from dgl import DropEdge, remove_self_loop, add_self_loop
from torch import nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class JKNet(nn.Module):
    def __init__(self, in_feat_dim: int, num_layers: int, drop_out: float, drop_edge: float):
        super().__init__()
        self.drop_out = nn.Dropout(drop_out)
        self.activation = nn.LeakyReLU()
        self.layers = [GraphConv(in_feat_dim, in_feat_dim, activation=None) for _ in range(num_layers)]
        self.drop_edge = DropEdge(p=drop_edge)

    def forward(self, g: dgl.DGLGraph, in_feat):
        with g.local_scope():
            feat_list = [in_feat]
            x = in_feat
            for layer_id, layer in enumerate(self.layers):
                x = self.drop_out(x)
                drop_g = self.drop_edge(g)
                drop_g = remove_self_loop(drop_g)
                drop_g = add_self_loop(drop_g)
                x = layer(drop_g, x)
                x = self.activation(x)
                feat_list.append(x)
            out = torch.max(torch.stack(feat_list, dim=-1), dim=-1)
        return out
