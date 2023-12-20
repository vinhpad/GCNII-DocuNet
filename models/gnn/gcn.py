import dgl
from dgl.nn.pytorch import GraphConv
from torch import nn, Tensor
from dgl.transforms import DropEdge, remove_self_loop, add_self_loop


class GCN(nn.Module):
    def __init__(self, in_feat_dim: int, num_layers: int, dropout: float, drop_edge: float):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.num_layers = num_layers
        self.activate = nn.LeakyReLU()
        self.gcn_layers = nn.ModuleList([GraphConv(in_feat_dim, in_feat_dim, activation=self.activate)
                                         for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.drop_edge = DropEdge(drop_edge)

    def forward(self, g: dgl.DGLGraph, in_feat: Tensor):
        with g.local_scope():
            x = in_feat
            for layer_id, gcn_layer in enumerate(self.gcn_layers):
                if layer_id != 0:
                    x = self.dropout(x)
                drop_g = self.drop_edge(g)
                drop_g = remove_self_loop(drop_g)
                drop_g = add_self_loop(drop_g)
                x = gcn_layer(drop_g, x)
            return x
