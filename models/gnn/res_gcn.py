from dgl import DropEdge, remove_self_loop, add_self_loop
from torch import nn, Tensor
from dgl.nn.pytorch import GraphConv
import dgl


class ResGCN(nn.Module):
    def __init__(self, in_feat_dim: int, num_layers: int, drop_out: float, drop_edge: float):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(in_feat_dim, in_feat_dim, activation=self.activation)
                                     for _ in range(num_layers)])
        self.drop_out = nn.Dropout(drop_out)
        self.drop_edge = DropEdge(drop_edge)

    def forward(self, g: dgl.DGLGraph, in_feat: Tensor):
        with g.local_scope():
            x = in_feat
            for layer_id, layer in enumerate(self.layers):
                if layer_id != 0:
                    x = self.drop_out(x)
                drop_g = self.drop_edge(g)
                drop_g = remove_self_loop(drop_g)
                drop_g = add_self_loop(drop_g)
                x = (layer(drop_g, x) + x) / 2
            return x
