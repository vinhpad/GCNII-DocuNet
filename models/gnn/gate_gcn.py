from torch import nn, Tensor
import torch
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.transforms import DropEdge, add_self_loop, remove_self_loop


class GatedLayer(nn.Module):
    def __init__(self, in_feat_dim: int):
        super().__init__()
        self.first_linear = nn.Linear(in_feat_dim, in_feat_dim, bias=False)
        self.second_linear = nn.Linear(in_feat_dim, in_feat_dim, bias=True)

    def forward(self, first_input: Tensor, second_input: Tensor):
        scale = torch.sigmoid(self.first_linear(first_input) + self.second_linear(second_input))
        return first_input * scale + second_input * (1 - scale)


class GateGCN(nn.Module):
    def __init__(self, in_feat_dim: int, num_layers: int, drop_out: float, share_gate_layer: bool,
                 drop_edge: float):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(in_feat_dim, in_feat_dim, activation=None)
                                     for _ in range(num_layers)])
        self.share_gate_layer = share_gate_layer
        if self.share_gate_layer:
            self.gated_layer = nn.ModuleList([GatedLayer(in_feat_dim) for _ in range(num_layers)])
        else:
            self.gated_layer = GatedLayer(in_feat_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.drop_edge = DropEdge(p=drop_edge)

    def forward(self, g: dgl.DGLGraph, in_feat: Tensor):
        with g.local_scope():
            x = in_feat
            for layer_id, layer in enumerate(self.layers):
                x = self.drop_out(x)
                drop_g = self.drop_edge(g)
                drop_g = remove_self_loop(drop_g)
                drop_g = add_self_loop(drop_g)
                next_x = layer(drop_g, x)
                next_x = self.activation(next_x)
                if not self.share_gate_layer:
                    x = self.gated_layer(x, next_x)
                else:
                    x = self.gated_layer[layer_id](x, next_x)
            x = self.drop_out(x)
            return x
