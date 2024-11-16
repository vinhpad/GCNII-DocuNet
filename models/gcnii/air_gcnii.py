import math

from dgl import DropEdge, remove_self_loop, add_self_loop
from torch import nn
import torch
import dgl.function as fn
from torch.nn import functional as F
from torch import nn, Tensor

from models.gcnii.gcnii import cal_gain

class GatedLayer(nn.Module):
    def __init__(self, in_feat_dim: int):
        super().__init__()
        self.first_linear = nn.Linear(in_feat_dim, in_feat_dim, bias=False)
        self.second_linear = nn.Linear(in_feat_dim, in_feat_dim, bias=True)

    def forward(self, first_input: Tensor, second_input: Tensor):
        scale = torch.sigmoid(self.first_linear(first_input) + self.second_linear(second_input))
        return first_input * scale + second_input * (1 - scale)

class AIRGCNIILayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None, graph_norm=True,
                 beta: float = 0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.beta = beta
        self.reset_parameters()
        self.gate_alpha_layer = GatedLayer(in_dim)
        self.gate_beta_layer = GatedLayer(in_dim)

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, graph, features, initial_features):
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            h = features * norm
        g.ndata['h'] = h
        w = torch.ones(g.number_of_edges(), 1).to(features.device)
        g.edata['w'] = w
        g.update_all(fn.u_mul_e('h', 'w', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        if self.graph_norm:
            h = h * norm
        h = self.gate_alpha_layer(h, initial_features)
        # h = self.gate_beta_layer(h, self.linear(h))
        h = (1 - self.beta) * h + self.beta * self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class AIRGCNII(nn.Module):
    def __init__(self, num_hidden, num_layers, bias=True,
                 activation=None, graph_norm=True, drop_out=0, lambda_=0,
                 drop_edge=0):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            beta = math.log(lambda_ / (i + 1) + 1)
            self.convs.append(AIRGCNIILayer(num_hidden, num_hidden, bias, activation,
                                            graph_norm, beta))
        self.params1 = list(self.convs.parameters())
        self.dropout = drop_out
        self.drop_edge = DropEdge(p=drop_edge)

    def forward(self, graph, features):
        h0 = F.dropout(features, self.dropout, self.training)
        h = h0
        # torch.save(h, f"debug/gcnii/x.pth")
        for layer_id, con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, self.training)
            drop_g = self.drop_edge(graph)
            drop_g = remove_self_loop(drop_g)
            drop_g = add_self_loop(drop_g)
            h = con(drop_g, h, h0)
            # torch.save(h, f"debug/gcnii/x_{layer_id}.pth")
            h = self.activation(h)
        h = F.dropout(h, self.dropout, self.training)
        return h