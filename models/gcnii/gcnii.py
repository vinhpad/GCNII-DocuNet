import math

from dgl import DropEdge, remove_self_loop, add_self_loop
from torch import nn
import torch
import dgl.function as fn
from torch.nn import functional as F


def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


class GCNIILayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None, graph_norm=True,
                 alpha: float = 0,
                 beta: float = 0):
        super(GCNIILayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.alpha = alpha
        self.beta = beta
        self.reset_parameters()

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
        h = (1 - self.alpha) * h + self.alpha * initial_features
        h = (1 - self.beta) * h + self.beta * self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GCNII(nn.Module):
    def __init__(self, num_hidden, num_layers, bias=True,
                 activation=None, graph_norm=True, drop_out=0, alpha=0, lambda_=0,
                 drop_edge=0):
        super(GCNII, self).__init__()
        self.activation = nn.LeakyReLU()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            beta = math.log(lambda_ / (i + 1) + 1)
            self.convs.append(GCNIILayer(num_hidden, num_hidden, bias, activation,
                                         graph_norm, alpha, beta))
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