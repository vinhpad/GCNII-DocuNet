import dgl
from dgl.nn.pytorch import GATConv
from torch import nn, Tensor
import torch
from .gcnii import cal_gain
import torch.nn.functional as F


class SingleHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.elu, batch_norm=False, residual=False, dropout=0, alpha=0.1):
        super(SingleHeadGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        self.alpha = alpha

    """这儿有点问题"""
    def reset_parameters(self):
        gain = cal_gain("leaky_relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=cal_gain(self.activation))

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, features):
        g = graph.local_var()
        h_pre = features
        z = self.dropout(features)
        z = self.fc(z)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        if self.batch_norm:
            h = self.bn(h)
        if self.activation:
            h = self.activation(h)
        if self.residual:
            h = (1 - self.alpha) * h + self.alpha * self.res_fc(h_pre)
        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='mean', activation=F.elu,
                 batch_norm=False, residual=False, dropout: float = 0, alpha: float = 0.1):
        super(GATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SingleHeadGATLayer(in_dim, out_dim, activation, batch_norm, residual, dropout, alpha))
        self.merge = merge

    def forward(self, g, features):
        head_outs = [attn_head(g, features) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)


class GATNet(nn.Module):
    def __init__(self, in_feat_dim: int, num_heads: int,
                 num_layers: int, drop_out: float):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.layers = nn.ModuleList([GATLayer(in_feat_dim, in_feat_dim, num_heads,
                                              residual=True, dropout=drop_out, alpha=0.5) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(in_feat_dim) for _ in range(num_layers)])

    def forward(self, g: dgl.DGLGraph, in_feat: Tensor):
        with g.local_scope():
            x = in_feat
            num_nodes = in_feat.shape[0]
            torch.save(x, "./debug/gat/x.pth")
            for layer_id, layer in enumerate(self.layers):
                x = layer(g, x)
                x = x.view((num_nodes, -1))
                # x = self.norm_layers[layer_id](x)
                torch.save(x, f"./debug/gat/x_{layer_id}.pth")
                x = self.activation(x)
        return x
