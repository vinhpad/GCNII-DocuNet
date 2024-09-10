import dgl
import torch

from torch import nn, Tensor
from dgl.nn.pytorch import GraphConv
from dgl.transforms import remove_self_loop, add_self_loop

class GCN(nn.Module):
    def __init__(self, in_feat_dim: int, num_layers: int):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.num_layers = num_layers
        self.activate = nn.LeakyReLU()
        self.gcn_layers = nn.ModuleList([GraphConv(in_feat_dim, in_feat_dim, activation=self.activate)
            for _ in range(self.num_layers)])        

    def forward(self, g: dgl.DGLGraph, in_feat: Tensor):
        with g.local_scope():
            x = in_feat
            for _, gcn_layer in enumerate(self.gcn_layers):
                x = gcn_layer(g, x)
            return x


class GNN(nn.Module):
    def __init__(self, node_type_embedding, num_layers, in_feat_dim: int, out_feat_dim, device: torch.device):
        super().__init__()
        self.node_type_embedding = nn.Embedding(3, node_type_embedding)
        self.gcn = GCN(in_feat_dim, num_layers)
        self.layer_norm = nn.LayerNorm(in_feat_dim)
        self.fc = nn.Linear(in_feat_dim, out_feat_dim)
        self.device = device

    def forward(self, inputs):
        (
            mention_hidden_state,
            entity_hidden_state,
            sent_hidden_state,
            graph,
        ) = inputs
        
        batch_size, num_mention, _ = mention_hidden_state.shape
        num_entity = int(entity_hidden_state.shape[1])
        num_sent = int(sent_hidden_state.shape[1])
        
        mention_type_embedding = self.node_type_embedding(torch.tensor(0).to(self.device)).view(1, 1, -1)
        entity_type_embedding = self.node_type_embedding(torch.tensor(1).to(self.device)).view(1, 1, -1)
        sent_type_embedding = self.node_type_embedding(torch.tensor(2).to(self.device)).view(1, 1, -1)

        mention_type_embedding = torch.broadcast_to(mention_type_embedding, (batch_size, num_mention, -1))
        entity_type_embedding = torch.broadcast_to(entity_type_embedding, (batch_size, num_entity, -1))
        sent_type_embedding = torch.broadcast_to(sent_type_embedding, (batch_size, num_sent, -1))
       
        mention_hidden_state = torch.concat((mention_hidden_state, mention_type_embedding), dim=2)
        entity_hidden_state = torch.concat((entity_hidden_state, entity_type_embedding), dim=2)
        sent_hidden_state = torch.concat((sent_hidden_state, sent_type_embedding), dim=2)
        
        node_hidden_state = torch.concat((mention_hidden_state, entity_hidden_state, sent_hidden_state), dim=1)
        node_hidden_state = self.layer_norm(node_hidden_state)  # Try out

        num_node = int(node_hidden_state.shape[1])
        node_hidden_state = torch.reshape(node_hidden_state, (batch_size * num_node, -1))

        output_node_hidden_state = self.fc(self.gcn(graph, node_hidden_state))

        output_node_hidden_state = torch.reshape(output_node_hidden_state, (batch_size, num_node, -1))
        entity_hidden_state = output_node_hidden_state[:, num_mention:num_mention + num_entity]
      
        return entity_hidden_state