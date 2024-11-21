import torch
import numpy as np
from torch import nn
from .gcnii.air_gcnii import AIRGCNII

class GNN(nn.Module):
    def __init__(self, num_node_type, node_type_embedding, hidden_feat_dim: int, device: torch.device):
        
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_type, node_type_embedding)

        self.gcn = AIRGCNII(
            num_hidden=hidden_feat_dim,
            num_layers=4,
            drop_out=0.2,
            drop_edge=0,
            lambda_=0.5
        )

        self.gcn_for_one_hot = AIRGCNII(
            num_hidden=hidden_feat_dim + num_node_type,
            num_layers=4,
            drop_out=0.2,
            drop_edge=0,
            lambda_=0.5
        )

        self.layer_norm = nn.LayerNorm(hidden_feat_dim)
        self.hidden_feat_dim = hidden_feat_dim
        self.num_node_type = num_node_type

        self.device = device

    def forward(self, inputs):
        (
            one_hot_encoding,
            mention_hidden_state,
            entity_hidden_state,
            sent_hidden_state,
            token_hidden_state,
            graph,
        ) = inputs

        # print(one_hot_encoding.shape)
        # original_o = np.random.rand(179, 179)
        num_node = one_hot_encoding.shape[0]
        batch_size, num_mention, _ = mention_hidden_state.shape
        num_entity = int(entity_hidden_state.shape[1])
        num_sent = int(sent_hidden_state.shape[1])
        num_token = int(token_hidden_state.shape[1])

        new_one_hot_encoding = torch.zeros((num_node, self.hidden_feat_dim + self.num_node_type), device=self.device)
        new_one_hot_encoding[:num_node, self.num_node_type: self.num_node_type +num_node] = one_hot_encoding

        new_one_hot_encoding[:num_mention, 0]= 1
        new_one_hot_encoding[num_mention:num_mention+num_entity, 1] = 1
        new_one_hot_encoding[num_mention+num_entity:num_mention+num_entity+num_sent, 2] = 1
        new_one_hot_encoding[num_mention+num_entity+num_sent:num_mention+num_entity+num_sent+num_token,3] = 1
    
    
        
    
        mention_type_embedding = self.node_type_embedding(torch.tensor(0).to(self.device)).view(1, 1, -1)
        entity_type_embedding = self.node_type_embedding(torch.tensor(1).to(self.device)).view(1, 1, -1)
        sent_type_embedding = self.node_type_embedding(torch.tensor(2).to(self.device)).view(1, 1, -1)
        token_type_embedding = self.node_type_embedding(torch.tensor(3).to(self.device)).view(1, 1, -1)
        
        mention_type_embedding = torch.broadcast_to(mention_type_embedding, (batch_size, num_mention, -1))
        entity_type_embedding = torch.broadcast_to(entity_type_embedding, (batch_size, num_entity, -1))
        sent_type_embedding = torch.broadcast_to(sent_type_embedding, (batch_size, num_sent, -1))
        token_type_embedding = torch.broadcast_to(token_type_embedding, (batch_size, num_token, -1))
        
        mention_hidden_state = torch.concat((mention_hidden_state, mention_type_embedding), dim=2)
        entity_hidden_state = torch.concat((entity_hidden_state, entity_type_embedding), dim=2)
        sent_hidden_state = torch.concat((sent_hidden_state, sent_type_embedding), dim=2)
        token_hidden_state = torch.concat((token_hidden_state, token_type_embedding), dim=2)
        
        node_hidden_state = torch.concat((mention_hidden_state, entity_hidden_state, sent_hidden_state, token_hidden_state), dim=1)
        num_node = int(node_hidden_state.shape[1])
        node_hidden_state = torch.reshape(node_hidden_state, (batch_size * num_node, -1))

        output_node_hidden_state = self.gcn(graph, node_hidden_state)
        output_one_hot_encoding = self.gcn_for_one_hot(graph, new_one_hot_encoding)
        output_node_hidden_state = torch.cat([output_one_hot_encoding, output_node_hidden_state],dim=1)
        
        output_node_hidden_state = torch.reshape(output_node_hidden_state, (batch_size, num_node, -1))
        entity_hidden_state = output_node_hidden_state[:, num_mention:num_mention + num_entity]
        output_node_hidden_state = torch.reshape(output_node_hidden_state, (batch_size * num_node, -1))
        
        return entity_hidden_state, output_node_hidden_state