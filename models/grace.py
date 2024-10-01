import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.losses import ATLoss
from opt_einsum import contract
from models.gnn import GNN
from models.attn_unet import AttentionUNet
from utils import process_long_input

class GRACE(nn.Module):
    
    def __init__(self, config, bert_model):
        super().__init__()
        
        self.bert_model = bert_model
        self.bert_config = config.bert_config
        self.device = config.device
        self.tau = config.tau
        self.offset = 1
        self.gnn_in_feat_dim = config.bert_hidden_dim + config.gnn_node_type_embedding
        self.gnn = GNN(
            node_type_embedding=config.gnn_node_type_embedding,
            num_layers=config.gnn_num_layer,
            in_feat_dim=self.gnn_in_feat_dim,
            hidden_feat_dim=config.gnn_hidden_feat_dim,
            out_feat_dim=config.gnn_output_dim,
            device=config.device
        )

        self.fc1 = torch.nn.Linear(config.gnn_output_dim, config.grace_projection_hidden_feat_dim)
        self.fc2 = torch.nn.Linear(config.grace_projection_hidden_feat_dim, config.grace_projection_out_feat_dim)

    def encode(self, input_ids, attention_mask, entity_pos, sent_pos, num_mention, num_entity, num_sent, graph):
        
        config = self.bert_config

        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]

        sequence_output, _ = process_long_input(
            self.bert_model, 
            input_ids, 
            attention_mask, 
            start_tokens,
            end_tokens
        )
        
        mention_embed = self.get_mention_embed(sequence_output, entity_pos, num_mention)
        entity_embed = self.get_entity_embed(sequence_output, entity_pos, num_entity)
        sent_embed = self.get_sent_embed(sequence_output, sent_pos, num_sent)
        _, node_hidden_state = self.gnn([mention_embed, entity_embed, sent_embed, graph])

        return node_hidden_state

    def get_sent_embed(self, sequence_output, batch_sent_pos, num_sent):
        batch_size, _, embed_dim = sequence_output.shape
        sent_embed = torch.zeros((batch_size, num_sent, embed_dim)).to(self.device)
        for batch_id, sent_pos in enumerate(batch_sent_pos):
            for sent_id, pos in enumerate(sent_pos):
                sent_embed[batch_id, sent_id] = sequence_output[batch_id, pos[0] + self.offset]
        return sent_embed

    def get_mention_embed(self, sequence_output, batch_entity_pos, num_mention):
        batch_size, _, embed_dim = sequence_output.shape
        print(num_mention)
        mention_embed = torch.zeros((batch_size, num_mention, embed_dim)).to(self.device)
        for batch_id, entity_pos in enumerate(batch_entity_pos):
            mention_id = 0
            for ent_pos in entity_pos:
                for mention_pos in ent_pos:
                    mention_embed[batch_id, mention_id] = sequence_output[batch_id, mention_pos[0] + self.offset]
                    mention_id += 1
        return mention_embed

    def get_entity_embed(self, sequence_output, batch_entity_pos, num_entity):
        batch_size, _, embed_dim = sequence_output.shape
        entity_embed = torch.zeros((batch_size, num_entity, embed_dim)).to(self.device)
        for batch_id, entity_pos in enumerate(batch_entity_pos):
            for entity_id, ent_pos in enumerate(entity_pos):
                embeds = []
                for mention_pos in ent_pos:
                    embeds.append(sequence_output[batch_id, mention_pos[0] + self.offset])
                entity_embed[batch_id, entity_id] = torch.logsumexp(torch.stack(embeds, dim=0), dim=0)
        return entity_embed

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag()/(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
    
    def projection(self, features):
        features = self.fc1(features)
        features = self.fc2(features)
        return features

    def forward(self, features, attention_mask, entity_pos, sent_pos, graph, num_mention, num_entity, num_sent):

        features = self.encode(features, attention_mask, entity_pos, 
                               sent_pos, num_mention, num_entity, num_sent, graph)

        features = self.projection(features)
        
        return features