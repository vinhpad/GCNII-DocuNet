import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.losses import balanced_loss as ATLoss
from opt_einsum import contract
from gnn import GNN
from models.attn_unet import AttentionUNet
from utils import process_long_input
from augmentation_graph import augmentation
from .grace import GRACE

class DocREModel(nn.Module):
    def __init__(self, args, bert_model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        
        # BERT model config
        self.bert_model = bert_model
        self.bert_config = args.bert_config
        self.bert_drop = nn.Dropout(0.6)
        self.bert_hidden_size = args.bert_config.hidden_size
        
        # graph neural network model config
        self.gnn_config = args.gnn_config
        self.gnn_hidden_size = self.bert_hidden_size + self.gnn_config.node_type_embedding
        self.gnn = GNN(self.gnn_config, self.gnn_hidden_size, args.device)
        
        # GRACE model config
        self.grace_config = args.grace_config
        self.grace_model = GRACE(self.grace_config).to(args.device)  
        
        # Model config
        self.head_extractor = nn.Linear(args.grace_hidden_dim + self.hidden_size + args.node_type_embedding + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(args.grace_hidden_dim + self.hidden_size + args.node_type_embedding + args.unet_out_dim, emb_size)
        self.binary_linear = nn.Linear(emb_size * block_size, self.bert_config.num_labels)
        self.num_labels = num_labels
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = ATLoss()
        self.device = args.device
        self.offset = 1

    def encode(self, input_ids, attention_mask):
        config = self.bert_config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
            
        sequence_output, attention = process_long_input(self.bert_model, input_ids, attention_mask, start_tokens,
                                                        end_tokens)
        return sequence_output, attention

    def get_sent_embed(self, sequence_output, batch_sent_pos, num_sent):
        batch_size, _, embed_dim = sequence_output.shape
        sent_embed = torch.zeros((batch_size, num_sent, embed_dim)).to(self.device)
        for batch_id, sent_pos in enumerate(batch_sent_pos):
            for sent_id, pos in enumerate(sent_pos):
                sent_embed[batch_id, sent_id] = sequence_output[batch_id, pos[0] + self.offset]
        return sent_embed

    def get_mention_embed(self, sequence_output, batch_entity_pos, num_mention):
        batch_size, _, embed_dim = sequence_output.shape
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

    def get_pair_entity_embed(self, entity_hidden_state, hts):
        s_embed, t_embed = [], []
        for batch_id, ht in enumerate(hts):
            for pair in ht:
                s_embed.append(entity_hidden_state[batch_id, pair[0]])
                t_embed.append(entity_hidden_state[batch_id, pair[1]])
        s_embed = torch.stack(s_embed, dim=0)
        t_embed = torch.stack(t_embed, dim=0)
        return s_embed, t_embed


    def get_virtual_embed(self, sequence_output, batch_virtual_pos, num_virtual):
        batch_size, _, embed_dim = sequence_output.shape
        virtual_embed = torch.zeros((batch_size, num_virtual, embed_dim)).to(self.device)
        for batch_id, virtual_pos in enumerate(batch_virtual_pos):
            for virtual_id, vir_pos in enumerate(virtual_pos):
                if vir_pos[0] == vir_pos[1]:
                    virtual_embed[batch_id][virtual_id] = sequence_output[batch_id][vir_pos[0] + self.offset]
                else:
                    embeds = []
                    for _, token_pos in enumerate(virtual_pos):
                        if vir_pos[0] <= token_pos[0] < vir_pos[1]:
                            embeds.append(sequence_output[batch_id][token_pos[0] + self.offset])

                    virtual_embed[batch_id][virtual_id] = torch.logsumexp(torch.stack(embeds, dim=0), dim=0)
        return virtual_embed

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.bert_config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            for _ in range(self.min_height-entity_num-1):
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return entity_as

    def get_channel_map(self, sequence_output, entity_as):
        bs,_,d = sequence_output.size()
        ne = self.min_height

        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(self, 
                input_ids, 
                attention_mask,
                entity_pos, 
                sent_pos, 
                virtual_pos,
                graph, 
                num_mention, 
                num_entity, 
                num_sent, 
                num_virtual,
                labels=None, 
                hts=None
            ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        mention_embed = self.get_mention_embed(sequence_output, entity_pos, num_mention)
        entity_embed = self.get_entity_embed(sequence_output, entity_pos, num_entity)
        sent_embed = self.get_sent_embed(sequence_output, sent_pos, num_sent)
        virtual_embed = self.get_virtual_embed(sequence_output, virtual_pos, num_virtual)
        entity_hidden_state, output_node_hidden_state = self.gnn([mention_embed, entity_embed, sent_embed, virtual_embed, graph])
        entity_hidden_state_grace = self.grace_model.projection(entity_embed)

        local_context = self.get_hrt(sequence_output, attention, entity_pos, hts)
        s_embed, t_embed = self.get_pair_entity_embed(entity_hidden_state, hts)
        s_embed_grace, t_embed_grace = self.get_pair_entity_embed(entity_hidden_state_grace, hts)
        
        feature_map = self.get_channel_map(sequence_output, local_context)
        attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()

        attn_map = self.segmentation_net(attn_input)
        h_t = self.get_ht (attn_map, hts)
        s_embed = torch.tanh(self.head_extractor(torch.cat([s_embed_grace, s_embed, h_t], dim=1)))
        t_embed = torch.tanh(self.tail_extractor(torch.cat([t_embed_grace, t_embed, h_t], dim=1)))

        b1 = s_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = t_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.binary_linear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            
            graph_aug_1 = augmentation(graph=graph).to(self.device)
            graph_aug_2 = augmentation(graph=graph).to(self.device)
            z1 = self.grace_model(output_node_hidden_state, graph_aug_1)
            z2 = self.grace_model(output_node_hidden_state, graph_aug_2)
 
            grace_loss = self.grace_model.loss(z1, z2, batch_size = 0)
            
            altop_loss = self.loss_fnt(logits.float(), labels.float())
            total_loss = altop_loss + grace_loss
            
            output = (total_loss, altop_loss, grace_loss, ) + output
        return output