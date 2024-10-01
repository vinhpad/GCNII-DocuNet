import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.losses import ATLoss
from opt_einsum import contract
from models.gnn import GNN
from models.attn_unet import AttentionUNet


class DocREModel(nn.Module):
    def __init__(self, bert_config, args, bert_model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.bert_config = bert_config
        self.bert_model = bert_model
        self.hidden_size = bert_config.hidden_size
        self.loss_fnt = ATLoss()
        self.device = args.device
        self.tau = args.tau
        self.head_extractor = nn.Linear(self.gnn_output_dim + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(self.gnn_output_dim + args.unet_out_dim, emb_size)
        self.binary_linear = nn.Linear(emb_size * block_size, bert_config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.bert_drop = nn.Dropout(0.6)
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.liner = nn.Linear(bert_config.hidden_size, args.unet_in_dim)

        self.min_height = args.max_height
        self.channel_type = args.channel_type

        self.segmentation_net = AttentionUNet(
            input_channels=args.unet_in_dim,
            class_number=args.unet_out_dim,
            down_channel=args.down_dim
        )

        self.offset = 1
        
        self.gnn_output_dim = 256
        self.gnn = GNN(
            args.gnn_node_embedding,
            args.gnn_num_layer,
            bert_config.hidden_size + args.gnn_node_embedding,
            self.gnn_output_dim,
            args.device
        )


    def get_pair_entity_embed(self, entity_hidden_state, hts):
        s_embed, t_embed = [], []
        for batch_id, ht in enumerate(hts):
            for pair in ht:
                s_embed.append(entity_hidden_state[batch_id, pair[0]])
                t_embed.append(entity_hidden_state[batch_id, pair[1]])
        s_embed = torch.stack(s_embed, dim=0)
        t_embed = torch.stack(t_embed, dim=0)
        return s_embed, t_embed
    


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

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

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



    def forward(self, features, attention_mask, entity_pos, num_mention, 
                num_entity, labels=None, hts=None):
        
        sequence_output, attention = self.encode(features, attention_mask)
        entity_hidden_state = features[:, num_mention:num_mention + num_entity]
        
        entity_hidden_state = self.drop_out(entity_hidden_state)
        local_context = self.get_hrt(sequence_output, attention, entity_pos, hts)
        s_embed, t_embed = self.get_pair_entity_embed(entity_hidden_state, hts)

        if self.channel_type == 'context-based':
            feature_map = self.get_channel_map(sequence_output, local_context)
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception("channel_type must be specify correctly")

        
        attn_map = self.segmentation_net(attn_input)
        h_t = self.get_ht (attn_map, hts)
        s_embed = torch.tanh(self.head_extractor(torch.cat([s_embed, h_t], dim=1)))
        t_embed = torch.tanh(self.tail_extractor(torch.cat([t_embed, h_t], dim=1)))

        b1 = s_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = t_embed.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        logits = self.binary_linear(bl)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float()) 
            output = (loss.to(sequence_output),) + output

        return output
