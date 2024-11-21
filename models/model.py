import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.losses import balanced_loss as ATLoss
from opt_einsum import contract
from .gnn import GNN
from models.attn_unet import AttentionUNet
from utils import process_long_input
from .grace import GRACE
import torch.nn.functional as F


class DocREModel(nn.Module):
    def __init__(self, args, bert_model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        
        # BERT model config
        self.bert_model = bert_model
        self.bert_config = args.bert_config
        self.bert_drop = nn.Dropout(0.6)
        self.bert_hidden_size = args.bert_config.hidden_size

        
        # graph neural network model config
        self.gnn_hidden_size = self.bert_hidden_size + args.gnn_node_type_embedding
        self.gnn = GNN(
            num_node_type=args.gnn_num_node_type, 
            node_type_embedding=args.gnn_node_type_embedding,
            hidden_feat_dim=self.gnn_hidden_size, 
            device=args.device
        )
        
        # GRACE model config
        # self.edge_drop_prob_first = args.edge_prob_first
        # self.edge_drop_prob_second = args.edge_prob_second
        # self.grace_model = GRACE(
        #     encoder=None,
        #     num_hidden = self.gnn_hidden_size, 
        #     num_proj_hidden=args.grace_projection_hidden_feat_dim, 
        #     tau=args.tau
        # ).to(args.device)
        
        # Unet config
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.liner = nn.Linear(self.bert_hidden_size, args.unet_in_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type
        self.segmentation_net = AttentionUNet(input_channels=args.unet_in_dim,
                                              class_number=args.unet_out_dim,
                                              down_channel=args.down_dim)
        
        # Model config
        self.head_extractor = nn.Linear(self.gnn_hidden_size * 2 + args.gnn_num_node_type + self.bert_hidden_size , emb_size)
        self.tail_extractor = nn.Linear(self.gnn_hidden_size * 2 + args.gnn_num_node_type + self.bert_hidden_size , emb_size)
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
            
        sequence_output, attention = process_long_input(self.bert_model, input_ids, 
            attention_mask, start_tokens, end_tokens)
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


    def get_token_embed(self, sequence_output, batch_token_pos, num_token):
        batch_size, _, embed_dim = sequence_output.shape
        token_embed = torch.zeros((batch_size, num_token, embed_dim)).to(self.device)
        for batch_id, token_pos in enumerate(batch_token_pos):
            for token_idx, tok_pos in enumerate(token_pos):
                token_embed[batch_id][token_idx] = sequence_output[batch_id][tok_pos + self.offset]
                
        return token_embed

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

    def forward(
        self, 
        input_ids, 
        attention_mask,
        entity_pos, 
        sent_pos, 
        token_pos,
        graph, 
        num_mention, 
        num_entity, 
        num_sent, 
        num_token,
        on_hot_encoding,
        labels=None, 
        hts=None
    ):
        
        sequence_output, attention = self.encode(input_ids, attention_mask)
        bs, _, d = sequence_output.shape
        mention_embed = self.get_mention_embed(sequence_output, entity_pos, num_mention)
        entity_embed = self.get_entity_embed(sequence_output, entity_pos, num_entity)
        sent_embed = self.get_sent_embed(sequence_output, sent_pos, num_sent)
        token_embed = self.get_token_embed(sequence_output, token_pos, num_token)

        print(mention_embed.shape)
        print(entity_embed.shape)
        print(sent_embed.shape)
        print(token_embed.shape)
        entity_hidden_state, _ = self.gnn([on_hot_encoding, mention_embed, entity_embed, sent_embed, token_embed, graph])
        
        local_context = self.get_hrt(sequence_output, attention, entity_pos, hts)
        s_embed, t_embed = self.get_pair_entity_embed(entity_hidden_state, hts)

        if self.channel_type == 'context-based':
            feature_map = self.get_channel_map(sequence_output, attn_input)
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
            attn_map = self.segmentation_net(attn_input)
        elif self.channel_type == 'similarity-based':
            
            ent_encode = sequence_output.new_zeros(bs, self.min_height, d)

            for _b in range(bs):
                entity_emb = entity_hidden_state[_b]
                entity_num = entity_emb.size(0)
                ent_encode[_b, :entity_num, :] = entity_emb
            
            # similar1 = []
            # for ent_encode
            # similar1 = [ent_encode[i].dot(ent_encode[j]) for i in range()]
           
            # similar2 = CosineMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            
            # similar3 = BilinearMatrixAttention(self.emb_size, self.emb_size).to(ent_encode.device)(ent_encode, ent_encode).unsqueeze(-1)
            
            # attn_input = torch.cat([similar1,similar2,similar3],dim=-1).permute(0, 3, 1, 2).contiguous()
        else:
            # attn_input = self.get_channel_map(sequence_output, local_context)
            attn_map = self.get_channel_map(sequence_output, local_context)
            # raise Exception("channel_type must be specify correctly")

        
        h_t = self.get_ht (attn_map, hts)
        print(h_t.shape)
        print(s_embed.shape)
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
            output = (loss, ) + output
        return output