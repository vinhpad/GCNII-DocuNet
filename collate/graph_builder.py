import torch
import numpy as np

def create_graph(batch_entity_pos, batch_sent_pos):
    batch_size = len(batch_entity_pos)

    num_mentions = [sum([len(ent_pos) for ent_pos in entity_pos]) for entity_pos in batch_entity_pos]
    
    num_entities = [len(entity_pos) for entity_pos in batch_entity_pos]
    
    num_sents =  [len(sent_pos) for sent_pos in batch_sent_pos]

    graphs = []
    for batch_id in range(batch_size):
        num_mention = num_mentions[batch_id]
        num_sent = num_sents[batch_id]

        num_node = num_mention + num_sent

        entity_pos = batch_entity_pos[batch_id]
        sent_pos = batch_sent_pos[batch_id]

        nodes_adj = torch.zeros((num_node, num_node))

        mention_idx = 0
        mention_real_idx = []
        
        for mention_pos in entity_pos:
            num_mention_of_entity = len(mention_pos)
            mention_ids = []
            for _ in range(num_mention_of_entity):
                mention_ids.append(mention_idx)
                mention_idx += 1
            mention_real_idx.append(mention_ids)
                
        # add self-loop
        edges_cnt = 1
        for vertex_id in range(num_node):
            nodes_adj[vertex_id][vertex_id] = edges_cnt

        # get_sentence_to_sentence_edges
        edges_cnt = 2
        for sent_idx_1 in range(num_sent):
            for sent_idx_2 in range(num_sent):
                if sent_idx_1 != sent_idx_2:
                    nodes_adj[num_mention+sent_idx_1][num_mention+sent_idx_2] = edges_cnt

    
        # mention to sent
        edges_cnt = 3
        for sent_id, sent_pos_ in enumerate(sent_pos):
            for entity_id, mention_pos in enumerate(entity_pos):
                for mention_id, mention_pos_ in enumerate(mention_pos):
                    # inside sent
                    if sent_pos_[0] <= mention_pos_[0] <= sent_pos_[1]:
                        nodes_adj[num_mention + sent_id][mention_real_idx[entity_id][mention_id]] = edges_cnt
                        nodes_adj[mention_real_idx[entity_id][mention_id]][num_mention + sent_id] = edges_cnt


        # mention co-reference edges
        edges_cnt = 4
        for entity_id, mention_pos in enumerate(entity_pos):
            num_mention_of_entity = len(mention_pos)
            for i in range(num_mention_of_entity):
                for j in range(num_mention_of_entity):
                    if i != j:
                        nodes_adj[mention_real_idx[entity_id][i]][mention_real_idx[entity_id][j]] = edges_cnt    
                        
        # inter-entity edges
        edges_cnt = 5
        for entity_id, mention_pos in enumerate(entity_pos):
            num_mention_of_entity = len(mention_pos)
            for i in range(num_mention_of_entity):
                for j in range(num_mention):
                    if mention_real_idx[entity_id][i] != j:
                        nodes_adj[mention_real_idx[entity_id][i]][j] = edges_cnt
                    
        graphs.append(nodes_adj)
      
    return graphs, num_mentions, num_entities, num_sents