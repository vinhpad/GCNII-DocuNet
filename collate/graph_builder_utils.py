from typing import Tuple, List

def get_sentence_to_sentence_edges(num_sent, batch_sent_pos) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, sent_pos in enumerate(batch_sent_pos):
        for sent_1_idx in range(len(sent_pos)):
            for sent_2_idx in range(len(sent_pos)):
                if sent_1_idx == sent_2_idx:
                    continue
                u.append(get_id(num_sent, batch_id, sent_1_idx))
                v.append(get_id(num_sent, batch_id, sent_2_idx))
    return u, v


def get_mention_to_mention_edges(num_mention, batch_entity_pos) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, entity_pos in enumerate(batch_entity_pos):
        mention_idx = 0
        for ent_pos in entity_pos:
            for internal_idx_1, _ in enumerate(ent_pos):
                for internal_idx_2, _ in enumerate(ent_pos):
                    if internal_idx_1 == internal_idx_2:
                        continue
                    u.append(get_id(num_mention, batch_id, mention_idx + internal_idx_1))
                    v.append(get_id(num_mention, batch_id, mention_idx + internal_idx_2))
            mention_idx += len(ent_pos)
    return u, v


def get_mention_to_entity_edges(num_mention, num_entity, batch_entity_pos) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, entity_pos in enumerate(batch_entity_pos):
        mention_idx = 0
        for entity_idx, ent_pos in enumerate(entity_pos):
            for _ in ent_pos:
                u.append(get_id(num_mention, batch_id, mention_idx))
                v.append(get_id(num_entity, batch_id, entity_idx))
                mention_idx += 1
    return u, v


def get_mention_to_sentence_edges(num_mention, num_sent,
                                  batch_sent_pos, batch_entity_pos) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, sent_pos in enumerate(batch_sent_pos):
        entity_pos = batch_entity_pos[batch_id]
        mention_idx = 0
        for _, ent_pos in enumerate(entity_pos):
            for mention in ent_pos:
                for sent_idx, sent in enumerate(sent_pos):
                    if sent[0] <= mention[0] <= sent[1]:
                        u.append(get_id(num_mention, batch_id, mention_idx))
                        v.append(get_id(num_sent, batch_id, sent_idx))
                        break
                mention_idx += 1
    return u, v


def get_entity_to_sentence_edges(num_entity, num_sent,
                                 batch_sent_pos, batch_entity_pos) -> Tuple[List[int], List[int]]:
    u = []
    v = []
    for batch_id, sent_pos in enumerate(batch_sent_pos):
        entity_pos = batch_entity_pos[batch_id]
        for entity_idx, ent_pos in enumerate(entity_pos):
            for mention in ent_pos:
                for sent_idx, sent in enumerate(sent_pos):
                    if sent[0] <= mention[0] <= sent[1]:
                        u.append(get_id(num_entity, batch_id, entity_idx))
                        v.append(get_id(num_sent, batch_id, sent_idx))
                        break
                
    return u, v

# def get_token_to_sent_edges(num_token, num_sent, batch_token_pos, batch_sent_pos):
#     u = []
#     v = []
#     for batch_id, sent_pos in enumerate(batch_sent_pos):
#         token_pos = batch_token_pos[batch_id]
#         for token_idx, tok_pos in enumerate(token_pos):
#             for sent_idx, sent in enumerate(sent_pos):
#                 if sent[0] <= tok_pos <= sent[1]:
#                     u.append(get_id(num_token, batch_id, token_idx))
#                     v.append(get_id(num_sent, batch_id, sent_idx))
#                     break     
#     return u, v

# def get_token_to_mention_edges(
#         num_mention, num_token, 
#         batch_entity_pos, batch_token_pos
#     ) -> Tuple[List[int], List[int]]:
#     u = []
#     v = []
#     for batch_id, entities_pos in enumerate(batch_entity_pos):
#         mention_idx = 0
#         token_pos = batch_token_pos[batch_id]
#         mention_pos = []

        
#         for entity_pos in entities_pos:
#             for mention in entity_pos:
#                 mention_pos.append([mention,mention_idx])
#                 mention_idx = mention_idx + 1
#         sorted_mention = sorted(mention_pos, key=lambda x: x[0][0])

#         for mention_idx, mention_pos in enumerate(sorted_mention):
#             for token_idx, tok_pos in enumerate(token_pos):
#                 if mention_idx != len(sorted_mention)-1:
#                     if sorted_mention[mention_idx][0][1] <= tok_pos < sorted_mention[mention_idx+1][0][0]:

#                         mention_real_idx = sorted_mention[mention_idx][1]
#                         u.append(get_id(num_token, batch_id, token_idx))
#                         v.append(get_id(num_mention, batch_id, mention_real_idx))

#                         mention_real_idx_ = sorted_mention[mention_idx+1][1]
#                         u.append(get_id(num_token, batch_id, token_idx))
#                         v.append(get_id(num_mention, batch_id, mention_real_idx_))
#     return u, v

def get_id(num_col: int, row_idx: int, col_idx: int) -> int:
    return num_col * row_idx + col_idx