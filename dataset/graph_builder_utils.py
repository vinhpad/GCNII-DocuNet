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
        for entity_idx, ent_pos in enumerate(entity_pos):
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
        mention_idx = 0
        for entity_idx, ent_pos in enumerate(entity_pos):
            for mention in ent_pos:
                for sent_idx, sent in enumerate(sent_pos):
                    if sent[0] <= mention[0] <= sent[1]:
                        u.append(get_id(num_entity, batch_id, entity_idx))
                        v.append(get_id(num_sent, batch_id, sent_idx))
                        break
                mention_idx += 1
    return u, v

def get_virtual_node_to_mention(virtual_pos, entity_pos, num_virtual, num_mention):
    u = []
    v = []

    for batch_id, vir_pos in enumerate(virtual_pos):

        entity_pos_s = entity_pos[batch_id]
        for entity_idx, ent_pos in enumerate(entity_pos_s):
            for mention_id, mention in enumerate(ent_pos):
                u.append(get_id(num_virtual, batch_id, mention_id))
                v.append(get_id(num_mention, batch_id, mention_id))

                if mention_id + 1 < len(ent_pos):
                    u.append(get_id(num_virtual, batch_id, mention_id+1))
                    v.append(get_id(num_mention, batch_id, mention_id))

    return u, v

def get_id(num_col: int, row_idx: int, col_idx: int) -> int:
    return num_col * row_idx + col_idx