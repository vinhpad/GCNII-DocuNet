import torch
from .graph_builder import GraphBuilder

graph_builder = GraphBuilder()


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    batch_size = len(batch)
    batch_entity_pos = [f["entity_pos"] for f in batch]
    batch_sent_pos = [f['sent_pos'] for f in batch]
    batch_virtual_pos, batch_token_pos, batch_mention_pos = create_virtual_node(batch_entity_pos)
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    graph, num_mention, num_entity, num_sent, num_token = graph_builder.create_graph(batch_entity_pos, batch_sent_pos,
                                                                                     batch_virtual_pos, batch_token_pos)

    num_node_graph = num_mention + num_entity + num_sent + num_token
    labels_node = torch.zeros([batch_size, num_node_graph], dtype=torch.long)

    for batch_id, _ in enumerate(batch):
        # label for mention
        for mention_idx, _ in enumerate(batch_mention_pos[batch_id]):
            labels_node[batch_id, mention_idx] = 1

        #label for entity
        for entity_idx, _ in enumerate(batch_entity_pos[batch_id]):
            labels_node[batch_id, entity_idx + num_mention] = 2

        #label for sent
        for sent_idx, _ in enumerate(batch_sent_pos[batch_id]):
            labels_node[batch_id, sent_idx + num_mention + num_entity] = 3

        #label for token
        for token_idx, _ in enumerate(batch_token_pos[batch_id]):
            labels_node[batch_id, token_idx + num_mention + num_entity + num_sent] = 4

    labels_node = torch.reshape(labels_node, (batch_size * num_node_graph, 1))[0]

    output = (input_ids, input_mask,
              batch_entity_pos,
              batch_sent_pos,
              batch_virtual_pos,
              graph,
              num_mention,
              num_entity,
              num_sent,
              num_token,
              labels,
              labels_node,
              hts)
    return output


def create_virtual_node(batch_entity_pos):
    batch_virtual_node = []
    batch_token_node = []
    batch_mention_node = []
    for batch_id, entities_pos in enumerate(batch_entity_pos):
        virtual_node = []
        mentions = []
        for entity_pos in entities_pos:
            for mention in entity_pos:
                mentions.append(mention)

        mentions.sort(key=lambda mention: mention[0])
        if 0 < mentions[0][0]:
            virtual_node.append([0, mentions[0][0]])

        for idx in range(1, len(mentions)):
            if mentions[idx - 1][1] < mentions[idx][0]:
                virtual_node.append([mentions[idx - 1][1], mentions[idx][0]])

        tokens = []
        for vir_node in virtual_node:
            for token_pos in range(vir_node[0], vir_node[1]):
                tokens.append(token_pos)

        batch_virtual_node.append(virtual_node)
        batch_token_node.append(tokens)
        batch_mention_node.append(mentions)

    return batch_virtual_node, batch_token_node, batch_mention_node
