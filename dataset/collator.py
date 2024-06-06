import torch
from .graph_builder import GraphBuilder

graph_builder = GraphBuilder()


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    batch_size = len(batch)
    labels = [f["labels"] for f in batch]
    batch_entity_pos = [f["entity_pos"] for f in batch]
    batch_sent_pos = [f['sent_pos'] for f in batch]
    batch_virtual_pos, batch_token_pos = create_virtual_node(batch_entity_pos)
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    graph, num_mention, num_entity, num_sent, num_virtual = graph_builder.create_graph(batch_entity_pos, batch_sent_pos,
                                                                                       batch_virtual_pos)

    labels_node = None

    output = (input_ids, input_mask,
              batch_entity_pos,
              batch_sent_pos,
              batch_virtual_pos,
              graph,
              num_mention,
              num_entity,
              num_sent,
              num_virtual,
              labels,
              labels_node,
              hts)
    return output


def create_virtual_node(batch_entity_pos):
    batch_virtual_pos = []
    batch_token_pos = []

    for batch_id, entities_pos in enumerate(batch_entity_pos):
        virtual_node = []
        mentions = []
        for entity_pos in entities_pos:
            for mention in entity_pos:
                mentions.append(mention)

        mentions.sort(key=lambda item: item[0])

        if 0 < mentions[0][0]:
            virtual_node.append([0, mentions[0][0]])

        for idx in range(1, len(mentions)):
            if mentions[idx - 1][1] < mentions[idx][0]:
                virtual_node.append([mentions[idx - 1][1], mentions[idx][0]])

        tokens = []
        for vir_node in virtual_node:
            range_tokens = []
            for token_pos in range(vir_node[0], vir_node[1]):
                range_tokens.append(token_pos)
            tokens.append(range_tokens)

        batch_virtual_pos.append(virtual_node)

        batch_token_pos.append(tokens)

    return batch_virtual_pos, batch_token_pos
