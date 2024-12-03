import torch
import random
import numpy as np
from .graph_builder import create_graph

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    batch_entity_pos = [f["entity_pos"] for f in batch]
    batch_sent_pos = [f['sent_pos'] for f in batch]

    graphs, num_mentions, num_entities, num_sents = create_graph(batch_entity_pos, batch_sent_pos)

    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    
    output = (
        input_ids, 
        input_mask,
        batch_entity_pos, 
        batch_sent_pos,
        graphs, 
        num_mentions, 
        num_entities, 
        num_sents,
        labels,
        hts
    )

    return output