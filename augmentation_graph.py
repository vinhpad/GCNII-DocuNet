import dgl
import torch
import numpy

def drop_feature(features, drop_prob):
    drop_mask = torch.empty(
        (features.size(1),),
        dtype=torch.float32,
        device=features.device
    ).uniform_(0, 1) < drop_prob
    
    x = features.clone()
    x[:, drop_mask] = 0
    return x

def drop_edge(graph, edge_drop_prob):
    num_edges = graph.num_edges()

    mask_rates = torch.FloatTensor(
        numpy.ones(num_edges) * edge_drop_prob
    )
    
    masks = torch.bernoulli(1 - mask_rates)
    
    mask_idx = masks.nonzero().squeeze(1)
    
    return mask_idx


def augmentation(graph, features, feature_drop_prob, edge_drop_prob):
    edges_dropped = drop_edge(graph, edge_drop_prob)
    features_dropped = drop_feature(features, feature_drop_prob)
    num_nodes = graph.num_nodes()

    src = graph.edges()[0]
    dst = graph.edges()[1]
    
    new_src = src[edges_dropped]
    new_dst = dst[edges_dropped]
    
    new_graph = dgl.graph((new_src, new_dst), num_nodes=num_nodes)
    new_graph = new_graph.add_self_loop()
    
    return new_graph, features_dropped