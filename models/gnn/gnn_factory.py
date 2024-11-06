from .gate_gcn import GateGCN
from .gcnii import GCNII
from .gcn import GCN
from .res_gcn import ResGCN
from config.model_config import GNNConfig
from .gatconv_net import GATNet
from .air_gcnii import AIRGCNII


class GNNFactory:
    @staticmethod
    def from_config(config: GNNConfig, in_feat_dim):
        if config.gnn_type == "gate_gcn":
            return GateGCN(in_feat_dim, config.args['num_layers'], drop_out=config.args['drop_out'],
                           share_gate_layer=config.args['share_gate_layer'], drop_edge=config.args['drop_edge'])
        elif config.gnn_type == "gcnii":
            return GCNII(in_feat_dim, config.args['num_layers'], drop_out=config.args['drop_out'],
                         alpha=config.args['alpha'], lambda_=config.args['lambda'], drop_edge=config.args['drop_edge'])
        elif config.gnn_type == "air_gcnii":
            return AIRGCNII(in_feat_dim, config.args['num_layers'], drop_out=config.args['drop_out'],
                            lambda_=config.args['lambda'], drop_edge=config.args['drop_edge'])
        elif config.gnn_type == "gcn":
            return GCN(in_feat_dim, config.args['num_layers'],
                       config.args['drop_out'], config.args['drop_edge'])
        elif config.gnn_type == "res_gcn":
            return ResGCN(in_feat_dim, config.args['num_layers'],
                          config.args['drop_out'], config.args['drop_edge'])
        elif config.gnn_type == "gat_conv":
            return GATNet(in_feat_dim, config.args['num_heads'], config.args['num_layers'],
                          drop_out=config.args['drop_out'])
        raise NotImplementedError()
