import torch
import torch.nn as nn
from .topology_agnostic_embedding_layer import feature_emb_layer as feature_emb_layer
from .neighbor_selection_layer import semantic_layer as semantic_layer

class semantic_gnn(nn.Module):
    def __init__(self,
                 in_dim,
                 num_classes,
                 g,
                 device,
                 hid_dim,
                 num_layers,
                 activation,

                 num_head,
                 step_size=0.02,
                 node_in_feats_dims=None,
                 category_feat_indices=None, category_feat_nums=None, node_types=[]
                 ):
        super(semantic_gnn, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activation = activation

        self.step_size = step_size
        self.g = g
        self.device = device
        self.num_head = num_head

        self.emb_layer = feature_emb_layer(
            node_in_feats_dims, category_feat_indices, category_feat_nums, node_types, emb_dim=self.in_dim)

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            self.layers.append(semantic_layer(self.in_dim, self.out_dim, self.num_classes, self.num_head,self.g,self.device, activation=self.activation,
                                              step_size=self.step_size))

        else:
            self.layers.append(
                semantic_layer(self.in_dim, self.hid_dim, self.num_classes, self.num_head, self.g, self.device,
                               activation=self.activation,
                               step_size=self.step_size))

            for i in range(self.num_layers-2):
                self.layers.append(semantic_layer(self.hid_dim, self.hid_dim, self.num_classes, self.num_head, self.g, self.device,
                               activation=self.activation,
                               step_size=self.step_size))

            self.layers.append(
                semantic_layer(self.hid_dim, self.hid_dim, self.num_classes, self.num_head, self.g, self.device,
                               activation=self.activation,
                               step_size=self.step_size))

        self.MLP = nn.Sequential(nn.Linear(self.hid_dim+self.num_layers*self.in_dim, 20),
                                 nn.Linear(20, 10),
                                 nn.Sigmoid(),
                                 nn.Linear(10, 2),
                                 nn.Sigmoid())

    def forward(self, graph, feat):
        feat = feat.copy()

        feat = self.emb_layer(feat)
        org_company_feat = feat['company']
        graph.nodes['company'].data['feat'] = feat['company']
        graph.nodes['brand'].data['feat'] = feat['brand']
        graph.nodes['organize'].data['feat'] = feat['organize']

        res_feat = []
        pseudo_loss_per_layer = []
        for layer in self.layers:
            graph, feat = layer(graph, feat)
            self.in_dim += feat['company'].shape[1]
            res_feat.append(feat['company'])
            if 'pseudo_loss' in feat.keys() and not torch.isnan(feat['pseudo_loss']):
                pseudo_loss_per_layer.append(feat['pseudo_loss'])

        company_graph_feat = torch.cat([*res_feat, org_company_feat], dim=1)

        feat['company'] = self.MLP(company_graph_feat)
        feat['pseudo_loss'] = pseudo_loss_per_layer
        return feat
