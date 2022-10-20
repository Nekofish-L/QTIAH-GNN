import torch
import torch.nn as nn


class feature_emb_layer(nn.Module):
    def __init__(self, node_in_feats_dims, category_feat_indices=None, category_feat_nums=None, node_types=[],
                 emb_dim=20):
        super(feature_emb_layer, self).__init__()
        self.node_types = node_types
        self.category_feat_nums = category_feat_nums
        self.node_in_feats_dims = node_in_feats_dims
        self.emb_dim = emb_dim
        self.feat_proj = nn.ModuleDict()
        if self.category_feat_nums:

            self.category_feat_indices = category_feat_indices

            self.num_feat_indices = {}
            self.nn_emb_dict = nn.ModuleDict()
            for node_type in self.node_types:
                self.num_feat_indices[node_type] = []
                for i in range(node_in_feats_dims[node_type]):
                    if i not in self.category_feat_indices:
                        self.num_feat_indices[node_type].append(i)
                self.node_in_feats_dims[node_type] = len(
                    self.num_feat_indices[node_type])

                self.nn_emb_dict[node_type] = nn.ModuleList()
                for n in self.category_feat_nums[node_type]:
                    self.nn_emb_dict[node_type].append(
                        torch.nn.Embedding(int(n.item()), int(n.item() // 5)))
                    self.node_in_feats_dims[node_type] += int(n.item() // 5)

                self.feat_proj[node_type] = nn.Linear(
                    self.node_in_feats_dims[node_type], self.emb_dim)

    def forward(self, inputs):
        inputs_nn_emb = {}
        for node_type in self.node_types:
            inputs_temp = inputs[node_type][:,
                          self.num_feat_indices[node_type]]
            inputs_nn_emb[node_type] = []
            for i, nn_emb in enumerate(self.nn_emb_dict[node_type]):
                category = inputs[node_type][:,
                           self.category_feat_indices[node_type][i]].int()
                # if category.min()<0:
                #     category[category<0] = self.category_feat_nums[node_type][i]-1
                category_emb = nn_emb(category)
                inputs_nn_emb[node_type].append(category_emb)

            inputs[node_type] = self.feat_proj[node_type](
                torch.cat([inputs_temp, torch.cat(inputs_nn_emb[node_type], dim=1)], dim=1))

        return inputs
