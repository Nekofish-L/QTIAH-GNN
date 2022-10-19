import torch
import numpy as np
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split

class CompanyDataset(DGLDataset):
    def __init__(self, heter_g):
        self.graph = heter_g
        self.num_classes = 2
        super(CompanyDataset, self).__init__(name='company_hetergraph')

    def process(self):
        n_nodes = self.graph.nodes['company'].data['company_feat'].shape[0]

        l = np.array(range(n_nodes))
        train, val_test = train_test_split(l, train_size=0.6)
        val, test = train_test_split(val_test, train_size=0.5)


        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True

        self.graph.nodes['company'].data['train_mask'] = train_mask
        self.graph.nodes['company'].data['val_mask'] = val_mask
        self.graph.nodes['company'].data['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
