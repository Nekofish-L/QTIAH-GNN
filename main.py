import torch
import torch.nn as nn
from data.Company_Dataset import CompanyDataset
from model.qtiah_gnn import semantic_gnn
import pickle
from train import train
from model.loss import CBLoss


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def main():
    seed_everything(10)

    gpu=1
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    with open('data/Heterogeneous_graph.pickle','rb') as file:
        g = pickle.load(file)
        file.close()

    company_feat = g.nodes['company'].data['company_feat']
    g.nodes['company'].data['feat'] = company_feat
    brand_feat = g.nodes['brand'].data['brand_feat']
    g.nodes['brand'].data['feat'] = brand_feat
    org_feat = g.nodes['organize'].data['organize_feat']
    g.nodes['organize'].data['feat'] = org_feat


    # device = torch.device('cpu')
    dataset = CompanyDataset(g)
    h_g = dataset[0]
    # in_dim = h_g.ndata['feat']['company'].shape[1]
    num_classes = dataset.num_classes

    category_feat_indices = {}
    category_feat_nums = {}
    node_types = ['company','brand','organize']

    category_feat_indices['company'] = [2,3,4]
    category_feat_indices['brand'] = [6,7,8]
    category_feat_indices['organize'] = [10]

    category_feat_nums['company'] = [torch.max(h_g.nodes['company'].data['company_feat'][:,i])+2 for i in category_feat_indices['company']]
    category_feat_nums['brand'] = [torch.max(h_g.nodes['brand'].data['brand_feat'][:,i])+2 for i in category_feat_indices['brand']]
    category_feat_nums['organize'] = [torch.max(h_g.nodes['organize'].data['organize_feat'][:,i])+2 for i in category_feat_indices['organize']]
    node_in_feats_dims = {'company':11,'brand':11,'organize':11}
    in_dim = 30
    hid_dim = 30
    lr = 0.001
    eta_min = 1e-5
    beta = 0.99999
    num_epoch = 3000
    num_head = 5
    num_layers = 2
    step_size = 0.02


    node_types = ['company','brand','organize']


    features = {'company': g.ndata["company_feat"]['company'].to(device),
                'brand': g.ndata['brand_feat']['brand'].to(device),
                'organize': g.ndata['organize_feat']['organize'].to(device)}

    for node_type in node_types:
        for i ,_ in enumerate(category_feat_nums[node_type]):
            category = features[node_type][:, category_feat_indices[node_type][i]].int()
            if category.min() < 0:
                category[category < 0] = category_feat_nums[node_type][i] - 1
            features[node_type][:, category_feat_indices[node_type][i]] = category


    model = semantic_gnn(in_dim = in_dim,num_classes = num_classes,
                         g=h_g,
                         device=device,
                         num_head=num_head,
                         step_size=step_size,
                         hid_dim=hid_dim,num_layers=num_layers,activation=nn.Sigmoid(),
                         node_in_feats_dims=node_in_feats_dims,
                         category_feat_indices = category_feat_indices,
                         category_feat_nums=category_feat_nums,
                         node_types = node_types)

    train(h_g, model, features, num_epoch, lr, CBLoss, device, beta, eta_min)

if __name__ == '__main__':
    main()