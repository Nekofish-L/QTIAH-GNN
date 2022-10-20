import torch
import torch.nn as nn
import dgl.function as fn
import numpy as np
from tqdm import tqdm
from itertools import chain
import random

etype_map = dict(
    company=[
        ('company', 'business', 'company'),
        ('company', 'business_event', 'brand'),
        ('company', 'competition', 'company'),
        ('company', 'invest', 'company'),
        ('company', 'member', 'company'),
        ('company', 'shareholder', 'company'),
    ],
    organize=[
        ('organize', 'invest', 'brand'),
        ('organize', 'invest', 'company')
    ],
    brand=[
        ('brand', 'business_event', 'company'),
        ('brand', 'competition', 'brand'),
        ('brand', 'member', 'brand'),
    ]
)

RW = True

class Hadamard_product_layer(nn.Module):
    def __init__(self, num_nodes, in_dim, device,num_head=2):
        super(Hadamard_product_layer, self).__init__()
        self.num_head = num_head
        self.weight_head = []
        self.num_nodes = num_nodes
        self.in_dim = in_dim

        for i in range(num_head):
            self.weight_head.append(nn.Parameter(
                torch.randn(self.num_nodes, self.in_dim)).to(device))

    def forward(self, input):
        res = []
        for i in range(self.num_head):

            res.append(torch.mul(self.weight_head[i], input))

        return torch.mean(torch.stack(res), dim=0)


class semantic_layer(nn.Module):
    # one layer of semantic_graph
    def __init__(self, in_dim, out_dim, num_class,num_head,g,device, activation=None, step_size=0.02):
        super(semantic_layer, self).__init__()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_class

        self.device = device

        self.linear_company = nn.Linear(self.in_dim*2, self.out_dim)
        self.linear_brand = nn.Linear(self.in_dim, self.out_dim)
        self.linear_organize = nn.Linear(self.in_dim, self.out_dim)

        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.Hadamard_layer = Hadamard_product_layer(
            g.num_nodes(), in_dim,device=self.device, num_head=num_head)
        self.Hadamard_layer_list = nn.ModuleList()
        self.tmask = (g.nodes['company'].data['y']
                      == 1).nonzero(as_tuple=True)[0]
        self.fmask = (g.nodes['company'].data['y']
                      == 0).nonzero(as_tuple=True)[0]

        self.pseudo_loss_func = nn.CrossEntropyLoss()

    def weight_cos(self, input1, input2, cos):
        return cos(self.Hadamard_layer(input1), self.Hadamard_layer(input2))

    def choose_neighbor(self, g, etype):
        access = g.edges[etype].data['d']
        temp_neigh = g.in_edges(g.nodes(etype[2]), form='eid', etype=etype)
        return temp_neigh[torch.argwhere(access[temp_neigh])].squeeze()

    def cal_sam_neigh(self, edges):
        d = edges.src['pred_label'] == edges.dst['pred_label']
        # print("edges", d.shape)
        return {'d': d}

    def forward(self, g, sem_feat):
        with g.local_scope():

            g.nodes['company'].data['feat'] = sem_feat['company']
            g.nodes['brand'].data['feat'] = sem_feat['brand']

            tfeat_mean = sem_feat['company'][self.tmask].mean(dim=0)
            ffeat_mean = sem_feat['company'][self.fmask].mean(dim=0)

            t_sem_feat = self.weight_cos(torch.cat(list(
                [sem_feat['company'], sem_feat['brand'], sem_feat['organize']])), tfeat_mean, self.cos_sim)  # 与正样本的相似度
            f_sem_feat = self.weight_cos(torch.cat(list(
                [sem_feat['company'], sem_feat['brand'], sem_feat['organize']])), ffeat_mean, self.cos_sim)  # 与负样本的相似度

            # sem_feat.keys() :  company,brand,organize
            # sem_feat values: company: 63180,brand:34588,organize:4148
            company_indices = [0, 63180]
            brand_indices = [63180, 97768]
            organize_indices = [97768, 101916]
            semantic = torch.stack((t_sem_feat, f_sem_feat), 0)

            pseudo_loss = self.pseudo_loss_func(semantic[:, company_indices[0]:company_indices[1]].transpose(
                0, 1), g.nodes['company'].data['y'])  # 伪标签loss

            g.nodes['company'].data['pred_label'] = torch.argmax(
                semantic[:, company_indices[0]:company_indices[1]], dim=0)
            g.nodes['brand'].data['pred_label'] = torch.argmax(
                semantic[:, brand_indices[0]:brand_indices[1]], dim=0)
            g.nodes['organize'].data['pred_label'] = torch.argmax(
                semantic[:, organize_indices[0]:organize_indices[1]], dim=0)

            hr_company = {}
            hr_brand = {}
            hr_organize = {}

            # random walk

            # same pred label node
            comp_plabel = g.nodes['company'].data['pred_label']
            brand_plabel = g.nodes['brand'].data['pred_label']
            org_plabel = g.nodes['organize'].data['pred_label']

            comp_eq_comp = torch.broadcast_to(comp_plabel.unsqueeze(
                dim=1), (-1, 63180)) == comp_plabel.unsqueeze(dim=0)
            comp_eq_brand = torch.broadcast_to(comp_plabel.unsqueeze(
                dim=1), (-1, 34588)) == brand_plabel.unsqueeze(dim=0)
            comp_eq_org = torch.broadcast_to(comp_plabel.unsqueeze(
                dim=1), (-1, 4148)) == org_plabel.unsqueeze(dim=0)

            brand_eq_brand = torch.broadcast_to(brand_plabel.unsqueeze(
                dim=1), (-1, 34588)) == brand_plabel.unsqueeze(dim=0)
            brand_eq_org = torch.broadcast_to(brand_plabel.unsqueeze(
                dim=1), (-1, 4148)) == org_plabel.unsqueeze(dim=0)
            org_eq_org = torch.broadcast_to(org_plabel.unsqueeze(
                dim=1), (-1, 4148)) == org_plabel.unsqueeze(dim=0)

            global neighbor_cache
            # reset cache
            neighbor_cache = dict()

            def same_neighbors(g, node, etype) -> list:
                if etype[0] == 'company':
                    if etype[-1] == 'company':
                        equl_vec = comp_eq_comp[node]
                    elif etype[-1] == 'brand':
                        equl_vec = comp_eq_brand[node]
                    elif etype[-1] == 'organize':
                        equl_vec = comp_eq_org[node]

                elif etype[0] == 'brand':
                    if etype[-1] == 'company':
                        equl_vec = comp_eq_brand[:, node]
                    elif etype[-1] == 'brand':
                        equl_vec = brand_eq_brand[node]
                    elif etype[-1] == 'organize':
                        equl_vec = brand_eq_org[node]

                elif etype[0] == 'organize':
                    if etype[-1] == 'company':
                        equl_vec = comp_eq_org[:, node]
                    elif etype[-1] == 'brand':
                        equl_vec = brand_eq_org[:, node]
                    elif etype[-1] == 'organize':
                        equl_vec = org_eq_org[node]

                same_nodes = set(torch.argwhere(
                    equl_vec).flatten().cpu().numpy())
                neighbor = set(g[etype].out_edges(node)[-1].cpu().numpy())
                intersection = same_nodes.intersection(neighbor)
                return list(zip([etype for _ in intersection], intersection))

            def filtered_neighbor(g, node, src_ntype):
                global neighbor_cache
                if (node, src_ntype) in neighbor_cache:
                    return neighbor_cache[(node, src_ntype)]
                etype_list = etype_map[src_ntype]
                neighbors = list(
                    chain(*map(lambda x: same_neighbors(g, node, x), etype_list)))
                neighbor_cache[(node, src_ntype)] = neighbors
                return neighbors

            def sample_next(g, node, src_ntype):
                neighbors = filtered_neighbor(g, node, src_ntype)

                if len(neighbors) == 0:
                    return None, None, None

                etype, tgt_node = random.choice(neighbors)
                return tgt_node, etype, etype[-1]

            def random_walk_per_node(g, node, max_length=8, max_same_comp_neighbor=15, max_sample_path=20,restart_prob = 0.1):
                counter, sample_path = 0, 0
                result = set()
                order_1_num = len(filtered_neighbor(g, node, "company"))
                max_same_comp_neighbor -= order_1_num
                if max_same_comp_neighbor <= 0:
                    return result

                while counter < max_same_comp_neighbor and sample_path < max_sample_path:
                    depth, company_in_a_path = 1, 0
                    current_node, current_ntype = node, "company"
                    while (depth < max_length) and (current_node is not None) and (counter + company_in_a_path < max_same_comp_neighbor):
                        # sample a random neighbor with same pseudo label
                        if restart_prob is not None and random.random() <=restart_prob:
                            depth, company_in_a_path = 1, 0
                            current_node, current_etype, current_ntype = sample_next(
                                g, node, 'company'
                            )
                        else:
                            current_node, current_etype, current_ntype = sample_next(
                                g, current_node, current_ntype)
                        if current_node is None:
                            break
                        _, relation, current_ntype = current_etype
                        if current_ntype == "company": # and ((node, current_node), current_etype) not in result:
                            company_in_a_path += 1
                            if relation not in ['business', 'competition', 'invest', 'member', 'shareholder']:
                                relation = 'business'
                        if current_ntype == 'brand':
                            relation = 'business_event'
                        result.add(((node, current_node), ("company", relation, current_ntype)))
                        depth += 1
                    counter += company_in_a_path
                    sample_path += 1
                return result

            # start random walk
            bankrupt = torch.argwhere(comp_plabel == 1).flatten().cpu().numpy()
            # sample_num = min(int(len(bankrupt)*0.05), 10)
            sample_num = len(bankrupt)
            sampled_bankrupt = np.random.choice(bankrupt, sample_num)
            extra_edges = []

            if RW:
                for node in tqdm(sampled_bankrupt):
                    new_edges = random_walk_per_node(g, node)
                    if len(new_edges):
                        for (start, end), etype in new_edges:
                            g.add_edges(start, end, etype=etype)
                            extra_edges.append(((start, end), etype))

            for i, etype in enumerate(g.canonical_etypes):
                if etype[0] == 'company' or etype[2] == 'company':
                    g.apply_edges(self.cal_sam_neigh, etype=etype)
                    # print("after apply", g[etype].edata['d'].shape)
                else:
                    g.edges[etype].data['d'] = torch.ones(
                        g.num_edges(etype)).to(self.device)

                sampled_edges = self.choose_neighbor(g, etype).to(self.device)

                g.send_and_recv(sampled_edges, fn.copy_u('feat', 'm'), fn.mean(
                    'm', 'h_%s_%s_%s' % (etype[0], etype[1], etype[2])), etype=etype)

                for key in g.ndata['h_%s_%s_%s' % (etype[0], etype[1], etype[2])].keys():
                    if key == 'company':
                        hr_company[etype] = g.ndata['h_%s_%s_%s' %
                                                    (etype[0], etype[1], etype[2])]['company']
                    elif key == 'brand':
                        hr_brand[etype] = g.ndata['h_%s_%s_%s' %
                                                  (etype[0], etype[1], etype[2])]['brand']
                        print(hr_brand[etype].shape, etype)
                    elif key == 'organize':
                        hr_organize[etype] = g.ndata['h_%s_%s_%s' %
                                                     (etype[0], etype[1], etype[2])]['organize']
                    else:
                        continue

            # hr_company_tensor = torch.sum(torch.stack(list(hr_company.values())),dim=0) # 没有加中心节点的特征
            # hr_company_tensor = torch.sum(torch.stack(list(hr_company.values())),dim=0) + sem_feat['company']  # 加了中心节点特征
            #####################

            hr_company_tensor_nei = torch.sum(
                torch.stack(list(hr_company.values())), dim=0)
            hr_company_tensor_diff = torch.sum(torch.stack(
                list(hr_company.values())) - sem_feat['company'], dim=0)
            hr_company_tensor = torch.cat(
                [hr_company_tensor_nei, hr_company_tensor_diff], dim=1)
            ######################

            if len(hr_organize) == 0:
                hr_organize_tensor = sem_feat['organize']
            else:
                hr_organize_tensor = torch.sum(torch.stack(
                    list(hr_organize.values())), dim=0) + sem_feat['organize']

            if len(hr_brand) == 0:
                hr_brand_tensor = sem_feat['brand']
            else:
                print(sem_feat['brand'].shape)
                print([t.shape for t in hr_brand.values()])
                hr_brand_tensor = torch.sum(torch.stack(
                    list(hr_brand.values())), dim=0) + sem_feat['brand']

            company_tensor = self.activation(
                self.linear_company(hr_company_tensor))
            brand_tensor = self.activation(self.linear_brand(hr_brand_tensor))
            organize_tensor = self.activation(
                self.linear_organize(hr_organize_tensor))

            # print("remove temp edge")
            if RW:
                for (start, end), etype in extra_edges:

                    g.remove_edges(g[etype].edge_ids(
                        [start, ], [end, ]), etype=etype)

        return g, {'company': company_tensor,
                   'brand': brand_tensor,
                   'organize': organize_tensor,
                   'pseudo_loss': pseudo_loss}