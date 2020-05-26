import os as os
import numpy as np

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def prepare_data(args):
    dataset = args.dataset.title()

    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, '..', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    data = modify_train_mask(data)
    data = add_noise_features(data, args.num_noise)

    return data


def modify_train_mask(data):
    val_mask = data.val_mask
    test_mask = data.test_mask
    new_train_mask = ~(val_mask + test_mask)

    data.train_mask = new_train_mask
    
    return data


def add_noise_features(data, num_noise):
    if not num_noise:
        return data

    num_nodes = data.x.size(0)

    noise_feat = torch.randn((num_nodes, num_noise))
    noise_feat = noise_feat - noise_feat.mean(1, keepdim=True)

    data.x = torch.cat([data.x, noise_feat], dim=-1)
    
    return data


def extract_test_nodes(data, num_samples):
    test_indices = data.test_mask.cpu().numpy().nonzero()[0]
    node_indices = np.random.choice(test_indices, num_samples).tolist()

    return node_indices
