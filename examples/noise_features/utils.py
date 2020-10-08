import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def prepare_data(args):
    dataset = args.dataset.title()

    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, '..', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    data = modify_train_mask(data)
    data = add_noise_features(data, args.num_noise)

    return data


def modify_train_mask(data):
    num_nodes = data.x.size(0)
    num_train = int(num_nodes * 0.8)

    node_indices = np.random.choice(num_nodes, size=num_train, replace=False)
    new_train_mask = torch.zeros_like(data.train_mask)
    new_train_mask[node_indices] = 1
    new_train_mask = new_train_mask > 0

    new_val_mask = torch.zeros_like(data.val_mask)
    new_val_mask = new_val_mask > 0

    new_test_mask = ~(new_train_mask + new_val_mask)

    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask

    # val_mask = data.val_mask
    # test_mask = data.test_mask
    # new_train_mask = ~(val_mask + test_mask)

    # data.train_mask = new_train_mask
    
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


def plot_dist(noise_feats, label=None, ymax=1.0, color=None, title=None, save_path=None):
    sns.set_style('darkgrid')
    ax = sns.distplot(noise_feats, hist=False, kde=True, kde_kws={'label': label}, color=color)
    plt.xlim(-3, 11)
    plt.ylim(ymin=0.0, ymax=ymax)

    if title:
        plt.title(title)
        
    if save_path:
        plt.savefig(save_path)

    return ax


def train(model, data, args, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    train_loss_values, train_acc_values = [], []
    test_loss_values, test_acc_values = [], []

    best = np.inf
    bad_counter = 0

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')

    for epoch in tqdm(range(args.epochs), desc='Training', leave=False):
        if epoch == 0:
            print('       |     Trainging     |     Validation     |')
            print('       |-------------------|--------------------|')
            print(' Epoch |  loss    accuracy |  loss    accuracy  |')
            print('-------|-------------------|--------------------|')

        train_loss, train_acc = train_on_epoch(model, optimizer, data)
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        test_loss, test_acc = evaluate(model, data, data.test_mask)
        test_loss_values.append(test_loss.item())
        test_acc_values.append(test_acc.item())

        if test_loss_values[-1] < best:
            bad_counter = 0
            log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch + 1,
                                                                               train_loss.item(),
                                                                               train_acc.item(),
                                                                               test_loss.item(),
                                                                               test_acc.item())
            
            torch.save(model.state_dict(), model_path)
            log += ' save model to {}'.format(model_path)
            
            if verbose:
                tqdm.write(log)

            best = test_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')

    model.load_state_dict(torch.load(model_path))


def train_on_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)

    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])

    train_loss.backward()
    optimizer.step()

    return train_loss, train_acc


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])

    return loss, acc


def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)
