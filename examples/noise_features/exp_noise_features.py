import os
os.sys.path.append('/Users/william/Documents/AI/github/graphlime')
os.sys.path.append('/Users/william/Documents/AI/github/graphlime/graphlime')

import random
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from models import GAT
from graphlime import GraphLIME, LIME, Greedy, Random
from utils import prepare_data, extract_test_nodes, train, evaluate, plot_dist

warnings.filterwarnings('ignore')

INPUT_DIM = {
    'Cora': 1433,
    'Pubmed': 500
}


def build_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    parser.add_argument('--epochs', type=int, default=400, help='epochs for training a GNN model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--test_samples', type=int, default=200, help='number of test samples')
    parser.add_argument('--num_noise', type=int, default=10, help='number of noise features to add')

    # GraphLIME
    parser.add_argument('--hop', type=int, default=2, help='hops')
    parser.add_argument('--rho', type=float, default=0.15, help='rho')
    parser.add_argument('--K', type=int, default=300, help='top-K most importance features')

    # LIME
    parser.add_argument('--lime_samples', type=int, default=50, help='generate samples for LIME')

    # Greedy
    parser.add_argument('--greedy_threshold', type=float, default=0.03, help='threshold of features for Greedy')

    parser.add_argument('--ymax', type=float, default=1.10, help='max of y-axis')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    args = parser.parse_args()

    return args


def check_args(args):
    assert args.dataset.title() in ['Cora', 'Pubmed']


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def find_noise_feats_by_GraphLIME(model, data, args):
    explainer = GraphLIME(model, hop=args.hop, rho=args.rho)

    node_indices = extract_test_nodes(data, args.test_samples)

    num_noise_feats = []

    for node_idx in tqdm(node_indices, desc='explain node', leave=False):
        coefs = explainer.explain_node(node_idx, data.x, data.edge_index)

        feat_indices = coefs.argsort()[-args.K:]
        feat_indices = [idx for idx in feat_indices if coefs[idx] > 0.0]

        num_noise_feat = sum(idx >= INPUT_DIM[args.dataset] for idx in feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats


def find_noise_feats_by_LIME(model, data, args):
    explainer = LIME(model, args.lime_samples)

    node_indices = extract_test_nodes(data, args.test_samples)

    num_noise_feats = []
    for node_idx in tqdm(node_indices, desc='explain node', leave=False):
        coefs = explainer.explain_node(node_idx, data.x, data.edge_index)
        coefs = np.abs(coefs)

        feat_indices = coefs.argsort()[-args.K:]

        num_noise_feat = sum(idx >= INPUT_DIM[args.dataset] for idx in feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats


def find_noise_feats_by_greedy(model, data, args):
    explainer = Greedy(model)

    node_indices = extract_test_nodes(data, args.test_samples)

    delta_probas = explainer.explain_node(node_indices, data.x, data.edge_index)  # (#test_smaples, #feats)
    feat_indices = delta_probas.argsort(axis=-1)[:, -args.K:]  # (#test_smaples, K)

    num_noise_feats = []
    for node_proba, node_feat_indices in zip(delta_probas, feat_indices):
        node_feat_indices = [feat_idx for feat_idx in node_feat_indices if node_proba[feat_idx] > args.greedy_threshold]
        num_noise_feat = sum(feat_idx >= INPUT_DIM[args.dataset] for feat_idx in node_feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats


def find_noise_feats_by_random(data, args):
    num_feats = data.x.size(1)
    explainer = Random(num_feats, args.K)

    num_noise_feats = []
    for node_idx in tqdm(range(args.test_samples), desc='explain node', leave=False):
        feat_indices = explainer.explain_node()
        noise_feat = (feat_indices >= INPUT_DIM[args.dataset]).sum()
        num_noise_feats.append(noise_feat)

    return num_noise_feats


def main():
    args = build_args()
    check_args(args)
    
    fix_seed(args.seed)

    data = prepare_data(args)

    hparams = {
        'input_dim': data.x.size(1),
        'hidden_dim': 16,
        'output_dim': max(data.y).item() + 1
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(**hparams).to(device)
    data = data.to(device)

    # model_path = '/Users/william/Documents/AI/github/graphlime/examples/noise_features/model.pth'
    # model.load_state_dict(torch.load(model_path))
    train(model, data, args)
    test_loss, test_acc = evaluate(model, data, mask=data.test_mask)
    print('test_loss: {:.4f}, test_acc: {:.4f}'.format(test_loss, test_acc))

    if test_acc < 0.8:
        print('bad model. Please re-run.')
        exit()
    
    print('=== Explain by GraphLIME ===')
    noise_feats = find_noise_feats_by_GraphLIME(model, data, args)
    plot_dist(noise_feats, label='GraphLIME', ymax=args.ymax, color='g')

    print('=== Explain by LIME ===')
    noise_feats = find_noise_feats_by_LIME(model, data, args)
    plot_dist(noise_feats, label='LIME', ymax=args.ymax, color='C0')

    print('=== Explain by Greedy ===')
    noise_feats = find_noise_feats_by_greedy(model, data, args)
    plot_dist(noise_feats, label='Greedy', ymax=args.ymax, color='orange')

    print('=== Explain by Random ===')
    noise_feats = find_noise_feats_by_random(data, args)
    plot_dist(noise_feats, label='Random', ymax=args.ymax, color='k',
              title=f'Distribution of noisy features on {args.dataset} for {model.__class__.__name__}',
              save_path=f'./images/{args.dataset.lower()}.png')
    
    plt.show()


if __name__ == "__main__":
    main()
