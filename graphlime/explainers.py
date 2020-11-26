import copy
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge, LassoLars

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


class GraphLIME:
    
    def __init__(self, model, hop=2, rho=0.1, cached=True):
        self.hop = hop
        self.rho = rho
        self.model = model
        self.cached = cached
        self.cached_result = None

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow

        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # get the initial prediction
            with torch.no_grad():
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result
    
    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape
        
        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)
            
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))  # (n, n, 1) or (n, n, d)

        return K
    
    def __compute_gram_matrix__(self, x):
        # unstable implementation due to matrix product (HxH)
        # n = x.shape[0]
        # H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
        # G = np.dot(np.dot(H, x), H)

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G
        
    def explain_node(self, node_idx, x, edge_index, **kwargs):
        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_idx, x, probas, edge_index, **kwargs)

        x = x.detach().cpu().numpy()  # (n, d)
        y = probas.detach().cpu().numpy()  # (n, classes)

        n, d = x.shape

        K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
        L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

        K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
        L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

        solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)

        solver.fit(K_bar * n, L_bar * n)

        return solver.coef_


class LIME:

    def __init__(self, model, num_samples, cached=True):
        self.model = model
        self.num_samples = num_samples
        self.cached = cached
        self.cached_result = None

        self.model.eval()

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # get the initial prediction
            with torch.no_grad():
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        probas = self.__init_predict__(x, edge_index, **kwargs)
        proba, label = probas[node_idx, :].max(dim=0)
        
        x_ = copy.deepcopy(x)
        original_feats = x[node_idx, :]

        sample_x = [original_feats.detach().cpu().numpy()]
        sample_y = [proba.item()]
        
        for _ in tqdm(range(self.num_samples), desc='collect samples', leave=False):
            x_[node_idx, :] = original_feats + torch.randn_like(original_feats)
            
            with torch.no_grad():
                log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
                probas_ = log_logits.exp()

            proba_ = probas_[node_idx, label]

            sample_x.append(x_[node_idx, :].detach().cpu().numpy())
            sample_y.append(proba_.item())

        sample_x = np.array(sample_x)
        sample_y = np.array(sample_y)

        solver = Ridge(alpha=0.1)
        solver.fit(sample_x, sample_y)

        return solver.coef_


class Greedy:

    def __init__(self, model, cached=True):
        self.model = model
        self.cached = cached
        self.cached_result = None

        self.model.eval()

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # get the initial prediction
            with torch.no_grad():
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def explain_node(self, node_idices, x, edge_index, **kwargs):
        if isinstance(node_idices, int):
            node_idices = [node_idices]

        probas = self.__init_predict__(x, edge_index, **kwargs)
        probas, labels = probas[node_idices, :].max(dim=1)  # (m,), (m,)

        num_nodes, num_feats = len(node_idices), x.size(1)
        delta_probas = np.zeros((num_nodes, num_feats))  # (m, #feats)

        self.model.eval()

        for feat_idx in tqdm(range(num_feats), desc='search features', leave=False):
            x_ = copy.deepcopy(x)
            x_[:, feat_idx] = 0.0

            with torch.no_grad():
                log_logits = self.model(x=x_, edge_index=edge_index, **kwargs)
                probas_ = log_logits.exp()
            
            probas_ = probas_[node_idices, :]  # (m, #classes)

            for node_idx in range(num_nodes):
                proba = probas[node_idx].item()
                label = labels[node_idx]
                proba_ = probas_[node_idx, label].item()
                
                delta_probas[node_idx, feat_idx] = abs((proba - proba_) / proba)

        return delta_probas


class Random:

    def __init__(self, num_feats, K):
        self.num_feats = num_feats
        self.K = K

    def explain_node(self):
        return np.random.choice(self.num_feats, self.K)
