import copy
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge

import torch


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
